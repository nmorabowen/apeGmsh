"""``geomTransf -jntOffset`` against an explicit rigid-link reference (live).

Oracle: a rigid joint offset is a rigid arm from the node to the element
end. The same frame is built twice through the bridge:

(a) every beam-column runs node to node, its offsets carried by
    ``geomTransf ... -jntOffset`` (``jnt_offset=``);
(b) every offset end is an explicit node at ``node + d``, the element runs
    between those nodes with an offset-free transform, and each one is tied
    to its joint with ``rigidLink beam`` under the ``Transformation``
    handler.

Masses sit on the joints only (the elements carry none), so the two models
are the same linear system: the static joint displacements and the first
periods must agree to round-off. The reference offsets of the
cardinal-point cases are written out by hand from the ETABS convention
(``joint_offset`` module docstring), not computed by the helper under
test: cardinal 8 on a 0.6 m deep beam puts its axis 0.3 m below the joint.

PDelta runs the ETABS-style sequence: gravity, ``loadConst``, eigen on the
P-Delta-softened stiffness, then a lateral step about the gravity state.

Stock OpenSees: ``-jntOffset`` is parsed by stock ``Linear``/``PDelta``
(Tcl and Python), so this file runs on the ``live-stock`` lane.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.joint_offset import InsertionPoint

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

pytestmark = pytest.mark.live

E, G = 30e9, 12.5e9
COL = dict(A=0.25, E=E, G=G, Iz=5.2e-3, Iy=3.1e-3, J=6.0e-3)
BEAM = dict(A=0.18, E=E, G=G, Iz=5.4e-3, Iy=1.35e-3, J=3.0e-3)
DEPTH, WIDTH = 0.6, 0.3
RTOL = 1e-9
N_MODES = 6

#: Two-storey, one-bay-each-way 3-D frame: plan 6 x 5 m, storeys 3.5 m.
_XY = ((0.0, 0.0), (6.0, 0.0), (0.0, 5.0), (6.0, 5.0))
_Z = (0.0, 3.5, 7.0)


def _joints_3d() -> dict[int, tuple[float, float, float]]:
    return {1 + 4 * k + i: (x, y, z)
            for k, z in enumerate(_Z) for i, (x, y) in enumerate(_XY)}


def _members_3d() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    cols = [(1 + 4 * k + i, 1 + 4 * (k + 1) + i) for k in range(2) for i in range(4)]
    beams: list[tuple[int, int]] = []
    for k in (1, 2):
        b = 1 + 4 * k
        beams += [(b, b + 1), (b + 2, b + 3),      # along +X
                  (b, b + 2), (b + 1, b + 3)]      # along +Y
    return cols, beams


#: Constant column offsets: eccentric column, rigid top zone 0.3 m deep.
COL_JNT = ((0.05, -0.04, 0.0), (0.05, -0.04, -0.3))


def _beam_offsets_by_hand(case: str, p_i, p_j) -> tuple[np.ndarray, np.ndarray]:
    """The reference offsets of a horizontal beam, written out from the
    ETABS convention with ``axis_2="z"`` (``vecxz=(0, 0, 1)``): local 1
    along the beam, 2 = +Z (up), 3 = 1 x 2."""
    e1 = (np.asarray(p_j) - np.asarray(p_i)) / np.linalg.norm(
        np.asarray(p_j) - np.asarray(p_i))
    e2 = np.array([0.0, 0.0, 1.0])
    e3 = np.cross(e1, e2)
    if case == "cardinal8":
        # top-center on the joint: the axis hangs d/2 below.
        shift = -(DEPTH / 2) * e2
        return shift, shift
    # "cardinal9_ends": top-right on the joint (c2 = +d/2, c3 = +b/2),
    # plus 0.25 m rigid end zones along the axis at both ends.
    shift = -(DEPTH / 2) * e2 - (WIDTH / 2) * e3
    return shift + 0.25 * e1, shift - 0.25 * e1


def _beam_rule(case: str) -> InsertionPoint:
    if case == "cardinal8":
        return InsertionPoint(cardinal_point=8, depth=DEPTH, width=WIDTH, axis_2="z")
    return InsertionPoint(cardinal_point=9, depth=DEPTH, width=WIDTH, axis_2="z",
                          end_offset_i=0.25, end_offset_j=0.25)


def _model_3d(transf_type: str, case: str, explicit: bool,
              offsets: bool = True) -> LiveOpsEmitter:
    joints = _joints_3d()
    cols, beams = _members_3d()
    coords = dict(joints)
    links: list[tuple[int, int]] = []
    next_id = [1000]

    def end(node: int, d: np.ndarray) -> int:
        if not np.any(d):
            return node
        nid = next_id[0]
        next_id[0] += 1
        coords[nid] = tuple(float(c) for c in np.asarray(joints[node]) + d)
        links.append((node, nid))
        return nid

    if explicit:
        col_conn = [(end(i, np.asarray(COL_JNT[0])), end(j, np.asarray(COL_JNT[1])))
                    for i, j in cols]
        beam_conn = []
        for i, j in beams:
            d_i, d_j = _beam_offsets_by_hand(case, joints[i], joints[j])
            beam_conn.append((end(i, d_i), end(j, d_j)))
    else:
        col_conn, beam_conn = cols, beams

    ids = sorted(coords)
    nodes = _NodesStub(ids=ids, coords=[coords[n] for n in ids],
                       node_pgs={"Base": [1, 2, 3, 4]})
    elements = _ElementsStub(elem_pgs={
        # One PG (and transform) per storey: a column line of collinear
        # members with end offsets needs its curves (#2551) otherwise.
        "Cols1": _ElementGroupView(ids=tuple(range(1, 5)),
                                   connectivity=tuple(col_conn[:4])),
        "Cols2": _ElementGroupView(ids=tuple(range(5, 9)),
                                   connectivity=tuple(col_conn[4:])),
        "Beams": _ElementGroupView(ids=tuple(range(9, 17)),
                                   connectivity=tuple(beam_conn)),
    })
    fem = FEMStub(nodes=nodes, elements=elements)
    if links:
        fem.add_node_constraints([
            NodePairRecord(kind=ConstraintKind.RIGID_BEAM, master_node=m,
                           slave_node=s, dofs=[1, 2, 3, 4, 5, 6],
                           name=f"arm{s}")
            for m, s in links])

    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    make = getattr(ops.geomTransf, transf_type)
    if explicit or not offsets:
        tcs = [make(vecxz=(1.0, 0.0, 0.0)) for _ in range(2)]
        tb = make(vecxz=(0.0, 0.0, 1.0))
    else:
        tcs = [make(vecxz=(1.0, 0.0, 0.0), jnt_offset=COL_JNT) for _ in range(2)]
        tb = make(vecxz=(0.0, 0.0, 1.0), jnt_offset=_beam_rule(case))
    for pg, tc in zip(("Cols1", "Cols2"), tcs):
        ops.element.elasticBeamColumn(pg=pg, transf=tc, **COL)
    ops.element.elasticBeamColumn(pg="Beams", transf=tb, **BEAM)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.constraints.Transformation()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    return emitter


def _run(emitter: LiveOpsEmitter, joints: dict[int, Any], *, ndf: int,
         gravity: dict[int, tuple[float, ...]],
         lateral: dict[int, tuple[float, ...]],
         masses: dict[int, tuple[float, ...]],
         ) -> tuple[np.ndarray, np.ndarray]:
    """Gravity -> loadConst -> eigen -> lateral step. Returns (joint
    displacements after both steps, first periods)."""
    o = emitter.ops
    for n, m in masses.items():
        o.mass(n, *m)
    o.timeSeries("Linear", 901)
    o.pattern("Plain", 901, 901)
    for n, f in gravity.items():
        o.load(n, *f)
    o.constraints("Transformation")
    o.numberer("RCM")
    o.system("UmfPack")
    o.test("NormDispIncr", 1e-14, 30, 0)
    o.algorithm("Newton")
    o.integrator("LoadControl", 1.0)
    o.analysis("Static")
    assert o.analyze(1) == 0
    o.loadConst("-time", 0.0)
    lam = np.asarray(o.eigen("-fullGenLapack", N_MODES), dtype=float)
    o.timeSeries("Linear", 902)
    o.pattern("Plain", 902, 902)
    for n, f in lateral.items():
        o.load(n, *f)
    assert o.analyze(1) == 0
    disp = np.array([[o.nodeDisp(n, k + 1) for k in range(ndf)]
                     for n in sorted(joints)], dtype=float)
    return disp, 2.0 * np.pi / np.sqrt(lam)


def _assert_same(a: np.ndarray, b: np.ndarray, what: str) -> None:
    scale = float(np.max(np.abs(b)))
    err = float(np.max(np.abs(a - b))) / scale
    assert err < RTOL, f"{what}: max |a - b| / max |b| = {err:.3e}"


def _loads_3d() -> tuple[dict, dict, dict]:
    floors = {5, 6, 7, 8, 9, 10, 11, 12}
    gravity = {n: (0.0, 0.0, -400e3, 0.0, 0.0, 0.0) for n in floors}
    lateral = {n: (30e3 * (1 + (n > 8)), 12e3, 0.0, 0.0, 0.0, 5e3)
               for n in floors}
    masses = {n: (40e3, 40e3, 40e3, 1e3, 1e3, 2.0e5) for n in floors}
    return gravity, lateral, masses


@pytest.mark.parametrize("transf_type", ["Linear", "PDelta"])
@pytest.mark.parametrize("case", ["cardinal8", "cardinal9_ends"])
def test_3d_jnt_offset_matches_rigid_link_reference(transf_type: str, case: str) -> None:
    pytest.importorskip("openseespy.opensees")
    gravity, lateral, masses = _loads_3d()
    joints = _joints_3d()

    a = _model_3d(transf_type, case, explicit=False)
    disp_a, per_a = _run(a, joints, ndf=6, gravity=gravity,
                         lateral=lateral, masses=masses)
    b = _model_3d(transf_type, case, explicit=True)
    disp_b, per_b = _run(b, joints, ndf=6, gravity=gravity,
                         lateral=lateral, masses=masses)

    _assert_same(disp_a, disp_b, f"{transf_type}/{case} joint displacements")
    _assert_same(per_a, per_b, f"{transf_type}/{case} periods")
    # Not vacuous: the offsets change the answer well past the tolerance.
    disp_0, per_0 = _run(_model_3d(transf_type, case, explicit=False, offsets=False),
                         joints, ndf=6, gravity=gravity, lateral=lateral,
                         masses=masses)
    assert np.max(np.abs(disp_0 - disp_b)) / np.max(np.abs(disp_b)) > 1e-3
    assert np.max(np.abs(per_0 - per_b)) / np.max(per_b) > 1e-3


# ---------------------------------------------------------------------------
# 2-D portal: -jntOffset dXi dYi dXj dYj
# ---------------------------------------------------------------------------

_J2 = {1: (0.0, 0.0), 2: (6.0, 0.0), 3: (0.0, 4.0), 4: (6.0, 4.0),
       5: (0.0, 8.0), 6: (6.0, 8.0)}
_COLS2 = [(1, 3), (2, 4), (3, 5), (4, 6)]
_BEAMS2 = [(3, 4), (5, 6)]
COL2_JNT = ((0.1, 0.0), (0.1, -0.3))
#: Beam: cardinal 8 by hand (axis 0.3 below the joints) + 0.2 m end zones.
BEAM2_JNT = ((0.2, -0.3), (-0.2, -0.3))


def _model_2d(transf_type: str, explicit: bool, rule: bool) -> LiveOpsEmitter:
    coords = {n: (x, y, 0.0) for n, (x, y) in _J2.items()}
    links: list[tuple[int, int]] = []
    next_id = [100]

    def end(node: int, d: tuple[float, float]) -> int:
        if d == (0.0, 0.0):
            return node
        nid = next_id[0]
        next_id[0] += 1
        x, y = _J2[node]
        coords[nid] = (x + d[0], y + d[1], 0.0)
        links.append((node, nid))
        return nid

    if explicit:
        col_conn = [(end(i, COL2_JNT[0]), end(j, COL2_JNT[1])) for i, j in _COLS2]
        beam_conn = [(end(i, BEAM2_JNT[0]), end(j, BEAM2_JNT[1])) for i, j in _BEAMS2]
    else:
        col_conn, beam_conn = _COLS2, _BEAMS2
    ids = sorted(coords)
    nodes = _NodesStub(ids=ids, coords=[coords[n] for n in ids],
                       node_pgs={"Base": [1, 2]})
    elements = _ElementsStub(elem_pgs={
        "Cols1": _ElementGroupView(ids=(1, 2), connectivity=tuple(col_conn[:2])),
        "Cols2": _ElementGroupView(ids=(3, 4), connectivity=tuple(col_conn[2:])),
        "Beams": _ElementGroupView(ids=(5, 6), connectivity=tuple(beam_conn)),
    })
    fem = FEMStub(nodes=nodes, elements=elements)
    if links:
        fem.add_node_constraints([
            NodePairRecord(kind=ConstraintKind.RIGID_BEAM, master_node=m,
                           slave_node=s, dofs=[1, 2, 3], name=f"arm{s}")
            for m, s in links])
    ops = apeSees(cast("object", fem), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    make = getattr(ops.geomTransf, transf_type)
    if explicit:
        tcs, tb = [make(), make()], make()
    else:
        tcs = [make(jnt_offset=COL2_JNT) for _ in range(2)]
        tb = make(jnt_offset=(
            InsertionPoint(cardinal_point=8, depth=DEPTH, width=WIDTH,
                           end_offset_i=0.2, end_offset_j=0.2)
            if rule else BEAM2_JNT))
    for pg, tc in zip(("Cols1", "Cols2"), tcs):
        ops.element.elasticBeamColumn(pg=pg, transf=tc, A=0.25, E=E, Iz=5.2e-3)
    ops.element.elasticBeamColumn(pg="Beams", transf=tb, A=0.18, E=E, Iz=5.4e-3)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.constraints.Transformation()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    return emitter


@pytest.mark.parametrize("transf_type", ["Linear", "PDelta"])
@pytest.mark.parametrize("rule", [False, True], ids=["constant", "insertion_point"])
def test_2d_jnt_offset_matches_rigid_link_reference(transf_type: str, rule: bool) -> None:
    pytest.importorskip("openseespy.opensees")
    floors = {3, 4, 5, 6}
    gravity = {n: (0.0, -500e3, 0.0) for n in floors}
    lateral = {n: (25e3 * (1 + (n > 4)), 0.0, 4e3) for n in floors}
    masses = {n: (50e3, 50e3, 1e3) for n in floors}
    disp_a, per_a = _run(_model_2d(transf_type, False, rule), _J2, ndf=3,
                         gravity=gravity, lateral=lateral, masses=masses)
    disp_b, per_b = _run(_model_2d(transf_type, True, False), _J2, ndf=3,
                         gravity=gravity, lateral=lateral, masses=masses)
    _assert_same(disp_a, disp_b, f"2-D {transf_type} joint displacements")
    _assert_same(per_a[:4], per_b[:4], f"2-D {transf_type} periods")


# ---------------------------------------------------------------------------
# A from_model line load covers the rigid end zones (ETABS), #2470 review 1
# ---------------------------------------------------------------------------
#
# Oracle: the same fixed-fixed beam with its end zones modelled as explicit
# elements 1e6 times stiffer than the beam, every segment on the element
# axis carrying the load, so the load covers the full node-to-node length.
# Reactions must agree to the stiff zones' compliance (~1e-6). Without the
# end-zone transfer OpenSees loads only the flexible length: 6 of the 40 kN
# of the 2-D case go missing.

_STIFF = 1e6
_PROPS_2D = dict(A=0.01, E=200e9, Iz=1e-4)
_PROPS_3D = dict(A=0.01, E=200e9, G=80e9, J=1e-4, Iy=1e-4, Iz=1e-4)


def _static_reactions(o: Any, nodes: tuple[int, ...]) -> np.ndarray:
    o.constraints("Plain")
    o.numberer("RCM")
    o.system("UmfPack")
    o.test("NormDispIncr", 1e-12, 10, 0)
    o.algorithm("Linear")
    o.integrator("LoadControl", 1.0)
    o.analysis("Static")
    assert o.analyze(1) == 0
    o.reactions()
    return np.array([o.nodeReaction(n) for n in nodes], dtype=float)


def _stiff_zone_reference(o: Any, ndm: int, chain: list[tuple[float, float, float]],
                          loaded: list[bool], w: tuple[float, ...]) -> np.ndarray:
    """Joints 1 and 2 are ``chain[0]`` and ``chain[-1]``; the segments
    between consecutive points are elastic beams, the longest at the beam's
    properties and the rest ``_STIFF`` times stiffer; each ``loaded``
    segment carries ``-beamUniform *w``."""
    o.wipe()
    ndf = 3 if ndm == 2 else 6
    o.model("basic", "-ndm", ndm, "-ndf", ndf)
    tags = [1] + [100 + k for k in range(1, len(chain) - 1)] + [2]
    for t, xyz in zip(tags, chain):
        o.node(t, *xyz[:ndm])
    for t in (1, 2):
        o.fix(t, *([1] * ndf))
    lengths = [float(np.linalg.norm(np.subtract(b, a))) for a, b in zip(chain, chain[1:])]
    flex = int(np.argmax(lengths))
    o.timeSeries("Linear", 1)
    o.pattern("Plain", 1, 1)
    for k, (a, b) in enumerate(zip(tags, tags[1:])):
        scale = 1.0 if k == flex else _STIFF
        if ndm == 2:
            o.geomTransf("Linear", 10 + k)
            p2 = _PROPS_2D
            o.element("elasticBeamColumn", 10 + k, a, b, p2["A"], p2["E"] * scale,
                      p2["Iz"], 10 + k)
        else:
            axis = np.subtract(chain[k + 1], chain[k])
            vecxz = (1.0, 0.0, 0.0) if abs(axis[0]) < 1e-12 else (0.0, 0.0, 1.0)
            o.geomTransf("Linear", 10 + k, *vecxz)
            p3 = _PROPS_3D
            o.element("elasticBeamColumn", 10 + k, a, b, p3["A"], p3["E"] * scale,
                      p3["G"] * scale, p3["J"], p3["Iy"], p3["Iz"], 10 + k)
        if loaded[k]:
            o.eleLoad("-ele", 10 + k, "-type", "-beamUniform", *w)
    return _static_reactions(o, (1, 2))


@pytest.mark.parametrize("ndm", [2, 3])
def test_line_load_over_end_zones_matches_stiff_zone_reference(ndm: int) -> None:
    pytest.importorskip("openseespy.opensees")
    from tests.opensees.unit.test_jnt_offset import _loaded_beam

    chain: list[tuple[float, float, float]]
    if ndm == 2:
        jnt: Any = ((0.3, 0.0), (-0.3, 0.0))
        params = {"wx": 0.0, "wy": -10.0, "wz": 0.0}
        chain = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (3.7, 0.0, 0.0), (4.0, 0.0, 0.0)]
        loaded = [True, True, True]
        w: tuple[float, ...] = (-10.0, 0.0)            # Wy Wx; local = global here
    else:   # hung 0.3 m below the joints, 0.3 m end zones, vecxz = +Z
        jnt = ((0.3, 0.0, -0.3), (-0.3, 0.0, -0.3))
        params = {"wx": 1.0, "wy": 3.0, "wz": -10.0}
        chain = [(0.0, 0.0, 0.0), (0.0, 0.0, -0.3), (0.3, 0.0, -0.3),
                 (3.7, 0.0, -0.3), (4.0, 0.0, -0.3), (4.0, 0.0, 0.0)]
        loaded = [False, True, True, True, False]
        w = (3.0, -10.0, 1.0)                          # Wy Wz Wx: y = +Y, z = +Z
    emitter = LiveOpsEmitter(wipe=True)
    _loaded_beam(jnt, ndm=ndm, params=params).build().emit(emitter)
    o = emitter.ops
    react_a = _static_reactions(o, (1, 2))
    total = 4.0 * np.array([params["wx"], params["wy"], params["wz"]][:ndm])
    np.testing.assert_allclose(-react_a[:, :ndm].sum(axis=0), total, rtol=1e-9)
    react_b = _stiff_zone_reference(o, ndm, chain, loaded, w)
    scale = float(np.max(np.abs(react_b)))
    err = float(np.max(np.abs(react_a - react_b))) / scale
    assert err < 1e-5, f"{ndm}-D reactions: max |a - b| / max |b| = {err:.3e}"
