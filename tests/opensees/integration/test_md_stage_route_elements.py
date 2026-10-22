"""#2446: which element classes the plastic stage-flip sweep reaches.

The bridge emits a ``ManzariDafalias`` / ``LadrunoSANISAND`` flip to a plastic
stage as an element sweep (``setParameter -val 1 -eleRange ... materialState
<tag>``) followed by ``updateMaterialStage``
(``stage_emit.material_stage_flip_commands``).  The sweep initialises an
element only when the element hands ``materialState`` to its materials, so a
multi-element plastic flip passes only on a :data:`MATERIAL_STATE_ELEMENTS`
class and is refused, by class name, on any other.

* The unit tests pin the emitted lines and the stage-close rules.
* The live census runs two elements of each class on one material through a
  prescribed homogeneous strain path (every DOF driven, so any element runs
  the same way) and compares the emitted flip with the per-element reference
  (each element its own material and its own ``updateMaterialStage``).
"""
from __future__ import annotations

import io
import warnings
from typing import Any

import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.stage_emit import (
    MATERIAL_STATE_ELE_RANGE,
    material_stage_flip_commands,
)
from apeGmsh.opensees._target import OpenSeesTarget
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.stage._builder import (
    FIRST_ELEMENT_ONLY_PLASTIC_INIT,
    MATERIAL_STATE_ELEMENTS,
)

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)
from tests.opensees.integration.test_md_stage_all_elements import (
    MD_ARGS,
    emitted_flip_lines,
)

_MD = dict(
    G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712, lambda_c=0.019,
    e0=0.934, ksi=0.7, P_atm=100.0, m=0.01, h0=7.05, Ch=0.968, nb=1.1,
    A0=0.704, nd=3.5, z_max=4.0, cz=600.0, rho=1.42,
)


# ---------------------------------------------------------------------------
# The emitted lines
# ---------------------------------------------------------------------------


def test_the_flip_commands() -> None:
    lo, hi = MATERIAL_STATE_ELE_RANGE
    assert (lo, hi) == (1, 2147483647)
    assert material_stage_flip_commands(7, 0) == (
        ("updateMaterialStage", ("-material", 7, "-stage", 0)),)
    assert material_stage_flip_commands(7, 1) == (
        ("setParameter", ("-val", 1.0, "-eleRange", lo, hi,
                          "materialState", "7")),
        ("updateMaterialStage", ("-material", 7, "-stage", 1)),
    )


def test_the_py_lines_pass_the_tag_as_a_string() -> None:
    """``setParameter`` reads its trailing argv as strings: an int tag is
    'Invalid String Input!' on openseespy (measured, stock and fork)."""
    assert emitted_flip_lines(7) == [
        "ops.setParameter('-val', 1.0, '-eleRange', 1, 2147483647, "
        "'materialState', '7')",
        "ops.updateMaterialStage('-material', 7, '-stage', 1)",
    ]
    assert emitted_flip_lines(7, 0) == [
        "ops.updateMaterialStage('-material', 7, '-stage', 0)"]


def test_the_tables() -> None:
    assert FIRST_ELEMENT_ONLY_PLASTIC_INIT == {
        "ManzariDafalias", "LadrunoSANISAND"}
    assert MATERIAL_STATE_ELEMENTS == {
        "SixNodeTri", "FourNodeTetrahedron", "BezierTri6", "BezierTet10",
        "LadrunoUP"}
    # Every class of the table has a census row on each build it runs on.
    assert MATERIAL_STATE_ELEMENTS == (
        {cls for cls, _builds in _FORWARDING} | {"LadrunoUP"})


def _tri6_fem() -> FEMStub:
    """Two DISCONNECTED 6-node triangles: ``A`` (eid 1) and ``B`` (eid 2),
    plus a quad ``Q`` (eid 3) for the mixed-carrier case."""
    tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
           (0.0, 0.5)]
    coords = [(x, y, 0.0) for x, y in tri]
    coords += [(x + 3.0, y, 0.0) for x, y in tri]
    coords += [(6.0, 0.0, 0.0), (7.0, 0.0, 0.0), (7.0, 1.0, 0.0),
               (6.0, 1.0, 0.0)]
    return FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 17)),
            coords=coords,
            node_pgs={"A": [1, 2, 3, 4, 5, 6], "B": [7, 8, 9, 10, 11, 12],
                      "Q": [13, 14, 15, 16], "Base": [1, 2, 7, 8, 13, 14]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "A": _ElementGroupView(ids=(1,),
                                       connectivity=((1, 2, 3, 4, 5, 6),)),
                "B": _ElementGroupView(
                    ids=(2,), connectivity=((7, 8, 9, 10, 11, 12),)),
                "Q": _ElementGroupView(ids=(3,),
                                       connectivity=((13, 14, 15, 16),)),
            },
        ),
    )


def _chain(ops: apeSees) -> dict[str, object]:
    return {
        "test": ops.test.NormUnbalance(tol=1e-6, max_iter=50),
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def _bridge(
    cls: str = "ManzariDafalias", *, pgs: "tuple[str, ...]" = ("A", "B"),
    quad: bool = False, mode: "str | None" = None,
) -> tuple[apeSees, Any]:
    target = None if mode is None else OpenSeesTarget(mode=mode)  # type: ignore[arg-type]
    ops = apeSees(_tri6_fem(), default_orientation=None, opensees=target)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = getattr(ops.nDMaterial, cls)(name="sand", **_MD)
    for pg in pgs:
        ops.element.SixNodeTri(pg=pg, thickness=1.0, material=mat)
    if quad:
        ops.element.FourNodeQuad(pg="Q", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(0, 1))
    return ops, mat


def _flip(ops: apeSees, mat: Any, *, route: str = "stage", value: int = 1,
          pgs: "tuple[str, ...]" = ("A", "B")) -> None:
    with ops.stage(name="push") as s:
        if route == "stage":
            s.update_material_stage(materials=[mat], stage=value)
        else:
            for pg in pgs:
                s.update_parameter(route, float(value), pg=pg, material=mat)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
def test_the_decks_sweep_then_flip(cls: str) -> None:
    ops, mat = _bridge(cls)
    _flip(ops, mat)
    tag = ops.tag_for(mat)
    tcl = TclEmitter()
    ops.build().emit(tcl)
    lines = tcl.lines()
    sweep = (f"setParameter -val 1.0 -eleRange 1 2147483647 materialState "
             f"{tag}")
    ums = f"updateMaterialStage -material {tag} -stage 1"
    assert lines.index(sweep) + 1 == lines.index(ums)
    py = PyEmitter()
    ops.build().emit(py)
    buf = io.StringIO()
    py.write_to(buf)
    text = buf.getvalue()
    assert (
        "ops.setParameter('-val', 1.0, '-eleRange', 1, 2147483647, "
        f"'materialState', '{tag}')\n"
        f"ops.updateMaterialStage('-material', {tag}, '-stage', 1)"
    ) in text


def test_an_elastic_flip_is_one_line() -> None:
    ops, mat = _bridge()
    _flip(ops, mat, value=0)
    tcl = TclEmitter()
    ops.build().emit(tcl)
    assert not [ln for ln in tcl.lines() if ln.startswith("setParameter")]
    assert f"updateMaterialStage -material {ops.tag_for(mat)} -stage 0" in (
        tcl.lines())


# ---------------------------------------------------------------------------
# The stage-close rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", sorted(FIRST_ELEMENT_ONLY_PLASTIC_INIT))
@pytest.mark.parametrize("route", ["stage", "materialState"])
def test_a_forwarding_class_on_two_elements_passes(cls: str, route: str) -> None:
    ops, mat = _bridge(cls)
    _flip(ops, mat, route=route)
    assert len(ops._stage_records) == 1


@pytest.mark.parametrize("route", ["stage", "materialState"])
def test_a_non_forwarding_carrier_is_refused_by_name(route: str) -> None:
    ops, mat = _bridge(quad=True)
    with pytest.raises(BridgeError) as exc:
        _flip(ops, mat, route=route, pgs=("A", "B", "Q"))
    msg = str(exc.value)
    assert "FourNodeQuad carries it" in msg and "#2445" in msg
    assert "SixNodeTri carries" not in msg
    assert f"nDMaterial tag {ops.tag_for(mat)} (3 elements)" in msg
    assert not ops._stage_records


def test_a_materialstate_route_must_cover_every_element() -> None:
    ops, mat = _bridge()
    with pytest.raises(BridgeError) as exc:
        _flip(ops, mat, route="materialState", pgs=("A",))
    msg = str(exc.value)
    assert "leaves 1 of those elements out of its targets" in msg
    assert "#2445" in msg


def test_a_partial_materialstate_route_beside_the_stage_flip_passes() -> None:
    """``s.update_material_stage`` of the same material sweeps them all."""
    ops, mat = _bridge()
    with ops.stage(name="push") as s:
        s.update_material_stage(materials=[mat], stage=1)
        s.update_parameter("materialState", 1.0, pg="A", material=mat)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    assert len(ops._stage_records) == 1


def test_the_updatematerialstage_parameter_is_refused() -> None:
    ops, mat = _bridge()
    with pytest.raises(BridgeError, match="theInt"):
        _flip(ops, mat, route="updateMaterialStage")


def test_a_one_element_material_passes_on_any_class() -> None:
    ops, mat = _bridge(pgs=(), quad=True)
    _flip(ops, mat)
    assert len(ops._stage_records) == 1


def _flip_then_activate(ops: apeSees, mat: Any, *, reflip: bool) -> None:
    with ops.stage(name="plastic") as s:
        s.update_material_stage(materials=[mat], stage=1)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install") as s:
        s.activate(pgs=["B"])
        if reflip:
            s.update_material_stage(materials=[mat], stage=1)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)


@pytest.mark.parametrize("mode", ["stock", "auto", None])
def test_activating_an_element_after_the_plastic_flip_refuses(
    mode: "str | None",
) -> None:
    ops, mat = _bridge(mode=mode)
    with pytest.raises(BridgeError) as exc:
        _flip_then_activate(ops, mat, reflip=False)
    msg = str(exc.value)
    assert "SixNodeTri on pg 'B'" in msg and "#2445" in msg
    assert len(ops._stage_records) == 1


def test_reflipping_in_the_activating_stage_passes() -> None:
    ops, mat = _bridge(mode="stock")
    _flip_then_activate(ops, mat, reflip=True)
    assert len(ops._stage_records) == 2


# ---------------------------------------------------------------------------
# The archive and its replay (R1b)
# ---------------------------------------------------------------------------


def r1b_tri6_flip_bridge() -> apeSees:
    """One ``ManzariDafalias`` on two disconnected ``SixNodeTri`` (a real
    ``FEMData``, so the archive replays): an elastic stage, then a stage
    that flips the material to plastic.  An R1b corpus model
    (``tests/opensees/golden/test_roundtrip_oracle.py``)."""
    import numpy as np

    from apeGmsh.mesh._element_types import ElementGroup, make_type_info
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        ElementComposite,
        FEMData,
        MeshInfo,
        NodeComposite,
    )

    tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 0.5),
           (0.0, 0.5)]
    xy = [(x, y) for x, y in tri] + [(x + 3.0, y) for x, y in tri]
    ids = np.arange(1, 13, dtype=np.int64)
    coords = np.array([(x, y, 0.0) for x, y in xy], dtype=np.float64)
    info6 = make_type_info(code=9, gmsh_name="Triangle 6", dim=2, order=2,
                           npe=6, count=2)
    group = ElementGroup(
        element_type=info6, ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
                              dtype=np.int64),
    )

    def pg(name: str, nodes: "list[int]", eles: "list[int]") -> dict:
        sel = np.array(nodes, dtype=np.int64)
        return {"name": name, "node_ids": sel,
                "node_coords": coords[[i - 1 for i in nodes]],
                "element_ids": np.array(eles, dtype=np.int64)}

    groups = {
        (2, 301): pg("A", [1, 2, 3, 4, 5, 6], [1]),
        (2, 302): pg("B", [7, 8, 9, 10, 11, 12], [2]),
        (0, 303): pg("Base", [1, 2, 4, 7, 8, 10], []),
    }
    fem = FEMData(
        nodes=NodeComposite(node_ids=ids, node_coords=coords,
                            physical=PhysicalGroupSet(groups),
                            labels=LabelSet({})),
        elements=ElementComposite(groups={9: group},
                                  physical=PhysicalGroupSet(groups),
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=12, n_elems=2, bandwidth=6, types=[info6]),
        model_name="tri6_flip",
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    sand = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=sand)
    ops.element.SixNodeTri(pg="B", thickness=1.0, material=sand)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="gravity") as s:
        s.update_material_stage(materials=[sand], stage=0)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="push") as s:
        s.update_material_stage(materials=[sand], stage=1)
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    return ops


def test_the_replayed_deck_sweeps_too(tmp_path: Any) -> None:
    """The archive keeps the flip as ``(tag, stage)`` rows, as before
    #2446 (so ``model_hash`` does not move), and the replay renders the
    same two lines through the same emitter verb."""
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    ops = r1b_tri6_flip_bridge()
    tag = ops.tag_for(ops._names["sand"])
    deck = tmp_path / "bridge.tcl"
    ops.tcl(str(deck), flat=True)
    archive = tmp_path / "model.h5"
    ops.h5(str(archive))
    model = OpenSeesModel.from_h5(str(archive))
    _gravity, push = model.stages()
    assert push.update_material_stages == ((tag, 1),)
    replayed = model.build("tcl", flat=True)
    sweep = (f"setParameter -val 1.0 -eleRange 1 2147483647 materialState "
             f"{tag}")
    assert deck.read_text(encoding="utf-8").count(sweep) == 1
    assert replayed.count(sweep) == 1


def _deck_tri6_run(tmp_path: Any, *, shared: bool) -> list[float]:
    """The bridge's own openseespy deck, run on the resolved backend: two
    SixNodeTri, hydrostatic 100 kPa elastic, a plastic flip, then 40 kPa
    more vertical.  ``shared``: one material on both (the swept flip);
    else each its own (one element per flip, the per-element reference).
    """
    from apeGmsh.opensees.emitter.live import get_ops

    ops = apeSees(_tri6_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mats = [ops.nDMaterial.ManzariDafalias(name=f"sand_{i}", **_MD)
            for i in range(1 if shared else 2)]
    ops.element.SixNodeTri(pg="A", thickness=1.0, material=mats[0])
    ops.element.SixNodeTri(pg="B", thickness=1.0, material=mats[-1])
    # Nodes 13-16 (the stub's quad, unused here) carry no element.
    ops.fix(nodes=[1, 7, 13, 14, 15, 16], dofs=(1, 1))
    ops.fix(nodes=[2, 4, 8, 10], dofs=(0, 1))
    ops.fix(nodes=[3, 6, 9, 12], dofs=(1, 0))

    def chain() -> dict[str, object]:
        return {
            "test": ops.test.NormUnbalance(tol=1e-8, max_iter=100),
            "algorithm": ops.algorithm.Newton(),
            "integrator": ops.integrator.LoadControl(dlam=0.05),
            "constraints": ops.constraints.Transformation(),
            "numberer": ops.numberer.RCM(),
            "system": ops.system.UmfPack(),
            "analysis": ops.analysis.Static(),
        }

    # Consistent loads of a quadratic edge: 1/6, 4/6, 1/6.
    edge = ((2, 1 / 6), (3, 1 / 6), (5, 4 / 6))
    with ops.stage(name="gravity") as s:
        s.update_material_stage(materials=mats, stage=0)
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for b in (0, 6):
                for n, w in edge:
                    p.load(node=b + n, forces=(-100.0 * w, -100.0 * w))
        s.analysis(**chain())
        s.run(n_increments=20)
    with ops.stage(name="push") as s:
        s.update_material_stage(materials=mats, stage=1)
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for b in (0, 6):
                for n, w in edge:
                    p.load(node=b + n, forces=(0.0, -40.0 * w))
        s.analysis(**chain())
        s.run(n_increments=20)
    path = tmp_path / ("shared.py" if shared else "reference.py")
    ops.py(str(path))
    src = path.read_text(encoding="utf-8")
    header = "import openseespy.opensees as ops\n"
    assert src.count(header) == 1
    live = get_ops()
    try:
        exec(compile(src.replace(header, ""), str(path), "exec"),  # noqa: S102
             {"__name__": "deck", "ops": live})
        return [d for n in sorted(live.getNodeTags())
                for d in live.nodeDisp(n)]
    finally:
        live.wipe()


@pytest.mark.live
def test_the_bridge_deck_initialises_every_element(tmp_path: Any) -> None:
    """End to end: the bridge's deck with one shared material matches the
    deck that gives each element its own (stock and fork)."""
    pytest.importorskip("openseespy.opensees")
    ref = _deck_tri6_run(tmp_path, shared=False)
    swept = _deck_tri6_run(tmp_path, shared=True)
    scale = max(abs(v) for v in ref)
    assert scale > 1e-4
    assert _diff(swept, ref) <= 1e-10 * max(1.0, scale)


# ---------------------------------------------------------------------------
# The live census
# ---------------------------------------------------------------------------

_TRI3 = [(0, 0), (1, 0), (0, 1)]
_TRI6 = _TRI3 + [(0.5, 0), (0.5, 0.5), (0, 0.5)]
_QUAD4 = [(0, 0), (1, 0), (1, 1), (0, 1)]
_TET4 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


def _mid(pts: "list[tuple[int, ...]]", a: int, b: int) -> tuple[float, ...]:
    return tuple((x + y) / 2 for x, y in zip(pts[a], pts[b]))


_TET10 = _TET4 + [_mid(_TET4, a, b)
                  for a, b in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))]
_HEX8 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

_PS = "PlaneStrain"
#: apeGmsh class -> (Tcl token, ndm, node coordinates, argv after the tag).
_CLASSES: dict[str, Any] = {
    "SixNodeTri": ("tri6n", 2, _TRI6, lambda n, m: (*n, 1.0, _PS, m)),
    "BezierTri6": ("BezierTri6", 2, _TRI6, lambda n, m: (*n, 1.0, _PS, m)),
    "FourNodeTetrahedron": ("FourNodeTetrahedron", 3, _TET4,
                            lambda n, m: (*n, m)),
    "BezierTet10": ("BezierTet10", 3, _TET10, lambda n, m: (*n, m)),
    "FourNodeQuad": ("quad", 2, _QUAD4, lambda n, m: (*n, 1.0, _PS, m)),
    "Tri31": ("tri31", 2, _TRI3, lambda n, m: (*n, 1.0, _PS, m)),
    "stdBrick": ("stdBrick", 3, _HEX8, lambda n, m: (*n, m)),
}

#: The table's classes, and whether their census needs the fork.
_FORWARDING = [
    ("SixNodeTri", "both"), ("FourNodeTetrahedron", "both"),
    ("BezierTri6", "fork"), ("BezierTet10", "fork"),
]
#: Upstream classes that drop the sweep (on stock and at FORK_PIN alike).
_DROPPING = ["FourNodeQuad", "Tri31", "stdBrick"]


def _census(cls: str, how: str, mat: str = "ManzariDafalias") -> list[float]:
    """Two elements of ``cls``, every DOF on a homogeneous strain path to
    t=1 elastic, flip, then on to t=2; returns their stresses."""
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    token, ndm, coords, argv = _CLASSES[cls]
    eps = (-1e-2, -1.5e-2, 0.0)[:ndm]
    ops.wipe()
    try:
        ops.model("basic", "-ndm", ndm, "-ndf", ndm)
        tags = [1, 2] if how == "reference" else [1, 1]
        for t in sorted(set(tags)):
            ops.nDMaterial(mat, t, *MD_ARGS)
        ops.timeSeries("Path", 1, "-time", 0.0, 1.0, 2.0,
                       "-values", 0.0, 1.0, 2.0)
        ops.pattern("Plain", 1, 1)
        for q, t in enumerate(tags):
            nodes = []
            for i, c in enumerate(coords):
                n = 100 * q + i + 1
                xyz = [float(v) for v in c]
                ops.node(n, xyz[0] + 3.0 * q, *xyz[1:])
                nodes.append(n)
                for d in range(ndm):
                    ops.sp(n, d + 1, eps[d] * xyz[d])
            ops.element(token, q + 1, *argv(nodes, t))
        for t in sorted(set(tags)):
            ops.updateMaterialStage("-material", t, "-stage", 0)
        ops.constraints("Penalty", 1.0e12, 1.0e12)
        ops.numberer("Plain")
        ops.system("FullGeneral")
        ops.test("NormUnbalance", 1e-8, 20, 0)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 0.05)
        ops.analysis("Static")
        assert ops.analyze(20) == 0
        if how == "emitted":
            for line in emitted_flip_lines(1):
                exec(line, {"ops": ops})  # noqa: S102 - the emitted deck line
        else:
            for t in sorted(set(tags)):
                ops.updateMaterialStage("-material", t, "-stage", 1)
        assert ops.analyze(20) == 0
        return [
            s for e in (1, 2)
            for s in (ops.eleResponse(e, "stresses")
                      or ops.eleResponse(e, "stress"))
        ]
    finally:
        ops.wipe()


def _forwarding_params() -> list[Any]:
    out = []
    for cls, builds in _FORWARDING:
        marks = [pytest.mark.live]
        if builds == "fork":
            marks.append(pytest.mark.ladruno_fork)
        out.append(pytest.param(cls, marks=marks, id=cls))
    return out


def _diff(a: "list[float]", b: "list[float]") -> float:
    assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b))


@pytest.mark.parametrize("cls", _forwarding_params())
def test_census_the_sweep_initialises_every_element(cls: str) -> None:
    ref = _census(cls, "reference")
    emitted = _census(cls, "emitted")
    first = _census(cls, "first")
    scale = max(1.0, max(abs(v) for v in ref))
    assert _diff(emitted, ref) <= 1e-10 * scale
    assert _diff(first, ref) > 1e-8 * scale, "the case must tell them apart"


@pytest.mark.live
@pytest.mark.parametrize("cls", _DROPPING)
def test_census_a_refused_class_drops_the_sweep(cls: str) -> None:
    ref = _census(cls, "reference")
    emitted = _census(cls, "emitted")
    first = _census(cls, "first")
    assert _diff(emitted, first) == 0.0
    assert _diff(emitted, ref) > 0.0
    assert cls not in MATERIAL_STATE_ELEMENTS
