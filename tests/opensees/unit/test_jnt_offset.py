"""``geomTransf -jntOffset``: tokens, refusals, the per-element fan-out,
the ETABS insertion-point helper's sign conventions, and the H5 round trip.

The live numeric oracle (offsets vs an explicit ``rigidLink beam``
reference) is ``tests/opensees/integration/test_jnt_offset_live.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.joint_offset import InsertionPoint, insertion_point_offsets
from apeGmsh.opensees.transform import Corotational, Linear, PDelta

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

J3 = ((0.1, -0.2, 0.3), (-0.4, 0.5, -0.6))
J3_FLAT = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
J2 = ((0.1, -0.2), (0.3, -0.4))
J2_FLAT = (0.1, -0.2, 0.3, -0.4)


# ---------------------------------------------------------------------------
# Primitive emit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [Linear, PDelta])
def test_constant_offset_emits_the_jnt_offset_kwarg(cls: Any) -> None:
    rec = RecordingEmitter()
    cls(vecxz=(0.0, 0.0, 1.0), jnt_offset=J3)._emit(rec, tag=7)
    assert rec.calls == [("geomTransf", (cls.__name__, 7, 0.0, 0.0, 1.0),
                          {"jnt_offset": J3_FLAT})]


def test_all_zero_offset_emits_no_flag() -> None:
    rec = RecordingEmitter()
    Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=((0, 0, 0), (0, 0, 0)))._emit(rec, tag=1)
    assert rec.calls == [("geomTransf", ("Linear", 1, 0.0, 0.0, 1.0), {})]


def test_insertion_point_needs_the_bridge() -> None:
    t = Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=InsertionPoint(depth=0.6))
    with pytest.raises(NotImplementedError, match="resolved per element"):
        t._emit(RecordingEmitter(), tag=1)


@pytest.mark.parametrize("bad", [
    ((0.0, 0.0, 0.0),),                         # one end
    ((0.0, 0.0), (0.0, 0.0, 0.0)),              # mixed widths
    ((0.0,), (0.0,)),                           # 1-D
    ((0.0, float("nan"), 0.0), (0, 0, 0)),      # non-finite
    (("a", 0.0, 0.0), (0, 0, 0)),               # not a number
])
def test_malformed_pair_refuses_at_construction(bad: Any) -> None:
    with pytest.raises(ValueError, match="jnt_offset"):
        Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=bad)


def test_corotational_3d_refuses_at_construction() -> None:
    with pytest.raises(ValueError, match="CorotCrdTransf3d"):
        Corotational(vecxz=(0.0, 0.0, 1.0), jnt_offset=J3)


def test_corotational_2d_constant_is_accepted() -> None:
    rec = RecordingEmitter()
    t = Corotational(jnt_offset=J2)
    assert t.jnt_offset == J2
    del rec


def test_corotational_2d_with_global_z_vecxz_emits_its_offsets(tmp_path: Path) -> None:
    """#2470 review 5: a 2-D pair with ``vecxz=(0, 0, 1)`` (what the 2-D
    bridge accepts) is not visibly 3-D, so construction must not refuse it."""
    ops = _bridge(
        2,
        lambda o: o.geomTransf.Corotational(vecxz=(0.0, 0.0, 1.0), jnt_offset=J2),
        lambda o: o.geomTransf.Linear(),
    )
    assert _deck(ops, tmp_path, "tcl")[0] == (
        "geomTransf Corotational 1 -jntOffset 0.1 -0.2 0.3 -0.4")


def test_corotational_3d_with_a_vecxz_still_refuses_at_build(tmp_path: Path) -> None:
    ops = _bridge(
        3,
        lambda o: o.geomTransf.Corotational(
            vecxz=(1.0, 0.0, 0.0), jnt_offset=InsertionPoint(depth=0.6)),
        lambda o: o.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0)),
    )
    with pytest.raises(BridgeError, match="CorotCrdTransf3d"):
        ops.tcl(str(tmp_path / "m.tcl"))


# ---------------------------------------------------------------------------
# Bridge emit: Tcl / py tokens, 3-D and 2-D
# ---------------------------------------------------------------------------

def _frame(ndm: int) -> FEMStub:
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0, 0, 0), (0, 0, 3), (5, 0, 3), (5, 4, 3)] if ndm == 3
        else [(0, 0, 0), (0, 3, 0), (5, 3, 0), (9, 3, 0)],
        node_pgs={"Base": [1]})
    elements = _ElementsStub(elem_pgs={
        "Cols": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
        "Beams": _ElementGroupView(ids=(2, 3), connectivity=((2, 3), (3, 4))),
    })
    return FEMStub(nodes=nodes, elements=elements)


def _deck(ops: apeSees, tmp_path: Path, kind: str) -> list[str]:
    path = tmp_path / f"m.{kind}"
    getattr(ops, kind)(str(path))
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if "geomTransf" in ln and not ln.lstrip().startswith("#")]


def _bridge(ndm: int, col: Any, beam: Any, **kw: Any) -> apeSees:
    ops = apeSees(cast("object", _frame(ndm)), **kw)
    ops.model(ndm=ndm, ndf=6 if ndm == 3 else 3)
    props: dict[str, float] = dict(A=0.01, E=200e9, Iz=1e-4)
    if ndm == 3:
        props.update(Iy=1e-4, G=80e9, J=1e-4)
    ops.element.elasticBeamColumn(pg="Cols", transf=col(ops), **props)
    ops.element.elasticBeamColumn(pg="Beams", transf=beam(ops), **props)
    ops.fix(pg="Base", dofs=(1,) * (6 if ndm == 3 else 3))
    return ops


@pytest.mark.parametrize("token", ["Linear", "PDelta"])
def test_3d_tokens_tcl_and_py(token: str, tmp_path: Path) -> None:
    ops = _bridge(
        3,
        lambda o: getattr(o.geomTransf, token)(vecxz=(1.0, 0.0, 0.0), jnt_offset=J3),
        lambda o: getattr(o.geomTransf, token)(vecxz=(0.0, 0.0, 1.0)),
    )
    tcl = _deck(ops, tmp_path, "tcl")
    assert tcl == [
        f"geomTransf {token} 1 1.0 0.0 0.0 -jntOffset 0.1 -0.2 0.3 -0.4 0.5 -0.6",
        f"geomTransf {token} 2 0.0 0.0 1.0",
    ]
    py = _deck(ops, tmp_path, "py")
    assert any(f"'{token}', 1, 1.0, 0.0, 0.0, '-jntOffset', 0.1, -0.2, 0.3, "
               "-0.4, 0.5, -0.6" in ln for ln in py), py


@pytest.mark.parametrize("token", ["Linear", "PDelta", "Corotational"])
def test_2d_tokens(token: str, tmp_path: Path) -> None:
    ops = _bridge(
        2,
        lambda o: getattr(o.geomTransf, token)(jnt_offset=J2),
        lambda o: getattr(o.geomTransf, token)(),
    )
    assert _deck(ops, tmp_path, "tcl") == [
        f"geomTransf {token} 1 -jntOffset 0.1 -0.2 0.3 -0.4",
        f"geomTransf {token} 2",
    ]


def test_2d_insertion_point_fans_out(tmp_path: Path) -> None:
    """Cardinal 8 on a 0.6 deep beam: both beams run along +X, so one
    line; 2-D local 2 is local y (x turned +90 deg = +Y). A pure cardinal
    shift is the same at both ends, so a collinear run takes it."""
    ops = _bridge(
        2,
        lambda o: o.geomTransf.Linear(),
        lambda o: o.geomTransf.Linear(jnt_offset=InsertionPoint(
            cardinal_point=8, depth=0.6)),
    )
    assert _deck(ops, tmp_path, "tcl") == [
        "geomTransf Linear 1",
        "geomTransf Linear 2 -jntOffset 0.0 -0.3 0.0 -0.3",
    ]


def test_2d_end_length_on_a_collinear_run_refuses(tmp_path: Path) -> None:
    """The two beam elements are collinear through node 3: a member meshed
    in two or a girder of two members (#2551)."""
    ops = _bridge(
        2,
        lambda o: o.geomTransf.Linear(),
        lambda o: o.geomTransf.Linear(jnt_offset=InsertionPoint(
            cardinal_point=8, depth=0.6, end_offset_i=0.2)),
    )
    with pytest.raises(BridgeError, match=r"meet at node 3 turning 0 deg, outside the 30-150 deg.*#2551"):
        ops.tcl(str(tmp_path / "m.tcl"))


def test_3d_insertion_point_fans_out_per_direction(tmp_path: Path) -> None:
    """Cardinal 7 (top-left) with ``axis_2="z"`` (``vecxz`` = +Z = ETABS 2):
    local 3 = 1 x 2, so left (-3) is +Y on the X beam and -X on the Y
    beam; the centroid sits at minus the cardinal point's position."""
    rule = InsertionPoint(cardinal_point=7, depth=0.6, width=0.3, axis_2="z",
                          end_offset_i=0.2)
    ops = _bridge(
        3,
        lambda o: o.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        lambda o: o.geomTransf.PDelta(vecxz=(0.0, 0.0, 1.0), jnt_offset=rule),
    )
    lines = _deck(ops, tmp_path, "tcl")
    assert lines[1] == "geomTransf PDelta 2 0.0 0.0 1.0 -jntOffset 0.2 -0.15 -0.3 0.0 -0.15 -0.3"
    assert lines[2] == "geomTransf PDelta 3 0.0 0.0 1.0 -jntOffset 0.15 0.2 -0.3 0.15 0.0 -0.3"
    # Each beam references its own line.
    elem = [ln for ln in (tmp_path / "m.tcl").read_text().splitlines()
            if ln.startswith("element elasticBeamColumn")]
    assert [ln.split()[-1] for ln in elem] == ["1", "2", "3"]


def test_insertion_point_dedupes_equal_offsets(tmp_path: Path) -> None:
    """Cardinal 8 is direction-free for horizontal beams (the axis drops
    d/2 along +Z whatever the plan direction): one line for both."""
    ops = _bridge(
        3,
        lambda o: o.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        lambda o: o.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), jnt_offset=InsertionPoint(
            cardinal_point=8, depth=0.6, axis_2="z")),
    )
    assert _deck(ops, tmp_path, "tcl")[1:] == [
        "geomTransf Linear 2 0.0 0.0 1.0 -jntOffset 0.0 0.0 -0.3 0.0 0.0 -0.3",
    ]


# ---------------------------------------------------------------------------
# Build-time refusals
# ---------------------------------------------------------------------------

def test_corotational_insertion_point_refuses_in_3d(tmp_path: Path) -> None:
    ops = _bridge(
        3,
        lambda o: o.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        lambda o: o.geomTransf.Corotational(
            jnt_offset=InsertionPoint(depth=0.6), orientation=None,
            vecxz=None),
        default_orientation=None,
    )
    with pytest.raises(BridgeError, match="CorotCrdTransf3d"):
        ops.tcl(str(tmp_path / "m.tcl"))


@pytest.mark.parametrize("ndm, jnt", [(3, J2), (2, J3)])
def test_width_mismatch_refuses(ndm: int, jnt: Any, tmp_path: Path) -> None:
    vec = (1.0, 0.0, 0.0) if ndm == 3 else None
    ops = _bridge(
        ndm,
        lambda o: o.geomTransf.Linear(vecxz=vec, jnt_offset=jnt),
        lambda o: o.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0) if ndm == 3 else None),
    )
    with pytest.raises(BridgeError, match="components per end"):
        ops.tcl(str(tmp_path / "m.tcl"))


def test_3d_rule_without_local_axes_refuses(tmp_path: Path) -> None:
    ops = _bridge(
        3,
        lambda o: o.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0)),
        lambda o: o.geomTransf.Linear(jnt_offset=InsertionPoint(depth=0.6)),
        default_orientation=None,
    )
    with pytest.raises(BridgeError, match="needs each element's local axes"):
        ops.tcl(str(tmp_path / "m.tcl"))


def test_2d_left_right_cardinal_refuses(tmp_path: Path) -> None:
    ops = _bridge(
        2,
        lambda o: o.geomTransf.Linear(),
        lambda o: o.geomTransf.Linear(jnt_offset=InsertionPoint(
            cardinal_point=7, depth=0.6, width=0.3)),
    )
    with pytest.raises(BridgeError, match="out of plane"):
        ops.tcl(str(tmp_path / "m.tcl"))


_PI2, _PJ2 = np.array([0.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0])


def _loaded_beam(jnt: Any, *, ndm: int = 2, params: dict | None = None,
                 vecxz: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 partitioned: bool = False, staged: bool = False) -> apeSees:
    """A fixed-fixed beam 1 -> 2 along +X, L = 4, under one ``from_model``
    global line load (``wy = -10`` unless ``params`` says otherwise).
    ``partitioned`` adds an unloaded beam 3 -> 4 owned by rank 0; the
    loaded one is rank 1's."""
    from apeGmsh._kernel.record_sets import ElementLoadSet
    from apeGmsh._kernel.records._loads import ElementLoadRecord

    ids = [1, 2]
    coords = [tuple(_PI2), tuple(_PJ2)]
    eids: tuple[int, ...] = (10,)
    conn: tuple[tuple[int, int], ...] = ((1, 2),)
    if partitioned:
        ids += [3, 4]
        coords += [(0.0, 5.0, 0.0), (4.0, 5.0, 0.0)]
        eids, conn = (10, 20), ((1, 2), (3, 4))
    nodes = _NodesStub(ids=ids, coords=coords, node_pgs={"Base": ids})
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs={
        "B": _ElementGroupView(ids=eids, connectivity=conn)}))
    if partitioned:
        fem.set_partitions([(0, [3, 4], [20]), (1, [1, 2], [10])])
    fem.elements.loads = ElementLoadSet([ElementLoadRecord(
        element_id=10, load_type="beamUniform",
        params=params or {"wx": 0.0, "wy": -10.0, "wz": 0.0}, pattern="grav")])
    ops = apeSees(cast("object", fem))
    ndf = 3 if ndm == 2 else 6
    ops.model(ndm=ndm, ndf=ndf)
    props: dict[str, float] = dict(A=0.01, E=200e9, Iz=1e-4)
    if ndm == 3:
        props.update(Iy=1e-4, G=80e9, J=1e-4)
        transf = ops.geomTransf.Linear(vecxz=vecxz, jnt_offset=jnt)
    else:
        transf = ops.geomTransf.Linear(jnt_offset=jnt)
    ops.element.elasticBeamColumn(pg="B", transf=transf, **props)
    ops.fix(pg="Base", dofs=(1,) * ndf)
    if staged:
        with ops.stage(name="s1") as st:
            with st.pattern(series=ops.timeSeries.Linear()) as p:
                p.from_model("grav")
            st.analysis(
                system=ops.system.BandGeneral(), numberer=ops.numberer.RCM(),
                constraints=ops.constraints.Plain(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                algorithm=ops.algorithm.Linear(), analysis=ops.analysis.Static())
            st.run(n_increments=1, dt=1.0)
    else:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("grav")
    return ops


def _pad3(v: Any) -> np.ndarray:
    out = np.zeros(3)
    out[:len(v)] = v
    return out


def _simpson(f: Any, u: float, v: float) -> np.ndarray:
    """Simpson's rule: exact for the (at most quadratic) integrands here."""
    return (v - u) / 6.0 * (f(u) + 4.0 * f(0.5 * (u + v)) + f(v))


def _emitted_resultant(calls: list, jnt: Any, ndm: int,
                       vecxz: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Force and moment about the origin of every load emitted on beam
    1 -> 2: each ``eleLoad -beamUniform`` read as OpenSees applies it (local
    axes, on the flexible element between the offset ends, ``aL``/``bL`` of
    its length), plus each nodal ``load`` at its joint."""
    q_i, q_j = _PI2 + _pad3(jnt[0]), _PJ2 + _pad3(jnt[1])
    flex = float(np.linalg.norm(q_j - q_i))
    e_x = (q_j - q_i) / flex
    if ndm == 2:
        e_y, e_z = np.array([-e_x[1], e_x[0], 0.0]), np.array([0.0, 0.0, 1.0])
    else:
        e_y = np.cross(np.asarray(vecxz, dtype=float), e_x)
        e_y /= np.linalg.norm(e_y)
        e_z = np.cross(e_x, e_y)
    force, moment = np.zeros(3), np.zeros(3)
    at = {1: _PI2, 2: _PJ2}
    n = 2 if ndm == 2 else 3
    for name, args, _kw in calls:
        if name == "eleLoad":
            w = [float(a) for a in args[args.index("-beamUniform") + 1:]]
            a, b = (w[n], w[n + 1]) if len(w) > n else (0.0, 1.0)
            wa = w[:n]
            wb = w[n + 2:] if len(w) > n + 2 else wa
            axes = (e_y, e_x) if ndm == 2 else (e_y, e_z, e_x)
            qa = sum(c * ax for c, ax in zip(wa, axes))
            qb = sum(c * ax for c, ax in zip(wb, axes))
            u, v = a * flex, b * flex

            def q(s: float, qa: Any = qa, qb: Any = qb, u: float = u,
                  v: float = v) -> np.ndarray:
                return qa + (qb - qa) * (s - u) / (v - u)
            force += _simpson(q, u, v)
            moment += _simpson(lambda s, q=q: np.cross(q_i + s * e_x, q(s)), u, v)
        elif name == "load":
            node, comps = int(args[0]), [float(c) for c in args[1:]]
            f = _pad3(comps[:2]) if ndm == 2 else np.array(comps[:3])
            m = np.array([0.0, 0.0, comps[2]]) if ndm == 2 else np.array(comps[3:6])
            force += f
            moment += np.cross(at[node], f) + m
    return force, moment


def _full_length_resultant(jnt: Any, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """ETABS: the global load over the node chord ``[aL L, bL L]``, on the
    element axis (the chord shifted by the offsets' transverse part)."""
    e_x = np.array([1.0, 0.0, 0.0])
    d_i = _pad3(jnt[0])
    t = d_i - float(np.dot(d_i, e_x)) * e_x
    q1 = np.array([params.get(k, 0.0) for k in ("wx", "wy", "wz")])
    q2 = np.array([params.get(k + "_end", params.get(k, 0.0))
                   for k in ("wx", "wy", "wz")])
    u, v = params.get("aL", 0.0) * 4.0, params.get("bL", 1.0) * 4.0

    def q(s: float) -> np.ndarray:
        return q1 + (q2 - q1) * (s - u) / (v - u)
    force = _simpson(q, u, v)
    moment = _simpson(lambda s: np.cross(_PI2 + t + s * e_x, q(s)), u, v)
    return force, moment


def _load_calls(rec: RecordingEmitter) -> list[tuple[str, tuple]]:
    return [(c[0], c[1]) for c in rec.calls if c[0] in ("eleLoad", "load")]


def test_global_line_load_on_a_tilted_offset_axis_refuses() -> None:
    ops = _loaded_beam(((0.0, 0.0), (0.0, -0.5)))   # J end dropped: axis tilts
    with pytest.raises(BridgeError, match="turns the element axis"):
        ops.build().emit(RecordingEmitter())


def test_global_line_load_with_overlapping_end_zones_refuses() -> None:
    ops = _loaded_beam(((2.5, 0.0), (-2.5, 0.0)))   # zones longer than the beam
    with pytest.raises(BridgeError, match="no flexible length"):
        ops.build().emit(RecordingEmitter())


def test_global_line_load_on_end_zones_loads_the_full_length() -> None:
    """#2470 review 1, maintainer ruling 2026-10-22: w L reaches the joints,
    as in ETABS. Hand calc, w = 10 on L = 4 with 0.3 m rigid ends: the
    flexible 3.4 m keeps ``-beamUniform -10``; each zone puts F = -w a = -3
    and M = -/+ w a^2 / 2 = -/+ 0.45 on its joint. The emit lost 6 of 40."""
    jnt = ((0.3, 0.0), (-0.3, 0.0))
    rec = RecordingEmitter()
    _loaded_beam(jnt).build().emit(rec)
    calls = _load_calls(rec)
    # The joint loads precede the eleLoad, the order an H5 replay writes.
    assert calls[:2] == [
        ("load", (1, 0.0, pytest.approx(-3.0), pytest.approx(-0.45))),
        ("load", (2, 0.0, pytest.approx(-3.0), pytest.approx(0.45))),
    ]
    assert calls[2][0] == "eleLoad" and calls[2][1][3:] == ("-beamUniform", -10.0, 0.0)
    force, moment = _emitted_resultant(rec.calls, jnt, 2)
    assert force[1] == pytest.approx(-40.0, rel=1e-12)       # w L, not w (L - 0.6)
    assert moment[2] == pytest.approx(-80.0, rel=1e-12)      # -w L^2 / 2 about node 1


@pytest.mark.parametrize("jnt, params", [
    (((0.3, -0.3), (-0.3, -0.3)), {"wx": 5.0, "wy": -10.0}),
    (((0.3, 0.0), (-0.5, 0.0)), {"wy": -10.0, "wy_end": -20.0, "wx": 2.0,
                                 "aL": 0.0, "bL": 0.5}),
    (((0.3, 0.0), (-0.5, 0.0)), {"wy": -10.0, "wy_end": -4.0,
                                 "aL": 0.25, "bL": 0.95}),
    (((0.3, 0.0), (-0.5, 0.0)), {"wy": -10.0, "aL": 0.0, "bL": 0.05}),
    (((0.3, 0.0), (-0.5, 0.0)), {"wy": -10.0, "aL": 0.9, "bL": 1.0}),
    (((-0.2, 0.0), (0.0, 0.0)), {"wy": -10.0}),
], ids=["shift_and_ends", "trapezoid_over_i_zone", "partial_into_j_zone",
        "i_zone_only", "j_zone_only", "outward_end"])
def test_2d_line_load_over_end_zones_keeps_the_full_resultant(
        jnt: Any, params: dict) -> None:
    full = {"wx": 0.0, "wy": 0.0, "wz": 0.0, **params}
    rec = RecordingEmitter()
    _loaded_beam(jnt, params=full).build().emit(rec)
    force, moment = _emitted_resultant(rec.calls, jnt, 2)
    want_f, want_m = _full_length_resultant(jnt, full)
    np.testing.assert_allclose(force, want_f, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(moment, want_m, rtol=1e-12, atol=1e-12)


def test_load_wholly_on_an_end_zone_emits_no_ele_load() -> None:
    rec = RecordingEmitter()
    _loaded_beam(((0.3, 0.0), (-0.5, 0.0)), params={
        "wx": 0.0, "wy": -10.0, "wz": 0.0, "aL": 0.0, "bL": 0.05}).build().emit(rec)
    assert [name for name, _a in _load_calls(rec)] == ["load"]


def test_cardinal_shift_without_end_zones_adds_no_joint_load() -> None:
    rec = RecordingEmitter()
    _loaded_beam(((0.0, -0.3), (0.0, -0.3))).build().emit(rec)
    assert [name for name, _a in _load_calls(rec)] == ["eleLoad"]


@pytest.mark.parametrize("jnt", [
    ((0.3, 0.0, -0.3), (-0.3, 0.0, -0.3)),       # hung 0.3 m, 0.3 m end zones
    ((0.25, 0.15, -0.3), (-0.4, 0.15, -0.3)),    # corner-like, unequal zones
], ids=["hung_ends", "corner_unequal"])
@pytest.mark.parametrize("vecxz", [(0.0, 0.0, 1.0), (0.0, -1.0, 0.0)],
                         ids=["vecxz_z", "vecxz_-y"])
def test_3d_line_load_over_end_zones_keeps_the_full_resultant(
        jnt: Any, vecxz: tuple[float, float, float]) -> None:
    params = {"wx": 1.0, "wy": 3.0, "wz": -10.0, "wz_end": -6.0}
    rec = RecordingEmitter()
    _loaded_beam(jnt, ndm=3, params=params, vecxz=vecxz).build().emit(rec)
    force, moment = _emitted_resultant(rec.calls, jnt, 3, vecxz)
    want_f, want_m = _full_length_resultant(jnt, params)
    np.testing.assert_allclose(force, want_f, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(moment, want_m, rtol=1e-12, atol=1e-12)
    assert [name for name, _a in _load_calls(rec)] == ["load", "load", "eleLoad"]


def _staged_gated_with_offsets() -> apeSees:
    """ADR 0099 S7: a stage-activated gated quad brackets mid-deck, so the
    Tcl deck re-declares the purged ``geomTransf`` lines at bracket close
    (``replay_builder_scoped_declarations`` on the bridge, the staged
    ``_replay_staged_into`` on H5 replay). The beam's 2-D transform carries
    a constant offset, so it is a bare-2-D ``line`` entry of the log."""
    from tests.opensees.h5.test_h5_builder_scope_replay import build_mixed_ndf_fem

    ops = apeSees(build_mixed_ndf_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    sec = ops.section.Elastic(E=2.0e8, A=0.01, Iz=1e-5)
    transf = ops.geomTransf.Linear(jnt_offset=((0.0, 0.1), (0.0, -0.1)))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.dispBeamColumn(pg="Liner", transf=transf, integration=integ)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="LinerBase", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.stage(name="dig") as s:
        s.activate(pgs=["Rock"])
        s.fix(pg="Base", dofs=(1, 1))
        with s.pattern(series=ts) as p:
            p.load(pg="LinerBase", forces=(10.0, 0.0, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.5),
            constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(), analysis=ops.analysis.Static())
        s.run(n_increments=2)
    return ops


@pytest.mark.parametrize("route", ["bridge", "h5_replay"])
def test_stage_bracket_redeclares_the_offset_line(route: str, tmp_path: Path) -> None:
    """#2470 review 2: the re-declared ``geomTransf`` after the stage
    bracket's ``model`` line keeps its ``-jntOffset`` on both routes."""
    from apeGmsh.opensees import OpenSeesModel

    ops = _staged_gated_with_offsets()
    if route == "bridge":
        path = tmp_path / "m.tcl"
        ops.tcl(str(path))
        text = path.read_text(encoding="utf-8")
    else:
        path = tmp_path / "m.h5"
        ops.h5(str(path))
        text = OpenSeesModel.from_h5(str(path)).build("tcl")
    lines = [ln.strip() for ln in text.splitlines()]
    models = [i for i, ln in enumerate(lines) if ln.startswith("model ")]
    assert len(models) == 3      # header, the stage bracket, its restore
    transf = [(i, ln) for i, ln in enumerate(lines) if ln.startswith("geomTransf")]
    want = "geomTransf Linear 1 -jntOffset 0.0 0.1 0.0 -0.1"
    assert [ln for _i, ln in transf] == [want, want]
    assert transf[0][0] < models[1] and transf[1][0] > models[-1]


@pytest.mark.parametrize("staged", [False, True], ids=["flat", "staged"])
def test_partitioned_end_zone_loads_land_once_on_the_owner_rank(staged: bool) -> None:
    jnt = ((0.3, 0.0), (-0.3, 0.0))
    rec = RecordingEmitter()
    _loaded_beam(jnt, partitioned=True, staged=staged).build().emit(rec)
    rank = None
    by_rank: list[tuple[int | None, str]] = []
    beam_calls = []
    for name, args, kw in rec.calls:
        if name == "partition_open":
            rank = int(args[0])
        elif name == "partition_close":
            rank = None
        elif name in ("eleLoad", "load"):
            by_rank.append((rank, name))
            beam_calls.append((name, args, kw))
    assert by_rank == [(1, "load"), (1, "load"), (1, "eleLoad")]
    force, moment = _emitted_resultant(beam_calls, jnt, 2)
    assert force[1] == pytest.approx(-40.0, rel=1e-12)
    assert moment[2] == pytest.approx(-80.0, rel=1e-12)


# ---------------------------------------------------------------------------
# The helper: ETABS sign conventions
# ---------------------------------------------------------------------------

_PI, _PJ = (0.0, 0.0, 3.0), (6.0, 0.0, 3.0)   # beam along +X


@pytest.mark.parametrize("point, c2, c3", [
    (1, -0.3, -0.15), (2, -0.3, 0.0), (3, -0.3, 0.15),
    (4, 0.0, -0.15), (5, 0.0, 0.0), (6, 0.0, 0.15),
    (7, 0.3, -0.15), (8, 0.3, 0.0), (9, 0.3, 0.15),
    (10, 0.0, 0.0), (11, 0.0, 0.0),
])
@pytest.mark.parametrize("axis_2, vecxz", [
    ("z", (0.0, 0.0, 1.0)),    # vecxz along ETABS 2: OpenSees z = +Z
    ("y", (0.0, -1.0, 0.0)),   # vecxz along ETABS 3: OpenSees y = +Z
])
def test_cardinal_points_on_an_x_beam(point: int, c2: float, c3: float,
                                      axis_2: Any, vecxz: Any) -> None:
    """ETABS 2 = +Z (up), 3 = 1 x 2 = X x Z = -Y. The offset is minus the
    cardinal point's (c2, c3): ``-(c2 * Z + c3 * (-Y))``."""
    d_i, d_j = insertion_point_offsets(
        p_i=_PI, p_j=_PJ, cardinal_point=point, depth=0.6, width=0.3,
        vecxz=vecxz, axis_2=axis_2)
    want = (0.0, c3, -c2)
    assert d_i == pytest.approx(want, abs=1e-15)
    assert d_j == pytest.approx(want, abs=1e-15)


def test_end_offsets_run_inward_along_the_axis() -> None:
    d_i, d_j = insertion_point_offsets(
        p_i=_PI, p_j=_PJ, depth=0.6, end_offset_i=0.4, end_offset_j=0.5,
        rigid_zone_factor=0.5)
    assert d_i == pytest.approx((0.2, 0.0, 0.0))
    assert d_j == pytest.approx((-0.25, 0.0, 0.0))


def test_end_offsets_follow_the_offset_axis() -> None:
    # Cardinal 8 then 0.5 m rigid zones: the zones run along the hung axis.
    d_i, d_j = insertion_point_offsets(
        p_i=_PI, p_j=_PJ, cardinal_point=8, depth=0.6, vecxz=(0, 0, 1),
        axis_2="z", end_offset_i=0.5, end_offset_j=0.5)
    assert d_i == pytest.approx((0.5, 0.0, -0.3))
    assert d_j == pytest.approx((-0.5, 0.0, -0.3))


def test_local_joint_offsets_use_the_nominal_axes() -> None:
    # (1, 2, 3) = (0, 0.1, 0.2) local; 2 = +Z, 3 = -Y.
    d_i, _ = insertion_point_offsets(
        p_i=_PI, p_j=_PJ, depth=0.6, vecxz=(0, 0, 1), axis_2="z",
        offset_i=(0.0, 0.1, 0.2), offset_system="local")
    assert d_i == pytest.approx((0.0, -0.2, 0.1))


def test_global_joint_offsets_add_to_the_cardinal_shift() -> None:
    d_i, d_j = insertion_point_offsets(
        p_i=_PI, p_j=_PJ, cardinal_point=8, depth=0.6, vecxz=(0, 0, 1),
        axis_2="z", offset_i=(0.0, 0.0, -0.3), offset_j=(0.0, 0.1, 0.0))
    assert d_i == pytest.approx((0.0, 0.0, -0.6))
    assert d_j == pytest.approx((0.0, 0.1, -0.3))


def test_mirror_flips_an_asymmetric_box() -> None:
    # bbox (min2, max2, min3, max3) = T-beam-like: flange to +3.
    box = (-0.4, 0.2, -0.1, 0.5)
    base = insertion_point_offsets(p_i=_PI, p_j=_PJ, cardinal_point=3,
                                   bbox=box, vecxz=(0, 0, 1), axis_2="z")
    mir = insertion_point_offsets(p_i=_PI, p_j=_PJ, cardinal_point=3,
                                  bbox=box, vecxz=(0, 0, 1), axis_2="z",
                                  mirror_2=True)
    # right (+3) face: max3 = 0.5, mirrored -min3 = 0.1; -3 is +Y.
    assert base[0] == pytest.approx((0.0, 0.5, 0.4))
    assert mir[0] == pytest.approx((0.0, 0.1, 0.4))


def test_shear_center_of_an_asymmetric_box_refuses() -> None:
    with pytest.raises(ValueError, match="doubly symmetric"):
        insertion_point_offsets(p_i=_PI, p_j=_PJ, cardinal_point=11,
                                bbox=(-0.4, 0.2, -0.1, 0.1), vecxz=(0, 0, 1))


@pytest.mark.parametrize("kw, match", [
    (dict(cardinal_point=12, depth=0.6), "1-11"),
    (dict(depth=0.6, width=0.3, bbox=(-1, 1, -1, 1)), "not both"),
    (dict(), "depth="),
    (dict(depth=0.6, rigid_zone_factor=1.5), r"\[0, 1\]"),
    (dict(depth=0.6, end_offset_i=3.5, end_offset_j=3.0), "reach past"),
    (dict(cardinal_point=8, depth=0.6), "pass vecxz="),
    (dict(cardinal_point=8, depth=0.6, vecxz=(1, 0, 0)), "parallel"),
])
def test_helper_refusals(kw: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        insertion_point_offsets(p_i=_PI, p_j=_PJ, **kw)


def test_2d_helper() -> None:
    d_i, d_j = insertion_point_offsets(
        p_i=(0.0, 0.0), p_j=(0.0, 4.0), cardinal_point=2, depth=0.5)
    # Column along +Y: local y = x turned +90 deg = -X; bottom (-2) on the
    # joint puts the centroid at +d/2 along +2 = -X.
    assert d_i == pytest.approx((-0.25, 0.0))
    assert d_j == pytest.approx((-0.25, 0.0))


# ---------------------------------------------------------------------------
# H5: the offsets archive and read back
# ---------------------------------------------------------------------------

def test_h5_round_trips_the_offsets(tmp_path: Path) -> None:
    rule = InsertionPoint(cardinal_point=7, depth=0.6, width=0.3, axis_2="z",
                          end_offset_i=0.2)
    ops = _bridge(
        3,
        lambda o: o.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0), jnt_offset=J3),
        lambda o: o.geomTransf.PDelta(vecxz=(0.0, 0.0, 1.0), jnt_offset=rule),
    )
    path = tmp_path / "m.h5"
    ops.h5(str(path))
    with h5_reader.open(str(path)) as model:
        got = {r.tag: (r.type_token, r.vec, r.jnt_offset) for r in model.transforms()}
    assert got[1] == ("Linear", (1.0, 0.0, 0.0), J3_FLAT)
    assert got[2][2] == pytest.approx((0.2, -0.15, -0.3, 0.0, -0.15, -0.3))
    assert got[3][2] == pytest.approx((0.15, 0.2, -0.3, 0.15, 0.0, -0.3))
    with h5py.File(str(path), "r") as f:
        shapes = {n: g["jnt_offset"].shape for n, g in f["opensees/transforms"].items()}
    assert set(shapes.values()) == {(1, 6)}


def test_h5_offset_free_transform_writes_no_dataset(tmp_path: Path) -> None:
    ops = _bridge(
        2,
        lambda o: o.geomTransf.Linear(jnt_offset=J2),
        lambda o: o.geomTransf.Linear(),
    )
    path = tmp_path / "m.h5"
    ops.h5(str(path))
    with h5py.File(str(path), "r") as f:
        grp = f["opensees/transforms"]
        has = {n: "jnt_offset" in g for n, g in grp.items()}
        widths = {n: g["jnt_offset"].shape for n, g in grp.items() if "jnt_offset" in g}
    assert sorted(has.values()) == [False, True]
    assert list(widths.values()) == [(1, 4)]
    with h5_reader.open(str(path)) as model:
        recs = sorted(model.transforms(), key=lambda r: r.tag)
    assert [r.jnt_offset for r in recs] == [J2_FLAT, ()]


def test_h5_malformed_offset_refuses(tmp_path: Path) -> None:
    ops = _bridge(
        2,
        lambda o: o.geomTransf.Linear(jnt_offset=J2),
        lambda o: o.geomTransf.Linear(),
    )
    path = tmp_path / "m.h5"
    ops.h5(str(path))
    with h5py.File(str(path), "r+") as f:
        for g in f["opensees/transforms"].values():
            if "jnt_offset" in g:
                del g["jnt_offset"]
                g.create_dataset("jnt_offset", data=np.zeros((1, 5)))
    with h5_reader.open(str(path)) as model:
        with pytest.raises(h5_reader.MalformedH5Error, match="jnt_offset"):
            model.transforms()
