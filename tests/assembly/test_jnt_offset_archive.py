"""Joint offsets survive the archive: H5 replay and a rotated instance.

* ``OpenSeesModel.from_h5(...).build("tcl")`` writes every ``geomTransf``
  line of the source deck, ``-jntOffset`` included (the fan-out of an
  ``InsertionPoint`` rule replays as its resolved lines).
* An ``Assembly`` instance turned by ``R`` (120 deg about (1, 1, 1), the
  cyclic map x -> y -> z -> x) writes ``R d`` for each offset half: an
  offset is a direction (node -> element end), never translated.
* A gmsh curve meshed into several elements is one member (#2551): its
  end offsets sit on its two end elements. The split survives the
  neutral round trip (the per-element source curve is in model.h5), the
  H5 replay, and an Assembly rehydrate (each archived end-sensitive line
  covers one element per curve, so re-splitting it is the identity).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.joint_offset import InsertionPoint

ROT = ((1.0, 1.0, 1.0), 2.0 * math.pi / 3.0)
COL_JNT = ((0.05, -0.04, 0.0), (0.05, -0.04, -0.3))
PROPS = dict(A=0.25, E=30e9, Iz=5e-3, Iy=3e-3, G=12e9, J=6e-3)


def _frame_fem(n_nodes: int = 2):
    """One column (0,0,0)-(0,0,3) and two beams at z = 3 along +X, +Y,
    each curve meshed with ``n_nodes`` nodes (``n_nodes - 1`` elements)."""
    with apeGmsh(model_name="jnt", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(0.0, 0.0, 3.0)
        p2 = geo.add_point(5.0, 0.0, 3.0)
        p3 = geo.add_point(0.0, 4.0, 3.0)
        col = geo.add_line(p0, p1)
        bx = geo.add_line(p1, p2)
        by = geo.add_line(p1, p3)
        g.model.sync()
        g.physical.add(1, [col], name="Col")
        g.physical.add(1, [bx, by], name="Beams")
        g.physical.add(0, [p0], name="Base")
        g.mesh.structured.set_transfinite_curve([col, bx, by], n_nodes)
        g.mesh.generation.generate(1)
        return g.mesh.queries.get_fem_data(dim=1)


def _archive(tmp_path: Path, fem: Any = None) -> tuple[Path, list[str]]:
    from apeGmsh.opensees import apeSees

    ops = apeSees(_frame_fem() if fem is None else fem)
    ops.model(ndm=3, ndf=6)
    tc = ops.geomTransf.PDelta(vecxz=(1.0, 0.0, 0.0), jnt_offset=COL_JNT, name="tc")
    tb = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0), name="tb", jnt_offset=InsertionPoint(
        cardinal_point=9, depth=0.6, width=0.3, axis_2="z", end_offset_i=0.25))
    ops.element.elasticBeamColumn(pg="Col", transf=tc, **PROPS)
    ops.element.elasticBeamColumn(pg="Beams", transf=tb, **PROPS)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    path = tmp_path / "jnt.h5"
    ops.h5(str(path))
    deck = tmp_path / "src.tcl"
    ops.tcl(str(deck))
    return path, _transf_lines(deck.read_text(encoding="utf-8"))


def _transf_lines(text: str) -> list[str]:
    return sorted(ln.strip() for ln in text.splitlines()
                  if ln.strip().startswith("geomTransf"))


def _offsets(lines: list[str]) -> list[tuple[float, ...]]:
    out = []
    for ln in lines:
        tok = ln.split()
        k = tok.index("-jntOffset")
        out.append(tuple(round(float(v), 12) + 0.0 for v in tok[k + 1:k + 7]))
    return sorted(out)


def test_h5_replay_writes_the_offset_lines(tmp_path: Path) -> None:
    from apeGmsh.opensees import OpenSeesModel

    path, src = _archive(tmp_path)
    assert len(src) == 3 and all("-jntOffset" in ln for ln in src)
    replayed = _transf_lines(OpenSeesModel.from_h5(str(path)).build("tcl"))
    assert replayed == src


def test_model_data_rewrite_keeps_the_offsets(tmp_path: Path) -> None:
    import warnings

    from apeGmsh.opensees import ModelData
    from apeGmsh.opensees.emitter import h5_reader

    path, _src = _archive(tmp_path)
    with h5_reader.open(str(path)) as m:
        want = sorted((r.tag, r.jnt_offset) for r in m.transforms())
    out = tmp_path / "rewritten.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ModelData.from_h5(str(path)).write(str(out))
    with h5_reader.open(str(out)) as m:
        got = sorted((r.tag, r.jnt_offset) for r in m.transforms())
    assert got == want and all(j for _t, j in got)


def test_rotated_instance_turns_each_offset_half(tmp_path: Path) -> None:
    from apeGmsh.assembly import Assembly

    path, src = _archive(tmp_path)
    ops = (Assembly("turned")
           .instance("c", path, rotate=ROT, translate=(100.0, -50.0, 7.0))
           .bridge(ndm=3, ndf=6))
    out = tmp_path / "turned.tcl"
    ops.tcl(str(out), flat=True)
    got = _offsets(_transf_lines(out.read_text(encoding="utf-8")))

    def r(v: np.ndarray) -> np.ndarray:   # (x, y, z) -> (z, x, y)
        return np.array([v[2], v[0], v[1]])

    want = []
    for d in _offsets(src):
        a, b = np.asarray(d[:3]), np.asarray(d[3:])
        want.append(tuple(round(float(c), 12) + 0.0
                          for c in (*r(a), *r(b))))
    assert got == sorted(want)


# ---------------------------------------------------------------------------
# A gmsh curve meshed into several elements is one member (#2551)
# ---------------------------------------------------------------------------

#: By hand, the 3-element frame of ``_archive`` (``_frame_fem(4)``):
#: the column's constant pair has an end length at j only, so its first two
#: elements keep the transverse part; cardinal 9 with axis_2 = z hangs each
#: beam 0.3 below (local 2 = +Z) and 0.15 right (local 3 = 1 x 2), plus a
#: 0.25 m i-end zone on its first element only.
_COL_T = (0.05, -0.04, 0.0)
_X_T, _Y_T = (0.0, 0.15, -0.3), (-0.15, 0.0, -0.3)


def _element_tails(text: str) -> dict[tuple[tuple[float, ...], tuple[float, ...]],
                                      tuple[float, ...]]:
    """``(p_i, p_j) -> -jntOffset tail`` of every beam element in a deck."""
    nodes: dict[str, tuple[float, ...]] = {}
    tails: dict[str, tuple[float, ...]] = {}
    out = {}
    for ln in text.splitlines():
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "node":
            nodes[tok[1]] = tuple(round(float(v), 9) + 0.0 for v in tok[2:5])
        elif tok[0] == "geomTransf":
            k = tok.index("-jntOffset") if "-jntOffset" in tok else len(tok)
            tails[tok[2]] = tuple(round(float(v), 12) + 0.0 for v in tok[k + 1:k + 7])
        elif tok[:2] == ["element", "elasticBeamColumn"]:
            out[(nodes[tok[3]], nodes[tok[4]])] = tails[tok[-1]]
    return out


def _want_tail(p_i: tuple[float, ...], p_j: tuple[float, ...]) -> tuple[float, ...]:
    if p_i[:2] == p_j[:2] == (0.0, 0.0):                       # the column
        end = (0.05, -0.04, -0.3) if p_j[2] == 3.0 else _COL_T
        return (*_COL_T, *end)
    if p_i[1] == p_j[1] == 0.0:                                # the +X beam
        start = (0.25, 0.15, -0.3) if p_i[0] == 0.0 else _X_T
        return (*start, *_X_T)
    start = (-0.15, 0.25, -0.3) if p_i[1] == 0.0 else _Y_T     # the +Y beam
    return (*start, *_Y_T)


def test_meshed_curves_take_their_offsets_at_their_ends(tmp_path: Path) -> None:
    from apeGmsh.opensees import OpenSeesModel

    path, src = _archive(tmp_path, _frame_fem(4))
    deck = (tmp_path / "src.tcl").read_text(encoding="utf-8")
    got = _element_tails(deck)
    assert len(got) == 9
    for (p_i, p_j), tail in got.items():
        assert tail == _want_tail(p_i, p_j), (p_i, p_j)
    replayed = _transf_lines(OpenSeesModel.from_h5(str(path)).build("tcl"))
    assert replayed == src


def test_neutral_round_trip_keeps_the_member_split(tmp_path: Path) -> None:
    """model.h5 carries each element's source curve: the reloaded FEM
    splits each curve exactly as the live one does."""
    from apeGmsh.mesh.FEMData import FEMData

    fem = _frame_fem(4)
    fem.to_h5(str(tmp_path / "model.h5"))
    (tmp_path / "live").mkdir()
    (tmp_path / "re").mkdir()
    _p1, live = _archive(tmp_path / "live", fem)
    _p2, reloaded = _archive(tmp_path / "re", FEMData.from_h5(str(tmp_path / "model.h5")))
    assert reloaded == live and len(live) == 6


def test_rotated_instance_of_a_meshed_member_turns_each_offset(tmp_path: Path) -> None:
    """Rehydrate: every archived end-sensitive line covers one element
    per curve, so re-splitting it is the identity and each element's
    offset comes back turned by R."""
    from apeGmsh.assembly import Assembly

    path, _src = _archive(tmp_path, _frame_fem(4))
    want = sorted(_element_tails((tmp_path / "src.tcl").read_text(encoding="utf-8")).values())
    ops = (Assembly("turned")
           .instance("c", path, rotate=ROT, translate=(100.0, -50.0, 7.0))
           .bridge(ndm=3, ndf=6))
    out = tmp_path / "turned.tcl"
    ops.tcl(str(out), flat=True)
    got = sorted(_element_tails(out.read_text(encoding="utf-8")).values())

    def r(v: tuple[float, ...]) -> tuple[float, ...]:   # (x, y, z) -> (z, x, y)
        return (v[2], v[0], v[1])
    assert got == sorted(tuple(round(c, 12) + 0.0 for c in (*r(t[:3]), *r(t[3:])))
                         for t in want)


def test_a_curved_gmsh_curve_with_end_offsets_refuses_naming_it() -> None:
    """Maintainer ruling (2026-10-26, #2551 Q3): split the arc into
    straight curves, or drop the end offsets."""
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.build.errors import BridgeError
    from apeGmsh.opensees.emitter.recording import RecordingEmitter

    with apeGmsh(model_name="arc", verbose=False) as g:
        geo = g.model.geometry
        c = geo.add_point(0.0, 0.0, 3.0)
        a = geo.add_point(4.0, 0.0, 3.0)
        b = geo.add_point(0.0, 4.0, 3.0)
        arc = geo.add_arc(a, c, b)
        g.model.sync()
        g.physical.add(1, [arc], name="Arc")
        g.physical.add(0, [a], name="Base")
        g.mesh.structured.set_transfinite_curve([arc], 5)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.elasticBeamColumn(pg="Arc", transf=ops.geomTransf.Linear(
        vecxz=(0.0, 0.0, 1.0), jnt_offset=InsertionPoint(
            cardinal_point=8, depth=0.6, width=0.3, axis_2="z", end_offset_i=0.25)),
        **PROPS)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    with pytest.raises(BridgeError, match=rf"curve {arc} \(elements .*\) is not straight"):
        ops.build().emit(RecordingEmitter())


@pytest.mark.parametrize("staged", [False, True], ids=["flat", "staged"])
def test_h5_replay_writes_a_split_members_pattern_block(staged: bool, tmp_path: Path) -> None:
    """A 2-element gmsh member (0,0)-(4,0) with 0.3 m end zones under a
    from_model line load: the joint loads land at its two ends only (one
    each, before the eleLoads), and the H5 replay writes the same block."""
    from apeGmsh._kernel.record_sets import ElementLoadSet
    from apeGmsh._kernel.records._loads import ElementLoadRecord
    from apeGmsh.opensees import OpenSeesModel, apeSees

    with apeGmsh(model_name="span", verbose=False) as g:
        geo = g.model.geometry
        p0 = geo.add_point(0.0, 0.0, 0.0)
        p1 = geo.add_point(4.0, 0.0, 0.0)
        ln = geo.add_line(p0, p1)
        g.model.sync()
        g.physical.add(1, [ln], name="Beams")
        g.physical.add(0, [p0, p1], name="Base")
        g.mesh.structured.set_transfinite_curve([ln], 3)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)
    eids = [int(e) for grp in fem.elements.select(pg="Beams").groups() for e in grp.ids]
    assert len(eids) == 2
    fem.elements.loads = ElementLoadSet([
        ElementLoadRecord(element_id=e, load_type="beamUniform",
                          params={"wx": 0.0, "wy": -10.0, "wz": 0.0}, pattern="dl")
        for e in eids])
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    ops.element.elasticBeamColumn(
        pg="Beams", transf=ops.geomTransf.Linear(jnt_offset=((0.3, 0.0), (-0.3, 0.0))),
        A=0.01, E=200e9, Iz=1e-4)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    if staged:
        with ops.stage(name="s1") as st:
            with st.pattern(series=ops.timeSeries.Linear()) as p:
                p.from_model("dl")
            st.analysis(
                system=ops.system.BandGeneral(), numberer=ops.numberer.RCM(),
                constraints=ops.constraints.Plain(),
                integrator=ops.integrator.LoadControl(dlam=1.0),
                algorithm=ops.algorithm.Linear(), analysis=ops.analysis.Static())
            st.run(n_increments=1, dt=1.0)
    else:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("dl")
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    h5 = tmp_path / "m.h5"
    ops.h5(str(h5))
    replayed = OpenSeesModel.from_h5(str(h5)).build("tcl")

    def block(text: str) -> list[str]:
        lines = [ln.strip() for ln in text.splitlines()]
        return [ln for ln in lines if ln.startswith(("load ", "eleLoad "))]
    want = block(deck.read_text(encoding="utf-8"))
    assert [ln.split()[0] for ln in want] == ["load", "load", "eleLoad", "eleLoad"]
    assert block(replayed) == want
