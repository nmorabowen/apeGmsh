"""LadrunoMembrane(pg=): per-panel frame filter and the interior-node skip.

Two side-by-side 4 m x 4 m panels share the edge x = 4 and each is meshed
by 2 x 2 quads, so each has one strictly interior node ((2, 2) and
(6, 2)).  Edge beams run along every grid edge of both perimeters, plus
one out-of-plane column under (4, 0).

* ``frames="Beams"`` (a group) keeps, per panel, only the beams on that
  panel's boundary: the fork kernel goes singular on a beam outside the
  panel or on an out-of-plane column, and transfers no load.
* An explicitly named frame outside the panel is refused.
* The interior nodes are touched only by the panels' gmsh cells, so they
  are not emitted, and any record that targets one is refused.
"""
from __future__ import annotations

import re
import warnings
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.pg_expand import MembraneFramesFilteredWarning
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

# "Beams" spans both panels, so most builds here drop frames per panel by
# design; the tests that pin that warning catch it themselves.
pytestmark = pytest.mark.filterwarnings(
    "ignore::apeGmsh.opensees._internal.build.pg_expand.MembraneFramesFilteredWarning")

# Grid nodes: x in 0, 2, 4, 6, 8 (columns 0..4), y in 0, 2, 4 (rows 0..2).
_XS = (0.0, 2.0, 4.0, 6.0, 8.0)
_YS = (0.0, 2.0, 4.0)


def _nid(ix: int, iy: int) -> int:
    return 1 + ix + 5 * iy


INTERIOR = (_nid(1, 1), _nid(3, 1))   # (2, 2) and (6, 2)
COL_BASE = 100                        # (4, 0, -3)


def _fem() -> FEMStub:
    ids = [_nid(ix, iy) for iy in range(3) for ix in range(5)]
    coords = [(_XS[ix], _YS[iy], 0.0) for iy in range(3) for ix in range(5)]
    ids.append(COL_BASE)
    coords.append((4.0, 0.0, -3.0))

    def quads(ix0: int) -> tuple[tuple[int, ...], ...]:
        return tuple(
            (_nid(ix, iy), _nid(ix + 1, iy), _nid(ix + 1, iy + 1), _nid(ix, iy + 1))
            for iy in range(2) for ix in (ix0, ix0 + 1)
        )

    beams: list[tuple[int, int]] = []
    for iy in (0, 2):                       # y = 0 and y = 4 edges
        beams += [(_nid(ix, iy), _nid(ix + 1, iy)) for ix in range(4)]
    for ix in (0, 2, 4):                    # x = 0, 4, 8 edges
        beams += [(_nid(ix, iy), _nid(ix, iy + 1)) for iy in range(2)]
    beam_ids = tuple(range(201, 201 + len(beams)))
    nodes = _NodesStub(
        ids=ids, coords=coords,
        node_pgs={
            "Base": [COL_BASE],
            "Corners": [_nid(0, 0), _nid(4, 0), _nid(0, 2), _nid(4, 2)],
            "PanelA": sorted({n for q in quads(0) for n in q}),
        },
    )
    # PanelA's own edge beams (x <= 4), as an element label.
    beams_a = [eid for eid, (i, j) in zip(beam_ids, beams)
               if max(_XS[(i - 1) % 5], _XS[(j - 1) % 5]) <= 4.0]
    elements = _ElementsStub(labels={"BeamsA": beams_a}, elem_pgs={
        "PanelA": _ElementGroupView(ids=(1, 2, 3, 4), connectivity=quads(0)),
        "PanelB": _ElementGroupView(ids=(5, 6, 7, 8), connectivity=quads(2)),
        "Beams": _ElementGroupView(ids=beam_ids, connectivity=tuple(beams)),
        "Col": _ElementGroupView(ids=(301,), connectivity=((COL_BASE, _nid(2, 0)),)),
        # A beam from PanelA's interior node to a corner (used by one test).
        "Link": _ElementGroupView(ids=(302,), connectivity=((INTERIOR[0], _nid(0, 0)),)),
    })
    fem = FEMStub(nodes=nodes, elements=elements)
    # FEMData always carries an interface list, which the interior-node
    # guard reads directly; the shared stub has none.
    fem.elements.interfaces = []  # type: ignore[attr-defined]
    return fem


def _bridge(
    frames: Any = "Beams", *, staged: bool = False, **mem: Any,
) -> "tuple[apeSees, Any, list[Any]]":
    """``staged=True`` leaves the panels' load pattern to the caller's stage."""
    fem = _fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    tc = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    beams = ops.element.elasticBeamColumn(
        pg="Beams", transf=t, A=0.1, E=30e6, Iz=1e-3, Iy=1e-3, G=12e6, J=1e-4)
    ops.element.elasticBeamColumn(
        pg="Col", transf=tc, A=0.1, E=30e6, Iz=1e-3, Iy=1e-3, G=12e6, J=1e-4)
    mems = [
        ops.element.LadrunoMembrane(
            pg=pg, frames=frames, pressure=-5.0, stiff=(30e6, 0.2, 0.2), **mem)
        for pg in ("PanelA", "PanelB")
    ]
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    if not staged:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            for m in mems:
                p.surface_load(ele=m)
    return ops, beams, mems


def _membrane_args(rec: RecordingEmitter) -> list[tuple[Any, ...]]:
    return [args for name, args, _kw in rec.calls
            if name == "element" and args[0] == "LadrunoMembrane"]


def _frames_of(args: tuple[Any, ...]) -> set[int]:
    i = args.index("-frames")
    out = set()
    for a in args[i + 1:]:
        if isinstance(a, str):
            break
        out.add(int(a))
    return out


def _beam_ends(rec: RecordingEmitter) -> dict[int, tuple[int, int]]:
    return {int(args[1]): (int(args[2]), int(args[3]))
            for name, args, _kw in rec.calls
            if name == "element" and args[0] == "elasticBeamColumn"}


class TestFrameFilter:
    def test_group_keeps_each_panels_own_beams_and_drops_the_column(self) -> None:
        ops, _beams, _mems = _bridge()
        rec = RecordingEmitter()
        ops.build().emit(rec)
        fem = ops._fem
        xyz = {int(n): c for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
        ends = _beam_ends(rec)
        col_tag = next(t for t, (i, j) in ends.items() if COL_BASE in (i, j))

        def on_panel(x0: float, x1: float) -> set[int]:
            return {
                t for t, (i, j) in ends.items()
                if t != col_tag and all(
                    x0 <= xyz[n][0] <= x1 and xyz[n][2] == 0.0 for n in (i, j))
            }

        a, b = _membrane_args(rec)
        assert _frames_of(a) == on_panel(0.0, 4.0)
        assert _frames_of(b) == on_panel(4.0, 8.0)
        assert len(_frames_of(a)) == 8 and len(_frames_of(b)) == 8
        # The shared x = 4 edge beams load both panels; the column neither.
        assert len(_frames_of(a) & _frames_of(b)) == 2
        assert col_tag not in _frames_of(a) | _frames_of(b)

    def test_explicit_tag_outside_the_panel_is_refused(self) -> None:
        ops, _beams, _mems = _bridge()
        rec = RecordingEmitter()
        ops.build().emit(rec)
        ends = _beam_ends(rec)
        fem = ops._fem
        xyz = {int(n): c for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
        # A beam of panel B only (both ends at x >= 6).
        outside = next(t for t, (i, j) in ends.items()
                       if xyz[i][0] >= 6.0 and xyz[j][0] >= 6.0)
        ops2, _b, _m = _bridge(frames=(outside,))
        with pytest.raises(BridgeError, match=r"PanelA.*does not lie in this panel"):
            ops2.build().emit(RecordingEmitter())

    def test_group_with_no_frame_in_the_panel_is_refused(self) -> None:
        ops, _beams, _mems = _bridge(frames="Col")
        with pytest.raises(BridgeError, match=r"PanelA.*none of its frames="):
            ops.build().emit(RecordingEmitter())

    def test_a_partly_dropped_group_warns_naming_the_dropped_frames(self) -> None:
        """``"Beams"`` spans both panels: each panel keeps its own 8 beams
        and drops the other panel's 6, and the build names those 6."""
        ops, _beams, _mems = _bridge()
        rec = RecordingEmitter()
        with pytest.warns(MembraneFramesFilteredWarning) as caught:
            ops.build().emit(rec)
        ends = _beam_ends(rec)
        col_tag = next(t for t, (i, j) in ends.items() if COL_BASE in (i, j))
        beam_tags = set(ends) - {col_tag}
        a, b = _membrane_args(rec)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, MembraneFramesFilteredWarning)]
        for pg, kept in (("PanelA", _frames_of(a)), ("PanelB", _frames_of(b))):
            (msg,) = [m for m in msgs if f"pg={pg!r}" in m]
            assert "frames= group 'Beams': 6 of its 14 frame element(s)" in msg
            named = re.search(r"OpenSees element\(s\) \[([0-9, ]+)\]", msg)
            assert named is not None
            assert {int(t) for t in named.group(1).split(",")} == beam_tags - kept

    def test_a_group_wholly_in_the_panel_does_not_warn(self) -> None:
        fem = _fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.1, E=30e6, Iz=1e-3, Iy=1e-3, G=12e6, J=1e-4)
        ops.element.LadrunoMembrane(pg="PanelA", frames="BeamsA")
        rec = RecordingEmitter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", MembraneFramesFilteredWarning)
            ops.build().emit(rec)
        (args,) = _membrane_args(rec)
        assert len(_frames_of(args)) == 8

    def test_support_nodes_keep_a_frameless_panel_legal(self) -> None:
        """Legal, but not silent: the whole "Col" group is dropped."""
        ops, _beams, _mems = _bridge(frames="Col", support_nodes="Corners")
        rec = RecordingEmitter()
        with pytest.warns(MembraneFramesFilteredWarning) as caught:
            ops.build().emit(rec)
        for args in _membrane_args(rec):
            assert "-frames" not in args
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, MembraneFramesFilteredWarning)]
        for pg in ("PanelA", "PanelB"):
            (msg,) = [m for m in msgs if f"pg={pg!r}" in m]
            assert "frames= group 'Col': all 1 frame element(s)" in msg


class TestInteriorNodes:
    def test_interior_nodes_are_not_emitted(self) -> None:
        ops, _beams, _mems = _bridge()
        rec = RecordingEmitter()
        ops.build().emit(rec)
        emitted = {int(args[0]) for name, args, _kw in rec.calls if name == "node"}
        all_ids = {int(n) for n in ops._fem.nodes.ids}
        assert emitted == all_ids - set(INTERIOR)

    def test_a_node_another_element_touches_stays(self) -> None:
        """An interior node a beam touches is part of the model."""
        fem = _fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.1, E=30e6, Iz=1e-3, Iy=1e-3, G=12e6, J=1e-4)
        # A beam from PanelA's interior node to a boundary node.
        ops.element.elasticBeamColumn(
            pg="Link", transf=t, A=0.1, E=30e6, Iz=1e-3, Iy=1e-3, G=12e6, J=1e-4)
        ops.element.LadrunoMembrane(pg="PanelA", frames="Beams")
        rec = RecordingEmitter()
        ops.build().emit(rec)
        emitted = {int(args[0]) for name, args, _kw in rec.calls if name == "node"}
        assert INTERIOR[0] in emitted

    def test_no_pg_membrane_emits_every_node(self) -> None:
        fem = _fem()
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            nodes=(_nid(0, 0), _nid(2, 0), _nid(2, 2), _nid(0, 2)))
        rec = RecordingEmitter()
        ops.build().emit(rec)
        emitted = {int(args[0]) for name, args, _kw in rec.calls if name == "node"}
        assert emitted == {int(n) for n in fem.nodes.ids}

    @pytest.mark.parametrize("verb", [
        "fix", "mass", "load", "sp", "recorder",
        "stage_fix", "stage_support", "stage_mass", "stage_recorder",
    ])
    def test_a_record_on_an_interior_node_is_refused(self, verb: str) -> None:
        ops, _beams, mems = _bridge(staged=verb.startswith("stage_"))
        if verb.startswith("stage_"):
            rec_spec = (
                ops.recorder.Node(file="d.out", response="disp", pg="PanelA", dofs=(3,))
                if verb == "stage_recorder" else None)
            with ops.stage(name="s1") as s:
                with s.pattern(series=ops.timeSeries.Linear()) as sp:
                    for m in mems:
                        sp.surface_load(ele=m)
                if verb == "stage_fix":
                    s.fix(nodes=(INTERIOR[0],), dofs=(0, 0, 1, 0, 0, 0))
                    hint = "support_nodes="
                elif verb == "stage_support":
                    s.support(pg="PanelA", dofs=(0, 0, 1, 0, 0, 0))
                    hint = "support_nodes="
                elif verb == "stage_mass":
                    s.mass(pg="PanelA", values=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
                    hint = "rho="
                else:
                    assert rec_spec is not None
                    s.recorder(rec_spec)
                    hint = "Record the panel's boundary"
                s.analysis(**_chain(ops))
                s.run(n_increments=1, dt=1.0)
        elif verb == "fix":
            ops.fix(nodes=(INTERIOR[0],), dofs=(0, 0, 1, 0, 0, 0))
            hint = "support_nodes="
        elif verb == "mass":
            ops.mass(pg="PanelA", values=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
            hint = "rho="
        elif verb == "load":
            with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
                p.load(node=INTERIOR[1], forces=(0.0, 0.0, -1.0, 0.0, 0.0, 0.0))
            hint = "surface_load"
        elif verb == "sp":
            with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
                p.sp(node=INTERIOR[1], dof=3, value=-0.01)
            hint = "boundary / frame nodes"
        else:
            ops.recorder.Node(file="d.out", response="disp", pg="PanelA", dofs=(3,))
            hint = "Record the panel's boundary"
        with pytest.raises(BridgeError, match=r"inside a LadrunoMembrane\(pg=\.\.\.\) panel") as ei:
            ops.build().emit(RecordingEmitter())
        assert hint in str(ei.value)
        assert str(INTERIOR[0]) in str(ei.value) or str(INTERIOR[1]) in str(ei.value)

    def test_records_on_boundary_nodes_stay_legal(self) -> None:
        ops, _beams, _mems = _bridge()
        ops.mass(pg="Corners", values=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
        ops.recorder.Node(file="d.out", response="disp", pg="Corners", dofs=(3,))
        ops.build().emit(RecordingEmitter())


def test_interior_node_ids_match_the_skip_set() -> None:
    from apeGmsh.opensees._internal.build.ndf import membrane_interior_nodes
    from apeGmsh.opensees._internal.types import Element

    ops, _beams, _mems = _bridge()
    elements = [p for p in ops._primitives if isinstance(p, Element)]
    assert membrane_interior_nodes(ops._fem, elements) == frozenset(INTERIOR)
    assert np.isin(np.asarray(INTERIOR), np.asarray(ops._fem.nodes.ids)).all()


def test_a_hole_node_no_element_touches_warns() -> None:
    """The membrane stiffens a node in its plane at most (this one has no
    in-plane stiffness at all), so an unframed hole node has free DOFs:
    the build says so."""
    from apeGmsh.opensees._internal.build.ndf import MembraneHoleNodeFreeWarning

    fem = _fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Beams", transf=t, A=0.1, E=30e6, Iz=1e-3, Iy=1e-3, G=12e6, J=1e-4)
    ops.element.LadrunoMembrane(
        nodes=(_nid(0, 0), _nid(4, 0), _nid(4, 2), _nid(0, 2)),
        holes=[(INTERIOR[0], INTERIOR[1], _nid(2, 1))])
    with pytest.warns(MembraneHoleNodeFreeWarning, match=r"2 node\(s\) of its hole"):
        ops.build().emit(RecordingEmitter())


def _chain(ops: apeSees) -> dict[str, object]:
    return {
        "test": ops.test.NormDispIncr(tol=1e-6, max_iter=10),
        "algorithm": ops.algorithm.Linear(),
        "integrator": ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.Plain(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def test_staged_emit_skips_the_interior_nodes() -> None:
    fem = _fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Beams", transf=t, A=0.1, E=30e6, Iz=1e-3, Iy=1e-3, G=12e6, J=1e-4)
    mems = [ops.element.LadrunoMembrane(pg=pg, frames="Beams", pressure=-5.0)
            for pg in ("PanelA", "PanelB")]
    ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
    with ops.stage(name="gravity") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            for m in mems:
                p.surface_load(ele=m)
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    emitted = {int(args[0]) for name, args, _kw in rec.calls if name == "node"}
    assert not set(INTERIOR) & emitted
    assert {int(n) for n in fem.nodes.ids} - emitted == set(INTERIOR)


def test_partitioned_emit_skips_the_interior_nodes() -> None:
    fem = _fem()
    # A macro-element lives wholly on one rank: the slab on rank 0.
    grid = sorted({_nid(ix, iy) for ix in range(5) for iy in range(3)})
    fem.set_partitions([(0, grid, list(range(1, 9))), (1, [COL_BASE], [])])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    ops.element.LadrunoMembrane(pg="PanelA", support_nodes="Corners")
    ops.element.LadrunoMembrane(pg="PanelB", support_nodes="Corners")
    rec = RecordingEmitter()
    ops.build().emit(rec)
    emitted = {int(args[0]) for name, args, _kw in rec.calls if name == "node"}
    assert any(name == "partition_open" for name, _a, _k in rec.calls)
    assert emitted and not set(INTERIOR) & emitted
