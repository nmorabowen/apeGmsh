"""End-to-end emit through :class:`RecordingEmitter` (Phase 4 Step 2).

This test is the acceptance gate for the bridge build pipeline:
materials, sections, transforms, elements, BCs, patterns, and the
analysis chain are all driven through ``BuiltModel.emit`` and the
recording is asserted against an expected sequence.

The fixture is a hand-rolled FEMData stub from
:mod:`tests.opensees.fixtures.fem_stub`. It carries everything the
build pipeline needs (nodes, elements per PG, ``index`` / ``coords``
slots) and nothing else.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh._kernel.records._loads import NodalLoadRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.section.fiber import FiberPoint, RectPatch

from tests.opensees.fixtures.fem_stub import (
    make_arch_with_orientation_fan_out,
    make_two_column_frame,
    make_two_node_beam,
)


def _logical_calls(rec: RecordingEmitter) -> list[tuple[str, tuple[object, ...]]]:
    """Strip the kwargs and section/pattern braces for sequence-only checks."""
    return [(name, args) for name, args, _ in rec.calls]


def test_minimal_force_beam_recording_sequence() -> None:
    """A single force-beam element + base fix + one nodal load drives a
    fully-formed recording sequence in the documented order."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )

    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(100e3, 0.0, 0.0))

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    names = [c[0] for c in rec.calls]

    # The model directive comes first.
    assert names[0] == "model"
    # uniaxialMaterial Steel02 emitted before section.
    idx_mat = names.index("uniaxialMaterial")
    idx_section_open = names.index("section_open")
    assert idx_mat < idx_section_open
    # Section close before transform.
    idx_section_close = names.index("section_close")
    idx_transf = names.index("geomTransf")
    assert idx_section_close < idx_transf
    # beamIntegration after section (it composes a section by tag) and
    # before the element (which references the integration tag).
    idx_integ = names.index("beamIntegration")
    idx_element = names.index("element")
    assert idx_section_close < idx_integ < idx_element
    # Transform before element.
    assert idx_transf < idx_element
    # Element before fix (fixes / masses come after element fan-out).
    idx_fix = names.index("fix")
    assert idx_element < idx_fix
    # timeSeries before pattern_open.
    idx_ts = names.index("timeSeries")
    idx_pattern_open = names.index("pattern_open")
    assert idx_ts < idx_pattern_open
    # pattern_open then load then pattern_close.
    idx_load = names.index("load")
    idx_pattern_close = names.index("pattern_close")
    assert idx_pattern_open < idx_load < idx_pattern_close


def test_node_recorder_pg_resolves_to_explicit_node_ids() -> None:
    """``ops.recorder.Node(pg=...)`` is materialised by the build
    pipeline into the same recorder line as the explicit ``nodes=...``
    form.  Direct ``_emit`` calls on a ``pg=`` spec still raise as a
    defense-in-depth guard (unit-tested); this test confirms the
    bridge-driven path actually works end-to-end.
    """
    fem = make_two_column_frame()  # PG "Base" -> nodes (1, 3)

    # Bridge run with pg=.
    ops_pg = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops_pg.model(ndm=3, ndf=6)
    ops_pg.recorder.Node(
        file="disp.out", response="disp", pg="Base", dofs=(1, 2, 3),
    )
    rec_pg = RecordingEmitter()
    ops_pg.build().emit(rec_pg)

    # Bridge run with explicit nodes= matching the pg's resolution.
    ops_ids = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops_ids.model(ndm=3, ndf=6)
    ops_ids.recorder.Node(
        file="disp.out", response="disp", nodes=(1, 3), dofs=(1, 2, 3),
    )
    rec_ids = RecordingEmitter()
    ops_ids.build().emit(rec_ids)

    # The recorder lines must match byte-for-byte (pg= form rewrites
    # via dataclasses.replace before driving _emit, so the emitted call
    # is the same as the manual-ids form).
    pg_recorder_calls = [c for c in rec_pg.calls if c[0] == "recorder"]
    ids_recorder_calls = [c for c in rec_ids.calls if c[0] == "recorder"]
    assert pg_recorder_calls == ids_recorder_calls
    # And it includes "-node 1 3" in order.
    args = pg_recorder_calls[0][1]
    node_idx = args.index("-node")
    assert args[node_idx + 1: node_idx + 3] == (1, 3)


def test_element_recorder_pg_resolves_to_explicit_element_ids() -> None:
    """Same as the Node test but for ``ops.recorder.Element(pg=...)``."""
    fem = make_two_column_frame()  # PG "Cols" -> elements (1, 2)

    ops_pg = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops_pg.model(ndm=3, ndf=6)
    ops_pg.recorder.Element(
        file="force.out", response=("globalForce",), pg="Cols",
    )
    rec_pg = RecordingEmitter()
    ops_pg.build().emit(rec_pg)

    ops_ids = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops_ids.model(ndm=3, ndf=6)
    ops_ids.recorder.Element(
        file="force.out", response=("globalForce",), elements=(1, 2),
    )
    rec_ids = RecordingEmitter()
    ops_ids.build().emit(rec_ids)

    pg_recorder_calls = [c for c in rec_pg.calls if c[0] == "recorder"]
    ids_recorder_calls = [c for c in rec_ids.calls if c[0] == "recorder"]
    assert pg_recorder_calls == ids_recorder_calls
    args = pg_recorder_calls[0][1]
    ele_idx = args.index("-ele")
    assert args[ele_idx + 1: ele_idx + 3] == (1, 2)


def test_mpco_nodes_pg_emits_region_and_R_flag() -> None:
    """``ops.recorder.MPCO(nodes_pg=...)`` auto-emits a ``region`` line
    holding the resolved node ids and passes ``-R $tag`` to the MPCO
    command — this is the LHS-scaling lever from the Cerro Lindo
    feature spec (MPCO records whole-model output without filter; the
    region lets us target hombros/crown only).
    """
    fem = make_two_column_frame()  # PG "Base" -> nodes (1, 3)

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.recorder.MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
        nodes_pg="Base",
    )
    rec = RecordingEmitter()
    ops.build().emit(rec)

    region_calls = [c for c in rec.calls if c[0] == "region"]
    mpco_calls = [c for c in rec.calls if c[0] == "recorder" and c[1][0] == "mpco"]

    # Exactly one region + one MPCO emitted; region precedes MPCO.
    assert len(region_calls) == 1
    assert len(mpco_calls) == 1
    region_idx = rec.calls.index(region_calls[0])
    mpco_idx = rec.calls.index(mpco_calls[0])
    assert region_idx < mpco_idx

    # Region carries the resolved node ids in declaration order.
    region_args = region_calls[0][1]
    region_tag = region_args[0]
    assert "-node" in region_args
    node_flag_idx = region_args.index("-node")
    assert region_args[node_flag_idx + 1: node_flag_idx + 3] == (1, 3)

    # MPCO command references the region's tag via -R.
    mpco_args = mpco_calls[0][1]
    assert "-R" in mpco_args
    r_idx = mpco_args.index("-R")
    assert mpco_args[r_idx + 1] == region_tag


def test_mpco_explicit_nodes_matches_pg_form() -> None:
    """User-spec acceptance: explicit ``nodes=`` and ``nodes_pg=`` that
    resolve to the same id list produce byte-identical MPCO output."""
    fem = make_two_column_frame()  # PG "Base" -> nodes (1, 3)

    # Bridge run with nodes_pg=.
    ops_pg = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops_pg.model(ndm=3, ndf=6)
    ops_pg.recorder.MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
        nodes_pg="Base",
    )
    rec_pg = RecordingEmitter()
    ops_pg.build().emit(rec_pg)

    # Bridge run with explicit nodes= matching the pg's resolution.
    ops_ids = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops_ids.model(ndm=3, ndf=6)
    ops_ids.recorder.MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
        nodes=(1, 3),
    )
    rec_ids = RecordingEmitter()
    ops_ids.build().emit(rec_ids)

    # The region + recorder pair must match byte-for-byte (both forms
    # land in the same materialised state after the build-pipeline
    # rewrite).
    pg_filter = [c for c in rec_pg.calls if c[0] in ("region", "recorder")]
    ids_filter = [c for c in rec_ids.calls if c[0] in ("region", "recorder")]
    assert pg_filter == ids_filter


def test_mpco_elements_pg_emits_region_with_ele_flag() -> None:
    """``elements_pg=`` resolves to ``-ele e1 e2 ...`` inside the region."""
    fem = make_two_column_frame()  # PG "Cols" -> elements (1, 2)

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.recorder.MPCO(
        file="run.mpco",
        elem_responses=("section.force",),
        elements_pg="Cols",
    )
    rec = RecordingEmitter()
    ops.build().emit(rec)

    region_calls = [c for c in rec.calls if c[0] == "region"]
    assert len(region_calls) == 1
    region_args = region_calls[0][1]
    assert "-ele" in region_args
    ele_flag_idx = region_args.index("-ele")
    assert region_args[ele_flag_idx + 1: ele_flag_idx + 3] == (1, 2)


def test_mpco_both_nodes_and_elements_pg_emits_one_region() -> None:
    """Single MPCO with both node and element filters → ONE region
    carrying both ``-node`` and ``-ele`` flags (the OpenSees ``region``
    syntax allows a hybrid).  MPCO ``-R`` filters both nodal and element
    output through this single region's members.
    """
    fem = make_two_column_frame()

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.recorder.MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
        elem_responses=("section.force",),
        nodes_pg="Top",        # PG "Top" -> nodes (2, 4)
        elements_pg="Cols",    # PG "Cols" -> elements (1, 2)
    )
    rec = RecordingEmitter()
    ops.build().emit(rec)

    region_calls = [c for c in rec.calls if c[0] == "region"]
    assert len(region_calls) == 1
    region_args = region_calls[0][1]
    # Both -node and -ele present on the same region line.
    assert "-node" in region_args
    assert "-ele" in region_args


def test_mpco_empty_nodes_pg_raises_bridge_error() -> None:
    """Empty PG → empty region → OpenSees runtime failure.  The
    bridge must refuse at build time with a clear BridgeError, never
    emit a bare ``region $tag`` line.
    """
    from apeGmsh.opensees._internal.build import BridgeError
    from tests.opensees.fixtures.fem_stub import (
        FEMStub, _ElementGroupView, _ElementsStub, _NodesStub,
    )

    nodes = _NodesStub(
        ids=[1, 2],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        node_pgs={"Base": [1], "Empty": []},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.recorder.MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
        nodes_pg="Empty",
    )
    with pytest.raises(BridgeError, match="resolved to zero nodes"):
        ops.build().emit(RecordingEmitter())


def test_mpco_empty_explicit_nodes_raises_bridge_error() -> None:
    """Build-pipeline guard catches the empty-tuple edge case that
    construction-time validation does not (an MPCO with nodes=() and
    matching response set construction-validates but the materialize
    pipeline must still refuse the empty region)."""
    from dataclasses import replace
    from apeGmsh.opensees._internal.build import BridgeError
    from apeGmsh.opensees._internal.tag_allocator import TagAllocator
    from apeGmsh.opensees.recorder import MPCO as _MPCO
    from tests.opensees.fixtures.fem_stub import (
        FEMStub, _ElementGroupView, _ElementsStub, _NodesStub,
    )

    nodes = _NodesStub(
        ids=[1, 2],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
        node_pgs={"Base": [1]},
    )
    elements = _ElementsStub(
        elem_pgs={"Cols": _ElementGroupView(ids=(1,), connectivity=((1, 2),))},
    )
    fem = FEMStub(nodes=nodes, elements=elements)

    spec = _MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
        nodes=(1,),
    )
    spec = replace(spec, nodes=())

    with pytest.raises(BridgeError, match=r"nodes=\(\) is empty"):
        spec.materialize(RecordingEmitter(), fem, TagAllocator())


def test_mpco_without_filter_emits_no_region() -> None:
    """Bare MPCO (no filter selectors) records the whole model — no
    region command, no ``-R`` flag.  Backward-compatible with code
    written before this feature."""
    fem = make_two_node_beam()

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.recorder.MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
    )
    rec = RecordingEmitter()
    ops.build().emit(rec)

    region_calls = [c for c in rec.calls if c[0] == "region"]
    mpco_calls = [c for c in rec.calls if c[0] == "recorder" and c[1][0] == "mpco"]
    assert region_calls == []
    assert len(mpco_calls) == 1
    assert "-R" not in mpco_calls[0][1]


def test_w_armado_fiber_section_in_force_beam_column() -> None:
    """``ops.section.W_fiber`` ships an end-to-end fiber section that
    a ``forceBeamColumn`` element can consume — the Cerro Lindo
    use case is a tapered cimbra arch with built-up W fibers.

    Verifies: the W_fiber-built section emits as a complete
    ``section_open`` block with three rectangular patches (top/bot
    flanges + web), all sharing the user-supplied material.
    """
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.Steel02(fy=337.5e6, E=200e9, b=0.01)
    sec = ops.section.W_fiber(
        bf=150e-3, tf=12e-3, hw=160e-3, tw=8e-3,
        material=steel,
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )

    rec = RecordingEmitter()
    ops.build().emit(rec)

    # One section_open / section_close pair (the Fiber block).
    names = [c[0] for c in rec.calls]
    assert names.count("section_open") == 1
    assert names.count("section_close") == 1
    open_idx = names.index("section_open")
    close_idx = names.index("section_close")
    # Exactly three patch calls inside the block (top + bot flanges + web).
    patches = [c for c in rec.calls[open_idx + 1: close_idx]
               if c[0] == "patch"]
    assert len(patches) == 3
    # Every patch references the same material tag — uniform material.
    steel_tag = ops.tag_for(steel)
    for kind, args, _ in patches:
        assert args[0] == "rect"
        # patch args: ("rect", mat_tag, ny, nz, yI, zI, yJ, zJ)
        assert args[1] == steel_tag


def test_aggregator_section_emits_after_its_dependencies() -> None:
    """``section.Aggregator`` composes one or more uniaxials and an
    optional base section.  The build pipeline must emit the base
    section (if any) and every material *before* the Aggregator so the
    aggregator's tag references are valid.

    Wires through a ``zeroLengthSection`` element so the build graph
    actually reaches the Aggregator via the element-section edge.
    """
    from apeGmsh.opensees.section.aggregator import Aggregator

    fem = make_two_node_beam()  # 2-node line; element 1 with nodes (1,2)

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    k_axial = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
    k_bend = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
    base = ops.section.Elastic(
        E=2e11, A=0.01, Iz=1e-4, Iy=1e-4, G=8e10, J=1e-4,
    )
    agg = ops.section.Aggregator(
        materials_by_dof={"P": k_axial, "Mz": k_bend},
        base_section=base,
    )
    # zeroLengthSection element uses the Aggregator section.
    ops.element.ZeroLengthSection(pg="Cols", section=agg)

    rec = RecordingEmitter()
    ops.build().emit(rec)

    # Locate the relevant emit indices.
    mat_calls = [(i, c) for i, c in enumerate(rec.calls)
                 if c[0] == "uniaxialMaterial"]
    sec_calls = [(i, c) for i, c in enumerate(rec.calls)
                 if c[0] == "section"]
    # Two uniaxials + a base Elastic section + the Aggregator section.
    assert len(mat_calls) == 2
    assert len(sec_calls) == 2  # Elastic base + Aggregator
    base_idx = next(
        i for i, c in sec_calls if c[1][0] == "Elastic"
    )
    agg_idx = next(
        i for i, c in sec_calls if c[1][0] == "Aggregator"
    )
    # All materials AND the base section must precede the Aggregator.
    for mi, _ in mat_calls:
        assert mi < agg_idx
    assert base_idx < agg_idx
    # The Aggregator references its dependencies' allocated tags.
    agg_args = rec.calls[agg_idx][1]
    mat_tag_for = {
        rec.calls[mi][1][2]: rec.calls[mi][1][1]  # type: ignore[index]
        for mi, c in mat_calls
    }
    del mat_tag_for  # tags already inferred from emit order; not asserted directly
    # Aggregator emission shape:
    # ("section", "Aggregator", <agg_tag>, <P_tag>, "P", <Mz_tag>, "Mz",
    #  "-section", <base_tag>)
    assert agg_args[0] == "Aggregator"
    assert "P" in agg_args
    assert "Mz" in agg_args
    assert "-section" in agg_args


def test_initial_stress_wrapper_emits_after_base_material() -> None:
    """``InitialStress`` wraps a base uniaxial; the build pipeline must
    emit the base material BEFORE the wrapper so the wrapper's
    ``$base_tag`` reference is valid in Tcl/Py.
    """
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    base = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    wrapped = ops.uniaxialMaterial.InitialStress(
        base_material=base, sigma_init=0.5 * 250e6,
    )
    # Use the wrapped material in a single-fiber section so the build
    # graph reaches it via the element -> section -> material edge.
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=wrapped, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    # Find the two uniaxialMaterial calls (base + wrapper) and confirm
    # the base lands first.
    mat_calls = [(i, c) for i, c in enumerate(rec.calls)
                 if c[0] == "uniaxialMaterial"]
    assert len(mat_calls) == 2
    (i_base, base_call), (i_wrap, wrap_call) = mat_calls
    assert i_base < i_wrap
    assert base_call[1][0] == "Steel02"
    assert wrap_call[1][0] == "InitialStressMaterial"
    # The wrapper's $base_tag references the base material's allocated tag.
    base_tag = base_call[1][1]
    assert wrap_call[1][2] == base_tag
    # And the sigma_init is forwarded verbatim.
    assert wrap_call[1][3] == 0.5 * 250e6


def test_section_open_close_brackets_patches_and_fibers() -> None:
    """A Fiber section emits ``section_open`` then patch/fiber/layer
    calls then ``section_close`` — the protocol's block contract."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    ops.section.Fiber(
        patches=(
            RectPatch(
                material=steel,
                ny=4, nz=2,
                yI=-0.1, zI=-0.05, yJ=0.1, zJ=0.05,
            ),
        ),
        fibers=(
            FiberPoint(material=steel, y=0.0, z=0.06, area=1e-4),
        ),
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    names = [c[0] for c in rec.calls]
    idx_open = names.index("section_open")
    idx_close = names.index("section_close")
    assert idx_open < idx_close
    # Between open and close: at least one patch and one fiber.
    inner = names[idx_open + 1: idx_close]
    assert "patch" in inner
    assert "fiber" in inner


def test_element_fan_out_across_pg() -> None:
    """A spec on a 2-element PG produces 2 element commands."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    element_calls = [c for c in rec.calls if c[0] == "element"]
    # Two elements in the PG -> two element commands.
    assert len(element_calls) == 2
    # Per-element fan-out allocates tags after the spec's own tag (which
    # is never emitted; see Phase-4 spec). With one element spec
    # registered (consuming tag 1), the two PG elements take tags 2 and 3.
    assert element_calls[0][1][1] == 2
    assert element_calls[1][1][1] == 3
    # Element 1 sees nodes (1, 2); element 2 sees nodes (3, 4).
    assert element_calls[0][1][2:4] == (1, 2)
    assert element_calls[1][1][2:4] == (3, 4)


def test_fix_pg_fans_to_per_node_calls() -> None:
    """``ops.fix(pg=...)`` emits one ``fix`` per node in the PG."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    fix_calls = [c for c in rec.calls if c[0] == "fix"]
    # PG "Base" has 2 nodes -> 2 fix commands.
    assert len(fix_calls) == 2
    fixed_node_tags = {c[1][0] for c in fix_calls}
    assert fixed_node_tags == {1, 3}
    # All 6 dofs propagate.
    for c in fix_calls:
        assert c[1][1:] == (1, 1, 1, 1, 1, 1)


def test_fix_explicit_nodes_emits_directly() -> None:
    """``ops.fix(nodes=[...])`` skips the fan-out path and emits as listed."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    ops.fix(nodes=(2,), dofs=(1, 0, 1, 0, 0, 0))

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    fix_calls = [c for c in rec.calls if c[0] == "fix"]
    assert len(fix_calls) == 1
    assert fix_calls[0][1] == (2, 1, 0, 1, 0, 0, 0)


def test_pattern_pg_load_fans_to_per_node_calls() -> None:
    """``p.load(pg=...)`` inside a pattern emits one ``load`` per node."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="Top", forces=(0.0, 0.0, -1000.0, 0.0, 0.0, 0.0))

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    load_calls = [c for c in rec.calls if c[0] == "load"]
    # PG "Top" has 2 nodes -> 2 load commands.
    assert len(load_calls) == 2
    load_node_tags = {c[1][0] for c in load_calls}
    assert load_node_tags == {2, 4}
    for c in load_calls:
        assert c[1][1:] == (0.0, 0.0, -1000.0, 0.0, 0.0, 0.0)


def test_unregistered_dependency_raises_bridge_error() -> None:
    """A standalone primitive used as a dependency without registering it
    must raise BridgeError at emit time (per ADR P11 Option A)."""
    from apeGmsh.opensees._internal.build import BridgeError
    from apeGmsh.opensees.material.uniaxial import Steel02
    from apeGmsh.opensees.section.fiber import Fiber, FiberPoint
    import pytest

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    # Construct a Steel02 standalone — never registered.
    standalone_steel = Steel02(fy=420e6, E=200e9, b=0.01)
    # Wire it into a Fiber section and register only the section.
    sec = Fiber(
        fibers=(FiberPoint(material=standalone_steel, y=0.0, z=0.0, area=0.01),),
    )
    ops.register(sec)

    bm = ops.build()
    rec = RecordingEmitter()
    with pytest.raises(BridgeError, match="reachable through"):
        bm.emit(rec)


def test_analysis_chain_ordering() -> None:
    """The analysis chain emits in the documented order.

    constraints -> numberer -> system -> test -> algorithm -> integrator -> analysis.
    """
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-6, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.05)
    ops.analysis.Static()

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    names = [c[0] for c in rec.calls]
    expected = ["constraints", "numberer", "system", "test", "algorithm",
                "integrator", "analysis"]
    indices = [names.index(name) for name in expected]
    assert indices == sorted(indices), (
        f"analysis chain emitted out of order: {indices} for {expected}"
    )


def test_recorder_pg_nodes_fans_to_explicit_list() -> None:
    """A Node recorder declared with ``pg=`` emits one ``recorder`` call
    with the resolved list of nodes."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    ops.recorder.Node(
        file="disp.out",
        response="disp",
        pg="Top",
        dofs=(1, 2, 3),
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    rec_calls = [c for c in rec.calls if c[0] == "recorder"]
    assert len(rec_calls) == 1
    args = rec_calls[0][1]
    # Args layout: ('Node', '-file', 'disp.out', '-node', 2, 4, '-dof', 1, 2, 3, 'disp')
    assert args[0] == "Node"
    # The fan-out includes both top nodes as -node positional ids.
    assert 2 in args
    assert 4 in args


def test_orientation_transform_fans_one_geomtransf_per_distinct_vecxz() -> None:
    """ADR 0010: an orientation-bearing GeomTransf emits one
    ``geomTransf`` line per distinct per-element vecxz across the
    elements that reference it. Curved members (arch) produce multiple
    geomTransf lines under a Spherical orientation (the radial
    direction varies along the arch)."""
    from apeGmsh.opensees.transform import Spherical

    fem = make_arch_with_orientation_fan_out()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    orientation = Spherical(origin=(0.0, 0.0, 0.0))
    transf = ops.geomTransf.Linear(orientation=orientation)
    ops.element.elasticBeamColumn(
        pg="Arch",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    geomtransf_calls = [c for c in rec.calls if c[0] == "geomTransf"]
    # Three arch segments at distinct angles -> three distinct vecxz
    # under spherical orientation (e_r varies with position) -> three
    # geomTransf lines, one per distinct vecxz.
    assert len(geomtransf_calls) == 3
    distinct_vecxzs = {tuple(c[1][2:5]) for c in geomtransf_calls}
    assert len(distinct_vecxzs) == 3


def test_alongbeam_orientation_is_bound_by_bridge_fan_out() -> None:
    """AlongBeam declares a bind_fem hook; the bridge calls it before
    per-element vecxz fan-out so triad_at(p) can query the cached
    reference-curve tangents."""
    from apeGmsh.opensees import AlongBeam
    from tests.opensees.fixtures.fem_stub import (
        FEMStub, _ElementGroupView, _ElementsStub, _NodesStub,
    )
    # FEM: two collinear PGs. "MainBar" is the reference curve along
    # +X; "Stirrups" is a single short stirrup perpendicular to it.
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5, 6],
        coords=[
            (0.0, 0.0, 0.0),   # MainBar node 1
            (1.0, 0.0, 0.0),   # MainBar node 2 (also stirrup midpoint sits near here)
            (2.0, 0.0, 0.0),   # MainBar node 3
            (3.0, 0.0, 0.0),   # MainBar node 4
            (1.5, -0.5, 0.0),  # Stirrup node 1
            (1.5,  0.5, 0.0),  # Stirrup node 2
        ],
        node_pgs={"Base": [1]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "MainBar": _ElementGroupView(
                ids=(1, 2, 3),
                connectivity=((1, 2), (2, 3), (3, 4)),
            ),
            "Stirrups": _ElementGroupView(
                ids=(4,), connectivity=((5, 6),),
            ),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    orient = AlongBeam(reference_pg="MainBar")
    # Before build, the orientation is unbound.
    assert orient._p_a is None

    transf = ops.geomTransf.Linear(orientation=orient)
    ops.element.elasticBeamColumn(
        pg="Stirrups",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    # After emit, AlongBeam.bind_fem was called: segments are cached.
    assert orient._p_a is not None
    assert orient._p_a.shape == (3, 3)   # 3 reference segments

    # One geomTransf line per distinct stirrup vecxz; with a single
    # stirrup we expect exactly one.
    geomtransf_calls = [c for c in rec.calls if c[0] == "geomTransf"]
    assert len(geomtransf_calls) == 1


def test_broker_nodal_loads_emit_as_synth_plain_pattern() -> None:
    """ADR 0001: ``fem.nodes.loads`` (from session-side g.loads) is
    emitted by the bridge as a synthesized Linear timeSeries + Plain
    pattern per pattern name, with the force mapped onto ndf."""
    fem = make_two_column_frame()
    fem.nodes.loads = [  # type: ignore[attr-defined]
        NodalLoadRecord(node_id=1, force_xyz=(10.0, 20.0, 0.0),
                        pattern="Pressure"),
        NodalLoadRecord(node_id=3, force_xyz=(30.0, 40.0, 0.0),
                        pattern="Pressure"),
    ]
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    names = [c[0] for c in rec.calls]
    i_ts = names.index("timeSeries")
    i_open = names.index("pattern_open")
    i_close = names.index("pattern_close")
    assert i_ts < i_open < i_close
    assert rec.calls[i_ts][1][0] == "Linear"
    ts_tag = rec.calls[i_ts][1][1]
    assert rec.calls[i_open][1] == ("Plain", rec.calls[i_open][1][1], ts_tag)
    load_calls = [c for c in rec.calls if c[0] == "load"]
    # ndf=3 -> (node, fx, fy, mz); moment absent -> mz = 0.0
    assert (1, 10.0, 20.0, 0.0) in [c[1] for c in load_calls]
    assert (3, 30.0, 40.0, 0.0) in [c[1] for c in load_calls]


def test_2d_geomtransf_emits_bare_form_without_vecxz() -> None:
    """A 2-D model (ndm=2) with a vecxz-less, orientation-less Linear
    transform emits the bare ``geomTransf Linear $tag`` — no vecxz
    vector. Regression for the 2-D beam-column emit path."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)

    transf = ops.geomTransf.Linear()  # no vecxz, no orientation
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf, A=0.01, E=200e9, Iz=1e-4,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    geomtransf_calls = [c for c in rec.calls if c[0] == "geomTransf"]
    assert len(geomtransf_calls) == 1
    # Bare 2-D form: ("Linear", tag) only — no trailing vecxz floats.
    args = geomtransf_calls[0][1]
    assert args[0] == "Linear"
    assert len(args) == 2, f"expected bare 2-D geomTransf, got {args!r}"


def test_orientation_transform_collinear_elements_share_one_geomtransf() -> None:
    """When multiple elements share the same vecxz under an
    orientation, only one ``geomTransf`` line is emitted and reused
    across them."""
    from apeGmsh.opensees.transform import Cartesian

    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    # Both columns are vertical and parallel — Cartesian orientation
    # yields one vecxz for both.
    orientation = Cartesian()
    transf = ops.geomTransf.Linear(orientation=orientation)
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    geomtransf_calls = [c for c in rec.calls if c[0] == "geomTransf"]
    # Both columns yield identical vecxz -> one geomTransf line shared.
    assert len(geomtransf_calls) == 1


def test_recorder_pg_elements_fans_to_explicit_list() -> None:
    """An Element recorder with ``pg=`` resolves to the PG's elements."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    ops.recorder.Element(
        file="ele_force.out",
        response=("globalForce",),
        pg="Cols",
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    rec_calls = [c for c in rec.calls if c[0] == "recorder"]
    assert len(rec_calls) == 1
    args = rec_calls[0][1]
    assert args[0] == "Element"
    # Fan-out includes both element tags 1, 2.
    assert 1 in args
    assert 2 in args


# ---------------------------------------------------------------------------
# Red-team gap closures — Tcl byte-shape, Windows paths, all-DOF Aggregator,
# nested wrappers
# ---------------------------------------------------------------------------

def test_mpco_region_tcl_byte_shape() -> None:
    """T1 — Tcl-emitter byte-shape assertion for the region + MPCO line.

    The recording-emitter tuple shape is one thing; what actually
    runs on OpenSees is the Tcl text.  This test pins the Tcl line
    shape for both the region declaration and the MPCO recorder
    pointing at it.
    """
    from apeGmsh.opensees.emitter.tcl import TclEmitter

    fem = make_two_column_frame()  # PG "Base" → nodes (1, 3)

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.recorder.MPCO(
        file="run.mpco",
        nodal_responses=("displacement",),
        nodes_pg="Base",
    )
    rec = TclEmitter()
    ops.build().emit(rec)
    lines = rec.lines()

    region_lines = [ln for ln in lines if ln.startswith("region ")]
    mpco_lines = [ln for ln in lines if ln.startswith("recorder mpco ")]
    assert len(region_lines) == 1
    assert len(mpco_lines) == 1

    # Region line carries the resolved node ids in order.
    region_line = region_lines[0]
    assert "-node 1 3" in region_line

    # MPCO line carries -R pointing at the region tag.
    region_tag = int(region_line.split()[1])
    mpco_line = mpco_lines[0]
    assert f"-R {region_tag}" in mpco_line
    assert "-N displacement" in mpco_line


def test_mpco_recorder_windows_style_path_passes_through() -> None:
    """T2 — Windows-style backslash paths must round-trip into the
    recorder Tcl line unchanged.  The user spec flags this as a real
    gotcha (Tcl interprets ``\\r`` as a control char) — the bridge's
    job is to pass through the string the user supplied; downstream
    advice in the docstring covers escaping.
    """
    from apeGmsh.opensees.emitter.tcl import TclEmitter

    fem = make_two_node_beam()

    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    # Construct a typical Windows MPCO target path.  The user spec
    # warns about this exact shape needing forward-slash conversion at
    # the use site; here we only assert the bridge does NOT silently
    # transform the path on the way out.
    path = "C:\\runs\\out.mpco"
    ops.recorder.MPCO(
        file=path,
        nodal_responses=("displacement",),
    )
    rec = TclEmitter()
    ops.build().emit(rec)
    mpco_lines = [ln for ln in rec.lines() if ln.startswith("recorder mpco ")]
    assert len(mpco_lines) == 1
    assert path in mpco_lines[0]


def test_aggregator_all_six_dof_emit_end_to_end() -> None:
    """T3 — all six DOF codes through the bridge to RecordingEmitter."""
    from apeGmsh.opensees.section.aggregator import (
        AGGREGATOR_DOF_CODES, Aggregator,
    )

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    # Six distinct materials so we can verify the per-DOF tag wiring.
    mats = {
        code: ops.uniaxialMaterial.ElasticMaterial(E=2e11 * (i + 1))
        for i, code in enumerate(AGGREGATOR_DOF_CODES)
    }
    agg = ops.section.Aggregator(materials_by_dof=mats)
    ops.element.ZeroLengthSection(pg="Cols", section=agg)

    rec = RecordingEmitter()
    ops.build().emit(rec)

    sec_calls = [c for c in rec.calls if c[0] == "section"]
    aggregator_calls = [c for c in sec_calls if c[1][0] == "Aggregator"]
    assert len(aggregator_calls) == 1
    args = aggregator_calls[0][1]
    # Args: ("Aggregator", agg_tag, mat_tag1, "P", mat_tag2, "Vy", ...)
    # Six DOF codes should appear in declaration order.
    codes_in_args = [a for a in args if a in AGGREGATOR_DOF_CODES]
    assert codes_in_args == list(AGGREGATOR_DOF_CODES)


def test_nested_aggregator_emits_with_correct_topo_order() -> None:
    """T4 — Aggregator-as-base_section of another Aggregator.  Both
    Aggregators must emit (base before outer), and the outer's
    ``-section`` flag must reference the base aggregator's tag.
    """
    from apeGmsh.opensees.section.aggregator import Aggregator

    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    m_axial = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
    m_bend = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
    inner = ops.section.Aggregator(materials_by_dof={"P": m_axial})
    outer = ops.section.Aggregator(
        materials_by_dof={"Mz": m_bend}, base_section=inner,
    )
    ops.element.ZeroLengthSection(pg="Cols", section=outer)

    rec = RecordingEmitter()
    ops.build().emit(rec)

    aggregator_calls = [
        (i, c) for i, c in enumerate(rec.calls)
        if c[0] == "section" and c[1][0] == "Aggregator"
    ]
    assert len(aggregator_calls) == 2
    (i_inner, inner_call), (i_outer, outer_call) = aggregator_calls
    assert i_inner < i_outer
    # The outer aggregator references the inner's allocated tag.
    inner_tag = inner_call[1][1]
    outer_args = outer_call[1]
    assert "-section" in outer_args
    sec_flag_idx = outer_args.index("-section")
    assert outer_args[sec_flag_idx + 1] == inner_tag


def test_nested_initial_stress_wrapper_emits_in_dependency_order() -> None:
    """T5 — ``InitialStress(base_material=InitialStress(...))``.  Both
    wrapper layers must emit; the outer's ``$base_tag`` references
    the inner wrapper's tag; the inner wrapper's ``$base_tag``
    references the leaf material.
    """
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    leaf = ops.uniaxialMaterial.Steel02(fy=420e6, E=2e11, b=0.01)
    inner_wrap = ops.uniaxialMaterial.InitialStress(
        base_material=leaf, sigma_init=0.1 * 250e6,
    )
    outer_wrap = ops.uniaxialMaterial.InitialStress(
        base_material=inner_wrap, sigma_init=0.4 * 250e6,
    )
    # Use the outermost in a FiberPoint so the build graph reaches it.
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=outer_wrap, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )

    rec = RecordingEmitter()
    ops.build().emit(rec)

    # Each entry is (index_in_rec.calls, args_tuple) where args_tuple
    # has shape (type_token, tag, *params).
    mat_entries = [
        (i, call[1]) for i, call in enumerate(rec.calls)
        if call[0] == "uniaxialMaterial"
    ]
    # Three materials: leaf + two wrappers.
    assert len(mat_entries) == 3

    leaf_entries = [(i, a) for i, a in mat_entries if a[0] == "Steel02"]
    wrapper_entries = [
        (i, a) for i, a in mat_entries if a[0] == "InitialStressMaterial"
    ]
    assert len(leaf_entries) == 1
    assert len(wrapper_entries) == 2

    leaf_idx, leaf_args = leaf_entries[0]
    leaf_tag = leaf_args[1]
    inner_idx, inner_args = wrapper_entries[0]
    outer_idx, outer_args = wrapper_entries[1]

    # Leaf precedes both wrappers; inner wrapper precedes outer.
    assert leaf_idx < inner_idx < outer_idx
    # Inner wrapper references leaf tag; outer wrapper references inner
    # wrapper's tag.  Wrapper args shape: ("InitialStressMaterial",
    # wrap_tag, base_tag, sigma_init).
    inner_tag = inner_args[1]
    assert inner_args[2] == leaf_tag
    assert outer_args[2] == inner_tag
