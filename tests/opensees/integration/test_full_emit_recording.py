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
