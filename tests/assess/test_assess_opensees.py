"""ADR 0094 Amendment 2 — OpenSeesModel.assess() acceptance.

``_build_osm`` builds a real deck through the public ``apeSees`` API
(fix + a ``Plain`` pattern importing a node load) and writes/reads it
through ``ops.h5()`` / ``OpenSeesModel.from_h5`` — a real broker
round trip. ``apeSees.h5()`` never archives ``_analyze_call`` (only a
**live** ``apeSees.analyze()`` sets it, on a throwaway
``LiveOpsEmitter`` that ``ops.h5()`` never touches), so tests that
need an analyze call use ``dataclasses.replace`` on the frozen,
already-real broker to add/remove exactly the field the catalog is
judging. This keeps the suite openseespy-free while still exercising
one genuine ``OpenSeesModel`` end to end.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from apeGmsh.assess import CATALOG_SEVERITY, CONDITIONAL_SEVERITY, AssessmentReport
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter

from tests.conftest import _open_model_from_h5
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


def _codes(report: AssessmentReport) -> list[str]:
    return [f.code for f in report.findings]


def _assert_catalog_severities(report: AssessmentReport) -> None:
    for finding in report.findings:
        assert finding.code in CATALOG_SEVERITY, finding
        # ADR 0094 Amendment 4: RES.ENERGY_ERR's severity is runtime-
        # conditional (see CONDITIONAL_SEVERITY); every other code stays
        # single-valued against CATALOG_SEVERITY.
        allowed = CONDITIONAL_SEVERITY.get(
            finding.code, frozenset({CATALOG_SEVERITY[finding.code]}),
        )
        assert finding.severity in allowed, finding


# ---------------------------------------------------------------------------
# Real OpenSeesModel fixtures
# ---------------------------------------------------------------------------


def _build_osm(tmp_path: Path, *, name: str = "osm_fixture.h5") -> Any:
    """Fix on node 1 + a Plain pattern importing a load on node 2.

    No ``ops.analyze`` (see module docstring) — ``_analyze_call`` is
    ``None`` on the returned broker.
    """
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    fem = build_simple_frame_fem()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.fix(nodes=[1], dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(100e3, 0.0, 0.0, 0.0, 0.0, 0.0))
    out = tmp_path / name
    ops.h5(str(out))
    return _open_model_from_h5(out)


def _build_osm_with_unimported_loads(tmp_path: Path) -> Any:
    """Broker carries a nodal load; the (single) pattern imports nothing."""
    from apeGmsh._kernel.records._loads import NodalLoadRecord
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    node_ids = np.array([1, 2], dtype=np.int64)
    node_coords = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([1], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    pg = {(1, 100): {
        "name": "Cols",
        "node_ids": node_ids,
        "node_coords": node_coords,
        "element_ids": np.array([1], dtype=np.int64),
    }}
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
        loads=[NodalLoadRecord(node_id=2, force_xyz=(100e3, 0.0, 0.0))],
    )
    elements = ElementComposite(
        groups={1: line_group}, physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=2, n_elems=1, bandwidth=1, types=[line_info])
    fem = FEMData(nodes=nodes, elements=elements, info=info)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.fix(nodes=[1], dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts):
        pass  # broker load declared; from_model never called

    out = tmp_path / "osm_unimported.h5"
    ops.h5(str(out))
    return _open_model_from_h5(out)


# ---------------------------------------------------------------------------
# analyze_call() accessor
# ---------------------------------------------------------------------------


def test_analyze_call_is_none_when_not_archived(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)
    assert model.analyze_call() is None


def test_analyze_call_public_accessor(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)
    with_call = dataclasses.replace(model, _analyze_call=(10, 0.05))
    assert with_call.analyze_call() == (10, 0.05)


# ---------------------------------------------------------------------------
# Catalog: healthy deck / OSM.NO_SUPPORT / OSM.NO_PATTERN
# ---------------------------------------------------------------------------


def test_healthy_deck_is_zero_findings(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)
    healthy = dataclasses.replace(model, _analyze_call=(10, 0.1))
    report = healthy.assess()
    assert isinstance(report, AssessmentReport)
    assert report.findings == ()
    _assert_catalog_severities(report)


def test_no_fix_lines_yields_no_support(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)
    no_fix = dataclasses.replace(model, _analyze_call=(10, 0.1), _fixes=())
    report = no_fix.assess()
    assert "OSM.NO_SUPPORT" in _codes(report)
    finding = next(f for f in report.findings if f.code == "OSM.NO_SUPPORT")
    assert finding.severity == "warning"
    _assert_catalog_severities(report)


def test_no_analyze_is_info_and_skips_no_pattern(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)  # _analyze_call is None by construction
    report = model.assess()
    assert "OSM.NO_ANALYZE" in _codes(report)
    finding = next(f for f in report.findings if f.code == "OSM.NO_ANALYZE")
    assert finding.severity == "info"
    assert "eigen" in finding.message.lower()
    assert "OSM.NO_PATTERN" not in _codes(report)
    assert any(code == "OSM.NO_PATTERN" for code, _ in report.skipped)
    _assert_catalog_severities(report)


def test_analyze_present_no_pattern_is_no_pattern_finding(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)
    variant = dataclasses.replace(model, _analyze_call=(4, None), _patterns=())
    report = variant.assess()
    assert "OSM.NO_PATTERN" in _codes(report)
    finding = next(f for f in report.findings if f.code == "OSM.NO_PATTERN")
    assert finding.severity == "warning"
    _assert_catalog_severities(report)


# ---------------------------------------------------------------------------
# OSM.LOADS_UNIMPORTED
# ---------------------------------------------------------------------------


def test_broker_loads_unimported_is_finding(tmp_path: Path) -> None:
    model = _build_osm_with_unimported_loads(tmp_path)
    report = model.assess()
    assert "OSM.LOADS_UNIMPORTED" in _codes(report)
    finding = next(f for f in report.findings if f.code == "OSM.LOADS_UNIMPORTED")
    assert finding.severity == "warning"
    assert any(
        code == "OSM.LOADS_UNIMPORTED" and "per-case" in reason
        for code, reason in report.skipped
    )
    _assert_catalog_severities(report)


def test_no_pattern_with_declared_loads_fires_both(tmp_path: Path) -> None:
    """analyze archived + broker loads + zero patterns: the classic
    forgot-``from_model`` deck needs OSM.LOADS_UNIMPORTED, not only a
    shrug-able OSM.NO_PATTERN (whose message says free vibration is
    legal)."""
    model = _build_osm_with_unimported_loads(tmp_path)
    variant = dataclasses.replace(
        model, _analyze_call=(10, 0.1), _patterns=(),
    )
    report = variant.assess()
    assert "OSM.NO_PATTERN" in _codes(report)
    assert "OSM.LOADS_UNIMPORTED" in _codes(report)
    _assert_catalog_severities(report)


def test_no_analyze_with_declared_loads_is_skipped_not_flagged(
    tmp_path: Path,
) -> None:
    """Eigen-only / no-run deck with declared loads is a legal
    multi-deck workflow — skipped, not judged."""
    model = _build_osm_with_unimported_loads(tmp_path)
    variant = dataclasses.replace(model, _patterns=())  # analyze stays None
    report = variant.assess()
    assert "OSM.LOADS_UNIMPORTED" not in _codes(report)
    assert any(
        code == "OSM.LOADS_UNIMPORTED" and "later deck" in reason
        for code, reason in report.skipped
    )


def test_loads_imported_by_pattern_is_not_a_finding(tmp_path: Path) -> None:
    # _build_osm's pattern DOES import (p.load(...)) — the broker itself
    # carries no NodalLoadRecord, so the check is vacuously clean either
    # way; this asserts the healthy-path shape stays clean too.
    model = _build_osm(tmp_path)
    report = model.assess()
    assert "OSM.LOADS_UNIMPORTED" not in _codes(report)


# ---------------------------------------------------------------------------
# Staged decks — whole catalog skipped
# ---------------------------------------------------------------------------


def test_staged_deck_skips_whole_catalog(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)
    staged = dataclasses.replace(
        model, _analyze_call=(10, 0.1), _stages=("stage-1",),
    )
    report = staged.assess()
    assert report.findings == ()
    codes_skipped = {code for code, _ in report.skipped}
    assert {
        "OSM.NO_ANALYZE", "OSM.NO_SUPPORT", "OSM.NO_PATTERN",
        "OSM.LOADS_UNIMPORTED", "RES.ZERO_U",
    } <= codes_skipped
    for _, reason in report.skipped:
        assert "staged deck" in reason


# ---------------------------------------------------------------------------
# RES.ZERO_U (results= cross-check)
# ---------------------------------------------------------------------------


def _write_ux(tmp_path: Path, name: str, *, node_ids: np.ndarray, ux: np.ndarray) -> Path:
    path = tmp_path / name
    time = np.arange(float(ux.shape[0]))
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(name="g", kind="static", time=time)
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_x": ux},
        )
        w.end_stage()
    return path


def test_res_zero_u_all_zero_last_step_is_warning(tmp_path: Path) -> None:
    # ADR 0094 Amendment 4: info -> warning (six dogfood runs, one true
    # positive, zero false positives).
    model = _build_osm(tmp_path)
    ready = dataclasses.replace(model, _analyze_call=(10, 0.1))
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    ux = np.zeros((1, node_ids.size))
    path = _write_ux(tmp_path, "zero_u.h5", node_ids=node_ids, ux=ux)
    with Results.from_native(path, model=_open_model_from_h5(path), fem=fem) as r:
        report = ready.assess(results=r)
    assert "RES.ZERO_U" in _codes(report)
    finding = next(f for f in report.findings if f.code == "RES.ZERO_U")
    assert finding.severity == "warning"
    _assert_catalog_severities(report)


def test_res_zero_u_no_loads_no_pattern_has_no_cross_reference(
    tmp_path: Path,
) -> None:
    """A free-vibration-style deck (zero patterns, no broker loads)
    with all-zero results fires RES.ZERO_U but must NOT point at an
    OSM.LOADS_UNIMPORTED finding that does not exist."""
    model = _build_osm(tmp_path)  # broker carries no load records
    variant = dataclasses.replace(model, _analyze_call=(10, 0.1), _patterns=())
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    ux = np.zeros((1, node_ids.size))
    path = _write_ux(tmp_path, "zero_u_free.h5", node_ids=node_ids, ux=ux)
    with Results.from_native(path, model=_open_model_from_h5(path), fem=fem) as r:
        report = variant.assess(results=r)
    assert "OSM.LOADS_UNIMPORTED" not in _codes(report)
    finding = next(f for f in report.findings if f.code == "RES.ZERO_U")
    assert "OSM.LOADS_UNIMPORTED" not in finding.message


def test_res_zero_u_omitted_results_is_skipped(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)
    ready = dataclasses.replace(model, _analyze_call=(10, 0.1))
    report = ready.assess()
    assert "RES.ZERO_U" not in _codes(report)
    assert any(code == "RES.ZERO_U" for code, _ in report.skipped)


def test_res_zero_u_nonzero_last_step_is_not_a_finding(tmp_path: Path) -> None:
    model = _build_osm(tmp_path)
    ready = dataclasses.replace(model, _analyze_call=(10, 0.1))
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    ux = np.array([[0.0, 0.25]])
    path = _write_ux(tmp_path, "nonzero_u.h5", node_ids=node_ids, ux=ux)
    with Results.from_native(path, model=_open_model_from_h5(path), fem=fem) as r:
        report = ready.assess(results=r)
    assert "RES.ZERO_U" not in _codes(report)
    _assert_catalog_severities(report)


# ---------------------------------------------------------------------------
# ADR 0094 Amendment 3 — run evidence from results
# ---------------------------------------------------------------------------


def _write_mode(tmp_path: Path, name: str, *, node_ids: np.ndarray) -> Path:
    """A modal-only results file (``kind='mode'``, no other stages)."""
    path = tmp_path / name
    with NativeWriter(path) as w:
        w.open(source_type="domain_capture")
        sid = w.begin_stage(
            name="mode1", kind="mode", time=np.array([0.0]),
            eigenvalue=1.0, frequency_hz=1.0, period_s=1.0, mode_index=1,
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": np.array([[0.0, 1.0]])},
        )
        w.end_stage()
    return path


def test_broken_shoebuckle_fires_both_loads_unimported_and_zero_u(
    tmp_path: Path,
) -> None:
    """Broker loads + zero patterns + no archived analyze + results= an
    all-zero non-mode last step (the shoebuckle dogfood shape, PR #990):
    both codes fire, and RES.ZERO_U cross-references OSM.LOADS_UNIMPORTED.
    """
    base = _build_osm_with_unimported_loads(tmp_path)  # _analyze_call stays None
    model = dataclasses.replace(base, _patterns=())  # zero patterns
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    ux = np.zeros((1, node_ids.size))
    path = _write_ux(tmp_path, "shoebuckle_zero_u.h5", node_ids=node_ids, ux=ux)
    with Results.from_native(path, model=_open_model_from_h5(path), fem=fem) as r:
        report = model.assess(results=r)
    assert "OSM.LOADS_UNIMPORTED" in _codes(report)
    unimported = next(
        f for f in report.findings if f.code == "OSM.LOADS_UNIMPORTED"
    )
    assert unimported.severity == "warning"
    assert "RES.ZERO_U" in _codes(report)
    zero_u = next(f for f in report.findings if f.code == "RES.ZERO_U")
    assert zero_u.severity == "warning"  # ADR 0094 Amendment 4
    assert "OSM.LOADS_UNIMPORTED" in zero_u.message
    _assert_catalog_severities(report)


def test_broken_shoebuckle_modal_only_results_keeps_both_skipped(
    tmp_path: Path,
) -> None:
    """Same shape, but ``results=`` is modal-only — not run evidence."""
    base = _build_osm_with_unimported_loads(tmp_path)
    model = dataclasses.replace(base, _patterns=())  # zero patterns
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    path = _write_mode(tmp_path, "shoebuckle_mode.h5", node_ids=node_ids)
    with Results.from_native(path, model=_open_model_from_h5(path), fem=fem) as r:
        report = model.assess(results=r)
    assert "OSM.LOADS_UNIMPORTED" not in _codes(report)
    assert "RES.ZERO_U" not in _codes(report)
    codes_skipped = {code for code, _ in report.skipped}
    assert "OSM.LOADS_UNIMPORTED" in codes_skipped
    assert "RES.ZERO_U" in codes_skipped


def test_results_nonzero_last_step_no_zero_u_but_loads_unimported_fires(
    tmp_path: Path,
) -> None:
    """A non-mode stage with a nonzero last step is still run evidence —
    OSM.LOADS_UNIMPORTED fires (zero-pattern deck) even though RES.ZERO_U
    does not (displacement is not all-zero)."""
    base = _build_osm_with_unimported_loads(tmp_path)
    model = dataclasses.replace(base, _patterns=())  # zero patterns
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    ux = np.array([[0.0, 0.25]])
    path = _write_ux(tmp_path, "shoebuckle_nonzero_u.h5", node_ids=node_ids, ux=ux)
    with Results.from_native(path, model=_open_model_from_h5(path), fem=fem) as r:
        report = model.assess(results=r)
    assert "RES.ZERO_U" not in _codes(report)
    assert "OSM.LOADS_UNIMPORTED" in _codes(report)
    _assert_catalog_severities(report)


def test_no_results_no_analyze_matches_amendment2_behavior(tmp_path: Path) -> None:
    """No results= and no archived analyze: identical to Amendment 2 —
    both codes land in skipped, neither is a finding."""
    base = _build_osm_with_unimported_loads(tmp_path)
    model = dataclasses.replace(base, _patterns=())  # zero patterns
    report = model.assess()
    assert "OSM.LOADS_UNIMPORTED" not in _codes(report)
    assert "RES.ZERO_U" not in _codes(report)
    codes_skipped = {code for code, _ in report.skipped}
    assert "OSM.LOADS_UNIMPORTED" in codes_skipped
    assert "RES.ZERO_U" in codes_skipped


def test_no_analyze_message_names_results_based_evidence(tmp_path: Path) -> None:
    """OSM.NO_ANALYZE's message names the custom-driver reading when
    results= itself supplies the run evidence."""
    model = _build_osm(tmp_path)  # _analyze_call is None by construction
    fem = build_simple_frame_fem()
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    ux = np.zeros((1, node_ids.size))
    path = _write_ux(tmp_path, "no_analyze_with_results.h5", node_ids=node_ids, ux=ux)
    with Results.from_native(path, model=_open_model_from_h5(path), fem=fem) as r:
        report = model.assess(results=r)
    finding = next(f for f in report.findings if f.code == "OSM.NO_ANALYZE")
    assert finding.severity == "info"
    assert "custom analyze loop" in finding.message


def test_no_analyze_message_unchanged_without_results(tmp_path: Path) -> None:
    """No results= at all: OSM.NO_ANALYZE keeps the Amendment 2 message."""
    model = _build_osm(tmp_path)
    report = model.assess()
    finding = next(f for f in report.findings if f.code == "OSM.NO_ANALYZE")
    assert "custom analyze loop" not in finding.message
