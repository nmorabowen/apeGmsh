"""Phase SSI-2.D PR-C — `s.region` / `s.recorder` builder methods,
recorder claiming mechanism, V4 validator, and introspection
properties.

Partitioned region tag-cache tests live in
`tests/opensees/integration/test_emit_partitioned_stage_bound_regions.py`.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import _StageBuilder, apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    RegionAssignmentRecord,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.recorder import Element as ElementRec
from apeGmsh.opensees.recorder import Node as NodeRec

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_two_pg_fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
            ],
            node_pgs={
                "Left":       [1, 4],
                "CimbraOnly": [5, 6],
            },
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock":   _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "cimbra": _ElementGroupView(
                    ids=(2,), connectivity=((2, 5, 6, 3),),
                ),
            },
        ),
    )


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _two_stage_ops_with_cimbra_activation() -> apeSees:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    return ops


# ===========================================================================
# __slots__ — sanity assertion that PR-C extends the builder
# ===========================================================================


def test_stage_builder_slots_include_pr_c_pools() -> None:
    """`_StageBuilder.__slots__` must declare `_region_records` and
    `_recorder_specs` for PR-C; without them, `s.region` / `s.recorder`
    raise `AttributeError` at first call."""
    assert "_region_records" in _StageBuilder.__slots__
    assert "_recorder_specs" in _StageBuilder.__slots__


# ===========================================================================
# Builder positive — region / recorder records flow into StageRecord
# ===========================================================================


def test_s_region_populates_stage_record_region_records() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="lining") as s:
        s.region(name="lining_r", pg="Left")
        s.region(name="probe_r", nodes=[2])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    recs = ops._stage_records[0].region_records
    assert len(recs) == 2
    assert recs[0] == RegionAssignmentRecord(
        name="lining_r", pg="Left", nodes=None,
    )
    assert recs[1] == RegionAssignmentRecord(
        name="probe_r", pg=None, nodes=(2,),
    )


def test_s_recorder_pulls_spec_into_stage_pool() -> None:
    """`s.recorder(spec)` claims the spec — bridge marks it and the
    stage's `recorder_specs` carries the reference."""
    ops = _two_stage_ops_with_cimbra_activation()
    rec_spec = ops.recorder.Node(
        file="disp.out", response="disp", nodes=(1,), dofs=(1, 2),
    )
    with ops.stage(name="lining") as s:
        s.recorder(rec_spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert ops._stage_records[0].recorder_specs == (rec_spec,)
    assert id(rec_spec) in ops._stage_claimed_recorder_ids


def test_s_region_recorder_default_empty_when_not_called() -> None:
    """A stage that never calls s.region / s.recorder exposes empty
    tuples on its StageRecord."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="bare") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = ops._stage_records[0]
    assert rec.region_records == ()
    assert rec.recorder_specs == ()


# ===========================================================================
# Builder negative — XOR validation + recorder type / membership checks
# ===========================================================================


def test_s_region_rejects_both_pg_and_nodes() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or nodes="):
            s.region(name="x", pg="Left", nodes=[1])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_region_rejects_empty_name() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="name= must be non-empty"):
            s.region(name="", pg="Left")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_recorder_rejects_non_recorder() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="bad") as s:
        with pytest.raises(TypeError, match="expected a Recorder"):
            s.recorder("not a recorder")  # type: ignore[arg-type]
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_recorder_rejects_unregistered_spec() -> None:
    """A Recorder constructed directly (NOT through ops.recorder.X)
    is not in the bridge's _primitives — refuse to claim it."""
    ops = _two_stage_ops_with_cimbra_activation()
    # Construct directly, bypassing the namespace registration.
    spec = NodeRec(file="x", response="disp", nodes=(1,), dofs=(1, 2))
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="not in the bridge's _primitives"):
            s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_recorder_rejects_double_claim() -> None:
    """The same recorder spec cannot be claimed by two stages."""
    ops = _two_stage_ops_with_cimbra_activation()
    spec = ops.recorder.Node(
        file="x", response="disp", nodes=(1,), dofs=(1, 2),
    )
    with ops.stage(name="A") as s:
        s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="B") as s:
        with pytest.raises(ValueError, match="already claimed"):
            s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


# ===========================================================================
# Claiming — global emit skips claimed recorders; stage emits them
# ===========================================================================


def test_claimed_recorder_skipped_in_global_emit() -> None:
    """A recorder claimed by a stage must NOT appear in the global
    post-element emit; it appears inside the stage block instead."""
    ops = _two_stage_ops_with_cimbra_activation()
    ops.fix(pg="Left", dofs=(1, 1))
    spec = ops.recorder.Node(
        file="lining.out", response="disp", nodes=(1,), dofs=(1, 2),
    )
    with ops.stage(name="rock_only") as s:
        s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)

    # Bucket calls by stage scope (-1 = pre-stage global; 0 = stage 1).
    stage_idx = -1
    recorder_calls_per_stage: dict[int, int] = {}
    for name, _args, _kwargs in rec.calls:
        if name == "stage_open":
            stage_idx += 1
            continue
        if name == "recorder":
            recorder_calls_per_stage[stage_idx] = (
                recorder_calls_per_stage.get(stage_idx, 0) + 1
            )
    assert recorder_calls_per_stage.get(-1, 0) == 0, (
        f"claimed recorder must NOT emit in global zone; got "
        f"{recorder_calls_per_stage.get(-1, 0)} recorder lines"
    )
    assert recorder_calls_per_stage.get(0, 0) == 1, (
        f"claimed recorder must emit inside stage 0's block; got "
        f"{recorder_calls_per_stage}"
    )


def test_unclaimed_recorder_still_emits_globally() -> None:
    """A recorder NOT claimed by any stage still emits in the global
    post-element loop."""
    ops = _two_stage_ops_with_cimbra_activation()
    ops.recorder.Node(
        file="global.out", response="disp", nodes=(1,), dofs=(1, 2),
    )
    with ops.stage(name="A") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    stage_idx = -1
    recorder_calls_per_stage: dict[int, int] = {}
    for name, _args, _kwargs in rec.calls:
        if name == "stage_open":
            stage_idx += 1
            continue
        if name == "recorder":
            recorder_calls_per_stage[stage_idx] = (
                recorder_calls_per_stage.get(stage_idx, 0) + 1
            )
    assert recorder_calls_per_stage.get(-1, 0) == 1
    assert recorder_calls_per_stage.get(0, 0) == 0


# ===========================================================================
# Emit ordering — recorder lands after chain, before analyze
# ===========================================================================


def test_stage_bound_recorder_emits_after_chain_before_analyze() -> None:
    """Within a stage block: chain → recorder → analyze.  Locks the
    PR-C slot ordering documented in `_emit_stages_flat`."""
    ops = _two_stage_ops_with_cimbra_activation()
    spec = ops.recorder.Node(
        file="disp.out", response="disp", nodes=(1,), dofs=(1, 2),
    )
    with ops.stage(name="probe") as s:
        s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()

    stage_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: probe")
    )
    chain_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage_idx and ln.strip().startswith("constraints ")
    )
    rec_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage_idx and ln.strip().startswith("recorder Node")
    )
    analyze_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage_idx and ln.strip().startswith("analyze ")
    )
    assert chain_idx < rec_idx < analyze_idx, (
        f"slot order broken: chain={chain_idx} recorder={rec_idx} "
        f"analyze={analyze_idx}"
    )


def test_stage_bound_region_emits_before_domain_change() -> None:
    """Stage-bound region lines emit alongside fix/mass, BEFORE the
    unified domain_change barrier."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="probe") as s:
        s.activate(pgs=["cimbra"])
        s.region(name="cimbra_r", pg="CimbraOnly")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()

    stage_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: probe")
    )
    region_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage_idx and ln.strip().startswith("region ")
    )
    dc_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage_idx and ln.strip() == "domainChange"
    )
    assert region_idx < dc_idx, (
        f"slot order broken: region={region_idx} domainChange={dc_idx}"
    )


# ===========================================================================
# V4 — recorder ownership-tier check
# ===========================================================================


def test_v4_stage_recorder_targets_later_stage_node_raises() -> None:
    """Stage 1 (rock_only) recorder targets node 5 — but node 5 is
    owned by stage 2 (cimbra activation).  Refuse at build."""
    ops = _two_stage_ops_with_cimbra_activation()
    spec = ops.recorder.Node(
        file="x", response="disp", nodes=(5,), dofs=(1, 2),
    )
    with ops.stage(name="rock_only") as s:
        s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    with pytest.raises(
        BridgeError, match=r"recorder.*LATER stage|recorders reference topology"
    ):
        bm.emit(TclEmitter())


def test_v4_stage_recorder_targets_later_stage_element_raises() -> None:
    """Stage 1 (rock_only) element recorder targets cimbra elements
    (owned by stage 2).  Refuse."""
    ops = _two_stage_ops_with_cimbra_activation()
    spec = ops.recorder.Element(
        file="x", response=("globalForce",), pg="cimbra",
    )
    with ops.stage(name="rock_only") as s:
        s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    with pytest.raises(BridgeError, match=r"LATER stage"):
        bm.emit(TclEmitter())


def test_v4_stage_recorder_targets_own_stage_passes() -> None:
    """Stage 2 recorder targets its own stage-bound topology — legal."""
    ops = _two_stage_ops_with_cimbra_activation()
    spec = ops.recorder.Node(
        file="x", response="disp", nodes=(5,), dofs=(1, 2),
    )
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    bm.emit(TclEmitter())  # must not raise


def test_v4_stage_recorder_targets_global_node_passes() -> None:
    """Stage recorder targets a globally-emitted node — legal."""
    ops = _two_stage_ops_with_cimbra_activation()
    spec = ops.recorder.Node(
        file="x", response="disp", nodes=(1,), dofs=(1, 2),
    )
    with ops.stage(name="probe") as s:
        s.recorder(spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    bm.emit(TclEmitter())


# ===========================================================================
# Introspection (Red #19 extended) — all_region_records, all_recorder_specs
# ===========================================================================


def test_all_region_records_combines_global_and_stages() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    ops.region(name="global_r", nodes=[1])
    with ops.stage(name="A") as s:
        s.region(name="stage_r", nodes=[2])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    all_regions = ops.all_region_records
    origins = [origin for origin, _ in all_regions]
    assert origins == ["global", "stage 'A'"]
    assert all_regions[0][1].name == "global_r"
    assert all_regions[1][1].name == "stage_r"


def test_all_recorder_specs_combines_global_and_stages() -> None:
    """Global recorders (unclaimed) + stage-bound recorders both appear,
    tagged by origin.  Claimed recorders only appear under their owning
    stage, never under 'global'."""
    ops = _two_stage_ops_with_cimbra_activation()
    g_spec = ops.recorder.Node(
        file="g.out", response="disp", nodes=(1,), dofs=(1, 2),
    )
    s_spec = ops.recorder.Node(
        file="s.out", response="disp", nodes=(1,), dofs=(1, 2),
    )
    with ops.stage(name="A") as s:
        s.recorder(s_spec)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    all_recs = ops.all_recorder_specs
    origins = [origin for origin, _ in all_recs]
    assert origins == ["global", "stage 'A'"]
    assert all_recs[0][1] is g_spec
    assert all_recs[1][1] is s_spec
