"""Phase SSI-2.D PR-B — `s.fix` / `s.mass` builder methods + emit pipeline.

Covers the builder surface (`_StageBuilder.fix` / `.mass`), the
introspection properties (`bridge.all_fix_records` /
`.all_mass_records`), and the single-partition emit slot in
`_emit_stages_flat`.

The partitioned emit slot in `_emit_stages_partitioned` is exercised
by `tests/opensees/integration/test_emit_partitioned_staged.py` —
extended in this PR to cover the per-rank BC fan-out + empty-bracket
skip.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import _StageBuilder, apeSees
from apeGmsh.opensees._internal.build import (
    FixRecord,
    MassRecord,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# Fixtures — same 2-PG quad pair the H1 / SSI-2.B tests use
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
# __slots__ — Red #26 mechanical assertion
# ===========================================================================


def test_stage_builder_slots_include_pr_b_pools() -> None:
    """`_StageBuilder.__slots__` must declare `_fix_records` and
    `_mass_records`; without them, `s.fix(...)` raises `AttributeError`
    at first call."""
    assert "_fix_records" in _StageBuilder.__slots__
    assert "_mass_records" in _StageBuilder.__slots__


# ===========================================================================
# Builder positive — records flow from `s.fix` / `s.mass` into StageRecord
# ===========================================================================


def test_s_fix_populates_stage_record_fix_records() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="lining") as s:
        s.fix(pg="Left", dofs=(1, 1))
        s.fix(nodes=[2], dofs=(1, 0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert len(ops._stage_records) == 1
    fix_recs = ops._stage_records[0].fix_records
    assert len(fix_recs) == 2
    assert fix_recs[0] == FixRecord(pg="Left", nodes=None, dofs=(1, 1))
    assert fix_recs[1] == FixRecord(pg=None, nodes=(2,), dofs=(1, 0))


def test_s_mass_populates_stage_record_mass_records() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="lining") as s:
        s.mass(pg="Left", values=(100.0, 100.0))
        s.mass(nodes=[2], values=(50.0, 50.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    mass_recs = ops._stage_records[0].mass_records
    assert len(mass_recs) == 2
    assert mass_recs[0] == MassRecord(pg="Left", nodes=None, values=(100.0, 100.0))
    assert mass_recs[1] == MassRecord(pg=None, nodes=(2,), values=(50.0, 50.0))


def test_s_fix_mass_default_empty_when_not_called() -> None:
    """A stage that never calls s.fix / s.mass exposes empty tuples."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="bare") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = ops._stage_records[0]
    assert rec.fix_records == ()
    assert rec.mass_records == ()


# ===========================================================================
# Builder negative — pg / nodes XOR + dofs / values mandatory
# ===========================================================================


def test_s_fix_rejects_both_pg_and_nodes() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or nodes="):
            s.fix(pg="Left", nodes=[1], dofs=(1, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_fix_rejects_neither_pg_nor_nodes() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or nodes="):
            s.fix(dofs=(1, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_mass_rejects_both_pg_and_nodes() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or nodes="):
            s.mass(pg="Left", nodes=[1], values=(1.0, 1.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


# ===========================================================================
# Introspection (Red #19) — bridge.all_fix_records / all_mass_records
# ===========================================================================


def test_all_fix_records_combines_global_and_stages() -> None:
    """`bridge.all_fix_records` returns tagged (origin, record) tuples
    spanning the global pool and every stage's pool."""
    ops = _two_stage_ops_with_cimbra_activation()
    ops.fix(pg="Left", dofs=(1, 1))  # global
    with ops.stage(name="A") as s:
        s.fix(nodes=[2], dofs=(1, 0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="B") as s:
        s.activate(pgs=["cimbra"])
        s.fix(nodes=[5], dofs=(1, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    all_fix = ops.all_fix_records
    assert len(all_fix) == 3
    origins = [origin for origin, _ in all_fix]
    assert origins == ["global", "stage 'A'", "stage 'B'"]
    # Records carried through unchanged.
    assert all_fix[0][1].pg == "Left"
    assert all_fix[1][1].nodes == (2,)
    assert all_fix[2][1].nodes == (5,)


def test_all_mass_records_combines_global_and_stages() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    ops.mass(nodes=[1], values=(10.0, 10.0))
    with ops.stage(name="A") as s:
        s.mass(nodes=[2], values=(20.0, 20.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    all_mass = ops.all_mass_records
    assert len(all_mass) == 2
    assert all_mass[0] == ("global", MassRecord(pg=None, nodes=(1,), values=(10.0, 10.0)))
    assert all_mass[1] == ("stage 'A'", MassRecord(pg=None, nodes=(2,), values=(20.0, 20.0)))


# ===========================================================================
# Single-partition emit shape — stage-bound fix / mass land in stage block
# ===========================================================================


def _bucket_flat(rec: RecordingEmitter) -> dict[int, list[tuple[str, tuple, dict]]]:
    """Bucket recorded calls by stage_idx (-1 = pre-stage)."""
    buckets: dict[int, list[tuple[str, tuple, dict]]] = {}
    stage_idx = -1
    for name, args, kwargs in rec.calls:
        if name == "stage_open":
            stage_idx += 1
            continue
        if name == "stage_close":
            continue
        buckets.setdefault(stage_idx, []).append((name, args, kwargs))
    return buckets


def test_flat_stage_bound_fix_emits_inside_stage_block() -> None:
    """s.fix on stage 2 emits `fix` lines INSIDE stage 2's block,
    after the topology and before initial_stress / chain."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.fix(nodes=[5, 6], dofs=(1, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_flat(rec)

    stage2 = buckets.get(1, [])
    fix_calls = [c for c in stage2 if c[0] == "fix"]
    # 2 nodes × 1 fix call each.
    assert len(fix_calls) == 2
    fix_node_ids = sorted(int(c[1][0]) for c in fix_calls)
    assert fix_node_ids == [5, 6]

    # fix lines must NOT appear in pre-stage (-1) or stage 1 (0) blocks.
    for sidx in (-1, 0):
        assert not [c for c in buckets.get(sidx, []) if c[0] == "fix"], (
            f"unexpected fix in stage_idx={sidx}: "
            f"{[c for c in buckets.get(sidx, []) if c[0] == 'fix']}"
        )


def test_flat_stage_bound_mass_emits_inside_stage_block() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.mass(nodes=[5], values=(100.0, 100.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_flat(rec)

    stage2 = buckets.get(1, [])
    mass_calls = [c for c in stage2 if c[0] == "mass"]
    assert len(mass_calls) == 1
    assert int(mass_calls[0][1][0]) == 5
    assert mass_calls[0][1][1:] == (100.0, 100.0)


def test_flat_stage_bound_fix_order_topology_fix_domain_change() -> None:
    """Within a stage block: stage_open → topology (nodes, elements) →
    fix → domain_change → initial_stress / chain / analyze / stage_close.

    Locks the slot ordering documented in `_emit_stages_flat`."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.fix(nodes=[5], dofs=(1, 1))
        s.mass(nodes=[5], values=(50.0, 50.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()

    stage2_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: install_cimbra")
    )
    elem_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage2_idx and ln.lstrip().startswith("element quad ")
    )
    fix_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage2_idx and ln.lstrip().startswith("fix 5 ")
    )
    mass_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage2_idx and ln.lstrip().startswith("mass 5 ")
    )
    dc_idx = next(
        i for i, ln in enumerate(lines)
        if i > stage2_idx and ln.strip() == "domainChange"
    )
    assert elem_idx < fix_idx < mass_idx < dc_idx, (
        f"slot order broken in stage 2: element={elem_idx} fix={fix_idx} "
        f"mass={mass_idx} domainChange={dc_idx}"
    )


def test_flat_domain_change_fires_on_bcs_only_no_activation() -> None:
    """A stage with stage-bound BCs but NO topology activation must
    still emit `domain_change` (Red #21 unified gate)."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="rock_only") as s:
        s.fix(nodes=[1], dofs=(1, 1))  # BC on globally-emitted node
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_flat(rec)

    stage1 = buckets.get(0, [])
    fix_calls = [c for c in stage1 if c[0] == "fix"]
    dc_calls = [c for c in stage1 if c[0] == "domain_change"]
    assert len(fix_calls) == 1
    assert len(dc_calls) == 1, (
        f"unified domain_change gate broken: expected 1 domain_change "
        f"after BC-only stage, got {len(dc_calls)}"
    )
