"""Phase SSI-2.D / PR-A — stage-bound BC ownership-tier validators.

Covers V1, V2, V3 (V4 lands in PR-C alongside ``s.recorder``):

* **V1** — stage N's BC targets must resolve to a globally-emitted
  node OR a node owned by stage M <= N.  Stage M > N targets crash
  OpenSees at parse time (stage-N block emits before the stage-M
  topology), so refuse at build time.
* **V2** — duplicate ``(node, DOF)`` fix or duplicate node mass
  across global + per-stage tiers — OpenSees rejects duplicate SP
  constraints and silently overwrites mass; refuse explicitly.
* **V3** — region ``name=`` collisions across scopes — OpenSees
  silently appends on duplicate region tag; mangle the name to make
  scope explicit.

Also locks the H1 regression and the new ``_run_staged_bc_validators``
orchestrator path on both ``_emit_flat`` and ``_emit_partitioned``
(the latter previously skipped H1 entirely — PR-A bug fix).

PR-A ships the validators only; ``_StageBuilder.fix`` / ``.mass`` /
``.region`` / ``.recorder`` builder methods land in PR-B / PR-C.
Until then we exercise the validators by direct ``StageRecord``
construction (``dataclasses.replace`` on the builder's appended
record).
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    FixRecord,
    MassRecord,
    RegionAssignmentRecord,
    StageRecord,
)
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# Fixtures (same 2-PG quad pair the H1 / SSI-2.B tests use)
# ---------------------------------------------------------------------------


def _make_two_pg_fem() -> FEMStub:
    """Rock (left quad, global) + cimbra (right quad, stage-bound).

    Nodes 1, 4 belong to the ``Left`` node-PG (global, used by rock).
    Nodes 5, 6 belong to the ``CimbraOnly`` node-PG (stage-bound when
    a stage activates ``cimbra``).  Nodes 2, 3 are shared between
    rock + cimbra and stay global.
    """
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


def _ops_two_stage_with_cimbra_activation() -> apeSees:
    """Stage 1 = rock only, Stage 2 = install cimbra (activates PG)."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    return ops


def _attach_stage_pools(
    ops: apeSees,
    stage_idx: int,
    *,
    fix_records: tuple[FixRecord, ...] = (),
    mass_records: tuple[MassRecord, ...] = (),
    region_records: tuple[RegionAssignmentRecord, ...] = (),
) -> None:
    """Replace ``ops._stage_records[stage_idx]`` with one carrying the
    supplied BC pools.

    PR-A ships the dataclass fields + validators; builder methods
    land in PR-B / PR-C.  Tests use direct construction in the
    meantime to exercise V1 / V2 / V3 paths.
    """
    existing = ops._stage_records[stage_idx]
    ops._stage_records[stage_idx] = replace(
        existing,
        fix_records=fix_records,
        mass_records=mass_records,
        region_records=region_records,
    )


def _stage_pair(ops: apeSees) -> None:
    """Open the canonical 2-stage chain on ``ops`` (rock-only, then
    install-cimbra).  Used by every V1/V2/V3 test that needs to
    populate stage-bound pools after the fact via ``_attach_stage_pools``.
    """
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


# ===========================================================================
# StageRecord shape — new fields default to () and accept tuples
# ===========================================================================


def test_stage_record_new_fields_default_empty() -> None:
    """StageRecord constructed without the new SSI-2.D fields exposes
    them as empty tuples — backcompat for existing builders."""
    sr = StageRecord(
        name="probe",
        initial_stress_records=(),
        test=None, algorithm=None, integrator=None,
        constraints=None, numberer=None, system=None, analysis=None,
        n_increments=1, dt=None,
    )
    assert sr.fix_records == ()
    assert sr.mass_records == ()
    assert sr.region_records == ()
    assert sr.recorder_specs == ()


def test_stage_record_accepts_populated_pools() -> None:
    """StageRecord constructed with all four new pools round-trips
    the tuples losslessly."""
    fix = FixRecord(pg=None, nodes=(5,), dofs=(1, 1))
    mass = MassRecord(pg=None, nodes=(6,), values=(1.0, 1.0))
    region = RegionAssignmentRecord(name="lining_r", pg=None, nodes=(5, 6))
    sr = StageRecord(
        name="probe",
        initial_stress_records=(),
        test=None, algorithm=None, integrator=None,
        constraints=None, numberer=None, system=None, analysis=None,
        n_increments=1, dt=None,
        fix_records=(fix,),
        mass_records=(mass,),
        region_records=(region,),
        recorder_specs=(),
    )
    assert sr.fix_records == (fix,)
    assert sr.mass_records == (mass,)
    assert sr.region_records == (region,)


# ===========================================================================
# V1 — stage N's BC targeting stage M > N
# ===========================================================================


def test_v1_stage_n_fix_targeting_later_stage_node_raises() -> None:
    """Stage 1 (``rock_only``) attaches s.fix on node 5 — but node 5
    only comes online in stage 2 (cimbra activation).  Refuse."""
    ops = _ops_two_stage_with_cimbra_activation()
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=0,  # stage_idx=0 is rock_only
        fix_records=(FixRecord(pg=None, nodes=(5,), dofs=(1, 1)),),
    )
    bm = ops.build()
    with pytest.raises(BridgeError, match="reference nodes owned by a LATER"):
        bm.emit(TclEmitter())


def test_v1_stage_n_mass_targeting_later_stage_node_raises() -> None:
    ops = _ops_two_stage_with_cimbra_activation()
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=0,
        mass_records=(MassRecord(pg=None, nodes=(6,), values=(1.0, 1.0)),),
    )
    bm = ops.build()
    with pytest.raises(BridgeError, match=r"s\.mass"):
        bm.emit(TclEmitter())


def test_v1_stage_n_region_targeting_later_stage_node_raises() -> None:
    ops = _ops_two_stage_with_cimbra_activation()
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=0,
        region_records=(
            RegionAssignmentRecord(name="probe_r", pg=None, nodes=(5,)),
        ),
    )
    bm = ops.build()
    with pytest.raises(BridgeError, match=r"s\.region"):
        bm.emit(TclEmitter())


def test_v1_stage_n_targeting_global_node_passes() -> None:
    """Stage 2 (``install_cimbra``) attaches s.fix on global node 1
    (rock's anchor) — legal, the node exists from the pre-stage block."""
    ops = _ops_two_stage_with_cimbra_activation()
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=1,  # install_cimbra
        fix_records=(FixRecord(pg=None, nodes=(1,), dofs=(1, 1)),),
    )
    bm = ops.build()
    bm.emit(TclEmitter())  # must not raise.


def test_v1_stage_n_targeting_own_stage_node_passes() -> None:
    """Stage 2 attaches s.fix on its own stage-bound node 5 — legal,
    the node emits in stage 2's own topology block before its BCs."""
    ops = _ops_two_stage_with_cimbra_activation()
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=1,
        fix_records=(FixRecord(pg=None, nodes=(5,), dofs=(1, 1)),),
    )
    bm = ops.build()
    bm.emit(TclEmitter())  # must not raise.


# ===========================================================================
# V2 — cross-tier duplicate fix / mass
# ===========================================================================


def test_v2_duplicate_fix_global_and_stage_raises() -> None:
    """Same (node, DOF) fixed globally AND in a stage — OpenSees
    rejects duplicate SP constraints."""
    ops = _ops_two_stage_with_cimbra_activation()
    ops.fix(nodes=[1], dofs=(1, 1))  # node 1 DOF 1 + DOF 2, globally
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=1,
        fix_records=(FixRecord(pg=None, nodes=(1,), dofs=(1, 0)),),
        # duplicate of node 1 DOF 1
    )
    bm = ops.build()
    with pytest.raises(BridgeError, match=r"Duplicate fix / mass"):
        bm.emit(TclEmitter())


def test_v2_duplicate_fix_across_two_stages_raises() -> None:
    ops = _ops_two_stage_with_cimbra_activation()
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=0,
        fix_records=(FixRecord(pg=None, nodes=(1,), dofs=(1, 0)),),
    )
    _attach_stage_pools(
        ops, stage_idx=1,
        fix_records=(FixRecord(pg=None, nodes=(1,), dofs=(1, 0)),),
    )
    bm = ops.build()
    with pytest.raises(
        BridgeError, match=r"fix on node 1 DOF 1.*rock_only.*install_cimbra"
    ):
        bm.emit(TclEmitter())


def test_v2_duplicate_mass_across_tiers_raises() -> None:
    """Same node mass-assigned globally AND in a stage — OpenSees
    setMass silently overwrites; refuse explicitly."""
    ops = _ops_two_stage_with_cimbra_activation()
    ops.mass(nodes=[1], values=(2.0, 2.0))
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=1,
        mass_records=(MassRecord(pg=None, nodes=(1,), values=(3.0, 3.0)),),
    )
    bm = ops.build()
    with pytest.raises(BridgeError, match=r"mass on node 1"):
        bm.emit(TclEmitter())


def test_v2_disjoint_dofs_same_node_passes() -> None:
    """Fix DOF 1 globally + DOF 2 in a stage — no collision; pass."""
    ops = _ops_two_stage_with_cimbra_activation()
    ops.fix(nodes=[1], dofs=(1, 0))  # node 1 DOF 1
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=1,
        fix_records=(FixRecord(pg=None, nodes=(1,), dofs=(0, 1)),),
        # node 1 DOF 2 only — disjoint
    )
    bm = ops.build()
    bm.emit(TclEmitter())  # must not raise.


# ===========================================================================
# V3 — region name collision across scopes
# ===========================================================================


def test_v3_region_name_collision_global_and_stage_raises() -> None:
    ops = _ops_two_stage_with_cimbra_activation()
    ops.region(name="probe_r", nodes=[1, 4])
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=1,
        region_records=(
            RegionAssignmentRecord(name="probe_r", pg=None, nodes=(5,)),
        ),
    )
    bm = ops.build()
    with pytest.raises(BridgeError, match="Region ``name=`` collision"):
        bm.emit(TclEmitter())


def test_v3_region_name_collision_across_two_stages_raises() -> None:
    ops = _ops_two_stage_with_cimbra_activation()
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=0,
        region_records=(
            RegionAssignmentRecord(name="probe_r", pg=None, nodes=(1,)),
        ),
    )
    _attach_stage_pools(
        ops, stage_idx=1,
        region_records=(
            RegionAssignmentRecord(name="probe_r", pg=None, nodes=(5,)),
        ),
    )
    bm = ops.build()
    with pytest.raises(
        BridgeError, match=r"region name 'probe_r'.*rock_only.*install_cimbra"
    ):
        bm.emit(TclEmitter())


def test_v3_distinct_region_names_pass() -> None:
    """Different region names in global + stages emit cleanly."""
    ops = _ops_two_stage_with_cimbra_activation()
    ops.region(name="global_r", nodes=[1, 4])
    _stage_pair(ops)
    _attach_stage_pools(
        ops, stage_idx=1,
        region_records=(
            RegionAssignmentRecord(name="stage_r", pg=None, nodes=(5,)),
        ),
    )
    bm = ops.build()
    bm.emit(TclEmitter())  # must not raise.


# ===========================================================================
# Orchestrator path — _run_staged_bc_validators wires H1 + V1 + V2 + V3
# ===========================================================================


def test_orchestrator_runs_h1_before_v1() -> None:
    """When both H1 and V1 would fire, H1 (the older + more user-
    facing check) wins because ``_run_staged_bc_validators`` calls
    it first.  Locks call ordering for diagnostic-message stability."""
    ops = _ops_two_stage_with_cimbra_activation()
    # H1 trigger: global fix on stage-bound node 5.
    ops.fix(nodes=[5], dofs=(1, 1))
    _stage_pair(ops)
    # V1 trigger: stage_idx=0 fix on node 6 (stage_idx=1's territory).
    _attach_stage_pools(
        ops, stage_idx=0,
        fix_records=(FixRecord(pg=None, nodes=(6,), dofs=(1, 1)),),
    )
    bm = ops.build()
    with pytest.raises(
        BridgeError, match="Stage-bound nodes referenced by GLOBAL"
    ):
        bm.emit(TclEmitter())
