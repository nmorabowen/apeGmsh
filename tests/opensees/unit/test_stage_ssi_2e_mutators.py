"""Phase SSI-2.E — between-stage Domain mutator verbs.

Covers:

- ``s.set_time(t)`` / ``s.set_creep(on)`` / ``s.reset()`` — three
  imperative time-state mutators.
- ``s.remove_sp(*, pg=, nodes=, dofs=)`` — SP removal + V5 validator.
- ``s.remove_element(*, pg=, elements=)`` — element removal + V6
  validator.
- ``s.mass(... overwrite=True)`` — V2 relaxation when the user opts in
  explicitly to mid-run mass overwrite.

Single-partition emit shape is locked here; the partitioned emit slot
in ``_emit_stages_partitioned`` is exercised by
``tests/opensees/integration/test_emit_partitioned_stage_ssi_2e.py``.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import _StageBuilder, apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    ElementRemovalRecord,
    MassRecord,
    SPRemovalRecord,
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
# Fixtures
# ---------------------------------------------------------------------------


def _make_two_pg_fem() -> FEMStub:
    """Two-PG quad pair: rock (1) + cimbra (2), 6 nodes."""
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


def _two_stage_ops() -> apeSees:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    return ops


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


# ===========================================================================
# __slots__ + dataclass field declarations
# ===========================================================================


def test_stage_builder_slots_include_ssi_2e_fields() -> None:
    """SSI-2.E adds five `_StageBuilder` slots — without them, calling
    the new verbs raises AttributeError at first call."""
    for slot in (
        "_remove_sp_records",
        "_remove_element_records",
        "_set_time",
        "_set_creep_on",
        "_pre_analyze_reset",
    ):
        assert slot in _StageBuilder.__slots__, (
            f"_StageBuilder missing SSI-2.E slot {slot!r}"
        )


def test_mass_record_has_overwrite_field_defaulting_false() -> None:
    """`MassRecord.overwrite` defaults to False so existing call sites
    that omit the kwarg keep V2 active."""
    assert "overwrite" in MassRecord.__dataclass_fields__
    rec = MassRecord(pg=None, nodes=(1,), values=(1.0, 1.0))
    assert rec.overwrite is False


def test_removal_record_dataclasses() -> None:
    """The two new removal dataclasses accept the expected fields."""
    sp = SPRemovalRecord(pg="Left", nodes=None, dofs=(1, 2))
    assert sp.pg == "Left"
    assert sp.nodes is None
    assert sp.dofs == (1, 2)
    ele = ElementRemovalRecord(pg=None, elements=(7, 8))
    assert ele.pg is None
    assert ele.elements == (7, 8)


# ===========================================================================
# Builder positive — records flow into StageRecord
# ===========================================================================


def test_s_remove_sp_populates_stage_record() -> None:
    ops = _two_stage_ops()
    ops.fix(pg="Left", dofs=(1, 1))  # global SP that stage 1 will release
    with ops.stage(name="release") as s:
        s.remove_sp(pg="Left", dofs=(1, 2))
        s.remove_sp(nodes=[1], dofs=(1,))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    recs = ops._stage_records[0].remove_sp_records
    assert len(recs) == 2
    assert recs[0] == SPRemovalRecord(pg="Left", nodes=None, dofs=(1, 2))
    assert recs[1] == SPRemovalRecord(pg=None, nodes=(1,), dofs=(1,))


def test_s_remove_element_populates_stage_record() -> None:
    ops = _two_stage_ops()  # globally emits rock (eid=1) and cimbra (eid=2)
    with ops.stage(name="excavate") as s:
        s.remove_element(pg="rock")
        s.remove_element(elements=[2])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    recs = ops._stage_records[0].remove_element_records
    assert len(recs) == 2
    assert recs[0] == ElementRemovalRecord(pg="rock", elements=None)
    assert recs[1] == ElementRemovalRecord(pg=None, elements=(2,))


def test_s_set_time_creep_reset_populate_stage_record() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="dyn") as s:
        s.set_time(t=10.0)
        s.set_creep(on=True)
        s.reset()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = ops._stage_records[0]
    assert rec.set_time == 10.0
    assert rec.set_creep_on is True
    assert rec.pre_analyze_reset is True


def test_stage_without_ssi_2e_verbs_has_empty_defaults() -> None:
    """A stage that never uses the SSI-2.E verbs keeps empty / None
    defaults — old fixtures are unaffected."""
    ops = _two_stage_ops()
    with ops.stage(name="bare") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = ops._stage_records[0]
    assert rec.remove_sp_records == ()
    assert rec.remove_element_records == ()
    assert rec.set_time is None
    assert rec.set_creep_on is None
    assert rec.pre_analyze_reset is False


# ===========================================================================
# Builder negative — XOR + empty-dofs guard
# ===========================================================================


def test_s_remove_sp_rejects_both_pg_and_nodes() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or nodes="):
            s.remove_sp(pg="Left", nodes=[1], dofs=(1,))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_remove_sp_rejects_empty_dofs() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="dofs= must contain at least one"):
            s.remove_sp(nodes=[1], dofs=())
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_remove_element_rejects_both_pg_and_elements() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or elements="):
            s.remove_element(pg="rock", elements=[1])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_s_remove_element_rejects_neither_pg_nor_elements() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="bad") as s:
        with pytest.raises(ValueError, match="exactly one of pg= or elements="):
            s.remove_element()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


# ===========================================================================
# Flat emit shape — single-partition
# ===========================================================================


def test_flat_set_time_emits_right_after_stage_open() -> None:
    """``setTime`` must emit between ``stage_open`` and any topology /
    BC / domain_change.  Position is load-bearing: emitting it AFTER
    fix would overwrite a value the user just set."""
    ops = _two_stage_ops()
    with ops.stage(name="stg") as s:
        s.set_time(t=42.0)
        s.fix(nodes=[1], dofs=(1, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    stage_open_i = names.index("stage_open")
    set_time_i = names.index("set_time")
    fix_i = names.index("fix")
    assert stage_open_i < set_time_i < fix_i


def test_flat_set_creep_emits_after_set_time() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="stg") as s:
        s.set_time(t=5.0)
        s.set_creep(on=True)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    assert names.index("set_time") < names.index("set_creep")
    # set_creep True -> emitter call carries True
    creep_call = next(c for c in rec.calls if c[0] == "set_creep")
    assert creep_call[1] == (True,)


def test_flat_reset_emits_right_before_analyze() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="stg") as s:
        s.reset()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    reset_i = names.index("reset")
    analyze_i = names.index("analyze")
    assert reset_i < analyze_i
    # No other top-level emit (other than chain bind) between reset and analyze.
    between = names[reset_i + 1:analyze_i]
    assert between == [], f"unexpected calls between reset and analyze: {between}"


def test_flat_removals_emit_before_new_fix_in_same_stage() -> None:
    """Atomic-replace pattern: a stage that removes a global SP and
    re-fixes the same DOF must emit ``remove sp`` BEFORE the new
    ``fix`` line so OpenSees parses the release-then-set sequence."""
    ops = _two_stage_ops()
    ops.fix(pg="Left", dofs=(1, 1))  # global SP on nodes 1, 4
    with ops.stage(name="replace") as s:
        s.remove_sp(nodes=[1], dofs=(2,))
        s.fix(nodes=[1], dofs=(0, 1))  # opposite SP pattern on same node
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    # remove_sp emits with node 1
    rsp_i = next(
        i for i, c in enumerate(rec.calls)
        if c[0] == "remove_sp" and c[1] == (1, 2)
    )
    # Find the fix call for node 1 INSIDE the stage block (skip the
    # global fix at the top of the deck).
    stage_open_i = names.index("stage_open")
    fix_in_stage = [
        i for i, c in enumerate(rec.calls)
        if i > stage_open_i and c[0] == "fix" and c[1][0] == 1
    ]
    assert fix_in_stage, "expected an in-stage fix on node 1"
    assert rsp_i < fix_in_stage[0]


def test_flat_remove_element_pg_resolves_to_ops_tag() -> None:
    """``s.remove_element(pg=...)`` resolves the PG via
    expand_pg_to_elements and emits one ``remove_element($ops_tag)``
    per element — the FEM eid is translated via fem_eid_to_ops_tag
    so the deck carries OpenSees tags, not FEM eids."""
    ops = _two_stage_ops()
    with ops.stage(name="excavate") as s:
        s.remove_element(pg="rock")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    # The first ``element`` call in the deck is rock (registration
    # order); the remove_element call must use the SAME ops tag.
    elem_calls = [c for c in rec.calls if c[0] == "element"]
    rock_ops_tag = elem_calls[0][1][1]
    re_calls = [c for c in rec.calls if c[0] == "remove_element"]
    assert len(re_calls) == 1
    assert re_calls[0][1] == (rock_ops_tag,)


def test_flat_remove_element_explicit_tag_passes_through() -> None:
    """``elements=[fem_eid]`` translates through fem_eid_to_ops_tag
    so the emitted ``remove_element($ops_tag)`` carries the OpenSees
    tag the rest of the deck uses (recorder.Element convention)."""
    ops = _two_stage_ops()
    with ops.stage(name="excavate") as s:
        s.remove_element(elements=[2])  # fem_eid 2 = cimbra
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    elem_calls = [c for c in rec.calls if c[0] == "element"]
    # Second ``element`` call is cimbra (registration order: rock, cimbra).
    cimbra_ops_tag = elem_calls[1][1][1]
    re_calls = [c for c in rec.calls if c[0] == "remove_element"]
    assert re_calls == [("remove_element", (cimbra_ops_tag,), {})]


def test_flat_removals_trigger_domain_change_with_no_other_content() -> None:
    """A stage that does ONLY a remove_sp (no fix/mass/region/activation)
    must still emit ``domain_change`` — the Domain mutated."""
    ops = _two_stage_ops()
    ops.fix(pg="Left", dofs=(1, 1))
    with ops.stage(name="release") as s:
        s.remove_sp(nodes=[1], dofs=(1,))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    buckets = _bucket_flat(rec)
    stage0 = buckets.get(0, [])
    assert [c for c in stage0 if c[0] == "remove_sp"]
    assert [c for c in stage0 if c[0] == "domain_change"]


# ===========================================================================
# Tcl emitter — concrete deck text
# ===========================================================================


def test_tcl_emit_set_time_set_creep_reset_remove_lines() -> None:
    """Locks the actual emitted Tcl tokens for the five SSI-2.E verbs.

    The ``remove element`` line carries the OpenSees ops tag (not the
    FEM eid); read it back from the deck so the test is robust against
    allocator-order changes.
    """
    ops = _two_stage_ops()
    ops.fix(pg="Left", dofs=(1, 1))
    with ops.stage(name="all") as s:
        s.set_time(t=10.0)
        s.set_creep(on=True)
        s.remove_sp(nodes=[1], dofs=(2,))
        s.remove_element(elements=[2])  # fem_eid 2 = cimbra
        s.reset()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    text = "\n".join(lines)
    assert "setTime 10.0" in text
    assert "setCreep 1" in text
    assert "remove sp 1 2" in text
    # Find the cimbra element line (the second ``element`` directive)
    # to pick its OpenSees tag, then verify the remove line cites it.
    elem_lines = [ln for ln in lines if ln.startswith("element ")]
    assert len(elem_lines) >= 2
    cimbra_tokens = elem_lines[1].split()
    cimbra_ops_tag = int(cimbra_tokens[2])  # "element <type> <tag> ..."
    assert f"remove element {cimbra_ops_tag}" in text
    # ``reset`` must appear exactly once (this stage requested one).
    assert text.count("\nreset\n") + (1 if text.startswith("reset\n") else 0) == 1


def test_tcl_set_creep_off_emits_zero() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="no_creep") as s:
        s.set_creep(on=False)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    assert "setCreep 0" in "\n".join(emitter.lines())


# ===========================================================================
# Mass overwrite — V2 relaxation
# ===========================================================================


def test_mass_overwrite_false_default_keeps_v2_active() -> None:
    """Without ``overwrite=True``, two mass declarations on the same
    node (one global, one stage) trip V2 with both tiers named."""
    ops = _two_stage_ops()
    ops.mass(nodes=[1], values=(10.0, 10.0))
    with ops.stage(name="dyn") as s:
        s.mass(nodes=[1], values=(20.0, 20.0))  # NO overwrite — should trip
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="Duplicate fix / mass"):
        ops.build().emit(RecordingEmitter())


def test_mass_overwrite_true_skips_v2() -> None:
    """``overwrite=True`` opts the record out of V2's duplicate-mass
    refusal.  The emitted ``mass`` line is unchanged (overwrite is a
    build-time validator-bypass marker, not an emit-time differentiator)."""
    ops = _two_stage_ops()
    ops.mass(nodes=[1], values=(10.0, 10.0))
    with ops.stage(name="dyn") as s:
        s.mass(nodes=[1], values=(20.0, 20.0), overwrite=True)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)  # no raise
    mass_calls = [c for c in rec.calls if c[0] == "mass"]
    # Two mass calls — global first, stage second — both targeting node 1.
    assert [c[1] for c in mass_calls] == [
        (1, 10.0, 10.0),
        (1, 20.0, 20.0),
    ]


def test_apesees_mass_accepts_overwrite_kwarg() -> None:
    """``apeSees.mass`` (global) accepts the same ``overwrite=`` kwarg
    for symmetry with the stage-bound builder."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    ops.mass(nodes=[1], values=(1.0, 1.0), overwrite=True)
    assert ops._mass_records[0].overwrite is True


# ===========================================================================
# V5 — remove_sp target must exist in an earlier tier
# ===========================================================================


def test_v5_passes_for_valid_remove_from_global_pool() -> None:
    """remove_sp on an SP declared in the global pool is legal."""
    ops = _two_stage_ops()
    ops.fix(pg="Left", dofs=(1, 1))  # alive for stage
    with ops.stage(name="release") as s:
        s.remove_sp(nodes=[1, 4], dofs=(1,))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    ops.build().emit(RecordingEmitter())  # no raise


def test_v5_passes_for_remove_of_earlier_stage_fix() -> None:
    """remove_sp on an SP declared by a strictly-earlier stage is legal."""
    ops = _two_stage_ops()
    with ops.stage(name="setup") as s:
        s.fix(nodes=[1], dofs=(1, 1))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="release") as s:
        s.remove_sp(nodes=[1], dofs=(1, 2))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    ops.build().emit(RecordingEmitter())  # no raise


def test_v5_fails_when_sp_never_declared() -> None:
    """remove_sp on (node, dof) that no earlier tier declared is V5."""
    ops = _two_stage_ops()
    # No global fix — but stage tries to remove node 1 DOF 1.
    with ops.stage(name="bad") as s:
        s.remove_sp(nodes=[1], dofs=(1,))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="s.remove_sp targets"):
        ops.build().emit(RecordingEmitter())


def test_v5_fails_when_sp_already_removed_in_earlier_stage() -> None:
    """Double-removal across stages is V5."""
    ops = _two_stage_ops()
    ops.fix(nodes=[1], dofs=(1, 1))
    with ops.stage(name="release_once") as s:
        s.remove_sp(nodes=[1], dofs=(1,))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="release_again") as s:
        s.remove_sp(nodes=[1], dofs=(1,))  # already gone — V5
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="s.remove_sp targets"):
        ops.build().emit(RecordingEmitter())


def test_v5_fails_for_same_stage_fix_plus_remove() -> None:
    """Same-stage ``s.fix`` does NOT make the SP available for same-
    stage ``s.remove_sp`` (removal emits BEFORE fix in the stage
    block).  V5 must reject."""
    ops = _two_stage_ops()
    with ops.stage(name="bad") as s:
        s.fix(nodes=[1], dofs=(1, 1))  # would emit AFTER remove_sp
        s.remove_sp(nodes=[1], dofs=(1,))  # tries to release prior — none exists
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="s.remove_sp targets"):
        ops.build().emit(RecordingEmitter())


# ===========================================================================
# V6 — remove_element target must exist in an earlier scope (or this stage)
# ===========================================================================


def test_v6_passes_for_remove_of_global_element() -> None:
    ops = _two_stage_ops()  # rock + cimbra both global
    with ops.stage(name="excavate") as s:
        s.remove_element(pg="rock")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    ops.build().emit(RecordingEmitter())  # no raise


def test_v6_passes_for_remove_of_same_stage_activation() -> None:
    """A stage may remove what it just activated (activation emits
    BEFORE the removal block within the same stage)."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    with ops.stage(name="install_then_remove") as s:
        s.activate(pgs=["cimbra"])
        s.remove_element(pg="cimbra")  # legal — activated by this stage
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    ops.build().emit(RecordingEmitter())  # no raise


def test_v6_fails_when_explicit_tag_not_alive() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="bad") as s:
        s.remove_element(elements=[9999])  # nonexistent
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="s.remove_element targets"):
        ops.build().emit(RecordingEmitter())


def test_v6_fails_when_element_already_removed_earlier_stage() -> None:
    ops = _two_stage_ops()
    with ops.stage(name="first") as s:
        s.remove_element(pg="rock")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="again") as s:
        s.remove_element(pg="rock")  # already removed
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="s.remove_element targets"):
        ops.build().emit(RecordingEmitter())


def test_v6_fails_when_pg_unknown_on_fem() -> None:
    """A typo in the PG name surfaces inside V6's offender list."""
    ops = _two_stage_ops()
    with ops.stage(name="bad") as s:
        s.remove_element(pg="doesnotexist")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="s.remove_element pg='doesnotexist'"):
        ops.build().emit(RecordingEmitter())


# ===========================================================================
# Atomic-replace — V5 + V2 cooperate
# ===========================================================================


def test_atomic_replace_global_sp_then_restage_with_new_value() -> None:
    """The end-to-end "release prior + re-fix in same stage" workflow
    must pass BOTH V5 (removal targets prior SP) AND V2 (new fix is
    not a duplicate because the prior SP was just released)."""
    ops = _two_stage_ops()
    ops.fix(nodes=[1], dofs=(1, 1))
    with ops.stage(name="swap") as s:
        s.remove_sp(nodes=[1], dofs=(1, 2))
        s.fix(nodes=[1], dofs=(0, 1))  # different fixity pattern
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    ops.build().emit(RecordingEmitter())  # both validators pass
