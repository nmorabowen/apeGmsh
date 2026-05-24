"""Regression tests for the SSI post-merge cleanup (red-team findings).

Covers:

* H1: fix / mass / region directives targeting stage-bound nodes →
  ``BridgeError`` at emit time.
* H2: duplicate ``initial_stress(name=...)`` across stages (or
  between global + stage pool) → ``BridgeError`` at emit time.
* H3: ``ops.imposed_displacement(uz=...)`` on an ``ndf=2`` model →
  ``ValueError`` at registration.
* M1: ``s.activate(pgs=[unknown])`` referencing a PG that no
  Element primitive carries → ``BridgeError`` at emit time.
* M2: ``ops.eigen(...)`` on a staged model → ``NotImplementedError``
  upfront.
* M4: nested ``with ops.stage(...)`` blocks → ``RuntimeError`` on
  the inner ``apeSees.stage(...)`` call.
* M6: ``ops.initial_stress(elements=[bad_id])`` where ``bad_id`` is
  not on any registered Element primitive → ``BridgeError`` at
  emit time.

Each fix also has a "negative" test confirming the validation
does NOT false-positive on correct inputs.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
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
    """Rock (left quad) + cimbra (right quad), sharing nodes 2 & 3."""
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
                "CimbraOnly": [5, 6],  # nodes exclusive to cimbra
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
# H1 — fix / mass / region on stage-bound nodes
# ===========================================================================


def test_h1_fix_on_stage_bound_node_raises() -> None:
    """``ops.fix(pg='CimbraOnly')`` would emit before stage 2's
    cimbra topology declares those nodes — should raise."""
    ops = _two_stage_ops_with_cimbra_activation()
    ops.fix(pg="CimbraOnly", dofs=(1, 1))   # nodes 5, 6 — stage-bound
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    with pytest.raises(BridgeError, match="Stage-bound nodes referenced by"):
        bm.emit(TclEmitter())


def test_h1_mass_on_stage_bound_node_raises() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    ops.mass(pg="CimbraOnly", values=(1.0, 1.0))
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    with pytest.raises(BridgeError, match=r"mass \('CimbraOnly'\)"):
        bm.emit(TclEmitter())


def test_h1_global_fix_on_global_node_passes() -> None:
    """Negative: ``ops.fix(pg='Left')`` on rock's global nodes
    1 & 4 is fine — they exist in the global emit zone."""
    ops = _two_stage_ops_with_cimbra_activation()
    ops.fix(pg="Left", dofs=(1, 1))  # nodes 1, 4 — global (rock)
    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    bm.emit(TclEmitter())  # must not raise.


# ===========================================================================
# H2 — duplicate initial_stress names across stages
# ===========================================================================


def test_h2_duplicate_name_across_stages_raises() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="A") as s:
        s.add(ops.initial_stress(
            name="rock_in", pg="rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=10,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="B") as s:
        s.add(ops.initial_stress(
            name="rock_in",  # duplicate name!
            pg="rock",
            sigma_xx=-200.0, sigma_yy=-200.0, sigma_zz=-200.0,
            ramp_steps=5,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    with pytest.raises(BridgeError, match="registered twice"):
        bm.emit(TclEmitter())


def test_h2_duplicate_name_global_and_stage_raises() -> None:
    """Same name registered globally AND in a stage — also collides."""
    ops = _two_stage_ops_with_cimbra_activation()
    # Don't ``s.add()`` this one — leaves it in the global pool.
    ops.initial_stress(
        name="rock_in", pg="rock",
        sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
        ramp_steps=10,
    )
    with ops.stage(name="A") as s:
        s.add(ops.initial_stress(
            name="rock_in",  # duplicate of the global one!
            pg="rock",
            sigma_xx=-200.0, sigma_yy=-200.0, sigma_zz=-200.0,
            ramp_steps=5,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    with pytest.raises(BridgeError, match="registered twice"):
        bm.emit(TclEmitter())


def test_h2_unique_names_pass() -> None:
    """Negative: different names across stages emit cleanly."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="A") as s:
        s.add(ops.initial_stress(
            name="rock_in_A", pg="rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=10,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="B") as s:
        s.add(ops.initial_stress(
            name="rock_in_B",
            pg="rock",
            sigma_xx=-200.0, sigma_yy=-200.0, sigma_zz=-200.0,
            ramp_steps=5,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    bm.emit(TclEmitter())  # must not raise.


# ===========================================================================
# H3 — imposed_displacement(uz=) on ndf=2
# ===========================================================================


def test_h3_uz_on_ndf2_raises() -> None:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match=r"uz= targets DOF 3"):
        ops.imposed_displacement(pg="rock", ux=1.0, uz=2.0)


def test_h3_uy_on_ndf1_raises() -> None:
    """ndf=1 only has DOF 1 — uy targets DOF 2."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=1)
    with pytest.raises(ValueError, match=r"uy= targets DOF 2"):
        ops.imposed_displacement(pg="rock", ux=1.0, uy=2.0)


def test_h3_ux_uy_on_ndf2_passes() -> None:
    """Negative: ux + uy on ndf=2 is fine."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    ops.imposed_displacement(pg="rock", ux=1.0, uy=2.0)


def test_h3_uz_on_ndf3_passes() -> None:
    """Negative: uz on ndf=3 is fine."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=3)
    ops.imposed_displacement(pg="rock", ux=1.0, uz=3.0)


def test_h3_before_model_call_skips_check() -> None:
    """If model() hasn't been called yet, the helper accepts any
    uz= — the validation is best-effort and runs only when ``_ndf``
    is known."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    # No ops.model() call yet.
    ops.imposed_displacement(pg="rock", ux=1.0, uz=3.0)


# ===========================================================================
# M1 — activate(pgs=) on unknown PG
# ===========================================================================


def test_m1_activate_unknown_pg_raises() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="A") as s:
        s.activate(pgs=["nonexistent_pg"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    with pytest.raises(BridgeError, match="references unknown element PGs"):
        bm.emit(TclEmitter())


def test_m1_activate_typo_in_pg_raises() -> None:
    """Realistic scenario: user typos a PG name — silent failure
    would emit no domainChange and no cimbra topology."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="A") as s:
        s.activate(pgs=["cimba"])  # typo: missing 'r'
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    with pytest.raises(BridgeError, match=r"'cimba'"):
        bm.emit(TclEmitter())


def test_m1_activate_real_pg_passes() -> None:
    """Negative: activating an actual element PG works."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="A") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    bm = ops.build()
    bm.emit(TclEmitter())


# ===========================================================================
# M2 — eigen on staged model
# ===========================================================================


def test_m2_eigen_on_staged_raises() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="A") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(
        NotImplementedError, match="does not support staged",
    ):
        ops.eigen(num_modes=1)


# ===========================================================================
# M4 — nested ops.stage(...)
# ===========================================================================


def test_m4_nested_stage_raises() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    with pytest.raises(
        RuntimeError, match="a stage is already open",
    ):
        with ops.stage(name="outer") as s_outer:
            with ops.stage(name="inner") as s_inner:  # should raise
                pass


def test_m4_sequential_stages_pass() -> None:
    """Negative: two sequential (non-nested) ``with ops.stage(...)``
    blocks work correctly."""
    ops = _two_stage_ops_with_cimbra_activation()
    with ops.stage(name="A") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="B") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert len(ops._stage_records) == 2


def test_m4_stage_exception_clears_builder_slot() -> None:
    """An exception inside ``with ops.stage(...)`` must still clear
    the open-builder slot so subsequent stages work."""
    ops = _two_stage_ops_with_cimbra_activation()
    with pytest.raises(RuntimeError, match="user"):
        with ops.stage(name="bad") as s:
            raise RuntimeError("user exception")
    # Slot cleared — opening a new stage works.
    assert ops._open_stage_builder is None
    with ops.stage(name="ok") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert len(ops._stage_records) == 1


# ===========================================================================
# M6 — initial_stress(elements=[bad_id])
# ===========================================================================


def test_m6_unregistered_element_id_raises() -> None:
    ops = _two_stage_ops_with_cimbra_activation()
    # Element id 9999 doesn't match any registered Element primitive
    # (the only PGs are "rock" → eid 1 and "cimbra" → eid 2).
    ops.initial_stress(
        name="r", elements=[9999],
        sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
        ramp_steps=10,
    )
    bm = ops.build()
    with pytest.raises(BridgeError, match="not registered with any Element"):
        bm.emit(TclEmitter())


def test_m6_registered_element_id_passes() -> None:
    """Negative: explicit ``elements=[1]`` (rock's element id) emits
    correctly."""
    ops = _two_stage_ops_with_cimbra_activation()
    ops.initial_stress(
        name="r", elements=[1],  # matches rock's element id
        sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
        ramp_steps=10,
    )
    bm = ops.build()
    bm.emit(TclEmitter())
