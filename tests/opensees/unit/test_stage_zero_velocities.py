"""``s.zero_velocities(...)`` — the transient → static handover verb.

A static stage inherits the previous transient stage's committed nodal
velocities and accelerations (a static integrator writes neither), so
``-dynamic`` reactions in the static stage keep reporting the previous
stage's inertial / damping terms.  ``s.zero_velocities()`` hands the
stage a quiescent kinematic state.

Locks the emitted deck text on BOTH targets (Tcl + openseespy), the
per-node DOF range (from the effective ndf map, not the envelope), the
line count, and the emit slot (last thing before ``analyze``).
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build import ZeroVelocityRecord
from apeGmsh.opensees.emitter.py import PyEmitter
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


def _make_quad_fem() -> FEMStub:
    """One quad on nodes 1-4, uniform ndf."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            node_pgs={"Left": [1, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
            },
        ),
    )


def _make_mixed_ndf_fem() -> FEMStub:
    """Quad (ndf 2 in 2D) + a disjoint beam leg (ndf 3).

    Nodes 1-4 are quad-only → effective ndf 2.  Nodes 5, 6 carry the
    beam → effective ndf 3.  Exercises the per-node DOF range: the
    zeroing must read the ADR 0048 inferred map, not the ``ops.model``
    envelope.  (A u-p node reaches ndf 4 through the same map.)  The two
    PGs share no node — a node shared between disjoint-ndf element types
    is refused at build time.
    """
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (3.0, 0.0, 0.0),
                (3.0, 1.0, 0.0),
            ],
            node_pgs={"Left": [1, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "brace": _ElementGroupView(
                    ids=(2,), connectivity=((5, 6),),
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


def _quad_ops() -> apeSees:
    ops = apeSees(_make_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    return ops


def _mixed_ops() -> apeSees:
    ops = apeSees(_make_mixed_ndf_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    transf = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(
        pg="brace", transf=transf, A=0.01, E=2e11, Iz=1e-5,
    )
    return ops


# ===========================================================================
# Builder surface
# ===========================================================================


def test_stage_builder_has_zero_velocity_slot() -> None:
    ops = _quad_ops()
    with ops.stage(name="s1") as s:
        assert hasattr(s, "_zero_velocity_records")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_zero_velocities_whole_domain_records_none() -> None:
    ops = _quad_ops()
    with ops.stage(name="s1") as s:
        rec = s.zero_velocities()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert rec == ZeroVelocityRecord(nodes=None)
    assert ops._stage_records[0].zero_velocity_records == (rec,)


def test_zero_velocities_node_set_records_tags() -> None:
    ops = _quad_ops()
    with ops.stage(name="s1") as s:
        rec = s.zero_velocities(nodes=[2, 3])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert rec == ZeroVelocityRecord(nodes=(2, 3))


def test_zero_velocities_rejects_empty_node_set() -> None:
    ops = _quad_ops()
    with ops.stage(name="s1") as s:
        with pytest.raises(ValueError, match="nodes= is empty"):
            s.zero_velocities(nodes=[])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


# ===========================================================================
# Emit slot
# ===========================================================================


def test_zero_velocities_emits_last_before_analyze() -> None:
    """The zeroing must be the LAST thing in the stage block — after
    ``reset`` (which would otherwise restore the velocities) and with
    nothing between it and ``analyze``."""
    ops = _quad_ops()
    with ops.stage(name="quiet") as s:
        s.reset()
        s.zero_velocities(nodes=[1])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    reset_i = names.index("reset")
    analyze_i = names.index("analyze")
    vel_idx = [i for i, n in enumerate(names) if n == "set_node_vel"]
    accel_idx = [i for i, n in enumerate(names) if n == "set_node_accel"]
    assert vel_idx and accel_idx
    assert reset_i < min(vel_idx)
    assert max(accel_idx) == analyze_i - 1


def test_zero_velocities_absent_emits_nothing() -> None:
    ops = _quad_ops()
    with ops.stage(name="plain") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    names = [c[0] for c in rec.calls]
    assert "set_node_vel" not in names
    assert "set_node_accel" not in names


def test_zero_velocities_scoped_to_its_own_stage() -> None:
    ops = _quad_ops()
    with ops.stage(name="dyn") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="static") as s:
        s.zero_velocities(nodes=[1])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    stage_idx = -1
    owner: list[int] = []
    for name, _args, _kw in rec.calls:
        if name == "stage_open":
            stage_idx += 1
        elif name in ("set_node_vel", "set_node_accel"):
            owner.append(stage_idx)
    assert owner and set(owner) == {1}


# ===========================================================================
# Tcl deck text
# ===========================================================================


def test_tcl_whole_domain_lines_and_count() -> None:
    ops = _quad_ops()
    with ops.stage(name="static") as s:
        s.zero_velocities()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    vel = [ln for ln in lines if ln.startswith("setNodeVel ")]
    accel = [ln for ln in lines if ln.startswith("setNodeAccel ")]
    # 4 nodes x ndf 2, one line per (node, DOF), for each of vel + accel.
    assert len(vel) == 8
    assert len(accel) == 8
    assert vel[:2] == [
        "setNodeVel 1 1 0.0 -commit",
        "setNodeVel 1 2 0.0 -commit",
    ]
    assert accel[-1] == "setNodeAccel 4 2 0.0 -commit"
    # Every line carries -commit: without it the stock handler writes
    # only the TRIAL vector and re-reads the OLD committed vector on the
    # next DOF, so only the last DOF would end up zeroed.
    assert all(ln.endswith(" -commit") for ln in vel + accel)


def test_tcl_node_set_form_targets_only_those_nodes() -> None:
    ops = _quad_ops()
    with ops.stage(name="static") as s:
        s.zero_velocities(nodes=[2, 4])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    vel = [ln for ln in lines if ln.startswith("setNodeVel ")]
    accel = [ln for ln in lines if ln.startswith("setNodeAccel ")]
    assert vel == [
        "setNodeVel 2 1 0.0 -commit",
        "setNodeVel 2 2 0.0 -commit",
        "setNodeVel 4 1 0.0 -commit",
        "setNodeVel 4 2 0.0 -commit",
    ]
    assert len(accel) == 4
    assert {ln.split()[1] for ln in accel} == {"2", "4"}


def test_tcl_dof_range_follows_per_node_effective_ndf() -> None:
    """Nodes 5, 6 carry the beam (ndf 3); nodes 1-4 are quad-only
    (ndf 2).  The DOF range per node comes from that map, not from the
    ``ops.model(ndf=3)`` envelope."""
    ops = _mixed_ops()
    with ops.stage(name="static") as s:
        s.zero_velocities()
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    per_node: dict[int, list[int]] = {}
    for ln in emitter.lines():
        if ln.startswith("setNodeVel "):
            _cmd, node, dof, _val, _flag = ln.split()
            per_node.setdefault(int(node), []).append(int(dof))
    assert per_node == {
        1: [1, 2],
        2: [1, 2],
        3: [1, 2],
        4: [1, 2],
        5: [1, 2, 3],
        6: [1, 2, 3],
    }


def test_tcl_repeated_calls_do_not_double_the_deck() -> None:
    ops = _quad_ops()
    with ops.stage(name="static") as s:
        s.zero_velocities(nodes=[1, 2])
        s.zero_velocities(nodes=[2, 3])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    vel = [ln for ln in emitter.lines() if ln.startswith("setNodeVel ")]
    assert [ln.split()[1] for ln in vel] == ["1", "1", "2", "2", "3", "3"]


# ===========================================================================
# openseespy deck text
# ===========================================================================


def test_py_emits_setnodevel_and_setnodeaccel() -> None:
    ops = _quad_ops()
    with ops.stage(name="static") as s:
        s.zero_velocities(nodes=[3])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    emitter = PyEmitter()
    ops.build().emit(emitter)
    lines = [ln.strip() for ln in emitter.lines()]
    assert "ops.setNodeVel(3, 1, 0.0, '-commit')" in lines
    assert "ops.setNodeVel(3, 2, 0.0, '-commit')" in lines
    assert "ops.setNodeAccel(3, 1, 0.0, '-commit')" in lines
    assert "ops.setNodeAccel(3, 2, 0.0, '-commit')" in lines


# ===========================================================================
# H5 archival — deferred, fail-loud
# ===========================================================================


def test_h5_archive_refuses_a_stage_that_zeroes_velocities(tmp_path) -> None:
    """The stage block has no store for the zeroing, so the archive would
    replay the static stage on the transient stage's velocities — exactly
    the artefact the verb removes.  Refuse rather than drop it silently."""
    ops = _quad_ops()
    with ops.stage(name="static") as s:
        s.zero_velocities(nodes=[1])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(NotImplementedError, match="s.zero_velocities"):
        ops.h5(str(tmp_path / "model.h5"))


def test_h5_archive_unaffected_when_the_stage_does_not_zero(tmp_path) -> None:
    ops = _quad_ops()
    with ops.stage(name="static") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    ops.h5(str(tmp_path / "model.h5"))
