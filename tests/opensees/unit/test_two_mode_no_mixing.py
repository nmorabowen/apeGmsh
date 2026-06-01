"""ADR 0051 §5 (BL-4) — two-mode no-mixing guard.

A model is either non-staged (a global ``ops.pattern.*`` + the analysis
chain + ``ops.analyze``/``ops.eigen``) OR staged (every pattern
stage-scoped via ``s.pattern(series=...)``). Registering a global
pattern AND opening a stage raises ``BridgeError`` at build — a global
pattern would fire in every stage's analyze loop and double-apply its
loads across the staged ``loadConst`` boundaries.

(The geometry-case "unconsumed model loads" reconciliation warning +
``ops.ignore_model_loads`` were removed: with loads opt-in, the bridge
deck is authoritative — the user decides which cases to import, and the
bridge does not audit the geometry's case list against it.)

All deck-level; no GPU.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import make_two_column_frame


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


def _frame_ops(fem: Any) -> apeSees:
    ops = apeSees(cast("object", fem), default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="Cols", transf=t, A=0.01, E=200e9, Iz=1e-4)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    return ops


def _add_stage(ops: apeSees, name: str = "push") -> None:
    with ops.stage(name=name) as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def _emit(ops: apeSees) -> RecordingEmitter:
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return rec


def test_global_pattern_plus_stage_raises() -> None:
    ops = _frame_ops(make_two_column_frame())
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=2, forces=(50.0, 0.0, 0.0))
    _add_stage(ops, "push")

    with pytest.raises(BridgeError) as exc:
        ops.build().emit(RecordingEmitter())
    msg = str(exc.value)
    assert "global" in msg
    assert "Plain" in msg            # names the offending pattern kind
    assert "'push'" in msg           # names the stage
    assert "s.pattern" in msg        # points at the staged alternative


def test_imposed_displacement_plus_stage_raises() -> None:
    """``ops.imposed_displacement`` registers a GLOBAL Plain pattern, so
    combining it with a stage trips the no-mixing guard."""
    ops = _frame_ops(make_two_column_frame())
    ops.imposed_displacement(pg="Top", ux=0.01)
    _add_stage(ops, "push")
    with pytest.raises(BridgeError, match="global ops.pattern"):
        ops.build().emit(RecordingEmitter())


def test_uniform_excitation_global_plus_stage_raises() -> None:
    """The guard catches ANY global pattern class, not just Plain."""
    ops = _frame_ops(make_two_column_frame())
    ops.pattern.UniformExcitation(direction=1, series=ops.timeSeries.Linear())
    _add_stage(ops, "shake")
    with pytest.raises(BridgeError) as exc:
        ops.build().emit(RecordingEmitter())
    assert "UniformExcitation" in str(exc.value)


def test_stage_only_patterns_do_not_trip_guard() -> None:
    """A staged model whose only pattern is stage-scoped (s.pattern) is
    valid — it is NOT a global pattern."""
    ops = _frame_ops(make_two_column_frame())
    with ops.stage(name="push") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(node=2, forces=(50.0, 0.0, 0.0))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = _emit(ops)              # must not raise
    assert [c[0] for c in rec.calls].count("pattern_open") == 1


def test_global_pattern_without_stages_is_fine() -> None:
    ops = _frame_ops(make_two_column_frame())
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=2, forces=(50.0, 0.0, 0.0))
    rec = _emit(ops)              # must not raise
    assert "stage_open" not in [c[0] for c in rec.calls]


def test_declared_but_unimported_case_is_silent() -> None:
    """Removal regression: a g.loads case that no pattern imports no
    longer warns — the explicit bridge deck is authoritative, and there
    is no WarnUnconsumedModelLoads / ops.ignore_model_loads surface."""
    import warnings

    from apeGmsh._kernel.record_sets import NodalLoadSet
    from apeGmsh._kernel.records._loads import NodalLoadRecord

    fem = make_two_column_frame()
    fem.nodes.loads = NodalLoadSet([  # type: ignore[attr-defined]
        NodalLoadRecord(node_id=2, force_xyz=(10.0, 0.0, 0.0), pattern="dead"),
    ])
    ops = _frame_ops(fem)
    # No p.from_model("dead"): the build emits NO unconsumed-case warning
    # (filter on the message so unrelated UserWarnings don't matter).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ops.build().emit(RecordingEmitter())
    reconciliation = [
        w for w in caught
        if "declared on the geometry" in str(w.message)
        or "no bridge pattern imported" in str(w.message)
    ]
    assert reconciliation == []
    # And the removed silencer is gone.
    assert not hasattr(ops, "ignore_model_loads")
