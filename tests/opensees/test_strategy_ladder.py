"""ADR 0057 Phase A — solution-strategy ladder contract.

Pins, per the ADR:

1. **Established profiles** — canonical names + aliases resolve, the
   evidence-based exclusion holds (``"non-smooth"`` has NO line-search
   rung), unknown names fail loud listing the catalog.
2. **Ladder validation** — algorithm-primitives-only rungs (tolerance /
   test / integrator changes are excluded by design), bounded depth.
3. **Spec resolution** — the chain's algorithm becomes rung 0
   (deduplicated when listed explicitly); a ladder that resolves to a
   single rung is rejected (it is just the plain loop).
4. **Deck emission (py + tcl)** — ``strategy=None`` stays byte-shape
   identical to the pre-0057 fail-loud loop; with a ladder, the loop
   carries the rungs literal, the loud per-escalation print, the
   rung-0 restore, and the exhaustion banner naming the ladder.
5. **Stage plumbing** — ``s.run(..., strategy=)`` reaches the emitted
   deck through the full build pipeline (FEMStub one-quad model).
6. **Runtime behavior** (openseespy subprocess): a converging deck
   runs clean with a ladder attached (no escalation prints); a
   non-converging deck walks every rung and aborts with the
   exhaustion banner (exit != 0) — the #587 fail-loud floor holds.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from apeGmsh.opensees.analysis.algorithm import (
    KrylovNewton,
    ModifiedNewton,
    Newton,
    NewtonLineSearch,
)
from apeGmsh.opensees.analysis.strategy import PROFILE_NAMES, Ladder, profile
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.base import StrategySpec
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# 1. Established profiles
# ---------------------------------------------------------------------------


def test_profile_canonical_names() -> None:
    for name in PROFILE_NAMES:
        lad = profile(name)
        assert isinstance(lad, Ladder)
        assert lad.name == name
        assert lad.rungs


def test_profile_aliases_resolve_to_canonical() -> None:
    assert profile("geotech").name == "non-smooth"
    assert profile("mohr-coulomb").name == "non-smooth"
    assert profile("metal").name == "smooth-hardening"


def test_profile_non_smooth_has_no_line_search_rung() -> None:
    # The 2026-06-10 zoned-twin evidence: NewtonLineSearch IS the
    # failure mode on MC/DP yield-surface kinks — escalating into it
    # would be harmful, so the profile excludes it BY DESIGN.
    lad = profile("non-smooth")
    assert not any(isinstance(r, NewtonLineSearch) for r in lad.rungs)


def test_profile_unknown_name_fails_loud_with_catalog() -> None:
    with pytest.raises(ValueError, match="non-smooth"):
        profile("does-not-exist")


def test_profile_returns_fresh_editable_values() -> None:
    a = profile("standard")
    b = profile("standard")
    assert a is not b
    extended = Ladder(rungs=a.rungs + (KrylovNewton(),), name="mine")
    assert len(extended.rungs) == len(a.rungs) + 1


# ---------------------------------------------------------------------------
# 2. Ladder validation
# ---------------------------------------------------------------------------


def test_ladder_rejects_empty_rungs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Ladder(rungs=())


def test_ladder_rejects_excessive_depth() -> None:
    with pytest.raises(ValueError, match="at most"):
        Ladder(rungs=tuple(Newton() for _ in range(9)))


def test_ladder_rejects_non_algorithm_rungs() -> None:
    # Tolerance/test/integrator rungs are excluded by design (ADR 0057
    # §6) — only solution algorithms may appear.
    with pytest.raises(TypeError, match="solution-algorithm"):
        Ladder(rungs=("Newton",))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Spec resolution
# ---------------------------------------------------------------------------


def test_to_spec_prepends_base_as_rung0() -> None:
    spec = Ladder(
        rungs=(ModifiedNewton(tangent="initial"),), name="x",
    ).to_spec(base=Newton())
    assert spec.rungs == (("Newton",), ("ModifiedNewton", "-initial"))


def test_to_spec_deduplicates_explicit_base() -> None:
    spec = Ladder(
        rungs=(Newton(), ModifiedNewton(tangent="initial")), name="x",
    ).to_spec(base=Newton())
    assert spec.rungs == (("Newton",), ("ModifiedNewton", "-initial"))


def test_to_spec_base_none_promotes_first_rung() -> None:
    spec = Ladder(
        rungs=(Newton(), KrylovNewton()), name="x",
    ).to_spec(base=None)
    assert spec.rungs[0] == ("Newton",)


def test_spec_rejects_single_rung() -> None:
    # base == the only rung → 1-rung spec == the plain fail-loud loop;
    # reject rather than emit a degenerate ladder.
    with pytest.raises(ValueError, match="at least 2 rungs"):
        Ladder(rungs=(Newton(),), name="x").to_spec(base=Newton())


# ---------------------------------------------------------------------------
# 4. Deck emission shape
# ---------------------------------------------------------------------------


def _spec() -> StrategySpec:
    return profile("non-smooth").to_spec(base=Newton())


def test_py_emission_without_strategy_unchanged() -> None:
    em = PyEmitter()
    em.analyze(steps=5, label="S")
    text = "\n".join(em.lines())
    assert "for _apesees_i in range(5):" in text
    assert "_apesees_rungs" not in text
    assert "analyze FAILED at increment" in text


def test_py_emission_with_strategy() -> None:
    em = PyEmitter()
    em.analyze(steps=5, dt=0.1, label="Gravity", strategy=_spec())
    text = "\n".join(em.lines())
    assert "_apesees_rungs = [('Newton',), ('ModifiedNewton', '-initial',), ('KrylovNewton',)]" in text
    assert "ops.algorithm(*_apesees_rungs[_apesees_r])" in text   # escalation
    assert "ops.algorithm(*_apesees_rungs[0])" in text            # rung-0 restore
    assert "apeGmsh strategy 'non-smooth': increment" in text     # loud provenance
    assert "exhausting strategy ladder 'non-smooth' (3 rungs)" in text
    assert "of stage 'Gravity'" in text


def test_tcl_emission_without_strategy_unchanged() -> None:
    em = TclEmitter()
    em.analyze(steps=5, label="S")
    text = "\n".join(em.lines())
    assert "_apesees_rungs" not in text
    assert "analyze FAILED at increment" in text


def test_tcl_emission_with_strategy() -> None:
    em = TclEmitter()
    em.analyze(steps=5, dt=0.1, label="Gravity", strategy=_spec())
    text = "\n".join(em.lines())
    assert "set _apesees_rungs {{Newton} {ModifiedNewton -initial} {KrylovNewton}}" in text
    assert "eval algorithm $_apesees_rung" in text
    assert "eval algorithm [lindex $_apesees_rungs 0]" in text
    assert "exhausting strategy ladder 'non-smooth' (3 rungs)" in text


# ---------------------------------------------------------------------------
# 5. Stage plumbing (full build pipeline over a one-quad stub)
# ---------------------------------------------------------------------------


def _quad_fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            node_pgs={"Rock": [1, 2, 3, 4], "Base": [1, 2], "Top": [3, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
            },
        ),
    )


def _staged_bridge(*, max_iter: int, strategy: Ladder | None) -> apeSees:
    ops = apeSees(_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=[1, 1])
    ts = ops.timeSeries.Linear()
    with ops.stage(name="Push") as s:
        with s.pattern(series=ts) as p:
            p.load(pg="Top", forces=(0.0, -10.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=max_iter),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.5),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=2, dt=0.5, strategy=strategy)
    return ops


def test_stage_strategy_reaches_py_deck(tmp_path: Path) -> None:
    ops = _staged_bridge(max_iter=50, strategy=profile("non-smooth"))
    deck = tmp_path / "deck.py"
    ops.py(str(deck), run=False)
    text = deck.read_text()
    assert "_apesees_rungs = [('Newton',), ('ModifiedNewton', '-initial',), ('KrylovNewton',)]" in text
    assert "exhausting strategy ladder 'non-smooth'" in text
    assert "of stage 'Push'" in text


def test_stage_strategy_reaches_tcl_deck(tmp_path: Path) -> None:
    ops = _staged_bridge(max_iter=50, strategy=profile("non-smooth"))
    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck), run=False)
    text = deck.read_text()
    assert "set _apesees_rungs {{Newton} {ModifiedNewton -initial} {KrylovNewton}}" in text
    assert "exhausting strategy ladder 'non-smooth'" in text


def test_stage_without_strategy_deck_unchanged(tmp_path: Path) -> None:
    ops = _staged_bridge(max_iter=50, strategy=None)
    deck = tmp_path / "deck.py"
    ops.py(str(deck), run=False)
    assert "_apesees_rungs" not in deck.read_text()


# ---------------------------------------------------------------------------
# 6. Runtime behavior (openseespy subprocess)
# ---------------------------------------------------------------------------


def _run_deck(deck: Path) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, str(deck)],
        capture_output=True, text=True, check=False, timeout=120,
        env={**os.environ, "LADRUNO_OPENSEES_QUIET": "1",
             "PYTHONIOENCODING": "utf-8"},
    )


@pytest.mark.live
def test_laddered_deck_converging_runs_clean(tmp_path: Path) -> None:
    pytest.importorskip("openseespy.opensees")
    ops = _staged_bridge(max_iter=50, strategy=profile("non-smooth"))
    deck = tmp_path / "deck.py"
    ops.py(str(deck), run=False)
    proc = _run_deck(deck)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # rung 0 carried every increment — no escalation prints.
    assert "apeGmsh strategy" not in proc.stdout


@pytest.mark.live
def test_laddered_deck_exhaustion_fails_loud(tmp_path: Path) -> None:
    pytest.importorskip("openseespy.opensees")
    # max_iter=1: Newton on even a LINEAR step needs 2 iterations to
    # certify NormDispIncr (solve + zero-correction check), so every
    # rung fails and the deck must abort with the exhaustion banner —
    # the #587 fail-loud floor survives the ladder.
    ops = _staged_bridge(max_iter=1, strategy=profile("non-smooth"))
    deck = tmp_path / "deck.py"
    ops.py(str(deck), run=False)
    proc = _run_deck(deck)
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "apeGmsh strategy 'non-smooth': increment 1/2 of stage 'Push' -> rung 1" in out
    assert "exhausting strategy ladder 'non-smooth' (3 rungs)" in out
