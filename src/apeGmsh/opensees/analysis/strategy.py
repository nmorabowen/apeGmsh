"""
Typed solution-strategy primitives — ADR 0057 Phase A.

A :class:`Ladder` declares an ordered escalation of solution
algorithms for an analyze loop: rung 0 (the analysis chain's own
algorithm, implicit) gets first shot at every increment; on a failed
``analyze 1`` the emitted loop walks to the next rung (re-issuing its
``algorithm`` command) and retries the *same* increment; a rescued
increment restores rung 0 so the fast chain leads again; exhausting
the ladder aborts with the #587 fail-loud banner plus the ladder name.

:func:`profile` returns the **established profiles** — named,
documented, evidence-revisable ladder presets (ADR 0057 §3).  Profile
orderings are grounded in the 2026-06-10 zoned-twin campaign, where
``NewtonLineSearch`` stalled in five independent mesh/element
configurations that plain ``Newton`` carried at identical tolerance:
on non-smooth multi-surface plasticity the line search oscillates
across the yield-surface kink, so ``"non-smooth"`` deliberately has
**no line-search rung**.

Hard exclusions (ADR 0057 §6): rungs are solution algorithms ONLY —
no tolerance relaxation, no test swaps, no integrator changes.
``Substep`` rungs are Phase B.
"""
from __future__ import annotations

from dataclasses import dataclass

from .._internal.types import SolutionAlgorithm
from ..emitter.base import StrategySpec
from .algorithm import (
    BFGS,
    KrylovNewton,
    ModifiedNewton,
    Newton,
    NewtonLineSearch,
)

__all__ = [
    "Ladder",
    "profile",
    "PROFILE_NAMES",
]


class _AlgorithmCapture:
    """Minimal emitter shim: records the ``algorithm`` command args.

    Algorithm primitives know their OpenSees argument shape only
    through ``_emit(emitter, tag)`` — capturing through the same path
    keeps the ladder's rung args byte-identical to what the chain
    would emit, with zero duplicated per-class argument logic.
    """

    def __init__(self) -> None:
        self.captured: tuple[int | float | str, ...] | None = None

    def algorithm(self, a_type: str, *args: int | float | str) -> None:
        self.captured = (a_type, *args)


def _rung_args(alg: SolutionAlgorithm) -> tuple[int | float | str, ...]:
    """Return the ``(a_type, *args)`` tuple ``alg`` would emit."""
    shim = _AlgorithmCapture()
    alg._emit(shim, 0)  # type: ignore[arg-type]  # shim implements the one method _emit uses
    if shim.captured is None:  # pragma: no cover - defensive
        raise ValueError(
            f"{type(alg).__name__}._emit did not call emitter.algorithm —"
            " not a solution-algorithm primitive?"
        )
    return shim.captured


_MAX_RUNGS = 8


@dataclass(frozen=True, kw_only=True, slots=True)
class Ladder:
    """An ordered escalation of solution algorithms (ADR 0057 §1–§2).

    ``rungs`` lists the *escalation* algorithms in order; the analysis
    chain's own algorithm is implicitly rung 0 (listing it first is
    optional sugar — :meth:`to_spec` deduplicates).  ``name`` labels
    the provenance prints and the exhaustion banner.

    Pass to ``s.run(..., strategy=ladder)`` (staged) or
    ``apeSees.analyze(..., strategy=ladder)`` (flat live runs).
    """

    rungs: tuple[SolutionAlgorithm, ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("Ladder: rungs must not be empty.")
        if len(self.rungs) > _MAX_RUNGS:
            raise ValueError(
                f"Ladder: at most {_MAX_RUNGS} rungs, got "
                f"{len(self.rungs)} — an escalation this deep is a "
                "modeling problem, not a solver problem."
            )
        for r in self.rungs:
            if not isinstance(r, SolutionAlgorithm):
                raise TypeError(
                    "Ladder rungs must be solution-algorithm primitives "
                    f"(ops.algorithm.*); got {type(r).__name__!r}. "
                    "Tolerance/test/integrator changes are excluded by "
                    "design (ADR 0057 §6)."
                )
        # Normalize: tuple() guards against list inputs under frozen.
        object.__setattr__(self, "rungs", tuple(self.rungs))

    def to_spec(self, *, base: SolutionAlgorithm | None) -> StrategySpec:
        """Resolve to the emitter-ready :class:`StrategySpec`.

        ``base`` is the analysis chain's algorithm — it becomes rung 0
        so the emitted loop can restore it after a rescued increment.
        If the ladder's first rung already equals the base (same
        emitted args), it is not duplicated.  ``base=None`` (flat runs
        that never declared a chain algorithm through the bridge)
        promotes the ladder's first rung to rung 0.
        """
        rung_args = [_rung_args(r) for r in self.rungs]
        if base is not None:
            base_args = _rung_args(base)
            if not rung_args or rung_args[0] != base_args:
                rung_args.insert(0, base_args)
        return StrategySpec(name=self.name, rungs=tuple(rung_args))


# ---------------------------------------------------------------------------
# Established profiles (ADR 0057 §3)
# ---------------------------------------------------------------------------

def _build_profile(name: str) -> Ladder:
    if name == "standard":
        # General default: the initial tangent survives states that
        # poison the current tangent; the line search goes last.
        return Ladder(name="standard", rungs=(
            ModifiedNewton(tangent="initial"),
            NewtonLineSearch(line_search="Bisection"),
        ))
    if name == "non-smooth":
        # 2026-06-10 zoned-twin evidence: NewtonLineSearch oscillates
        # across MC/DP yield-surface kinks and IS the failure mode —
        # five independent configs stalled under it while plain Newton
        # at identical tolerance carried all of them.  NO line-search
        # rung, by design.
        return Ladder(name="non-smooth", rungs=(
            ModifiedNewton(tangent="initial"),
            KrylovNewton(),
        ))
    if name == "smooth-hardening":
        # Smooth J2-type response is where the line search genuinely
        # helps first.
        return Ladder(name="smooth-hardening", rungs=(
            NewtonLineSearch(line_search="Bisection"),
            KrylovNewton(),
        ))
    if name == "penalty-stiff":
        # Embed/contact penalties (K ~ 1e8+) poison the current
        # tangent; the initial tangent is the classic remedy.
        return Ladder(name="penalty-stiff", rungs=(
            ModifiedNewton(tangent="initial"),
            KrylovNewton(),
            NewtonLineSearch(line_search="Bisection"),
        ))
    if name == "exhaustive":
        # Last resort: "get me A converged state to debug from".
        return Ladder(name="exhaustive", rungs=(
            ModifiedNewton(tangent="initial"),
            NewtonLineSearch(line_search="Bisection"),
            KrylovNewton(),
            BFGS(),
            Newton(tangent="secant"),
        ))
    raise KeyError(name)


_ALIASES = {
    "geotech": "non-smooth",
    "mohr-coulomb": "non-smooth",
    "metal": "smooth-hardening",
}

PROFILE_NAMES: tuple[str, ...] = (
    "standard", "non-smooth", "smooth-hardening", "penalty-stiff",
    "exhaustive",
)
"""Canonical established-profile names (aliases: ``geotech`` /
``mohr-coulomb`` → ``non-smooth``; ``metal`` → ``smooth-hardening``)."""


def profile(name: str) -> Ladder:
    """Return an established profile :class:`Ladder` by name.

    The returned ladder is a plain value — extend or trim it with
    ``Ladder(rungs=profile("standard").rungs + (...,), name="mine")``.
    See the module docstring and ADR 0057 §3 for the per-profile
    rationale; profile names are a stable contract, orderings are
    evidence-revisable.
    """
    canonical = _ALIASES.get(name, name)
    try:
        return _build_profile(canonical)
    except KeyError:
        raise ValueError(
            f"unknown strategy profile {name!r}. Established profiles: "
            f"{', '.join(PROFILE_NAMES)} (aliases: "
            f"{', '.join(sorted(_ALIASES))})."
        ) from None
