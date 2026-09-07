"""
Typed solution-strategy primitives — ADR 0057 Phase A + Phase B.

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

Phase B (ADR 0104) adds the :class:`Substep` rung — the adaptive
step-size controller ported from the TIMs response-curve harness:
base step, cap, regrow after consecutive successes, halving on
failure, a floor, a subdivision budget, a wall-clock budget, and
run-to-criterion termination (target displacement, or a tail-slope
plateau).  It runs against a :class:`SubstepDriver` via
:meth:`Substep.drive`; deck emission of the substep loop is deferred
(ADR 0104 D6), so :meth:`Ladder.to_spec` REFUSES a ladder that
carries one rather than silently emitting a fixed-step deck.

Hard exclusions (ADR 0057 §6): algorithm rungs are solution
algorithms ONLY — no tolerance relaxation, no test swaps, no
integrator identity changes beyond the ``Substep`` step scaling.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

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
    "Substep",
    "SubstepDriver",
    "SubstepResult",
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


# ---------------------------------------------------------------------------
# Substep rung — the adaptive step controller (ADR 0057 Phase B / ADR 0104)
# ---------------------------------------------------------------------------

Verdict = Literal["target", "plateau", "budget", "floor", "wall"]

#: Verdicts that mean the run got what it was asked for.  Every other
#: verdict is a FAILURE — a spent budget is never dressed up as
#: success (ADR 0104 D4; the #587 fail-loud floor).
_SUCCESS_VERDICTS: frozenset[str] = frozenset({"target", "plateau"})

#: Minimum committed rows before the plateau criterion may fire.  Two
#: three-point least-squares fits (initial tangent, tail tangent) read
#: off fewer rows than this are noise, not a trend — ported from the
#: TIMs harness ``_rolling_k`` / ``_verdict`` guard.
_MIN_PLATEAU_ROWS = 8


class SubstepDriver(Protocol):
    """What :meth:`Substep.drive` needs from the analysis it is driving.

    Deliberately three methods: the controller owns the *step-size
    policy* and nothing else, so the driver keeps every OpenSees
    idiom that varies between pushes — which integrator carries the
    increment (``DisplacementControl`` vs an ``sp`` ramp under
    ``LoadControl``), and whether a failed ``analyze`` first walks an
    algorithm ladder (ADR 0057 Phase A) before reporting failure.
    """

    def analyze(self, ds: float) -> int:
        """Attempt ONE increment of size ``ds``; 0 == converged.

        Mirrors the ``ops.analyze`` return convention, so a driver
        that sets its integrator and forwards the call is a two-liner.
        """
        ...

    def disp(self, node: int, dof: int) -> float:
        """Displacement at ``node`` / ``dof`` (``ops.nodeDisp``)."""
        ...

    def load(self) -> float:
        """The load conjugate to the control displacement.

        Under ``DisplacementControl`` against a unit reference load
        this is the load factor (``ops.getTime()``) and IS the push
        force; the plateau criterion reads its slope.
        """
        ...


@dataclass(frozen=True, slots=True)
class SubstepResult:
    """Outcome of one :meth:`Substep.drive`.

    ``verdict`` is the whole contract; :attr:`ok` is derived from it
    and never from "we took some steps".
    """

    verdict: Verdict
    reason: str
    steps: int
    subdivisions: int
    disp: float
    ds_final: float
    wall: float
    curve: tuple[tuple[float, float], ...] = field(default=())

    @property
    def ok(self) -> bool:
        """True only for a run-to-criterion verdict (ADR 0104 D4)."""
        return self.verdict in _SUCCESS_VERDICTS


def _slope(points: "Sequence[tuple[float, float]]") -> float:
    """Least-squares d(load)/d(disp) over ``points``; NaN under 3 points."""
    n = len(points)
    if n < 3:
        return math.nan
    su = sum(u for u, _ in points)
    sq = sum(q for _, q in points)
    suu = sum(u * u for u, _ in points)
    suq = sum(u * q for u, q in points)
    den = n * suu - su * su
    if den == 0.0:
        return math.nan
    return (n * suq - su * sq) / den


@dataclass(frozen=True, kw_only=True, slots=True)
class Substep:
    """Adaptive step size + run-to-criterion termination (ADR 0104).

    Declares the *policy*; :meth:`drive` executes it against a
    :class:`SubstepDriver`.  Step sizes and ``target`` are magnitudes
    in the control DOF's own units — the datum is the displacement
    read at entry, so a push that starts from a settled gravity state
    measures its own advance, not the absolute nodal value.

    ``ds_max`` defaults to ``ds``: ADR 0057 §2 says the step never
    regrows above the nominal, so growth only ever *recovers* ground
    lost to halving unless a cap is declared explicitly.

    ``budget`` is the number of **consecutive** halvings allowed to
    rescue ONE increment (ADR 0057 §2's ``max_halvings``), not a
    run-wide total: the regrow probe re-fails by construction every
    ``regrow_after`` good steps, so a run-wide total is spent by
    *probing* rather than by the model failing — measured, and the
    reason for the divergence from the TIMs harness (ADR 0104 F1).
    :attr:`SubstepResult.subdivisions` still reports the run total.
    """

    node: int
    dof: int
    target: float
    ds: float
    ds_min: float
    ds_max: float | None = None
    regrow: float = 2.0
    regrow_after: int = 2
    budget: int = 8
    wall_budget: float | None = None
    plateau: float | None = None
    plateau_window: float | None = None

    def __post_init__(self) -> None:
        if self.dof < 1:
            raise ValueError(
                f"Substep: dof is 1-based (OpenSees), got {self.dof}."
            )
        for name in ("target", "ds", "ds_min"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(
                    f"Substep: {name} must be > 0, got "
                    f"{getattr(self, name)!r} — every step size and the "
                    "target are MAGNITUDES of advance in the control DOF."
                )
        if self.ds_min > self.ds:
            raise ValueError(
                f"Substep: ds_min ({self.ds_min!r}) is above the base step "
                f"ds ({self.ds!r}) — the floor would refuse the very first "
                "halving, so the controller could never subdivide."
            )
        if self.ds_max is None:
            object.__setattr__(self, "ds_max", float(self.ds))
        elif self.ds_max < self.ds:
            raise ValueError(
                f"Substep: ds_max ({self.ds_max!r}) is below the base step "
                f"ds ({self.ds!r}) — a cap under the nominal step is a "
                "smaller base step, declared confusingly."
            )
        if self.regrow <= 1.0:
            raise ValueError(
                f"Substep: regrow must be > 1, got {self.regrow!r} "
                "(it is the growth factor, not a fraction)."
            )
        if self.regrow_after < 1:
            raise ValueError(
                f"Substep: regrow_after must be >= 1, got "
                f"{self.regrow_after!r}."
            )
        if self.budget < 0:
            raise ValueError(
                f"Substep: budget must be >= 0, got {self.budget!r} "
                "(0 means no halving is allowed at all — the first "
                "failed increment ends the run)."
            )
        if self.wall_budget is not None and self.wall_budget <= 0.0:
            raise ValueError(
                f"Substep: wall_budget must be > 0 seconds, got "
                f"{self.wall_budget!r} — a zero budget is a run that "
                "cannot start, which is a mistake and not a policy."
            )
        if (self.plateau is None) != (self.plateau_window is None):
            raise ValueError(
                "Substep: plateau and plateau_window are declared "
                "together — an epsilon with no window has no tail to "
                "fit, and a window with no epsilon has no threshold."
            )
        if self.plateau is not None and not 0.0 < self.plateau < 1.0:
            raise ValueError(
                f"Substep: plateau must be a fraction in (0, 1), got "
                f"{self.plateau!r} — it is the share of the INITIAL "
                "tangent the tail slope must fall below."
            )
        if self.plateau_window is not None and self.plateau_window <= 0.0:
            raise ValueError(
                f"Substep: plateau_window must be > 0, got "
                f"{self.plateau_window!r}."
            )

    # -- the criterion ---------------------------------------------------

    def _plateau_reached(
        self, curve: "Sequence[tuple[float, float]]",
    ) -> bool:
        """Tail slope has fallen to ``plateau`` × the initial tangent.

        Ported from the TIMs harness ``_verdict``: both tangents are
        least-squares fits, the window must be **covered** by the
        curve's own range (a run shorter than the window cannot show a
        sustained tangent either way), and a non-positive or NaN
        initial tangent disables the criterion rather than dividing by
        it.
        """
        if self.plateau is None or self.plateau_window is None:
            return False
        n = len(curve)
        if n < _MIN_PLATEAU_ROWS:
            return False
        u_end = curve[-1][0]
        if u_end - curve[0][0] < self.plateau_window:
            return False
        k0 = _slope(curve[: max(4, n // 50)])
        if math.isnan(k0) or k0 <= 0.0:
            return False
        lo = u_end - self.plateau_window
        k_tail = _slope([p for p in curve if p[0] >= lo])
        if math.isnan(k_tail):
            return False
        return k_tail <= self.plateau * k0

    # -- the controller --------------------------------------------------

    def drive(
        self,
        driver: SubstepDriver,
        *,
        clock: "Callable[[], float]" = time.monotonic,
    ) -> SubstepResult:
        """Run to criterion, adapting the step size; never silently.

        Termination is checked at the TOP of every attempt, success
        criteria first: a run that meets its target or plateaus on the
        last permitted second is a success, and a budget can only be
        *spent* by a run that had not yet got what it came for.
        """
        ds = float(self.ds)
        ds_max = float(self.ds_max if self.ds_max is not None else self.ds)
        t0 = clock()
        u0 = float(driver.disp(self.node, self.dof))
        tol = 1e-12 * max(abs(self.target), 1.0)
        curve: list[tuple[float, float]] = []
        steps = good = nsub = depth = 0
        u = 0.0
        verdict: Verdict
        while True:
            u = abs(float(driver.disp(self.node, self.dof)) - u0)
            if not math.isfinite(u):
                # A NaN control reading fails EVERY comparison below —
                # the target test, the floor test and the step-1
                # tripwire alike — so the loop would spin forever on a
                # model that has already blown up.  Measured (ADR 0104
                # F2); it is the commonest way an OpenSees solve dies.
                raise ValueError(
                    f"Substep: node {self.node} dof {self.dof} read back "
                    f"a non-finite displacement ({u!r}) — the model "
                    "state is not a number and no step size can rescue "
                    "it. Aborting rather than stepping into it."
                )
            if u >= self.target - tol:
                verdict, reason = "target", (
                    f"reached the target displacement {self.target:g} "
                    f"at node {self.node} dof {self.dof}"
                )
                break
            if self._plateau_reached(curve):
                verdict, reason = "plateau", (
                    f"tail slope fell to {self.plateau:g} of the initial "
                    f"tangent over a window of {self.plateau_window:g} "
                    f"at u = {u:g}"
                )
                break
            if (
                self.wall_budget is not None
                and clock() - t0 > self.wall_budget
            ):
                verdict, reason = "wall", (
                    f"wall-clock budget of {self.wall_budget:g} s spent "
                    f"at u = {u:g}"
                )
                break
            # Land exactly on the target: the last step is whatever is
            # left, never an overshoot (ADR 0057 §2, the loadConst
            # contract — unloading a yielded model to correct an
            # overshoot corrupts the plastic state).
            ds_use = min(ds, self.target - u)
            if driver.analyze(ds_use) != 0:
                nsub += 1
                depth += 1
                good = 0
                ds *= 0.5
                print(
                    f"apeGmsh substep: subdivision {depth}/{self.budget} "
                    f"(run total {nsub}) at u = {u:.6g} -> ds = {ds:.6g}"
                )
                if depth > self.budget:
                    verdict, reason = "budget", (
                        f"subdivision budget of {self.budget} spent on one "
                        f"increment at u = {u:g}"
                    )
                    break
                if ds < self.ds_min:
                    verdict, reason = "floor", (
                        f"step size fell below the floor {self.ds_min:g} "
                        f"at u = {u:g}"
                    )
                    break
                continue
            steps += 1
            depth = 0
            u_new = abs(float(driver.disp(self.node, self.dof)) - u0)
            if steps == 1 and u_new <= u:
                # The step-1 fidelity tripwire, ported from the TIMs
                # harness: a converged first increment that moved the
                # declared control DOF by nothing means the DOF being
                # driven is not the one being measured.  Refuse here,
                # not after an hour of steps that cannot terminate.
                raise ValueError(
                    f"Substep: the first converged increment did not "
                    f"advance node {self.node} dof {self.dof} (still "
                    f"{u_new:g} from the datum).  The control DOF the "
                    f"driver steps and the one this Substep measures "
                    f"are not the same DOF."
                )
            q = float(driver.load())
            if self.plateau is not None and not math.isfinite(q):
                # A NaN load silently DISABLES the plateau criterion
                # (the initial tangent fits to NaN), so a declared
                # criterion would never fire and the run would only
                # ever stop on the target or the wall (ADR 0104 F3).
                raise ValueError(
                    f"Substep: the driver's load reading is non-finite "
                    f"({q!r}) and a plateau criterion is declared — the "
                    "tail-slope test would silently never fire."
                )
            curve.append((u_new, q))
            good += 1
            if good >= self.regrow_after and ds < ds_max:
                ds, good = min(self.regrow * ds, ds_max), 0
        wall = clock() - t0
        result = SubstepResult(
            verdict=verdict, reason=reason, steps=steps,
            subdivisions=nsub, disp=u, ds_final=ds, wall=wall,
            curve=tuple(curve),
        )
        print(
            f"apeGmsh substep: {verdict.upper()} after {steps} steps, "
            f"{nsub} subdivisions, {wall:.3g} s -- {reason}"
        )
        return result


_MAX_RUNGS = 8


@dataclass(frozen=True, kw_only=True, slots=True)
class Ladder:
    """An ordered escalation of solution algorithms (ADR 0057 §1–§2).

    ``rungs`` lists the *escalation* algorithms in order; the analysis
    chain's own algorithm is implicitly rung 0 (listing it first is
    optional sugar — :meth:`to_spec` deduplicates).  ``name`` labels
    the provenance prints and the exhaustion banner.

    At most ONE rung may be a :class:`Substep` (ADR 0057 §1) — the
    step-size policy of a loop is singular.  It is reachable as
    :attr:`substep` and is driven by :meth:`Substep.drive`; it is not
    an ``algorithm`` command, so it never appears in the emitted spec.

    Pass to ``s.run(..., strategy=ladder)`` (staged) or
    ``apeSees.analyze(..., strategy=ladder)`` (flat live runs).
    """

    rungs: tuple[SolutionAlgorithm | Substep, ...]
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
            if not isinstance(r, (SolutionAlgorithm, Substep)):
                raise TypeError(
                    "Ladder rungs must be solution-algorithm primitives "
                    f"(ops.algorithm.*) or one ops.strategy.Substep; got "
                    f"{type(r).__name__!r}. Tolerance/test/integrator "
                    "changes are excluded by design (ADR 0057 §6)."
                )
        if sum(isinstance(r, Substep) for r in self.rungs) > 1:
            raise ValueError(
                "Ladder: at most one Substep rung — a loop has one "
                "step-size policy, and two would each undo the other's "
                "halving."
            )
        # Normalize: tuple() guards against list inputs under frozen.
        object.__setattr__(self, "rungs", tuple(self.rungs))

    @property
    def substep(self) -> Substep | None:
        """The ladder's :class:`Substep` rung, if it declared one."""
        for r in self.rungs:
            if isinstance(r, Substep):
                return r
        return None

    def to_spec(self, *, base: SolutionAlgorithm | None) -> StrategySpec:
        """Resolve to the emitter-ready :class:`StrategySpec`.

        ``base`` is the analysis chain's algorithm — it becomes rung 0
        so the emitted loop can restore it after a rescued increment.
        If the ladder's first rung already equals the base (same
        emitted args), it is not duplicated.  ``base=None`` (flat runs
        that never declared a chain algorithm through the bridge)
        promotes the ladder's first rung to rung 0.

        A ladder carrying a :class:`Substep` is **refused**: the
        substep loop is not emitted into decks yet (ADR 0104 D6), and
        resolving it to the fixed-step Phase A spec would emit a deck
        that quietly ignores the declared step policy.
        """
        if self.substep is not None:
            raise ValueError(
                f"Ladder {self.name!r} carries a Substep rung, which is "
                "not emitted into decks yet (ADR 0104 D6). Emitting the "
                "algorithm rungs alone would silently drop the declared "
                "step-size policy and run a fixed-step deck. Drive it "
                "in-process with Substep.drive(driver) instead."
            )
        rung_args = [
            _rung_args(r) for r in self.rungs
            if isinstance(r, SolutionAlgorithm)
        ]
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
