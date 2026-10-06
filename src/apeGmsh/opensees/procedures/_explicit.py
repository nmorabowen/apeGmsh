"""Explicit-integrator procedures.

Moved verbatim from ``apesees.apeSees`` (ADR 0114 S1-b); the mixin holds no
state and no ``__init__``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._host import _ProcedureHost

# ---------------------------------------------------------------------------
# Explicit critical-time-step (dt_cr) helpers (Ladruno fork).  Pure functions
# — the math + sentinel handling, unit-testable without openseespy.
# ---------------------------------------------------------------------------

# Tiny priming step: triggers the integrator's domainChanged() so dt_cr is
# computed, without meaningfully advancing the solution (negligible vs any
# real integration duration).  Per the explicit-dynamics ADR (D5) recipe.
# NOTE: required on the deployed fork build (605affeb) — criticalTimeStep()
# returns the 0.0 NOT_COMPUTED sentinel until the first step runs.
_DTCR_PRIME_DT = 1e-12


@dataclass(frozen=True, slots=True)
class ExplicitRunResult:
    """Result of :meth:`apeSees.analyze_explicit`.

    ``n`` sub-steps of size ``dt`` tiled the requested duration; ``dt_cr``
    is the critical time step queried on the initial stiffness (``dt`` is
    ``safety * dt_cr``, optionally capped by ``dt_max``).
    """

    n: int
    dt: float
    dt_cr: float


def _dtcr_or_raise(dtcr: float) -> float:
    """Return ``dtcr`` if it is a usable (> 0) critical time step, else raise.

    ``ops.criticalTimeStep()`` sentinels: ``0.0`` not-computed (no prior
    ``analyze`` / ``domainChanged``), ``-1.0`` not-applicable or disabled
    (no ``cfl`` flag, a non-explicit integrator, or a model whose *elements*
    yield no finite estimate — e.g. a pure nodal-mass model).
    """
    if dtcr > 0:
        return dtcr
    if dtcr == 0.0:
        raise ValueError(
            "critical_time_step: dt_cr not computed yet (sentinel 0.0). "
            "An explicit integrator with cfl=True must be registered and "
            "the domain primed — this is handled internally, so seeing 0.0 "
            "means the priming step did not run."
        )
    raise ValueError(
        "critical_time_step: dt_cr not applicable (sentinel -1.0). Requires "
        "an explicit integrator (ExplicitBathe / ExplicitBatheLNVD / "
        "CentralDifferenceLadruno) with cfl=True, and element mass density "
        "(e.g. element -rho / -mass). The dt_cr eigensolve loops ELEMENTS and "
        "uses element mass + stiffness; ops.mass nodal mass is excluded (the "
        "estimate is computed on a different mass operator than the run uses), "
        "so a pure nodal-mass model yields no finite estimate."
    )


def _explicit_substep_count(
    duration: float,
    dtcr: float,
    *,
    safety: float,
    dt_max: float | None,
) -> tuple[int, float]:
    """Stable sub-step count for an explicit run of length ``duration``.

    ``dt_stable = safety * dtcr`` (capped by ``dt_max`` if given);
    ``n = max(1, ceil(duration / dt))``; returns ``(n, duration / n)`` so
    the steps tile ``duration`` exactly.  Assumes ``dtcr > 0`` (the caller
    routes through :func:`_dtcr_or_raise` first).
    """
    import math

    if duration <= 0:
        raise ValueError(
            f"analyze_explicit: duration must be > 0, got {duration}."
        )
    if not (0.0 < safety <= 1.0):
        raise ValueError(
            f"analyze_explicit: safety must be in (0, 1], got {safety}."
        )
    if dt_max is not None and dt_max <= 0:
        raise ValueError(
            f"analyze_explicit: dt_max must be > 0, got {dt_max}."
        )
    # ``safety`` scales the value criticalTimeStep() returns — NOT any
    # larger Noh-Bathe bound. Empirically 0.9 of the returned value runs
    # stable for ExplicitBathe; re-basing it on a bigger bound would erase
    # the margin.
    dt = safety * dtcr
    if dt_max is not None:
        dt = min(dt, dt_max)
    n = max(1, math.ceil(duration / dt))
    return n, duration / n


class _ExplicitMixin(_ProcedureHost):
    def critical_time_step(self) -> float:
        """Query the active explicit integrator's critical time step ``dt_cr``.

        **Fork-only** (Ladruno): builds + emits a throwaway live model
        (like :meth:`eigen`), primes one tiny step to trigger the
        integrator's ``dt_cr`` computation, then returns the usable
        (Noh-Bathe) limit.

        Requires a complete analysis chain with an **explicit**
        integrator constructed with ``cfl=True`` (e.g.
        ``ops.integrator.ExplicitBathe(cfl=True)``), a ``Transient``
        analysis, and **element mass density** (``-rho`` / ``-mass``) —
        the ``dt_cr`` eigensolve uses element mass+stiffness, not
        ``ops.mass`` nodal mass.

        Raises
        ------
        BridgeError
            If the analysis chain is incomplete.
        NotImplementedError
            If the model has registered stages (live execution of staged
            models is unsupported — emit Tcl/Py instead).
        ValueError
            If ``dt_cr`` is not usable (no ``cfl`` flag, a non-explicit
            integrator, or a pure nodal-mass model).
        """
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.critical_time_step: live execution does not "
                "support staged models "
                f"(got {len(self._stage_records)} stage(s)). Emit Tcl/Py "
                "and query criticalTimeStep() there instead."
            )
        self._check_analysis_chain_for_analyze()
        self._check_explicit_solver_compat()

        from ..emitter.live import LiveOpsEmitter

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        self._artifacts.after_emit(self)
        # Prime one negligible step so the integrator computes dt_cr.
        live_emitter.analyze(steps=1, dt=_DTCR_PRIME_DT)
        return _dtcr_or_raise(live_emitter.critical_time_step())

    def analyze_explicit(
        self,
        *,
        duration: float,
        safety: float = 0.9,
        dt_max: float | None = None,
    ) -> "ExplicitRunResult":
        """Run an explicit transient over ``duration``, auto-sized to ``dt_cr``.

        **Fork-only** (Ladruno) driver implementing the explicit-dynamics
        sub-stepping recipe (ADR D5): build + emit, prime one tiny step,
        query the critical time step, then integrate ``duration`` in
        ``n = ceil(duration / (safety * dt_cr))`` equal sub-steps via a
        single ``analyze(n, duration / n)``.

        .. warning::
           ``dt_cr`` is queried **once**, on the initial stiffness. For a
           model whose tangent *stiffens* mid-run (contact closing,
           geometric / material stiffening) the true critical step shrinks
           and a fixed ``dt`` can go supercritical and diverge. Guard such
           runs by constructing the integrator with ``cfl_abort=True`` (and
           ``recompute=N``) so a recomputed CFL violation aborts the run —
           this method then re-raises that abort as an error rather than
           returning silently. A one-shot run with an unguarded integrator
           emits :class:`OpenSeesExplicitSolverWarning`.

        Parameters
        ----------
        duration
            Total physical time to integrate (``> 0``).
        safety
            Fraction of ``dt_cr`` used as the step (``0 < safety <= 1``;
            default ``0.9``). Scales the value ``criticalTimeStep()``
            returns — do not re-base it on any larger Noh-Bathe bound.
        dt_max
            Optional upper bound on the sub-step — use a step finer than
            stability requires (e.g. for output resolution). ``> 0``.

        Returns
        -------
        ExplicitRunResult
            ``(n, dt, dt_cr)`` — the sub-step count, the step actually used,
            and the queried critical time step.

        Raises
        ------
        BridgeError / NotImplementedError / ValueError
            As for :meth:`critical_time_step`, plus ``ValueError`` for an
            out-of-range ``duration`` / ``safety`` / ``dt_max``.
        RuntimeError
            If the explicit ``analyze`` returns non-zero (divergence, or a
            mid-run ``-cflAbort`` when the integrator is guarded).
        """
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.analyze_explicit: live execution does not support "
                f"staged models (got {len(self._stage_records)} stage(s)). "
                "Emit Tcl/Py and drive the explicit run there instead."
            )
        self._check_analysis_chain_for_analyze()
        self._check_explicit_solver_compat()
        self._warn_if_unguarded_explicit_run()

        from ..emitter.live import LiveOpsEmitter

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        self._artifacts.after_emit(self)
        # Prime, query dt_cr, then size + run the sub-stepped analysis on
        # the SAME emitter (the prime step's tiny dt is stable).
        live_emitter.analyze(steps=1, dt=_DTCR_PRIME_DT)
        dtcr = _dtcr_or_raise(live_emitter.critical_time_step())
        n, dt = _explicit_substep_count(
            duration, dtcr, safety=safety, dt_max=dt_max,
        )
        ret = int(live_emitter.analyze(steps=n, dt=dt))
        if ret != 0:
            raise RuntimeError(
                f"apeSees.analyze_explicit: explicit run failed (analyze "
                f"returned {ret}) after sizing dt={dt:.3e} from "
                f"dt_cr={dtcr:.3e} over {n} sub-steps. The solution likely "
                "diverged — on a stiffening model the critical step can fall "
                "below dt mid-run. Lower safety=, pass a smaller dt_max=, or "
                "construct the integrator with cfl_abort=True / recompute=N."
            )
        return ExplicitRunResult(n=n, dt=dt, dt_cr=dtcr)
