"""
Typed ``test`` (convergence test) primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``test <Type> ...`` command. The matching
:class:`apeGmsh.opensees._internal.ns.analysis._TestNS` methods take
the same kwargs and call ``self._bridge._register(Cls(...))``.

Convergence tests are singletons in OpenSees (no tag in the command).
The ``tag`` parameter to :meth:`_emit` is consumed by the allocator
but not rendered in the emitted command.

OpenSees command shapes::

    test NormDispIncr        tol max_iter [print_flag norm_type]
    test NormUnbalance       tol max_iter [print_flag norm_type]
    test EnergyIncr          tol max_iter [print_flag norm_type]
    test FixedNumIter        max_iter    [print_flag norm_type]
    test RelativeNormDispIncr tol max_iter [print_flag norm_type]

Note: The file is named ``test.py`` to mirror the OpenSees command
spelling (``ops.test.NormDispIncr``). Pytest's collection only matches
``test_*.py``, so the name does not conflict with test discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import ConvergenceTest, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "NormDispIncr",
    "NormUnbalance",
    "EnergyIncr",
    "FixedNumIter",
    "RelativeNormDispIncr",
    "LadrunoStabilizedUnbalance",
]


# ---------------------------------------------------------------------------
# NormDispIncr — converge on the displacement-increment norm
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class NormDispIncr(ConvergenceTest):
    """``test NormDispIncr tol maxIter [pFlag normType]``."""

    tol: float
    max_iter: int
    print_flag: int = 0
    norm_type: int = 2

    def __post_init__(self) -> None:
        if self.tol <= 0:
            raise ValueError(
                f"NormDispIncr: tol must be > 0, got {self.tol}"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"NormDispIncr: max_iter must be >= 1, got {self.max_iter}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.test(
            "NormDispIncr",
            self.tol, self.max_iter, self.print_flag, self.norm_type,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# NormUnbalance — converge on the residual (out-of-balance) force norm
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class NormUnbalance(ConvergenceTest):
    """``test NormUnbalance tol maxIter [pFlag normType]``."""

    tol: float
    max_iter: int
    print_flag: int = 0
    norm_type: int = 2

    def __post_init__(self) -> None:
        if self.tol <= 0:
            raise ValueError(
                f"NormUnbalance: tol must be > 0, got {self.tol}"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"NormUnbalance: max_iter must be >= 1, got {self.max_iter}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.test(
            "NormUnbalance",
            self.tol, self.max_iter, self.print_flag, self.norm_type,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# EnergyIncr — converge on incremental energy (du * dF) norm
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class EnergyIncr(ConvergenceTest):
    """``test EnergyIncr tol maxIter [pFlag normType]``."""

    tol: float
    max_iter: int
    print_flag: int = 0
    norm_type: int = 2

    def __post_init__(self) -> None:
        if self.tol <= 0:
            raise ValueError(
                f"EnergyIncr: tol must be > 0, got {self.tol}"
            )
        if self.max_iter < 1:
            raise ValueError(
                f"EnergyIncr: max_iter must be >= 1, got {self.max_iter}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.test(
            "EnergyIncr",
            self.tol, self.max_iter, self.print_flag, self.norm_type,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# FixedNumIter — always run a fixed number of iterations (no tolerance)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class FixedNumIter(ConvergenceTest):
    """``test FixedNumIter maxIter [pFlag normType]``.

    Runs ``max_iter`` Newton iterations every step regardless of the
    residual. Useful for explicit-style schemes and pseudo-static
    runs where the algorithm relies on a fixed number of corrections.
    """

    max_iter: int
    print_flag: int = 0
    norm_type: int = 2

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError(
                f"FixedNumIter: max_iter must be >= 1, got {self.max_iter}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.test(
            "FixedNumIter",
            self.max_iter, self.print_flag, self.norm_type,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# RelativeNormDispIncr — relative variant of NormDispIncr
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class RelativeNormDispIncr(ConvergenceTest):
    """``test RelativeNormDispIncr tol maxIter [pFlag normType]``.

    Converges when ``||du_k|| / ||du_0||`` falls below ``tol`` —
    relative to the first-iteration increment of the step.
    """

    tol: float
    max_iter: int
    print_flag: int = 0
    norm_type: int = 2

    def __post_init__(self) -> None:
        if self.tol <= 0:
            raise ValueError(
                f"RelativeNormDispIncr: tol must be > 0, got {self.tol}"
            )
        if self.max_iter < 1:
            raise ValueError(
                "RelativeNormDispIncr: max_iter must be >= 1, "
                f"got {self.max_iter}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.test(
            "RelativeNormDispIncr",
            self.tol, self.max_iter, self.print_flag, self.norm_type,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoStabilizedUnbalance — true-residual unbalance test (Ladruno fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoStabilizedUnbalance(ConvergenceTest):
    """``test LadrunoStabilizedUnbalance tol maxIter [pFlag normType]`` — **fork-only**.

    The *Ladruno fork*'s unbalance test (``CONVERGENCE_TEST_TAG`` 33000).
    Like stock :class:`NormUnbalance` it converges on the residual norm, but
    when the active integrator is a stabilizing ``LadrunoArcLength
    -stabilize`` it norms the **true static residual** rather than the SOE
    ``B`` vector (which the stabilization perturbs); otherwise it falls back
    to ``||B||``, identical to ``NormUnbalance``. Emission works on any
    build; the fork is required only to *run*.
    """

    tol: float
    max_iter: int
    print_flag: int = 0
    norm_type: int = 2

    def __post_init__(self) -> None:
        if self.tol <= 0:
            raise ValueError(
                f"LadrunoStabilizedUnbalance: tol must be > 0, got {self.tol}"
            )
        if self.max_iter < 1:
            raise ValueError(
                "LadrunoStabilizedUnbalance: max_iter must be >= 1, "
                f"got {self.max_iter}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.test(
            "LadrunoStabilizedUnbalance",
            self.tol, self.max_iter, self.print_flag, self.norm_type,
        )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
