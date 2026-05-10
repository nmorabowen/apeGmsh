"""
Typed ``algorithm`` (solution algorithm) primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``algorithm <Type> ...`` command. The
matching :class:`apeGmsh.opensees._internal.ns.analysis._AlgorithmNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

Solution algorithms are singletons in OpenSees (no tag in the
command). The ``tag`` parameter to :meth:`_emit` is consumed by the
allocator but not rendered in the emitted command.

OpenSees command shapes::

    algorithm Linear
    algorithm Newton             [-secant | -initial]
    algorithm ModifiedNewton     [-secant | -initial]
    algorithm NewtonLineSearch -type type [-tol tol] [-maxIter n]
                              [-minEta v] [-maxEta v]
    algorithm KrylovNewton       [-iterate t] [-increment t] [-maxDim n]
    algorithm BFGS               [count]
    algorithm Broyden            [count]

The ``-secant`` / ``-initial`` flags select which tangent the Newton
family uses. They are mutually exclusive (no tangent and one tangent
flag are both valid; two tangent flags are not). The dataclasses
expose them as a single ``tangent`` enum-string field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .._internal.types import Primitive, SolutionAlgorithm

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Linear",
    "Newton",
    "ModifiedNewton",
    "NewtonLineSearch",
    "KrylovNewton",
    "BFGS",
    "Broyden",
]


# -- Shared types ------------------------------------------------------------

NewtonTangent = Literal["tangent", "secant", "initial"]
"""Tangent-stiffness flavor for Newton / ModifiedNewton.

* ``"tangent"`` — emit no flag (OpenSees default: current tangent).
* ``"secant"`` — emit ``-secant``.
* ``"initial"`` — emit ``-initial``.
"""

LineSearchType = Literal["Bisection", "Secant", "RegulaFalsi", "InitialInterpolated"]
"""Line-search algorithm selector for ``NewtonLineSearch``."""


# ---------------------------------------------------------------------------
# Linear — single iteration per step (linear analyses)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Linear(SolutionAlgorithm):
    """``algorithm Linear`` — one solve per step (no iteration)."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.algorithm("Linear")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Newton — full Newton-Raphson
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Newton(SolutionAlgorithm):
    """``algorithm Newton [-secant | -initial]`` — full Newton-Raphson.

    ``tangent`` selects the stiffness flavor: the OpenSees default
    ("current tangent") if ``"tangent"``, ``-secant`` for the secant
    stiffness, ``-initial`` for the initial-tangent (sometimes called
    initial-stiffness Newton).
    """

    tangent: NewtonTangent = "tangent"

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.tangent == "secant":
            emitter.algorithm("Newton", "-secant")
        elif self.tangent == "initial":
            emitter.algorithm("Newton", "-initial")
        else:
            emitter.algorithm("Newton")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ModifiedNewton — re-uses one tangent across a step
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ModifiedNewton(SolutionAlgorithm):
    """``algorithm ModifiedNewton [-secant | -initial]``.

    Forms one tangent at the start of each step and re-uses it for
    every iteration in that step. ``tangent`` semantics match
    :class:`Newton`.
    """

    tangent: NewtonTangent = "tangent"

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.tangent == "secant":
            emitter.algorithm("ModifiedNewton", "-secant")
        elif self.tangent == "initial":
            emitter.algorithm("ModifiedNewton", "-initial")
        else:
            emitter.algorithm("ModifiedNewton")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# NewtonLineSearch — Newton with a one-dimensional line search per iteration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class NewtonLineSearch(SolutionAlgorithm):
    """``algorithm NewtonLineSearch -type T [-tol t] [-maxIter n]
    [-minEta v] [-maxEta v]``.

    ``line_search`` (required) picks the 1-D search algorithm. The
    other keys are optional and emit only when set.
    """

    line_search: LineSearchType
    tol: float | None = None
    max_iter: int | None = None
    min_eta: float | None = None
    max_eta: float | None = None

    def __post_init__(self) -> None:
        if self.tol is not None and self.tol <= 0:
            raise ValueError(
                f"NewtonLineSearch: tol must be > 0, got {self.tol}"
            )
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError(
                "NewtonLineSearch: max_iter must be >= 1, "
                f"got {self.max_iter}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = ["-type", self.line_search]
        if self.tol is not None:
            args += ["-tol", self.tol]
        if self.max_iter is not None:
            args += ["-maxIter", self.max_iter]
        if self.min_eta is not None:
            args += ["-minEta", self.min_eta]
        if self.max_eta is not None:
            args += ["-maxEta", self.max_eta]
        emitter.algorithm("NewtonLineSearch", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# KrylovNewton — Newton with a Krylov-subspace acceleration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class KrylovNewton(SolutionAlgorithm):
    """``algorithm KrylovNewton [-iterate t] [-increment t] [-maxDim n]``.

    The two ``-iterate`` / ``-increment`` flags pick which tangent is
    kept across a step (``"current"``, ``"initial"``, or ``"noTangent"``).
    ``max_dim`` caps the size of the Krylov subspace.
    """

    iterate: Literal["current", "initial", "noTangent"] | None = None
    increment: Literal["current", "initial", "noTangent"] | None = None
    max_dim: int | None = None

    def __post_init__(self) -> None:
        if self.max_dim is not None and self.max_dim < 1:
            raise ValueError(
                f"KrylovNewton: max_dim must be >= 1, got {self.max_dim}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = []
        if self.iterate is not None:
            args += ["-iterate", self.iterate]
        if self.increment is not None:
            args += ["-increment", self.increment]
        if self.max_dim is not None:
            args += ["-maxDim", self.max_dim]
        emitter.algorithm("KrylovNewton", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# BFGS — quasi-Newton with BFGS rank-2 update
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class BFGS(SolutionAlgorithm):
    """``algorithm BFGS [count]`` — quasi-Newton with BFGS rank-2 update.

    ``count`` caps the number of stored rank-2 updates per step.
    """

    count: int | None = None

    def __post_init__(self) -> None:
        if self.count is not None and self.count < 1:
            raise ValueError(
                f"BFGS: count must be >= 1, got {self.count}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.count is None:
            emitter.algorithm("BFGS")
        else:
            emitter.algorithm("BFGS", self.count)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Broyden — quasi-Newton with Broyden rank-1 update
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Broyden(SolutionAlgorithm):
    """``algorithm Broyden [count]`` — quasi-Newton with rank-1 update.

    ``count`` caps the number of stored rank-1 updates per step.
    """

    count: int | None = None

    def __post_init__(self) -> None:
        if self.count is not None and self.count < 1:
            raise ValueError(
                f"Broyden: count must be >= 1, got {self.count}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.count is None:
            emitter.algorithm("Broyden")
        else:
            emitter.algorithm("Broyden", self.count)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
