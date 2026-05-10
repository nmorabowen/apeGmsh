"""
Typed ``integrator`` primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``integrator <Type> ...`` command. The
matching :class:`apeGmsh.opensees._internal.ns.analysis._IntegratorNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

Integrators are singletons in OpenSees (no tag in the command). The
``tag`` parameter to :meth:`_emit` is consumed by the allocator but
not rendered in the emitted command.

OpenSees command shapes::

    integrator LoadControl          dlam [num_iter [min_lam max_lam]]
    integrator DisplacementControl  node dof dU [num_iter [min_dU max_dU]]
    integrator ArcLength            s alpha
    integrator Newmark              gamma beta
    integrator HHT                  alpha [gamma beta]
    integrator CentralDifference
    integrator ExplicitDifference

The ``min_*`` / ``max_*`` step-bracket parameters on LoadControl and
DisplacementControl are only meaningful in tandem with ``num_iter``;
the dataclasses reject "min/max set but num_iter unset" at
construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import Integrator, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "LoadControl",
    "DisplacementControl",
    "ArcLength",
    "Newmark",
    "HHT",
    "CentralDifference",
    "ExplicitDifference",
]


# ---------------------------------------------------------------------------
# LoadControl — static, prescribed load-factor increment
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LoadControl(Integrator):
    """``integrator LoadControl dlam [num_iter [min_lam max_lam]]``.

    Static analysis with a fixed (or adaptively bracketed) load-factor
    increment. ``dlam`` is the nominal increment per step; ``num_iter``
    is the target convergence-iteration count used to scale the
    increment adaptively when supplied.
    """

    dlam: float
    num_iter: int | None = None
    min_lam: float | None = None
    max_lam: float | None = None

    def __post_init__(self) -> None:
        if self.num_iter is not None and self.num_iter < 1:
            raise ValueError(
                f"LoadControl: num_iter must be >= 1, got {self.num_iter}"
            )
        if (self.min_lam is None) != (self.max_lam is None):
            raise ValueError(
                "LoadControl: supply both min_lam and max_lam, or "
                f"neither (got min_lam={self.min_lam!r}, "
                f"max_lam={self.max_lam!r})."
            )
        if self.min_lam is not None and self.num_iter is None:
            raise ValueError(
                "LoadControl: min_lam/max_lam require num_iter to be set."
            )
        if (
            self.min_lam is not None
            and self.max_lam is not None
            and self.min_lam > self.max_lam
        ):
            raise ValueError(
                "LoadControl: min_lam must be <= max_lam, got "
                f"min_lam={self.min_lam}, max_lam={self.max_lam}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int] = [self.dlam]
        if self.num_iter is not None:
            args.append(self.num_iter)
            if self.min_lam is not None:
                assert self.max_lam is not None  # __post_init__ guarantee
                args += [self.min_lam, self.max_lam]
        emitter.integrator("LoadControl", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# DisplacementControl — static, prescribed displacement increment at one DOF
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class DisplacementControl(Integrator):
    """``integrator DisplacementControl node dof dU
    [num_iter [min_dU max_dU]]``.

    Static analysis driven by a prescribed displacement increment
    ``dU`` at ``node``'s ``dof``-th DOF. ``num_iter`` enables adaptive
    bracketing as in :class:`LoadControl`.
    """

    node: int
    dof: int
    dU: float
    num_iter: int | None = None
    min_dU: float | None = None
    max_dU: float | None = None

    def __post_init__(self) -> None:
        if self.dof < 1:
            raise ValueError(
                f"DisplacementControl: dof must be >= 1, got {self.dof}"
            )
        if self.num_iter is not None and self.num_iter < 1:
            raise ValueError(
                "DisplacementControl: num_iter must be >= 1, "
                f"got {self.num_iter}"
            )
        if (self.min_dU is None) != (self.max_dU is None):
            raise ValueError(
                "DisplacementControl: supply both min_dU and max_dU, or "
                f"neither (got min_dU={self.min_dU!r}, "
                f"max_dU={self.max_dU!r})."
            )
        if self.min_dU is not None and self.num_iter is None:
            raise ValueError(
                "DisplacementControl: min_dU/max_dU require num_iter "
                "to be set."
            )
        if (
            self.min_dU is not None
            and self.max_dU is not None
            and self.min_dU > self.max_dU
        ):
            raise ValueError(
                "DisplacementControl: min_dU must be <= max_dU, got "
                f"min_dU={self.min_dU}, max_dU={self.max_dU}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int] = [self.node, self.dof, self.dU]
        if self.num_iter is not None:
            args.append(self.num_iter)
            if self.min_dU is not None:
                assert self.max_dU is not None  # __post_init__ guarantee
                args += [self.min_dU, self.max_dU]
        emitter.integrator("DisplacementControl", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ArcLength — static arc-length method
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ArcLength(Integrator):
    """``integrator ArcLength s alpha``.

    Arc-length continuation. ``s`` is the arc-length increment per step;
    ``alpha`` weights the load contribution to the arc-length norm.
    """

    s: float
    alpha: float

    def __post_init__(self) -> None:
        if self.s <= 0:
            raise ValueError(
                f"ArcLength: s must be > 0, got {self.s}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.integrator("ArcLength", self.s, self.alpha)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Newmark — implicit transient (the standard structural choice)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Newmark(Integrator):
    """``integrator Newmark gamma beta``.

    The classical Newmark scheme. ``gamma=0.5, beta=0.25`` recovers
    the unconditionally stable average-acceleration variant; the user
    is responsible for selecting parameters consistent with their
    accuracy + stability requirements.
    """

    gamma: float
    beta: float

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.integrator("Newmark", self.gamma, self.beta)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# HHT — Hilber-Hughes-Taylor alpha-method
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class HHT(Integrator):
    """``integrator HHT alpha [gamma beta]``.

    Hilber-Hughes-Taylor alpha-method. Supplying ``gamma`` and
    ``beta`` overrides the OpenSees defaults derived from ``alpha``;
    omit both to use the defaults. Either supply both or neither.
    """

    alpha: float
    gamma: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        if (self.gamma is None) != (self.beta is None):
            raise ValueError(
                "HHT: supply both gamma and beta, or neither "
                f"(got gamma={self.gamma!r}, beta={self.beta!r})."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.gamma is None:
            emitter.integrator("HHT", self.alpha)
        else:
            assert self.beta is not None  # __post_init__ guarantee
            emitter.integrator("HHT", self.alpha, self.gamma, self.beta)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# CentralDifference — explicit transient (no parameters)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class CentralDifference(Integrator):
    """``integrator CentralDifference`` — explicit central-difference."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.integrator("CentralDifference")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ExplicitDifference — explicit transient (no parameters)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ExplicitDifference(Integrator):
    """``integrator ExplicitDifference`` — explicit difference scheme."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.integrator("ExplicitDifference")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
