"""
Typed ``constraints`` primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``constraints <Type> ...`` command. The
matching :class:`apeGmsh.opensees._internal.ns.analysis._ConstraintsNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

Constraint handlers are singletons in OpenSees (no tag in the command
syntax). The ``tag`` parameter to :meth:`_emit` is consumed by the
allocator but not rendered in the emitted command.

OpenSees command shapes::

    constraints Plain
    constraints Penalty alphaSP alphaMP
    constraints Transformation
    constraints Lagrange [alphaSP alphaMP]
    constraints Auto <-verbose> <-autoPenalty $oom> <-userPenalty $val>

See ``architecture/api-design.md`` for the namespace surface and
``architecture/emitter.md`` for the underlying ``constraints(c_type,
*args: int | float | str)`` Protocol method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import ConstraintHandler, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Plain",
    "Penalty",
    "Transformation",
    "Lagrange",
    "Auto",
    "LadrunoProjection",
]


# ---------------------------------------------------------------------------
# Plain — direct application of homogeneous SPs (default)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Plain(ConstraintHandler):
    """``constraints Plain`` — direct application, homogeneous SPs only."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag  # constraint handlers are singletons; no tag in the command
        emitter.constraints("Plain")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Penalty — penalty method with user-chosen weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Penalty(ConstraintHandler):
    """``constraints Penalty alphaSP alphaMP`` — penalty method.

    Both alphas are required: there is no sensible OpenSees default —
    suitable values depend on the model's stiffness scale.
    """

    alpha_sp: float
    alpha_mp: float

    def __post_init__(self) -> None:
        if self.alpha_sp <= 0:
            raise ValueError(
                f"Penalty: alpha_sp must be > 0, got {self.alpha_sp}"
            )
        if self.alpha_mp <= 0:
            raise ValueError(
                f"Penalty: alpha_mp must be > 0, got {self.alpha_mp}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.constraints("Penalty", self.alpha_sp, self.alpha_mp)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Transformation — exact, no spurious modes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Transformation(ConstraintHandler):
    """``constraints Transformation`` — exact transformation method."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.constraints("Transformation")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Lagrange — Lagrange multipliers, optional alpha weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Lagrange(ConstraintHandler):
    """``constraints Lagrange [alphaSP alphaMP]`` — Lagrange multipliers.

    Both alphas are optional; OpenSees uses 1.0 by default. Supplying
    one requires supplying the other (the OpenSees command parser
    expects both or neither).
    """

    alpha_sp: float | None = None
    alpha_mp: float | None = None

    def __post_init__(self) -> None:
        if (self.alpha_sp is None) != (self.alpha_mp is None):
            raise ValueError(
                "Lagrange: supply both alpha_sp and alpha_mp, or neither "
                f"(got alpha_sp={self.alpha_sp!r}, "
                f"alpha_mp={self.alpha_mp!r})."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.alpha_sp is None:
            emitter.constraints("Lagrange")
        else:
            assert self.alpha_mp is not None  # __post_init__ guarantee
            emitter.constraints("Lagrange", self.alpha_sp, self.alpha_mp)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Auto — auto-selecting handler (Petracca, June 2024)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Auto(ConstraintHandler):
    """``constraints Auto <-verbose> <-autoPenalty $oom> <-userPenalty $val>``.

    Uses Transformation for SP constraints and PenaltyMP for MP
    constraints. By default chooses the penalty value automatically
    from the stiffness scale of the DOFs involved (``auto_penalty=True``,
    ``auto_penalty_oom=3.0``).

    Set ``user_penalty`` (with ``auto_penalty=False``) to override with
    a fixed value. The two penalty modes are mutually exclusive; the
    underlying OpenSees parser errors if both ``-autoPenalty`` and
    ``-userPenalty`` flags are emitted.
    """

    verbose: bool = False
    auto_penalty: bool = True
    auto_penalty_oom: float = 3.0
    user_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not self.auto_penalty and self.user_penalty <= 0:
            raise ValueError(
                "Auto: when auto_penalty=False, user_penalty must be > 0 "
                f"(got user_penalty={self.user_penalty})."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[int | float | str] = []
        if self.verbose:
            args.append("-verbose")
        if self.auto_penalty:
            # Only emit the flag when the OOM differs from the parser
            # default (3.0) — keeps the emitted command minimal when
            # the user is just asking for "auto" with defaults.
            if self.auto_penalty_oom != 3.0:
                args.extend(("-autoPenalty", self.auto_penalty_oom))
        else:
            args.extend(("-userPenalty", self.user_penalty))
        emitter.constraints("Auto", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoProjection — momentum-conserving explicit constraint projection
# (Ladruno fork, ADR-30 / HANDLER_TAG 33001). ADR 0068: the explicit-safe
# enforcement for enforce="equation" ties — a projection (not a stiffness
# addition), so it is exact and Δt-neutral. Also enforces equalDOF /
# rigidLink / rigidDiaphragm and EQ_Constraint groups (ADR-30 P3).
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoProjection(ConstraintHandler):
    """``constraints LadrunoProjection <-verbose> <-projectICs> <-icTol $tol>``.

    **Fork-only** (Ladruno OpenSees, ADR-30). A stock OpenSees build
    fails loud at this command. Use for explicit transient analysis with
    constraints (equalDOF / rigidLink / rigidDiaphragm / ``enforce=
    "equation"`` ties): enforcement is a momentum-conserving *projection*
    of the acceleration, so unlike a penalty it does not shrink the
    critical time step. Requires a projection-aware explicit integrator.
    """

    verbose: bool = False
    project_ics: bool = False
    ic_tol: float | None = None

    def __post_init__(self) -> None:
        if self.ic_tol is not None and not self.project_ics:
            raise ValueError(
                "LadrunoProjection: ic_tol is only meaningful with "
                "project_ics=True (the -icTol flag rides -projectICs)."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[int | float | str] = []
        if self.verbose:
            args.append("-verbose")
        if self.project_ics:
            args.append("-projectICs")
            if self.ic_tol is not None:
                args.extend(("-icTol", self.ic_tol))
        emitter.constraints("LadrunoProjection", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
