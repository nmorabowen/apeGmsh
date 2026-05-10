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

See ``architecture/api-design.md`` for the namespace surface and
``architecture/emitter.md`` for the underlying ``constraints(c_type,
*args: float)`` Protocol method.
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
