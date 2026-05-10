"""
Coordinate-system primitives + ``geomTransf`` typed primitives.

This module re-exports the CS classes shipped in
:mod:`apeGmsh.solvers._opensees_csys` per ADR 0010 (the **only**
sanctioned dependency from :mod:`apeGmsh.opensees` into
:mod:`apeGmsh.solvers` — see ADR 0009) and ships the typed
``geomTransf`` primitives (Linear, PDelta, Corotational) that
replace the legacy ``add_geom_transf`` flow.

OpenSees command shape::

    geomTransf <Linear|PDelta|Corotational> tag <vecxzX vecxzY vecxzZ>

In 3D, ``vecxz`` is required: a vector in the local x-z plane that
fixes the section's strong axis. In 2D, ``vecxz`` is omitted and
local axes are derived from the element's node order.

Two ways to pin orientation at construction:

* **Explicit ``vecxz``** — one 3-tuple applied to every element that
  references this transform. Suitable for prismatic frames where all
  members share the same orientation (e.g. all columns + all
  beams in a regular Cartesian frame).

* **Coordinate system (``csys=``)** — the bridge derives a per-element
  ``vecxz`` from the CS triad and the element's tangent at build
  time. Curved members emit one ``geomTransf`` line per distinct
  vecxz (per ADR 0010). Suitable for ring beams, dome ribs, arches.

Exactly one of ``csys`` / ``vecxz`` must be supplied (or neither, for
2D models — the bridge will reject "neither" in 3D at build time).

Phase 1D scope
--------------
The typed primitives accept BOTH ``csys`` and explicit ``vecxz`` at
construction. ``_emit`` only handles the ``vecxz`` case for now;
csys-derived per-element vecxz fan-out lands when the build pipeline
materializes it (Phase 4). When ``csys`` is set and ``vecxz`` is
``None``, ``_emit`` raises :class:`NotImplementedError` with a clear
message.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from apeGmsh.solvers._opensees_csys import (
    Cartesian,
    Cylindrical,
    Spherical,
    resolve_vecxz,
)

from ._internal.types import GeomTransf, Primitive

if TYPE_CHECKING:
    from .emitter.base import Emitter


__all__ = [
    # CS classes (ADR 0010)
    "Cartesian",
    "Cylindrical",
    "Spherical",
    "resolve_vecxz",
    # GeomTransf primitives (Phase 1D)
    "CoordSys",
    "Linear",
    "PDelta",
    "Corotational",
]


# Union of CS classes for typed kwargs. The CS classes are not part of
# the typed-primitive hierarchy (they're standalone helpers from
# apeGmsh.solvers); we accept any of the three or None.
CoordSys: TypeAlias = Cartesian | Cylindrical | Spherical


# ---------------------------------------------------------------------------
# Shared validation helper
# ---------------------------------------------------------------------------

def _check_csys_xor_vecxz(
    type_name: str,
    csys: CoordSys | None,
    vecxz: tuple[float, float, float] | None,
) -> None:
    """Reject "both csys and vecxz supplied"; "neither" is permitted.

    The "neither" case is the 2D-model construction path — the bridge
    rejects it at build time when the model is 3D. We don't reject it
    at construction because that would force users to pass a sentinel
    in 2D.
    """
    if csys is not None and vecxz is not None:
        raise ValueError(
            f"{type_name}: supply either csys= or vecxz=, not both."
        )


# ---------------------------------------------------------------------------
# Linear — small-displacement linearized transform
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Linear(GeomTransf):
    """``geomTransf Linear`` — small-displacement linearized transform.

    Either ``csys`` (the bridge derives per-element ``vecxz`` at build
    time via the CS rule from ADR 0010) or ``vecxz`` (one vector
    applied to all elements that reference this transform) must be
    supplied for 3D models. Supplying neither is permitted at
    construction (2D-only path); the bridge will raise if used in a
    3D model.

    Parameters
    ----------
    csys
        Coordinate system used to derive ``vecxz`` per element at
        build time. Mutually exclusive with ``vecxz``.
    vecxz
        Explicit local-x-z vector (one 3-tuple). Mutually exclusive
        with ``csys``.
    roll_deg
        Rotation about the element tangent applied AFTER the CS rule
        produces a candidate ``vecxz``. Useful for asymmetric
        sections (channels, angles) on curved members.
    """

    csys: CoordSys | None = None
    vecxz: tuple[float, float, float] | None = None
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        _check_csys_xor_vecxz("Linear", self.csys, self.vecxz)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        # Phase 1D scope: ``_emit`` requires an explicit ``vecxz``.
        # When ``csys`` is set, the bridge build pipeline (Phase 4)
        # is responsible for substituting one or more concrete
        # ``vecxz`` vectors prior to ``_emit`` — see ADR 0010 for the
        # one-line-per-distinct-vecxz behavior on curved members.
        if self.vecxz is None:
            raise NotImplementedError(
                "Linear._emit: csys-derived per-element vecxz fan-out "
                "is the bridge build pipeline's responsibility (Phase "
                "4). For Phase 1D unit tests, construct with explicit "
                "vecxz=."
            )
        emitter.geomTransf("Linear", tag, *self.vecxz)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# PDelta — adds P-Δ effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class PDelta(GeomTransf):
    """``geomTransf PDelta`` — linear transform plus P-Δ effects.

    See :class:`Linear` for parameter semantics — PDelta shares the
    same construction shape, differing only in the OpenSees command
    type token emitted.
    """

    csys: CoordSys | None = None
    vecxz: tuple[float, float, float] | None = None
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        _check_csys_xor_vecxz("PDelta", self.csys, self.vecxz)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if self.vecxz is None:
            raise NotImplementedError(
                "PDelta._emit: csys-derived per-element vecxz fan-out "
                "is the bridge build pipeline's responsibility (Phase "
                "4). For Phase 1D unit tests, construct with explicit "
                "vecxz=."
            )
        emitter.geomTransf("PDelta", tag, *self.vecxz)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Corotational — finite-displacement rotations
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Corotational(GeomTransf):
    """``geomTransf Corotational`` — finite-displacement transform.

    See :class:`Linear` for parameter semantics — Corotational shares
    the same construction shape, differing only in the OpenSees
    command type token emitted.
    """

    csys: CoordSys | None = None
    vecxz: tuple[float, float, float] | None = None
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        _check_csys_xor_vecxz("Corotational", self.csys, self.vecxz)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if self.vecxz is None:
            raise NotImplementedError(
                "Corotational._emit: csys-derived per-element vecxz "
                "fan-out is the bridge build pipeline's responsibility "
                "(Phase 4). For Phase 1D unit tests, construct with "
                "explicit vecxz=."
            )
        emitter.geomTransf("Corotational", tag, *self.vecxz)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
