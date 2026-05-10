"""
Truss-family elements — typed primitives for ``element Truss``,
``element CorotTruss``, and ``element InertiaTruss``.

OpenSees commands::

    element Truss       tag iNode jNode A matTag [-rho rho] [-cMass c] [-doRayleigh r]
    element CorotTruss  tag iNode jNode A matTag [-rho rho] [-cMass c] [-doRayleigh r]
    element InertiaTruss tag iNode jNode mass

``Truss`` and ``CorotTruss`` share the same parameter shape; only
the OpenSees type token differs. ``InertiaTruss`` is a mass-only
element — no area, no material, no rho.

Element fan-out
===============

The bridge fans the spec across its physical group at build time. The
typed class:

* reads the per-element node tags via
  :func:`apeGmsh.opensees._internal.tag_resolution.current_element_nodes`,
* resolves the composed material's tag via
  :func:`~apeGmsh.opensees._internal.tag_resolution.resolve_tag`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.tag_resolution import (
    current_element_nodes,
    resolve_tag,
)
from .._internal.types import Element, Primitive, UniaxialMaterial

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Truss",
    "CorotTruss",
    "InertiaTruss",
]


# ---------------------------------------------------------------------------
# Truss / CorotTruss — same shape, different type token
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Truss(Element):
    """``element Truss`` — uniaxial-material truss.

    OpenSees command::

        element Truss tag iNode jNode A matTag [-rho rho] \
            [-cMass cFlag] [-doRayleigh rFlag]

    Parameters
    ----------
    pg
        Physical-group label whose 2-node line elements receive this
        spec. The bridge fans the spec across the PG at build time.
    A
        Cross-sectional area. Must be > 0.
    material
        The :class:`UniaxialMaterial` integrated along the truss axis.
    rho
        Optional mass per unit length (``-rho``). Must be >= 0 if
        supplied.
    c_mass
        Use consistent (rather than lumped) mass formulation
        (``-cMass``).
    do_rayleigh
        Include the element in Rayleigh damping (``-doRayleigh``).
    """

    pg: str
    A: float
    material: UniaxialMaterial
    rho: float | None = None
    c_mass: bool = False
    do_rayleigh: bool = False

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise ValueError(f"Truss: A must be > 0, got {self.A!r}")
        if self.rho is not None and self.rho < 0:
            raise ValueError(f"Truss: rho must be >= 0, got {self.rho!r}")

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 2:
            raise ValueError(
                f"Truss: expected 2 node tags, got {len(nodes)}"
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, self.A, mat_tag]
        if self.rho is not None:
            args += ["-rho", self.rho]
        if self.c_mass:
            args += ["-cMass", 1]
        if self.do_rayleigh:
            args += ["-doRayleigh", 1]
        emitter.element("Truss", tag, *args)


@dataclass(frozen=True, kw_only=True, slots=True)
class CorotTruss(Element):
    """``element CorotTruss`` — corotational variant of :class:`Truss`.

    Same parameter shape as :class:`Truss`; only the OpenSees type
    token emitted is ``"CorotTruss"``. CorotTruss tracks large
    rotations exactly via a corotational frame and is the right
    choice for cable / brace problems with significant geometric
    nonlinearity.
    """

    pg: str
    A: float
    material: UniaxialMaterial
    rho: float | None = None
    c_mass: bool = False
    do_rayleigh: bool = False

    def __post_init__(self) -> None:
        if self.A <= 0:
            raise ValueError(f"CorotTruss: A must be > 0, got {self.A!r}")
        if self.rho is not None and self.rho < 0:
            raise ValueError(
                f"CorotTruss: rho must be >= 0, got {self.rho!r}"
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return (self.material,)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 2:
            raise ValueError(
                f"CorotTruss: expected 2 node tags, got {len(nodes)}"
            )
        mat_tag = resolve_tag(emitter, self.material)
        args: list[int | float | str] = [*nodes, self.A, mat_tag]
        if self.rho is not None:
            args += ["-rho", self.rho]
        if self.c_mass:
            args += ["-cMass", 1]
        if self.do_rayleigh:
            args += ["-doRayleigh", 1]
        emitter.element("CorotTruss", tag, *args)


# ---------------------------------------------------------------------------
# InertiaTruss — mass-only truss
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class InertiaTruss(Element):
    """``element InertiaTruss`` — mass-only truss between two nodes.

    OpenSees command::

        element InertiaTruss tag iNode jNode mass

    No stiffness, no material — just a lumped mass distributed across
    the two end nodes. Used to model added mass on cable / brace
    elements without changing their stiffness contribution.

    Parameters
    ----------
    pg
        Physical-group label whose 2-node line elements receive this
        spec.
    mass
        Mass per unit length. Must be > 0.
    """

    pg: str
    mass: float

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError(
                f"InertiaTruss: mass must be > 0, got {self.mass!r}"
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        nodes = current_element_nodes(emitter)
        if len(nodes) != 2:
            raise ValueError(
                f"InertiaTruss: expected 2 node tags, got {len(nodes)}"
            )
        emitter.element("InertiaTruss", tag, *nodes, self.mass)
