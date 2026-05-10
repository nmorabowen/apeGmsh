"""
``_ElementNS`` — backs ``ops.element.<Type>(pg=..., ...)``.

Phase 0 shipped the empty stub; Phase 2 populates it with one typed
method per OpenSees element. Multiple Phase 2 slice agents
(beam_column, truss, zero_length, shell, solid, joint) extend this
class — each appends its own typed methods. Methods are kept in
alphabetical-ish order to minimize merge conflicts; coordinator
ordering is final.
"""
from __future__ import annotations

from ...element.truss import CorotTruss, InertiaTruss, Truss
from ...element.zero_length import (
    ZeroLength,
    ZeroLengthMatDir,
    ZeroLengthSection,
)
from ..types import Section, UniaxialMaterial
from ._base import _BridgeNamespace


__all__ = ["_ElementNS"]


class _ElementNS(_BridgeNamespace):
    """``ops.element.<Type>(pg=..., ...)`` — Phase 2 population.

    Each method constructs the typed Element primitive, registers it
    with the bridge (allocating its tag), and returns the typed
    instance. Per the namespace contract in :mod:`api-design`, every
    signature is fully kw-only with explicit types — no ``**kwargs``
    (P12).

    The bridge fans the spec across its physical group at build time;
    the typed class never carries node tags.
    """

    # -- Truss family ---------------------------------------------------

    def CorotTruss(
        self,
        *,
        pg: str,
        A: float,
        material: UniaxialMaterial,
        rho: float | None = None,
        c_mass: bool = False,
        do_rayleigh: bool = False,
    ) -> CorotTruss:
        """``element CorotTruss`` — corotational uniaxial-material truss."""
        return self._bridge._register(
            CorotTruss(
                pg=pg, A=A, material=material,
                rho=rho, c_mass=c_mass, do_rayleigh=do_rayleigh,
            )
        )

    def InertiaTruss(
        self,
        *,
        pg: str,
        mass: float,
    ) -> InertiaTruss:
        """``element InertiaTruss`` — mass-only truss."""
        return self._bridge._register(
            InertiaTruss(pg=pg, mass=mass)
        )

    def Truss(
        self,
        *,
        pg: str,
        A: float,
        material: UniaxialMaterial,
        rho: float | None = None,
        c_mass: bool = False,
        do_rayleigh: bool = False,
    ) -> Truss:
        """``element Truss`` — uniaxial-material truss."""
        return self._bridge._register(
            Truss(
                pg=pg, A=A, material=material,
                rho=rho, c_mass=c_mass, do_rayleigh=do_rayleigh,
            )
        )

    # -- ZeroLength family ----------------------------------------------

    def ZeroLength(
        self,
        *,
        pg: str,
        mat_dirs: tuple[ZeroLengthMatDir, ...],
        orient: tuple[float, float, float, float, float, float]
        | None = None,
        do_rayleigh: bool = False,
    ) -> ZeroLength:
        """``element zeroLength`` — coupled (material, dof) springs."""
        return self._bridge._register(
            ZeroLength(
                pg=pg,
                mat_dirs=mat_dirs,
                orient=orient,
                do_rayleigh=do_rayleigh,
            )
        )

    def ZeroLengthSection(
        self,
        *,
        pg: str,
        section: Section,
        orient: tuple[float, float, float, float, float, float]
        | None = None,
        do_rayleigh: bool = False,
    ) -> ZeroLengthSection:
        """``element zeroLengthSection`` — section-coupled zero-length."""
        return self._bridge._register(
            ZeroLengthSection(
                pg=pg,
                section=section,
                orient=orient,
                do_rayleigh=do_rayleigh,
            )
        )
