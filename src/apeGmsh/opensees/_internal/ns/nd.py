"""
``_NDMaterialNS`` — backs ``ops.nDMaterial.<Type>(...)``.

Phase 1B populates this with one typed method per OpenSees nD material.
Each method constructs the matching ``@dataclass(frozen=...)`` instance
from :mod:`apeGmsh.opensees.material.nd` and registers it with the
bridge so a tag is allocated.
"""
from __future__ import annotations

from ...material.nd import DruckerPrager, ElasticIsotropic, J2Plasticity
from ._base import _BridgeNamespace


__all__ = ["_NDMaterialNS"]


class _NDMaterialNS(_BridgeNamespace):
    """``ops.nDMaterial.<Type>(...)`` — Phase 1B materials."""

    def ElasticIsotropic(
        self,
        *,
        E: float,
        nu: float,
        rho: float = 0.0,
    ) -> ElasticIsotropic:
        """Register an :class:`ElasticIsotropic` continuum material."""
        return self._bridge._register(
            ElasticIsotropic(E=E, nu=nu, rho=rho)
        )

    def J2Plasticity(
        self,
        *,
        K: float,
        G: float,
        sig0: float,
        sigInf: float,
        delta: float,
        H: float,
        eta: float = 0.0,
    ) -> J2Plasticity:
        """Register a :class:`J2Plasticity` continuum material."""
        return self._bridge._register(
            J2Plasticity(
                K=K,
                G=G,
                sig0=sig0,
                sigInf=sigInf,
                delta=delta,
                H=H,
                eta=eta,
            )
        )

    def DruckerPrager(
        self,
        *,
        K: float,
        G: float,
        sigmaY: float,
        rho: float,
        rhoBar: float,
        Kinf: float,
        Ko: float,
        delta1: float,
        delta2: float,
        H: float,
        theta: float,
    ) -> DruckerPrager:
        """Register a :class:`DruckerPrager` continuum material."""
        return self._bridge._register(
            DruckerPrager(
                K=K,
                G=G,
                sigmaY=sigmaY,
                rho=rho,
                rhoBar=rhoBar,
                Kinf=Kinf,
                Ko=Ko,
                delta1=delta1,
                delta2=delta2,
                H=H,
                theta=theta,
            )
        )
