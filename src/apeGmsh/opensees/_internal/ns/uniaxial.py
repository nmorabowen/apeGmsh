"""
``_UniaxialMaterialNS`` — backs ``ops.uniaxialMaterial.<Type>(...)``.

Phase 1A populates the namespace with one typed method per OpenSees
uniaxial material primitive. Each method has a fully-typed, kw-only
signature (per charter P12 — no ``**kwargs``), constructs the typed
dataclass (which validates), and routes through the bridge's
``_register`` to allocate a tag.
"""
from __future__ import annotations

from ...material.uniaxial import (
    ENT,
    Concrete01,
    Concrete02,
    ElasticMaterial,
    Hysteretic,
    Steel01,
    Steel02,
)
from ._base import _BridgeNamespace


__all__ = ["_UniaxialMaterialNS"]


class _UniaxialMaterialNS(_BridgeNamespace):
    """``ops.uniaxialMaterial.<Type>(...)`` — Phase 1A."""

    def Steel01(
        self, *,
        fy: float,
        E:  float,
        b:  float,
        a1: float | None = None,
        a2: float | None = None,
        a3: float | None = None,
        a4: float | None = None,
    ) -> Steel01:
        return self._bridge._register(
            Steel01(fy=fy, E=E, b=b, a1=a1, a2=a2, a3=a3, a4=a4)
        )

    def Steel02(
        self, *,
        fy:       float,
        E:        float,
        b:        float,
        R0:       float = 20.0,
        cR1:      float = 0.925,
        cR2:      float = 0.15,
        a1:       float | None = None,
        a2:       float | None = None,
        a3:       float | None = None,
        a4:       float | None = None,
        sig_init: float | None = None,
    ) -> Steel02:
        return self._bridge._register(
            Steel02(
                fy=fy, E=E, b=b,
                R0=R0, cR1=cR1, cR2=cR2,
                a1=a1, a2=a2, a3=a3, a4=a4,
                sig_init=sig_init,
            )
        )

    def Concrete01(
        self, *,
        fpc:   float,
        epsc0: float,
        fpcu:  float,
        epsU:  float,
    ) -> Concrete01:
        return self._bridge._register(
            Concrete01(fpc=fpc, epsc0=epsc0, fpcu=fpcu, epsU=epsU)
        )

    def Concrete02(
        self, *,
        fpc:        float,
        epsc0:      float,
        fpcu:       float,
        epsU:       float,
        lambda_val: float,
        ft:         float,
        Ets:        float,
    ) -> Concrete02:
        return self._bridge._register(
            Concrete02(
                fpc=fpc, epsc0=epsc0, fpcu=fpcu, epsU=epsU,
                lambda_val=lambda_val, ft=ft, Ets=Ets,
            )
        )

    def Hysteretic(
        self, *,
        s1p:     float,
        e1p:     float,
        s2p:     float,
        e2p:     float,
        s1n:     float,
        e1n:     float,
        s2n:     float,
        e2n:     float,
        pinch_x: float,
        pinch_y: float,
        damage1: float,
        damage2: float,
        s3p:     float | None = None,
        e3p:     float | None = None,
        s3n:     float | None = None,
        e3n:     float | None = None,
        beta:    float = 0.0,
    ) -> Hysteretic:
        return self._bridge._register(
            Hysteretic(
                s1p=s1p, e1p=e1p, s2p=s2p, e2p=e2p,
                s1n=s1n, e1n=e1n, s2n=s2n, e2n=e2n,
                pinch_x=pinch_x, pinch_y=pinch_y,
                damage1=damage1, damage2=damage2,
                s3p=s3p, e3p=e3p, s3n=s3n, e3n=e3n,
                beta=beta,
            )
        )

    def ElasticMaterial(
        self, *,
        E:   float,
        eta: float = 0.0,
    ) -> ElasticMaterial:
        return self._bridge._register(ElasticMaterial(E=E, eta=eta))

    def ENT(self, *, E: float) -> ENT:
        return self._bridge._register(ENT(E=E))
