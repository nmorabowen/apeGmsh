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
    ASDConcrete1D,
    ASDSteel1D,
    Concrete01,
    Concrete02,
    ElasticMaterial,
    Hysteretic,
    InitialStress,
    Maxwell,
    Steel01,
    Steel02,
    Viscous,
    ViscousDamper,
)
from ..types import UniaxialMaterial
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
        name: str | None = None,
    ) -> Steel01:
        return self._bridge._register(
            Steel01(fy=fy, E=E, b=b, a1=a1, a2=a2, a3=a3, a4=a4), name=name
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
        name: str | None = None,
    ) -> Steel02:
        return self._bridge._register(
            Steel02(
                fy=fy, E=E, b=b,
                R0=R0, cR1=cR1, cR2=cR2,
                a1=a1, a2=a2, a3=a3, a4=a4,
                sig_init=sig_init,
            ),
            name=name,
        )

    def ASDSteel1D(
        self, *,
        E:  float,
        sy: float,
        su: float,
        eu: float,
        implex: bool = False,
        implex_control: tuple[float, float] | None = None,
        auto_regularization: bool = False,
        buckling_lch: float | None = None,
        fracture: bool = False,
        slip_material: UniaxialMaterial | str | None = None,
        radius: float | None = None,
        K_alpha:  float | None = None,
        max_iter: int | None = None,
        tolU: float | None = None,
        tolR: float | None = None,
        name: str | None = None,
    ) -> ASDSteel1D:
        """``uniaxialMaterial ASDSteel1D`` — ASDEA plastic-damage steel.

        Backbone hardening is derived internally from ``(E, sy, su,
        eu)``. ``slip_material`` accepts a UniaxialMaterial handle or its
        registered name. See :class:`ASDSteel1D` for the full contract.
        """
        slip_material = self._bridge._resolve(
            slip_material, base=UniaxialMaterial
        )
        return self._bridge._register(
            ASDSteel1D(
                E=E, sy=sy, su=su, eu=eu,
                implex=implex,
                implex_control=implex_control,
                auto_regularization=auto_regularization,
                buckling_lch=buckling_lch,
                fracture=fracture,
                slip_material=slip_material,
                radius=radius,
                K_alpha=K_alpha,
                max_iter=max_iter,
                tolU=tolU,
                tolR=tolR,
            ),
            name=name,
        )

    def ASDConcrete1D(
        self, *,
        E: float,
        fc: float,
        ft: float | None = None,
        Gf: float | None = None,
        Gc: float | None = None,
        lch_ref: float | None = None,
        eta: float = 0.0,
        implex: bool = False,
        name: str | None = None,
    ) -> ASDConcrete1D:
        """``uniaxialMaterial ASDConcrete1D`` — Petracca plastic-damage (1-D).

        Builds an **unconfined** backbone in Python from ``(fc, ft, Gf,
        Gc)`` and emits the explicit curve + ``-autoRegularization
        $lch_ref`` (ADR 0044). Confinement-blind — bake a Mander curve into
        an explicit :class:`ASDConcrete1D` for confined members. See
        :meth:`ASDConcrete1D.from_fc` for the parameter contract.
        """
        return self._bridge._register(
            ASDConcrete1D.from_fc(
                E=E, fc=fc, ft=ft, Gf=Gf, Gc=Gc, lch_ref=lch_ref,
                eta=eta, implex=implex,
            ),
            name=name,
        )

    def ConfinedConcrete1D(
        self, *,
        E: float,
        fc: float,
        eps_cu: float,
        fcc: float | None = None,
        fl: float | None = None,
        eps_co: float = 0.002,
        plastic_ratio: float = 0.7,
        n_comp: int = 12,
        ft: float | None = None,
        Gf: float | None = None,
        lch_ref: float | None = None,
        auto_regularize: bool = False,
        eta: float = 0.0,
        implex: bool = False,
        name: str | None = None,
    ) -> ASDConcrete1D:
        """``ASDConcrete1D`` with a Mander confined-concrete backbone.

        Confinement is baked into the compression envelope (the uniaxial model
        is confinement-blind). Provide exactly one of ``fcc`` (confined
        strength) or ``fl`` (effective lateral pressure); ``eps_cu`` is the
        confined ultimate strain. See :meth:`ASDConcrete1D.from_mander` for the
        full contract — note ``auto_regularize`` defaults to ``False`` (the
        Mander envelope is physical, not crack-band-rescaled).
        """
        return self._bridge._register(
            ASDConcrete1D.from_mander(
                E=E, fc=fc, eps_cu=eps_cu, fcc=fcc, fl=fl, eps_co=eps_co,
                plastic_ratio=plastic_ratio, n_comp=n_comp,
                ft=ft, Gf=Gf, lch_ref=lch_ref, auto_regularize=auto_regularize,
                eta=eta, implex=implex,
            ),
            name=name,
        )

    def Concrete01(
        self, *,
        fpc:   float,
        epsc0: float,
        fpcu:  float,
        epsU:  float,
        name: str | None = None,
    ) -> Concrete01:
        return self._bridge._register(
            Concrete01(fpc=fpc, epsc0=epsc0, fpcu=fpcu, epsU=epsU), name=name
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
        name: str | None = None,
    ) -> Concrete02:
        return self._bridge._register(
            Concrete02(
                fpc=fpc, epsc0=epsc0, fpcu=fpcu, epsU=epsU,
                lambda_val=lambda_val, ft=ft, Ets=Ets,
            ),
            name=name,
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
        name: str | None = None,
    ) -> Hysteretic:
        return self._bridge._register(
            Hysteretic(
                s1p=s1p, e1p=e1p, s2p=s2p, e2p=e2p,
                s1n=s1n, e1n=e1n, s2n=s2n, e2n=e2n,
                pinch_x=pinch_x, pinch_y=pinch_y,
                damage1=damage1, damage2=damage2,
                s3p=s3p, e3p=e3p, s3n=s3n, e3n=e3n,
                beta=beta,
            ),
            name=name,
        )

    def ElasticMaterial(
        self, *,
        E:   float,
        eta: float = 0.0,
        name: str | None = None,
    ) -> ElasticMaterial:
        return self._bridge._register(
            ElasticMaterial(E=E, eta=eta), name=name
        )

    def ENT(self, *, E: float, name: str | None = None) -> ENT:
        return self._bridge._register(ENT(E=E), name=name)

    def Viscous(
        self, *,
        C: float,
        alpha: float = 1.0,
        min_vel: float = 1.0e-11,
        name: str | None = None,
    ) -> Viscous:
        """``uniaxialMaterial Viscous`` — pure dashpot ``F = C·|v|^alpha``.

        The canonical absorbing-boundary / Lysmer dashpot for a
        ``ZeroLength``. Has zero static stiffness — parallel it with an
        elastic spring on the same DOF when a static tangent is formed.
        See :class:`Viscous`.
        """
        return self._bridge._register(
            Viscous(C=C, alpha=alpha, min_vel=min_vel), name=name
        )

    def ViscousDamper(
        self, *,
        K: float,
        C: float,
        alpha: float,
        l_gap: float | None = None,
        name: str | None = None,
    ) -> ViscousDamper:
        """``uniaxialMaterial ViscousDamper`` — spring ``K`` in series with a
        nonlinear dashpot. Transient-only. See :class:`ViscousDamper`.
        """
        return self._bridge._register(
            ViscousDamper(K=K, C=C, alpha=alpha, l_gap=l_gap), name=name
        )

    def Maxwell(
        self, *,
        K: float,
        C: float,
        alpha: float,
        length: float,
        name: str | None = None,
    ) -> Maxwell:
        """``uniaxialMaterial Maxwell`` — Maxwell viscoelastic with nonzero
        tangent ``K`` and closed-form relaxation. See :class:`Maxwell`.
        """
        return self._bridge._register(
            Maxwell(K=K, C=C, alpha=alpha, length=length), name=name
        )

    def InitialStress(
        self, *,
        base_material: UniaxialMaterial | str,
        sigma_init:    float,
        name: str | None = None,
    ) -> InitialStress:
        """``uniaxialMaterial InitialStressMaterial`` — wrap a uniaxial
        material with a per-fiber initial stress.

        ``base_material`` accepts a UniaxialMaterial handle or its
        registered name. See :class:`InitialStress` for the full
        contract.
        """
        base_material = self._bridge._resolve(
            base_material, base=UniaxialMaterial
        )
        return self._bridge._register(
            InitialStress(base_material=base_material, sigma_init=sigma_init),
            name=name,
        )
