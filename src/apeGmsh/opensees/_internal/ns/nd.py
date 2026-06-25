"""
``_NDMaterialNS`` — backs ``ops.nDMaterial.<Type>(...)``.

Phase 1B populates this with one typed method per OpenSees nD material.
Each method constructs the matching ``@dataclass(frozen=...)`` instance
from :mod:`apeGmsh.opensees.material.nd` and registers it with the
bridge so a tag is allocated.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from ...material.nd import (
    ASDConcrete3D,
    ASDPlasticMaterial3D as _ASDPlasticMaterial3DCls,
    DruckerPrager,
    ElasticIsotropic,
    InitDefGrad,
    J2Plasticity,
    LadrunoCohesiveHingeBiaxial,
    LadrunoConcrete3D,
    LadrunoJ2,
    LadrunoJ2Finite,
    LadrunoRCConcrete,
    LadrunoRCFiniteStrain,
    LogStrain,
    MohrCoulombSoil as _build_mohr_coulomb_soil,
    PlaneStrain,
    StagedStrain,
)
from ..types import NDMaterial
from ._base import _BridgeNamespace



__all__ = ["_NDMaterialNS"]

#: The two LadrunoRC subclasses share the identical ``from_fc`` grammar; this
#: restricted TypeVar lets :meth:`_NDMaterialNS._build_rc` flow the concrete
#: subclass type through (``from_fc`` returns ``Self``), so each public method
#: keeps its narrow declared return type.
_RC = TypeVar("_RC", LadrunoRCConcrete, LadrunoRCFiniteStrain)


class _NDMaterialNS(_BridgeNamespace):
    """``ops.nDMaterial.<Type>(...)`` — Phase 1B materials."""

    def ElasticIsotropic(
        self,
        *,
        E: float,
        nu: float,
        rho: float = 0.0,
        name: str | None = None,
    ) -> ElasticIsotropic:
        """Register an :class:`ElasticIsotropic` continuum material."""
        return self._bridge._register(
            ElasticIsotropic(E=E, nu=nu, rho=rho), name=name
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
        name: str | None = None,
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
            ),
            name=name,
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
        name: str | None = None,
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
            ),
            name=name,
        )

    # -- ASDPlasticMaterial3D family (Phase SSI-1.5) ----------------------

    def ASDPlasticMaterial3D(
        self,
        *,
        yf: str,
        pf: str,
        el: str,
        iv: str,
        internal_variables: dict[str, float | tuple[float, ...]] | None = None,
        model_parameters: dict[str, float] | None = None,
        integration_options: dict[str, float | int | str] | None = None,
        name: str | None = None,
    ) -> _ASDPlasticMaterial3DCls:
        """Register a generic :class:`ASDPlasticMaterial3D`.

        Accepts dicts for the three keyed blocks; the bridge converts
        them to tuples internally for the frozen-dataclass storage.
        Insertion order in the resulting Tcl emission matches the
        dict iteration order (Python 3.7+ insertion-ordered).

        ``internal_variables`` values may be scalars (for 1-element
        IVs like ``DP_cohesion``, ``YieldStress``) or tuples (for
        N-element IVs like ``BackStress`` which is a 6-vector); both
        are normalized to tuples for storage.

        Prefer :meth:`MohrCoulombSoil` for the standard SSI rock /
        soil case — it pre-fills the parameter dict so callers don't
        repeat ~25 zero-fills per call site.
        """
        iv_tuples = tuple(
            (
                name,
                (float(values),) if isinstance(values, (int, float))
                else tuple(float(v) for v in values),
            )
            for name, values in (internal_variables or {}).items()
        )
        mp_tuples = tuple(
            (name, float(value))
            for name, value in (model_parameters or {}).items()
        )
        io_tuples = tuple(
            (name, value)
            for name, value in (integration_options or {}).items()
        )
        return self._bridge._register(
            _ASDPlasticMaterial3DCls(
                yf=yf, pf=pf, el=el, iv=iv,
                internal_variables=iv_tuples,
                model_parameters=mp_tuples,
                integration_options=io_tuples,
            ),
            name=name,
        )

    def ASDConcrete3D(
        self,
        *,
        E: float,
        v: float,
        fc: float,
        ft: float | None = None,
        Gf: float | None = None,
        Gc: float | None = None,
        lch_ref: float | None = None,
        rho: float = 0.0,
        Kc: float = 2.0 / 3.0,
        eta: float = 0.0,
        cdf: float = 0.0,
        implex: bool = False,
        name: str | None = None,
    ) -> ASDConcrete3D:
        """Register a Petracca plastic-damage :class:`ASDConcrete3D` from physics.

        Builds the backbone in Python from ``(fc, ft, Gf, Gc)`` and emits
        the explicit curve + ``-autoRegularization $lch_ref`` (ADR 0044).
        ``ft``/``Gf``/``Gc``/``lch_ref`` default to the CEB-FIP / native
        self-derived values; pass a representative element size as
        ``lch_ref`` for better-conditioned softening. For 2-D/shell
        elements wrap the result in :meth:`PlaneStrain`.
        """
        return self._bridge._register(
            ASDConcrete3D.from_fc(
                E=E, v=v, fc=fc, ft=ft, Gf=Gf, Gc=Gc, lch_ref=lch_ref,
                rho=rho, Kc=Kc, eta=eta, cdf=cdf, implex=implex,
            ),
            name=name,
        )

    def PlaneStrain(
        self, *, base: NDMaterial | str, name: str | None = None
    ) -> PlaneStrain:
        """Register a :class:`PlaneStrain` 2-D wrapper around a 3-D nDMaterial.

        Use whenever a 2-D element (``FourNodeQuad``, ``Tri31``) needs
        to consume a 3-D-only constitutive law (e.g.
        ``ASDPlasticMaterial3D``).  ``base`` accepts the registered
        nDMaterial handle or the name it was registered under.
        """
        base = self._bridge._resolve(base, base=NDMaterial)
        return self._bridge._register(PlaneStrain(base=base), name=name)

    # -- Ladruno fork — J2 plasticity family ------------------------------

    def LadrunoJ2(
        self,
        *,
        K: float,
        G: float,
        sig0: float,
        Qinf: float = 0.0,
        b: float = 0.0,
        Hiso: float = 0.0,
        backstresses: Sequence[tuple[float, float]] = (),
        rho: float = 0.0,
        lch_ref: float | None = None,
        damage: tuple[float, float, float, float] | None = None,
        implex: bool = False,
        name: str | None = None,
    ) -> LadrunoJ2:
        """Register a :class:`LadrunoJ2` combined-hardening von Mises material.

        Ladruno fork (``ND_TAG`` 33011); see :class:`LadrunoJ2`. ``Qinf``/
        ``b``/``Hiso`` set the Voce + linear isotropic hardening;
        ``backstresses`` a list of ``(C, gamma)`` Chaboche pairs (<= 8);
        ``damage`` the optional Lemaitre ``(r, s, pD, Dc)`` mode.

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        return self._bridge._register(
            LadrunoJ2(
                K=K, G=G, sig0=sig0, Qinf=Qinf, b=b, Hiso=Hiso,
                backstresses=tuple((float(C), float(g)) for C, g in backstresses),
                rho=rho, lch_ref=lch_ref, damage=damage, implex=implex,
            ),
            name=name,
        )

    def LadrunoJ2Finite(
        self,
        *,
        K: float,
        G: float,
        sig0: float,
        Qinf: float = 0.0,
        b: float = 0.0,
        Hiso: float = 0.0,
        backstresses: Sequence[tuple[float, float]] = (),
        rho: float = 0.0,
        implex: bool = False,
        name: str | None = None,
    ) -> LadrunoJ2Finite:
        """Register a :class:`LadrunoJ2Finite` finite-strain-native J2 material.

        Ladruno fork (``ND_TAG`` 33012); see :class:`LadrunoJ2Finite`. Use
        for combined hardening **with** large rotation; the sole consumer is
        ``LadrunoBrick ... -geom finite``. No ``-damage`` /
        ``-autoRegularization`` here (the finite material rejects them).

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        return self._bridge._register(
            LadrunoJ2Finite(
                K=K, G=G, sig0=sig0, Qinf=Qinf, b=b, Hiso=Hiso,
                backstresses=tuple((float(C), float(g)) for C, g in backstresses),
                rho=rho, implex=implex,
            ),
            name=name,
        )

    # -- Ladruno fork — concrete plastic-damage family --------------------

    def LadrunoConcrete3D(
        self,
        *,
        E: float,
        nu: float,
        fc: float,
        ft: float,
        Gf: float,
        Gc: float,
        e: float | None = None,
        kupfer: float = 1.16,
        Df: float = 1.0,
        As: float = 2.0,
        rho: float = 0.0,
        hardening: tuple[float, float] = (0.3, 0.5),
        ductility: tuple[float, float, float, float] = (
            0.08, 0.003, 2.0, 1.0e-6),
        lch: float = 1.0,
        auto_regularize: bool = False,
        implex: bool = False,
        eta: float = 0.0,
        ct_temper: str = "none",
        hoop_k: float = 0.0,
        hoop_fy: float = 1.0e30,
        name: str | None = None,
    ) -> LadrunoConcrete3D:
        """Register a :class:`LadrunoConcrete3D` CDPM2-grade solid concrete.

        Ladruno fork (``ND_TAG`` 33017); see :class:`LadrunoConcrete3D`.
        ``fc``/``ft``/``Gf``/``Gc`` are positive magnitudes (``ft < fc``).
        The consistent tangent is **non-symmetric** — pair with an
        unsymmetric solver (``system UmfPack`` / ``FullGeneral``). Supply
        ``e`` directly or let it derive from ``kupfer`` (not both).

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        return self._bridge._register(
            LadrunoConcrete3D(
                E=E, nu=nu, fc=fc, ft=ft, Gf=Gf, Gc=Gc,
                e=e, kupfer=kupfer, Df=Df, As=As, rho=rho,
                hardening=hardening, ductility=ductility,
                lch=lch, auto_regularize=auto_regularize, implex=implex,
                eta=eta, ct_temper=ct_temper, hoop_k=hoop_k, hoop_fy=hoop_fy,
            ),
            name=name,
        )

    def _build_rc(
        self,
        cls: "type[_RC]",
        *,
        name: str | None,
        **kw: Any,
    ) -> "_RC":
        """Shared ``from_fc`` build + register for the RC namespace methods.

        ``LadrunoRCConcrete`` and ``LadrunoRCFiniteStrain`` carry the
        identical command grammar, so both namespace methods funnel their
        full typed surface through here (forwarded to ``cls.from_fc``).
        """
        return self._bridge._register(cls.from_fc(**kw), name=name)

    def LadrunoRCConcrete(
        self,
        *,
        E: float,
        nu: float,
        fc: float,
        ft: float | None = None,
        Gf: float | None = None,
        Gc: float | None = None,
        lch_ref: float | None = None,
        rho: float = 0.0,
        regularize: bool = True,
        Kc: float = 2.0 / 3.0,
        beta: bool = False,
        beta_floor: float = 0.1,
        lubliner_reduced: bool = False,
        tangent: str = "consistent",
        interlock: bool = False,
        cyclic: bool = False,
        agg: float = 16.0,
        crack_strain: float = 0.0,
        crack_spacing: float = 0.0,
        lch: float = 0.0,
        beta_sr_min: float = 0.01,
        xcrack: bool = False,
        deg_kappa: float = 0.5,
        deg_slip_ref: float = 0.01,
        deg_min: float = 0.1,
        implex: bool = False,
        implex_alpha: float = 1.0,
        implex_control: tuple[float, float] | None = None,
        shear_retention: str = "mcft",
        shear_ret_factor: float = 0.4,
        tens_stiff: str = "off",
        tens_stiff_c: float = 500.0,
        tens_stiff_alpha: float = 1.0,
        name: str | None = None,
    ) -> LadrunoRCConcrete:
        """Register a :class:`LadrunoRCConcrete` RC plastic-damage + MCFT material.

        Ladruno fork; see :class:`LadrunoRCConcrete`. Backbones are built in
        Python from ``(fc, ft, Gf, Gc)`` (CEB-FIP defaults), with
        ``-autoRegularization`` wired when ``regularize`` (default). The full
        MCFT aggregate-interlock / tension-stiffening / IMPL-EX flag surface
        is exposed here; construct :class:`LadrunoRCConcrete` directly only to
        supply raw (non-``from_fc``) backbone points.

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        return self._build_rc(
            LadrunoRCConcrete, name=name,
            E=E, nu=nu, fc=fc, ft=ft, Gf=Gf, Gc=Gc, lch_ref=lch_ref, rho=rho,
            regularize=regularize, Kc=Kc, beta=beta, beta_floor=beta_floor,
            lubliner_reduced=lubliner_reduced, tangent=tangent,
            interlock=interlock, cyclic=cyclic, agg=agg,
            crack_strain=crack_strain, crack_spacing=crack_spacing, lch=lch,
            beta_sr_min=beta_sr_min, xcrack=xcrack, deg_kappa=deg_kappa,
            deg_slip_ref=deg_slip_ref, deg_min=deg_min, implex=implex,
            implex_alpha=implex_alpha, implex_control=implex_control,
            shear_retention=shear_retention, shear_ret_factor=shear_ret_factor,
            tens_stiff=tens_stiff, tens_stiff_c=tens_stiff_c,
            tens_stiff_alpha=tens_stiff_alpha,
        )

    def LadrunoRCFiniteStrain(
        self,
        *,
        E: float,
        nu: float,
        fc: float,
        ft: float | None = None,
        Gf: float | None = None,
        Gc: float | None = None,
        lch_ref: float | None = None,
        rho: float = 0.0,
        regularize: bool = True,
        Kc: float = 2.0 / 3.0,
        beta: bool = False,
        beta_floor: float = 0.1,
        lubliner_reduced: bool = False,
        tangent: str = "consistent",
        interlock: bool = False,
        cyclic: bool = False,
        agg: float = 16.0,
        crack_strain: float = 0.0,
        crack_spacing: float = 0.0,
        lch: float = 0.0,
        beta_sr_min: float = 0.01,
        xcrack: bool = False,
        deg_kappa: float = 0.5,
        deg_slip_ref: float = 0.01,
        deg_min: float = 0.1,
        implex: bool = False,
        implex_alpha: float = 1.0,
        implex_control: tuple[float, float] | None = None,
        shear_retention: str = "mcft",
        shear_ret_factor: float = 0.4,
        tens_stiff: str = "off",
        tens_stiff_c: float = 500.0,
        tens_stiff_alpha: float = 1.0,
        name: str | None = None,
    ) -> LadrunoRCFiniteStrain:
        """Register a :class:`LadrunoRCFiniteStrain` finite-strain RC material.

        Ladruno fork; the Hencky finite-strain view of
        :class:`LadrunoRCConcrete` — same plastic-damage + MCFT law at large
        rotation / strain. A ``FiniteStrainNDMaterial`` consumed by
        ``LadrunoBrick ... -geom finite``. Same full flag surface as
        :meth:`LadrunoRCConcrete`.

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        return self._build_rc(
            LadrunoRCFiniteStrain, name=name,
            E=E, nu=nu, fc=fc, ft=ft, Gf=Gf, Gc=Gc, lch_ref=lch_ref, rho=rho,
            regularize=regularize, Kc=Kc, beta=beta, beta_floor=beta_floor,
            lubliner_reduced=lubliner_reduced, tangent=tangent,
            interlock=interlock, cyclic=cyclic, agg=agg,
            crack_strain=crack_strain, crack_spacing=crack_spacing, lch=lch,
            beta_sr_min=beta_sr_min, xcrack=xcrack, deg_kappa=deg_kappa,
            deg_slip_ref=deg_slip_ref, deg_min=deg_min, implex=implex,
            implex_alpha=implex_alpha, implex_control=implex_control,
            shear_retention=shear_retention, shear_ret_factor=shear_ret_factor,
            tens_stiff=tens_stiff, tens_stiff_c=tens_stiff_c,
            tens_stiff_alpha=tens_stiff_alpha,
        )

    def LadrunoCohesiveHingeBiaxial(
        self,
        *,
        Mcz: float,
        Gfz: float,
        Mcy: float,
        Gfy: float,
        softening: str = "exponential",
        penalty_ratio: float = 1000.0,
        bk_eta: float = 1.0,
        name: str | None = None,
    ) -> LadrunoCohesiveHingeBiaxial:
        """Register a :class:`LadrunoCohesiveHingeBiaxial` coupled hinge surface.

        Ladruno fork (``ND_TAG`` 33004); the coupled Mz–My cohesive
        interaction surface for ``LadrunoDispBeamColumn -hingeBiaxial``. Each
        axis carries its own ``Mc``/``Gf``; ``bk_eta`` is the
        Benzeggagh-Kenane mode-mix exponent. See
        :class:`LadrunoCohesiveHingeBiaxial`.

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        return self._bridge._register(
            LadrunoCohesiveHingeBiaxial(
                Mcz=Mcz, Gfz=Gfz, Mcy=Mcy, Gfy=Gfy, softening=softening,
                penalty_ratio=penalty_ratio, bk_eta=bk_eta,
            ),
            name=name,
        )

    # -- Ladruno fork — finite-strain & staged-birth wrappers -------------

    def LogStrain(
        self, *, inner: NDMaterial | str, name: str | None = None
    ) -> LogStrain:
        """Register a :class:`LogStrain` Hencky finite-strain lift wrapper.

        Ladruno fork (``ND_TAG`` 33010); see :class:`LogStrain`. Lifts an
        isotropic small-strain 3-D ``inner`` to finite strain for
        ``LadrunoBrick ... -geom finite``. ``inner`` accepts the registered
        nDMaterial handle or its registered name.

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        inner = self._bridge._resolve(inner, base=NDMaterial)
        return self._bridge._register(LogStrain(inner=inner), name=name)

    def InitDefGrad(
        self, *,
        inner: NDMaterial | str,
        no_init_f: bool = False,
        F0: tuple[float, ...] | None = None,
        name: str | None = None,
    ) -> InitDefGrad:
        """Register an :class:`InitDefGrad` finite staged stress-free birth wrapper.

        Ladruno fork (``ND_TAG`` 33013); see :class:`InitDefGrad`. Makes a
        continuum element born stress-free at the deformed geometry in a
        staged build. ``inner`` must be a finite-strain material
        (``LogStrain`` / ``LadrunoJ2Finite``); ``F0`` is an optional 9
        row-major birth gradient.

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        inner = self._bridge._resolve(inner, base=NDMaterial)
        return self._bridge._register(
            InitDefGrad(inner=inner, no_init_f=no_init_f, F0=F0), name=name
        )

    def StagedStrain(
        self, *,
        inner: NDMaterial | str,
        no_init: bool = False,
        eps0: tuple[float, ...] | None = None,
        name: str | None = None,
    ) -> StagedStrain:
        """Register a :class:`StagedStrain` small-strain staged-birth wrapper.

        Ladruno fork (``ND_TAG`` 33014); see :class:`StagedStrain`. The
        everyday small-strain staged-build case (2-D or 3-D) — the inner is
        born virgin at its birth strain. ``eps0`` is an optional 6-component
        Voigt birth strain.

        Fork-only: emits on any build, errors at ``ops.run()`` on stock
        ``openseespy``.
        """
        inner = self._bridge._resolve(inner, base=NDMaterial)
        return self._bridge._register(
            StagedStrain(inner=inner, no_init=no_init, eps0=eps0), name=name
        )

    def MohrCoulombSoil(
        self,
        *,
        c: float,
        phi: float,
        psi: float,
        E: float,
        nu: float,
        rho: float = 0.0,
        ds: float = 1e-5,
        yield_stress: float = 1e10,
        initial_p0: float = 0.0,
        integration_method: str = "Backward_Euler",
        tangent_type: str = "Secant",
        f_absolute_tol: float = 1e-6,
        stress_absolute_tol: float = 1e-6,
        n_max_iterations: int = 100,
        return_to_yield_surface: str = "Disabled",
        rk45_dT_min: float = 0.01,
        rk45_niter_max: int = 100,
        name: str | None = None,
    ) -> _ASDPlasticMaterial3DCls:
        """Register an ASDPlasticMaterial3D wired for Mohr-Coulomb soil/rock.

        Convenience over :meth:`ASDPlasticMaterial3D` for the standard
        SSI case: MohrCoulomb_YF + MohrCoulomb_PF + LinearIsotropic3D_EL
        + BackStress(NullHardeningTensorFunction).  See
        :func:`apeGmsh.opensees.material.nd.MohrCoulombSoil` for the
        parameter docstring.
        """
        return self._bridge._register(
            _build_mohr_coulomb_soil(
                c=c, phi=phi, psi=psi, E=E, nu=nu, rho=rho, ds=ds,
                yield_stress=yield_stress, initial_p0=initial_p0,
                integration_method=integration_method,
                tangent_type=tangent_type,
                f_absolute_tol=f_absolute_tol,
                stress_absolute_tol=stress_absolute_tol,
                n_max_iterations=n_max_iterations,
                return_to_yield_surface=return_to_yield_surface,
                rk45_dT_min=rk45_dT_min,
                rk45_niter_max=rk45_niter_max,
            ),
            name=name,
        )
