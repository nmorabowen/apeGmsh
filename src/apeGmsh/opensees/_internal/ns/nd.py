"""
``_NDMaterialNS`` — backs ``ops.nDMaterial.<Type>(...)``.

Phase 1B populates this with one typed method per OpenSees nD material.
Each method constructs the matching ``@dataclass(frozen=...)`` instance
from :mod:`apeGmsh.opensees.material.nd` and registers it with the
bridge so a tag is allocated.
"""
from __future__ import annotations

from ...material.nd import (
    ASDConcrete3D,
    ASDPlasticMaterial3D,
    DruckerPrager,
    ElasticIsotropic,
    J2Plasticity,
    MohrCoulombSoil as _build_mohr_coulomb_soil,
    NDMaterial,
    PlaneStrain,
)
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
    ) -> ASDPlasticMaterial3D:
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
            ASDPlasticMaterial3D(
                yf=yf, pf=pf, el=el, iv=iv,
                internal_variables=iv_tuples,
                model_parameters=mp_tuples,
                integration_options=io_tuples,
            )
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
            )
        )

    def PlaneStrain(self, *, base: NDMaterial) -> PlaneStrain:
        """Register a :class:`PlaneStrain` 2-D wrapper around a 3-D nDMaterial.

        Use whenever a 2-D element (``FourNodeQuad``, ``Tri31``) needs
        to consume a 3-D-only constitutive law (e.g.
        ``ASDPlasticMaterial3D``).  The ``base`` must be the registered
        primitive instance — not a tag.
        """
        return self._bridge._register(PlaneStrain(base=base))

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
    ) -> ASDPlasticMaterial3D:
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
            )
        )
