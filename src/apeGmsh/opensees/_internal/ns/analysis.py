"""
Analysis-chain namespaces — backed by Phase 3C.

Holds the seven analysis-component family namespaces in one module
because Phase 3C ships them as a single slice (one agent owns all
seven; they share patterns and the volume per file is small).

Each namespace mirrors a section of the OpenSees analysis-chain
command surface:

* ``_ConstraintsNS``  — ``ops.constraints.<Type>(...)``
* ``_NumbererNS``     — ``ops.numberer.<Type>()``
* ``_SystemNS``       — ``ops.system.<Type>()``
* ``_TestNS``         — ``ops.test.<Type>(...)``
* ``_AlgorithmNS``    — ``ops.algorithm.<Type>(...)``
* ``_IntegratorNS``   — ``ops.integrator.<Type>(...)``
* ``_AnalysisNS``     — ``ops.analysis.<Type>()``

Each method registers its primitive with the bridge via
``self._bridge._register(Cls(...))``.
"""
from __future__ import annotations

from typing import Literal, Sequence

from ...analysis.algorithm import (
    BFGS,
    Broyden,
    KrylovNewton,
    LineSearchType,
    ModifiedNewton,
    Newton,
    NewtonLineSearch,
    NewtonTangent,
)
from ...analysis.algorithm import Linear as AlgorithmLinear
from ...analysis.analysis import Static, Transient, VariableTransient
from ...analysis.constraint_handler import (
    Auto as ConstraintsAuto,
)
from ...analysis.constraint_handler import (
    Lagrange,
    LadrunoProjection,
    Penalty,
    Transformation,
)
from ...analysis.constraint_handler import Plain as ConstraintsPlain
from ...analysis.integrator import (
    ArcLength,
    CentralDifference,
    CentralDifferenceLadruno,
    DampingMode,
    DisplacementControl,
    ExplicitBathe,
    ExplicitBatheLNVD,
    ExplicitDifference,
    HHT,
    LadrunoArcLength,
    LadrunoDynamicRelaxation,
    LadrunoIndirectControl,
    LoadControl,
    Lump,
    MassMode,
    Newmark,
)
from ...analysis.numberer import AMD, RCM, ParallelPlain, ParallelRCM
from ...analysis.numberer import Plain as NumbererPlain
from ...analysis.system import (
    BandGeneral,
    BandSPD,
    Diagonal,
    FullGeneral,
    MPIDiagonal,
    Mumps,
    ParallelProfileSPD,
    ProfileSPD,
    SparseGeneral,
    SparseSYM,
    SProfileSPD,
    UmfPack,
)
from ...analysis.test import (
    EnergyIncr,
    FixedNumIter,
    NormDispIncr,
    NormUnbalance,
    RelativeNormDispIncr,
)
from ._base import _BridgeNamespace


__all__ = [
    "_ConstraintsNS",
    "_NumbererNS",
    "_SystemNS",
    "_TestNS",
    "_AlgorithmNS",
    "_IntegratorNS",
    "_AnalysisNS",
]


# ---------------------------------------------------------------------------
# _ConstraintsNS — ops.constraints.<Type>(...)
# ---------------------------------------------------------------------------

class _ConstraintsNS(_BridgeNamespace):
    """``ops.constraints.<Type>(...)`` — Phase 3C typed methods."""

    def Plain(self) -> ConstraintsPlain:
        """``constraints Plain`` — direct application (homogeneous SPs)."""
        return self._bridge._register(ConstraintsPlain())

    def Penalty(self, *, alpha_sp: float, alpha_mp: float) -> Penalty:
        """``constraints Penalty alphaSP alphaMP`` — penalty method."""
        return self._bridge._register(
            Penalty(alpha_sp=alpha_sp, alpha_mp=alpha_mp)
        )

    def Transformation(self) -> Transformation:
        """``constraints Transformation`` — exact transformation method."""
        return self._bridge._register(Transformation())

    def Lagrange(
        self,
        *,
        alpha_sp: float | None = None,
        alpha_mp: float | None = None,
    ) -> Lagrange:
        """``constraints Lagrange [alphaSP alphaMP]`` — Lagrange multipliers.

        Both alphas are optional; supply both or neither.
        """
        return self._bridge._register(
            Lagrange(alpha_sp=alpha_sp, alpha_mp=alpha_mp)
        )

    def Auto(
        self,
        *,
        verbose: bool = False,
        auto_penalty: bool = True,
        auto_penalty_oom: float = 3.0,
        user_penalty: float = 0.0,
    ) -> ConstraintsAuto:
        """``constraints Auto <-verbose> <-autoPenalty $oom> <-userPenalty $val>``.

        Auto-selecting handler (Petracca, 2024): Transformation for SP,
        PenaltyMP for MP, with the penalty value picked from the local
        stiffness scale by default.
        """
        return self._bridge._register(
            ConstraintsAuto(
                verbose=verbose,
                auto_penalty=auto_penalty,
                auto_penalty_oom=auto_penalty_oom,
                user_penalty=user_penalty,
            )
        )

    def LadrunoProjection(
        self,
        *,
        verbose: bool = False,
        project_ics: bool = False,
        ic_tol: float | None = None,
    ) -> LadrunoProjection:
        """``constraints LadrunoProjection`` — momentum-conserving explicit
        constraint projection (**Ladruno fork**, ADR-30).

        The Δt-neutral, exact enforcement for explicit transient analysis
        with constraints — including ``enforce='equation'`` ties (ADR
        0068). A stock OpenSees build fails loud at this command.
        """
        return self._bridge._register(
            LadrunoProjection(
                verbose=verbose,
                project_ics=project_ics,
                ic_tol=ic_tol,
            )
        )


# ---------------------------------------------------------------------------
# _NumbererNS — ops.numberer.<Type>()
# ---------------------------------------------------------------------------

class _NumbererNS(_BridgeNamespace):
    """``ops.numberer.<Type>()`` — Phase 3C typed methods."""

    def Plain(self) -> NumbererPlain:
        """``numberer Plain`` — number DOFs in node-add order."""
        return self._bridge._register(NumbererPlain())

    def RCM(self) -> RCM:
        """``numberer RCM`` — reverse Cuthill-McKee bandwidth reduction."""
        return self._bridge._register(RCM())

    def AMD(self) -> AMD:
        """``numberer AMD`` — approximate minimum degree reduction."""
        return self._bridge._register(AMD())

    def ParallelPlain(self) -> ParallelPlain:
        """``numberer ParallelPlain`` — parallel plain numbering.

        Only available in OpenSees builds with ``_PARALLEL_INTERPRETERS``
        (e.g. ``OpenSeesMP``). Emit-only against serial ``OpenSees.exe``
        will error at runtime.
        """
        return self._bridge._register(ParallelPlain())

    def ParallelRCM(self) -> ParallelRCM:
        """``numberer ParallelRCM`` — parallel reverse Cuthill-McKee.

        Only available in OpenSees builds with ``_PARALLEL_INTERPRETERS``
        (e.g. ``OpenSeesMP``). Emit-only against serial ``OpenSees.exe``
        will error at runtime.
        """
        return self._bridge._register(ParallelRCM())


# ---------------------------------------------------------------------------
# _SystemNS — ops.system.<Type>()
# ---------------------------------------------------------------------------

class _SystemNS(_BridgeNamespace):
    """``ops.system.<Type>()`` — Phase 3C typed methods."""

    def BandGeneral(self) -> BandGeneral:
        """``system BandGeneral`` — banded general (non-symmetric)."""
        return self._bridge._register(BandGeneral())

    def BandSPD(self) -> BandSPD:
        """``system BandSPD`` — banded symmetric-positive-definite."""
        return self._bridge._register(BandSPD())

    def ProfileSPD(self) -> ProfileSPD:
        """``system ProfileSPD`` — skyline symmetric-positive-definite."""
        return self._bridge._register(ProfileSPD())

    def UmfPack(self) -> UmfPack:
        """``system UmfPack`` — direct sparse LU (UMFPACK)."""
        return self._bridge._register(UmfPack())

    def Mumps(self) -> Mumps:
        """``system Mumps`` — MUMPS multifrontal sparse solver."""
        return self._bridge._register(Mumps())

    def SparseGeneral(self) -> SparseGeneral:
        """``system SparseGeneral`` — generic sparse general."""
        return self._bridge._register(SparseGeneral())

    def FullGeneral(self) -> FullGeneral:
        """``system FullGeneral`` — dense general (small systems only)."""
        return self._bridge._register(FullGeneral())

    def SProfileSPD(self) -> SProfileSPD:
        """``system SProfileSPD`` — single-precision skyline SPD."""
        return self._bridge._register(SProfileSPD())

    def SparseSYM(self) -> SparseSYM:
        """``system SparseSYM`` — symmetric sparse direct solver."""
        return self._bridge._register(SparseSYM())

    def Diagonal(self) -> Diagonal:
        """``system Diagonal`` — direct diagonal solver.

        The system for explicit time integration (``CentralDifference``
        / Ladruno ``ExplicitBathe`` family) — inverts a lumped diagonal
        mass element-wise. Not valid for static / implicit analysis.
        """
        return self._bridge._register(Diagonal())

    def MPIDiagonal(self) -> MPIDiagonal:
        """``system MPIDiagonal`` — distributed diagonal solver.

        Parallel counterpart of :meth:`Diagonal`. Only available in
        OpenSees builds with ``_PARALLEL_INTERPRETERS`` (e.g.
        ``OpenSeesMP``); a serial build falls back to the plain diagonal.
        """
        return self._bridge._register(MPIDiagonal())

    def ParallelProfileSPD(self) -> ParallelProfileSPD:
        """``system ParallelProfileSPD`` — distributed skyline SPD.

        Only available in OpenSees builds with ``_PARALLEL_INTERPRETERS``
        (e.g. ``OpenSeesMP``). Emit-only against serial ``OpenSees.exe``
        will error at runtime.
        """
        return self._bridge._register(ParallelProfileSPD())


# ---------------------------------------------------------------------------
# _TestNS — ops.test.<Type>(...)
# ---------------------------------------------------------------------------

class _TestNS(_BridgeNamespace):
    """``ops.test.<Type>(...)`` — Phase 3C typed methods."""

    def NormDispIncr(
        self,
        *,
        tol: float,
        max_iter: int,
        print_flag: int = 0,
        norm_type: int = 2,
    ) -> NormDispIncr:
        """``test NormDispIncr tol maxIter [pFlag normType]``."""
        return self._bridge._register(
            NormDispIncr(
                tol=tol,
                max_iter=max_iter,
                print_flag=print_flag,
                norm_type=norm_type,
            )
        )

    def NormUnbalance(
        self,
        *,
        tol: float,
        max_iter: int,
        print_flag: int = 0,
        norm_type: int = 2,
    ) -> NormUnbalance:
        """``test NormUnbalance tol maxIter [pFlag normType]``."""
        return self._bridge._register(
            NormUnbalance(
                tol=tol,
                max_iter=max_iter,
                print_flag=print_flag,
                norm_type=norm_type,
            )
        )

    def EnergyIncr(
        self,
        *,
        tol: float,
        max_iter: int,
        print_flag: int = 0,
        norm_type: int = 2,
    ) -> EnergyIncr:
        """``test EnergyIncr tol maxIter [pFlag normType]``."""
        return self._bridge._register(
            EnergyIncr(
                tol=tol,
                max_iter=max_iter,
                print_flag=print_flag,
                norm_type=norm_type,
            )
        )

    def FixedNumIter(
        self,
        *,
        max_iter: int,
        print_flag: int = 0,
        norm_type: int = 2,
    ) -> FixedNumIter:
        """``test FixedNumIter maxIter [pFlag normType]``."""
        return self._bridge._register(
            FixedNumIter(
                max_iter=max_iter,
                print_flag=print_flag,
                norm_type=norm_type,
            )
        )

    def RelativeNormDispIncr(
        self,
        *,
        tol: float,
        max_iter: int,
        print_flag: int = 0,
        norm_type: int = 2,
    ) -> RelativeNormDispIncr:
        """``test RelativeNormDispIncr tol maxIter [pFlag normType]``."""
        return self._bridge._register(
            RelativeNormDispIncr(
                tol=tol,
                max_iter=max_iter,
                print_flag=print_flag,
                norm_type=norm_type,
            )
        )


# ---------------------------------------------------------------------------
# _AlgorithmNS — ops.algorithm.<Type>(...)
# ---------------------------------------------------------------------------

class _AlgorithmNS(_BridgeNamespace):
    """``ops.algorithm.<Type>(...)`` — Phase 3C typed methods."""

    def Linear(self) -> AlgorithmLinear:
        """``algorithm Linear`` — one solve per step (no iteration)."""
        return self._bridge._register(AlgorithmLinear())

    def Newton(
        self, *, tangent: NewtonTangent = "tangent",
    ) -> Newton:
        """``algorithm Newton [-secant | -initial]`` — full Newton-Raphson."""
        return self._bridge._register(Newton(tangent=tangent))

    def ModifiedNewton(
        self, *, tangent: NewtonTangent = "tangent",
    ) -> ModifiedNewton:
        """``algorithm ModifiedNewton [-secant | -initial]``."""
        return self._bridge._register(ModifiedNewton(tangent=tangent))

    def NewtonLineSearch(
        self,
        *,
        line_search: LineSearchType,
        tol: float | None = None,
        max_iter: int | None = None,
        min_eta: float | None = None,
        max_eta: float | None = None,
    ) -> NewtonLineSearch:
        """``algorithm NewtonLineSearch -type T [-tol t] [-maxIter n]
        [-minEta v] [-maxEta v]``."""
        return self._bridge._register(
            NewtonLineSearch(
                line_search=line_search,
                tol=tol,
                max_iter=max_iter,
                min_eta=min_eta,
                max_eta=max_eta,
            )
        )

    def KrylovNewton(
        self,
        *,
        iterate: Literal["current", "initial", "noTangent"] | None = None,
        increment: Literal["current", "initial", "noTangent"] | None = None,
        max_dim: int | None = None,
    ) -> KrylovNewton:
        """``algorithm KrylovNewton [-iterate t] [-increment t] [-maxDim n]``."""
        return self._bridge._register(
            KrylovNewton(
                iterate=iterate,
                increment=increment,
                max_dim=max_dim,
            )
        )

    def BFGS(self, *, count: int | None = None) -> BFGS:
        """``algorithm BFGS [count]`` — quasi-Newton rank-2 update."""
        return self._bridge._register(BFGS(count=count))

    def Broyden(self, *, count: int | None = None) -> Broyden:
        """``algorithm Broyden [count]`` — quasi-Newton rank-1 update."""
        return self._bridge._register(Broyden(count=count))


# ---------------------------------------------------------------------------
# _IntegratorNS — ops.integrator.<Type>(...)
# ---------------------------------------------------------------------------

class _IntegratorNS(_BridgeNamespace):
    """``ops.integrator.<Type>(...)`` — Phase 3C typed methods."""

    def LoadControl(
        self,
        *,
        dlam: float,
        num_iter: int | None = None,
        min_lam: float | None = None,
        max_lam: float | None = None,
    ) -> LoadControl:
        """``integrator LoadControl dlam [numIter [minLam maxLam]]``."""
        return self._bridge._register(
            LoadControl(
                dlam=dlam,
                num_iter=num_iter,
                min_lam=min_lam,
                max_lam=max_lam,
            )
        )

    def DisplacementControl(
        self,
        *,
        node: int,
        dof: int,
        dU: float,
        num_iter: int | None = None,
        min_dU: float | None = None,
        max_dU: float | None = None,
    ) -> DisplacementControl:
        """``integrator DisplacementControl node dof dU
        [numIter [minDU maxDU]]``."""
        return self._bridge._register(
            DisplacementControl(
                node=node,
                dof=dof,
                dU=dU,
                num_iter=num_iter,
                min_dU=min_dU,
                max_dU=max_dU,
            )
        )

    def ArcLength(self, *, s: float, alpha: float) -> ArcLength:
        """``integrator ArcLength s alpha`` — arc-length method."""
        return self._bridge._register(ArcLength(s=s, alpha=alpha))

    def Newmark(self, *, gamma: float, beta: float) -> Newmark:
        """``integrator Newmark gamma beta`` — implicit transient."""
        return self._bridge._register(Newmark(gamma=gamma, beta=beta))

    def HHT(
        self,
        *,
        alpha: float,
        gamma: float | None = None,
        beta: float | None = None,
    ) -> HHT:
        """``integrator HHT alpha [gamma beta]`` — HHT alpha-method."""
        return self._bridge._register(
            HHT(alpha=alpha, gamma=gamma, beta=beta)
        )

    def CentralDifference(self) -> CentralDifference:
        """``integrator CentralDifference`` — explicit central difference."""
        return self._bridge._register(CentralDifference())

    def ExplicitDifference(self) -> ExplicitDifference:
        """``integrator ExplicitDifference`` — explicit difference scheme."""
        return self._bridge._register(ExplicitDifference())

    def ExplicitBathe(
        self,
        *,
        p: float = 0.54,
        cfl: bool = False,
        cfl_abort: bool = False,
        tangent: bool = False,
        recompute: int | None = None,
        lump: Lump | None = None,
        verbose: bool = False,
        divergence: float | None = None,
    ) -> ExplicitBathe:
        """``integrator ExplicitBathe p [flags...]`` — **fork-only**.

        Explicit Noh-Bathe two-sub-step scheme (``p`` ∈(0,1), default
        ``0.54``). Emission works on any build; the *Ladruno fork* is
        required only to *run* the deck. See
        :class:`apeGmsh.opensees.analysis.integrator.ExplicitBathe` for
        the option-flag semantics.
        """
        return self._bridge._register(
            ExplicitBathe(
                p=p,
                cfl=cfl,
                cfl_abort=cfl_abort,
                tangent=tangent,
                recompute=recompute,
                lump=lump,
                verbose=verbose,
                divergence=divergence,
            )
        )

    def ExplicitBatheLNVD(
        self,
        *,
        p: float = 0.54,
        alpha: float = 0.8,
        cfl: bool = False,
        cfl_abort: bool = False,
        tangent: bool = False,
        recompute: int | None = None,
        lump: Lump | None = None,
        verbose: bool = False,
        divergence: float | None = None,
    ) -> ExplicitBatheLNVD:
        """``integrator ExplicitBatheLNVD p alpha [flags...]`` — **fork-only**.

        :meth:`ExplicitBathe` plus FLAC local non-viscous damping
        (``alpha`` ∈[0,1), default ``0.80``) for dynamic relaxation /
        quasi-static solving. Fork required only to *run*.
        """
        return self._bridge._register(
            ExplicitBatheLNVD(
                p=p,
                alpha=alpha,
                cfl=cfl,
                cfl_abort=cfl_abort,
                tangent=tangent,
                recompute=recompute,
                lump=lump,
                verbose=verbose,
                divergence=divergence,
            )
        )

    def CentralDifferenceLadruno(
        self,
        *,
        cfl: bool = False,
        cfl_abort: bool = False,
        tangent: bool = False,
        recompute: int | None = None,
        lump: Lump | None = None,
        verbose: bool = False,
        divergence: float | None = None,
    ) -> CentralDifferenceLadruno:
        """``integrator CentralDifferenceLadruno [flags...]`` — **fork-only**.

        The *Ladruno fork*'s robust central-difference integrator
        (correct first-step starter, built-in ``dt_cr``, ``βK`` guard).
        No positional parameter; ``lump`` defaults to Diagonal when
        omitted. Fork required only to *run*.
        """
        return self._bridge._register(
            CentralDifferenceLadruno(
                cfl=cfl,
                cfl_abort=cfl_abort,
                tangent=tangent,
                recompute=recompute,
                lump=lump,
                verbose=verbose,
                divergence=divergence,
            )
        )

    def LadrunoArcLength(
        self,
        *,
        s: float,
        alpha: float,
        jd: int | None = None,
        ell_min: float | None = None,
        ell_max: float | None = None,
        p_exp: float | None = None,
        stabilize: bool = False,
        f_target: float | None = None,
        adapt_stab: bool = False,
        c_visc: float | None = None,
        mass: MassMode | None = None,
        mass_scale: float | None = None,
    ) -> LadrunoArcLength:
        """``integrator LadrunoArcLength s alpha [flags...]`` — **fork-only**.

        Adaptive / viscous-stabilized arc-length (superset of upstream
        ``ArcLength``; flag-free it is bit-identical). Supply the
        ``jd``/``ell_min``/``ell_max`` triple for Ramm adaptive radius, or
        ``stabilize=True`` (+ ``f_target``/``c_visc``/``mass``/
        ``adapt_stab``) for the viscous-regularized limit-point path. Fork
        required only to *run*. See
        :class:`apeGmsh.opensees.analysis.integrator.LadrunoArcLength`.
        """
        return self._bridge._register(
            LadrunoArcLength(
                s=s,
                alpha=alpha,
                jd=jd,
                ell_min=ell_min,
                ell_max=ell_max,
                p_exp=p_exp,
                stabilize=stabilize,
                f_target=f_target,
                adapt_stab=adapt_stab,
                c_visc=c_visc,
                mass=mass,
                mass_scale=mass_scale,
            )
        )

    def LadrunoDynamicRelaxation(
        self,
        *,
        mass: MassMode | None = None,
        mass_scale: float | None = None,
        dt: float | None = None,
        recompute: int | None = None,
        damping: DampingMode | None = None,
        zeta: float | None = None,
        no_auto_refresh: bool = False,
        interp: bool = False,
        divergence: float | None = None,
        verbose: bool = False,
    ) -> LadrunoDynamicRelaxation:
        """``integrator LadrunoDynamicRelaxation [flags...]`` — **fork-only**.

        Matrix-free dynamic relaxation to static equilibrium (no tangent
        factorization). Drive with ``ops.system.Diagonal()`` +
        ``ops.algorithm.Linear()`` + ``ops.analysis.Transient()``. Fork
        required only to *run*. See
        :class:`apeGmsh.opensees.analysis.integrator.LadrunoDynamicRelaxation`.
        """
        return self._bridge._register(
            LadrunoDynamicRelaxation(
                mass=mass,
                mass_scale=mass_scale,
                dt=dt,
                recompute=recompute,
                damping=damping,
                zeta=zeta,
                no_auto_refresh=no_auto_refresh,
                interp=interp,
                divergence=divergence,
                verbose=verbose,
            )
        )

    def LadrunoIndirectControl(
        self,
        *,
        incr: float,
        controls: Sequence[tuple[int, int, float]],
        num_iter: int | None = None,
        dmin: float | None = None,
        dmax: float | None = None,
    ) -> LadrunoIndirectControl:
        """``integrator LadrunoIndirectControl incr -dof ...`` — **fork-only**.

        Weighted multi-DOF indirect displacement control (monotone
        ``c^T U`` through snap-back). ``controls`` is a non-empty sequence
        of ``(node, dof, coef)`` entries (1-based ``dof``). Fork required
        only to *run*. See
        :class:`apeGmsh.opensees.analysis.integrator.LadrunoIndirectControl`.
        """
        return self._bridge._register(
            LadrunoIndirectControl(
                incr=incr,
                controls=tuple(controls),
                num_iter=num_iter,
                dmin=dmin,
                dmax=dmax,
            )
        )


# ---------------------------------------------------------------------------
# _AnalysisNS — ops.analysis.<Type>()
# ---------------------------------------------------------------------------

class _AnalysisNS(_BridgeNamespace):
    """``ops.analysis.<Type>()`` — Phase 3C typed methods."""

    def Static(self) -> Static:
        """``analysis Static`` — static (pseudo-time)."""
        return self._bridge._register(Static())

    def Transient(self) -> Transient:
        """``analysis Transient`` — fixed-step transient."""
        return self._bridge._register(Transient())

    def VariableTransient(self) -> VariableTransient:
        """``analysis VariableTransient`` — adaptive-step transient."""
        return self._bridge._register(VariableTransient())
