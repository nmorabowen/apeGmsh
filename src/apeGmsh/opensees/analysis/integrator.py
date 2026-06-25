"""
Typed ``integrator`` primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``integrator <Type> ...`` command. The
matching :class:`apeGmsh.opensees._internal.ns.analysis._IntegratorNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

Integrators are singletons in OpenSees (no tag in the command). The
``tag`` parameter to :meth:`_emit` is consumed by the allocator but
not rendered in the emitted command.

OpenSees command shapes::

    integrator LoadControl          dlam [num_iter [min_lam max_lam]]
    integrator DisplacementControl  node dof dU [num_iter [min_dU max_dU]]
    integrator ArcLength            s alpha
    integrator Newmark              gamma beta
    integrator HHT                  alpha [gamma beta]
    integrator CentralDifference
    integrator ExplicitDifference
    integrator LadrunoArcLength          s alpha [-adapt Jd ellMin ellMax]
                                         [-p exp] [-stabilize f] [-adaptStab]
                                         [-cVisc c] [-mass mode [scale]]
    integrator LadrunoDynamicRelaxation  [-mass mode [scale]] [-dt dt]
                                         [-recompute N] [-damping mode [zeta]]
                                         [-noAutoRefresh] [-interp]
                                         [-divergence f] [-verbose]
    integrator LadrunoIndirectControl    incr -dof node dof coef [-dof ...]
                                         [-iter numIter [dMin dMax]]

The ``min_*`` / ``max_*`` step-bracket parameters on LoadControl and
DisplacementControl are only meaningful in tandem with ``num_iter``;
the dataclasses reject "min/max set but num_iter unset" at
construction.

Six integrators are **fork-only** — they require the OpenSees *Ladruno
fork* build to *run*. Emission works on any build (it's just an
``integrator <Type> ...`` line); the fork requirement bites only at
``ops.analyze(...)`` / ``ops.run()``:

* The three *explicit* schemes :class:`ExplicitBathe`,
  :class:`ExplicitBatheLNVD` and :class:`CentralDifferenceLadruno` share
  an order-free option grammar (``-cfl``/``-cflAbort``/``-tangent``/
  ``-recompute N``/``-lump rowsum|diagonal``/``-verbose``/``-divergence
  f``); see :func:`_render_explicit_cfl_options`.
* The two static *path-following* integrators
  :class:`LadrunoArcLength` (adaptive / viscous-stabilized arc-length)
  and :class:`LadrunoIndirectControl` (weighted multi-DOF control),
  plus the matrix-free quasi-static :class:`LadrunoDynamicRelaxation`.

.. note:: openseespy greedy-read quirk

   The fork's ``-stabilize [f]``, ``-mass lumped [scale]`` and
   ``-damping viscous [zeta]`` options take an *optional* trailing
   number. On the openseespy parser ``PythonModule::getDouble``
   increments the arg cursor *before* the type check, so a failed
   optional read **consumes** the following token (a ``-flag`` would be
   silently eaten). To stay robust on both the Tcl and Python paths
   these emitters **always render the explicit trailing value** (the
   fork default when the user leaves it unset), so a greedy flag is
   never emitted bare-then-followed-by-another-flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .._internal.types import Integrator, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "LoadControl",
    "DisplacementControl",
    "ArcLength",
    "Newmark",
    "HHT",
    "CentralDifference",
    "ExplicitDifference",
    "ExplicitBathe",
    "ExplicitBatheLNVD",
    "CentralDifferenceLadruno",
    "LadrunoArcLength",
    "LadrunoDynamicRelaxation",
    "LadrunoIndirectControl",
    "LadrunoHHT",
    "LadrunoGeneralizedAlpha",
]


# ---------------------------------------------------------------------------
# Shared explicit-integrator option grammar (Ladruno fork)
# ---------------------------------------------------------------------------

Lump = Literal["rowsum", "diagonal"]

#: Artificial-mass construction mode for the fork's matrix-free path-following
#: integrators (:class:`LadrunoArcLength` stabilization, :class:`LadrunoDynamicRelaxation`).
MassMode = Literal["gershgorin", "lumped", "unity"]

#: Dynamic-relaxation damping mode (:class:`LadrunoDynamicRelaxation`).
DampingMode = Literal["kinetic", "viscous"]


def _validate_explicit_cfl_options(
    *,
    who: str,
    recompute: int | None,
    lump: Lump | None,
    divergence: float | None,
) -> None:
    """Validate the shared explicit-integrator option flags.

    Enforces only what the fork C++ itself rejects / treats as an
    authoring slip — no cross-flag coupling (the C++ parser is
    permissive: e.g. ``-cflAbort`` without a ``dt_cr`` source is
    silently inert, not an error).
    """
    if recompute is not None and recompute < 1:
        raise ValueError(
            f"{who}: recompute must be >= 1 (every-N committed steps), "
            f"got {recompute}."
        )
    if lump is not None and lump not in ("rowsum", "diagonal"):
        raise ValueError(
            f"{who}: lump must be 'rowsum' or 'diagonal', got {lump!r}."
        )
    if divergence is not None and divergence <= 0:
        raise ValueError(
            f"{who}: divergence factor must be > 0, got {divergence}."
        )


def _render_explicit_cfl_options(
    args: list[float | int | str],
    *,
    cfl: bool,
    cfl_abort: bool,
    tangent: bool,
    recompute: int | None,
    lump: Lump | None,
    verbose: bool,
    divergence: float | None,
) -> None:
    """Append the shared explicit-integrator flags to ``args`` in a
    fixed, byte-stable canonical order.

    Omitted ``lump`` is left to the fork's per-integrator default
    (RowSum for the Bathe schemes, Diagonal for
    ``CentralDifferenceLadruno``) — we never re-emit a default.
    """
    if cfl:
        args.append("-cfl")
    if cfl_abort:
        args.append("-cflAbort")
    if tangent:
        args.append("-tangent")
    if recompute is not None:
        args += ["-recompute", recompute]
    if lump is not None:
        args += ["-lump", lump]
    if verbose:
        args.append("-verbose")
    if divergence is not None:
        args += ["-divergence", divergence]


# ---------------------------------------------------------------------------
# LoadControl — static, prescribed load-factor increment
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LoadControl(Integrator):
    """``integrator LoadControl dlam [num_iter [min_lam max_lam]]``.

    Static analysis with a fixed (or adaptively bracketed) load-factor
    increment. ``dlam`` is the nominal increment per step; ``num_iter``
    is the target convergence-iteration count used to scale the
    increment adaptively when supplied.
    """

    dlam: float
    num_iter: int | None = None
    min_lam: float | None = None
    max_lam: float | None = None

    def __post_init__(self) -> None:
        if self.num_iter is not None and self.num_iter < 1:
            raise ValueError(
                f"LoadControl: num_iter must be >= 1, got {self.num_iter}"
            )
        if (self.min_lam is None) != (self.max_lam is None):
            raise ValueError(
                "LoadControl: supply both min_lam and max_lam, or "
                f"neither (got min_lam={self.min_lam!r}, "
                f"max_lam={self.max_lam!r})."
            )
        if self.min_lam is not None and self.num_iter is None:
            raise ValueError(
                "LoadControl: min_lam/max_lam require num_iter to be set."
            )
        if (
            self.min_lam is not None
            and self.max_lam is not None
            and self.min_lam > self.max_lam
        ):
            raise ValueError(
                "LoadControl: min_lam must be <= max_lam, got "
                f"min_lam={self.min_lam}, max_lam={self.max_lam}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int] = [self.dlam]
        if self.num_iter is not None:
            args.append(self.num_iter)
            if self.min_lam is not None:
                assert self.max_lam is not None  # __post_init__ guarantee
                args += [self.min_lam, self.max_lam]
        emitter.integrator("LoadControl", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# DisplacementControl — static, prescribed displacement increment at one DOF
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class DisplacementControl(Integrator):
    """``integrator DisplacementControl node dof dU
    [num_iter [min_dU max_dU]]``.

    Static analysis driven by a prescribed displacement increment
    ``dU`` at ``node``'s ``dof``-th DOF. ``num_iter`` enables adaptive
    bracketing as in :class:`LoadControl`.
    """

    node: int
    dof: int
    dU: float
    num_iter: int | None = None
    min_dU: float | None = None
    max_dU: float | None = None

    def __post_init__(self) -> None:
        if self.dof < 1:
            raise ValueError(
                f"DisplacementControl: dof must be >= 1, got {self.dof}"
            )
        if self.num_iter is not None and self.num_iter < 1:
            raise ValueError(
                "DisplacementControl: num_iter must be >= 1, "
                f"got {self.num_iter}"
            )
        if (self.min_dU is None) != (self.max_dU is None):
            raise ValueError(
                "DisplacementControl: supply both min_dU and max_dU, or "
                f"neither (got min_dU={self.min_dU!r}, "
                f"max_dU={self.max_dU!r})."
            )
        if self.min_dU is not None and self.num_iter is None:
            raise ValueError(
                "DisplacementControl: min_dU/max_dU require num_iter "
                "to be set."
            )
        if (
            self.min_dU is not None
            and self.max_dU is not None
            and self.min_dU > self.max_dU
        ):
            raise ValueError(
                "DisplacementControl: min_dU must be <= max_dU, got "
                f"min_dU={self.min_dU}, max_dU={self.max_dU}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int] = [self.node, self.dof, self.dU]
        if self.num_iter is not None:
            args.append(self.num_iter)
            if self.min_dU is not None:
                assert self.max_dU is not None  # __post_init__ guarantee
                args += [self.min_dU, self.max_dU]
        emitter.integrator("DisplacementControl", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ArcLength — static arc-length method
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ArcLength(Integrator):
    """``integrator ArcLength s alpha``.

    Arc-length continuation. ``s`` is the arc-length increment per step;
    ``alpha`` weights the load contribution to the arc-length norm.
    """

    s: float
    alpha: float

    def __post_init__(self) -> None:
        if self.s <= 0:
            raise ValueError(
                f"ArcLength: s must be > 0, got {self.s}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.integrator("ArcLength", self.s, self.alpha)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Newmark — implicit transient (the standard structural choice)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Newmark(Integrator):
    """``integrator Newmark gamma beta``.

    The classical Newmark scheme. ``gamma=0.5, beta=0.25`` recovers
    the unconditionally stable average-acceleration variant; the user
    is responsible for selecting parameters consistent with their
    accuracy + stability requirements.
    """

    gamma: float
    beta: float

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.integrator("Newmark", self.gamma, self.beta)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# HHT — Hilber-Hughes-Taylor alpha-method
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class HHT(Integrator):
    """``integrator HHT alpha [gamma beta]``.

    Hilber-Hughes-Taylor alpha-method. Supplying ``gamma`` and
    ``beta`` overrides the OpenSees defaults derived from ``alpha``;
    omit both to use the defaults. Either supply both or neither.
    """

    alpha: float
    gamma: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        if (self.gamma is None) != (self.beta is None):
            raise ValueError(
                "HHT: supply both gamma and beta, or neither "
                f"(got gamma={self.gamma!r}, beta={self.beta!r})."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.gamma is None:
            emitter.integrator("HHT", self.alpha)
        else:
            assert self.beta is not None  # __post_init__ guarantee
            emitter.integrator("HHT", self.alpha, self.gamma, self.beta)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# CentralDifference — explicit transient (no parameters)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class CentralDifference(Integrator):
    """``integrator CentralDifference`` — explicit central-difference."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.integrator("CentralDifference")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ExplicitDifference — explicit transient (no parameters)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ExplicitDifference(Integrator):
    """``integrator ExplicitDifference`` — explicit difference scheme."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.integrator("ExplicitDifference")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ExplicitBathe — explicit Noh-Bathe two-sub-step (Ladruno fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ExplicitBathe(Integrator):
    """``integrator ExplicitBathe p [flags...]`` — **fork-only**.

    The Noh-Bathe two-sub-step explicit scheme (2nd order, controllable
    high-frequency dissipation via ``p``). Requires the OpenSees
    *Ladruno fork* build to run; emission works on any build.

    ``p`` is the sub-step parameter ``∈(0,1)`` (default ``0.54``). The
    option flags enable + tune the critical-time-step (``dt_cr``)
    machinery and per-step diagnostics:

    * ``cfl`` — estimate ``dt_cr`` (queryable via ``criticalTimeStep()``).
    * ``cfl_abort`` — abort if ``dt`` exceeds the Noh-Bathe limit
      (inert unless a ``dt_cr`` source — ``cfl``/``tangent``/``recompute``
      — is also enabled).
    * ``tangent`` — estimate ``dt_cr`` from the current tangent.
    * ``recompute`` — refresh ``dt_cr`` every N committed steps (N >= 1).
    * ``lump`` — element mass lumping for ``dt_cr`` (default RowSum).
    * ``verbose`` — per-step dt/energy reporting.
    * ``divergence`` — abort if kinetic energy grows by this factor.
    """

    p: float = 0.54
    cfl: bool = False
    cfl_abort: bool = False
    tangent: bool = False
    recompute: int | None = None
    lump: Lump | None = None
    verbose: bool = False
    divergence: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(
                f"ExplicitBathe: p must be in (0, 1), got {self.p}."
            )
        _validate_explicit_cfl_options(
            who="ExplicitBathe",
            recompute=self.recompute,
            lump=self.lump,
            divergence=self.divergence,
        )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = [self.p]
        _render_explicit_cfl_options(
            args,
            cfl=self.cfl,
            cfl_abort=self.cfl_abort,
            tangent=self.tangent,
            recompute=self.recompute,
            lump=self.lump,
            verbose=self.verbose,
            divergence=self.divergence,
        )
        emitter.integrator("ExplicitBathe", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ExplicitBatheLNVD — Noh-Bathe + FLAC local non-viscous damping (fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ExplicitBatheLNVD(Integrator):
    """``integrator ExplicitBatheLNVD p alpha [flags...]`` — **fork-only**.

    :class:`ExplicitBathe` plus FLAC local non-viscous damping, for
    dynamic relaxation / quasi-static solving. Requires the *Ladruno
    fork* build to run.

    ``p`` is the sub-step parameter ``∈(0,1)`` (default ``0.54``);
    ``alpha`` is the FLAC local-damping coefficient ``∈[0,1)`` (default
    ``0.80``; ``0`` disables damping). Both are always emitted
    explicitly — the fork reads them as a *pair* of leading numerics, so
    eliding ``alpha`` would shift a following flag into its slot. The
    option flags are identical to :class:`ExplicitBathe`.
    """

    p: float = 0.54
    alpha: float = 0.8
    cfl: bool = False
    cfl_abort: bool = False
    tangent: bool = False
    recompute: int | None = None
    lump: Lump | None = None
    verbose: bool = False
    divergence: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(
                f"ExplicitBatheLNVD: p must be in (0, 1), got {self.p}."
            )
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(
                "ExplicitBatheLNVD: alpha (FLAC damping) must be in "
                f"[0, 1), got {self.alpha}."
            )
        _validate_explicit_cfl_options(
            who="ExplicitBatheLNVD",
            recompute=self.recompute,
            lump=self.lump,
            divergence=self.divergence,
        )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = [self.p, self.alpha]
        _render_explicit_cfl_options(
            args,
            cfl=self.cfl,
            cfl_abort=self.cfl_abort,
            tangent=self.tangent,
            recompute=self.recompute,
            lump=self.lump,
            verbose=self.verbose,
            divergence=self.divergence,
        )
        emitter.integrator("ExplicitBatheLNVD", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# CentralDifferenceLadruno — robust central difference (fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class CentralDifferenceLadruno(Integrator):
    """``integrator CentralDifferenceLadruno [flags...]`` — **fork-only**.

    The *Ladruno fork*'s robust central-difference integrator: a correct
    first-step starter, built-in ``dt_cr``, and a ``βK`` guard. Requires
    the fork build to run.

    Takes no positional parameter; the option flags match
    :class:`ExplicitBathe`, except ``lump`` defaults to **Diagonal**
    (diagonal-of-consistent) rather than RowSum when omitted. (The
    dropped *coupled* mode is served by ``NewmarkExplicit 0.5`` — out of
    scope here.)
    """

    cfl: bool = False
    cfl_abort: bool = False
    tangent: bool = False
    recompute: int | None = None
    lump: Lump | None = None
    verbose: bool = False
    divergence: float | None = None

    def __post_init__(self) -> None:
        _validate_explicit_cfl_options(
            who="CentralDifferenceLadruno",
            recompute=self.recompute,
            lump=self.lump,
            divergence=self.divergence,
        )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = []
        _render_explicit_cfl_options(
            args,
            cfl=self.cfl,
            cfl_abort=self.cfl_abort,
            tangent=self.tangent,
            recompute=self.recompute,
            lump=self.lump,
            verbose=self.verbose,
            divergence=self.divergence,
        )
        emitter.integrator("CentralDifferenceLadruno", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoArcLength — adaptive / viscous-stabilized arc-length (fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoArcLength(Integrator):
    """``integrator LadrunoArcLength s alpha [flags...]`` — **fork-only**.

    The *Ladruno fork*'s adaptive / stabilized arc-length integrator — a
    strict superset of upstream :class:`ArcLength`. With no flags it is
    bit-identical to stock ``ArcLength``; the flags add a Ramm adaptive
    radius and a viscous stabilization path. Emission works on any build;
    the fork is required only to *run*.

    ``s`` is the arc-length increment (> 0), ``alpha`` weights the load
    contribution to the constraint norm.

    **Layer A — Ramm adaptive radius** (``-adapt Jd ellMin ellMax``):
    supply ``jd``, ``ell_min`` and ``ell_max`` *together* to rescale the
    radius each step toward ``jd`` desired corrector iterations, clamped
    to ``[ell_min, ell_max]``. ``p_exp`` (``-p``) is the adaptation
    exponent (``1.0`` LoadControl-style, ``0.5`` gentler Crisfield).

    **Viscous stabilization** (``stabilize=True``, a code path *disjoint*
    from the arc-length quadratic): regularizes the tangent with an
    artificial dashpot so ordinary Newton clears limit points.
    ``f_target`` is the target dissipated-energy fraction (fork default
    ``2e-4``); ``c_visc`` sets the dashpot coefficient directly instead;
    ``adapt_stab`` drives the *load* increment adaptively; ``mass``
    selects the artificial mass (``gershgorin`` row-sum, ``lumped`` real
    mass scaled by ``mass_scale``, or ``unity``). The artificial-mass /
    dashpot knobs (``c_visc``, ``mass``, ``mass_scale``, ``adapt_stab``,
    ``f_target``) require ``stabilize=True``. **Does not follow true
    snap-back.**
    """

    s: float
    alpha: float
    # Layer A — Ramm adaptive radius (-adapt Jd ellMin ellMax / -p exp)
    jd: int | None = None
    ell_min: float | None = None
    ell_max: float | None = None
    p_exp: float | None = None
    # Viscous stabilization (-stabilize [f] / -adaptStab / -cVisc / -mass)
    stabilize: bool = False
    f_target: float | None = None
    adapt_stab: bool = False
    c_visc: float | None = None
    mass: MassMode | None = None
    mass_scale: float | None = None

    def __post_init__(self) -> None:
        if self.s <= 0:
            raise ValueError(f"LadrunoArcLength: s must be > 0, got {self.s}")
        # -adapt is an all-or-none triple
        adapt_triple = (self.jd, self.ell_min, self.ell_max)
        n_set = sum(v is not None for v in adapt_triple)
        if n_set not in (0, 3):
            raise ValueError(
                "LadrunoArcLength: supply all of jd, ell_min, ell_max (the "
                "-adapt triple) or none "
                f"(got jd={self.jd!r}, ell_min={self.ell_min!r}, "
                f"ell_max={self.ell_max!r})."
            )
        if self.jd is not None:
            assert self.ell_min is not None and self.ell_max is not None
            if self.jd < 1:
                raise ValueError(
                    f"LadrunoArcLength: jd must be >= 1, got {self.jd}."
                )
            if self.ell_min < 0:
                raise ValueError(
                    "LadrunoArcLength: ell_min must be >= 0, got "
                    f"{self.ell_min}."
                )
            if self.ell_min > self.ell_max:
                raise ValueError(
                    "LadrunoArcLength: ell_min must be <= ell_max, got "
                    f"ell_min={self.ell_min}, ell_max={self.ell_max}."
                )
        if self.p_exp is not None and self.p_exp <= 0:
            raise ValueError(
                f"LadrunoArcLength: p_exp must be > 0, got {self.p_exp}."
            )
        # Stabilization-only knobs require stabilize=True (otherwise inert).
        if not self.stabilize:
            stray = [
                name for name, val in (
                    ("f_target", self.f_target),
                    ("c_visc", self.c_visc),
                    ("mass", self.mass),
                    ("mass_scale", self.mass_scale),
                ) if val is not None
            ]
            if self.adapt_stab:
                stray.append("adapt_stab")
            if stray:
                raise ValueError(
                    "LadrunoArcLength: "
                    f"{', '.join(sorted(stray))} require stabilize=True "
                    "(they only affect the viscous-stabilization path)."
                )
        if self.f_target is not None and self.f_target <= 0:
            raise ValueError(
                f"LadrunoArcLength: f_target must be > 0, got {self.f_target}."
            )
        if self.c_visc is not None and self.c_visc <= 0:
            raise ValueError(
                f"LadrunoArcLength: c_visc must be > 0, got {self.c_visc}."
            )
        if self.mass is not None and self.mass not in (
            "gershgorin", "lumped", "unity",
        ):
            raise ValueError(
                "LadrunoArcLength: mass must be 'gershgorin', 'lumped' or "
                f"'unity', got {self.mass!r}."
            )
        if self.mass_scale is not None:
            if self.mass != "lumped":
                raise ValueError(
                    "LadrunoArcLength: mass_scale only applies to "
                    f"mass='lumped' (got mass={self.mass!r})."
                )
            if self.mass_scale <= 0:
                raise ValueError(
                    "LadrunoArcLength: mass_scale must be > 0, got "
                    f"{self.mass_scale}."
                )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = [self.s, self.alpha]
        if self.jd is not None:
            assert self.ell_min is not None and self.ell_max is not None
            args += ["-adapt", self.jd, self.ell_min, self.ell_max]
        if self.p_exp is not None:
            args += ["-p", self.p_exp]
        if self.stabilize:
            # Always render the explicit f after -stabilize (fork default
            # 2e-4 when unset): the openseespy parser would otherwise eat a
            # following -flag on the optional read (see module note).
            args += ["-stabilize", 2.0e-4 if self.f_target is None
                     else self.f_target]
            if self.adapt_stab:
                args.append("-adaptStab")
            if self.c_visc is not None:
                args += ["-cVisc", self.c_visc]
            if self.mass is not None:
                args += ["-mass", self.mass]
                if self.mass == "lumped":
                    # Always render the lumped scale (fork default 1.0) — same
                    # greedy-read defense as -stabilize above.
                    args.append(
                        1.0 if self.mass_scale is None else self.mass_scale
                    )
        emitter.integrator("LadrunoArcLength", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoDynamicRelaxation — matrix-free quasi-static (fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoDynamicRelaxation(Integrator):
    """``integrator LadrunoDynamicRelaxation [flags...]`` — **fork-only**.

    The *Ladruno fork*'s matrix-free dynamic-relaxation integrator: drives
    a model to static equilibrium by integrating a fictitious damped
    transient to rest, never assembling or factorizing the tangent — so it
    sails through the limit points and indefinite tangents that defeat
    implicit Newton. Drive it with ``ops.system.Diagonal()`` +
    ``ops.algorithm.Linear()`` + ``ops.analysis.Transient()``. Emission
    works on any build; the fork is required only to *run*.

    Takes no positional parameter. Options:

    * ``mass`` — fictitious-mass construction: ``gershgorin`` (scale-free
      Gershgorin row-sum, the robust default), ``lumped`` (real element
      mass scaled by ``mass_scale``), or ``unity``.
    * ``dt`` — pseudo time step (fork default ``1.0``, safe with
      ``gershgorin``).
    * ``recompute`` — refresh the fictitious mass every ``N`` committed
      steps (``N >= 1``; omit to keep the one-shot mass — the fork default
      0 = no refresh). Use under strong softening.
    * ``damping`` — ``kinetic`` (parameter-free Cundall velocity zeroing
      at kinetic-energy peaks, the default) or ``viscous`` (mass-
      proportional, ratio ``zeta``).
    * ``no_auto_refresh`` — disable the Gershgorin mass refresh at KE peaks.
    * ``interp`` — enable the fork's interpolated velocity output.
    * ``divergence`` — abort if kinetic energy grows by this factor in a
      step.
    * ``verbose`` — per-step ``max|a|`` / residual / KE reporting.
    """

    mass: MassMode | None = None
    mass_scale: float | None = None
    dt: float | None = None
    recompute: int | None = None
    damping: DampingMode | None = None
    zeta: float | None = None
    no_auto_refresh: bool = False
    interp: bool = False
    divergence: float | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.mass is not None and self.mass not in (
            "gershgorin", "lumped", "unity",
        ):
            raise ValueError(
                "LadrunoDynamicRelaxation: mass must be 'gershgorin', "
                f"'lumped' or 'unity', got {self.mass!r}."
            )
        if self.mass_scale is not None:
            if self.mass != "lumped":
                raise ValueError(
                    "LadrunoDynamicRelaxation: mass_scale only applies to "
                    f"mass='lumped' (got mass={self.mass!r})."
                )
            if self.mass_scale <= 0:
                raise ValueError(
                    "LadrunoDynamicRelaxation: mass_scale must be > 0, got "
                    f"{self.mass_scale}."
                )
        if self.dt is not None and self.dt <= 0:
            raise ValueError(
                f"LadrunoDynamicRelaxation: dt must be > 0, got {self.dt}."
            )
        if self.recompute is not None and self.recompute < 1:
            raise ValueError(
                "LadrunoDynamicRelaxation: recompute must be >= 1 (every-N "
                f"committed steps), got {self.recompute}."
            )
        if self.damping is not None and self.damping not in (
            "kinetic", "viscous",
        ):
            raise ValueError(
                "LadrunoDynamicRelaxation: damping must be 'kinetic' or "
                f"'viscous', got {self.damping!r}."
            )
        if self.zeta is not None:
            if self.damping != "viscous":
                raise ValueError(
                    "LadrunoDynamicRelaxation: zeta only applies to "
                    f"damping='viscous' (got damping={self.damping!r})."
                )
            if self.zeta <= 0:
                raise ValueError(
                    "LadrunoDynamicRelaxation: zeta must be > 0, got "
                    f"{self.zeta}."
                )
        if self.divergence is not None and self.divergence <= 0:
            raise ValueError(
                "LadrunoDynamicRelaxation: divergence factor must be > 0, "
                f"got {self.divergence}."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = []
        if self.mass is not None:
            args += ["-mass", self.mass]
            if self.mass == "lumped":
                # Always render the lumped scale (fork default 1.0) — the
                # openseespy parser would eat a following -flag otherwise.
                args.append(
                    1.0 if self.mass_scale is None else self.mass_scale
                )
        if self.dt is not None:
            args += ["-dt", self.dt]
        if self.recompute is not None:
            args += ["-recompute", self.recompute]
        if self.damping is not None:
            args += ["-damping", self.damping]
            if self.damping == "viscous":
                # Always render the trailing zeta (fork default 1.0).
                args.append(1.0 if self.zeta is None else self.zeta)
        if self.no_auto_refresh:
            args.append("-noAutoRefresh")
        if self.interp:
            args.append("-interp")
        if self.divergence is not None:
            args += ["-divergence", self.divergence]
        if self.verbose:
            args.append("-verbose")
        emitter.integrator("LadrunoDynamicRelaxation", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoIndirectControl — weighted multi-DOF control (fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoIndirectControl(Integrator):
    """``integrator LadrunoIndirectControl incr -dof ... [-iter ...]`` —
    **fork-only**.

    The *Ladruno fork*'s indirect-displacement-control integrator —
    generalizes upstream ``DisplacementControl`` from a single nodal DOF
    to a **weighted multi-DOF control quantity** ``zeta = c^T U`` that
    stays monotone through snap-back even when individual DOFs reverse
    (e.g. crack-mouth-opening ``u_A - u_B``). Emission works on any build;
    the fork is required only to *run*.

    ``incr`` is the control-quantity increment per step. ``controls`` is a
    non-empty sequence of ``(node, dof, coef)`` participation entries (the
    1-based ``dof`` matches :class:`DisplacementControl`); each emits a
    ``-dof node dof coef`` group. Optional Ramm adaptation via
    ``num_iter`` (``-iter``) scales the increment toward ``num_iter``
    desired iterations, optionally clamped to ``[dmin, dmax]``.
    """

    incr: float
    controls: tuple[tuple[int, int, float], ...]
    num_iter: int | None = None
    dmin: float | None = None
    dmax: float | None = None

    def __post_init__(self) -> None:
        if not self.controls:
            raise ValueError(
                "LadrunoIndirectControl: needs at least one (node, dof, "
                "coef) control entry."
            )
        for entry in self.controls:
            if len(entry) != 3:
                raise ValueError(
                    "LadrunoIndirectControl: each control entry must be "
                    f"(node, dof, coef), got {entry!r}."
                )
            _node, dof, _coef = entry
            if dof < 1:
                raise ValueError(
                    "LadrunoIndirectControl: dof must be >= 1 (1-based), "
                    f"got {dof} in entry {entry!r}."
                )
        if self.num_iter is not None and self.num_iter < 1:
            raise ValueError(
                "LadrunoIndirectControl: num_iter must be >= 1, got "
                f"{self.num_iter}."
            )
        if (self.dmin is None) != (self.dmax is None):
            raise ValueError(
                "LadrunoIndirectControl: supply both dmin and dmax, or "
                f"neither (got dmin={self.dmin!r}, dmax={self.dmax!r})."
            )
        if self.dmin is not None and self.num_iter is None:
            raise ValueError(
                "LadrunoIndirectControl: dmin/dmax require num_iter to be set."
            )
        if (
            self.dmin is not None
            and self.dmax is not None
            and self.dmin > self.dmax
        ):
            raise ValueError(
                "LadrunoIndirectControl: dmin must be <= dmax, got "
                f"dmin={self.dmin}, dmax={self.dmax}."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = [self.incr]
        for node, dof, coef in self.controls:
            args += ["-dof", node, dof, coef]
        if self.num_iter is not None:
            args += ["-iter", self.num_iter]
            if self.dmin is not None:
                assert self.dmax is not None  # __post_init__ guarantee
                args += [self.dmin, self.dmax]
        emitter.integrator("LadrunoIndirectControl", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoHHT — DDM-sensitivity HHT-alpha (Ladruno fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoHHT(Integrator):
    """``integrator LadrunoHHT alpha [gamma beta]`` — **fork-only**.

    The *Ladruno fork*'s sensitivity-carrying Hilber-Hughes-Taylor
    alpha-method (``INTEGRATOR_TAG`` 33013). The primal (non-sensitivity)
    path is byte-identical to stock :class:`HHT`; the fork adds the
    consistent direct-differentiation (DDM) sensitivity methods for
    reliability / optimization. Supply both ``gamma`` and ``beta`` or
    neither — omitted, the fork derives ``gamma = 1.5 - alpha`` and
    ``beta = (2 - alpha)^2 / 4``. Emission works on any build; the fork is
    required only to *run*.
    """

    alpha: float
    gamma: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        if (self.gamma is None) != (self.beta is None):
            raise ValueError(
                "LadrunoHHT: supply both gamma and beta, or neither "
                f"(got gamma={self.gamma!r}, beta={self.beta!r})."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.gamma is None:
            emitter.integrator("LadrunoHHT", self.alpha)
        else:
            assert self.beta is not None  # __post_init__ guarantee
            emitter.integrator(
                "LadrunoHHT", self.alpha, self.gamma, self.beta)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# LadrunoGeneralizedAlpha — DDM-sensitivity generalized-alpha (Ladruno fork)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class LadrunoGeneralizedAlpha(Integrator):
    """``integrator LadrunoGeneralizedAlpha alphaM alphaF [gamma beta]`` — **fork-only**.

    The *Ladruno fork*'s sensitivity-carrying Chung-Hulbert generalized-alpha
    integrator (``INTEGRATOR_TAG`` 33014). The primal path matches stock
    ``GeneralizedAlpha``; during sensitivity solves the fork re-forms the
    tangent with the consistent ``c3*M`` mass coefficient (stock emits an
    inconsistent ``alphaM*c3*M`` Jacobian the sensitivity residual can't
    match). Supply both ``gamma`` and ``beta`` or neither — omitted, the fork
    derives ``gamma = 0.5 + alphaM - alphaF`` and
    ``beta = (1 + alphaM - alphaF)^2 / 4``. Emission works on any build; the
    fork is required only to *run*.
    """

    alpha_m: float
    alpha_f: float
    gamma: float | None = None
    beta: float | None = None

    def __post_init__(self) -> None:
        if (self.gamma is None) != (self.beta is None):
            raise ValueError(
                "LadrunoGeneralizedAlpha: supply both gamma and beta, or "
                f"neither (got gamma={self.gamma!r}, beta={self.beta!r})."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.gamma is None:
            emitter.integrator(
                "LadrunoGeneralizedAlpha", self.alpha_m, self.alpha_f)
        else:
            assert self.beta is not None  # __post_init__ guarantee
            emitter.integrator(
                "LadrunoGeneralizedAlpha",
                self.alpha_m, self.alpha_f, self.gamma, self.beta,
            )

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
