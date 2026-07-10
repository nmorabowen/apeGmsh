"""
Typed ``algorithm`` (solution algorithm) primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``algorithm <Type> ...`` command. The
matching :class:`apeGmsh.opensees._internal.ns.analysis._AlgorithmNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

Solution algorithms are singletons in OpenSees (no tag in the
command). The ``tag`` parameter to :meth:`_emit` is consumed by the
allocator but not rendered in the emitted command.

OpenSees command shapes::

    algorithm Linear             [-secant | -initial] [-factorOnce]
    algorithm Newton             [-secant | -initial | -intialThenCurrent
                                  | -hall [iFactor cFactor]]
    algorithm ModifiedNewton     [-secant | -initial | -factorOnce
                                  | -hall [iFactor cFactor]]
    algorithm NewtonLineSearch -type type [-tol tol] [-maxIter n]
                              [-minEta v] [-maxEta v]
    algorithm KrylovNewton       [-iterate t] [-increment t] [-maxDim n]
    algorithm BFGS               [count]
    algorithm Broyden            [count]

The ``-secant`` / ``-initial`` flags select which tangent the Newton
family uses. They are mutually exclusive (no tangent and one tangent
flag are both valid; two tangent flags are not). The dataclasses
expose them as a single ``tangent`` enum-string field.

The ``-factorOnce`` flag tells the algorithm to form + factor the
system tangent only once (on the first iteration) and re-use that
factorization for the rest of the analysis — a large speedup for
linear-elastic transient runs where the tangent never changes. On
``Linear`` the OpenSees parser loops over its options, so ``-factorOnce``
combines freely with a ``-secant`` / ``-initial`` tangent flag. On
``ModifiedNewton`` the parser reads a *single* option token, so
``factor_once`` is mutually exclusive with a non-default ``tangent``
(supplying both would silently drop one); the dataclass rejects that
combination. The emitted spelling is ``-FactorOnce`` — the one casing
accepted by *both* the ``Linear`` (``-factorOnce`` / ``-FactorOnce``) and
the ``ModifiedNewton`` (``-factoronce`` / ``-FactorOnce``) parsers on the
openseespy and Tcl paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .._internal.types import Primitive, SolutionAlgorithm

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Linear",
    "Newton",
    "ModifiedNewton",
    "NewtonLineSearch",
    "KrylovNewton",
    "BFGS",
    "Broyden",
]


# -- Shared types ------------------------------------------------------------

NewtonTangent = Literal["tangent", "secant", "initial"]
"""Base tangent-stiffness flavor (``Linear``, and the core Newton family).

* ``"tangent"`` — emit no flag (OpenSees default: current tangent).
* ``"secant"`` — emit ``-secant``.
* ``"initial"`` — emit ``-initial``.
"""

NewtonRaphsonTangent = Literal[
    "tangent", "secant", "initial", "initialThenCurrent", "hall",
]
"""Tangent flavor for full :class:`Newton`. Superset of :data:`NewtonTangent`:

* ``"initialThenCurrent"`` — predict with the initial tangent, correct with
  the current one (emits ``-intialThenCurrent`` — note the upstream OpenSees
  typo the parser actually matches).
* ``"hall"`` — Hall tangent-blending; ``hall_i_factor`` / ``hall_c_factor``
  weight the initial vs. current tangent (emits ``-hall [iFactor cFactor]``).
"""

ModifiedNewtonTangent = Literal["tangent", "secant", "initial", "hall"]
"""Tangent flavor for :class:`ModifiedNewton` (``Newton`` minus
``initialThenCurrent``, which the ModifiedNewton parser does not accept)."""

LineSearchType = Literal["Bisection", "Secant", "RegulaFalsi", "InitialInterpolated"]
"""Line-search algorithm selector for ``NewtonLineSearch``."""

# Upstream OpenSees matches these exact option tokens (``-intial…`` is a
# real typo in the C++ parser — do NOT "correct" it or the flag stops
# matching). We expose the correctly-spelled value in the Python API and
# map it here.
_TANGENT_FLAG: dict[str, str] = {
    "secant": "-secant",
    "initial": "-initial",
    "initialThenCurrent": "-intialThenCurrent",
    "hall": "-hall",
}


def _validate_hall(
    who: str,
    tangent: str,
    i_factor: float | None,
    c_factor: float | None,
) -> None:
    """Validate the optional Hall tangent-blending factors."""
    if (i_factor is None) != (c_factor is None):
        raise ValueError(
            f"{who}: supply both hall_i_factor and hall_c_factor, or neither "
            f"(got hall_i_factor={i_factor!r}, hall_c_factor={c_factor!r})."
        )
    if i_factor is not None and tangent != "hall":
        raise ValueError(
            f"{who}: hall_i_factor/hall_c_factor require tangent='hall' "
            f"(got tangent={tangent!r})."
        )


def _tangent_args(
    tangent: str,
    i_factor: float | None,
    c_factor: float | None,
) -> list[float | str]:
    """Render the tangent flag (+ trailing Hall factors) for the Newton family.

    The Hall factors are emitted last — the OpenSees ``-hall`` parser reads
    them only when they are the final two tokens.
    """
    args: list[float | str] = []
    flag = _TANGENT_FLAG.get(tangent)
    if flag is not None:
        args.append(flag)
    if tangent == "hall" and i_factor is not None:
        assert c_factor is not None  # _validate_hall guarantee
        args += [i_factor, c_factor]
    return args


# ---------------------------------------------------------------------------
# Linear — single iteration per step (linear analyses)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Linear(SolutionAlgorithm):
    """``algorithm Linear [-secant | -initial] [-factorOnce]``.

    One solve per step (no iteration). ``tangent`` selects the stiffness
    flavor as in :class:`Newton`. ``factor_once=True`` forms + factors the
    tangent only on the first step and re-uses that factorization — a large
    speedup for linear-elastic transient runs. The OpenSees ``Linear``
    parser loops its options, so ``factor_once`` combines freely with a
    non-default ``tangent``.
    """

    tangent: NewtonTangent = "tangent"
    factor_once: bool = False

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[str] = []
        if self.tangent == "secant":
            args.append("-secant")
        elif self.tangent == "initial":
            args.append("-initial")
        if self.factor_once:
            args.append("-FactorOnce")
        emitter.algorithm("Linear", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Newton — full Newton-Raphson
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Newton(SolutionAlgorithm):
    """``algorithm Newton [-secant | -initial | -intialThenCurrent
    | -hall [iFactor cFactor]]`` — full Newton-Raphson.

    ``tangent`` selects the stiffness flavor: the OpenSees default
    ("current tangent") if ``"tangent"``, ``-secant`` for the secant
    stiffness, ``-initial`` for the initial-tangent (initial-stiffness
    Newton), ``"initialThenCurrent"`` to predict with the initial tangent
    then correct with the current one, or ``"hall"`` for Hall
    tangent-blending. With ``tangent="hall"`` the optional
    ``hall_i_factor`` / ``hall_c_factor`` weight the initial vs. current
    tangent (OpenSees defaults ``0.1`` / ``0.9`` when omitted); supply both
    or neither.
    """

    tangent: NewtonRaphsonTangent = "tangent"
    hall_i_factor: float | None = None
    hall_c_factor: float | None = None

    def __post_init__(self) -> None:
        _validate_hall(
            "Newton", self.tangent, self.hall_i_factor, self.hall_c_factor
        )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args = _tangent_args(
            self.tangent, self.hall_i_factor, self.hall_c_factor
        )
        emitter.algorithm("Newton", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ModifiedNewton — re-uses one tangent across a step
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ModifiedNewton(SolutionAlgorithm):
    """``algorithm ModifiedNewton [-secant | -initial | -factorOnce
    | -hall [iFactor cFactor]]``.

    Forms one tangent at the start of each step and re-uses it for
    every iteration in that step. ``tangent`` semantics match
    :class:`Newton` (minus ``initialThenCurrent``, which the ModifiedNewton
    parser does not accept); with ``tangent="hall"`` the optional
    ``hall_i_factor`` / ``hall_c_factor`` weight the initial vs. current
    tangent. ``factor_once=True`` additionally re-uses the *first*
    factorization across the whole analysis (``-factorOnce``).

    The OpenSees ``ModifiedNewton`` parser reads a *single* option token,
    so ``factor_once`` is mutually exclusive with a non-default
    ``tangent`` — supplying both would silently drop one, so the dataclass
    rejects that combination.
    """

    tangent: ModifiedNewtonTangent = "tangent"
    factor_once: bool = False
    hall_i_factor: float | None = None
    hall_c_factor: float | None = None

    def __post_init__(self) -> None:
        if self.factor_once and self.tangent != "tangent":
            raise ValueError(
                "ModifiedNewton: factor_once cannot be combined with "
                f"tangent={self.tangent!r} — the OpenSees ModifiedNewton "
                "parser reads a single option token, so one flag would be "
                "silently dropped. Use factor_once with the default "
                "(current) tangent."
            )
        _validate_hall(
            "ModifiedNewton", self.tangent,
            self.hall_i_factor, self.hall_c_factor,
        )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args = _tangent_args(
            self.tangent, self.hall_i_factor, self.hall_c_factor
        )
        if not args and self.factor_once:
            args = ["-FactorOnce"]
        emitter.algorithm("ModifiedNewton", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# NewtonLineSearch — Newton with a one-dimensional line search per iteration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class NewtonLineSearch(SolutionAlgorithm):
    """``algorithm NewtonLineSearch -type T [-tol t] [-maxIter n]
    [-minEta v] [-maxEta v]``.

    ``line_search`` (required) picks the 1-D search algorithm. The
    other keys are optional and emit only when set.
    """

    line_search: LineSearchType
    tol: float | None = None
    max_iter: int | None = None
    min_eta: float | None = None
    max_eta: float | None = None

    def __post_init__(self) -> None:
        if self.tol is not None and self.tol <= 0:
            raise ValueError(
                f"NewtonLineSearch: tol must be > 0, got {self.tol}"
            )
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError(
                "NewtonLineSearch: max_iter must be >= 1, "
                f"got {self.max_iter}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = ["-type", self.line_search]
        if self.tol is not None:
            args += ["-tol", self.tol]
        if self.max_iter is not None:
            args += ["-maxIter", self.max_iter]
        if self.min_eta is not None:
            args += ["-minEta", self.min_eta]
        if self.max_eta is not None:
            args += ["-maxEta", self.max_eta]
        emitter.algorithm("NewtonLineSearch", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# KrylovNewton — Newton with a Krylov-subspace acceleration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class KrylovNewton(SolutionAlgorithm):
    """``algorithm KrylovNewton [-iterate t] [-increment t] [-maxDim n]``.

    The two ``-iterate`` / ``-increment`` flags pick which tangent is
    kept across a step (``"current"``, ``"initial"``, or ``"noTangent"``).
    ``max_dim`` caps the size of the Krylov subspace.
    """

    iterate: Literal["current", "initial", "noTangent"] | None = None
    increment: Literal["current", "initial", "noTangent"] | None = None
    max_dim: int | None = None

    def __post_init__(self) -> None:
        if self.max_dim is not None and self.max_dim < 1:
            raise ValueError(
                f"KrylovNewton: max_dim must be >= 1, got {self.max_dim}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: list[float | int | str] = []
        if self.iterate is not None:
            args += ["-iterate", self.iterate]
        if self.increment is not None:
            args += ["-increment", self.increment]
        if self.max_dim is not None:
            args += ["-maxDim", self.max_dim]
        emitter.algorithm("KrylovNewton", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# BFGS — quasi-Newton with BFGS rank-2 update
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class BFGS(SolutionAlgorithm):
    """``algorithm BFGS [count]`` — quasi-Newton with BFGS rank-2 update.

    ``count`` caps the number of stored rank-2 updates per step.
    """

    count: int | None = None

    def __post_init__(self) -> None:
        if self.count is not None and self.count < 1:
            raise ValueError(
                f"BFGS: count must be >= 1, got {self.count}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.count is None:
            emitter.algorithm("BFGS")
        else:
            emitter.algorithm("BFGS", self.count)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Broyden — quasi-Newton with Broyden rank-1 update
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Broyden(SolutionAlgorithm):
    """``algorithm Broyden [count]`` — quasi-Newton with rank-1 update.

    ``count`` caps the number of stored rank-1 updates per step.
    """

    count: int | None = None

    def __post_init__(self) -> None:
        if self.count is not None and self.count < 1:
            raise ValueError(
                f"Broyden: count must be >= 1, got {self.count}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        if self.count is None:
            emitter.algorithm("Broyden")
        else:
            emitter.algorithm("Broyden", self.count)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
