"""
Typed ``system`` primitives — Phase 3C.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``system <Type>`` command. The matching
:class:`apeGmsh.opensees._internal.ns.analysis._SystemNS` methods take
no parameters (the seven core variants ship as flag-only) and call
``self._bridge._register(Cls())``.

Linear-system solvers are singletons in OpenSees (no tag in the
command). The ``tag`` parameter to :meth:`_emit` is consumed by the
allocator but not rendered.

OpenSees command shapes::

    system BandGeneral
    system BandSPD
    system ProfileSPD
    system SProfileSPD
    system UmfPack
    system Pardiso <-matrixType 0|1|2> <-stats>
    system Mumps <-ICNTL14 n> <-ICNTL7 n> <-matrixType 0|1|2>
                 <-BLR eps> <-ICNTL35 n> <-CNTL7 v>
                 <-commSplit color> <-stats>
    system SparseGeneral
    system SparseSYM
    system FullGeneral
    system Diagonal
    system MPIDiagonal
    system ParallelProfileSPD

``Pardiso`` is fork-only *and* build-conditional: the Ladruno fork
registers it only when the build found Intel MKL (fork ADR-75 P1,
fork PR #623), and only on the serial/desktop targets. Everything
else here emits on any build.

``Diagonal`` is the solver paired with explicit time integration
(``CentralDifference`` / the Ladruno ``ExplicitBathe`` family): it
inverts a lumped, diagonal mass matrix element-wise. It cannot solve a
coupled stiffness system, so it is **not** valid for static / implicit
analysis. ``MPIDiagonal`` / ``ParallelProfileSPD`` are parallel-only —
they require an OpenSees build with ``_PARALLEL_INTERPRETERS`` (e.g.
``OpenSeesMP``); emission works anywhere but a serial run errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import LinearSystem, Primitive

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "BandGeneral",
    "BandSPD",
    "ProfileSPD",
    "SProfileSPD",
    "UmfPack",
    "Pardiso",
    "Mumps",
    "SparseGeneral",
    "SparseSYM",
    "FullGeneral",
    "Diagonal",
    "MPIDiagonal",
    "ParallelProfileSPD",
]


# ---------------------------------------------------------------------------
# BandGeneral — banded, general (non-symmetric)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class BandGeneral(LinearSystem):
    """``system BandGeneral`` — banded general (non-symmetric) solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("BandGeneral")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# BandSPD — banded, symmetric positive-definite
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class BandSPD(LinearSystem):
    """``system BandSPD`` — banded symmetric-positive-definite solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("BandSPD")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ProfileSPD — variable-bandwidth (skyline) symmetric positive-definite
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ProfileSPD(LinearSystem):
    """``system ProfileSPD`` — skyline symmetric-positive-definite solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("ProfileSPD")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# UmfPack — direct sparse LU (UMFPACK)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class UmfPack(LinearSystem):
    """``system UmfPack`` — direct sparse LU via UMFPACK."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("UmfPack")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Pardiso — threaded sparse LU (Intel MKL PARDISO), fork + MKL only
# ---------------------------------------------------------------------------

#: ``matrix_type`` -> the ``-matrixType`` integer.  ``system Pardiso`` and
#: ``system Mumps`` deliberately share this numbering (fork ADR-75 P1d): 0
#: unsymmetric (full storage), 1 symmetric positive-definite, 2 symmetric
#: general.  For PARDISO these map onto MKL ``mtype`` 11 / 2 / -2; for MUMPS
#: onto ``SYM`` 0 / 1 / 2.
_MATRIX_TYPES: "dict[str, int]" = {
    "unsymmetric": 0,
    "spd": 1,
    "symmetric": 2,
}


@dataclass(frozen=True, kw_only=True, slots=True)
class Pardiso(LinearSystem):
    """``system Pardiso`` — shared-memory threaded sparse LU (Intel MKL).

    The **desktop** counterpart of :class:`Mumps`: same unsymmetric
    storage as :class:`UmfPack` (MKL matrix type 11, real + unsymmetric),
    but the factorization runs on several threads.

    **The win compounds with model size** — it is not a constant factor.
    On the fork's 3-D solid benchmark at four threads: **1.61× UmfPack at
    11.5k DOF, 2.15× at 26k, 3.40× at 51k**, because UmfPack scales
    ~O(N²) while PARDISO scales ~O(N^1.45) (fork ADR-75 P1c). Results are
    bit-identical to UmfPack (rel err 0.0) at every size and thread count.

    **It also raises the ceiling, which matters more than the ratio.**
    UmfPack ran *out of memory* at 86,490 DOF on the same machine;
    PARDISO solved that model in 30.4 s and a 136,080-DOF one in 68.6 s.
    Memory — not time — is what stops a large 3-D direct solve, so a
    model that previously forced the cluster may now fit on a
    workstation. (The wall is machine-RAM-dependent; its *ordering* is
    the durable result. Largest-model ceiling untested.)

    Thread count is **not** a parameter: MKL reads it from
    ``MKL_NUM_THREADS`` / ``OMP_NUM_THREADS``, which must be set in the
    environment *before* the process starts (before ``import openseespy``
    for a live run, or in the job script for a Tcl / HPC run). Four is
    the recommended desktop value.

    Only the factor/solve is threaded — assembly and element state
    determination stay serial — so the win lands on solve-bound models
    (3-D solids, large brick meshes) and not on element-bound ones
    (fiber frames). Since the solve fraction *grows* with N, the bigger
    the solid model the more this is the right choice.

    Availability: **Ladruno fork built with Intel MKL, serial targets
    only** (fork PRs #623 / #624 / #630 — before #630 the verb existed in
    ``openseespy`` but not in ``OpenSees.exe``). Stock OpenSees, a
    non-MKL fork build, and the MPI targets all reject the command with
    "unknown system type" — emission works anywhere, running does not.
    Under ``OpenSeesMP`` use :class:`Mumps`; PARDISO is a serial SOE with
    no distributed assembly.

    Parameters
    ----------
    matrix_type
        Storage / factorization mode (fork ADR-75 P1d, ``-matrixType``).

        * ``"unsymmetric"`` (default) — full CSR, MKL mtype 11. The only
          universally safe choice.
        * ``"symmetric"`` — upper-triangle half-storage, mtype −2
          (LDLᵀ with Bunch–Kaufman pivoting). **1.96× UmfPack and 42%
          less peak memory** on the fork's Lane-B benchmark, bit-identical
          results. The right symmetric choice for structural tangents.
        * ``"spd"`` — half-storage, mtype 2 (Cholesky). Within 0.7% of
          ``"symmetric"`` on time, but it **fails outright on an
          indefinite tangent**, which any softening or buckling model
          eventually has. Prefer ``"symmetric"``.

        ⚠ **Half-storage reads only the ``col >= row`` half of each
        element matrix — no averaging, no detection.** On a genuinely
        unsymmetric tangent it silently solves a *different system*. This
        fork produces those in production: ``LadrunoContact``,
        non-associated flow, follower loads, ``LadrunoUP``, and concrete
        plastic-damage. Only pick a symmetric mode when you know the
        tangent is symmetric. apeGmsh refuses the combination outright
        for ``LadrunoUP`` (ADR 0074 D4).
    krylov
        Emit ``-krylov <L>`` (fork ADR-75 P1e): reuse the previous
        factorization as a **CGS preconditioner** instead of refactorizing
        every iteration, stopping at a residual of ``10**-L`` (the fork's
        recipe recommends ``6``). The payoff exists only under **full
        Newton on solve-bound models** — a further **1.51× at 51k DOF** on
        the fork's benchmark — because full Newton is what refactorizes
        per iteration; under ``ModifiedNewton``/``KrylovNewton`` there is
        nothing to save. Below ~25k DOF the bookkeeping cancels the win.

        Not available with ``matrix_type="symmetric"`` (Intel documents
        the CG reuse for SPD only; the fork warns and falls back to a
        direct solve — apeGmsh refuses the combination instead).

        ⚠ Near limit points the inexact solve can change **which
        post-peak equilibrium branch** a ``LoadControl`` continuation
        lands on (fork recipe trap 5). Keep it off when the deliverable
        is a post-peak path, not just the peak.
    stats
        Emit ``-stats``: dump PARDISO factor nnz and peak memory once per
        sparsity pattern. The only way to see whether a model fits.

    .. warning::
        **Threaded PARDISO is not byte-reproducible run-to-run** (fork
        recipe trap 7): at ``MKL_NUM_THREADS`` > 1 the reduction order
        varies, and repeated identical runs differ by ~1 ULP. Pin
        ``MKL_NUM_THREADS=1`` for byte-identical regression baselines;
        otherwise compare with a relative tolerance (1e-12 is generous).
    """

    matrix_type: str = "unsymmetric"
    krylov: int | None = None
    stats: bool = False

    def __post_init__(self) -> None:
        if self.matrix_type not in _MATRIX_TYPES:
            raise ValueError(
                f"Pardiso: matrix_type={self.matrix_type!r} is not one of "
                f"{sorted(_MATRIX_TYPES)}. Note the fork DEGRADES a "
                f"bad -matrixType to unsymmetric with only a warning (and a "
                f"string-valued one silently fell back to ProfileSPD before "
                f"fork PR #630), so apeGmsh validates it here instead."
            )
        if self.krylov is not None:
            if isinstance(self.krylov, bool) or not isinstance(
                self.krylov, int
            ):
                raise ValueError(
                    f"Pardiso: krylov={self.krylov!r} — the CGS stopping "
                    f"exponent L (residual 10**-L) must be an int."
                )
            if self.krylov < 1:
                raise ValueError(
                    f"Pardiso: krylov={self.krylov!r} — the CGS stopping "
                    f"exponent L must be >= 1."
                )
            if self.matrix_type == "symmetric":
                raise ValueError(
                    "Pardiso: krylov= is not available with "
                    "matrix_type='symmetric' (MKL documents the CGS "
                    "factorization reuse for mtype 11 and 2 only; the fork "
                    "warns and silently falls back to a direct solve — "
                    "fork ADR-75 P1e). Drop krylov= or use "
                    "matrix_type='unsymmetric'/'spd'."
                )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: "list[int | float | str]" = []
        code = _MATRIX_TYPES[self.matrix_type]
        # Always emit -matrixType, including 0 (unsymmetric). A bare
        # ``if code:`` treats 0 as false and silently drops the flag, so
        # ``Pardiso(matrix_type="unsymmetric")`` looked like an unset
        # default in the deck. Emit an INT, never a str: the fork parses
        # with OPS_GetIntInput (fork PR #630).
        args += ["-matrixType", code]
        if self.krylov is not None:
            args += ["-krylov", int(self.krylov)]
        if self.stats:
            args.append("-stats")
        emitter.system("Pardiso", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Mumps — multifrontal massively-parallel sparse direct solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Mumps(LinearSystem):
    """``system Mumps <options>`` — MUMPS multifrontal sparse solver.

    The cluster/MPI counterpart of :class:`Pardiso`, and the system to
    use under ``OpenSeesMP``. Optional knobs are emitted only when set;
    ``matrix_type`` is always emitted (including ``0`` for the default
    unsymmetric mode) so the deck does not hide the storage choice.

    Parameters
    ----------
    blr
        Block Low-Rank dropping tolerance (fork ADR-75 P2, ``-BLR eps``;
        sets ``ICNTL(35)=1`` and ``CNTL(7)=eps``). ``None`` (default)
        leaves BLR off and the factorization exact.

        ⚠ **BLR is an APPROXIMATE factorization** — it must stay off on
        byte-identical / oracle lanes. And the fork's own measurements
        (#626, ~32k DOF on 2 ranks) found it a win on **no** axis at that
        scale: at ``1e-9`` it was 1.70× slower with +8.4% peak memory; at
        ``1e-4`` it shrank stored factors 21.8% but moved *peak* memory
        only −4.6% while running 3.17× slower — peak is dominated by the
        active frontal space, not by what is stored. In a nonlinear loop
        a looser tolerance also returns a less accurate correction, so
        Newton needs more iterations. The crossover to production-size
        fronts is **untested**; measure with ``stats=True`` before
        trusting it.
    stats
        Emit ``-stats``: dump MUMPS ``INFOG(9)/(21)/(22)``, ``RINFOG(3)``
        (and the BLR ``RINFOG(14)/(15)`` when BLR is on) after each
        numeric factorization on rank 0 (fork ADR-75 P2b). The only way
        to see per-rank factor memory.

    icntl14
        ``ICNTL(14)`` — the percentage by which MUMPS grows its working
        space over the estimate (fork default 20). **The knob to raise on
        an out-of-memory / workspace failure**; 100–200 is routine on hard
        3-D factorizations.
    icntl7
        ``ICNTL(7)`` — the sequential ordering MUMPS uses when it does not
        pick one itself (fork default 7 = automatic; 5 = METIS,
        4 = PORD, 2 = AMF, 0 = AMD).
    matrix_type
        Storage mode, same numbering as :class:`Pardiso` — ``ICNTL SYM``
        0 / 1 / 2. ``"unsymmetric"`` (default) is the only universally
        safe choice; the symmetric modes read only half of each element
        matrix and silently solve a *different* system on an unsymmetric
        tangent (contact, non-associated flow, follower loads,
        ``LadrunoUP``, concrete plastic-damage). apeGmsh refuses the
        combination outright for ``LadrunoUP`` (ADR 0074 D4).
    icntl35, cntl7
        Raw BLR controls, for driving compression directly instead of via
        ``blr``. ``icntl35`` selects the BLR variant (1 = factorization +
        solve); ``cntl7`` is the dropping tolerance. Mutually exclusive
        with ``blr``, which is sugar for exactly this pair.
    comm_split
        ``-commSplit <color>`` (fork ADR-43 P3a): split
        ``MPI_COMM_WORLD`` by colour so this rank's MUMPS factor/solve
        runs on a sub-communicator; ranks sharing a colour form one solve
        group. ⚠ ``MPI_Comm_split`` is **collective over the world
        communicator** — *every* rank must pass some colour or the run
        deadlocks. Must be >= 0.
    """

    icntl14: int | None = None
    icntl7: int | None = None
    matrix_type: str = "unsymmetric"
    blr: float | None = None
    icntl35: int | None = None
    cntl7: float | None = None
    comm_split: int | None = None
    stats: bool = False

    def __post_init__(self) -> None:
        if self.matrix_type not in _MATRIX_TYPES:
            raise ValueError(
                f"Mumps: matrix_type={self.matrix_type!r} is not one of "
                f"{sorted(_MATRIX_TYPES)}."
            )
        # Every bad value below makes the fork's parser `return 0`, leaving
        # theSOE null — OpenSees then falls back to the ProfileSPD default
        # with only a warning, so the run looks merely slow rather than
        # wrong (fork ADR-75 P1d). Refuse up front instead.
        if self.blr is not None and self.blr < 0.0:
            raise ValueError(
                f"Mumps: blr={self.blr!r} — the BLR dropping tolerance must "
                f"be >= 0."
            )
        if self.cntl7 is not None and self.cntl7 < 0.0:
            raise ValueError(
                f"Mumps: cntl7={self.cntl7!r} — the BLR dropping tolerance "
                f"must be >= 0."
            )
        if self.blr is not None and (
            self.icntl35 is not None or self.cntl7 is not None
        ):
            raise ValueError(
                "Mumps: blr= is sugar for icntl35=1 + cntl7=<eps>, so "
                "combining it with icntl35=/cntl7= is ambiguous — the fork "
                "parses options left-to-right and the last one silently "
                "wins. Pass either blr= or the raw pair."
            )
        if self.comm_split is not None and self.comm_split < 0:
            raise ValueError(
                f"Mumps: comm_split={self.comm_split!r} — the colour must be "
                f">= 0."
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        args: "list[int | float | str]" = []
        if self.icntl14 is not None:
            args += ["-ICNTL14", int(self.icntl14)]
        if self.icntl7 is not None:
            args += ["-ICNTL7", int(self.icntl7)]
        code = _MATRIX_TYPES[self.matrix_type]
        # Always emit -matrixType, including 0 (unsymmetric). Same
        # falsy-0 trap as Pardiso — ``if code:`` would omit the flag.
        # An INT, never a str — the fork parses with OPS_GetIntInput.
        args += ["-matrixType", code]
        if self.blr is not None:
            args += ["-BLR", float(self.blr)]
        if self.icntl35 is not None:
            args += ["-ICNTL35", int(self.icntl35)]
        if self.cntl7 is not None:
            args += ["-CNTL7", float(self.cntl7)]
        if self.comm_split is not None:
            args += ["-commSplit", int(self.comm_split)]
        if self.stats:
            args.append("-stats")
        emitter.system("Mumps", *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# SparseGeneral — generic sparse general (SuperLU-backed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class SparseGeneral(LinearSystem):
    """``system SparseGeneral`` — generic sparse general solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("SparseGeneral")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# FullGeneral — dense, non-symmetric (small-system fallback)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class FullGeneral(LinearSystem):
    """``system FullGeneral`` — dense general solver (small systems only)."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("FullGeneral")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# SProfileSPD — single-precision skyline symmetric positive-definite
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class SProfileSPD(LinearSystem):
    """``system SProfileSPD`` — single-precision skyline SPD solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("SProfileSPD")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# SparseSYM — symmetric sparse direct solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class SparseSYM(LinearSystem):
    """``system SparseSYM`` — symmetric sparse direct solver."""

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("SparseSYM")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Diagonal — direct diagonal solver (explicit time integration)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Diagonal(LinearSystem):
    """``system Diagonal`` — direct diagonal solver.

    The system paired with explicit time integration
    (``CentralDifference`` / Ladruno ``ExplicitBathe`` family): it
    inverts a lumped, diagonal mass matrix element-wise, with no global
    factorization. It only uses the diagonal of the assembled matrix, so
    it is **not** valid for static / implicit analysis (a coupled
    stiffness solve will fail to converge).
    """

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("Diagonal")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# MPIDiagonal — parallel diagonal solver
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class MPIDiagonal(LinearSystem):
    """``system MPIDiagonal`` — distributed diagonal solver.

    Parallel counterpart of :class:`Diagonal` for explicit runs under
    ``OpenSeesMP``. Only available in OpenSees builds with
    ``_PARALLEL_INTERPRETERS``; a serial build falls back to the plain
    diagonal solver. Emission works on any build.
    """

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("MPIDiagonal")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# ParallelProfileSPD — distributed skyline symmetric positive-definite
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ParallelProfileSPD(LinearSystem):
    """``system ParallelProfileSPD`` — distributed skyline SPD solver.

    Parallel counterpart of :class:`ProfileSPD`. Only available in
    OpenSees builds with ``_PARALLEL_INTERPRETERS`` (e.g.
    ``OpenSeesMP``); emit-only against a serial ``OpenSees.exe`` will
    error at runtime.
    """

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _ = tag
        emitter.system("ParallelProfileSPD")

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
