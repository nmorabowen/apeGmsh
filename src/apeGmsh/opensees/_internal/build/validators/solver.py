"""solver: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations

from typing import Iterable, Sequence

from ...types import Element
from ..errors import BridgeError


#: Linear systems a LadrunoUP deck may declare (ADR 0074 D4 — the fork
#: guide §2 allow-list): general (unsymmetric-storage) solvers only.
#: Serial: UmfPack / SparseGeneral (SuperLU) / FullGeneral / BandGeneral /
#: Pardiso (MKL matrix type 11 — real and unsymmetric, same storage as
#: UmfPack); MPI: Mumps (apeGmsh's typed Mumps emits ``-matrixType 0``,
#: SYM=0 — general).  Everything else either
#: stores only one triangle (BandSPD / ProfileSPD / SProfileSPD /
#: ParallelProfileSPD / SparseSYM — one Q coupling block is silently
#: dropped at assembly) or solves only the diagonal (Diagonal /
#: MPIDiagonal — ALL coupling dropped).  Shared by every gate that needs
#: a solver able to hold a genuinely unsymmetric tangent: the u-p one
#: below, and the Manzari consistent-tangent one after it.
_UNSYMMETRIC_SAFE_SYSTEMS: frozenset[str] = frozenset(
    {"UmfPack", "SparseGeneral", "FullGeneral", "BandGeneral", "Mumps",
     "Pardiso"}
)


_GENERAL_SOLVER_MSG = (
    "ops.system.UmfPack() (serial first choice), Pardiso (fork + MKL, "
    "threaded), SparseGeneral, FullGeneral, BandGeneral, or Mumps "
    "(MPI, SYM=0)"
)


def validate_ladruno_up_solver(
    elements: "Iterable[Element]",
    *,
    enforce: bool,
    staged: bool,
    partitioned: bool,
    flat_systems: "Sequence[object]",
    stage_systems: "Sequence[tuple[str, object | None]]",
) -> None:
    """ADR 0074 D4 — refuse to build a LadrunoUP deck that WILL SOLVE on a
    wrong (or absent) linear system.

    The honest-p contract makes the effective tangent UNSYMMETRIC (``−Q``
    in the u-rows' stiffness, ``+Qᵀ`` in the p-rows' damp): symmetric-
    storage solvers keep only the upper triangle, silently discard one
    coupling block at assembly, and return plausible-looking garbage
    (measured p ≈ 1e88 with every ``analyze()`` returning 0 on ProfileSPD
    — the no-``system``-command default).  No framework hook lets an
    element reject an SOE, and the fork can only print a notice — this
    gate actually stops the run.  Deliberately NO escape hatch.

    Scope (the gate validates what the deck will EMIT-and-SOLVE, not merely
    what was registered):

    * ``enforce`` is False for emits that never drive a u-p solve — H5
      archival, model-only export with no analysis chain, and eigen-only
      decks — so those are skipped entirely (the caller decides).
    * **staged**: each stage owns its analysis chain and ``wipeAnalysis``
      re-defaults the SOE to ProfileSPD between stages, so EVERY stage that
      analyzes must declare a legal system of its own; a globally-registered
      system is never emitted in staged mode and is ignored (fixes the
      stray-global false-reject).  A stage with ``system=None`` runs on the
      ProfileSPD default → raise, naming the stage (fixes the mixed
      declared/undeclared-stage false-accept).
    * **flat**: the EFFECTIVE (last-declared) system is the one OpenSees
      uses at analyze; it must be legal.  With no system declared, a
      **partitioned** deck is fine — ADR-0027 INV-5 auto-emits a general
      ``Mumps``/``UmfPack`` fallback — while a serial flat analysis deck
      raises the no-system footgun.
    """
    from ....element.solid import LadrunoUP

    up_pg = next(
        (str(getattr(spec, "pg", "?"))
         for spec in elements if isinstance(spec, LadrunoUP)),
        None,
    )
    if up_pg is None:
        return

    def _check(system: object, where: str) -> None:
        token = type(system).__name__
        if token not in _UNSYMMETRIC_SAFE_SYSTEMS:
            raise BridgeError(
                f"LadrunoUP (pg {up_pg!r}) with system {token!r} ({where}): "
                f"the honest-p tangent is UNSYMMETRIC and {token} does not "
                f"store the full matrix — one u-p coupling block is silently "
                f"dropped at assembly and the run returns plausible garbage "
                f"pore pressures (measured ~1e88 with rc=0; fork guide §2). "
                f"Declare a general solver: {_GENERAL_SOLVER_MSG}."
            )
        # Pardiso / Mumps are legal only in their UNSYMMETRIC mode (fork
        # ADR-75 P1d).  A class-name-only check would pass
        # `Pardiso(matrix_type="symmetric")` straight through — and
        # half-storage reads only the col >= row half of each element matrix,
        # which is exactly the silent coupling-block drop this gate exists to
        # stop.
        mtype = getattr(system, "matrix_type", "unsymmetric")
        if mtype != "unsymmetric":
            raise BridgeError(
                f"LadrunoUP (pg {up_pg!r}) with system {token}"
                f"(matrix_type={mtype!r}) ({where}): the honest-p tangent is "
                f"UNSYMMETRIC, but this mode stores only the upper triangle. "
                f"{token} would read half of each element matrix — no "
                f"averaging, no detection — and silently solve a DIFFERENT "
                f"system, returning garbage pore pressures with rc=0 (fork "
                f"ADR-75 P1d / guide §2). Use ops.system.{token}() with its "
                f"default matrix_type='unsymmetric'."
            )

    # A DECLARED symmetric/diagonal system is unambiguously wrong with u-p —
    # validate it whether or not THIS emit runs analyze (so a model-only Tcl
    # export still catches `ops.system.ProfileSPD()`).  The MISSING-system
    # branch, by contrast, is gated on ``enforce`` (an archival / eigen-only /
    # model-only-skeleton emit that never solves must not be refused for a
    # solver it never needs).
    if staged:
        for name, system in stage_systems:
            if system is not None:
                _check(system, f"stage {name}")
        if enforce:
            for name, system in stage_systems:
                if system is None:
                    raise BridgeError(
                        f"LadrunoUP (pg {up_pg!r}): stage {name} analyzes with "
                        f"no linear system, so it runs on the OpenSees "
                        f"ProfileSPD default after the prior stage's "
                        f"wipeAnalysis — which silently drops a u-p coupling "
                        f"block and returns garbage pore pressures (fork guide "
                        f"§2). Give every stage its own general solver in "
                        f"s.analysis(system=...): {_GENERAL_SOLVER_MSG}."
                    )
        return

    # Flat: the last-declared system is the effective one at analyze time.
    if flat_systems:
        _check(flat_systems[-1], "global")
        return
    if not enforce:
        # Archival / eigen-only / model-only skeleton — never solves, so a
        # missing system is not (yet) a footgun.
        return
    if partitioned:
        # ADR-0027 INV-5 auto-emits a general Mumps/UmfPack fallback.
        return
    raise BridgeError(
        f"LadrunoUP (pg {up_pg!r}) with no linear system declared: the "
        f"no-`system`-command OpenSees default is ProfileSPD, which silently "
        f"drops one u-p coupling block and returns plausible garbage pore "
        f"pressures (fork guide §2, the #1 u-p footgun). Declare a general "
        f"solver before build: {_GENERAL_SOLVER_MSG}."
    )


_SERIAL_MUMPS_MSG = (
    "`system Mumps` declared on a serial deck (`len(fem.partitions) <= 1`): "
    "the Ladruno fork's desktop targets do not compile the serial "
    "`MumpsSolver`, so `system Mumps` answers *unknown system type* on "
    "`OpenSees.exe` and on the desktop openseespy build — leaving the run "
    "on whatever SOE was already in place (ProfileSPD by default) rather "
    "than stopping. Declare `ops.system.Pardiso()` for a threaded desktop "
    "solve, or partition the mesh (`g.mesh.partitioning`) and run under "
    "`OpenSeesMP`."
)


#: Class names emitting ``system Mumps``. apeGmsh's typed system primitives
#: (``analysis/system.py``) expose a single :class:`Mumps`; ADR 0106 D5 also
#: names a ``MumpsParallel`` that does not exist as a Python type.
_SERIAL_MUMPS_CLASSES = frozenset({"Mumps"})


def validate_serial_mumps(
    *,
    enforce: bool,
    staged: bool,
    partitioned: bool,
    flat_systems: "Sequence[object]",
    stage_systems: "Sequence[tuple[str, object | None]]",
) -> None:
    """ADR 0106 D5 — refuse an explicit ``Mumps`` system on a serial deck.

    The Ladruno fork's desktop targets (``OpenSees.exe``, the desktop
    openseespy build) never compile the serial ``MumpsSolver``: a
    declared ``system Mumps`` answers *unknown system type* at runtime,
    and a rejected ``system`` command does not abort the deck
    (``OpenSees.exe`` exits 0 on a Tcl error) — the model silently solves
    on whatever SOE was already in place instead. Refusing at build time
    turns that wrong answer into a sentence.

    Scope, mirroring :func:`validate_ladruno_up_solver`'s seam:

    * ``enforce`` is False for emits that never drive a solve — H5
      archival, model-only export, eigen-only decks — so those are
      skipped entirely.
    * **partitioned** decks are untouched: that is what Mumps is for
      (ADR 0027 INV-5's auto-emitted fallback and ADR 0077 INV-8's
      parallel-ARPACK requirement both only fire under partitioning).
    * **staged**: each stage owns its analysis chain, so every stage's
      own declared system is checked independently; a stage with no
      declared system is not this gate's concern (that is
      ``validate_ladruno_up_solver``'s missing-system case).
    * **flat**: the effective (last-declared) system is the one
      OpenSees uses at analyze; only it is checked.

    Deliberately NO escape hatch — the declared solver does not exist on
    these targets, so there is no reading under which the deck does what
    it says.
    """
    if not enforce or partitioned:
        return

    if staged:
        for name, system in stage_systems:
            if system is not None and type(system).__name__ in _SERIAL_MUMPS_CLASSES:
                raise BridgeError(f"{_SERIAL_MUMPS_MSG} (stage {name})")
        return

    if flat_systems and type(flat_systems[-1]).__name__ in _SERIAL_MUMPS_CLASSES:
        raise BridgeError(_SERIAL_MUMPS_MSG)


_STAGE_MARKER_UNSAFE = str.maketrans({
    '"': "'", "[": "(", "]": ")", "{": "(", "}": ")", "\\": "/", "$": "_",
})


def stage_marker_name(name: str) -> str:
    r"""The stage name as it appears on an ``APEGMSH_STAGE`` marker line.

    ADR 0106 D2 — the marker is a runtime ``puts "..."`` / ``print("...")``
    whose only job is to survive the S1 parser's ``(.+)$``, so the name is
    normalised ONCE here for both emitters: ``"`` / ``[`` / ``]`` / ``{``
    / ``}`` / ``\`` / ``$`` would close the string, trigger Tcl command
    or variable substitution, or escape in Python; any whitespace run
    (a newline included) collapses to one space.  Deterministic, so the
    tcl and py lanes attribute to the same name.
    """
    return " ".join(str(name).translate(_STAGE_MARKER_UNSAFE).split())


def deck_requests_solver_stats(
    *,
    flat_systems: "Sequence[object]",
    stage_systems: "Sequence[tuple[str, object | None]]",
) -> bool:
    """ADR 0106 D2 — does this deck ask a solver for ``-stats`` anywhere?

    Resolves the flat/staged system declarations the same way
    :func:`validate_ladruno_up_solver` already resolves them (its
    ``flat_systems`` / ``stage_systems`` shapes, reused verbatim by the
    caller). Duck-types on ``.stats`` rather than naming ``Pardiso`` /
    ``Mumps`` — both carry the flag and nothing else does. Answers a
    plain "was it requested", independent of whether the deck's
    analysis chain ever runs.
    """
    for system in flat_systems:
        if getattr(system, "stats", False):
            return True
    for _name, system in stage_systems:
        if system is not None and getattr(system, "stats", False):
            return True
    return False
