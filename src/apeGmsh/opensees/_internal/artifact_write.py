"""The bridge's automatic ``model.h5`` write (ADR 0112 D1 for ``apeSees``).

A run that uses the bridge and never calls ``ops.h5(path)`` still leaves
the full ``model.h5`` (neutral zone + ``/opensees`` + ``/provenance``,
stamped with the snapshot's ``session_id``) at the session's
conventional path, where the session's ``end()`` leaves the neutral file
and the ``<stem>.geometry.h5`` sibling (#1307, V2d-4b).  ``apeSees``
calls :meth:`BridgeArtifactWriter.after_emit` once at the end of every
terminal deck emit and live build; this module decides whether that
call writes, through the D1 overwrite policy
(:mod:`apeGmsh._artifact_policy`, rules P1 to P4, built in 4a and called
here, not re-implemented) and two rules of its own:

* **P5, the dirty flag.**  A loop of emits on one model writes once,
  and so does a loop of fresh bridges on one unchanged model (one
  bridge per ground motion): the write runs only when the
  declarations that would reach the file differ from what the file at
  that path last received.  The declarations are digested whole
  (:func:`_declaration_digest`: every primitive with its contents, a
  pattern's recorded loads included, every record list, the model
  flags, the names and the snapshot's content hash), never counted, so
  a second ``p.load()`` on a registered pattern or a sweep that changes
  one material value in a fresh bridge is a new state.  The digest of
  the last write is kept process-wide, keyed by the resolved target.
* **P6, the opt-out.**  ``apeSees(fem, _artifacts=False)`` is for the
  library's own bridges (the ETABS and STKO importers, ``strut_tie``,
  the emit-cost bench): they write nothing and warn nothing.

The write goes through the one composition path, :meth:`apeSees.h5`,
into ``<target>.tmp-<uuid>`` beside the target and an atomic replace,
so a failed write leaves the previous file untouched and no temp
behind.  An automatic write is silent about deferred archive features
(``H5FeatureDeferredWarning`` belongs to the explicit saves: ``g.save()``,
``save_to=``, ``ops.h5(path)``).  Nothing here raises into the user's
run: a refused write (the policy's own warning), a snapshot with no
name that no session warned about (P2), a kernel-partitioned snapshot
(P3), or a failure of the write itself (an ``OSError``, an h5py error)
warns **once per bridge** and stops; an MPI rank other than 0 is silent
(P3).  A session snapshot with no name is silent too: its session
already gave the one P2 warning at ``end()``.  A stub snapshot (not a
:class:`FEMData`; the bridge-only file ``h5()`` writes for it carries no
neutral zone) belongs to no run and is skipped silently.  An explicit
``ops.h5(path)`` never comes here (P7).

**The lineage memo.**  Every archive :meth:`apeSees.h5` writes,
the automatic one and an explicit ``ops.h5(path)`` alike, leaves its
``(model_hash, fem_hash)`` in :data:`_LINEAGE_BY_DIGEST` under
:func:`lineage_key`, the P5 digest of the declarations it was composed
from.  A deck exported later from the same declarations stamps its run
record with that lineage and composes no archive
(:func:`apeGmsh.opensees.emitter.py.deck_run_lineage`): the only route to
a record for a partitioned model, whose archive P3 never writes
automatically and a deck export never composes.
"""
from __future__ import annotations

import hashlib
import pickle
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # The bridge, as the procedure mixins and ``apeSees`` itself see it:
    # the typing scaffold declares every attribute read here.
    from ..procedures._host import _ProcedureHost

__all__ = [
    "BridgeArtifactWarning", "BridgeArtifactWriter", "lineage_key",
    "remember_lineage",
]


class BridgeArtifactWarning(UserWarning):
    """The bridge's automatic ``model.h5`` write was skipped or failed.

    Issued once per bridge: for a snapshot with no name that no session
    warned about (P2), a kernel-partitioned snapshot (P3) and a write
    that failed.  A target the D1 policy refuses warns through the
    policy's own ``UserWarning``.
    """


#: ``warnings.warn`` depth from :meth:`BridgeArtifactWriter.after_emit`
#: to the user's line: the method, the emit site (``tcl``, ``analyze``,
#: ...), the user's call.
_STACKLEVEL = 3

#: The declaration digest last written at each resolved target, for
#: the whole process (P5 across bridges).  A target whose file is gone
#: is written again whatever the memo says.
_LAST_WRITTEN: dict[Path, str] = {}


def _declaration_digest(bridge: "_ProcedureHost", name: str) -> str:
    """One digest of everything the write depends on (P5): the name, the
    snapshot's ``session_id`` (a re-run of the session in one process, a
    notebook cell run again, is a new run whose file must carry its id)
    and content hash (``snapshot_id``, the stored ``fem_hash``), the
    registered primitives with their full contents (a frozen primitive's
    fields and a pattern's private accumulators alike), the name
    aliases, every record list and the model flags.  Pickled, not
    ``repr``'d, so a long array is never truncated into a collision; an
    object that cannot be pickled refuses, and the caller warns.
    """
    state = (
        name,
        bridge._fem.session_id,
        bridge._fem.snapshot_id,
        bridge._ndm,
        bridge._ndf,
        bridge._element_tags,
        bridge._default_orientation,
        bridge._mass_from_model,
        bridge._fix_from_model,
        bridge._primitives,
        bridge._names,
        bridge._fix_records,
        bridge._equation_constraint_records,
        bridge._mass_records,
        bridge._ndf_records,
        bridge._region_records,
        bridge._rayleigh_records,
        bridge._damping_attach_records,
        bridge._modal_damping_records,
        bridge._initial_stress_records,
        bridge._stage_records,
    )
    raw = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


#: ``(model_hash, fem_hash)`` by :func:`lineage_key`, for every archive of
#: a bridge this process composed (``ops.h5(path)``, the automatic write, a
#: deck record's in-memory compose).  The lineage is a function of what the
#: digest covers: neither the name the file is written under, its path,
#: nor its cuts and sweeps enter ``model_hash``
#: (:func:`~apeGmsh.opensees._internal.lineage.compute_model_hash`).
_LINEAGE_BY_DIGEST: dict[str, tuple[str, str]] = {}


def lineage_key(bridge: "_ProcedureHost") -> str | None:
    """The digest :data:`_LINEAGE_BY_DIGEST` keys ``bridge``'s current
    declarations by: P5's, under the snapshot's own name.  ``None`` for a
    snapshot that is not a :class:`FEMData` (its bridge-only archive has
    no neutral zone, so no lineage to reuse) or a declaration that cannot
    be digested (whatever its pickling raises); nothing is then memoised,
    and a deck composes or refuses by its own rule."""
    from ...mesh.FEMData import FEMData

    fem = bridge._fem
    if not isinstance(fem, FEMData):
        return None
    try:
        return _declaration_digest(bridge, fem.model_name or "")
    except Exception:  # noqa: BLE001 - a declaration that cannot be digested
        # Whatever pickle or an object's own ``__reduce_ex__`` raises: the
        # export and ``h5()`` go on, and the deck says why it has no record.
        return None


def remember_lineage(bridge: "_ProcedureHost", archive: object) -> None:
    """Memoise the lineage of ``archive``, a ``model.h5`` (a path or a
    file object) just composed from ``bridge``'s current declarations.  An
    archive with no ``model_hash`` (a stub snapshot's) is not memoised."""
    key = lineage_key(bridge)
    if key is None:
        return
    import h5py

    from .lineage import read_stored_lineage

    with h5py.File(archive, "r") as f:
        fem_hash, model_hash, _ = read_stored_lineage(f["meta"])
    if model_hash is not None:
        _LINEAGE_BY_DIGEST[key] = (model_hash, fem_hash or "")


def emit_deck_tail(
    bridge: Any, emitter: Any, bm: Any, pre_prof: Any, post_prof: Any,
    analyze_steps: "int | None", analyze_dt: Any, path: Any,
) -> None:
    """Emit ``bm`` into ``emitter``, then the deck's tail: the profiler
    records, the ``analyze`` line and, last, the run record (#2210; it
    reuses the automatic ``model.h5``'s lineage).  ``apeSees.py`` and
    ``apeSees.tcl`` share it (#2289)."""
    bm.emit(emitter)
    for _verb, _vargs in pre_prof:
        emitter.profiler(_verb, *_vargs)
    if analyze_steps is not None:
        emitter.analyze(steps=int(analyze_steps), dt=analyze_dt)
    for _verb, _vargs in post_prof:
        emitter.profiler(_verb, *_vargs)
    if analyze_steps is not None:
        bridge._artifacts.after_emit(bridge)
        emitter._deck_run_record(
            lambda p: bridge._write_h5(p, bm), int(analyze_steps),
            analyze_dt, bridge, path,
        )


class BridgeArtifactWriter:
    """One bridge's automatic-write state: the opt-out and whether it
    has warned.  What was last written lives in :data:`_LAST_WRITTEN`,
    shared by every bridge in the process."""

    __slots__ = ("enabled", "_warned")

    #: :meth:`apeSees.h5` memoises every archive's lineage through the
    #: bridge's writer, whatever its opt-out (P6 opts out of files only).
    remember_lineage = staticmethod(remember_lineage)

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._warned = False

    def after_emit(self, bridge: "_ProcedureHost") -> None:
        """Write ``bridge``'s ``model.h5`` at the conventional path if it
        is due.  Called by every terminal emit site; never raises."""
        if not self.enabled or self._warned:
            return
        target: Path | None = None
        try:
            from ...mesh.FEMData import FEMData

            fem = bridge._fem
            if not isinstance(fem, FEMData):
                return
            from ..._artifact_policy import (
                artifact_verdict,
                content_hash,
                mpi_rank,
                provenance_scripts,
                record_bridge_write,
            )
            from ..._atomic_io import replace_with_retry
            from ..._core import default_artifact_dir
            from ..emitter.h5 import H5FeatureDeferredWarning
            from .build.ownership import is_partitioned
            from .schema_version import NEUTRAL, OPENSEES, PROVENANCE, SEQUENCE

            # P3: under MPI only rank 0 writes automatically.
            if mpi_rank() not in (None, 0):
                return
            # P1/P2: the name is the one the session gave the snapshot.
            # A session snapshot (it carries the session's provenance
            # table) with no name already drew the one P2 warning, at
            # the session's ``end()``: stay silent.  A snapshot no
            # session extracted (from_msh, an import) gets the warning
            # here, once.
            name = fem.model_name
            if not name:
                if fem.provenance is None:
                    self._warned = True
                    warnings.warn(
                        "no model name: the snapshot carries no model_name "
                        "(a from_msh, import or hand-built snapshot; a "
                        "session's would have been named by it, or by the "
                        "script it ran from), so there is no conventional "
                        "model.h5 path; the bridge writes nothing "
                        "automatically. Call ops.h5(path).",
                        BridgeArtifactWarning,
                        stacklevel=_STACKLEVEL,
                    )
                return
            # P3: a mesh the kernel partitioned is an MPI deck's; its
            # model.h5 is outside V2's scope.  A composed model reports
            # its modules as partitions (ADR 0038) and is not
            # partitioned for D1: it writes, as it does for the session.
            if is_partitioned(fem) and not fem.composed_from:
                self._warned = True
                warnings.warn(
                    f"partitioned run ({len(fem.partitions)} partitions): "
                    f"no automatic write of {name}.h5, the partitioned "
                    f"model.h5 being outside V2's scope. Call ops.h5(path) "
                    f"to write the model anyway.",
                    BridgeArtifactWarning,
                    stacklevel=_STACKLEVEL,
                )
                return
            target = (default_artifact_dir() / f"{name}.h5").resolve()
            digest = _declaration_digest(bridge, name)
            if _LAST_WRITTEN.get(target) == digest and target.exists():
                return
            # The zones ``h5()`` writes: ``/sequence`` always (derived
            # from ``/opensees``, ADR 0112 D2b), ``/provenance`` only when
            # the merged table has records (compose.py, ``_merge_provenance``).
            bridge_prov = bridge._provenance.snapshot()
            writes = frozenset({NEUTRAL, OPENSEES, SEQUENCE})
            if fem.provenance is not None or bridge_prov.records:
                writes |= {PROVENANCE}
            scripts = provenance_scripts(fem.provenance) | provenance_scripts(
                bridge_prov,
            )
            verdict = artifact_verdict(
                target, writes=writes, overwrite=True,
                session_id=fem.session_id,
                content=lambda: content_hash(fem),
                scripts=scripts, explicit=False,
            )
            if verdict == "refuse":
                self._warned = True
                return
            if verdict == "write":
                tmp = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
                try:
                    with warnings.catch_warnings():
                        # A deferred archive feature is the explicit
                        # save's warning; the automatic write is silent.
                        warnings.simplefilter("ignore", H5FeatureDeferredWarning)
                        bridge.h5(str(tmp), model_name=name)
                    replace_with_retry(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
                # The session's end() then compares this file with the
                # same view (perhaps a narrower get_fem_data(dim=) than
                # its own) re-extracted at that moment: a view is equal,
                # a declaration made after this write is not.
                record_bridge_write(target, fem.session_id, fem.extract_view)
            _LAST_WRITTEN[target] = digest
        except Exception as exc:  # noqa: BLE001
            self._warned = True
            warnings.warn(
                f"model.h5 not written at "
                f"{target if target is not None else '<unresolved>'}: "
                f"{exc!r}. The bridge does not try again.",
                BridgeArtifactWarning,
                stacklevel=_STACKLEVEL,
            )
