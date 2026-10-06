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

* **P5, the dirty flag.**  A loop of ``analyze`` calls on one model
  writes once: the write runs only when the bridge's declarations
  changed since its last write.  The key is the primitive count and
  the snapshot's content hash, as the ruling says, widened to the
  bridge's record lists and model flags (a ``fix``, ``mass``,
  ``region``, ``stage`` or ``mass_from_model`` is not a primitive, and
  without them an edit that only adds one would leave the file stale:
  the gap both K1-3d briefs flagged on #1307).
* **P6, the opt-out.**  ``apeSees(fem, _artifacts=False)`` is for the
  library's own bridges (the ETABS and STKO importers, ``strut_tie``,
  the emit-cost bench): they write nothing and warn nothing.

The write goes through the one composition path, :meth:`apeSees.h5`,
into ``<target>.tmp-<uuid>`` beside the target and an atomic replace,
so a failed write leaves the previous file untouched and no temp
behind.  Nothing here raises into the user's run: a refused write (the
policy's own warning), a snapshot with no name (P2), a partitioned
snapshot (P3), or a failure of the write itself (an ``OSError``, an
h5py error) warns **once per bridge** and stops; an MPI rank other
than 0 is silent (P3).  A stub snapshot (not a :class:`FEMData`; the
bridge-only file ``h5()`` writes for it carries no neutral zone)
belongs to no run and is skipped silently.  An explicit ``ops.h5(path)``
never comes here (P7).
"""
from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # The bridge, as the procedure mixins and ``apeSees`` itself see it:
    # the typing scaffold declares every attribute read here.
    from ..procedures._host import _ProcedureHost

__all__ = ["BridgeArtifactWarning", "BridgeArtifactWriter"]


class BridgeArtifactWarning(UserWarning):
    """The bridge's automatic ``model.h5`` write was skipped or failed.

    Issued once per bridge: for a snapshot with no name (P2), a
    partitioned snapshot (P3) and a write that failed.  A target the
    D1 policy refuses warns through the policy's own ``UserWarning``.
    """


#: ``warnings.warn`` depth from :meth:`BridgeArtifactWriter.after_emit`
#: to the user's line: the method, the emit site (``tcl``, ``analyze``,
#: ...), the user's call.
_STACKLEVEL = 3


def _dirty_key(bridge: "_ProcedureHost") -> tuple[object, ...]:
    """What the write depends on (P5): the registered primitives, every
    record list and model flag the build reads, and the snapshot's
    content hash (``snapshot_id``, the stored ``fem_hash``; cached on
    the snapshot, so a repeat is a lookup).  Explicit attributes, not a
    name list, so the type checker holds them to the bridge."""
    return (
        len(bridge._primitives),
        len(bridge._fix_records),
        len(bridge._equation_constraint_records),
        len(bridge._mass_records),
        len(bridge._ndf_records),
        len(bridge._region_records),
        len(bridge._rayleigh_records),
        len(bridge._damping_attach_records),
        len(bridge._modal_damping_records),
        len(bridge._initial_stress_records),
        len(bridge._stage_records),
        bridge._mass_from_model,
        bridge._fix_from_model,
        bridge._ndm,
        bridge._ndf,
        bridge._fem.snapshot_id,
    )


class BridgeArtifactWriter:
    """One bridge's automatic-write state: the opt-out, the P5 key of its
    last write, and whether it has warned."""

    __slots__ = ("enabled", "_last_key", "_warned")

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._last_key: tuple[object, ...] | None = None
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
            key = _dirty_key(bridge)
            if key == self._last_key:
                return
            from ..._artifact_policy import (
                artifact_verdict,
                content_hash,
                mpi_rank,
                provenance_scripts,
            )
            from ..._atomic_io import replace_with_retry
            from ..._core import default_artifact_dir
            from .build import is_partitioned
            from .schema_version import NEUTRAL, OPENSEES, PROVENANCE

            # P3: under MPI only rank 0 writes automatically.
            if mpi_rank() not in (None, 0):
                return
            # P1/P2: the name is the one the session gave the snapshot.
            name = fem.model_name
            if not name:
                self._warned = True
                warnings.warn(
                    "no model name: the snapshot carries no model_name (no "
                    "session named it: a session with no model_name= run "
                    "from a notebook, -c, stdin or a console-script launcher, "
                    "or a from_msh / import / compose snapshot), so there is "
                    "no conventional model.h5 path; the bridge writes nothing "
                    "automatically. Pass model_name= to the session, or call "
                    "ops.h5(path).",
                    BridgeArtifactWarning,
                    stacklevel=_STACKLEVEL,
                )
                return
            # P3: a partitioned snapshot emits per rank; its model.h5 is
            # outside V2's scope.  The bridge's own predicate, the one
            # BuiltModel.emit routes on.
            if is_partitioned(fem):
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
            target = default_artifact_dir() / f"{name}.h5"
            # The zones ``h5()`` writes: ``/provenance`` only when the
            # merged table has records (compose.py, ``_merge_provenance``).
            bridge_prov = bridge._provenance.snapshot()
            writes = frozenset({NEUTRAL, OPENSEES})
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
                    bridge.h5(str(tmp), model_name=name)
                    replace_with_retry(tmp, target)
                finally:
                    tmp.unlink(missing_ok=True)
            self._last_key = key
        except Exception as exc:  # noqa: BLE001
            self._warned = True
            warnings.warn(
                f"model.h5 not written at "
                f"{target if target is not None else '<unresolved>'}: "
                f"{exc!r}. The bridge does not try again.",
                BridgeArtifactWarning,
                stacklevel=_STACKLEVEL,
            )
