"""
_Ingest — pull resolved load / mass records out of a :class:`FEMData`
snapshot and into the OpenSees bridge's internal tables.

Accessed via ``g.opensees.ingest``.  These two methods are the
bridge-side half of the "define on the session, resolve on the
snapshot" pipeline:

1. ``g.loads`` / ``g.masses`` accumulate *definitions*
2. ``g.mesh.queries.get_fem_data()`` resolves them into
   ``fem.loads`` / ``fem.masses``
3. ``g.opensees.ingest.loads(fem)`` / ``g.opensees.ingest.masses(fem)``
   translates those records into the internal dicts consumed by
   :meth:`OpenSees.build`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .OpenSees import OpenSees


class _Ingest:
    """Pull resolved loads and masses out of a :class:`FEMData` snapshot."""

    def __init__(self, parent: "OpenSees") -> None:
        self._opensees = parent

    def loads(self, fem) -> "_Ingest":
        """Ingest resolved load records from a :class:`FEMData` snapshot.

        Translates ``fem.loads`` (populated by the ``g.loads``
        auto-resolve inside ``get_fem_data``) into the internal
        load-pattern dict consumed by :meth:`OpenSees.build`.

        After calling this, :meth:`OpenSees.build` will emit the loads
        as ``pattern Plain`` blocks.

        Parameters
        ----------
        fem : FEMData
            Snapshot from ``g.mesh.queries.get_fem_data()``.
        """
        loads = getattr(fem, "loads", None)
        if not loads:
            return self
        from apeGmsh.solvers.Loads import NodalLoadRecord, ElementLoadRecord

        ops = self._opensees
        for rec in loads:
            pat = rec.pattern
            if pat not in ops._load_patterns:
                ops._load_patterns[pat] = []
            if isinstance(rec, NodalLoadRecord):
                ops._load_patterns[pat].append({
                    "type":    "nodal_direct",
                    "node_id": int(rec.node_id),
                    "forces":  list(rec.forces),
                })
            elif isinstance(rec, ElementLoadRecord):
                ops._load_patterns[pat].append({
                    "type":       "element_direct",
                    "element_id": int(rec.element_id),
                    "load_type":  rec.load_type,
                    "params":     dict(rec.params),
                })
        ops._log(
            f"ingest.loads(): {len(loads)} load record(s) "
            f"across {len(loads.patterns())} pattern(s)"
        )
        return self

    def masses(self, fem) -> "_Ingest":
        """Ingest resolved nodal mass records from a :class:`FEMData` snapshot.

        Translates ``fem.masses`` (populated by the ``g.masses``
        auto-resolve) into the internal mass dict consumed by
        :meth:`OpenSees.build`.  Each record becomes one
        ``ops.mass(node, mx, my, mz, ...)`` command.

        Parameters
        ----------
        fem : FEMData
            Snapshot from ``g.mesh.queries.get_fem_data()``.
        """
        masses = getattr(fem, "masses", None)
        if not masses:
            return self
        ops = self._opensees
        for r in masses:
            ops._mass_records.append({
                "node_id": int(r.node_id),
                "mass":    list(r.mass),
            })
        ops._log(f"ingest.masses(): {len(masses)} mass record(s)")
        return self
