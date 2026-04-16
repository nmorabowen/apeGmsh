"""
_Ingest — pull resolved load / mass records out of a :class:`FEMData`
snapshot and into the OpenSees bridge's internal tables.

Accessed via ``g.opensees.ingest``.  These two methods are the
bridge-side half of the "define on the session, resolve on the
snapshot" pipeline:

1. ``g.loads`` / ``g.masses`` accumulate *definitions*
2. ``g.mesh.queries.get_fem_data()`` resolves them into
   ``fem.nodes.loads`` / ``fem.elements.loads`` / ``fem.nodes.masses``
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

        Translates ``fem.nodes.loads`` and ``fem.elements.loads``
        (populated by the ``g.loads`` auto-resolve inside
        ``get_fem_data``) into the internal load-pattern dict consumed
        by :meth:`OpenSees.build`.

        After calling this, :meth:`OpenSees.build` will emit the loads
        as ``pattern Plain`` blocks.

        Parameters
        ----------
        fem : FEMData
            Snapshot from ``g.mesh.queries.get_fem_data()``.
        """
        nodal_loads = getattr(getattr(fem, "nodes", None), "loads", None)
        elem_loads = getattr(getattr(fem, "elements", None), "loads", None)
        if not nodal_loads and not elem_loads:
            return self

        ops = self._opensees
        total = 0

        if nodal_loads:
            for rec in nodal_loads:
                pat = rec.pattern
                if pat not in ops._load_patterns:
                    ops._load_patterns[pat] = []
                # The bridge dict keeps a length-6 (Fx,Fy,Fz,Mx,My,Mz) list
                # so _opensees_build/export/inspect can slice to ops._ndf.
                f = rec.force_xyz or (0.0, 0.0, 0.0)
                m = rec.moment_xyz or (0.0, 0.0, 0.0)
                ops._load_patterns[pat].append({
                    "type":    "nodal_direct",
                    "node_id": int(rec.node_id),
                    "forces":  [f[0], f[1], f[2], m[0], m[1], m[2]],
                })
            total += len(nodal_loads)

        if elem_loads:
            for rec in elem_loads:
                pat = rec.pattern
                if pat not in ops._load_patterns:
                    ops._load_patterns[pat] = []
                ops._load_patterns[pat].append({
                    "type":       "element_direct",
                    "element_id": int(rec.element_id),
                    "load_type":  rec.load_type,
                    "params":     dict(rec.params),
                })
            total += len(elem_loads)

        # Collect unique patterns from both sets
        all_patterns: list[str] = []
        if nodal_loads:
            all_patterns.extend(nodal_loads.patterns())
        if elem_loads:
            for p in elem_loads.patterns():
                if p not in all_patterns:
                    all_patterns.append(p)

        ops._log(
            f"ingest.loads(): {total} load record(s) "
            f"across {len(all_patterns)} pattern(s)"
        )
        return self

    def sp(self, fem) -> "_Ingest":
        """Ingest resolved SP records from a :class:`FEMData` snapshot.

        Translates ``fem.nodes.sp`` (populated by ``g.loads.face_sp``
        auto-resolve) into the internal SP dict consumed by
        :meth:`OpenSees.build`.

        Parameters
        ----------
        fem : FEMData
            Snapshot from ``g.mesh.queries.get_fem_data()``.
        """
        sp_set = getattr(getattr(fem, "nodes", None), "sp", None)
        if not sp_set:
            return self
        ops = self._opensees
        for rec in sp_set:
            ops._sp_records.append({
                "node_id": int(rec.node_id),
                "dof": int(rec.dof),
                "value": float(rec.value),
                "is_homogeneous": bool(rec.is_homogeneous),
            })
        ops._log(f"ingest.sp(): {len(sp_set)} SP record(s)")
        return self

    def masses(self, fem) -> "_Ingest":
        """Ingest resolved nodal mass records from a :class:`FEMData` snapshot.

        Translates ``fem.nodes.masses`` (populated by the ``g.masses``
        auto-resolve) into the internal mass dict consumed by
        :meth:`OpenSees.build`.  Each record becomes one
        ``ops.mass(node, mx, my, mz, ...)`` command.

        Parameters
        ----------
        fem : FEMData
            Snapshot from ``g.mesh.queries.get_fem_data()``.
        """
        masses = getattr(getattr(fem, "nodes", None), "masses", None)
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

    def constraints(
        self,
        fem,
        *,
        tie_penalty: float | None = None,
    ) -> "_Ingest":
        """Ingest resolved constraint records from a :class:`FEMData` snapshot.

        Stores ``fem.nodes.constraints`` and ``fem.elements.constraints``
        on the broker for emission during :meth:`OpenSees.build`.
        Currently the emitter only handles **tie** interpolation
        records (``kind == "tie"``) from the element-side set — they
        become ``element ASDEmbeddedNodeElement`` commands in the
        exported script.  Node-pair records (``equal_dof`` /
        ``rigid_beam`` / ``rigid_rod``), rigid diaphragms, and embedded
        rebars are ingested but emission is deferred to later phases.

        Parameters
        ----------
        fem : FEMData
            Snapshot from ``g.mesh.queries.get_fem_data()``.
        tie_penalty : float, optional
            Penalty stiffness passed to ``element ASDEmbeddedNodeElement``
            via the ``-K`` flag.  When ``None`` (default) OpenSees
            uses its built-in default of ``1.0e18`` — drop to ``1e10``
            – ``1e12`` if you see conditioning issues.  The element
            is a penalty formulation, so you only need the stiffness
            several orders of magnitude above the parent element
            stiffness, not infinite.

        Example
        -------
        ::

            fem = g.mesh.queries.get_fem_data(dim=3)
            (g.opensees.ingest
                .loads(fem)
                .masses(fem)
                .constraints(fem, tie_penalty=1e12))
            g.opensees.build()
        """
        node_cs = getattr(getattr(fem, "nodes", None), "constraints", None)
        elem_cs = getattr(getattr(fem, "elements", None), "constraints", None)
        if (node_cs is None or not node_cs) and (elem_cs is None or not elem_cs):
            return self
        ops = self._opensees
        # Element-side constraints (SurfaceConstraintSet) are consumed by
        # emit_tie_elements via ops._constraint_records.interpolations().
        ops._constraint_records = elem_cs
        # Node-side constraints (NodeConstraintSet) are stored separately
        # for future node-pair / extra-node emission.
        ops._node_constraint_records = node_cs
        ops._tie_penalty = tie_penalty

        total = 0
        kinds: list[str] = []
        for cs in (node_cs, elem_cs):
            if cs is not None and cs:
                total += len(cs)
                try:
                    kinds.extend(cs.summary().index.tolist())
                except Exception:
                    pass
        ops._log(
            f"ingest.constraints(): {total} record(s) "
            f"(kinds={kinds}, tie_penalty={tie_penalty})"
        )
        return self
