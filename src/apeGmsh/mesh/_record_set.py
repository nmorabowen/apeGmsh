"""
_record_set — Sub-composite classes for FEM boundary conditions.
=================================================================

Provides the shared ``_RecordSetBase`` and all sub-composite classes
that live on ``NodeComposite`` and ``ElementComposite``:

Node-side:
    ``NodeConstraintSet``   — equal_dof, rigid_beam, node_to_surface, …
    ``NodalLoadSet``        — point forces on nodes
    ``MassSet``             — lumped nodal masses

Element-side:
    ``SurfaceConstraintSet`` — tie, mortar, tied_contact, …
    ``ElementLoadSet``       — surface pressure, body forces

Also defines ``ConstraintKind`` and ``LoadKind`` constant classes
for linter-friendly kind comparisons (no magic strings).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


# =====================================================================
# Kind constants — linter-friendly, autocomplete-friendly
# =====================================================================

class ConstraintKind:
    """String constants for constraint record ``kind`` values.

    Exposed as ``Kind`` on each constraint sub-composite so the user
    gets autocomplete right where they need it::

        K = fem.nodes.constraints.Kind
        for c in fem.nodes.constraints.node_pairs():
            if c.kind == K.RIGID_BEAM:
                ...
    """
    EQUAL_DOF          = "equal_dof"
    RIGID_BEAM         = "rigid_beam"
    RIGID_ROD          = "rigid_rod"
    RIGID_DIAPHRAGM    = "rigid_diaphragm"
    RIGID_BODY         = "rigid_body"
    KINEMATIC_COUPLING = "kinematic_coupling"
    PENALTY            = "penalty"
    NODE_TO_SURFACE    = "node_to_surface"
    TIE                = "tie"
    DISTRIBUTING       = "distributing"
    EMBEDDED           = "embedded"
    TIED_CONTACT       = "tied_contact"
    MORTAR             = "mortar"

    # Classification for rendering / routing.
    # Keep these next to the constants above so additions are obvious.
    NODE_PAIR_KINDS = frozenset({
        EQUAL_DOF, RIGID_BEAM, RIGID_ROD, RIGID_DIAPHRAGM,
        RIGID_BODY, KINEMATIC_COUPLING, PENALTY, NODE_TO_SURFACE,
    })
    SURFACE_KINDS = frozenset({
        TIE, DISTRIBUTING, EMBEDDED, TIED_CONTACT, MORTAR,
    })


class LoadKind:
    """String constants for load record ``kind`` values.

    Exposed as ``Kind`` on each load sub-composite::

        K = fem.nodes.loads.Kind
    """
    NODAL   = "nodal"
    ELEMENT = "element"


# =====================================================================
# Record set base
# =====================================================================

class _RecordSetBase:
    """Shared protocol for all record sub-composites.

    Provides ``__iter__``, ``__len__``, ``__bool__``, and ``by_kind``
    so that sub-classes only add their domain-specific methods.
    """

    def __init__(self, records: list | None = None) -> None:
        self._records: list = list(records) if records else []

    def by_kind(self, kind: str) -> list:
        """Return all records matching a constraint/load kind."""
        return [r for r in self._records
                if getattr(r, 'kind', None) == kind]

    # ── Dunder ──────────────────────────────────────────────

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return len(self._records) > 0

    def __repr__(self) -> str:
        cls = type(self).__name__
        if not self._records:
            return f"{cls}(empty)"
        return f"{cls}({len(self._records)} records)"


# =====================================================================
# Node-side sub-composites
# =====================================================================

class NodeConstraintSet(_RecordSetBase):
    """Node-to-node constraints: equal_dof, rigid_beam, rigid_diaphragm,
    node_to_surface, penalty, etc.

    Also holds phantom nodes produced by compound constraints.

    Accessed via ``fem.nodes.constraints``.

    Example
    -------
    ::

        K = fem.nodes.constraints.Kind
        for nid, xyz in fem.nodes.constraints.extra_nodes():
            ops.node(nid, *xyz)

        for c in fem.nodes.constraints.node_pairs():
            if c.kind == K.RIGID_BEAM:
                ops.rigidLink("beam", c.master_node, c.slave_node)
            elif c.kind == K.EQUAL_DOF:
                ops.equalDOF(c.master_node, c.slave_node, *c.dofs)
    """

    Kind = ConstraintKind

    def node_pairs(self):
        """Iterate over every constraint as individual
        :class:`NodePairRecord` objects.

        Compound records (:class:`NodeGroupRecord`,
        :class:`NodeToSurfaceRecord`) are expanded automatically.

        Yields
        ------
        NodePairRecord
        """
        from apeGmsh.solvers.Constraints import (
            NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        )

        for rec in self._records:
            if isinstance(rec, NodePairRecord):
                yield rec
            elif isinstance(rec, NodeGroupRecord):
                yield from rec.expand_to_pairs()
            elif isinstance(rec, NodeToSurfaceRecord):
                yield from rec.expand()

    def extra_nodes(self):
        """Iterate ``(node_id, coords)`` for phantom nodes that solvers
        must create **before** emitting constraint commands.

        Yields
        ------
        (int, list[float])
            Node ID and ``[x, y, z]`` coordinates.
        """
        from apeGmsh.solvers.Constraints import NodeToSurfaceRecord

        for rec in self._records:
            if isinstance(rec, NodeToSurfaceRecord):
                coords = rec.phantom_coords
                if coords is None:
                    continue
                for tag, xyz in zip(rec.phantom_nodes, coords):
                    yield int(tag), xyz.tolist()

    def get_phantom_nodes(self):
        """Return phantom node IDs and coordinates as a ``NodeResult``.

        Returns
        -------
        NodeResult
            ``(ids, coords)`` — destructurable as
            ``ids, coords = constraints.get_phantom_nodes()``.
            Empty arrays if no phantom nodes exist.
        """
        from .FEMData import NodeResult
        ids_list, coords_list = [], []
        for nid, xyz in self.extra_nodes():
            ids_list.append(nid)
            coords_list.append(xyz)
        if not ids_list:
            return NodeResult(
                np.array([], dtype=np.int64),
                np.empty((0, 3), dtype=np.float64))
        return NodeResult(
            np.array(ids_list, dtype=np.int64),
            np.array(coords_list, dtype=np.float64))

    def summary(self) -> "pd.DataFrame":
        """DataFrame summarising the constraint set.

        Columns: ``kind``, ``count``, ``n_node_pairs``.
        """
        import pandas as pd
        from apeGmsh.solvers.Constraints import (
            NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        )

        counts: dict[str, dict] = {}
        for rec in self._records:
            kind = rec.kind
            if kind not in counts:
                counts[kind] = {"count": 0, "n_node_pairs": 0}
            counts[kind]["count"] += 1

            if isinstance(rec, NodePairRecord):
                counts[kind]["n_node_pairs"] += 1
            elif isinstance(rec, NodeGroupRecord):
                counts[kind]["n_node_pairs"] += len(rec.slave_nodes)
            elif isinstance(rec, NodeToSurfaceRecord):
                counts[kind]["n_node_pairs"] += (
                    len(rec.rigid_link_records) + len(rec.equal_dof_records)
                )

        if not counts:
            return pd.DataFrame(columns=["kind", "count", "n_node_pairs"])

        rows = [
            {"kind": k, "count": v["count"],
             "n_node_pairs": v["n_node_pairs"]}
            for k, v in counts.items()
        ]
        return pd.DataFrame(rows).set_index("kind").sort_index()

    def __repr__(self) -> str:
        if not self._records:
            return "NodeConstraintSet(empty)"
        kinds: dict[str, int] = {}
        for r in self._records:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        return f"NodeConstraintSet({len(self._records)} records: {parts})"


class NodalLoadSet(_RecordSetBase):
    """Nodal load records (point forces).

    Accessed via ``fem.nodes.loads``.

    Example
    -------
    ::

        for load in fem.nodes.loads:
            ops.load(load.node_id, *load.forces)

        wind = fem.nodes.loads.by_pattern("Wind")
    """

    Kind = LoadKind

    def patterns(self) -> list[str]:
        """Return all unique pattern names in insertion order."""
        seen: list[str] = []
        for r in self._records:
            if r.pattern not in seen:
                seen.append(r.pattern)
        return seen

    def by_pattern(self, name: str) -> list:
        """Return all records belonging to the named pattern."""
        return [r for r in self._records if r.pattern == name]

    def summary(self) -> "pd.DataFrame":
        """DataFrame of (pattern, kind) -> count."""
        import pandas as pd
        if not self._records:
            return pd.DataFrame(columns=["pattern", "kind", "count"])
        rows: dict[tuple, int] = {}
        for r in self._records:
            key = (r.pattern, r.kind)
            rows[key] = rows.get(key, 0) + 1
        data = [
            {"pattern": p, "kind": k, "count": c}
            for (p, k), c in rows.items()
        ]
        return (pd.DataFrame(data)
                .sort_values(["pattern", "kind"])
                .reset_index(drop=True))

    def __repr__(self) -> str:
        if not self._records:
            return "NodalLoadSet(empty)"
        n_pats = len(self.patterns())
        return (f"NodalLoadSet({len(self._records)} records, "
                f"{n_pats} pattern(s))")


class MassSet(_RecordSetBase):
    """Resolved nodal masses.

    Accessed via ``fem.nodes.masses``.

    Example
    -------
    ::

        for m in fem.nodes.masses:
            ops.mass(m.node_id, *m.mass)

        fem.nodes.masses.total_mass()
    """

    def by_node(self, node_id: int):
        """Return the MassRecord for a node, or ``None``."""
        for r in self._records:
            if r.node_id == int(node_id):
                return r
        return None

    def total_mass(self) -> float:
        """Sum of translational mass (mx) over all records.

        Assumes isotropic mass (mx == my == mz).
        """
        return sum(float(r.mass[0]) for r in self._records)

    def summary(self) -> "pd.DataFrame":
        """DataFrame with one row per node:
        ``node_id, mx, my, mz, Ixx, Iyy, Izz``.
        """
        import pandas as pd
        if not self._records:
            return pd.DataFrame(
                columns=["node_id", "mx", "my", "mz",
                         "Ixx", "Iyy", "Izz"])
        rows = []
        for r in self._records:
            m = r.mass
            rows.append({
                "node_id": int(r.node_id),
                "mx": float(m[0]), "my": float(m[1]), "mz": float(m[2]),
                "Ixx": float(m[3]), "Iyy": float(m[4]),
                "Izz": float(m[5]),
            })
        return (pd.DataFrame(rows)
                .sort_values("node_id")
                .reset_index(drop=True))

    def __repr__(self) -> str:
        if not self._records:
            return "MassSet(empty)"
        return (f"MassSet({len(self._records)} nodes, "
                f"total={self.total_mass():.6g})")


# =====================================================================
# Element-side sub-composites
# =====================================================================

class SurfaceConstraintSet(_RecordSetBase):
    """Surface coupling constraints: tie, mortar, tied_contact,
    distributing.

    Accessed via ``fem.elements.constraints``.

    Example
    -------
    ::

        K = fem.elements.constraints.Kind
        for interp in fem.elements.constraints.interpolations():
            # build MP constraint from weights
            ...
    """

    Kind = ConstraintKind

    def interpolations(self):
        """Iterate over :class:`InterpolationRecord` objects.

        Covers ``tie``, ``distributing``, and the slave records
        inside ``tied_contact`` / ``mortar``.

        Yields
        ------
        InterpolationRecord
        """
        from apeGmsh.solvers.Constraints import (
            InterpolationRecord, SurfaceCouplingRecord,
        )

        for rec in self._records:
            if isinstance(rec, InterpolationRecord):
                yield rec
            elif isinstance(rec, SurfaceCouplingRecord):
                yield from rec.slave_records

    def couplings(self):
        """Iterate over :class:`SurfaceCouplingRecord` objects.

        Yields the top-level coupling records (mortar, tied_contact).

        Yields
        ------
        SurfaceCouplingRecord
        """
        from apeGmsh.solvers.Constraints import SurfaceCouplingRecord

        for rec in self._records:
            if isinstance(rec, SurfaceCouplingRecord):
                yield rec

    def summary(self) -> "pd.DataFrame":
        """DataFrame summarising surface constraints.

        Columns: ``kind``, ``count``, ``n_interpolations``.
        """
        import pandas as pd
        from apeGmsh.solvers.Constraints import (
            InterpolationRecord, SurfaceCouplingRecord,
        )

        counts: dict[str, dict] = {}
        for rec in self._records:
            kind = rec.kind
            if kind not in counts:
                counts[kind] = {"count": 0, "n_interpolations": 0}
            counts[kind]["count"] += 1

            if isinstance(rec, InterpolationRecord):
                counts[kind]["n_interpolations"] += 1
            elif isinstance(rec, SurfaceCouplingRecord):
                counts[kind]["n_interpolations"] += len(rec.slave_records)

        if not counts:
            return pd.DataFrame(
                columns=["kind", "count", "n_interpolations"])

        rows = [
            {"kind": k, "count": v["count"],
             "n_interpolations": v["n_interpolations"]}
            for k, v in counts.items()
        ]
        return pd.DataFrame(rows).set_index("kind").sort_index()

    def __repr__(self) -> str:
        if not self._records:
            return "SurfaceConstraintSet(empty)"
        kinds: dict[str, int] = {}
        for r in self._records:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        return (f"SurfaceConstraintSet("
                f"{len(self._records)} records: {parts})")


class ElementLoadSet(_RecordSetBase):
    """Element load records (surface pressure, body forces).

    Accessed via ``fem.elements.loads``.

    Example
    -------
    ::

        for eload in fem.elements.loads:
            ops.eleLoad(eload.element_id, eload.load_type, **eload.params)
    """

    Kind = LoadKind

    def patterns(self) -> list[str]:
        """Return all unique pattern names in insertion order."""
        seen: list[str] = []
        for r in self._records:
            if r.pattern not in seen:
                seen.append(r.pattern)
        return seen

    def by_pattern(self, name: str) -> list:
        """Return all records belonging to the named pattern."""
        return [r for r in self._records if r.pattern == name]

    def summary(self) -> "pd.DataFrame":
        """DataFrame of (pattern, kind) -> count."""
        import pandas as pd
        if not self._records:
            return pd.DataFrame(columns=["pattern", "kind", "count"])
        rows: dict[tuple, int] = {}
        for r in self._records:
            key = (r.pattern, getattr(r, 'load_type', r.kind))
            rows[key] = rows.get(key, 0) + 1
        data = [
            {"pattern": p, "kind": k, "count": c}
            for (p, k), c in rows.items()
        ]
        return (pd.DataFrame(data)
                .sort_values(["pattern", "kind"])
                .reset_index(drop=True))

    def __repr__(self) -> str:
        if not self._records:
            return "ElementLoadSet(empty)"
        n_pats = len(self.patterns())
        return (f"ElementLoadSet({len(self._records)} records, "
                f"{n_pats} pattern(s))")
