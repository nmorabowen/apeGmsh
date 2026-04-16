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

from typing import TYPE_CHECKING, ClassVar, Generic, Iterator, TypeVar

import numpy as np

_R = TypeVar('_R')

if TYPE_CHECKING:
    import pandas as pd
    from apeGmsh.solvers.Constraints import (  # noqa: F401
        ConstraintRecord, NodePairRecord, NodeGroupRecord,
        NodeToSurfaceRecord, InterpolationRecord, SurfaceCouplingRecord,
    )
    from apeGmsh.solvers.Loads import NodalLoadRecord, ElementLoadRecord, SPRecord  # noqa: F401
    from apeGmsh.solvers.Masses import MassRecord  # noqa: F401


# =====================================================================
# Kind constants — linter-friendly, autocomplete-friendly
# =====================================================================

class ConstraintKind:
    """String constants for constraint record ``kind`` values.

    Exposed as ``Kind`` on each constraint sub-composite so the user
    gets autocomplete right where they need it::

        K = fem.nodes.constraints.Kind
        for c in fem.nodes.constraints.pairs():
            if c.kind == K.RIGID_BEAM:
                ops.rigidLink("beam", c.master_node, c.slave_node)

    The constants are typed as ``ClassVar[str]`` so Pylance/mypy
    recognise them as static attributes (not instance fields).
    """
    EQUAL_DOF:          ClassVar[str] = "equal_dof"
    RIGID_BEAM:         ClassVar[str] = "rigid_beam"
    RIGID_BEAM_STIFF:   ClassVar[str] = "rigid_beam_stiff"
    RIGID_ROD:          ClassVar[str] = "rigid_rod"
    RIGID_DIAPHRAGM:    ClassVar[str] = "rigid_diaphragm"
    RIGID_BODY:         ClassVar[str] = "rigid_body"
    KINEMATIC_COUPLING: ClassVar[str] = "kinematic_coupling"
    PENALTY:            ClassVar[str] = "penalty"
    NODE_TO_SURFACE:    ClassVar[str] = "node_to_surface"
    TIE:                ClassVar[str] = "tie"
    DISTRIBUTING:       ClassVar[str] = "distributing"
    EMBEDDED:           ClassVar[str] = "embedded"
    TIED_CONTACT:       ClassVar[str] = "tied_contact"
    MORTAR:             ClassVar[str] = "mortar"

    # Classification for rendering / routing.
    NODE_PAIR_KINDS: ClassVar[frozenset[str]] = frozenset({
        "equal_dof", "rigid_beam", "rigid_beam_stiff", "rigid_rod",
        "rigid_diaphragm", "rigid_body", "kinematic_coupling",
        "penalty", "node_to_surface",
    })
    SURFACE_KINDS: ClassVar[frozenset[str]] = frozenset({
        "tie", "distributing", "embedded", "tied_contact", "mortar",
    })


class LoadKind:
    """String constants for load record ``kind`` values.

    Exposed as ``Kind`` on each load sub-composite::

        K = fem.nodes.loads.Kind
    """
    NODAL:   ClassVar[str] = "nodal"
    ELEMENT: ClassVar[str] = "element"


# =====================================================================
# Record set base
# =====================================================================

class _RecordSetBase(Generic[_R]):
    """Shared protocol for all record sub-composites.

    Provides ``__iter__``, ``__len__``, ``__bool__``, and ``by_kind``
    so that sub-classes only add their domain-specific methods.

    Subclasses bind the type variable so that Pylance/mypy can infer
    the record type for iteration and query results.
    """

    def __init__(self, records: list[_R] | None = None) -> None:
        self._records: list[_R] = list(records) if records else []

    def by_kind(self, kind: str) -> list[_R]:
        """Return all records matching a constraint/load kind."""
        return [r for r in self._records
                if getattr(r, 'kind', None) == kind]

    # ── Dunder ──────────────────────────────────────────────

    def __iter__(self) -> Iterator[_R]:
        return iter(self._records)

    def __getitem__(self, idx: int) -> _R:
        return self._records[idx]

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

class NodeConstraintSet(_RecordSetBase["ConstraintRecord"]):
    """Node-to-node constraints.

    Accessed via ``fem.nodes.constraints``.

    Record types stored here
    ------------------------
    The set mixes three concrete record subclasses depending on the
    source constraint definition:

    =====================  ============================================
    Record type            Source constraint kinds
    =====================  ============================================
    ``NodePairRecord``     ``equal_dof``, ``rigid_beam``, ``rigid_rod``,
                           ``penalty``
    ``NodeGroupRecord``    ``rigid_diaphragm``, ``rigid_body``,
                           ``kinematic_coupling``
    ``NodeToSurfaceRecord`` ``node_to_surface`` (compound — also creates
                           phantom nodes)
    =====================  ============================================

    Record tiers: atomic vs compound
    --------------------------------
    Records fall into two tiers, and every iterator on this class
    deterministically returns one tier or the other:

    * **Atomic records** map 1:1 to a solver command.  The only
      atomic type here is ``NodePairRecord``.
    * **Compound records** wrap several atomic records and carry
      side-band data (phantom coordinates, rigid-arm offsets, …)
      that vanishes the moment they are flattened.  ``NodeGroupRecord``
      and ``NodeToSurfaceRecord`` are compound.

    Iterator convention:

    =====================================  =========  =================
    Iterator                               Returns    Expands compound?
    =====================================  =========  =================
    ``pairs()``                            atomic     yes (all compound)
    ``equal_dofs()``                       atomic     yes (NodeToSurface)
    ``rigid_link_groups()``                atomic+    yes (all rigid)
    ``rigid_diaphragms()``                 atomic+    no (diaphragm only)
    ``node_to_surfaces()``                 compound   no
    ``phantom_nodes()``                    side-band  n/a (NodeResult)
    direct iter (``for rec in …``)         mixed      no
    ``by_kind(kind)``                      mixed      no
    =====================================  =========  =================

    ``atomic+`` = grouped tuples ``(master, [slaves])`` built from
    atomic pair data — still side-band-free.

    **Rule of thumb.** If you need ``phantom_coords`` or any other
    compound-only field, iterate the compound accessor
    (``node_to_surfaces()``, ``phantom_nodes()``, or direct iter).
    If you just need flat solver commands, use an atomic iterator —
    compound records are expanded for you automatically.

    Common fields
    -------------
    All records have ``kind`` (str) and ``name`` (str | None).
    Beyond that, each subclass exposes its own fields — see the
    subclass docstrings in ``apeGmsh.solvers.Constraints``.

    OpenSees workflow (recommended)
    -------------------------------
    ::

        # 1. Create phantom nodes first (node_to_surface)
        for nid, xyz in fem.nodes.constraints.phantom_nodes():
            ops.node(nid, *xyz)

        # 2. Rigid links — grouped by master (physical hierarchy)
        for master, slaves in fem.nodes.constraints.rigid_link_groups():
            for slave in slaves:
                ops.rigidLink("beam", master, slave)

        # 3. Equal DOFs — flat iteration (each pair independent)
        for pair in fem.nodes.constraints.equal_dofs():
            ops.equalDOF(pair.master_node, pair.slave_node, *pair.dofs)

    Alternative: emit with native multi-slave commands
    ---------------------------------------------------
    ::

        # OpenSees rigidDiaphragm takes a list of slaves natively
        for master, slaves in fem.nodes.constraints.rigid_diaphragms():
            ops.rigidDiaphragm(3, master, *slaves)

    Filtering
    ---------
    ::

        # Kind filter — returns list[ConstraintRecord] (base type)
        rigid = fem.nodes.constraints.by_kind(
            fem.nodes.constraints.Kind.RIGID_BEAM)

        # Flat pair iteration — compound records expanded automatically
        for pair in fem.nodes.constraints.pairs():
            # pair is a NodePairRecord from any source (direct pair,
            # NodeGroupRecord expansion, or NodeToSurfaceRecord expansion)
            ...

        # Raw access to compound records (when you need fields the
        # flattened pairs can't expose, e.g. phantom_coords)
        for nts in fem.nodes.constraints.node_to_surfaces():
            ...

        # Or iterate the set directly for mixed-subclass access
        for rec in fem.nodes.constraints:
            ...

        # Direct indexing
        rec = fem.nodes.constraints[0]
    """

    Kind = ConstraintKind

    # ── Typed iterators ────────────────────────────────────

    def pairs(self) -> Iterator["NodePairRecord"]:
        """Iterate over every constraint as a flat sequence of pairs.

        Compound records (``NodeGroupRecord``, ``NodeToSurfaceRecord``)
        are **expanded automatically** into individual
        ``NodePairRecord`` objects — this is the natural emission
        order for solvers like OpenSees.

        For access to the compound records themselves (e.g. to reach
        ``NodeToSurfaceRecord.phantom_coords``), iterate the set
        directly or use :meth:`node_to_surfaces`.
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

    def node_to_surfaces(self) -> Iterator["NodeToSurfaceRecord"]:
        """Yield only ``NodeToSurfaceRecord`` instances.

        Each record carries the master node, slave surface nodes,
        and the generated phantom nodes for 6-DOF ↔ 3-DOF coupling.
        Use this when you need fields that the flattened
        :meth:`pairs` iterator can't expose (e.g. ``phantom_coords``).
        """
        from apeGmsh.solvers.Constraints import NodeToSurfaceRecord
        for rec in self._records:
            if isinstance(rec, NodeToSurfaceRecord):
                yield rec

    # ── Solver-ready grouped iterators ──────────────────────

    def rigid_link_groups(
        self,
    ) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(master, slaves)`` tuples for all rigid-link constraints.

        Combines every rigid-link source into one grouped stream:

        - ``NodePairRecord(kind='rigid_beam' | 'rigid_rod')``
          → one slave per master
        - ``NodeGroupRecord(kind='rigid_diaphragm' | 'rigid_body')``
          → many slaves per master
        - ``NodeToSurfaceRecord.rigid_link_records``
          → many phantom slaves per master

        Pairs with the same master are grouped together — the natural
        physical shape for OpenSees emission:

        ::

            for master, slaves in fem.nodes.constraints.rigid_link_groups():
                for slave in slaves:
                    ops.rigidLink("beam", master, slave)

        Yields
        ------
        (int, list[int])
            ``(master_node, [slave_node, ...])`` — slave list has at
            least one element.
        """
        from apeGmsh.solvers.Constraints import (
            NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        )

        # Accumulate slaves by master, preserving insertion order
        groups: dict[int, list[int]] = {}

        def _add(master: int, slave: int) -> None:
            groups.setdefault(int(master), []).append(int(slave))

        _RIGID_KINDS = {"rigid_beam", "rigid_rod"}

        for rec in self._records:
            if isinstance(rec, NodePairRecord):
                if rec.kind in _RIGID_KINDS:
                    _add(rec.master_node, rec.slave_node)
            elif isinstance(rec, NodeGroupRecord):
                # rigid_diaphragm, rigid_body, kinematic_coupling
                for sn in rec.slave_nodes:
                    _add(rec.master_node, sn)
            elif isinstance(rec, NodeToSurfaceRecord):
                # master → each phantom (6-DOF rigid link). Skip pairs
                # that the spring variant of node_to_surface generated —
                # those go out through stiff_beam_groups().
                for pair in rec.rigid_link_records:
                    if pair.kind in _RIGID_KINDS:
                        _add(pair.master_node, pair.slave_node)

        for master, slaves in groups.items():
            yield master, slaves

    def stiff_beam_groups(
        self,
    ) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(master, [slaves])`` tuples for stiff-beam-link records.

        Used by the spring variant of :meth:`node_to_surface_spring`:
        instead of OpenSees ``rigidLink('beam', …)`` constraints the
        emission side uses stiff ``elasticBeamColumn`` elements between
        the master and each phantom, giving the master rotation DOFs
        direct element stiffness. This avoids the ill-conditioned
        reduced stiffness that a pure constraint-based rigid link on
        an ``ndf=3`` solid face can produce when the master's rotation
        DOFs are free and directly loaded by a moment.

        Only pair records with ``kind='rigid_beam_stiff'`` are yielded —
        the regular ``rigid_link_groups()`` iterator skips them.

        ::

            # OpenSees emission for the spring variant
            for master, slaves in fem.nodes.constraints.stiff_beam_groups():
                for slave in slaves:
                    ops.element(
                        'elasticBeamColumn', next_eid,
                        master, slave,
                        A_big, E, I_big, I_big, J_big, transf_tag,
                    )
                    next_eid += 1
        """
        from apeGmsh.solvers.Constraints import (
            NodePairRecord, NodeToSurfaceRecord,
        )

        groups: dict[int, list[int]] = {}

        def _add(master: int, slave: int) -> None:
            groups.setdefault(int(master), []).append(int(slave))

        for rec in self._records:
            if isinstance(rec, NodePairRecord):
                if rec.kind == "rigid_beam_stiff":
                    _add(rec.master_node, rec.slave_node)
            elif isinstance(rec, NodeToSurfaceRecord):
                for pair in rec.rigid_link_records:
                    if pair.kind == "rigid_beam_stiff":
                        _add(pair.master_node, pair.slave_node)

        for master, slaves in groups.items():
            yield master, slaves

    def rigid_diaphragms(
        self,
    ) -> Iterator[tuple[int, list[int]]]:
        """Yield ``(master, slaves)`` only for rigid_diaphragm records.

        Useful for OpenSees's native multi-slave command::

            for master, slaves in fem.nodes.constraints.rigid_diaphragms():
                ops.rigidDiaphragm(3, master, *slaves)

        Yields
        ------
        (int, list[int])
            ``(master_node, [slave_node, ...])``
        """
        from apeGmsh.solvers.Constraints import NodeGroupRecord

        for rec in self._records:
            if (isinstance(rec, NodeGroupRecord)
                    and rec.kind == "rigid_diaphragm"):
                yield int(rec.master_node), [
                    int(s) for s in rec.slave_nodes]

    def equal_dofs(self) -> Iterator["NodePairRecord"]:
        """Yield equal_dof pairs — flat iteration.

        Includes:

        - Direct ``NodePairRecord(kind='equal_dof')``
        - ``NodeToSurfaceRecord.equal_dof_records``
          (phantom → original slave, translations only)

        Each pair is independent — the master varies per pair (every
        phantom is its own master in the node_to_surface case).

        Usage
        -----
        ::

            for pair in fem.nodes.constraints.equal_dofs():
                ops.equalDOF(pair.master_node, pair.slave_node, *pair.dofs)
        """
        from apeGmsh.solvers.Constraints import (
            NodePairRecord, NodeToSurfaceRecord,
        )

        for rec in self._records:
            if (isinstance(rec, NodePairRecord)
                    and rec.kind == "equal_dof"):
                yield rec
            elif isinstance(rec, NodeToSurfaceRecord):
                yield from rec.equal_dof_records

    def phantom_nodes(self):
        """Phantom nodes that solvers must create **before** emitting
        constraint commands (created by ``node_to_surface``).

        Returns a :class:`NodeResult` — iterate as ``(node_id, xyz)``
        pairs for clean solver emission::

            for nid, xyz in fem.nodes.constraints.phantom_nodes():
                ops.node(nid, *xyz)

        Or pull the arrays out::

            pn = fem.nodes.constraints.phantom_nodes()
            pn.ids       # ndarray(N,) object dtype
            pn.coords    # ndarray(N, 3) float64

        Returns
        -------
        NodeResult
            Empty NodeResult if no phantom nodes exist.
        """
        from .FEMData import NodeResult
        from apeGmsh.solvers.Constraints import NodeToSurfaceRecord

        ids_list: list[int] = []
        coords_list: list = []
        for rec in self._records:
            if not isinstance(rec, NodeToSurfaceRecord):
                continue
            coords = rec.phantom_coords
            if coords is None:
                continue
            for tag, xyz in zip(rec.phantom_nodes, coords):
                ids_list.append(int(tag))
                coords_list.append(xyz)
        if not ids_list:
            return NodeResult(
                np.array([], dtype=object),
                np.empty((0, 3), dtype=np.float64))
        return NodeResult(
            np.array(ids_list, dtype=object),
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


class NodalLoadSet(_RecordSetBase["NodalLoadRecord"]):
    """Nodal load records (resolved point forces / moments).

    Accessed via ``fem.nodes.loads``.

    Each record is a ``NodalLoadRecord`` with:

    - ``node_id``     : int — mesh node ID
    - ``force_xyz``   : (Fx, Fy, Fz) tuple or ``None``
    - ``moment_xyz``  : (Mx, My, Mz) tuple or ``None``
    - ``pattern``     : str — load pattern name (e.g. ``"default"``)
    - ``name``        : str | None — optional human label

    The record is DOF-agnostic: both vectors are pure 3D spatial
    quantities.  Mapping onto the solver's DOF space is the caller's
    responsibility.

    Examples
    --------
    ::

        # 3D frame (ndf=6)
        for load in fem.nodes.loads:
            fx, fy, fz = load.force_xyz  or (0.0, 0.0, 0.0)
            mx, my, mz = load.moment_xyz or (0.0, 0.0, 0.0)
            ops.load(load.node_id, fx, fy, fz, mx, my, mz)

        # 2D planar frame (ndf=3: ux, uy, rz)
        for load in fem.nodes.loads:
            fx, fy, _  = load.force_xyz  or (0.0, 0.0, 0.0)
            _, _, mz   = load.moment_xyz or (0.0, 0.0, 0.0)
            ops.load(load.node_id, fx, fy, mz)

        # 3D solid (ndf=3: ux, uy, uz)
        for load in fem.nodes.loads:
            fx, fy, fz = load.force_xyz or (0.0, 0.0, 0.0)
            ops.load(load.node_id, fx, fy, fz)

        # Group by pattern (OpenSees timeSeries + pattern)
        for pat in fem.nodes.loads.patterns():
            ops.timeSeries('Linear', tag)
            ops.pattern('Plain', tag, tag)
            for load in fem.nodes.loads.by_pattern(pat):
                fx, fy, fz = load.force_xyz  or (0.0, 0.0, 0.0)
                mx, my, mz = load.moment_xyz or (0.0, 0.0, 0.0)
                ops.load(load.node_id, fx, fy, fz, mx, my, mz)
    """

    Kind = LoadKind

    def patterns(self) -> list[str]:
        """Return all unique pattern names in insertion order."""
        seen: list[str] = []
        for r in self._records:
            if r.pattern not in seen:
                seen.append(r.pattern)
        return seen

    def by_pattern(self, name: str) -> list["NodalLoadRecord"]:
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


class SPSet(_RecordSetBase["SPRecord"]):
    """Single-point constraint records (prescribed displacements / fix).

    Accessed via ``fem.nodes.sp``.

    Each record is an ``SPRecord`` with:

    - ``node_id``        : int — mesh node ID
    - ``dof``            : int — 1-based DOF index
    - ``value``          : float — prescribed displacement (0.0 for fix)
    - ``is_homogeneous`` : bool — ``True`` for plain fix, ``False`` for ``ops.sp``

    Examples
    --------
    ::

        for rec in fem.nodes.sp:
            if rec.is_homogeneous:
                ops.fix(rec.node_id, ...)
            else:
                ops.sp(rec.node_id, rec.dof, rec.value)
    """

    def homogeneous(self) -> list["SPRecord"]:
        """Return only homogeneous (fix) records."""
        return [r for r in self._records if r.is_homogeneous]

    def prescribed(self) -> list["SPRecord"]:
        """Return only non-zero prescribed displacement records."""
        return [r for r in self._records if not r.is_homogeneous]

    def by_node(self, node_id: int) -> list["SPRecord"]:
        """All SP records for a given node."""
        return [r for r in self._records if r.node_id == int(node_id)]

    def __repr__(self) -> str:
        if not self._records:
            return "SPSet(empty)"
        n_hom = sum(1 for r in self._records if r.is_homogeneous)
        n_pre = len(self._records) - n_hom
        return f"SPSet({len(self._records)} records: {n_hom} fix, {n_pre} prescribed)"


class MassSet(_RecordSetBase["MassRecord"]):
    """Resolved nodal masses.

    Accessed via ``fem.nodes.masses``.

    Each record is a ``MassRecord`` with:

    - ``node_id`` : int — mesh node ID
    - ``mass``    : tuple(mx, my, mz, Ixx, Iyy, Izz) — 6-DOF mass vector

    Examples
    --------
    ::

        # Apply masses
        for m in fem.nodes.masses:
            ops.mass(m.node_id, *m.mass)

        # Query total
        print(fem.nodes.masses.total_mass())

        # Lookup by node
        m = fem.nodes.masses.by_node(42)
        if m is not None:
            print(m.mass)

        # Direct indexing
        rec = fem.nodes.masses[0]
        print(rec.node_id, rec.mass)
    """

    def by_node(self, node_id: int) -> "MassRecord | None":
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

class SurfaceConstraintSet(_RecordSetBase["ConstraintRecord"]):
    """Surface coupling constraints.

    Accessed via ``fem.elements.constraints``.

    Record types stored here
    ------------------------
    ==========================  ======================================
    Record type                 Source constraint kinds
    ==========================  ======================================
    ``InterpolationRecord``     ``tie``, ``distributing``, ``embedded``
    ``SurfaceCouplingRecord``   ``tied_contact``, ``mortar``
    ==========================  ======================================

    Record tiers: atomic vs compound
    --------------------------------
    Same two-tier convention as ``NodeConstraintSet``:

    * **Atomic** — ``InterpolationRecord`` (one slave ↔ N weighted
      masters; emission-ready).
    * **Compound** — ``SurfaceCouplingRecord`` wraps a list of
      ``InterpolationRecord`` in ``slave_records`` and additionally
      holds ``mortar_operator`` for the mortar method — that matrix
      is lost if you flatten the wrapper.

    Iterator convention:

    ==================================  =========  =================
    Iterator                            Returns    Expands compound?
    ==================================  =========  =================
    ``interpolations()``                atomic     yes
    ``couplings()``                     compound   no
    direct iter (``for rec in …``)      mixed      no
    ``by_kind(kind)``                   mixed      no
    ==================================  =========  =================

    **Rule of thumb.** If you need ``mortar_operator`` or any other
    compound-only field, iterate ``couplings()``.  Otherwise use
    ``interpolations()`` — coupling records are walked for you.

    Common fields
    -------------
    - ``kind`` (str), ``name`` (str | None) — base fields
    - Subclass-specific fields on ``InterpolationRecord``:
      ``slave_node``, ``master_nodes``, ``weights``, ``dofs``
    - Subclass-specific fields on ``SurfaceCouplingRecord``:
      ``slave_records``, ``mortar_operator``, ``master_nodes``,
      ``slave_nodes``, ``dofs``

    Examples
    --------
    ::

        K = fem.elements.constraints.Kind

        # Iterate interpolations (one per slave node)
        for interp in fem.elements.constraints.interpolations():
            # interp.slave_node, interp.master_nodes, interp.weights
            ...

        # Iterate top-level coupling records
        for coup in fem.elements.constraints.couplings():
            ...

        # Filter by kind
        ties = fem.elements.constraints.by_kind(K.TIE)

        # Direct indexing
        rec = fem.elements.constraints[0]
    """

    Kind = ConstraintKind

    def interpolations(self) -> Iterator["InterpolationRecord"]:
        """Yield ``InterpolationRecord`` objects.

        Covers ``tie``, ``distributing``, ``embedded``, and the
        slave records inside ``tied_contact`` / ``mortar`` (expanded
        automatically).
        """
        from apeGmsh.solvers.Constraints import (
            InterpolationRecord, SurfaceCouplingRecord,
        )

        for rec in self._records:
            if isinstance(rec, InterpolationRecord):
                yield rec
            elif isinstance(rec, SurfaceCouplingRecord):
                yield from rec.slave_records

    def couplings(self) -> Iterator["SurfaceCouplingRecord"]:
        """Yield ``SurfaceCouplingRecord`` objects.

        Returns the top-level coupling records (``mortar``,
        ``tied_contact``) without expanding their slave lists.
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


class ElementLoadSet(_RecordSetBase["ElementLoadRecord"]):
    """Element load records (surface pressure, body forces).

    Accessed via ``fem.elements.loads``.

    Each record is an ``ElementLoadRecord`` with:

    - ``element_id`` : int — mesh element ID
    - ``load_type``  : str — e.g. ``'beamUniform'``, ``'surfacePressure'``
    - ``params``     : dict — solver-specific parameters
    - ``pattern``    : str — load pattern name

    Examples
    --------
    ::

        for eload in fem.elements.loads:
            ops.eleLoad(eload.element_id, eload.load_type, **eload.params)

        # By pattern
        for eload in fem.elements.loads.by_pattern("Wind"):
            ...

        # Direct indexing
        rec = fem.elements.loads[0]
    """

    Kind = LoadKind

    def patterns(self) -> list[str]:
        """Return all unique pattern names in insertion order."""
        seen: list[str] = []
        for r in self._records:
            if r.pattern not in seen:
                seen.append(r.pattern)
        return seen

    def by_pattern(self, name: str) -> list["ElementLoadRecord"]:
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
