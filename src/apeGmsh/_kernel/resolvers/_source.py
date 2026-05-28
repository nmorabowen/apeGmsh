"""Resolver source adapter — Phase 3B.2d / ADR 0038.

The :class:`ResolverSource` Protocol abstracts the small surface that
session composites (``g.loads`` / ``g.masses`` / ``g.constraints``)
use to look up node tags, coordinates, and PG / label resolutions
while building :class:`~apeGmsh._kernel.records` instances.

Two concrete implementations ship in this module:

* :class:`GmshSource` — wraps a live gmsh session (build phase).  All
  queries route to ``gmsh.model.*`` plus the composites' existing
  resolution helpers.
* :class:`FEMDataSource` — wraps a :class:`FEMData` snapshot
  (chain phase).  Queries are answered out of the in-memory broker
  with no gmsh state required.

The Protocol is intentionally narrow — it covers only the lookup
operations the chain-phase shim retrofit (Phase 3B.2d step 3) needs to
resolve a constraint / load / mass def directly into a record using
the broker as the source of truth.  Wider operations (face areas,
edge tributaries, element connectivity walks) remain on the existing
:class:`apeGmsh._kernel.resolvers._load_resolver.LoadResolver` and
sibling resolvers — those are pure mesh math and don't need the source
abstraction.

Future work
-----------
As more chain-phase shim paths get retrofitted, additional methods can
be added here as needed — keep the Protocol minimal.  Foreign-format
adapters (LS-DYNA d3plot, xDMF, Exodus) that round-trip via
:class:`FEMData` get :class:`FEMDataSource` for free.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


@runtime_checkable
class ResolverSource(Protocol):
    """Adapter contract for chain-phase constraint / load / mass resolution.

    Methods return numpy arrays of int64 node IDs (and parallel coord
    arrays where applicable) so the consuming code can index into the
    broker's record-set machinery without any gmsh dependency.

    All methods are name-resolution-aware: pass either a label name
    (Tier 1), a physical-group name (Tier 2), or a concrete container
    that already resolves to nodes.
    """

    def node_ids(self) -> np.ndarray:
        """All node ids in the model (int64 ndarray)."""
        ...

    def node_coords(self) -> np.ndarray:
        """All node coordinates as an ``(N, 3)`` float64 ndarray.

        Indexed parallel to :meth:`node_ids`.
        """
        ...

    def nodes_for(self, target: str) -> np.ndarray:
        """Resolve a target name to a set of node ids.

        ``target`` is the label/PG name passed by the user.  Resolution
        order matches the rest of the API: label (Tier 1) → physical
        group (Tier 2).  Returns an int64 ndarray.

        Raises
        ------
        KeyError
            When the name resolves to neither a label nor a PG, and is
            not a known mesh-selection container either.
        """
        ...

    def has_target(self, target: str) -> bool:
        """``True`` when ``target`` resolves to something in the source.

        Cheap pre-check used by validators that want to surface a clear
        error message before kicking off the full resolution.
        """
        ...


class FEMDataSource:
    """:class:`ResolverSource` backed by a :class:`FEMData` snapshot.

    Used by the chain-phase shim retrofit to resolve constraint / load /
    mass defs into records without touching gmsh — every lookup is
    answered out of the broker's named-group / label / node arrays.

    Construction is cheap (just stashes the ``FEMData`` reference); the
    object is otherwise stateless.  Pass a fresh instance per shim call
    so the source always reflects the latest chain head.
    """

    __slots__ = ("_fem",)

    def __init__(self, fem: "FEMData") -> None:
        self._fem = fem

    # -- ResolverSource methods ------------------------------------

    def node_ids(self) -> np.ndarray:
        return np.asarray(self._fem.nodes.ids, dtype=np.int64)

    def node_coords(self) -> np.ndarray:
        return np.asarray(self._fem.nodes.coords, dtype=np.float64)

    def nodes_for(self, target: str) -> np.ndarray:
        """Walk the broker's name → entity machinery to resolve nodes.

        Looks up ``target`` first in the node-side labels, then the
        node-side physical groups, then the element-side labels / PGs
        (gathering every node touching the resolved elements).  This
        mirrors the live-gmsh ``resolve_to_dimtags`` Tier 1 → Tier 2
        ordering.
        """
        fem = self._fem

        # Tier 1 — labels on node side.
        from apeGmsh._kernel._label_prefix import add_prefix
        prefixed = add_prefix(target)

        for entry in fem.nodes.labels._groups.values():
            entry_name = entry.get("name", "")
            if entry_name in (target, prefixed):
                ids = entry.get("node_ids")
                if ids is not None:
                    return np.asarray(ids, dtype=np.int64)

        # Tier 1 — labels on element side: gather nodes from
        # connectivity of resolved elements.
        for entry in fem.elements.labels._groups.values():
            entry_name = entry.get("name", "")
            if entry_name in (target, prefixed):
                eids = entry.get("element_ids")
                if eids is not None:
                    return self._nodes_from_element_ids(
                        np.asarray(eids, dtype=np.int64)
                    )

        # Tier 2 — physical groups on node side.
        for entry in fem.nodes.physical._groups.values():
            entry_name = entry.get("name", "")
            if entry_name == target:
                ids = entry.get("node_ids")
                if ids is not None:
                    return np.asarray(ids, dtype=np.int64)

        # Tier 2 — physical groups on element side.
        for entry in fem.elements.physical._groups.values():
            entry_name = entry.get("name", "")
            if entry_name == target:
                eids = entry.get("element_ids")
                if eids is not None:
                    return self._nodes_from_element_ids(
                        np.asarray(eids, dtype=np.int64)
                    )

        raise KeyError(
            f"FEMDataSource: target {target!r} resolves to neither a "
            f"label (Tier 1) nor a physical group (Tier 2) in the "
            f"current FEMData chain head."
        )

    def has_target(self, target: str) -> bool:
        """``True`` when :meth:`nodes_for` would succeed without raising."""
        try:
            self.nodes_for(target)
            return True
        except KeyError:
            return False

    # -- Element-side queries (Compose v1.1-A.2 / ADR 0041) --------

    def host_subelements_for(self, target: str) -> np.ndarray:
        """Return ``ndarray(F, 3 | 4)`` of virtual sub-element rows for *target*.

        Element-side counterpart to :meth:`nodes_for` — resolves
        ``target`` to a set of element IDs via Tier 1 (labels) → Tier 2
        (physical groups), pulls each resolved element's ``(etype,
        connectivity_row)`` pair from the broker's element composite,
        and delegates to
        :func:`apeGmsh._kernel.geometry._host_decomposition.decompose_hosts_to_subelements`.

        The returned rows are virtual tri3 / tet4 sub-elements that
        :class:`~apeGmsh._kernel.resolvers._constraint_resolver.ConstraintResolver.resolve_embedded`
        consumes — see ADR 0036 for the coupling semantics and
        ADR 0041 §"Decision 3" for the chain-phase usage contract.

        Parameters
        ----------
        target : str
            Label name (Tier 1) or physical-group name (Tier 2) on the
            element side.  Node-side labels / PGs are NOT consulted —
            this method is element-side only, mirroring how the
            build-phase ``_collect_host_subelements`` resolves
            ``host_entities`` from ``g.parts`` / physical groups
            (never from node-side records).

        Returns
        -------
        ndarray
            ``(F, 4)`` of tet sub-element rows for 3D hosts, ``(F, 3)``
            for 2D hosts.  Empty array ``np.empty((0, 0), dtype=int)``
            when the resolved elements decompose to nothing (e.g. an
            element group that survives the resolution but has zero
            rows after a defensive filter).

        Raises
        ------
        KeyError
            When ``target`` resolves to no element-side label or PG.
        ValueError
            When the resolved elements carry no embeddable host types
            (matching the build-phase error message in
            :meth:`ConstraintsComposite._resolve_embedded`).  Also
            raised by the decomposition function for mixed-dim hosts
            and unsupported etypes.

        Notes
        -----
        Higher-order hosts emit a ``UserWarning`` once per
        ``(etype, target)`` per ADR 0041 §"Decision 8" — chain phase
        has no entity tags to name, so the target label is the natural
        granularity.  Build-phase callers using
        :meth:`ConstraintsComposite._collect_host_subelements` get the
        original per-``(etype, entity)`` warning cadence unchanged.
        """
        import warnings as _warnings

        # Resolve ``target`` to a set of element IDs on the element
        # side only.  Tier 1 (labels) → Tier 2 (PGs).
        eids = self._element_ids_for_target(target)

        # Pull (etype, conn) pairs from the broker for each resolved
        # element by walking every element-type group and filtering.
        fem = self._fem
        eids_set = set(int(x) for x in eids)
        groups: list[tuple[int, np.ndarray]] = []
        for code, grp in fem.elements._groups.items():
            grp_ids = np.asarray(grp.ids, dtype=np.int64)
            mask = np.array(
                [int(t) in eids_set for t in grp_ids], dtype=bool,
            )
            if not mask.any():
                continue
            conn = np.asarray(grp.connectivity, dtype=np.int64)[mask]
            groups.append((int(code), conn))

        if not groups:
            # Resolved label exists but the broker has no element rows
            # matching those ids — defensive, normally unreachable
            # because the element groups are the source of truth for
            # ids.
            raise ValueError(
                f"FEMDataSource.host_subelements_for: target "
                f"{target!r} resolved to {len(eids)} element ids but "
                f"none survive the broker's element-type filter.  "
                f"This is a broker invariant violation — the label "
                f"references element ids that no ElementGroup holds."
            )

        # Higher-order warning per (etype, target) — ADR 0041 §8.
        warned_codes: set[int] = set()

        def _warn(code: int, name: str) -> None:
            if code in warned_codes:
                return
            warned_codes.add(code)
            _warnings.warn(
                f"embedded: host target {target!r} carries {name} "
                f"elements — decomposing to corner-node-only linear "
                f"sub-elements.  The embedded coupling will be linear "
                f"regardless of the host's native interpolation "
                f"order; quadratic / bilinear / trilinear host "
                f"kinematics will NOT be felt by the embedded node.  "
                f"Set `host_coupling=` explicitly on the "
                f"`embedded(...)` call to acknowledge.",
                UserWarning, stacklevel=4,
            )

        from apeGmsh._kernel.geometry._host_decomposition import (
            decompose_hosts_to_subelements,
        )
        sub = decompose_hosts_to_subelements(
            groups, warn_higher_order=_warn,
        )
        if sub.size == 0:
            raise ValueError(
                f"FEMDataSource.host_subelements_for: target "
                f"{target!r} resolved to elements but none carry "
                f"embeddable host types.  Supported host types: "
                f"tri3 / tri6 / quad4 / quad8 / quad9 (2D); tet4 / "
                f"tet10 / hex8 / hex20 / prism6 / prism15 / pyramid5 "
                f"/ pyramid13 (3D).  Non-simplex and higher-order "
                f"hosts are decomposed to linear sub-tris / sub-tets "
                f"using corner nodes only.  Fix the broker's element "
                f"composition before composing a chain-phase "
                f"embedded constraint."
            )
        return sub

    def boundary_faces_for(self, target: str) -> np.ndarray:
        """Return ``ndarray(F, n_fpn)`` of surface face rows owned by *target*.

        Element-side counterpart to :meth:`host_subelements_for` for the
        tied-contact / mortar code path — resolves ``target`` to a node
        set via :meth:`nodes_for` (Tier 1 labels → Tier 2 PGs on both
        node and element sides), then filters the broker's **dim=2
        element groups** to the connectivity rows whose every node is
        in that set.  Mirrors the build-phase pattern in
        :meth:`PartsRegistry.build_face_map`
        (``src/apeGmsh/core/_parts_registry.py``).

        Parameters
        ----------
        target : str
            Label name (Tier 1) or physical-group name (Tier 2).  Unlike
            :meth:`host_subelements_for` this method walks the full
            node-resolution machinery (node-side and element-side
            tiers) — a tied-contact interface is naturally named by the
            surface PG that already lives on the node side of the
            broker.

        Returns
        -------
        ndarray
            ``(n_faces, n_fpn)`` int64 connectivity rows.  Returns
            ``np.empty((0, 0), dtype=int)`` when dim=2 element groups
            exist in the broker but no face row is fully owned by the
            target's node set — consistent with the build-phase
            :meth:`PartsRegistry.build_face_map` empty-instance
            convention.

        Raises
        ------
        KeyError
            When ``target`` resolves to no label / PG (propagated from
            :meth:`nodes_for`).
        ValueError
            When the broker contains **no dim=2 ElementGroups at all**
            (ADR 0041 §"Decision 5" — chain phase does not synthesize
            faces from volume elements; the broker must already carry
            the surface mesh).  Also raised when the broker carries
            multiple dim=2 element-types with different nodes-per-face
            (e.g. tri3 + quad4) and more than one of them survives the
            node-ownership filter — the resolver downstream expects a
            single rectangular ``(n_faces, n_fpn)`` array.

        Notes
        -----
        Per ADR 0041 §"Decision 3" this is a separate concrete-class
        method on :class:`FEMDataSource` (not on the
        :class:`ResolverSource` Protocol) and §"Decision 5" pins the
        no-volume-synthesis contract.  When you hit the "no dim=2
        ElementGroups" ``ValueError``, the remedy is to re-extract the
        broker with ``dim=None`` (so the surface mesh is captured
        alongside the volume) and re-save.
        """
        fem = self._fem

        # Resolve the target to its node set via the full Tier 1 → 2
        # walk on both node and element sides.  KeyError propagates.
        node_ids = self.nodes_for(target)
        node_set = set(int(n) for n in node_ids)

        # Find every dim=2 ElementGroup in the broker.
        surface_groups: list[tuple[int, np.ndarray]] = []
        for code, grp in fem.elements._groups.items():
            if int(grp.element_type.dim) != 2:
                continue
            surface_groups.append(
                (int(code), np.asarray(grp.connectivity, dtype=np.int64)),
            )

        if not surface_groups:
            raise ValueError(
                f"FEMDataSource.boundary_faces_for: target {target!r} "
                f"resolves to a node set, but the broker carries no "
                f"dim=2 ElementGroups — chain phase does not synthesize "
                f"face connectivity from volume elements (ADR 0041 "
                f"§\"Decision 5\").  Remedy: re-extract the FEMData "
                f"with `dim=None` so the surface mesh is captured "
                f"alongside the volume, then re-save."
            )

        # Filter each surface group's connectivity to rows fully owned
        # by the target's node set.  ``np.isin`` against the resolved
        # set vectorises the same predicate
        # :meth:`PartsRegistry.build_face_map` uses.
        if not node_set:
            return np.empty((0, 0), dtype=np.int64)

        node_arr = np.asarray(sorted(node_set), dtype=np.int64)
        kept_blocks: list[np.ndarray] = []
        seen_npe: set[int] = set()
        for _code, conn in surface_groups:
            mask = np.all(np.isin(conn, node_arr), axis=1)
            if not mask.any():
                continue
            kept = conn[mask]
            seen_npe.add(int(kept.shape[1]))
            kept_blocks.append(kept)

        if not kept_blocks:
            # dim=2 groups exist but none of their rows are fully owned
            # by this target — consistent with build_face_map's empty-
            # instance return.  Caller treats this as "empty interface".
            return np.empty((0, 0), dtype=np.int64)

        if len(seen_npe) > 1:
            raise ValueError(
                f"FEMDataSource.boundary_faces_for: target {target!r} "
                f"resolves to mixed surface element types with "
                f"different nodes-per-face ({sorted(seen_npe)}).  The "
                f"chain-phase tied-contact resolver expects a single "
                f"rectangular (n_faces, n_fpn) connectivity array.  "
                f"Split the target into per-element-type sub-PGs and "
                f"compose them separately."
            )

        return np.vstack(kept_blocks)

    # -- Internal helpers ------------------------------------------

    def _element_ids_for_target(self, target: str) -> np.ndarray:
        """Resolve ``target`` to an int64 ndarray of element IDs.

        Element-side mirror of :meth:`nodes_for`'s walk: Tier 1
        labels on the element side first, then Tier 2 PGs.  Node-side
        records are never consulted — embedded host targets are
        element-side by construction.
        """
        from apeGmsh._kernel._label_prefix import add_prefix

        fem = self._fem
        prefixed = add_prefix(target)

        # Tier 1 — element-side labels.
        for entry in fem.elements.labels._groups.values():
            entry_name = entry.get("name", "")
            if entry_name in (target, prefixed):
                eids = entry.get("element_ids")
                if eids is not None:
                    return np.asarray(eids, dtype=np.int64)

        # Tier 2 — element-side physical groups.
        for entry in fem.elements.physical._groups.values():
            entry_name = entry.get("name", "")
            if entry_name == target:
                eids = entry.get("element_ids")
                if eids is not None:
                    return np.asarray(eids, dtype=np.int64)

        raise KeyError(
            f"FEMDataSource.host_subelements_for: target {target!r} "
            f"resolves to neither an element-side label (Tier 1) nor a "
            f"physical group (Tier 2) in the current FEMData chain "
            f"head.  Embedded hosts are element-side; pass a label or "
            f"PG name that names a set of host elements."
        )

    def _nodes_from_element_ids(self, eids: np.ndarray) -> np.ndarray:
        """Collect the union of node ids referenced by the given elements.

        Walks every element-type group in the broker's element
        composite; for each group whose ids overlap ``eids`` it pulls
        the corresponding connectivity rows and unions their nodes.
        Returns int64 ndarray (sorted, no duplicates).
        """
        fem = self._fem
        eids_set = set(int(x) for x in eids)
        out: set[int] = set()
        for code, grp in fem.elements._groups.items():
            grp_ids = np.asarray(grp.ids, dtype=np.int64)
            mask = np.array(
                [int(t) in eids_set for t in grp_ids], dtype=bool,
            )
            if not mask.any():
                continue
            conn = np.asarray(grp.connectivity, dtype=np.int64)[mask]
            for row in conn:
                out.update(int(n) for n in row)
        return np.array(sorted(out), dtype=np.int64)


class GmshSource:
    """:class:`ResolverSource` backed by the live gmsh session.

    Build-phase callers wrap their session in this adapter and pass it
    to the chain-phase router so the same code path works for both
    phases.  In practice the build-phase shims continue using their
    existing resolution helpers directly (no behaviour change); this
    adapter exists so future code that needs to be source-agnostic can
    consume the Protocol uniformly.

    Construction is cheap.  Queries are evaluated lazily.
    """

    __slots__ = ("_session",)

    def __init__(self, session) -> None:
        self._session = session

    # -- ResolverSource methods ------------------------------------

    def node_ids(self) -> np.ndarray:
        import gmsh
        tags, _coords, _ = gmsh.model.mesh.getNodes()
        return np.asarray(tags, dtype=np.int64)

    def node_coords(self) -> np.ndarray:
        import gmsh
        _tags, coords, _ = gmsh.model.mesh.getNodes()
        return np.asarray(coords, dtype=np.float64).reshape(-1, 3)

    def nodes_for(self, target: str) -> np.ndarray:
        """Use the central :func:`resolve_to_dimtags` helper + gmsh mesh
        queries to collect node tags.
        """
        import gmsh
        from apeGmsh.core._helpers import resolve_to_dimtags

        dts = resolve_to_dimtags(
            target, default_dim=3, session=self._session,
        )
        if not dts:
            raise KeyError(
                f"GmshSource: target {target!r} resolves to no entities."
            )

        out: set[int] = set()
        for d, t in dts:
            tags, _coords = gmsh.model.mesh.getNodesForPhysicalGroup(d, t) \
                if False else (np.array([], dtype=np.int64), None)
            # Use getNodes on the dim/tag (covers entity nodes, not PG):
            ntags, _ = gmsh.model.mesh.getNodes(dim=d, tag=t, includeBoundary=True)[:2]
            out.update(int(n) for n in ntags)
        return np.array(sorted(out), dtype=np.int64)

    def has_target(self, target: str) -> bool:
        try:
            self.nodes_for(target)
            return True
        except Exception:
            return False


def make_source(session_or_fem) -> ResolverSource:
    """Pick the right :class:`ResolverSource` for the given argument.

    Convenience constructor used by the chain-phase router — accepts
    either an :class:`apeGmsh` session or a :class:`FEMData` snapshot
    and returns the matching adapter.  Discriminates by attribute
    presence (FEMData has ``.nodes`` + ``.elements`` composites;
    session has ``.mesh`` plus ``.model``).
    """
    # FEMData carries ``.composed_from`` and ``.elements._groups`` —
    # session carries ``.mesh`` + ``.model``.  Use ``composed_from`` as
    # the discriminator since it is the most cohesive FEMData-only
    # attribute introduced in 2.9 schema.
    if hasattr(session_or_fem, "composed_from") and not hasattr(
        session_or_fem, "_compose_facade",
    ):
        return FEMDataSource(session_or_fem)
    return GmshSource(session_or_fem)
