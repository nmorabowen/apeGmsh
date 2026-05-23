"""
_Partitioning — mesh partitioning and node/element renumbering.

Accessed via ``g.mesh.partitioning``.  The single home for:

* **Renumbering** — contiguous IDs (``simple``) or bandwidth-optimised
  orderings (``rcm``, ``hilbert``, ``metis``).  Mutates the Gmsh model
  so that ``get_fem_data()`` produces solver-ready tags.
* **Partitioning** — MPI-style domain decomposition via Gmsh/METIS.
  Partition membership is captured in ``FEMData`` and queryable via
  ``fem.nodes.select(partition=2)``.

  Two flavours are supported:

  * **Flavor A** (default) — ``partition(n)`` calls Gmsh's native METIS
    binding, which balances element *count* only (no vertex weights).
  * **Flavor B** — ``partition(n, weights=...)`` routes through an
    external METIS binding (``pymetis`` or ``networkx-metis``) so the
    caller can supply per-element vertex weights.  apeGmsh builds the
    element dual graph from Gmsh's connectivity, calls the backend,
    then pushes the assignment back into Gmsh via ``partition_explicit``
    — downstream consumers (FEMData broker, h5 round-trip, OpenSees
    bridge) see the same model state as the native path.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import gmsh
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from .Mesh import Mesh

# Backend literal — public alias surfaced in the partition() signature.
PartitionBackend = Literal["gmsh", "pymetis", "networkx-metis"]


# =====================================================================
# Output contracts
# =====================================================================

class RenumberResult:
    """Result of a mesh renumbering operation.

    Attributes
    ----------
    method : str
        Algorithm used (``"simple"``, ``"rcm"``, ``"hilbert"``, ``"metis"``).
    n_nodes : int
        Number of nodes renumbered.
    n_elements : int
        Number of elements renumbered.
    bandwidth_before : int
        Semi-bandwidth before renumbering.
    bandwidth_after : int
        Semi-bandwidth after renumbering.
    """

    __slots__ = ('method', 'n_nodes', 'n_elements',
                 'bandwidth_before', 'bandwidth_after')

    def __init__(
        self,
        method: str,
        n_nodes: int,
        n_elements: int,
        bandwidth_before: int,
        bandwidth_after: int,
    ) -> None:
        self.method = method
        self.n_nodes = n_nodes
        self.n_elements = n_elements
        self.bandwidth_before = bandwidth_before
        self.bandwidth_after = bandwidth_after

    def __repr__(self) -> str:
        if self.bandwidth_after > 0:
            ratio = self.bandwidth_before / self.bandwidth_after
            return (
                f"RenumberResult({self.method}): "
                f"{self.n_nodes} nodes, {self.n_elements} elements, "
                f"bw {self.bandwidth_before}\u2192{self.bandwidth_after} "
                f"({ratio:.1f}\u00d7)")
        return (
            f"RenumberResult({self.method}): "
            f"{self.n_nodes} nodes, {self.n_elements} elements, "
            f"bw {self.bandwidth_before}\u2192{self.bandwidth_after}")


class PartitionInfo:
    """Result of a mesh partitioning operation.

    Attributes
    ----------
    n_parts : int
        Number of partitions created.
    elements_per_partition : dict[int, int]
        ``{partition_id: element_count}``.
    weights_per_partition : dict[int, float] | None
        ``{partition_id: total_weight}`` when ``partition()`` was called
        with ``weights=``, otherwise ``None``.  Populated by
        ``_gather_partition_info()`` from the per-element weight vector
        cached on ``_Partitioning`` during the weighted call.
    """

    __slots__ = (
        'n_parts',
        'elements_per_partition',
        'weights_per_partition',
    )

    def __init__(
        self,
        n_parts: int,
        elements_per_partition: dict[int, int],
        weights_per_partition: dict[int, float] | None = None,
    ) -> None:
        self.n_parts = n_parts
        self.elements_per_partition = elements_per_partition
        self.weights_per_partition = weights_per_partition

    def __repr__(self) -> str:
        counts = ", ".join(
            f"P{k}:{v}"
            for k, v in sorted(self.elements_per_partition.items()))
        if self.weights_per_partition is None:
            return f"PartitionInfo({self.n_parts} parts: {counts})"
        weights = ", ".join(
            f"P{k}:{v:.3g}"
            for k, v in sorted(self.weights_per_partition.items()))
        return (
            f"PartitionInfo({self.n_parts} parts: {counts}; "
            f"weights[{weights}])")


# =====================================================================
# Gmsh method name mapping
# =====================================================================

_METHOD_MAP: dict[str, str] = {
    "rcm":     "RCMK",
    "hilbert": "Hilbert",
    "metis":   "Metis",
}


# =====================================================================
# Composite
# =====================================================================

class _Partitioning:
    """Mesh partitioning plus node / element renumbering.

    Accessed via ``g.mesh.partitioning``.
    """

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh
        # Per-element weights from the most-recent weighted ``partition()``
        # call, keyed by Gmsh element tag. Cleared on every ``partition*()``
        # entry so ``summary()``/``_gather_partition_info()`` know whether
        # to populate ``PartitionInfo.weights_per_partition``.
        self._last_weights: dict[int, float] | None = None

    # ------------------------------------------------------------------
    # Renumbering
    # ------------------------------------------------------------------

    def renumber(
        self,
        dim: int = 2,
        *,
        method: str = "rcm",
        base: int = 1,
    ) -> RenumberResult:
        """Renumber nodes and elements in the Gmsh model.

        After this call every Gmsh query returns solver-ready contiguous
        IDs.  Call **once**, before extracting FEM data with
        :meth:`~_Queries.get_fem_data`.

        Parameters
        ----------
        dim : int
            Element dimension used to compute bandwidth and to collect
            element tags for renumbering.
        method : ``"simple"`` | ``"rcm"`` | ``"hilbert"`` | ``"metis"``
            ``"simple"``  — contiguous IDs, no optimisation.
            ``"rcm"``     — Reverse Cuthill-McKee (bandwidth reduction).
            ``"hilbert"`` — Hilbert space-filling curve (cache locality).
            ``"metis"``   — METIS graph-partitioner ordering.
        base : int
            Starting ID (default 1 = OpenSees / Abaqus convention).

        Returns
        -------
        RenumberResult
        """
        from ._fem_extract import extract_raw
        from .FEMData import _compute_bandwidth
        from ._fem_factory import _build_element_groups

        # 1. Bandwidth BEFORE ────────────────────────────────────
        raw = extract_raw(dim=dim)
        groups = _build_element_groups(raw['groups'])
        bw_before = _compute_bandwidth(groups)
        n_nodes = len(raw['node_tags'])
        n_elems = len(raw['elem_tags'])

        # 2. Node renumbering ────────────────────────────────────
        if method == "simple":
            self._renumber_nodes_simple(base)
        elif method in _METHOD_MAP:
            old, new = gmsh.model.mesh.computeRenumbering(
                method=_METHOD_MAP[method])
            gmsh.model.mesh.renumberNodes(
                oldTags=list(old), newTags=list(new))
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"Use 'simple', 'rcm', 'hilbert', or 'metis'.")

        # 3. Element renumbering (always simple contiguous) ──────
        self._renumber_elements_simple(dim, base)

        # 4. Bandwidth AFTER ─────────────────────────────────────
        raw_after = extract_raw(dim=dim)
        groups_after = _build_element_groups(raw_after['groups'])
        bw_after = _compute_bandwidth(groups_after)

        result = RenumberResult(
            method=method,
            n_nodes=n_nodes,
            n_elements=n_elems,
            bandwidth_before=bw_before,
            bandwidth_after=bw_after,
        )
        self._mesh._log(
            f"renumber(method={method!r}, dim={dim}): "
            f"{n_nodes} nodes, {n_elems} elements, "
            f"bw {bw_before}\u2192{bw_after}")
        return result

    # ── internal helpers ─────────────────────────────────────────

    @staticmethod
    def _renumber_nodes_simple(base: int) -> None:
        """Assign contiguous node tags starting from *base*."""
        tags, _, _ = gmsh.model.mesh.getNodes()
        old = np.sort(np.asarray(tags, dtype=np.int64))
        new = np.arange(base, base + len(old), dtype=np.int64)
        gmsh.model.mesh.renumberNodes(
            oldTags=old.tolist(), newTags=new.tolist())

    @staticmethod
    def _renumber_elements_simple(dim: int, base: int) -> None:
        """Assign contiguous element tags for *dim* starting from *base*."""
        _, etags_list, _ = gmsh.model.mesh.getElements(dim=dim, tag=-1)
        all_tags: list[int] = []
        for etags in etags_list:
            all_tags.extend(int(t) for t in etags)
        if not all_tags:
            return
        old = np.array(sorted(all_tags), dtype=np.int64)
        new = np.arange(base, base + len(old), dtype=np.int64)
        gmsh.model.mesh.renumberElements(
            oldTags=old.tolist(), newTags=new.tolist())

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def partition(
        self,
        n_parts: int,
        *,
        weights: Sequence[float] | np.ndarray | None = None,
        backend: PartitionBackend | None = None,
    ) -> PartitionInfo:
        """Partition the mesh into *n_parts* sub-domains.

        Must be called after ``g.mesh.generation.generate()``.

        Two flavours, dispatched by ``weights`` / ``backend``:

        ============= ================== ==================================
        ``weights``   ``backend``        Path
        ============= ================== ==================================
        ``None``      ``None``           Gmsh-native METIS (unweighted)
        ``None``      ``"gmsh"``         Gmsh-native METIS (explicit alias)
        ``None``      ``"pymetis"``      pymetis with ``vwgt = ones(n)``
        ``None``      ``"networkx-...``  networkx-metis, unit weights
        sequence      ``None``           pymetis (default for weighted)
        sequence      ``"pymetis"``      pymetis with the given weights
        sequence      ``"networkx-...``  networkx-metis with weights
        sequence      ``"gmsh"``         ``ValueError`` — Gmsh has no vwgt
        ============= ================== ==================================

        Parameters
        ----------
        n_parts : int
            Number of partitions (>= 1).
        weights : sequence of float, optional
            Per-element vertex weights for METIS.  Length must match the
            total number of elements across **all** dimensions, ordered
            by flattened ``gmsh.model.mesh.getElements(dim=-1, tag=-1)``
            traversal — the same order used to drive
            ``partition_explicit`` internally.
        backend : ``"gmsh"`` | ``"pymetis"`` | ``"networkx-metis"``, optional
            Backend selector.  See dispatch table above.

        Returns
        -------
        PartitionInfo
            ``weights_per_partition`` is populated when ``weights`` was
            passed, else ``None``.

        Raises
        ------
        ValueError
            On ``n_parts < 1``, ``backend="gmsh"`` with ``weights``, or
            wrong-length ``weights``.
        ImportError
            When the selected backend is not installed (with a
            ``pip install apeGmsh[partition-pymetis]`` hint).
        """
        if n_parts < 1:
            raise ValueError(f"n_parts must be >= 1, got {n_parts}")

        # Default backend resolution.
        if backend is None:
            backend = "pymetis" if weights is not None else "gmsh"

        if backend == "gmsh":
            if weights is not None:
                raise ValueError(
                    "Gmsh has no vwgt API; use backend='pymetis' or "
                    "backend='networkx-metis' to pass weights.")
            self._last_weights = None
            gmsh.model.mesh.partition(n_parts)
            info = self._gather_partition_info()
            self._mesh._log(f"partition(n_parts={n_parts})")
            return info

        if backend not in ("pymetis", "networkx-metis"):
            raise ValueError(
                f"Unknown backend {backend!r}. "
                "Use 'gmsh', 'pymetis', or 'networkx-metis'.")

        return self._partition_weighted(n_parts, weights, backend)

    def partition_explicit(
        self,
        n_parts: int,
        elem_tags: list[int],
        parts: list[int],
    ) -> PartitionInfo:
        """Partition with an explicit per-element assignment.

        Parameters
        ----------
        n_parts : int
            Total number of partitions declared.
        elem_tags : list[int]
            Element tags to assign.
        parts : list[int]
            Parallel list of 1-based partition IDs.

        Returns
        -------
        PartitionInfo
        """
        if len(elem_tags) != len(parts):
            raise ValueError(
                f"len(elem_tags)={len(elem_tags)} != "
                f"len(parts)={len(parts)}")
        # External callers of partition_explicit don't supply weights;
        # clear so ``_gather_partition_info`` reports unweighted state.
        # The internal weighted path sets ``_last_weights`` *before*
        # calling ``partition_explicit`` and re-asserts it after, so its
        # cache survives this clear.
        self._last_weights = None
        gmsh.model.mesh.partition(
            n_parts, elementTags=elem_tags, partitions=parts)
        info = self._gather_partition_info()
        self._mesh._log(
            f"partition_explicit(n_parts={n_parts}, "
            f"n_elements={len(elem_tags)})")
        return info

    def unpartition(self) -> "_Partitioning":
        """Remove the partition structure and restore a monolithic mesh."""
        self._last_weights = None
        gmsh.model.mesh.unpartition()
        self._mesh._log("unpartition()")
        return self

    # ── weighted path (Flavor B) ─────────────────────────────────

    def _partition_weighted(
        self,
        n_parts: int,
        weights: Sequence[float] | np.ndarray | None,
        backend: PartitionBackend,
    ) -> PartitionInfo:
        """Build the dual graph, call METIS, round-trip via Gmsh.

        Parameters
        ----------
        n_parts : int
            Number of partitions (already validated by caller).
        weights : sequence or None
            Per-element vertex weights, or ``None`` (= unit weights).
        backend : ``"pymetis"`` | ``"networkx-metis"``
            METIS binding to use.  Caller has already screened ``"gmsh"``.
        """
        # 1. Collect every element across every dim, in a stable order.
        elem_tags = self._collect_all_element_tags()
        n_elems = len(elem_tags)
        if n_elems == 0:
            raise RuntimeError(
                "No elements to partition. Call "
                "g.mesh.generation.generate() first.")

        # 2. Validate / normalise the weight vector.
        if weights is None:
            w_arr = np.ones(n_elems, dtype=np.int64)
        else:
            w_seq = list(weights)
            if len(w_seq) != n_elems:
                raise ValueError(
                    f"weights length mismatch: expected {n_elems} "
                    f"(total elements across all dims), got "
                    f"{len(w_seq)}.")
            # Preserve the user's float view for ``weights_per_partition``
            # but pass METIS an integer vector (pymetis demands int32+,
            # networkx-metis accepts either but is happiest with ints).
            # Scale so we don't lose resolution for sub-unit weights.
            w_arr_f = np.asarray(w_seq, dtype=np.float64)
            scale = self._weight_scale_for_metis(w_arr_f)
            w_arr = np.maximum(
                np.rint(w_arr_f * scale).astype(np.int64), 1)

        # 3. Build adjacency (element dual graph) from Gmsh connectivity.
        adjacency = self._build_dual_graph(elem_tags)

        # 4. Call the backend.
        parts_0 = self._call_backend(
            backend=backend,
            n_parts=n_parts,
            adjacency=adjacency,
            weights=w_arr,
        )

        # 5. Cache per-element float weights (or None for unit weights)
        #    so ``_gather_partition_info`` can populate
        #    ``weights_per_partition``.
        if weights is None:
            weight_cache: dict[int, float] | None = None
        else:
            weight_cache = {
                int(tag): float(w)
                for tag, w in zip(elem_tags, weights)
            }
        # Set BEFORE partition_explicit (which clears it) and restore
        # AFTER (which calls _gather_partition_info — needs the cache).
        # partition_explicit clears self._last_weights → we restore it
        # right after the call below.
        parts_1 = [int(p) + 1 for p in parts_0]  # Gmsh expects 1-based.
        self.partition_explicit(n_parts, elem_tags, parts_1)
        self._last_weights = weight_cache
        # Re-gather with the now-populated cache so the returned info
        # carries weights_per_partition.
        info = self._gather_partition_info()
        self._mesh._log(
            f"partition(n_parts={n_parts}, weights="
            f"{'<vec>' if weights is not None else 'None'}, "
            f"backend={backend!r})")
        return info

    @staticmethod
    def _weight_scale_for_metis(w: np.ndarray) -> float:
        """Pick a scale factor that maps float weights to ints.

        METIS uses integer vertex weights; sub-unit floats would round
        to zero.  We scale so the smallest non-zero weight maps to ~1000
        (enough resolution to distinguish a 100x ratio).
        """
        positive = w[w > 0]
        if positive.size == 0:
            return 1.0
        wmin = float(positive.min())
        if wmin >= 1.0:
            return 1.0
        return 1000.0 / wmin

    @staticmethod
    def _collect_all_element_tags() -> list[int]:
        """Flatten every element tag across every dim in Gmsh's order.

        This is the ordering contract for the ``weights`` argument:
        ``gmsh.model.mesh.getElements(dim=-1, tag=-1)`` traversed by
        ``(dim, etype)`` in Gmsh's native iteration order.
        """
        all_tags: list[int] = []
        for d in range(4):
            _, etags_list, _ = gmsh.model.mesh.getElements(dim=d, tag=-1)
            for etags in etags_list:
                all_tags.extend(int(t) for t in etags)
        return all_tags

    @staticmethod
    def _build_dual_graph(elem_tags: list[int]) -> list[np.ndarray]:
        """Build CSR-style adjacency from Gmsh's element connectivity.

        Adjacency rule per dimension:

        * **dim=3 (volumes)** — two elements adjacent iff they share a
          face (``getElementFaceNodes`` with the type's primary face).
        * **dim=2 (surfaces)** — two elements adjacent iff they share an
          edge (``getElementEdgeNodes``).
        * **dim=1 (lines)** — share a node.
        * **dim=0 (points)** — never adjacent (METIS will treat each as
          its own component; partitioning still works since Gmsh routes
          isolated points to any partition).

        Returns
        -------
        list of length ``len(elem_tags)`` — each entry is an ``int32``
        array of neighbour indices (into the ``elem_tags`` order).  This
        is the format both ``pymetis.part_graph`` (xadj/adjncy-style)
        and ``networkx-metis`` accept after a thin conversion.
        """
        # tag -> position in the flattened ``elem_tags`` ordering.
        idx_of = {tag: i for i, tag in enumerate(elem_tags)}

        # face/edge-key -> list of element indices sharing that key.
        # Keys are sorted-tuple-of-node-ids so orientation doesn't
        # produce false negatives.
        bucket: dict[tuple[int, ...], list[int]] = {}

        for d in (3, 2, 1):
            etypes, etags_list, _ = gmsh.model.mesh.getElements(
                dim=d, tag=-1)
            if len(etypes) == 0:
                continue

            for etype, etags in zip(etypes, etags_list):
                if len(etags) == 0:
                    continue

                # Resolve element family → (face/edge nodes-per-key, count
                # of keys per element).
                props = gmsh.model.mesh.getElementProperties(int(etype))
                _name, _dim, _order, num_nodes, _, _ = props

                if d == 3:
                    # Volume — primary face: pick the smallest face arity
                    # available (tet/hex/prism/pyramid all expose at
                    # least 3-node faces).
                    face_arity = 3
                    nodes_flat = (
                        gmsh.model.mesh.getElementFaceNodes(
                            int(etype), face_arity))
                    # nodes_flat has shape n_elem * n_faces_per_elem *
                    # face_arity, flattened.  We don't know n_faces a
                    # priori per type — derive from length.
                    n_e = len(etags)
                    n_keys_per_elem = (
                        len(nodes_flat) // (face_arity * n_e))
                elif d == 2:
                    # Surface — adjacency via shared *edges* (2-node
                    # keys).  getElementEdgeNodes packs node pairs.
                    face_arity = 2
                    nodes_flat = gmsh.model.mesh.getElementEdgeNodes(
                        int(etype))
                    n_e = len(etags)
                    n_keys_per_elem = (
                        len(nodes_flat) // (face_arity * n_e))
                else:  # d == 1
                    # Line — adjacency via shared *nodes* (1-node keys).
                    face_arity = 1
                    # All node tags for each element, in element order.
                    _, e_nodes = gmsh.model.mesh.getElementsByType(
                        int(etype))
                    nodes_flat = e_nodes
                    n_e = len(etags)
                    n_keys_per_elem = num_nodes

                # Walk each element's keys; bucket by sorted-node-tuple.
                ptr = 0
                for i_e, tag in enumerate(etags):
                    pos = idx_of[int(tag)]
                    for _ in range(n_keys_per_elem):
                        key_nodes = nodes_flat[ptr:ptr + face_arity]
                        ptr += face_arity
                        key = tuple(
                            sorted(int(n) for n in key_nodes))
                        bucket.setdefault(key, []).append(pos)

        # Convert bucket → adjacency lists.
        adj_sets: list[set[int]] = [set() for _ in elem_tags]
        for sharers in bucket.values():
            if len(sharers) < 2:
                continue
            # All-pairs within the bucket are neighbours.  In well-formed
            # meshes len(sharers) is 1 (boundary) or 2 (interior face/
            # edge); for non-manifold meshes it may be larger, which the
            # n^2 walk still handles correctly.
            for a in sharers:
                for b in sharers:
                    if a != b:
                        adj_sets[a].add(b)

        return [
            np.asarray(sorted(s), dtype=np.int32)
            for s in adj_sets
        ]

    def _call_backend(
        self,
        *,
        backend: PartitionBackend,
        n_parts: int,
        adjacency: list[np.ndarray],
        weights: np.ndarray,
    ) -> list[int]:
        """Invoke the chosen METIS binding; return 0-indexed parts list."""
        if backend == "pymetis":
            pymetis = self._import_backend(
                "pymetis", "partition-pymetis")
            # pymetis.part_graph signature:
            #   part_graph(nparts, adjacency=[...], vweights=[...])
            #     -> (edgecuts, membership)
            _cuts, membership = pymetis.part_graph(
                int(n_parts),
                adjacency=[a.tolist() for a in adjacency],
                vweights=weights.astype(np.int32).tolist(),
            )
            return [int(p) for p in membership]

        # networkx-metis
        nx = self._import_backend("networkx", "partition-networkx-metis")
        nxmetis = self._import_backend(
            "nxmetis", "partition-networkx-metis")
        g = nx.Graph()
        for i, w in enumerate(weights):
            g.add_node(i, weight=int(w))
        for i, neigh in enumerate(adjacency):
            for j in neigh:
                if int(j) > i:  # undirected — add each edge once
                    g.add_edge(i, int(j))
        _cuts, parts = nxmetis.partition(
            g, int(n_parts), node_weight='weight')
        # nxmetis returns a list of lists (one per partition).  Flatten
        # into a per-vertex partition vector.
        membership = [0] * len(adjacency)
        for pid, members in enumerate(parts):
            for v in members:
                membership[int(v)] = pid
        return membership

    @staticmethod
    def _import_backend(modname: str, extras_name: str):
        """Import *modname* or raise ``ImportError`` with a pip hint."""
        try:
            return importlib.import_module(modname)
        except ImportError as exc:
            raise ImportError(
                f"Backend {modname!r} is not installed. "
                f"Install it with: pip install apeGmsh[{extras_name}]"
            ) from exc

    # ── internal ─────────────────────────────────────────────────

    def _gather_partition_info(self) -> PartitionInfo:
        """Query Gmsh to build :class:`PartitionInfo` after partitioning.

        When ``self._last_weights`` is populated (set by the weighted
        path before/after the underlying ``partition_explicit`` call),
        the per-partition weight totals are computed by walking every
        partitioned entity's element tags and summing the cached
        per-element weights.  Otherwise ``weights_per_partition`` is
        left ``None``.
        """
        n = gmsh.model.getNumberOfPartitions()
        elems_per: dict[int, int] = {}
        weights = self._last_weights
        weights_per: dict[int, float] | None = (
            {} if weights is not None else None)

        for ent_dim, ent_tag in gmsh.model.getEntities():
            try:
                pparts = gmsh.model.getPartitions(ent_dim, ent_tag)
            except Exception:
                continue
            if len(pparts) == 0:
                continue
            _, etags_list, _ = gmsh.model.mesh.getElements(
                ent_dim, ent_tag)
            n_elems = sum(len(et) for et in etags_list)
            for p in pparts:
                elems_per[int(p)] = elems_per.get(int(p), 0) + n_elems

            if weights is not None and weights_per is not None:
                # Each entity is owned by exactly one partition in a
                # standard METIS decomposition; tally each element's
                # weight against every partition that claims this entity
                # (ghost entities replicate across partitions — they're
                # the same physical element, so we count their weight
                # only against the partition where the element actually
                # *lives*, which is the partition matching the entity's
                # parent-of-tag in Gmsh).  In practice ``len(pparts)==1``
                # for the owning entity and >1 for shared ghosts; we
                # tally only when ``len(pparts)==1`` to avoid
                # double-counting weight across ghosts.
                if len(pparts) == 1:
                    p = int(pparts[0])
                    total = 0.0
                    for et in etags_list:
                        for tag in et:
                            w = weights.get(int(tag))
                            if w is not None:
                                total += w
                    weights_per[p] = weights_per.get(p, 0.0) + total

        return PartitionInfo(
            n_parts=n,
            elements_per_partition=elems_per,
            weights_per_partition=weights_per)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def n_partitions(self) -> int:
        """Return the current number of partitions (0 if not partitioned)."""
        return gmsh.model.getNumberOfPartitions()

    def summary(self) -> str:
        """Concise text summary of the partition state."""
        n = self.n_partitions()
        model_name = getattr(
            getattr(self._mesh, '_parent', None), 'name', '?')
        if n == 0:
            return f"Partitioning(model={model_name!r}): not partitioned"
        lines = [
            f"Partitioning(model={model_name!r}): {n} partition(s)"]
        df = self.entity_table()
        if not df.empty:
            partitioned = df[df['partitions'] != '']
            counts = (
                partitioned
                .reset_index()
                .groupby('dim')
                .size()
                .rename(index={
                    0: 'points', 1: 'curves',
                    2: 'surfaces', 3: 'volumes'}))
            for dim_label, count in counts.items():
                lines.append(
                    f"  {dim_label:10s}: {count} partitioned entities")
        return "\n".join(lines)

    def entity_table(self, dim: int = -1) -> "pd.DataFrame":
        """DataFrame of all model entities and their partition membership.

        Parameters
        ----------
        dim : int
            Restrict to a single dimension (``-1`` = all).

        Returns
        -------
        pd.DataFrame
            Columns: ``dim``, ``tag``, ``partitions``,
            ``parent_dim``, ``parent_tag``.
        """
        import pandas as pd

        rows: list[dict] = []
        entities = (
            gmsh.model.getEntities(dim=dim)
            if dim != -1
            else gmsh.model.getEntities())
        for ent_dim, ent_tag in entities:
            try:
                parts = list(gmsh.model.getPartitions(ent_dim, ent_tag))
            except Exception:
                parts = []
            try:
                p_dim, p_tag = gmsh.model.getParent(ent_dim, ent_tag)
            except Exception:
                p_dim, p_tag = -1, -1
            rows.append({
                'dim':        ent_dim,
                'tag':        ent_tag,
                'partitions': ", ".join(str(p) for p in parts),
                'parent_dim': p_dim,
                'parent_tag': p_tag,
            })

        if not rows:
            return pd.DataFrame(
                columns=['dim', 'tag', 'partitions',
                         'parent_dim', 'parent_tag'])
        return pd.DataFrame(rows).set_index(['dim', 'tag'])

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def save(
        self,
        path: Path | str,
        *,
        one_file_per_partition: bool = False,
        create_topology: bool = False,
        create_physicals: bool = True,
    ) -> "_Partitioning":
        """Write the partitioned mesh to file(s).

        Parameters
        ----------
        path : Path or str
            Output file path (format inferred from extension).
        one_file_per_partition : bool
            Write one file per partition alongside the combined file.
        create_topology : bool
            Pass to ``Mesh.PartitionCreateTopology``.
        create_physicals : bool
            Pass to ``Mesh.PartitionCreatePhysicals``.

        Returns
        -------
        self — for chaining
        """
        path = Path(path)
        gmsh.option.setNumber(
            "Mesh.PartitionCreateTopology", int(create_topology))
        gmsh.option.setNumber(
            "Mesh.PartitionCreatePhysicals", int(create_physicals))
        gmsh.option.setNumber(
            "Mesh.PartitionSplitMeshFiles", int(one_file_per_partition))
        gmsh.write(str(path))
        self._mesh._log(
            f"save({path}, "
            f"one_file_per_partition={one_file_per_partition})")
        return self
