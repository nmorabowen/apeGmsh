"""``ElementChain`` — the point-family chainable over FEM elements.

Sibling of ``mesh/_node_chain.py``.  Imports only the package-root
leaf ``apeGmsh._chain`` and numpy at module load — it does **not**
import ``apeGmsh.core`` or ``apeGmsh.results`` (keeps the
eager/deferred import polarity intact — see
``docs/plans/selection-unification.md`` §3 and the
``tests/test_import_dag_polarity.py`` guard; the baseline set of eager
cross-package edges stays unchanged).

Atoms are **element ids**.  The point-family spatial verbs operate on
element **centroids** (the mean of each element's node coordinates),
exactly the same numpy half-open / ``inclusive=`` contract as
``NodeChain`` (the base ``in_box`` flows through ``_spatial_box``).

The engine is the ``ElementComposite``: it owns the per-type element
groups (``id`` + ``connectivity``) and — wired once in
``FEMData.__init__`` — a back-reference to the sibling
``NodeComposite`` so centroids can be computed in-memory (no live Gmsh
session required, so this works for import-origin FEMData too).
Connectivity that references an unknown node id is a **fail-loud**
error (never silently mapped to row 0 — that would corrupt centroids).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._chain import SelectionChain

if TYPE_CHECKING:
    # Type-only: the runtime import stays deferred inside _materialize
    # (TYPE_CHECKING is False at runtime, so the module's load-time
    # imports remain just the package-root leaf + numpy).
    from ._element_types import GroupResult

#: Engine attribute carrying the sibling ``NodeComposite`` (set by
#: ``FEMData.__init__``).  Named here so the wiring point and the
#: consumer agree on one private contract.
NODES_REF_ATTR = "_apegmsh_nodes_ref"


class ElementChain(SelectionChain):
    """Daisy-chainable element selection (point family, centroid-based)."""

    FAMILY = "point"

    __slots__ = ()

    # ── centroid cache (built once per engine) ──────────────
    def _centroid_map(self) -> dict:
        """``element_id -> (3,) float64 centroid`` for every element.

        Computed once per ``ElementComposite`` engine and memoised on
        it (mirrors ``NodeChain._row_map``'s engine-side cache).  The
        centroid is the mean of the element's node coordinates.

        Fails loud if the engine was not wired with its sibling
        ``NodeComposite`` (the ``FEMData.__init__`` contract), or if any
        element's connectivity references a node id that is not in the
        node set — silently substituting row 0 (as the generic
        ``_mesh_filters.element_centroids`` does) would corrupt the
        centroid and is explicitly rejected here.
        """
        cache = getattr(self._engine, "_apegmsh_elem_centroid", None)
        if cache is not None:
            return cache

        nodes = getattr(self._engine, NODES_REF_ATTR, None)
        if nodes is None:
            raise RuntimeError(
                "ElementChain centroids require the sibling "
                "NodeComposite, which FEMData.__init__ wires onto the "
                "ElementComposite. This engine is missing it — build "
                "the chain via fem.elements.select(...) on a FEMData."
            )

        node_ids = np.asarray(nodes.ids, dtype=np.int64)
        node_xyz = np.asarray(nodes.coords, dtype=np.float64)
        id_to_idx = {int(n): i for i, n in enumerate(node_ids)}

        cache = {}
        for grp in self._engine._groups.values():
            eids = np.asarray(grp.ids, dtype=np.int64)
            conn = np.asarray(grp.connectivity, dtype=np.int64)
            if eids.size == 0:
                continue
            for row in range(eids.shape[0]):
                try:
                    rows = [id_to_idx[int(n)] for n in conn[row]]
                except KeyError as e:
                    raise KeyError(
                        f"element {int(eids[row])} "
                        f"({grp.type_name}) references node {e.args[0]} "
                        f"which is not in the FEM node set — refusing "
                        f"to compute a corrupted centroid (fail loud)."
                    ) from None
                cache[int(eids[row])] = node_xyz[rows].mean(axis=0)

        setattr(self._engine, "_apegmsh_elem_centroid", cache)
        return cache

    # ── coordinate access — element centroid ───────────────
    def _coords_of(self, atoms: tuple) -> np.ndarray:
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        cmap = self._centroid_map()
        try:
            rows = [cmap[int(a)] for a in atoms]
        except KeyError as e:
            raise KeyError(
                f"element id {e.args[0]} is not in this FEM "
                f"(no centroid)."
            ) from None
        return np.asarray(rows, dtype=np.float64)

    # ── point-family spatial hooks (numpy kernel) ───────────
    # Same coordinate-containment contract as NodeChain: the base
    # `in_box` calls `_spatial_box` with `inclusive=` flowing through;
    # default is half-open [lo, hi) (canonical, R4), `inclusive=True`
    # restores the closed box [lo, hi].
    def _spatial_box(self, atoms, lo, hi, *, inclusive: bool) -> tuple:
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        if inclusive:                       # closed [lo, hi]
            mask = np.all((c >= lo) & (c <= hi), axis=1)
        else:                               # half-open [lo, hi)  (canonical)
            mask = np.all((c >= lo) & (c < hi), axis=1)
        return tuple(a for a, k in zip(atoms, mask) if k)

    def _spatial_sphere(self, atoms, center, radius: float) -> tuple:
        r = float(radius)
        if r < 0:
            raise ValueError(f"radius must be non-negative, got {r}.")
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        ctr = np.asarray(center, dtype=np.float64).reshape(3)
        mask = np.linalg.norm(c - ctr, axis=1) <= r          # closed ball
        return tuple(a for a, k in zip(atoms, mask) if k)

    def _spatial_plane(self, atoms, point, normal, tol: float) -> tuple:
        t = float(tol)
        if t < 0:
            raise ValueError(f"tolerance must be non-negative, got {t}.")
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        nn = np.linalg.norm(n)
        if nn == 0:
            raise ValueError("normal vector has zero length.")
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        p = np.asarray(point, dtype=np.float64).reshape(3)
        dist = np.abs((c - p) @ (n / nn))
        return tuple(a for a, k in zip(atoms, dist <= t) if k)

    # ── terminal ────────────────────────────────────────────
    def result(self) -> "GroupResult":
        return self._materialize()

    def _materialize(self) -> "GroupResult":
        """The selected elements as the **existing** ``GroupResult``.

        Identical type/shape to what ``fem.elements.get(...)`` returns
        today: a ``GroupResult`` of per-type ``ElementGroup`` blocks
        masked to the chain's selected element ids — the same
        ``np.isin`` block-masking ``ElementComposite.get`` performs for
        its id-set filter, so ``select(...).result()`` is the same
        materialised object family as ``get(...)``.

        ``GroupResult`` / ``ElementGroup`` are imported **deferred**
        (inside this method, not at module top): ``_element_types``
        pulls only stdlib/numpy so there is no cycle today, but the
        deferred import keeps this module's load-time imports to just
        the package-root leaf ``apeGmsh._chain`` + numpy — matching
        ``NodeChain`` and keeping the import-DAG polarity baseline
        unchanged.
        """
        from ._element_types import ElementGroup, GroupResult  # deferred

        keep = set(int(a) for a in self._items)
        result_groups: list = []
        for grp in self._engine._groups.values():
            gids = np.asarray(grp.ids, dtype=np.int64)
            mask = np.isin(gids, list(keep))
            if not mask.any():
                continue
            result_groups.append(
                ElementGroup(
                    element_type=grp.element_type,
                    ids=grp.ids[mask],
                    connectivity=grp.connectivity[mask],
                )
            )
        return GroupResult(result_groups)
