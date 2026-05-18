"""``MeshSelectionChain`` — the point-family chainable over the LIVE mesh.

Sibling of ``mesh/_node_chain.py`` / ``mesh/_elem_chain.py`` /
``results/_result_chain.py``, but bound to the **pre-snapshot**,
mutable, session-bound :class:`~apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet`
(the ``g.mesh_selection`` composite that reads live
``gmsh.model.mesh`` — distinct from the immutable
``MeshSelectionStore`` captured onto ``FEMData``).

Module load imports **only** the package-root leaf ``apeGmsh._chain``
+ numpy.  It does *not* import ``apeGmsh.core``/``viz``/``results``,
and it does *not* import ``gmsh`` at load — the live-mesh
node/element coordinates are obtained by **delegating to the existing
``MeshSelectionSet`` live-mesh path** (``_get_mesh_nodes`` /
``_get_mesh_elements`` — the exact methods ``add_nodes`` /
``add_elements`` already use), reached through a back-reference on the
engine adapter.  This keeps the
``tests/test_import_dag_polarity.py`` eager-edge baseline (8 triples,
``mesh``↔``core`` only) unchanged and keeps the live-mesh fetch in
one place (no reimplementation, no second copy of ``_mesh_filters``).

A ``MeshSelectionChain`` is **bi-level**, like ``ResultChain``:

* ``level == "node"``    — atoms are mesh node ids; spatial verbs
  operate on the live node coordinates
  (``MeshSelectionSet._get_mesh_nodes``).
* ``level == "element"`` — atoms are element ids for one element
  ``dim``; spatial verbs operate on element **centroids** (the mean
  of each element's node coordinates), computed **fail-loud** here: a
  connectivity entry referencing a node id absent from the live node
  set raises ``KeyError`` (never the silent ``dict.get(..., 0)``
  row-0 substitution that the generic
  ``_mesh_filters.element_centroids`` performs — that would corrupt
  the centroid).  This is the same correctness invariant
  ``_elem_chain`` / ``_result_chain`` already enforce.

The level (and, for the element level, the element ``dim``) is
carried on a tiny opaque engine adapter (:class:`_LiveMeshEngine`) so
the base ``SelectionChain`` contract is untouched — every refining
verb is ``type(self)(new_items, _engine=self._engine)`` (covariant),
so the single class daisy-chains identically at both levels.

The point-family spatial contract matches ``NodeChain`` /
``ElementChain`` exactly: ``in_box`` is half-open ``[lo, hi)`` by
default (canonical, R4) and closed ``[lo, hi]`` with
``inclusive=True`` (S2 parity), so a fluent
``g.mesh_selection.select().in_box(...).on_plane(...)`` is the same
node/element set the eager
``g.mesh_selection.add_nodes(in_box=..., on_plane=...)`` produces.

Terminal: :meth:`MeshSelectionChain.result` returns the **same shape**
``MeshSelectionSet.get_nodes`` / ``get_elements`` return today
(node level: ``{'tags': ndarray(object), 'coords': (N,3) float64}``;
element level: ``{'element_ids': ndarray(object),
'connectivity': ndarray(E,npe,object)}``), so the chain terminal is
the existing live-mesh result view of the selected ids.
:attr:`MeshSelectionChain.ids` exposes the raw selected id list.

**Persistence is deliberately out of scope** (sub-phase S3d): a chain
does NOT register itself into ``MeshSelectionSet._sets`` and there is
no ``.save_as(name)``.  Naming/round-tripping a chained selection as a
``selection=`` is deferred to the persistence decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._kernel.chain import SelectionChain

#: The two levels a ``MeshSelectionChain``'s atoms can live at.
VALID_LEVELS = ("node", "element")


class _LiveMeshEngine:
    """Opaque back-reference a :class:`MeshSelectionChain` is bound to.

    Holds only what the chain needs and a level discriminator:

    * ``ms`` — the spawning
      :class:`~apeGmsh.mesh.MeshSelectionSet.MeshSelectionSet`.  The
      chain reaches the **live mesh exclusively through** its existing
      ``_get_mesh_nodes`` / ``_get_mesh_elements`` methods (the same
      path ``add_nodes`` / ``add_elements`` use); the chain never
      touches ``gmsh`` itself.
    * ``level`` — ``"node"`` or ``"element"``.
    * ``dim``  — element dimension (1/2/3) when ``level == "element"``;
      ``0`` (unused) for the node level.

    Identity (not value) is what the base
    :meth:`SelectionChain._compatible` compares, so set-algebra is loud
    across two differently-bound live-mesh selections — same contract
    as the mesh / results chains.
    """

    # The two ``_apegmsh_lm_*`` slots are lazily-populated per-engine
    # coordinate / centroid caches (mirrors NodeChain / ElementChain /
    # ResultChain engine-side memoisation).
    __slots__ = (
        "ms", "level", "dim",
        "_apegmsh_lm_node_idrow", "_apegmsh_lm_elem_centroid",
    )

    def __init__(self, ms: Any, level: str, dim: int) -> None:
        if level not in VALID_LEVELS:
            raise ValueError(
                f"MeshSelectionChain level={level!r} invalid; expected "
                f"one of {VALID_LEVELS}."
            )
        self.ms = ms
        self.level = level
        self.dim = dim
        self._apegmsh_lm_node_idrow = None
        self._apegmsh_lm_elem_centroid = None


#: Attribute the per-``MeshSelectionSet`` engine-adapter map is
#: memoised under (mirrors ``ResultChain``'s per-composite adapter
#: cache, but keyed by ``(level, dim)`` because one
#: ``MeshSelectionSet`` spans both levels and every element ``dim``).
_ENGINE_CACHE_ATTR = "_apegmsh_mesh_selection_chain_engines"


def engine_for(ms: Any, level: str, dim: int) -> _LiveMeshEngine:
    """Return the **stable per-(ms, level, dim)** engine adapter.

    The base :meth:`SelectionChain._compatible` gates set-algebra by
    engine *identity* (``self._engine is other._engine``).  A live-mesh
    selection cannot use the ``MeshSelectionSet`` itself as the engine
    (the chain needs a level / dim discriminator the composite does not
    carry), so an adapter is built once per ``(level, dim)`` and
    memoised on the ``MeshSelectionSet``.  Consequences, matching the
    locked contract used by the other chains:

    * two node selections from the same ``g.mesh_selection`` share one
      adapter → ``select(ids=a) | select(ids=b)`` composes;
    * a node selection and an element selection (different ``level``)
      get different adapters → cross-level set-algebra is loud;
    * element selections at different ``dim`` get different adapters →
      combining a 2-D and a 3-D element selection is loud;
    * two different sessions have different ``MeshSelectionSet``
      objects → different adapters → cross-session set-algebra is loud.
    """
    cache = getattr(ms, _ENGINE_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(ms, _ENGINE_CACHE_ATTR, cache)
    key = (level, dim)
    eng = cache.get(key)
    if eng is None:
        eng = _LiveMeshEngine(ms, level, dim)
        cache[key] = eng
    return eng


class MeshSelectionChain(SelectionChain):
    """Daisy-chainable live-mesh selection (point family, bi-level)."""

    FAMILY = "point"

    __slots__ = ()

    # ── level ───────────────────────────────────────────────
    @property
    def _level(self) -> str:
        return self._engine.level

    # ── live-mesh node fetch (reuses the EXISTING path) ─────
    def _live_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """``(ids, coords)`` straight from the spawning
        ``MeshSelectionSet._get_mesh_nodes`` — the *exact* live-mesh
        path ``add_nodes`` uses (no reimplementation)."""
        return self._engine.ms._get_mesh_nodes()

    def _node_row_map(self) -> dict:
        cache = self._engine._apegmsh_lm_node_idrow
        if cache is None:
            ids, _ = self._live_nodes()
            cache = {int(n): i for i, n in enumerate(ids)}
            self._engine._apegmsh_lm_node_idrow = cache
        return cache

    # ── element-level centroid access (FAIL-LOUD) ───────────
    def _centroid_map(self) -> dict:
        """``element_id -> (3,) float64 centroid`` for every element of
        the engine's ``dim``.

        Computed once per engine adapter and memoised on it.  The
        centroid is the mean of the element's node coordinates, taken
        from the **live mesh** via the spawning
        ``MeshSelectionSet._get_mesh_elements`` / ``_get_mesh_nodes``
        (the same fetch ``add_elements`` uses).

        **Fail-loud**: a connectivity entry referencing a node id that
        is not in the live node set raises ``KeyError``.  This
        deliberately does *not* reuse
        ``_mesh_filters.element_centroids`` — that routine
        ``dict.get(int(nid), 0)``-substitutes row 0 for a missing node
        id, silently corrupting the centroid instead of failing (the
        same invariant ``_elem_chain`` / ``_result_chain`` enforce).
        """
        cache = self._engine._apegmsh_lm_elem_centroid
        if cache is not None:
            return cache

        ms = self._engine.ms
        dim = self._engine.dim
        node_ids, node_xyz = ms._get_mesh_nodes()
        id_to_idx = {int(n): i for i, n in enumerate(node_ids)}
        elem_ids, conn = ms._get_mesh_elements(dim)
        elem_ids = np.asarray(elem_ids, dtype=np.int64)
        conn = np.asarray(conn, dtype=np.int64)

        cache: dict = {}
        for row in range(elem_ids.shape[0]):
            try:
                rows = [id_to_idx[int(n)] for n in conn[row] if n >= 0]
            except KeyError as e:
                raise KeyError(
                    f"element {int(elem_ids[row])} (dim={dim}) "
                    f"references node {e.args[0]} which is not in the "
                    f"live mesh node set — refusing to compute a "
                    f"corrupted centroid (fail loud)."
                ) from None
            cache[int(elem_ids[row])] = node_xyz[rows].mean(axis=0)

        self._engine._apegmsh_lm_elem_centroid = cache
        return cache

    # ── abstract hook: coords of the given atoms ────────────
    def _coords_of(self, atoms: tuple) -> np.ndarray:
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        if self._level == "node":
            _, coords = self._live_nodes()
            coords = np.asarray(coords, dtype=np.float64)
            rm = self._node_row_map()
            try:
                rows = [rm[int(a)] for a in atoms]
            except KeyError as e:
                raise KeyError(
                    f"node id {e.args[0]} is not in the live mesh "
                    f"(no coordinate)."
                ) from None
            return coords[rows]
        # element level — centroids (fail-loud)
        cmap = self._centroid_map()
        try:
            rows = [cmap[int(a)] for a in atoms]
        except KeyError as e:
            raise KeyError(
                f"element id {e.args[0]} is not in the live mesh "
                f"(dim={self._engine.dim}; no centroid)."
            ) from None
        return np.asarray(rows, dtype=np.float64)

    # ── point-family spatial hooks (numpy kernel) ───────────
    # Identical coordinate-containment contract to NodeChain /
    # ElementChain / ResultChain: the base ``in_box`` calls
    # ``_spatial_box`` with ``inclusive=`` flowing through; default is
    # half-open ``[lo, hi)`` (canonical, R4), ``inclusive=True``
    # restores the closed box ``[lo, hi]`` (S2 parity).
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
    @property
    def ids(self) -> list[int]:
        """The selected live-mesh ids (node ids or element ids)."""
        return [int(a) for a in self._items]

    def result(self) -> dict:
        return self._materialize()

    def _materialize(self) -> dict:
        """The selected live-mesh ids as the **same-shape** dict
        ``MeshSelectionSet.get_nodes`` / ``get_elements`` return today.

        * node level → ``{'tags': ndarray(object),
          'coords': (N,3) float64}`` — exactly
          :meth:`MeshSelectionSet.get_nodes`'s shape, built from the
          chain's selected node ids and their live coordinates.
        * element level → ``{'element_ids': ndarray(object),
          'connectivity': ndarray(E, npe, object)}`` — exactly
          :meth:`MeshSelectionSet.get_elements`'s shape, the live
          connectivity rows for the chain's selected element ids.

        So ``select(...).result()`` is the existing live-mesh result
        view of the (already-narrowed) ids — no detached snapshot, no
        registration into ``_sets`` (persistence is out of S3d scope).
        """
        atoms = self._items
        if self._level == "node":
            ids = np.asarray(atoms, dtype=np.int64)
            coords = self._coords_of(atoms)
            return {
                "tags": ids.astype(object),
                "coords": np.asarray(coords, dtype=np.float64),
            }
        # element level — mask the live (ids, conn) to the selection,
        # preserving live-mesh row order (matches add_elements storage).
        ms = self._engine.ms
        all_ids, all_conn = ms._get_mesh_elements(self._engine.dim)
        all_ids = np.asarray(all_ids, dtype=np.int64)
        all_conn = np.asarray(all_conn, dtype=np.int64)
        keep = set(int(a) for a in atoms)
        mask = np.array([int(e) in keep for e in all_ids], dtype=bool)
        return {
            "element_ids": all_ids[mask].astype(object),
            "connectivity": all_conn[mask].astype(object),
        }
