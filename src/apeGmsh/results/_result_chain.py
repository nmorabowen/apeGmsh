"""``ResultChain`` — the point-family chainable over result IDs.

Sibling of ``mesh/_node_chain.py`` / ``mesh/_elem_chain.py``, but for
the **results** domain.  Module load imports **only** the package-root
leaf ``apeGmsh._chain`` + numpy — it does *not* import
``apeGmsh.core``/``apeGmsh.mesh``, and anything from ``apeGmsh.results``
is imported **deferred** (inside method bodies).  This keeps
``results`` runtime-clean of ``core``/``mesh`` (it is TYPE_CHECKING-only
today — see every other ``results/*.py``) and keeps the
``tests/test_import_dag_polarity.py`` eager-edge baseline (8 triples,
``results`` in none of them) unchanged.

A ``ResultChain`` is **bi-level**: its atoms are node ids *or* element
ids depending on the composite that spawned it.  The level is carried
on a tiny opaque engine adapter (:class:`_ResultChainEngine`) so the
base ``SelectionChain`` contract is untouched — every refining verb is
``type(self)(new_items, _engine=self._engine)`` (covariant), so the
single ``ResultChain`` class daisy-chains identically at both levels.

Coordinate source per level (point-family spatial verbs operate on it):

* ``level == "node"``   — the bound ``fem.nodes.coords`` (duck-typed,
  exactly as the existing ``results/_composites`` spatial helpers do).
* ``level == "element"`` — element **centroids** (mean of the element's
  node coordinates), computed **fail-loud** here: a connectivity entry
  that references a node id absent from the FEM raises (never the
  ``np.clip`` silent row-substitution that
  ``results/_composites._element_centroids`` does — that routine is
  flagged for the fail-loud sweep; it is *not* reused here).

Terminal: unlike the node/element broker chains (whose ``.result()``
materialises a detached ``NodeResult``/``GroupResult``), a results
selection needs a **component** to read a slab.  :meth:`ResultChain.get`
delegates to the spawning composite's existing ``.get(ids=...,
component=...)`` path, so it returns the *exact* existing slab type
(``NodeSlab`` / ``ElementSlab``) with id/value parity to
``results.<level>.get(ids=<equiv>, component=...)``.  ``.result()``
with no component fails loud (a bare results selection is meaningless
without a component).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._kernel.chain import SelectionChain

#: The two levels a ``ResultChain``'s atoms can live at.
VALID_LEVELS = ("node", "element")


class _ResultChainEngine:
    """Opaque back-reference a :class:`ResultChain` is bound to.

    Minimal level discriminator + the two things the chain needs:

    * ``results`` — the bound :class:`~apeGmsh.results.Results` (used
      only to reach ``results._fem`` for coordinates / centroids,
      duck-typed exactly as the existing results spatial helpers do);
    * ``host`` — the spawning composite (``NodeResultsComposite`` /
      ``ElementResultsComposite``), whose **existing** ``.get(...)`` the
      terminal delegates to (so the slab read reuses the existing
      reader path verbatim — automatic parity);
    * ``level`` — ``"node"`` or ``"element"``.

    Identity (not value) is what the base
    :meth:`SelectionChain._compatible` compares, so set-algebra is loud
    across two differently-bound results selections — same contract as
    the mesh chains.
    """

    # The two ``_apegmsh_rc_*`` slots are lazily-populated per-engine
    # coordinate/centroid caches (mirrors NodeChain/ElementChain's
    # engine-side memoisation, which write onto their composite engine).
    __slots__ = (
        "results", "host", "level",
        "_apegmsh_rc_node_idrow", "_apegmsh_rc_elem_centroid",
    )

    def __init__(self, results: Any, host: Any, level: str) -> None:
        if level not in VALID_LEVELS:
            raise ValueError(
                f"ResultChain level={level!r} invalid; expected one of "
                f"{VALID_LEVELS}."
            )
        self.results = results
        self.host = host
        self.level = level
        self._apegmsh_rc_node_idrow = None
        self._apegmsh_rc_elem_centroid = None


#: Attribute the per-composite engine adapter is memoised under.
_ENGINE_CACHE_ATTR = "_apegmsh_result_chain_engine"


def engine_for(results: Any, host: Any, level: str) -> _ResultChainEngine:
    """Return the **stable per-composite** engine adapter for ``host``.

    The base :meth:`SelectionChain._compatible` gates set-algebra by
    engine *identity* (``self._engine is other._engine``), exactly as it
    does for the mesh chains — whose engine *is* the composite (a stable
    singleton on the FEMData).  A results selection cannot use the
    composite itself as the engine (the chain needs a level
    discriminator the composite does not carry), so the adapter is built
    once per composite and memoised on it.  Consequences, matching the
    locked contract:

    * two selections from the *same* ``results.<level>`` share one
      adapter → ``select(ids=a) | select(ids=b)`` composes;
    * a node selection and an element selection come from *different*
      host composites → different adapters → cross-level set-algebra is
      loud;
    * two different ``Results`` have different composites → different
      adapters → cross-results set-algebra is loud.

    ``Results._derive`` builds fresh composites, so a derived
    ``Results`` gets its own adapter (cross-derive is loud too — the
    user must pair selections from one ``Results``).
    """
    cached = getattr(host, _ENGINE_CACHE_ATTR, None)
    if cached is not None and cached.level == level:
        # ``results`` can change underneath a re-derived composite that
        # somehow reuses the host object; keep the back-ref fresh while
        # preserving identity for the set-algebra contract.
        cached.results = results
        return cached
    eng = _ResultChainEngine(results, host, level)
    setattr(host, _ENGINE_CACHE_ATTR, eng)
    return eng


class ResultChain(SelectionChain):
    """Daisy-chainable results selection (point family, bi-level)."""

    FAMILY = "point"

    __slots__ = ()

    # ── level ───────────────────────────────────────────────
    @property
    def _level(self) -> str:
        return self._engine.level

    # ── bound FEM (duck-typed, same as results spatial helpers) ─
    def _fem(self):
        fem = getattr(self._engine.results, "_fem", None)
        if fem is None:
            raise RuntimeError(
                "ResultChain spatial / coordinate access requires a "
                "bound FEMData. Pass fem= when constructing Results, or "
                "call results.bind(fem)."
            )
        return fem

    # ── node-level coordinate access ────────────────────────
    def _node_row_map(self) -> dict:
        cache = self._engine._apegmsh_rc_node_idrow
        if cache is None:
            ids = np.asarray(self._fem().nodes.ids, dtype=np.int64)
            cache = {int(n): i for i, n in enumerate(ids)}
            # memoise on the engine adapter (mirrors NodeChain's
            # engine-side cache)
            self._engine._apegmsh_rc_node_idrow = cache
        return cache

    # ── element-level centroid access (FAIL-LOUD) ───────────
    def _centroid_map(self) -> dict:
        """``element_id -> (3,) float64 centroid`` for every element.

        Computed once per engine adapter and memoised on it.  The
        centroid is the mean of the element's node coordinates.

        **Fail-loud** (contract 4): a connectivity entry referencing a
        node id that is not in the FEM node set raises ``KeyError``.
        This deliberately does **not** reuse
        ``results/_composites._element_centroids`` — that routine
        ``np.clip``-s an out-of-range ``searchsorted`` index to the last
        node, silently corrupting the centroid instead of failing.  A
        local fail-loud computation is used here; the clip-silent
        routine is flagged for the orchestrator's fail-loud sweep.
        """
        cache = self._engine._apegmsh_rc_elem_centroid
        if cache is not None:
            return cache

        fem = self._fem()
        node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        node_xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
        id_to_idx = {int(n): i for i, n in enumerate(node_ids)}

        cache: dict = {}
        for type_info in fem.elements.types:
            ids, conn = fem.elements.resolve(element_type=type_info.name)
            ids = np.asarray(ids, dtype=np.int64)
            conn = np.asarray(conn, dtype=np.int64)
            if ids.size == 0:
                continue
            for row in range(ids.shape[0]):
                try:
                    rows = [id_to_idx[int(n)] for n in conn[row]]
                except KeyError as e:
                    raise KeyError(
                        f"element {int(ids[row])} ({type_info.name}) "
                        f"references node {e.args[0]} which is not in "
                        f"the FEM node set — refusing to compute a "
                        f"corrupted centroid (fail loud)."
                    ) from None
                cache[int(ids[row])] = node_xyz[rows].mean(axis=0)

        self._engine._apegmsh_rc_elem_centroid = cache
        return cache

    # ── abstract hook: coords of the given atoms ────────────
    def _coords_of(self, atoms: tuple) -> np.ndarray:
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        if self._level == "node":
            coords = np.asarray(self._fem().nodes.coords, dtype=np.float64)
            rm = self._node_row_map()
            try:
                rows = [rm[int(a)] for a in atoms]
            except KeyError as e:
                raise KeyError(
                    f"node id {e.args[0]} is not in this FEM "
                    f"(no coordinate)."
                ) from None
            return coords[rows]
        # element level — centroids (fail-loud)
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
    # Identical coordinate-containment contract to NodeChain /
    # ElementChain: the base ``in_box`` calls ``_spatial_box`` with
    # ``inclusive=`` flowing through; default is half-open ``[lo, hi)``
    # (canonical, R4), ``inclusive=True`` restores the closed box.
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
    def get(self, *, component: str, time=None, stage=None, **extra):
        """Materialise the slab for the chain's selected ids.

        Delegates to the spawning composite's **existing**
        ``.get(ids=<chain ids>, component=..., time=..., stage=...)``
        path — so the slab is read through the exact existing reader
        code and is the *same* slab type / id-and-value parity as
        ``results.<level>.get(ids=<equivalent>, component=...)``.  The
        spatial daisy-chain happens *before* this call (the chain's
        atoms are already narrowed); this terminal only reads.

        ``**extra`` forwards each spawning sub-composite's *additional*
        terminal kwargs verbatim — ``gp_indices=`` for
        ``results.elements.fibers``; ``gp_indices=`` / ``layer_indices=``
        for ``results.elements.layers``.  ``ResultChain`` stays generic:
        it never names a sub-composite kwarg, so the host's own ``.get``
        signature remains the single source of truth (an unknown kwarg
        fails loud there, not silently dropped here).  The node /
        element levels and the uniform sub-composites
        (``gauss`` / ``line_stations`` / ``springs``) pass nothing
        extra, so their call is byte-identical to before.
        """
        host = self._engine.host
        return host.get(
            ids=list(self._items),
            component=component,
            time=time,
            stage=stage,
            **extra,
        )

    def _materialize(self):
        """Bare ``.result()`` is meaningless for a results selection.

        A results selection identifies *where* to read; it still needs
        *what* (a component).  Fail loud with a directive message rather
        than return something meaningless (contract 3).
        """
        raise RuntimeError(
            "results selection needs .get(component=...): a ResultChain "
            "identifies node/element ids but a slab read requires a "
            "component. Use "
            "results.<nodes|elements>.select(...).<spatial...>"
            ".get(component=...) instead of .result()."
        )
