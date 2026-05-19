"""``MeshSelection`` — the v2 point-family terminal over FEM ids.

selection-unification-v2 P2-I (``docs/plans/selection-unification-v2.md``
§4/§5 R-v2-2, §6 P2-I, §6.1 STOP-2) + **P3-K** (§6.2 P3-K +
``selection-unification-v2-p3k-execmap.md``).  This is the
**chain==terminal** point-family type the *four* point host hooks
return from P2-I onward:

* ``fem.nodes.select(...)``        (broker node — was ``NodeChain``)
* ``fem.elements.select(...)``     (broker element — was ``ElementChain``)
* ``results.<nodes|elements>.select(...)`` /
  ``results.elements.{gauss,fibers,layers,line_stations,springs}
  .select(...)``                   (results — was ``ResultChain``)
* ``g.mesh_selection.select(...)`` (live mesh — was
  ``MeshSelectionChain``)

It is a **new leaf** (mirrors ``mesh/_node_chain.py`` exactly): the only
module-top ``apeGmsh`` imports are the package-root leaves
``from .._kernel.chain import SelectionChain`` and the pure
``from .._kernel import spatial`` (``_kernel.payloads`` is imported
**deferred** inside ``_materialize`` to avoid the
``FEMData``↔``_mesh_selection`` load cycle, mirroring
``_node_chain.py:107``).  ``mesh/_mesh_selection.py`` already carries the
frozen ``("mesh","_kernel","mesh/_mesh_selection.py")`` import-DAG
BASELINE triple, so the extra ``_kernel.spatial`` edge is the *same
polarity already frozen* — **no new BASELINE triple**, no ``core↔mesh``
edge, no deferred→eager flip (``tests/test_import_dag_polarity.py``).

Engine-polymorphism (P3-K: self-contained, no delegation)
---------------------------------------------------------
``MeshSelection`` is **engine-polymorphic**: it serves the four host
contexts above and in each behaves *identically* to the legacy chain
that host previously returned.  Through P2-I this was done by
delegating each per-engine hook to a freshly-constructed legacy chain
(``NodeChain`` / ``ElementChain`` / ``ResultChain`` /
``MeshSelectionChain``).  **P3-K collapsed that delegation**: the
per-engine ``_coords_of`` / centroid / ``_materialize`` bodies are now
**relocated verbatim into this class** (dispatched on
:func:`_engine_kind`), and the box / sphere / plane coordinate-mask math
— which was a *byte-identical* copy in all four chains — lives once in
the pure :mod:`apeGmsh._kernel.spatial` kernel.  The relocation is a
**behaviour-INVISIBLE** pure move (cache-attribute names, iteration
order, fail-loud messages, deferred imports and dtypes are preserved
character-for-character from the chains), proven by the legacy chains'
own tests + every ``select(...)`` parity test staying green and the
four proof files staying byte-unchanged.  The four legacy chain modules
are left **defined-but-dead** through P3-K (P3-R deletes them — Phase-0
deferral, ``…-p3k-execmap.md``); ``MeshSelection`` no longer constructs
or imports them.

``MeshSelection`` owns the *unified* surface: the ratified pair-view
``__iter__`` (HT8 / R3-C, §6.1 STOP-2(b) Option (i)), the ``.ids`` /
``.coords`` / ``.connectivity`` / ``.groups()`` accessors, the
``.values(...)`` results read (now calling the **retained** spawning
sub-composite ``host.get(...)`` directly — SC-2), the ``.result()``
identity alias, and ``.save_as(name)``.

Set-algebra is **unaffected**: it is inherited from
:class:`~apeGmsh._kernel.chain.SelectionChain` and operates on the
``_items`` atoms (node ids / element ids), not on ``__iter__``.
``_compatible`` gates by ``type(self)`` (now ``MeshSelection``) and
``self._engine`` *identity* — the four host hooks pass the same engine
object the legacy chains used (the composite itself for the broker
levels; the memoised ``engine_for`` singleton for the results / live
levels), so cross-context / cross-engine set-algebra stays loud exactly
as before.

Engine dispatch (most-specific attribute first; unambiguous — verified
at source):

* ``_LiveMeshEngine``   — has ``.ms`` (the ``MeshSelectionSet``)
  → live-mesh.
* ``_ResultChainEngine`` — has ``.host`` + ``.results`` (no ``.ms``)
  → results.
* ``ElementComposite``  — has ``._groups`` (per-type dict; no
  ``.ms``/``.host``) → broker element.
* ``NodeComposite``     — has ``.coords`` / ``.ids`` (no ``._groups`` /
  ``.host`` / ``.ms``) → broker node.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .._kernel.chain import SelectionChain
from .._kernel import spatial

#: Engine attribute carrying the sibling ``NodeComposite`` on an
#: ``ElementComposite`` (set by ``FEMData.__init__``).  Relocated
#: verbatim from the (now-dead) ``mesh/_elem_chain.py:42`` so the
#: broker-element centroid wiring point and consumer agree on one
#: private contract (P3-K invisible relocation).
NODES_REF_ATTR = "_apegmsh_nodes_ref"


# Engine "kind" discriminators (no import of the engine classes — duck
# typing on the attributes each carries, verified at source):
#   _LiveMeshEngine   : __slots__ = ("ms", "level", "dim", ...)
#   _ResultChainEngine: __slots__ = ("results", "host", "level", ...)
#   ElementComposite  : self._groups : dict[int, ElementGroup]
#   NodeComposite     : self._ids / self._coords  (no _groups)
def _engine_kind(engine: Any) -> str:
    if engine is None:
        return "none"
    if hasattr(engine, "ms"):
        return "live"
    if hasattr(engine, "host") and hasattr(engine, "results"):
        return "result"
    if hasattr(engine, "_groups"):
        return "element"
    return "node"


def _no_engine() -> "RuntimeError":
    return RuntimeError(
        "MeshSelection has _engine=None (constructed standalone) — "
        "it has no host engine to resolve coordinates / a terminal "
        "against. Build it via fem.nodes.select(...) / "
        "fem.elements.select(...) / results.<...>.select(...) / "
        "g.mesh_selection.select(...)."
    )


class MeshSelection(SelectionChain):
    """Daisy-chainable + terminal point-family selection (FEM ids).

    Engine-polymorphic across the four point host contexts (broker
    node / broker element / results / live mesh); the engine-specific
    bodies are relocated verbatim from the four legacy chains (P3-K),
    so behaviour per context is byte-faithful (the
    selection-unification-v2 P2-I invisibility contract, carried
    through the P3-K collapse).  The *unified* surface (pair-view
    ``__iter__``,
    ``.ids``/``.coords``/``.connectivity``/``.groups()``/``.values()``/
    ``.result()``/``.save_as``) lives here.
    """

    FAMILY = "point"

    __slots__ = ()

    # ── level discriminator (results / live carry it on the engine) ─
    @property
    def _level(self) -> str:
        """``"node"`` | ``"element"`` for the bi-level engines.

        Broker-node is always ``"node"``; broker-element always
        ``"element"``; results / live carry an explicit ``.level`` on
        their engine adapter (the same discriminator the legacy
        ``ResultChain`` / ``MeshSelectionChain`` read).
        """
        kind = _engine_kind(self._engine)
        if kind in ("result", "live"):
            return self._engine.level
        return "element" if kind == "element" else "node"

    # ════════════════════════════════════════════════════════
    #  Per-engine coordinate access — relocated VERBATIM from the
    #  four legacy chains (P3-K; behaviour-invisible).  Caches use
    #  the *same* attribute names on the *same* engine objects the
    #  chains used, so memoisation behaviour is byte-identical.
    # ════════════════════════════════════════════════════════

    # ── broker node — verbatim mesh/_node_chain.py:31-45 ────
    def _row_map_node(self) -> dict:
        cache = getattr(self._engine, "_apegmsh_chain_idrow", None)
        if cache is None:
            ids = np.asarray(self._engine.ids)
            cache = {int(n): i for i, n in enumerate(ids)}
            setattr(self._engine, "_apegmsh_chain_idrow", cache)
        return cache

    def _coords_of_node(self, atoms: tuple) -> np.ndarray:
        coords = np.asarray(self._engine.coords, dtype=np.float64)
        rm = self._row_map_node()
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        rows = [rm[int(a)] for a in atoms]
        return coords[rows]

    # ── broker element — verbatim mesh/_elem_chain.py:53-117 ─
    def _centroid_map_element(self) -> dict:
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

    def _coords_of_element(self, atoms: tuple) -> np.ndarray:
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        cmap = self._centroid_map_element()
        try:
            rows = [cmap[int(a)] for a in atoms]
        except KeyError as e:
            raise KeyError(
                f"element id {e.args[0]} is not in this FEM "
                f"(no centroid)."
            ) from None
        return np.asarray(rows, dtype=np.float64)

    # ── results — verbatim results/_result_chain.py:148-240 ─
    def _fem(self):
        fem = getattr(self._engine.results, "_fem", None)
        if fem is None:
            raise RuntimeError(
                "ResultChain spatial / coordinate access requires a "
                "bound FEMData. Pass fem= when constructing Results, or "
                "call results.bind(fem)."
            )
        return fem

    def _node_row_map_result(self) -> dict:
        cache = self._engine._apegmsh_rc_node_idrow
        if cache is None:
            ids = np.asarray(self._fem().nodes.ids, dtype=np.int64)
            cache = {int(n): i for i, n in enumerate(ids)}
            self._engine._apegmsh_rc_node_idrow = cache
        return cache

    def _centroid_map_result(self) -> dict:
        cache = self._engine._apegmsh_rc_elem_centroid
        if cache is not None:
            return cache

        fem = self._fem()
        node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        node_xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
        id_to_idx = {int(n): i for i, n in enumerate(node_ids)}

        cache: dict = {}
        # selection-unification v2 P3-R / §6.3 M-STOP-3: iterate the
        # element groups directly (the ``_centroid_map_element``
        # pattern) — byte-equivalent to the removed per-type
        # ``fem.elements.resolve(element_type=)`` loop (``_groups`` is
        # one group per type code; same dict order as the old
        # ``fem.elements.types``).
        for grp in fem.elements._groups.values():
            ids = np.asarray(grp.ids, dtype=np.int64)
            conn = np.asarray(grp.connectivity, dtype=np.int64)
            if ids.size == 0:
                continue
            for row in range(ids.shape[0]):
                try:
                    rows = [id_to_idx[int(n)] for n in conn[row]]
                except KeyError as e:
                    raise KeyError(
                        f"element {int(ids[row])} ({grp.type_name}) "
                        f"references node {e.args[0]} which is not in "
                        f"the FEM node set — refusing to compute a "
                        f"corrupted centroid (fail loud)."
                    ) from None
                cache[int(ids[row])] = node_xyz[rows].mean(axis=0)

        self._engine._apegmsh_rc_elem_centroid = cache
        return cache

    def _coords_of_result(self, atoms: tuple) -> np.ndarray:
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        if self._level == "node":
            coords = np.asarray(self._fem().nodes.coords, dtype=np.float64)
            rm = self._node_row_map_result()
            try:
                rows = [rm[int(a)] for a in atoms]
            except KeyError as e:
                raise KeyError(
                    f"node id {e.args[0]} is not in this FEM "
                    f"(no coordinate)."
                ) from None
            return coords[rows]
        # element level — centroids (fail-loud)
        cmap = self._centroid_map_result()
        try:
            rows = [cmap[int(a)] for a in atoms]
        except KeyError as e:
            raise KeyError(
                f"element id {e.args[0]} is not in this FEM "
                f"(no centroid)."
            ) from None
        return np.asarray(rows, dtype=np.float64)

    # ── live mesh — verbatim mesh/_mesh_selection_chain.py:171-257 ─
    def _live_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        return self._engine.ms._get_mesh_nodes()

    def _node_row_map_live(self) -> dict:
        cache = self._engine._apegmsh_lm_node_idrow
        if cache is None:
            ids, _ = self._live_nodes()
            cache = {int(n): i for i, n in enumerate(ids)}
            self._engine._apegmsh_lm_node_idrow = cache
        return cache

    def _centroid_map_live(self) -> dict:
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

    def _coords_of_live(self, atoms: tuple) -> np.ndarray:
        if not atoms:
            return np.empty((0, 3), dtype=np.float64)
        if self._level == "node":
            _, coords = self._live_nodes()
            coords = np.asarray(coords, dtype=np.float64)
            rm = self._node_row_map_live()
            try:
                rows = [rm[int(a)] for a in atoms]
            except KeyError as e:
                raise KeyError(
                    f"node id {e.args[0]} is not in the live mesh "
                    f"(no coordinate)."
                ) from None
            return coords[rows]
        # element level — centroids (fail-loud)
        cmap = self._centroid_map_live()
        try:
            rows = [cmap[int(a)] for a in atoms]
        except KeyError as e:
            raise KeyError(
                f"element id {e.args[0]} is not in the live mesh "
                f"(dim={self._engine.dim}; no centroid)."
            ) from None
        return np.asarray(rows, dtype=np.float64)

    # ── abstract hook: coords of the given atoms (dispatch) ──
    def _coords_of(self, atoms: tuple) -> np.ndarray:
        kind = _engine_kind(self._engine)
        if kind == "node":
            return self._coords_of_node(atoms)
        if kind == "element":
            return self._coords_of_element(atoms)
        if kind == "result":
            return self._coords_of_result(atoms)
        if kind == "live":
            return self._coords_of_live(atoms)
        raise _no_engine()

    # ── point-family spatial hooks (one kernel; validation +
    #    empty-guard order verbatim from the legacy chains) ──
    def _spatial_box(self, atoms, lo, hi, *, inclusive: bool) -> tuple:
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        mask = spatial.box_mask(c, lo, hi, inclusive=inclusive)
        return tuple(a for a, k in zip(atoms, mask) if k)

    def _spatial_sphere(self, atoms, center, radius: float) -> tuple:
        r = float(radius)
        if r < 0:
            raise ValueError(f"radius must be non-negative, got {r}.")
        if not atoms:
            return ()
        c = self._coords_of(atoms)
        mask = spatial.sphere_mask(c, center, r)
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
        mask = spatial.plane_mask(c, point, n / nn, t)
        return tuple(a for a, k in zip(atoms, mask) if k)

    # ── unified pair-view __iter__ (HT8 / R3-C; §6.1 STOP-2(b)) ──
    def __iter__(self) -> Iterator[Any]:
        """Yield ``(id, payload)`` pairs (the ratified HT8 design).

        * node level → ``(node_id, xyz)`` (as ``NodeResult`` iterates);
        * element level → ``(element_id, conn_row)`` (as
          ``ElementGroup`` iterates inside ``GroupResult``).

        This is the documented ``for nid, xyz`` / ``for eid, conn``
        idiom (chain==terminal, R-v2-2) — the 23+ existing
        ``(id, payload)`` production callers go through ``.get()`` /
        the legacy payloads (not this chain), so they are untouched.
        **Set-algebra is unaffected**: ``| & - ^`` operate on
        ``_items`` (the atoms), not on this iterator (verified by the
        ``test_selection_idiom`` / ``test_p2i_parity`` set-algebra
        assertions, which compare ``_items``).
        """
        if self._level == "node":
            ids = np.asarray(self._items, dtype=np.int64)
            coords = self._coords_of(self._items)
            for nid, xyz in zip(ids, coords):
                yield int(nid), xyz
            return
        # element level — (eid, conn_row).  The element payload differs
        # by engine and the pair-view follows it byte-faithfully:
        #   * broker / results → ``GroupResult`` (iterate its
        #     ``ElementGroup`` blocks, each yielding ``(eid, conn)``);
        #   * live mesh → the flat ``dict`` shape
        #     ``{'element_ids', 'connectivity'}`` (the legacy
        #     ``MeshSelectionChain._materialize`` return) — zip the two
        #     into the same ``(eid, conn_row)`` pair shape.
        res = self._materialize()
        if isinstance(res, dict):             # live-mesh element payload
            eids = res["element_ids"]
            conn = res["connectivity"]
            for eid, row in zip(eids, conn):
                yield int(eid), tuple(int(n) for n in row)
            return
        for grp in res:                       # GroupResult → ElementGroup
            for eid, conn_row in grp:         # ElementGroup → (eid, conn)
                yield eid, conn_row

    # ── accessors (the unified terminal surface) ────────────
    @property
    def ids(self) -> list[int]:
        """The selected ids (node ids or element ids) as Python ints."""
        return [int(a) for a in self._items]

    @property
    def coords(self) -> np.ndarray:
        """``(N, 3)`` float64 coordinates of the selected ids.

        Node level → node coordinates; element level → element
        centroids (the same fail-loud centroid the legacy element /
        results / live chains compute — never a silent row-0).
        """
        return self._coords_of(self._items)

    @property
    def connectivity(self) -> np.ndarray:
        """Connectivity of the selected **elements** (element level).

        Reuses the materialised element payload, so the shape /
        homogeneous-vs-mixed behaviour is byte-identical to the legacy
        chain's ``.result()`` for that engine:

        * broker / results element → ``GroupResult.connectivity``
          (raises ``TypeError`` for a mixed-type result, by design —
          use ``.groups()`` / iterate);
        * live-mesh element → the live ``connectivity`` ndarray.
        """
        if self._level != "element":
            raise TypeError(
                "connectivity is element-level only; this selection is "
                "node-level (use .coords / .ids)."
            )
        res = self._materialize()
        if isinstance(res, dict):             # live-mesh element payload
            return np.asarray(res["connectivity"])
        return res.connectivity               # GroupResult.connectivity

    def groups(self):
        """The per-type element blocks for an element selection.

        Returns the ``GroupResult`` (broker / results element) —
        ``list(sel.groups())`` yields the ``ElementGroup`` blocks,
        preserving per-type ``element_type`` (needed by the OpenSees
        emitter / beam viewer; R3-B / R-v2-4).  For the live-mesh
        element engine (whose terminal is a flat dict, not a
        ``GroupResult``) returns that same dict — byte-identical to the
        legacy ``MeshSelectionChain.result()``.
        """
        if self._level != "element":
            raise TypeError(
                "groups() is element-level only; this selection is "
                "node-level."
            )
        return self._materialize()

    # ── results read — verbatim ResultChain.get (SC-2: the
    #    spawning sub-composite reader is RETAINED, not removed) ─
    def values(self, *, component: str, time=None, stage=None, **extra):
        """Read the result slab for the selected ids (results engine).

        **Verbatim** behaviour of the legacy ``ResultChain.get`` — it
        forwards ``host.get(ids=list(self._items), component=,
        time=, stage=, **extra)`` to the spawning sub-composite's
        **retained** ``.get`` (the typed results reader →
        ``Results._reader.read_*`` + ``_resolve_*_ids``; SC-2 — only the
        *chain* ``.values()`` path is the P3-R removal target, the
        composite reader stays).  ``**extra`` is forwarded opaquely
        (``gp_indices=`` / ``layer_indices=`` for the fibers / layers
        sub-composites); this method **never names** ``gp_indices`` /
        ``layer_indices`` — the spawning ``.get`` signature stays the
        single source of truth (R5; the locked
        ``test_result_chain_subcomposites`` fail-loud invariant — an
        unknown kwarg fails loud *there*, not silently dropped here).

        Only valid on the results engine; on the broker / live engines
        a results read is meaningless (no component reader) → fail
        loud, exactly as the legacy ``ResultChain`` vs broker /
        live-mesh terminals differ.
        """
        if _engine_kind(self._engine) != "result":
            raise RuntimeError(
                ".values(component=...) reads a RESULT slab and is only "
                "valid on a results selection "
                "(results.<nodes|elements|...>.select(...)). This "
                "selection is over a "
                f"{_engine_kind(self._engine)!r} engine — use "
                ".result() / .ids / .coords for the broker / live-mesh "
                "terminal instead."
            )
        host = self._engine.host
        return host.get(
            ids=list(self._items),
            component=component,
            time=time,
            stage=stage,
            **extra,
        )

    # ── persistence — register into the mesh-selection store ─
    def save_as(self, name: str) -> "MeshSelection":
        """Register the current id set into the mesh-selection store.

        Reuses the **existing** registration surface
        ``MeshSelectionSet.add(dim, ids, name=name)`` (no reinvented
        store): that writes ``_sets`` → ``_snapshot()`` →
        ``MeshSelectionStore`` → FEMData HDF5, so the named set
        round-trips and becomes addressable as ``selection=`` (the
        ``docs/plans/selection-unification-v2.md`` §6 P2-I
        ``.save_as`` contract).  Returns ``self`` for chaining.

        Reachability (source-proven; see ADR 0015 / the v2 plan): the
        mutable mesh-selection store is the live ``g.mesh_selection``
        (``MeshSelectionSet``).  Only the **live-mesh** engine carries
        it (``_LiveMeshEngine.ms``).  The broker-node / broker-element
        / results engines hold no mutable ``MeshSelectionSet`` — a
        ``FEMData`` carries only the *immutable, read-only*
        ``MeshSelectionStore`` snapshot (no ``.add``), and is routinely
        a detached / import-origin object with no live gmsh session at
        all.  There is no non-reinventing way to register from those
        engines, so ``.save_as`` is **present-but-loud** there (the
        ``in_box`` ``inclusive=``→``TypeError`` precedent: explicit
        fail, never a silent no-op or a fake parallel store).  The
        legacy ``MeshSelectionChain`` had no ``.save_as`` at all, so
        this is strictly additive and breaks no P2-I parity.
        """
        kind = _engine_kind(self._engine)
        if kind != "live":
            raise RuntimeError(
                ".save_as(name) registers into the live mesh-selection "
                "store (g.mesh_selection / MeshSelectionSet), which "
                f"only the live-mesh engine carries. This selection is "
                f"over a {kind!r} engine: a FEMData / Results holds "
                "only the immutable read-only MeshSelectionStore "
                "snapshot (no registration surface) and may have no "
                "live gmsh session. Build the selection via "
                "g.mesh_selection.select(...) to use .save_as, or "
                "register through the existing g.mesh_selection "
                "surface (add / from_geometric) before snapshotting."
            )
        ms = self._engine.ms
        dim = 0 if self._level == "node" else int(self._engine.dim)
        ms.add(dim, self.ids, name=name)
        return self

    # ── terminal — the level/engine-appropriate payload ─────
    def result(self):
        return self._materialize()

    # ── per-engine _materialize — relocated VERBATIM from the
    #    four legacy chains' _materialize (P3-K invisible) ───
    def _materialize_node(self):
        # verbatim mesh/_node_chain.py:90-112
        from .FEMData import NodeResult  # deferred — avoids load cycle

        atoms = self._items
        ids = np.asarray(atoms, dtype=np.int64)
        coords = self._coords_of(atoms)
        return NodeResult(ids, coords)

    def _materialize_element(self):
        # verbatim mesh/_elem_chain.py:166-200 (iterates
        # self._engine._groups.values() in insertion order — m4)
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

    def _materialize_result(self):
        # verbatim results/_result_chain.py:317-330
        raise RuntimeError(
            "results selection needs .get(component=...): a ResultChain "
            "identifies node/element ids but a slab read requires a "
            "component. Use "
            "results.<nodes|elements>.select(...).<spatial...>"
            ".get(component=...) instead of .result()."
        )

    def _materialize_live(self):
        # verbatim mesh/_mesh_selection_chain.py:312-348
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

    def _materialize(self):
        """Identity alias → the legacy per-engine payload (R-v2-2).

        Byte-identical to what the legacy chain for this engine
        returned from ``.result()``:

        * broker node      → ``NodeResult``;
        * broker element   → ``GroupResult``;
        * live mesh        → the live-mesh ``dict`` shape;
        * results          → **raises**, directing to ``.values(...)``
          (a results selection needs a component) — exactly as the
          legacy ``ResultChain._materialize`` does today.
        """
        kind = _engine_kind(self._engine)
        if kind == "node":
            return self._materialize_node()
        if kind == "element":
            return self._materialize_element()
        if kind == "result":
            return self._materialize_result()
        if kind == "live":
            return self._materialize_live()
        raise _no_engine()
