"""``MeshSelection`` — the v2 point-family terminal over FEM ids.

selection-unification-v2 P2-I (``docs/plans/selection-unification-v2.md``
§4/§5 R-v2-2, §6 P2-I, §6.1 STOP-2).  This is the **chain==terminal**
point-family type the *four* point host hooks return from P2-I onward:

* ``fem.nodes.select(...)``        (broker node — was ``NodeChain``)
* ``fem.elements.select(...)``     (broker element — was ``ElementChain``)
* ``results.<nodes|elements>.select(...)`` /
  ``results.elements.{gauss,fibers,layers,line_stations,springs}
  .select(...)``                   (results — was ``ResultChain``)
* ``g.mesh_selection.select(...)`` (live mesh — was
  ``MeshSelectionChain``)

It is a **new leaf** (mirrors ``mesh/_node_chain.py`` exactly): the only
module-top import is the package-root leaf
``from .._kernel.chain import SelectionChain`` (+ numpy / stdlib).
``_kernel.payloads`` and the four legacy chains are imported **deferred**
inside ``_materialize`` / ``__iter__`` / the engine-delegate factory
(mirrors ``_node_chain.py:107``), so this module adds exactly **one**
downward ``("mesh","_kernel","mesh/_mesh_selection.py")`` BASELINE triple
— the same polarity already frozen, no ``core↔mesh`` edge, no
deferred→eager flip (``tests/test_import_dag_polarity.py``).

Engine-polymorphism
-------------------
``MeshSelection`` is **engine-polymorphic**: it serves the four host
contexts above and in each behaves *identically* to the legacy chain
that host previously returned.  Rather than re-implement four copies of
the per-engine coordinate / centroid / materialise logic (which would
risk semantic drift), the engine-specific hooks are delegated to a
freshly-constructed instance of the **matching legacy chain**
(``NodeChain`` / ``ElementChain`` / ``ResultChain`` /
``MeshSelectionChain``), built from the *same* ``(_items, _engine)``.
Those legacy chains stay defined-and-importable through P2-I (P3 deletes
them — ``docs/plans/selection-unification-v2.md`` §6 P3).  This makes
the per-engine behaviour **byte-faithful by construction** (it literally
calls the legacy code) while ``MeshSelection`` owns the *unified*
surface: the ratified pair-view ``__iter__`` (HT8 / R3-C, §6.1
STOP-2(b) Option (i)), the ``.ids`` / ``.coords`` / ``.connectivity`` /
``.groups()`` accessors, the ``.values(...)`` results read, the
``.result()`` identity alias, and ``.save_as(name)``.

Set-algebra is **unaffected** by the pair-view: it is inherited from
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
  → live-mesh, delegate :class:`MeshSelectionChain`.
* ``_ResultChainEngine`` — has ``.host`` + ``.results`` (no ``.ms``)
  → results, delegate :class:`ResultChain`.
* ``ElementComposite``  — has ``._groups`` (per-type dict; no
  ``.ms``/``.host``) → broker element, delegate :class:`ElementChain`.
* ``NodeComposite``     — has ``.coords`` / ``.ids`` (no ``._groups`` /
  ``.host`` / ``.ms``) → broker node, delegate :class:`NodeChain`.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .._kernel.chain import SelectionChain


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


class MeshSelection(SelectionChain):
    """Daisy-chainable + terminal point-family selection (FEM ids).

    Engine-polymorphic across the four point host contexts (broker
    node / broker element / results / live mesh); behaviour for the
    engine-specific hooks is delegated verbatim to the matching legacy
    chain so it is byte-faithful per context (the
    selection-unification-v2 P2-I invisibility contract).  The
    *unified* surface (pair-view ``__iter__``,
    ``.ids``/``.coords``/``.connectivity``/``.groups()``/``.values()``/
    ``.result()``/``.save_as``) lives here.
    """

    FAMILY = "point"

    __slots__ = ()

    # ── engine-delegate factory (DEFERRED legacy-chain imports) ──
    def _delegate(self) -> "SelectionChain":
        """A fresh legacy chain of the engine-appropriate class.

        Constructed on-demand from this selection's *own*
        ``(_items, _engine)`` and **never stored** — it exists only to
        run the legacy per-engine ``_coords_of`` / ``_spatial_*`` /
        ``_materialize`` / ``.get`` so behaviour is byte-faithful per
        context.  Imports are **deferred** here (not at module top) so
        the new leaf keeps its single downward ``_kernel`` edge — the
        legacy chain modules each import only ``_kernel.chain`` + numpy
        at load, identical to this module
        (``tests/test_import_dag_polarity.py`` baseline unchanged).
        """
        kind = _engine_kind(self._engine)
        if kind == "live":
            from ._mesh_selection_chain import MeshSelectionChain
            return MeshSelectionChain(self._items, _engine=self._engine)
        if kind == "result":
            from ..results._result_chain import ResultChain
            return ResultChain(self._items, _engine=self._engine)
        if kind == "element":
            from ._elem_chain import ElementChain
            return ElementChain(self._items, _engine=self._engine)
        if kind == "node":
            from ._node_chain import NodeChain
            return NodeChain(self._items, _engine=self._engine)
        raise RuntimeError(
            "MeshSelection has _engine=None (constructed standalone) — "
            "it has no host engine to resolve coordinates / a terminal "
            "against. Build it via fem.nodes.select(...) / "
            "fem.elements.select(...) / results.<...>.select(...) / "
            "g.mesh_selection.select(...)."
        )

    # ── abstract per-domain hooks → delegate verbatim ───────
    def _coords_of(self, atoms: tuple) -> np.ndarray:
        # Delegate must see exactly `atoms` (the chain may call this
        # with a subset, e.g. from `where`/`nearest_to`); build it on
        # the requested atoms, not on self._items.
        d = self._delegate()
        return d._coords_of(atoms)

    def _spatial_box(self, atoms: tuple, lo, hi, *, inclusive: bool) -> tuple:
        return self._delegate()._spatial_box(
            atoms, lo, hi, inclusive=inclusive
        )

    def _spatial_sphere(self, atoms: tuple, center, radius: float) -> tuple:
        return self._delegate()._spatial_sphere(atoms, center, radius)

    def _spatial_plane(self, atoms: tuple, point, normal, tol: float) -> tuple:
        return self._delegate()._spatial_plane(atoms, point, normal, tol)

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
        # element level — (eid, conn_row).  The legacy element payload
        # differs by engine and the pair-view follows it byte-faithfully:
        #   * broker / results → ``GroupResult`` (iterate its
        #     ``ElementGroup`` blocks, each yielding ``(eid, conn)``);
        #   * live mesh → the flat ``dict`` shape
        #     ``{'element_ids', 'connectivity'}`` (the legacy
        #     ``MeshSelectionChain._materialize`` return) — zip the two
        #     into the same ``(eid, conn_row)`` pair shape.
        res = self._delegate()._materialize()
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

        Reuses the legacy materialised element payload, so the shape /
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
        res = self._delegate()._materialize()
        if isinstance(res, dict):             # live-mesh element payload
            return np.asarray(res["connectivity"])
        return res.connectivity               # GroupResult.connectivity

    def groups(self):
        """The per-type element blocks for an element selection.

        Returns the legacy ``GroupResult`` (broker / results element)
        — ``list(sel.groups())`` yields the ``ElementGroup`` blocks,
        preserving per-type ``element_type`` (needed by the OpenSees
        emitter / beam viewer; R3-B / R-v2-4).  For the live-mesh
        element engine (whose legacy terminal is a flat dict, not a
        ``GroupResult``) returns that same dict — byte-identical to the
        legacy ``MeshSelectionChain.result()``.
        """
        if self._level != "element":
            raise TypeError(
                "groups() is element-level only; this selection is "
                "node-level."
            )
        return self._delegate()._materialize()

    # ── results read — verbatim rename of ResultChain.get ───
    def values(self, *, component: str, time=None, stage=None, **extra):
        """Read the result slab for the selected ids (results engine).

        **Verbatim** behaviour of the legacy ``ResultChain.get`` — it
        forwards ``host.get(ids=list(self._items), component=,
        time=, stage=, **extra)`` to the spawning sub-composite's
        existing ``.get`` (so the slab type / id-and-value parity is
        the exact existing reader path).  ``**extra`` is forwarded
        opaquely (``gp_indices=`` / ``layer_indices=`` for the
        fibers / layers sub-composites); this method **never names**
        ``gp_indices`` / ``layer_indices`` — the spawning ``.get``
        signature stays the single source of truth (R5; the locked
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
        return self._delegate().get(
            component=component, time=time, stage=stage, **extra
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

    # ── terminal — the level/engine-appropriate legacy payload ─
    def result(self):
        return self._materialize()

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
        return self._delegate()._materialize()
