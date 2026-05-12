"""UI tree perf benches — validate hot rebuild paths.

Marked ``@pytest.mark.bench`` so it only runs with ``pytest -m bench``.
Uses the offscreen Qt platform so the test does not require a display.

Two benches:

1. ``test_parts_tree_refresh_perf`` — measures PartsTreePanel.refresh()
   cost for one part containing 10 000 leaf entities. The current
   implementation calls ``QTreeWidgetItem(parent)`` 10 000 times — each
   triggers a tree-widget-side insertion notification because there is
   no ``setUpdatesEnabled(False)`` / ``blockSignals`` wrap. We expect
   the wrap to roughly halve the wall time.

2. ``test_outline_tree_full_rebuild_perf`` — measures the OutlineTree
   _refresh_diagrams full tear-down vs. the same number of nodes
   already present. Demonstrates the cost of ``takeChildren()`` +
   re-add on geometries × compositions matrices that grow with the
   user's session.
"""
from __future__ import annotations

import os
import time

import pytest


# Force offscreen Qt before importing qtpy.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.mark.bench
def test_parts_tree_refresh_perf() -> None:
    """PartsTreePanel.refresh() should beat a 1 s budget for 10k entities.

    More importantly, this exposes that one giant refresh path adds
    items in a tight Python loop without the ``setUpdatesEnabled``
    guard. Compare A (current) and B (with guard) wall times.
    """
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Stub registries: shape only — PartsTreePanel reads .instances and
    # .all_entities(); the values just need to iterate.
    class _Inst:
        def __init__(self, entities):
            self.entities = entities

    class _Parts:
        def __init__(self, instances):
            self.instances = instances

    class _Reg:
        def __init__(self, all_dts):
            self._all = all_dts

        def all_entities(self, dim=None):
            if dim is None:
                return list(self._all)
            return [(d, t) for (d, t) in self._all if d == dim]

    n = 10_000
    surfaces = list(range(1, n + 1))
    parts = _Parts({"BigPart": _Inst({2: surfaces})})
    entity_reg = _Reg([(2, t) for t in surfaces])

    from apeGmsh.viewers.ui._parts_tree import PartsTreePanel

    # ── A: current implementation ─────────────────────────────────────
    panel = PartsTreePanel(parts, entity_reg)
    # Already refreshed once in __init__ — measure a second call.
    t0 = time.perf_counter()
    panel.refresh()
    t_curr = time.perf_counter() - t0
    print(f"\n[parts_tree A] current refresh({n} leaves): {t_curr*1000:.1f} ms")

    # ── B: simulate the wrap by guarding the tree directly ───────────
    tree = panel._tree
    tree.setUpdatesEnabled(False)
    tree.blockSignals(True)
    try:
        t0 = time.perf_counter()
        panel.refresh()
        t_wrap = time.perf_counter() - t0
    finally:
        tree.blockSignals(False)
        tree.setUpdatesEnabled(True)
    print(f"[parts_tree B] wrapped refresh({n} leaves):  {t_wrap*1000:.1f} ms")

    speedup = t_curr / max(t_wrap, 1e-9)
    print(f"[parts_tree] speedup factor: {speedup:.2f}x")

    # Hard budget — protects against future regressions.
    assert t_curr < 5.0, f"refresh too slow: {t_curr*1000:.0f} ms"


@pytest.mark.bench
def test_outline_tree_full_rebuild_perf() -> None:
    """OutlineTree._refresh_diagrams full rebuild on geom×comp matrix.

    Demonstrates that the current "takeChildren + add" pattern scales
    O(G*C) with no incremental update; for sessions with G=10 geometries
    × C=20 compositions each, every geometry-events fires a complete
    teardown.
    """
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Stub director shape used by OutlineTree.
    class _Comp:
        def __init__(self, cid: str, name: str, layers):
            self.id = cid
            self.name = name
            self.layers = layers

    class _CompMgr:
        def __init__(self, comps, active_id):
            self.compositions = comps
            self.active_id = active_id

        def find(self, cid):
            for c in self.compositions:
                if c.id == cid:
                    return c
            return None

    class _Geom:
        def __init__(self, gid, name, comps):
            self.id = gid
            self.name = name
            self.compositions = _CompMgr(comps, comps[0].id if comps else None)
            self.deform_enabled = False
            self.deform_field = None
            self.deform_scale = 1.0

    class _GeomMgr:
        def __init__(self, geoms):
            self.geometries = geoms
            self.active_id = geoms[0].id if geoms else None

        def find(self, gid):
            for g in self.geometries:
                if g.id == gid:
                    return g
            return None

        def geometry_for_composition(self, cid):
            for g in self.geometries:
                if g.compositions.find(cid):
                    return g
            return None

        def subscribe(self, cb):
            return lambda: None

    class _Director:
        def __init__(self, geoms):
            self.geometries = _GeomMgr(geoms)
            self.stage_id = "stage0"

        def stages(self):
            return []

        def subscribe_stage(self, cb):
            return lambda: None

        def subscribe_diagrams(self, cb):
            return lambda: None

        @property
        def registry(self):
            class _R:
                def diagrams(self_inner):
                    return []
            return _R()

    # 10 geometries, 20 compositions each = 200 nodes per refresh.
    G, C = 10, 20
    geoms = [
        _Geom(
            f"g{gi}", f"Geometry {gi}",
            [_Comp(f"c{gi}_{ci}", f"Diagram {ci}", []) for ci in range(C)],
        )
        for gi in range(G)
    ]
    director = _Director(geoms)

    from apeGmsh.viewers.ui._outline_tree import OutlineTree

    # Avoid theme subscription triggering import paths during tear-down.
    tree = OutlineTree(director)

    # Warm one round.
    tree._refresh_diagrams()

    iterations = 50
    t0 = time.perf_counter()
    for _ in range(iterations):
        tree._refresh_diagrams()
    elapsed = time.perf_counter() - t0
    per_call = elapsed / iterations * 1000.0
    print(
        f"\n[outline] _refresh_diagrams × {iterations} on {G}×{C} matrix: "
        f"{elapsed*1000:.1f} ms total, {per_call:.2f} ms/call"
    )

    # On a typical workstation each refresh should fit well under 50 ms;
    # the current implementation easily blows past that on larger
    # session matrices because it tears the entire group every time.
    assert per_call < 200.0, f"refresh way too slow: {per_call:.1f} ms/call"


# =====================================================================
# Dispatcher migration — attach_dispatcher swaps the subscription
# =====================================================================

def _build_outline_with_director():
    """Construct an OutlineTree against a real GeometryManager + stub
    director. Returns (tree, director, geometries)."""
    from qtpy import QtWidgets
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from apeGmsh.viewers.diagrams._geometries import GeometryManager

    geometries = GeometryManager()

    class _Director:
        def __init__(self, geoms):
            self.geometries = geoms
            self.stage_id = None

        def stages(self):
            return []

        def subscribe_stage(self, _cb):
            return lambda: None

        def subscribe_diagrams(self, _cb):
            return lambda: None

        @property
        def registry(self):
            class _R:
                def diagrams(self_inner):
                    return []
            return _R()

    director = _Director(geometries)
    from apeGmsh.viewers.ui._outline_tree import OutlineTree
    tree = OutlineTree(director)
    return tree, director, geometries


def test_attach_dispatcher_replaces_legacy_subscription() -> None:
    """After ``attach_dispatcher``, a geometry mutation must NOT fire
    the rebuild via the legacy ``director.geometries.subscribe`` path
    (otherwise every mutation rebuilds twice — once legacy, once UI).

    Verified via the tree's observable state: child count under the
    Geometries group must change synchronously on legacy mutations,
    and only after a UI-lane drain on dispatcher-attached mutations.
    """
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    tree, director, geometries = _build_outline_with_director()

    initial = tree._group_diagrams.childCount()
    assert initial >= 1, "OutlineTree should render the bootstrap geometry"

    # Legacy path — synchronous rebuild on mutation.
    geometries.add("X")
    assert tree._group_diagrams.childCount() == initial + 1

    # Attach a real dispatcher with a recorded defer (manual flush).
    deferred = []
    dispatcher = Dispatcher(
        director=director,
        pump_step=lambda _l: None,
        pump_deform=lambda _l: None,
        pump_gate=lambda: None,
        pump_restack=lambda: None,
        render=lambda: None,
        defer_fn=deferred.append,
    )
    tree.attach_dispatcher(dispatcher)
    # Bridge the manager's typed events to the dispatcher (ResultsViewer
    # does this wiring in production; we replicate it for the test).
    geometries.subscribe_typed(
        lambda kind, payload: dispatcher.fire(kind, payload=payload),
    )

    before = tree._group_diagrams.childCount()
    geometries.add("Y")    # fires GEOMETRY_ADDED via typed → dispatcher.

    # Legacy path is gone → no synchronous rebuild.
    assert tree._group_diagrams.childCount() == before, (
        "Legacy subscription still firing — attach_dispatcher must "
        "have dropped it"
    )

    # Drain UI lane → tree updates.
    for fn in deferred:
        fn()
    assert tree._group_diagrams.childCount() == before + 1


@pytest.mark.bench
def test_storm_of_mutations_collapses_to_few_rebuilds() -> None:
    """Phase 2.1 + outline migration: 200 geometry mutations in one Qt
    tick should produce at most a handful of rebuilds (one per
    granular kind), NOT 200.

    We attach the dispatcher first so the rebuild path goes through
    the late-binding lambda installed by ``attach_dispatcher`` —
    ``lambda _k, _p: self._refresh_diagrams()`` — which DOES look up
    the attribute at call time. That's how we install a rebuild counter
    that the dispatcher path will use.
    """
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    tree, director, geometries = _build_outline_with_director()

    deferred = []
    dispatcher = Dispatcher(
        director=director,
        pump_step=lambda _l: None,
        pump_deform=lambda _l: None,
        pump_gate=lambda: None,
        pump_restack=lambda: None,
        render=lambda: None,
        defer_fn=deferred.append,
    )
    tree.attach_dispatcher(dispatcher)
    geometries.subscribe_typed(
        lambda kind, payload: dispatcher.fire(kind, payload=payload),
    )

    rebuilds = [0]
    original_refresh = tree._refresh_diagrams.__func__    # underlying fn

    def _counting_refresh():
        rebuilds[0] += 1
        original_refresh(tree)

    # The dispatcher subscription holds a lambda that resolves
    # ``self._refresh_diagrams`` at call time — overriding the
    # instance attribute makes the counter live.
    tree._refresh_diagrams = _counting_refresh    # type: ignore[method-assign]

    # 200 mutations of mixed granular kinds.
    boot = geometries.geometries[0]
    other = geometries.add("Other")
    # Drain the GEOMETRY_ADDED from "Other" so the dispatcher's
    # ``_ui_flush_scheduled`` flag resets — otherwise subsequent fires
    # see "flush already scheduled" and the loop below never
    # re-schedules one.
    for fn in list(deferred):
        fn()
    deferred.clear()
    rebuilds[0] = 0

    for i in range(50):
        geometries.set_active(boot.id)
        geometries.set_active(other.id)
        geometries.set_deformation(boot.id, scale=float(i))
        geometries.rename(boot.id, f"G{i}")

    # Nothing has flushed yet — all queued on UI lane.
    assert rebuilds[0] == 0

    # Drain — coalesce should fold the storm down dramatically.
    for fn in list(deferred):
        fn()

    print(
        f"\n[outline storm] 200 mutations (3 distinct granular kinds) -> "
        f"{rebuilds[0]} rebuilds after coalesce flush"
    )

    # Coalesce contract: 3 distinct granular kinds fired in the loop
    # (ACTIVE / DEFORM / RENAMED). Each coalesces to one handler call.
    # Lower bound prevents a silent "no rebuild happened" regression;
    # upper bound is 7 to absorb any cross-kind corner case while still
    # proving the storm doesn't reach 200.
    assert 1 <= rebuilds[0] <= 7, (
        f"Coalesce expected 1..7 rebuilds; got {rebuilds[0]} for "
        f"200 mutations"
    )
