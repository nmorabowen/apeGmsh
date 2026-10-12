"""UI tree perf benches — validate hot rebuild paths.

Marked ``@pytest.mark.bench`` so it only runs with ``pytest -m bench``.
Uses the offscreen Qt platform so the test does not require a display.

One bench: ``test_parts_tree_refresh_perf`` — measures
PartsTreePanel.refresh() cost for one part containing 10 000 leaf
entities. The current implementation calls ``QTreeWidgetItem(parent)``
10 000 times — each triggers a tree-widget-side insertion notification
because there is no ``setUpdatesEnabled(False)`` / ``blockSignals``
wrap. We expect the wrap to roughly halve the wall time.

(The legacy OutlineTree benches went with ``ui/_outline_tree.py``,
X3-d #1734.)
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
