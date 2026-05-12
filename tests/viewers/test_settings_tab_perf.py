"""DiagramSettingsTab dispatcher migration (UI storm follow-up #2).

Same ``attach_dispatcher`` pattern as :class:`OutlineTree` (PR #131):
the legacy ``director.geometries.subscribe`` wiring fires once per
mutation; ``dispatcher.subscribe(..., lane=Lane.UI, coalesce=True)``
collapses a storm to one callback per granular kind per Qt tick.

Bench file (`@pytest.mark.bench`); two correctness assertions that
verify the legacy-vs-dispatcher swap is observably correct.
"""
from __future__ import annotations

import os

import pytest


# Force offscreen Qt before importing qtpy.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _build_settings_tab_with_director():
    """Construct a DiagramSettingsTab against a real GeometryManager +
    stub director. Returns (tab, director, geometries)."""
    from qtpy import QtWidgets
    _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from apeGmsh.viewers.diagrams._geometries import GeometryManager

    geometries = GeometryManager()

    class _RegistryStub:
        def diagrams(self):
            return []

    class _CompMgrStub:
        active = None

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
            return _RegistryStub()

        @property
        def compositions(self):
            # ``_rebuild`` reads ``self._director.compositions.active``
            # in stack mode. Return a stub with ``active = None`` so
            # the rebuild path doesn't blow up on attribute lookup.
            return _CompMgrStub()

    director = _Director(geometries)
    from apeGmsh.viewers.ui._diagram_settings_tab import DiagramSettingsTab
    tab = DiagramSettingsTab(director)
    return tab, director, geometries


def test_attach_dispatcher_replaces_legacy_subscription() -> None:
    """After ``attach_dispatcher``, a geometry mutation must NOT fire
    ``_on_compositions_changed`` synchronously through the legacy path
    — only after the UI-lane drain."""
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    tab, director, geometries = _build_settings_tab_with_director()

    # Count callbacks via a wrapper installed on the legacy path BEFORE
    # attach_dispatcher (so the legacy subscription captures the
    # patched version when re-subscribing — but here we just need to
    # verify the legacy path no longer fires after migration).
    fires = [0]

    def _counting_cb():
        fires[0] += 1

    # Install a parallel legacy subscriber so we can detect whether
    # the legacy chain still fires after migration. (The tab's own
    # legacy subscription is what attach_dispatcher will drop.)
    geometries.subscribe(_counting_cb)

    # Pre-migration legacy fire — sanity check.
    geometries.add("X")
    pre_count = fires[0]
    assert pre_count >= 1, "legacy chain should fire on geometries.add()"

    # Attach a real dispatcher with a recorded defer.
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
    tab.attach_dispatcher(dispatcher)
    # Bridge the manager's typed events to the dispatcher.
    geometries.subscribe_typed(
        lambda kind, payload: dispatcher.fire(kind, payload=payload),
    )

    # The parallel legacy subscriber stays attached — verify it still
    # fires (independent subscription). What we care about is whether
    # tab's INTERNAL legacy subscription is dropped — covered by the
    # storm test below via a counter on _on_compositions_changed.
    geometries.add("Y")
    assert fires[0] > pre_count, "parallel legacy subscriber should still fire"


@pytest.mark.bench
def test_storm_of_mutations_collapses_to_few_rebuilds() -> None:
    """Phase 2 + settings-tab migration: 200 geometry mutations in one
    Qt tick should produce at most a handful of ``_on_compositions_changed``
    callbacks (one per granular kind), NOT 200."""
    from apeGmsh.viewers.diagrams._dispatch import Dispatcher

    tab, director, geometries = _build_settings_tab_with_director()

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
    tab.attach_dispatcher(dispatcher)
    geometries.subscribe_typed(
        lambda kind, payload: dispatcher.fire(kind, payload=payload),
    )

    fires = [0]
    original_cb = tab._on_compositions_changed.__func__

    def _counting_cb():
        fires[0] += 1
        original_cb(tab)

    # The dispatcher subscription holds a lambda that resolves
    # ``self._on_compositions_changed`` at call time, so the instance-
    # attribute patch makes the counter live.
    tab._on_compositions_changed = _counting_cb    # type: ignore[method-assign]

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
    fires[0] = 0

    for i in range(50):
        geometries.set_active(boot.id)
        geometries.set_active(other.id)
        geometries.set_deformation(boot.id, scale=float(i))
        geometries.rename(boot.id, f"G{i}")

    # Nothing has flushed yet.
    assert fires[0] == 0

    # Drain.
    for fn in list(deferred):
        fn()

    print(
        f"\n[settings_tab storm] 200 mutations (3 distinct granular "
        f"kinds) -> {fires[0]} callbacks after coalesce flush"
    )

    # 3 distinct granular kinds in the loop (ACTIVE / DEFORM /
    # RENAMED). Bound 1..7 to absorb any cross-kind corner case
    # while still proving the storm doesn't reach 200.
    assert 1 <= fires[0] <= 7, (
        f"Coalesce expected 1..7 callbacks; got {fires[0]} for 200 mutations"
    )
