"""Single source of truth for mesh.viewer overlay visibility state.

Pre-PR5 mesh.viewer kept two independent stores for overlay
visibility: the outline tree's eye-icons (which read their own
``_active_load_patterns()`` snapshot off Qt widget state) and the
right-side tab panels' checkboxes (which read their own
``active_patterns()`` / ``active_kinds()`` off Qt widget state).
Both fanned into the same ``_rebuild_*_overlay`` methods on
mesh_viewer.  Alternating between the two surfaces caused the overlay
to flip to whichever surface fired last — documented at
``_mesh_outline_tree.py:96-104`` as a deliberate follow-up.

This module is that follow-up.  :class:`OverlayVisibilityModel` is a
plain-Python (no Qt) state object holding the canonical
``{load_patterns, constraint_kinds, mass_visible}`` triple.  Both
the outline tree and the tab panels read from and write to this
model; mesh_viewer subscribes to the model's ``on_changed`` callback
for the actual ``_rebuild_*`` calls.

Observer pattern, not Qt signals, so the model stays testable
without a ``QApplication``.  Idempotent setters (no-op when the
new value equals the current value) keep the observer chain quiet
when a write reflects state that was already set by the other
surface — preventing the round-trip oscillation that a naive MVC
would create.
"""
from __future__ import annotations

from typing import Any, Callable

# Overlay scale keys -> the overlay-rebuild scope key the dispatcher's
# ``overlays`` pump receives (ADR 0056 V3). The scales used to live in
# a module-private dict on mesh_viewer (``_overlay_scales``); they are
# intent state and belong to this owner.
_SCALE_KEY_TO_OVERLAY: dict[str, str] = {
    "force_arrow": "loads",
    "moment_arrow": "loads",
    "mass_sphere": "mass",
    "constraint_marker": "constraints",
    "constraint_line": "constraints",
    "tangent_normal_arrow": "tangent",
}


class OverlayVisibilityModel:
    """Canonical state for mesh.viewer overlay visibility.

    Four fields:

    * ``load_patterns: frozenset[str]`` — names of load patterns
      currently rendered (LoadsTabPanel + Loads outline section).
    * ``constraint_kinds: frozenset[str]`` — kinds (e.g. ``"rigid_link"``,
      ``"node_to_surface"``) currently rendered.
    * ``mass_visible: bool`` — single flag for the mass overlay.
    * ``boundary_nodes_visible: bool`` — single flag for the
      cross-partition boundary-node glyph overlay (schema 2.10.0 /
      ADR 0027).  Toggled by the "Boundary nodes" row in the outline
      tree's Partitions section.

    Setters are idempotent: setting a value equal to the current one
    does NOT fire observers.  This is what breaks the
    outline-eye ↔ tab-checkbox oscillation: when the tab panel mirrors
    a state already set by the outline, the mirror write is a no-op
    and nobody re-renders.

    Observers receive zero arguments.  Subscribers read the model's
    public attributes for the new state (typically only one of the
    four fields changes per write, but observers don't need to know
    which one — ``_rebuild_*`` calls are cheap enough to refire all
    four on any change).
    """

    __slots__ = (
        "_load_patterns",
        "_constraint_kinds",
        "_mass_visible",
        "_boundary_nodes_visible",
        "_scales",
        "_observers",
        "dispatcher",
    )

    def __init__(self) -> None:
        self._load_patterns: frozenset[str] = frozenset()
        self._constraint_kinds: frozenset[str] = frozenset()
        self._mass_visible: bool = False
        self._boundary_nodes_visible: bool = False
        # Overlay glyph scale multipliers (ADR 0056 V3 — owned intent
        # state, formerly mesh_viewer's module-private _overlay_scales).
        self._scales: dict[str, float] = {
            key: 1.0 for key in _SCALE_KEY_TO_OVERLAY
        }
        self._observers: list[Callable[[], None]] = []
        # Injected by the mesh viewer (ADR 0056 V3): when set, every
        # successful mutation owner-fires MESH_OVERLAY_CHANGED with the
        # affected overlay key — the dispatcher's ``overlays`` pump
        # rebuilds just that overlay and the render coalesces. The
        # plain observers keep firing either way (UI-sync subscribers
        # like the outline tree).
        self.dispatcher: Any = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def load_patterns(self) -> frozenset[str]:
        return self._load_patterns

    @property
    def constraint_kinds(self) -> frozenset[str]:
        return self._constraint_kinds

    @property
    def mass_visible(self) -> bool:
        return self._mass_visible

    @property
    def boundary_nodes_visible(self) -> bool:
        return self._boundary_nodes_visible

    @property
    def scales(self) -> dict[str, float]:
        """Read-only view of the overlay glyph scale multipliers."""
        return dict(self._scales)

    def scale(self, key: str) -> float:
        return self._scales[key]

    def set_load_patterns(self, patterns) -> None:
        new = frozenset(patterns)
        if new == self._load_patterns:
            return
        self._load_patterns = new
        self._fire("loads")

    def set_constraint_kinds(self, kinds) -> None:
        new = frozenset(kinds)
        if new == self._constraint_kinds:
            return
        self._constraint_kinds = new
        self._fire("constraints")

    def set_mass_visible(self, visible: bool) -> None:
        new = bool(visible)
        if new == self._mass_visible:
            return
        self._mass_visible = new
        self._fire("mass")

    def set_boundary_nodes_visible(self, visible: bool) -> None:
        new = bool(visible)
        if new == self._boundary_nodes_visible:
            return
        self._boundary_nodes_visible = new
        self._fire("boundary")

    def set_scale(self, key: str, value: float) -> None:
        """Set one overlay glyph scale multiplier (owner-fired).

        ``key`` is one of the keys in ``_SCALE_KEY_TO_OVERLAY``;
        unknown keys fail loud (a typo'd scale silently doing nothing
        is the bug class ADR 0056 INV-6 forbids).
        """
        if key not in self._scales:
            raise KeyError(
                f"Unknown overlay scale {key!r}; expected one of "
                f"{sorted(self._scales)}."
            )
        new = float(value)
        if new == self._scales[key]:
            return
        self._scales[key] = new
        self._fire(_SCALE_KEY_TO_OVERLAY[key])

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Register a zero-argument callback.

        The model fires every subscribed callback on any successful
        state change.  Callbacks read the model's public properties
        for the new state.
        """
        self._observers.append(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        """Remove a previously-registered callback.  No-op if absent."""
        try:
            self._observers.remove(callback)
        except ValueError:
            pass

    def _fire(self, overlay_key: "str | None" = None) -> None:
        # Owner-fired dispatch first (ADR 0056 Part 2): the pump
        # rebuilds the affected overlay + the render coalesces; the
        # plain observers (UI-sync, e.g. outline tree) then see the
        # post-rebuild state. Snapshot the list so observers that
        # unsubscribe during their own callback don't mutate the
        # iteration.
        if self.dispatcher is not None:
            from ..diagrams._dispatch import MESH_OVERLAY_CHANGED
            self.dispatcher.fire(MESH_OVERLAY_CHANGED, layer=overlay_key)
        for cb in list(self._observers):
            cb()


__all__ = ["OverlayVisibilityModel"]
