"""OpacityController — per-actor SetOpacity + depth-peel auto-toggle.

Tracks per-actor opacity overrides so that:

* Picker correctness does **not** depend on depth peeling — the mode
  routing in :class:`PickEngine` (Phase 3.2) is what enables clean
  GP/fiber picking through translucent solids. Depth peeling here
  exists purely for rendering fidelity: when one or more actors drop
  below opacity 1.0, the plotter switches into peeled rendering so
  the scene composes correctly; when every actor returns to opaque,
  peeling switches off (it has a non-trivial GPU cost).

* The auto-toggle decision is one boolean — "is any registered actor
  translucent?" — so we don't need a full priority list. We just track
  a count: increment when a new actor goes from 1.0 to <1.0, decrement
  when one returns. The plotter call fires only on the 0↔1 transition.

* Every mutation publishes ``OPACITY_CHANGED`` through the injected
  dispatcher. RENDER-lane subscribers can update HUD labels (e.g., a
  "translucent" indicator) without polling.
"""
from __future__ import annotations

from typing import Any


# Default depth-peel knobs match the prompt's recommendation. 4 peels
# is enough for two overlapping translucent surfaces (substrate +
# diagram); occlusion ratio 0.1 caps the iteration cost on dense
# meshes.
_DEFAULT_NUMBER_OF_PEELS: int = 4
_DEFAULT_OCCLUSION_RATIO: float = 0.1


class OpacityController:
    """Per-actor opacity + depth-peel auto-toggle.

    Construct with the plotter; opt the dispatcher in afterwards.
    ``set_opacity(actor, value)`` is the only mutation surface — it
    routes through the actor's ``vtkProperty.SetOpacity``, tracks the
    translucent count, and decides whether the plotter needs peeling
    on or off.
    """

    def __init__(
        self,
        plotter: Any,
        *,
        number_of_peels: int = _DEFAULT_NUMBER_OF_PEELS,
        occlusion_ratio: float = _DEFAULT_OCCLUSION_RATIO,
    ) -> None:
        self._plotter = plotter
        self._n_peels = int(number_of_peels)
        self._occlusion = float(occlusion_ratio)
        # ``id(actor)`` -> ``(opacity, actor)``. The actor reference is
        # retained so ``restore_all`` can reset every tracked actor's
        # VTK opacity without the caller having to re-supply the list.
        self._opacities: dict[int, tuple[float, Any]] = {}
        # Whether the plotter is currently in peeled-render mode. The
        # toggle calls are idempotent on most pyvista versions but we
        # gate on this flag to keep telemetry honest.
        self._peeling_enabled: bool = False
        # Wired by ResultsViewer for OPACITY_CHANGED dispatch.
        self.dispatcher: Any = None

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def set_opacity(self, actor: Any, value: float) -> None:
        """Set ``actor``'s opacity in [0, 1]. Idempotent on same-value
        re-sets so RENDER-lane subscribers don't churn on slider drag."""
        if actor is None:
            return
        v = max(0.0, min(1.0, float(value)))
        prev_entry = self._opacities.get(id(actor))
        prev = prev_entry[0] if prev_entry is not None else None
        if prev is not None and prev == v:
            return
        try:
            actor.GetProperty().SetOpacity(v)
        except Exception:
            return
        was_translucent = prev is not None and prev < 1.0
        now_translucent = v < 1.0
        if v >= 1.0:
            # Restored — drop from tracking entirely so ``len(translucent)``
            # reflects only actors actually below 1.0.
            self._opacities.pop(id(actor), None)
        else:
            self._opacities[id(actor)] = (v, actor)
        # Auto-toggle depth peeling on the 0↔1 boundary.
        if now_translucent and not was_translucent:
            self._maybe_enable_peeling()
        elif not now_translucent and was_translucent:
            self._maybe_disable_peeling()
        self._fire_changed(actor, v)

    def restore_opacity(self, actor: Any) -> None:
        """Shortcut for ``set_opacity(actor, 1.0)``."""
        self.set_opacity(actor, 1.0)

    def restore_all(self) -> None:
        """Reset every tracked actor to fully opaque + disable peeling."""
        if not self._opacities:
            return
        # Snapshot the entries first because ``set_opacity`` mutates
        # ``self._opacities`` (pops the key when value reaches 1.0).
        entries = list(self._opacities.values())
        for _v, actor in entries:
            try:
                actor.GetProperty().SetOpacity(1.0)
            except Exception:
                continue
        self._opacities.clear()
        self._maybe_disable_peeling()
        self._fire_changed(None, 1.0)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def opacity_of(self, actor: Any) -> float:
        """Return the last value we set on ``actor`` (1.0 if untracked)."""
        entry = self._opacities.get(id(actor))
        return entry[0] if entry is not None else 1.0

    def n_translucent(self) -> int:
        return len(self._opacities)

    @property
    def peeling_enabled(self) -> bool:
        return self._peeling_enabled

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_enable_peeling(self) -> None:
        if self._peeling_enabled or self._plotter is None:
            return
        try:
            self._plotter.enable_depth_peeling(
                number_of_peels=self._n_peels,
                occlusion_ratio=self._occlusion,
            )
            self._peeling_enabled = True
        except Exception:
            pass

    def _maybe_disable_peeling(self) -> None:
        # Only disable when the last translucent actor returns to 1.0.
        if not self._peeling_enabled or self._opacities:
            return
        if self._plotter is None:
            return
        try:
            self._plotter.disable_depth_peeling()
            self._peeling_enabled = False
        except Exception:
            pass

    def _fire_changed(self, actor: Any, value: float) -> None:
        if self.dispatcher is None:
            return
        try:
            from ..diagrams._dispatch import OPACITY_CHANGED
            self.dispatcher.fire(
                OPACITY_CHANGED, payload=(actor, value),
            )
        except Exception:
            pass


__all__ = ["OpacityController"]
