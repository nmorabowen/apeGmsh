"""ScalarBarSupport — a diagram's *registration* with the viewer's legends.

Under ADR 0081 the colour scale is a view annotation owned by the
viewer's :class:`~apeGmsh.viewers.core._legend.LegendController`, not a
property of the diagram that happens to be coloured. This mixin is what
is left of the old per-diagram bar: two lifecycle calls and a handful of
setters that forward to the controller.

* ``_register_scalar_bar()`` from ``attach`` (after the layer exists),
  ``_unregister_scalar_bar()`` from ``detach``.
* ``set_show_scalar_bar`` / ``set_fmt`` / ``set_scalar_bar_vertical`` /
  ``set_scalar_bar_scale`` mutate the controller's entry. They keep
  their names because the settings tab and presets call them.

What this mixin deliberately no longer has: bar geometry, orientation
and size state of its own (the controller owns them, so a user-placed
legend survives every recolour and re-attach), a second layout
implementation for the un-migrated plotter path (there are no
un-migrated diagrams left), and an on-demand title that add and remove
could disagree about (the legend key is stable; the *title* collapses
in the controller).

The style dataclass's four scalar-bar fields survive as **initial
preferences**, consumed once at register time.
"""
from __future__ import annotations

from typing import Any, Optional


def _default_vertical(preference: Optional[bool]) -> bool:
    """Resolve a style's tri-state orientation preference.

    ``None`` means "no preference", and the default is **vertical**:
    ``vtkScalarBarActor``'s vertical layout is the one it gets right,
    and a bar down the side does not eat the bottom of the model view
    the way a full-width horizontal one does.
    """
    return True if preference is None else bool(preference)


class ScalarBarSupport:
    """Mixin: registers a diagram's layer with the viewer's legends."""

    def _init_scalar_bar_state(self) -> None:
        # The legend key this diagram registered under (None until
        # attach). The controller owns everything behind it.
        self._legend_key: Optional[tuple[str, str]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _legends(self) -> Any:
        """The controller owning this diagram's viewport, or ``None``."""
        backend = getattr(self, "_backend", None)
        if backend is None:
            return None
        from ..core._legend import controller_for
        return controller_for(backend)

    def _legend_geometry(self) -> str:
        """Concurrent-geometry discriminator for the legend key.

        ADR 0058 S2b stamps a resolver naming the owning geometry when
        more than one is visible. It feeds the *key* here rather than
        the title: keying keeps two geometries' scales apart even while
        only one is on screen, so add and remove can never disagree
        about which bar they mean.
        """
        resolver = getattr(self, "_bar_prefix_resolver", None)
        if resolver is None:
            return ""
        try:
            return str(resolver(self) or "")
        except Exception:
            return ""

    def _register_scalar_bar(self) -> None:
        """Bind this diagram's layer to its legend. Idempotent."""
        legends = self._legends()
        handle = getattr(self, "_handle", None)
        if legends is None or handle is None:
            return
        if not self._scalar_bar_is_enabled():
            return
        style = getattr(self.spec, "style", None)
        self._legend_key = (
            self._legend_geometry(), self.spec.selector.component,
        )
        legends.register(
            self._legend_key,
            handle.layer_id,
            handle,
            lut=self._current_lutspec(),
            fmt=getattr(style, "fmt", "%.3g") or "%.3g",
            vertical=_default_vertical(getattr(style, "scalar_bar_vertical", None)),
            font_scale=getattr(style, "scalar_bar_scale", None),
            visible=bool(getattr(style, "show_scalar_bar", True)),
        )

    def _unregister_scalar_bar(self) -> None:
        """Release this diagram's claim on its legend. Idempotent."""
        legends = self._legends()
        handle = getattr(self, "_handle", None)
        if legends is not None and handle is not None:
            legends.unregister(handle.layer_id)
        self._legend_key = None

    # ------------------------------------------------------------------
    # Live setters (settings tab, presets, context menu)
    # ------------------------------------------------------------------

    def set_show_scalar_bar(self, show: bool) -> None:
        """Show / hide this diagram's colour scale."""
        self._legend_mutate("set_visible", bool(show))

    def set_fmt(self, fmt: str) -> None:
        """Update the tick-label ``printf`` format.

        The box is derived from the formatted labels (ADR 0081 Part 3),
        so a wider format widens the legend rather than overflowing it.
        """
        self._legend_mutate("set_fmt", str(fmt))

    def set_scalar_bar_vertical(self, vertical: bool) -> None:
        self._legend_mutate("set_vertical", bool(vertical))

    def set_scalar_bar_scale(self, scale: float) -> None:
        """Scale the legend's text — and with it, its box."""
        self._legend_mutate("set_font_scale", float(scale))

    def _legend_mutate(self, method: str, value: Any) -> None:
        legends = self._legends()
        if legends is None or self._legend_key is None:
            return
        getattr(legends, method)(self._legend_key, value)

    # ------------------------------------------------------------------
    # Reads (settings tab restores its widgets from these)
    # ------------------------------------------------------------------

    def _legend_entry(self) -> Any:
        legends = self._legends()
        if legends is None or self._legend_key is None:
            return None
        return legends.entry(self._legend_key)

    def _effective_show_scalar_bar(self) -> bool:
        entry = self._legend_entry()
        if entry is not None:
            return bool(entry.visible)
        style = getattr(self.spec, "style", None)
        return bool(getattr(style, "show_scalar_bar", True))

    def _effective_bar_vertical(self) -> bool:
        entry = self._legend_entry()
        if entry is not None:
            return bool(entry.vertical)
        style = getattr(self.spec, "style", None)
        return _default_vertical(getattr(style, "scalar_bar_vertical", None))

    def _effective_bar_scale(self) -> float:
        entry = self._legend_entry()
        if entry is not None and entry.font_scale is not None:
            return float(entry.font_scale)
        style = getattr(self.spec, "style", None)
        return float(getattr(style, "scalar_bar_scale", 1.0) or 1.0)

    def _effective_fmt(self) -> str:
        entry = self._legend_entry()
        if entry is not None:
            return entry.fmt
        style = getattr(self.spec, "style", None)
        return getattr(style, "fmt", "%.3g") or "%.3g"

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _scalar_bar_is_enabled(self) -> bool:
        """Whether this diagram contributes a legend at all.

        Default: True. ``VectorGlyph`` overrides to gate on
        ``use_magnitude_colors``.
        """
        return True

    def _current_lutspec(self) -> Any:
        """Plain ``LutSpec`` snapshot of the host's Qt LUT mirror."""
        from ..scene_ir import LutSpec
        lut = getattr(self, "_lut", None)
        if lut is not None:
            return LutSpec(
                name=getattr(lut, "preset", "viridis"),
                vmin=float(getattr(lut, "vmin", 0.0)),
                vmax=float(getattr(lut, "vmax", 1.0)),
                log_scale=bool(getattr(lut, "log_scale", False)),
            )
        return LutSpec()
