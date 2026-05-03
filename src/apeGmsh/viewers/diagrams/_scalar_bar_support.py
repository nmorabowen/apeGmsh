"""ScalarBarSupport — shared scalar-bar lifecycle for diagram classes.

Every contour-bearing diagram (Contour, VectorGlyph, GaussPoint,
FiberSection, LayerStack) attaches a ``vtkScalarBarActor`` via
PyVista's ``add_mesh(show_scalar_bar=True, scalar_bar_args=...)``. The
mixin centralises three concerns that were duplicated five times:

* **Live show/hide** — ``set_show_scalar_bar(bool)`` removes the bar
  from the plotter's registry on ``False``, and re-adds it via
  ``add_scalar_bar`` against the active actor's mapper on ``True``.
  The bar is keyed by the component name (``spec.selector.component``)
  so multiple diagrams can coexist without colliding.
* **Live format** — ``set_fmt(str)`` updates the bar's
  ``SetLabelFormat`` directly when present; the runtime override also
  feeds future re-creations (after toggling visibility off then on).
* **Leak-free detach** — ``_remove_scalar_bar(title)`` is the cleanup
  call that ``detach()`` must invoke before tearing the actor down.

The mixin assumes the host class is a :class:`~apeGmsh.viewers.
diagrams._base.Diagram` — it relies on ``self._plotter``,
``self._actor``, and ``self.spec`` being set by the time methods
fire. Diagrams whose bar is conditional (e.g. VectorGlyph only paints
a bar when ``use_magnitude_colors`` is True) override
``_scalar_bar_is_enabled`` to gate ``set_show_scalar_bar``.
"""
from __future__ import annotations

from typing import Any, Optional


class ScalarBarSupport:
    """Mixin providing live show/hide + fmt for a diagram's scalar bar.

    Hosts must call ``self._init_scalar_bar_state()`` from their
    ``__init__``, and ``self._remove_scalar_bar(self.spec.selector.
    component)`` from their ``detach()`` before clearing ``_actor``.
    """

    # Runtime overrides (None means "fall back to style").
    _runtime_show_scalar_bar: Optional[bool]
    _runtime_fmt: Optional[str]

    def _init_scalar_bar_state(self) -> None:
        self._runtime_show_scalar_bar = None
        self._runtime_fmt = None

    # ------------------------------------------------------------------
    # Public live setters
    # ------------------------------------------------------------------

    def set_show_scalar_bar(self, show: bool) -> None:
        """Toggle the scalar bar's visibility live (no re-attach).

        ``False`` removes the bar from the plotter; ``True`` re-adds it
        against the active actor's mapper using the current ``fmt``.
        Idempotent and safe to call before / after attach.
        """
        show = bool(show)
        self._runtime_show_scalar_bar = show
        if not self._scalar_bar_is_enabled():
            # Diagram doesn't currently draw a bar (e.g. VectorGlyph
            # without magnitude colors). The runtime flag is recorded
            # for future re-attach, but there is nothing to mutate.
            return
        if getattr(self, "_actor", None) is None or self._plotter is None:
            return
        title = self._scalar_bar_title()
        if not show:
            self._remove_scalar_bar(title)
        else:
            self._ensure_scalar_bar(title)

    def set_fmt(self, fmt: str) -> None:
        """Update the bar's tick-label ``printf`` format string live."""
        fmt = str(fmt)
        self._runtime_fmt = fmt
        bar = self._scalar_bar_actor()
        if bar is None:
            return
        try:
            bar.SetLabelFormat(fmt)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hooks (override per diagram if its bar is conditional or its
    # title differs from the component name)
    # ------------------------------------------------------------------

    def _scalar_bar_is_enabled(self) -> bool:
        """Whether the diagram is currently drawing a scalar bar.

        Default: True. VectorGlyph overrides to gate on
        ``use_magnitude_colors``.
        """
        return True

    def _scalar_bar_title(self) -> str:
        """The plotter-registry key used for this diagram's bar."""
        return self.spec.selector.component

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scalar_bar_actor(self) -> Any:
        """Return the ``vtkScalarBarActor`` for this diagram, or ``None``."""
        plotter = getattr(self, "_plotter", None)
        if plotter is None:
            return None
        bars = getattr(plotter, "scalar_bars", None)
        if bars is None:
            return None
        try:
            return bars[self._scalar_bar_title()]
        except Exception:
            return None

    def _remove_scalar_bar(self, title: str) -> None:
        """Remove the bar for ``title`` from the plotter, idempotent."""
        plotter = getattr(self, "_plotter", None)
        if plotter is None:
            return
        try:
            plotter.remove_scalar_bar(title)
        except Exception:
            pass

    def _ensure_scalar_bar(self, title: str) -> None:
        """Re-attach a scalar bar against the active actor's mapper."""
        actor = getattr(self, "_actor", None)
        plotter = getattr(self, "_plotter", None)
        if actor is None or plotter is None:
            return
        try:
            mapper = actor.GetMapper()
        except Exception:
            return
        self._remove_scalar_bar(title)
        fmt = self._runtime_fmt or self._scalar_bar_default_fmt()
        try:
            plotter.add_scalar_bar(
                title=title,
                mapper=mapper,
                fmt=fmt,
                interactive=True,
            )
        except Exception:
            pass

    def _scalar_bar_default_fmt(self) -> str:
        """Fall back to the host style's ``fmt`` if it has one."""
        style = getattr(self, "spec", None)
        if style is None:
            return "%.3g"
        return getattr(style.style, "fmt", "%.3g") or "%.3g"

    # ------------------------------------------------------------------
    # Common scalar_bar_args builder
    # ------------------------------------------------------------------

    def _scalar_bar_args(self) -> Optional[dict]:
        """Return the dict to pass to ``add_mesh(scalar_bar_args=...)``.

        Returns ``None`` when the bar is suppressed (style or runtime
        override). The diagram should pair this with a matching
        ``show_scalar_bar`` boolean.
        """
        if not self._effective_show_scalar_bar():
            return None
        return {
            "title": self._scalar_bar_title(),
            "fmt": self._runtime_fmt or self._scalar_bar_default_fmt(),
            "interactive": True,
        }

    def _effective_show_scalar_bar(self) -> bool:
        """Resolve runtime override + style flag into a single boolean."""
        if self._runtime_show_scalar_bar is not None:
            return bool(self._runtime_show_scalar_bar)
        style = getattr(self, "spec", None)
        if style is None:
            return True
        return bool(getattr(style.style, "show_scalar_bar", True))
