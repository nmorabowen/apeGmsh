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
* **Live layout** — ``set_scalar_bar_vertical(bool)`` /
  ``set_scalar_bar_scale(float)`` re-add the bar with the new
  orientation / on-screen size; both overrides survive show toggles
  and LUT changes via the shared ``_make_scalar_bar_spec`` builder.
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
    _runtime_bar_vertical: Optional[bool]
    _runtime_bar_scale: Optional[float]

    def _init_scalar_bar_state(self) -> None:
        self._runtime_show_scalar_bar = None
        self._runtime_fmt = None
        self._runtime_bar_vertical = None
        self._runtime_bar_scale = None

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
        # Render-seam path (ADR 0042): migrated diagrams hold a backend +
        # layer handle; route the bar through the backend keyed by
        # layer_id. Un-migrated diagrams fall through to the plotter.
        if self._uses_backend():
            if show:
                self._backend.add_scalar_bar(
                    self._handle, self._make_scalar_bar_spec(),
                )
            else:
                self._backend.remove_scalar_bar(self._handle.layer_id)
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
        if self._uses_backend():
            self._backend.set_scalar_bar_format(self._handle.layer_id, fmt)
            return
        bar = self._scalar_bar_actor()
        if bar is None:
            return
        try:
            bar.SetLabelFormat(fmt)
        except Exception:
            pass

    def set_scalar_bar_vertical(self, vertical: bool) -> None:
        """Flip the bar between vertical and horizontal live."""
        self._runtime_bar_vertical = bool(vertical)
        self._refresh_scalar_bar()

    def set_scalar_bar_scale(self, scale: float) -> None:
        """Scale the bar's on-screen size live (``1.0`` = default)."""
        self._runtime_bar_scale = max(0.05, float(scale))
        self._refresh_scalar_bar()

    def _refresh_scalar_bar(self) -> None:
        """Re-add the bar (when currently shown) so layout changes land."""
        if not self._scalar_bar_is_enabled():
            return
        if not self._effective_show_scalar_bar():
            return
        if self._uses_backend():
            self._backend.add_scalar_bar(
                self._handle, self._make_scalar_bar_spec(),
            )
            return
        if getattr(self, "_actor", None) is not None and self._plotter is not None:
            self._ensure_scalar_bar(self._scalar_bar_title())

    # ------------------------------------------------------------------
    # Render-seam helpers (ADR 0042, R-B)
    # ------------------------------------------------------------------

    def _make_scalar_bar_spec(self) -> Any:
        """The ``ScalarBarSpec`` for the diagram's current runtime state.

        The single builder every add/refresh path uses, so runtime
        overrides (fmt, orientation, scale) survive re-creation.
        """
        from ..scene_ir import ScalarBarSpec
        return ScalarBarSpec(
            layer_id=self._handle.layer_id,
            title=self._scalar_bar_title(),
            lut=self._current_lutspec(),
            fmt=self._runtime_fmt or self._scalar_bar_default_fmt(),
            vertical=self._effective_bar_vertical(),
            size=self._effective_bar_scale(),
        )

    def _uses_backend(self) -> bool:
        """True when the host is a migrated diagram (backend + handle)."""
        return (
            getattr(self, "_backend", None) is not None
            and getattr(self, "_handle", None) is not None
        )

    def _current_lutspec(self) -> Any:
        """A plain ``LutSpec`` snapshot of the host's LUT mirror (if any).

        The Qt LUT mirror stays diagram-side; this is the plain-spec
        translation that crosses the seam.
        """
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
        """The plotter-registry key used for this diagram's bar.

        ADR 0058 S2b — when the registry stamped a
        ``_bar_prefix_resolver`` on the diagram (concurrent
        geometries) and it resolves to a name, the title is prefixed
        ``"<geometry> — <component>"``. The resolver returns ``None``
        while at most one geometry is visible, keeping
        single-geometry titles unchanged. Computed on demand, so add
        and remove agree on the key; a bar created while only one
        geometry was visible keeps its unprefixed title until
        re-created (accepted refresh-on-recreate behavior).
        """
        base = self.spec.selector.component
        resolver = getattr(self, "_bar_prefix_resolver", None)
        if resolver is not None:
            try:
                prefix = resolver(self)
            except Exception:
                prefix = None
            if prefix:
                return f"{prefix} — {base}"
        return base

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
        """Remove the bar from the plotter, idempotent.

        Routes through the backend (keyed by layer_id) for migrated
        diagrams; otherwise removes by title from the plotter.
        """
        if self._uses_backend():
            try:
                self._backend.remove_scalar_bar(self._handle.layer_id)
            except Exception:
                pass
            return
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
                **self._legacy_layout_kwargs(),
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
            **self._legacy_layout_kwargs(),
        }

    def _effective_show_scalar_bar(self) -> bool:
        """Resolve runtime override + style flag into a single boolean."""
        if self._runtime_show_scalar_bar is not None:
            return bool(self._runtime_show_scalar_bar)
        style = getattr(self, "spec", None)
        if style is None:
            return True
        return bool(getattr(style.style, "show_scalar_bar", True))

    def _effective_bar_vertical(self) -> Optional[bool]:
        """Runtime orientation override, else the style's, else ``None``
        (= backend theme default)."""
        if self._runtime_bar_vertical is not None:
            return bool(self._runtime_bar_vertical)
        style = getattr(self, "spec", None)
        if style is None:
            return None
        vertical = getattr(style.style, "scalar_bar_vertical", None)
        return None if vertical is None else bool(vertical)

    def _effective_bar_scale(self) -> float:
        """Runtime size-scale override, else the style's, else ``1.0``."""
        if self._runtime_bar_scale is not None:
            return float(self._runtime_bar_scale)
        style = getattr(self, "spec", None)
        if style is None:
            return 1.0
        return float(getattr(style.style, "scalar_bar_scale", 1.0) or 1.0)

    def _legacy_layout_kwargs(self) -> dict:
        """Orientation/size kwargs for the direct-plotter fallback path.

        Mirrors the backend's ``_scalar_bar_layout`` with pyvista's
        stock theme dimensions; empty when nothing is overridden so
        un-touched bars keep their exact historical placement.
        """
        vertical = self._effective_bar_vertical()
        scale = self._effective_bar_scale()
        kwargs: dict = {}
        if vertical is not None:
            kwargs["vertical"] = bool(vertical)
        if scale != 1.0:
            w0, h0 = (0.08, 0.45) if vertical else (0.6, 0.08)
            kwargs["width"] = min(w0 * scale, 0.95)
            kwargs["height"] = min(h0 * scale, 0.95)
        return kwargs
