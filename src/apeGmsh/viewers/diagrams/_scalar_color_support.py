"""ScalarColorSupport — shared LUT / clim / cmap state for diagrams.

Every scalar-coloured diagram (Contour, VectorGlyph, GaussPoint,
FiberSection, LayerStack) carries the same runtime colour state and
the same Qt LUT mirror, previously copy-pasted five times:

* **Runtime overrides** — ``_runtime_clim`` / ``_runtime_cmap``
  (None = fall back to the style), plus the attach-time
  ``_initial_clim`` autofit.
* **Live setters** — ``set_clim`` / ``set_cmap`` route through the
  LUT when attached (so the ColorMapEditor stays in sync) and park
  in the runtime overrides otherwise.
* **The LUT mirror** — ``_init_lut`` builds the Qt
  :class:`~apeGmsh.viewers.core._lut_manager.LUT` the ColorMapEditor
  binds to; ``_on_lut_changed`` mirrors mutations back into the
  runtime overrides and pushes a plain ``ColorSpec`` (+ scalar-bar
  refresh) through the backend; ``_teardown_lut`` is the leak-free
  detach half.

Extends :class:`ScalarBarSupport` (the scalar-bar lifecycle half) —
hosts inherit this one mixin and get both. Hosts must call
``self._init_scalar_color_state()`` from ``__init__`` and
``self._teardown_lut()`` from ``detach()``, and override
``_scalar_values_for_autofit`` (the one genuinely per-diagram piece:
which array feeds ``autofit_clim_at_current_step``).

One deliberate unification: ``_on_lut_changed``'s scalar-bar refresh
passes ``fmt`` (runtime override or the style default) — previously
only the contour did, so a ``set_fmt`` on the other diagrams was
lost on the next colormap change while ``set_show_scalar_bar``
(ScalarBarSupport) preserved it.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy import ndarray

from ._scalar_bar_support import ScalarBarSupport


class ScalarColorSupport(ScalarBarSupport):
    """Mixin: runtime colour state + LUT mirror for scalar diagrams.

    The host must be a :class:`~apeGmsh.viewers.diagrams._base.Diagram`
    — the methods rely on ``self.spec`` / ``self._backend`` /
    ``self._handle`` being managed by the diagram lifecycle.
    """

    # Host (Diagram) attributes the mixin reads — declared for typing.
    spec: Any
    _handle: Any
    _backend: Any

    # Runtime overrides (None means "fall back to style") + LUT mirror.
    _runtime_clim: Optional[tuple[float, float]]
    _runtime_cmap: Optional[str]
    _initial_clim: Optional[tuple[float, float]]
    _lut: Any
    _lut_conn: Any

    def _init_scalar_color_state(self) -> None:
        self._init_scalar_bar_state()
        self._runtime_clim = None
        self._runtime_cmap = None
        self._initial_clim = None
        # Diagram-side LUT mirror (Qt); changes pushed through the backend.
        self._lut = None
        self._lut_conn = None

    # ------------------------------------------------------------------
    # Public runtime setters (used by the settings tab / ColorMapEditor)
    # ------------------------------------------------------------------

    @property
    def lut(self) -> Any:
        """The shared lookup-table mirror for this diagram.

        ``None`` before :meth:`attach` and after :meth:`detach` (and
        for diagrams that aren't currently colouring by array). The
        ColorMapEditor reads this to bind its widgets and writes to it
        via the LUT's setters; mutations fire ``LUT.changed`` which the
        diagram translates into a plain ``ColorSpec`` and pushes
        through the backend.
        """
        return self._lut

    def set_clim(self, vmin: float, vmax: float) -> None:
        """Override the colormap range. Live update via the LUT."""
        if vmin == vmax:
            vmax = vmin + 1.0
        if self._lut is not None:
            self._lut.set_range(float(vmin), float(vmax))
            return
        self._runtime_clim = (float(vmin), float(vmax))

    def set_cmap(self, cmap: str) -> None:
        """Switch the colormap. Routes through the LUT when attached."""
        self._runtime_cmap = cmap
        if self._lut is not None:
            self._lut.set_preset(cmap)

    def current_clim(self) -> Optional[tuple[float, float]]:
        return self._runtime_clim or self._initial_clim

    def autofit_clim_at_current_step(self) -> Optional[tuple[float, float]]:
        """Re-fit clim to the current step's value range."""
        values = self._scalar_values_for_autofit()
        if values is None:
            return None
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        lo, hi = float(finite.min()), float(finite.max())
        if lo == hi:
            hi = lo + 1.0
        self.set_clim(lo, hi)
        return (lo, hi)

    # ------------------------------------------------------------------
    # Hooks (override per diagram)
    # ------------------------------------------------------------------

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        """The current step's scalar array, or ``None`` when absent.

        The one per-diagram piece of the autofit: contour feeds its
        substrate scalars, gauss its GP values, fiber its fiber
        values, layer-stack its aggregated cell values, vector-glyph
        its magnitude colours.
        """
        return None

    def _color_array_name(self) -> str:
        """The ColorSpec array name (hosts override with their fallback)."""
        return self.spec.selector.component

    # ------------------------------------------------------------------
    # LUT mirror (diagram-side; changes pushed through the backend)
    # ------------------------------------------------------------------

    def _init_lut(self) -> None:
        """Build the LUT mirror that the ColorMapEditor binds to."""
        if self._handle is None:
            return
        from ..core._lut_manager import LUT

        preset = (
            self._runtime_cmap
            or getattr(self.spec.style, "cmap", None)
            or "viridis"
        )
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        try:
            self._lut = LUT(
                array_name=self.spec.selector.component,
                preset=preset,
                vmin=float(clim[0]),
                vmax=float(clim[1]),
                show_scalar_bar=self._effective_show_scalar_bar(),
            )
            self._lut_conn = self._lut.changed.connect(self._on_lut_changed)
        except Exception:
            self._lut = None
            self._lut_conn = None

    def _on_lut_changed(self) -> None:
        """LUT mutated — mirror into runtime overrides and re-apply via
        the backend."""
        from ..scene_ir import ColorSpec

        if self._lut is None or self._handle is None or self._backend is None:
            return
        self._runtime_cmap = self._lut.preset
        self._runtime_clim = (self._lut.vmin, self._lut.vmax)
        color = ColorSpec(
            mode="by_array",
            array_name=self._color_array_name(),
            lut=self._current_lutspec(),
        )
        self._backend.set_layer_color(self._handle, color)
        # Refresh the bar so it reflects the new LUT.
        if self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle, self._make_scalar_bar_spec(),
            )

    def _teardown_lut(self) -> None:
        """Disconnect + drop the LUT mirror (call from ``detach`` FIRST,
        so a teardown-triggered ``changed.emit`` doesn't poke a
        half-dismantled layer)."""
        if self._lut is not None and self._lut_conn is not None:
            try:
                self._lut.changed.disconnect(self._lut_conn)
            except (TypeError, RuntimeError):
                pass
        self._lut = None
        self._lut_conn = None
