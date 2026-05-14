"""LUTManager + LUT — shared lookup-table registry for viewer diagrams.

Inspired by ParaView's ``vtkSMTransferFunctionManager`` (`Remoting/Views/
vtkSMTransferFunctionManager.h`), with three deliberate simplifications:

* **Per-viewer instance, not process-global.** One ``LUTManager`` per
  ``ResultsViewer``. Avoids cross-window leakage and keeps tests
  parallelisable — same pattern as :class:`ActiveObjects`.
* **No transfer-function curves (v1).** The LUT carries a preset name,
  scalar range, log-scale flag, and a scalar-bar visibility flag. Color
  stops are derived on demand from the preset by sampling matplotlib's
  colormap. A future v2 may add user-editable stops + opacity curves;
  v1 ships preset-only for scope discipline (see ``docs/plans/
  06-color-map-editor.md`` risk section).
* **Per-array, not per-diagram.** ``manager.get_or_create("stress_vm")``
  returns the same LUT instance for repeat lookups, so two contour
  diagrams that *opt in* to LUT sharing can color consistently. The
  v1 viewer wiring keeps diagrams per-layer (each diagram has its own
  LUT), but the manager already supports the v2 sharing flow.

Lazy qtpy import — the module is safe to import in headless contexts
(tests that don't need Qt should not be forced to construct a
``QApplication``).
"""
from __future__ import annotations

from typing import Any, Optional


# Curated 10-preset palette. All names match matplotlib's registry, so
# ``matplotlib.colormaps[name]`` resolves without translation.
PRESETS: tuple[str, ...] = (
    "viridis", "plasma", "cividis", "magma", "inferno",
    "coolwarm", "RdBu", "Spectral", "turbo", "jet",
)


def is_preset(name: str) -> bool:
    """Whether ``name`` is one of the curated v1 presets."""
    return name in PRESETS


def sample_preset(name: str, n: int = 256):
    """Sample ``name`` at ``n`` evenly spaced positions in ``[0, 1]``.

    Returns an ``(n, 4)`` RGBA float array in ``[0, 1]``. Lazy
    matplotlib import — only callers that actually request stops pay
    the cost.

    Falls back to ``viridis`` if ``name`` is unknown.
    """
    import numpy as np

    try:
        import matplotlib
        cmap = matplotlib.colormaps[name if is_preset(name) else "viridis"]
    except Exception:
        # If matplotlib refuses (rare — name validated above), fall
        # back to a simple gray ramp so callers see *something*.
        ramp = np.linspace(0.0, 1.0, n)
        return np.stack([ramp, ramp, ramp, np.ones_like(ramp)], axis=1)
    xs = np.linspace(0.0, 1.0, n)
    return np.asarray(cmap(xs), dtype=np.float64)


# Late-imported Qt — keep this module importable in headless contexts.
def _build_classes():
    from qtpy import QtCore

    class LUT(QtCore.QObject):
        """A named lookup table referenced by one or more diagrams.

        State::

            array_name        # immutable identity (e.g. "stress_vm")
            preset            # matplotlib cmap name
            vmin, vmax        # scalar range
            log_scale         # apply log10 mapping
            show_scalar_bar   # render the bar overlay

        Every mutating setter emits :attr:`changed` *once* with no
        payload — diagrams that reference the LUT re-color on the
        signal, the color-map editor refreshes its widgets. Setters
        are no-ops when the new value equals the current one, so the
        editor can echo back its own writes without a feedback loop
        (still, callers binding to ``changed`` should guard against
        re-entrant writes by checking ``signalsBlocked()`` or via a
        ``_self_setting`` flag).
        """

        changed = QtCore.Signal()

        def __init__(
            self,
            array_name: str,
            preset: str = "viridis",
            vmin: float = 0.0,
            vmax: float = 1.0,
            *,
            log_scale: bool = False,
            show_scalar_bar: bool = True,
            parent: Any = None,
        ) -> None:
            super().__init__(parent)
            self._array_name = str(array_name)
            self._preset = preset if is_preset(preset) else "viridis"
            self._vmin = float(vmin)
            self._vmax = float(vmax) if vmax != vmin else float(vmin) + 1.0
            self._log_scale = bool(log_scale)
            self._show_scalar_bar = bool(show_scalar_bar)

        # ── Identity ────────────────────────────────────────────────
        @property
        def array_name(self) -> str:
            return self._array_name

        # ── Preset ──────────────────────────────────────────────────
        @property
        def preset(self) -> str:
            return self._preset

        def set_preset(self, name: str) -> None:
            """Switch to ``name``. Unknown presets are clamped to
            ``viridis`` (no exception — the editor's combo can't
            propose anything invalid, but external callers might)."""
            new = name if is_preset(name) else "viridis"
            if new == self._preset:
                return
            self._preset = new
            self.changed.emit()

        # ── Range ───────────────────────────────────────────────────
        @property
        def vmin(self) -> float:
            return self._vmin

        @property
        def vmax(self) -> float:
            return self._vmax

        @property
        def range(self) -> tuple[float, float]:
            return (self._vmin, self._vmax)

        def set_range(self, vmin: float, vmax: float) -> None:
            """Update the scalar range. Collapses ``vmin == vmax`` to
            a unit-width range so downstream LUT builders don't divide
            by zero."""
            lo = float(vmin)
            hi = float(vmax)
            if hi == lo:
                hi = lo + 1.0
            if lo == self._vmin and hi == self._vmax:
                return
            self._vmin = lo
            self._vmax = hi
            self.changed.emit()

        # ── Log scale ───────────────────────────────────────────────
        @property
        def log_scale(self) -> bool:
            return self._log_scale

        def set_log_scale(self, on: bool) -> None:
            new = bool(on)
            if new == self._log_scale:
                return
            self._log_scale = new
            self.changed.emit()

        # ── Scalar-bar visibility ──────────────────────────────────
        @property
        def show_scalar_bar(self) -> bool:
            return self._show_scalar_bar

        def set_show_scalar_bar(self, on: bool) -> None:
            new = bool(on)
            if new == self._show_scalar_bar:
                return
            self._show_scalar_bar = new
            self.changed.emit()

        # ── Derived ─────────────────────────────────────────────────
        def color_stops(
            self, n: int = 8,
        ) -> "list[tuple[float, tuple[float, float, float]]]":
            """Return ``n`` evenly-spaced (t, rgb) stops from the
            current preset. ``t`` is in ``[0, 1]``, ``rgb`` channels
            in ``[0, 1]``. Read-only preview for the editor — the
            preset is the source of truth."""
            samples = sample_preset(self._preset, n)
            ts = [i / max(n - 1, 1) for i in range(n)]
            return [
                (ts[i], (
                    float(samples[i, 0]),
                    float(samples[i, 1]),
                    float(samples[i, 2]),
                ))
                for i in range(n)
            ]

        def to_pyvista_lookup_table(self):
            """Build a ``pv.LookupTable`` matching the current state.

            The LUT's scalar range is set; ``log_scale=True`` flips
            the table's log flag so the mapper applies log10 mapping.
            Lazy pyvista import — headless tests don't need it.
            """
            import pyvista as pv
            table = pv.LookupTable(self._preset)
            table.scalar_range = (self._vmin, self._vmax)
            if self._log_scale:
                try:
                    table.log_scale = True
                except Exception:
                    # Older PyVista lacked the property; fall back to
                    # the underlying VTK attribute.
                    try:
                        table.SetScaleToLog10()
                    except Exception:
                        pass
            return table

        # ── Repr ────────────────────────────────────────────────────
        def __repr__(self) -> str:
            return (
                f"<LUT array={self._array_name!r} preset={self._preset!r} "
                f"range=({self._vmin:.3g}, {self._vmax:.3g}) "
                f"log={self._log_scale} bar={self._show_scalar_bar}>"
            )

    class LUTManager(QtCore.QObject):
        """Per-viewer registry of named lookup tables.

        Owns ``LUT`` instances keyed by array name. ``get_or_create``
        is the canonical lookup — diagrams call it at attach time and
        cache the returned reference for the diagram's lifetime.
        """

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self._luts: "dict[str, LUT]" = {}

        def get_or_create(
            self,
            array_name: str,
            *,
            preset: str = "viridis",
            vmin: float = 0.0,
            vmax: float = 1.0,
            log_scale: bool = False,
            show_scalar_bar: bool = True,
        ) -> "LUT":
            """Return the LUT for ``array_name``, creating one on miss.

            The ``preset`` / ``vmin`` / ``vmax`` / ``log_scale`` /
            ``show_scalar_bar`` arguments are *initial defaults* —
            applied only when the LUT doesn't yet exist. A second
            call with different defaults returns the existing LUT
            unchanged (callers wanting to override should mutate via
            the LUT's setters)."""
            existing = self._luts.get(array_name)
            if existing is not None:
                return existing
            lut = LUT(
                array_name,
                preset=preset,
                vmin=vmin,
                vmax=vmax,
                log_scale=log_scale,
                show_scalar_bar=show_scalar_bar,
                parent=self,
            )
            self._luts[array_name] = lut
            return lut

        def get(self, array_name: str) -> "Optional[LUT]":
            """Return the LUT for ``array_name`` or ``None``."""
            return self._luts.get(array_name)

        def all(self) -> "list[LUT]":
            """All registered LUTs in insertion order."""
            return list(self._luts.values())

        def remove(self, array_name: str) -> None:
            """Drop ``array_name`` from the registry. Idempotent.

            The dropped LUT is *not* destroyed — its ``QObject``
            parent stays this manager, so callers still holding a
            reference can keep using it (it just won't be returned by
            ``get`` anymore). Detach scenarios where the LUT is no
            longer referenced rely on Qt's parent-tracked GC at
            manager destruction time."""
            self._luts.pop(array_name, None)

        def __contains__(self, array_name: str) -> bool:
            return array_name in self._luts

        def __len__(self) -> int:
            return len(self._luts)

    return LUT, LUTManager


_classes: "Optional[tuple[type, type]]" = None


def __getattr__(name: str):
    """Lazy-construct ``LUT`` / ``LUTManager`` on first access.

    Mirrors :mod:`apeGmsh.viewers.core._active_objects` — avoids pulling
    qtpy at module import time so headless code paths stay clean.
    """
    global _classes
    if name in ("LUT", "LUTManager"):
        if _classes is None:
            _classes = _build_classes()
        return _classes[0] if name == "LUT" else _classes[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
