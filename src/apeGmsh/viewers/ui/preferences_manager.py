"""
PreferencesManager — persistent user preferences for viewer defaults.

A singleton observable that mirrors ``ThemeManager``'s pattern. Reads
and writes a single JSON file at a platform-appropriate config path.

The JSON file is *for* preferences the user has explicitly changed;
defaults live in code (``DEFAULT_PREFERENCES``). Missing keys fall back
to defaults, so adding a new field to a future schema does not break
older config files.

Theme selection is intentionally NOT persisted here — it lives in
QSettings (``ThemeManager``) so on-the-fly theme switches stay fast and
don't require a JSON write.

Usage::

    from apeGmsh.viewers.ui.preferences_manager import PREFERENCES
    size = PREFERENCES.current.point_size
    PREFERENCES.subscribe(lambda p: print("new prefs:", p))
    PREFERENCES.update({"point_size": 8.0})
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable


# ======================================================================
# Preferences dataclass
# ======================================================================

@dataclass(frozen=True)
class Preferences:
    """Persistent viewer defaults.

    Applied to *new* viewer invocations. Already-open viewers keep
    their session state; close and reopen to pick up changes.
    """

    # ── BRep viewer defaults ────────────────────────────────────────
    point_size: float = 10.0              # BRep point glyph size
    line_width: float = 6.0               # BRep curve line width
    surface_opacity: float = 0.35         # BRep surface/volume opacity
    show_surface_edges: bool = False      # Mesh-edge overlay on BRep

    # ── Mesh viewer defaults ────────────────────────────────────────
    node_marker_size: float = 6.0
    mesh_line_width: float = 3.0
    mesh_surface_opacity: float = 1.0
    mesh_show_surface_edges: bool = True

    # ── Rendering ───────────────────────────────────────────────────
    smooth_shading: bool = False          # False = flat matte (CAD look)
    # One of: "none", "fxaa", "ssaa", "msaa"
    anti_aliasing: str = "ssaa"

    # ── Outlines ────────────────────────────────────────────────────
    feature_angle: float = 25.0           # Silhouette dihedral-angle threshold

    # ── Label font sizes (px) ───────────────────────────────────────
    node_label_font_size: int = 8
    element_label_font_size: int = 8
    entity_label_font_size: int = 10
    origin_marker_font_size: int = 10
    coord_precision: int = 2              # Decimal places in coord labels

    # ── Axis widget ─────────────────────────────────────────────────
    axis_line_width: float = 2.0
    axis_labels_visible: bool = True

    # ── Origin-marker overlay ───────────────────────────────────────
    origin_marker_size: float = 10.0
    origin_marker_show_coords: bool = True
    origin_marker_include_world_origin: bool = True

    # ── Geometry probes (tangent / normal arrows) ───────────────────
    tangent_normal_scale: float = 0.05    # Fraction of model diagonal

    # ── Interaction & UI ────────────────────────────────────────────
    drag_threshold: int = 8               # Pixels moved before click → drag
    # One of: "left", "top", "right", "bottom"
    tab_position: str = "left"
    dock_min_width: int = 320
    window_maximized: bool = True
    show_console: bool = False


DEFAULT_PREFERENCES = Preferences()

# Accepted values for Literal-style string fields. Used by the dialog's
# combo boxes and, softly, by _apply_overrides (unknown values are kept —
# we trust callers and let Qt validate on the UI side).
ANTI_ALIASING_CHOICES = ("none", "fxaa", "ssaa", "msaa")
TAB_POSITION_CHOICES = ("left", "top", "right", "bottom")


# ======================================================================
# Storage helpers
# ======================================================================

def _config_path() -> Path:
    """Platform-appropriate path for preferences.json.

    Uses ``QStandardPaths.AppConfigLocation`` when Qt is available
    (correct on Windows/macOS/Linux). Falls back to ``~/.config/apeGmsh/``
    in headless / Qt-less environments so tests can still run.
    """
    try:
        from qtpy.QtCore import QStandardPaths
        root = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppConfigLocation
        )
        if root:
            return Path(root) / "apeGmsh" / "preferences.json"
    except Exception:
        pass
    return Path.home() / ".config" / "apeGmsh" / "preferences.json"


def _apply_overrides(base: Preferences, data: dict) -> Preferences:
    """Return ``base`` with keys from ``data`` applied (unknown keys ignored)."""
    valid = {f.name for f in fields(Preferences)}
    kept = {k: v for k, v in data.items() if k in valid}
    return replace(base, **kept)


# ======================================================================
# Manager
# ======================================================================

class PreferencesManager:
    """Global current preferences + observer list.

    Observers fire on every successful ``update()`` / ``reset()`` with
    the new ``Preferences`` value.
    """

    def __init__(self, path: Path | None = None) -> None:
        # ``path`` is mainly for tests; production code constructs the
        # singleton without arguments and picks up the platform config
        # path from ``_config_path()``.
        self._path: Path = Path(path) if path is not None else _config_path()
        self._current: Preferences = self._load() or DEFAULT_PREFERENCES
        self._observers: list[Callable[[Preferences], None]] = []

    @property
    def current(self) -> Preferences:
        return self._current

    @property
    def path(self) -> Path:
        """Location of the JSON file (may not exist yet)."""
        return self._path

    # ── mutations ────────────────────────────────────────────────────

    def update(self, overrides: dict[str, Any]) -> Preferences:
        """Apply ``overrides`` (dict) and persist to JSON."""
        new = _apply_overrides(self._current, overrides)
        if new == self._current:
            return new
        self._current = new
        self._save(new)
        self._notify(new)
        return new

    def reset(self) -> Preferences:
        """Delete the JSON file and revert to defaults."""
        self._current = DEFAULT_PREFERENCES
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            pass
        self._notify(self._current)
        return self._current

    def subscribe(
        self, cb: Callable[[Preferences], None]
    ) -> Callable[[], None]:
        """Register observer; returns an unsubscribe callable."""
        self._observers.append(cb)

        def _unsub() -> None:
            try:
                self._observers.remove(cb)
            except ValueError:
                pass

        return _unsub

    # ── internal ─────────────────────────────────────────────────────

    def _notify(self, prefs: Preferences) -> None:
        for cb in list(self._observers):
            try:
                cb(prefs)
            except Exception:
                import logging
                logging.getLogger("apeGmsh.viewer.preferences").exception(
                    "preferences observer failed: %r", cb,
                )

    def _load(self) -> Preferences | None:
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            return _apply_overrides(DEFAULT_PREFERENCES, data)
        except Exception:
            import logging
            logging.getLogger("apeGmsh.viewer.preferences").exception(
                "failed to load preferences from %s", self._path,
            )
            return None

    def _save(self, prefs: Preferences) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(asdict(prefs), indent=2, sort_keys=True)
            self._path.write_text(payload, encoding="utf-8")
        except Exception:
            import logging
            logging.getLogger("apeGmsh.viewer.preferences").exception(
                "failed to save preferences to %s", self._path,
            )


PREFERENCES = PreferencesManager()
