"""Style presets — JSON save/load for ``DiagramStyle`` records.

Phase 6 closing item: the user can save a per-kind style snapshot
("Stress contour, viridis, [-100, 100] MPa") under a name and apply
it later in a fresh session. Presets live in
``<config>/apeGmsh/style_presets/<name>.json`` so they survive across
sessions and machines (when a config dir is synced).

The serializer handles only primitive fields plus ``tuple`` of
primitives — JSON has no tuple type, so tuples round-trip through
lists. Every shipped ``*Style`` dataclass uses primitive fields only,
so we don't need a richer codec.
"""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Optional

from ._styles import (
    ContourStyle,
    DeformedShapeStyle,
    DiagramStyle,
    FiberSectionStyle,
    GaussMarkerStyle,
    LayerStackStyle,
    LineForceStyle,
    SpringForceStyle,
    VectorGlyphStyle,
)

if TYPE_CHECKING:
    from pathlib import Path


# kind_id → style class. Kept in this module so the preset codec is
# self-contained; AddDiagramDialog and DiagramSettingsTab both read
# this map when serializing / deserializing.
KIND_TO_STYLE_CLASS: dict[str, type[DiagramStyle]] = {
    "contour":        ContourStyle,
    "deformed_shape": DeformedShapeStyle,
    "line_force":     LineForceStyle,
    "fiber_section":  FiberSectionStyle,
    "layer_stack":    LayerStackStyle,
    "vector_glyph":   VectorGlyphStyle,
    "gauss_marker":   GaussMarkerStyle,
    "spring_force":   SpringForceStyle,
}


# ======================================================================
# Serialization
# ======================================================================

def style_to_dict(style: DiagramStyle) -> dict:
    """Render ``style`` to a JSON-safe dict.

    Tuples are flattened to lists (JSON has no tuple); everything else
    must already be a primitive. Raises ``TypeError`` if the style
    contains a non-primitive field — keeps presets predictable.
    """
    if not is_dataclass(style):
        raise TypeError(f"style is not a dataclass: {style!r}")
    raw = asdict(style)
    return _to_jsonable(raw)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    raise TypeError(
        f"Style preset value of type {type(value).__name__!r} is not "
        f"JSON-safe: {value!r}"
    )


def style_from_dict(kind_id: str, data: dict) -> DiagramStyle:
    """Construct a style instance from ``kind_id`` and a JSON dict.

    Tuple fields are reconstructed from JSON arrays via the dataclass
    field's declared annotation. Unknown fields are ignored (forward
    compatibility — older presets must not break newer code with
    fewer fields).
    """
    cls = KIND_TO_STYLE_CLASS.get(kind_id)
    if cls is None:
        raise ValueError(f"Unknown kind_id: {kind_id!r}")
    valid_names = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for name, raw in data.items():
        if name not in valid_names:
            continue
        kwargs[name] = _coerce_field(cls, name, raw)
    return cls(**kwargs)


def _coerce_field(cls: type, name: str, raw: Any) -> Any:
    """Re-tuple a field if the dataclass declares a tuple annotation.

    JSON read-back gives lists; the dataclass annotation says
    ``tuple[...]`` for fields like ``clim`` / ``components``. ``Optional``
    annotations let ``None`` pass through unchanged.
    """
    if raw is None:
        return None
    annotations = getattr(cls, "__annotations__", {})
    ann = annotations.get(name, "")
    ann_str = str(ann)
    if "tuple" in ann_str.lower() and isinstance(raw, list):
        return tuple(raw)
    return raw


# ======================================================================
# Store
# ======================================================================

class StylePresetStore:
    """File-backed catalogue of named style presets.

    Defaults to ``<QSettings AppConfigLocation>/apeGmsh/style_presets/``,
    falling back to ``~/.config/apeGmsh/style_presets/`` if Qt is
    unavailable. Tests can pass an explicit ``directory`` to keep the
    user's real preset library out of the test fixture.
    """

    _filename_charset = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    )

    def __init__(self, directory: "Optional[Path]" = None) -> None:
        from pathlib import Path
        if directory is None:
            directory = self._default_dir()
        self._dir: Path = Path(directory)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Read-only filesystem — list / load still work, save fails.
            pass

    @property
    def directory(self) -> "Path":
        return self._dir

    @classmethod
    def _default_dir(cls) -> "Path":
        from pathlib import Path
        try:
            from qtpy.QtCore import QStandardPaths
            root = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppConfigLocation
            )
            if root:
                return Path(root) / "apeGmsh" / "style_presets"
        except Exception:
            pass
        return Path.home() / ".config" / "apeGmsh" / "style_presets"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(
        self, name: str, kind_id: str, style: DiagramStyle,
    ) -> "Path":
        """Write a preset to disk; returns the path written."""
        import json
        safe = self._sanitize_name(name)
        if not safe:
            raise ValueError(f"Invalid preset name: {name!r}")
        if kind_id not in KIND_TO_STYLE_CLASS:
            raise ValueError(f"Unknown kind_id: {kind_id!r}")
        payload = {
            "version": 1,
            "name": name,
            "kind_id": kind_id,
            "fields": style_to_dict(style),
        }
        path = self._dir / f"{safe}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def load(self, name: str) -> tuple[str, DiagramStyle]:
        """Read a preset by name. Returns ``(kind_id, style)``."""
        import json
        safe = self._sanitize_name(name)
        path = self._dir / f"{safe}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        kind_id = data.get("kind_id")
        fields_data = data.get("fields") or {}
        if not isinstance(kind_id, str):
            raise ValueError(f"Preset {name!r} missing kind_id")
        style = style_from_dict(kind_id, fields_data)
        return kind_id, style

    def list(self) -> list[tuple[str, str]]:
        """Return ``(name, kind_id)`` pairs for every parseable preset."""
        import json
        out: list[tuple[str, str]] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                name = data.get("name") or path.stem
                kind_id = data.get("kind_id")
                if isinstance(name, str) and isinstance(kind_id, str):
                    out.append((name, kind_id))
            except Exception:
                continue
        return out

    def list_for_kind(self, kind_id: str) -> list[str]:
        """Names of every preset matching ``kind_id``."""
        return [name for (name, kid) in self.list() if kid == kind_id]

    def delete(self, name: str) -> bool:
        """Remove a preset by name. Returns ``True`` if a file was deleted."""
        safe = self._sanitize_name(name)
        path = self._dir / f"{safe}.json"
        if path.exists():
            try:
                path.unlink()
                return True
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _sanitize_name(cls, name: str) -> str:
        """Reject path traversal / unprintable filename characters.

        Keeps a conservative whitelist; any character outside
        ``_filename_charset`` is dropped. Leading / trailing dots
        are then stripped so the result can't resolve to ``.`` or
        ``..`` (parent-directory escape). Empty result means the name
        is unusable as a filename and ``save`` will raise.
        """
        cleaned = "".join(ch for ch in name if ch in cls._filename_charset)
        cleaned = cleaned.strip(".")
        return cleaned


# Singleton — the dialog and the settings tab share one instance so
# saves in one are visible to the other without a reload.
_DEFAULT_STORE: Optional[StylePresetStore] = None


def default_store() -> StylePresetStore:
    """Lazy singleton accessor; tests can swap with :func:`reset_store`."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = StylePresetStore()
    return _DEFAULT_STORE


def reset_store(store: "Optional[StylePresetStore]" = None) -> None:
    """Replace (or clear) the singleton — for tests."""
    global _DEFAULT_STORE
    _DEFAULT_STORE = store
