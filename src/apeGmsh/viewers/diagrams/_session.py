"""Session persistence for the post-solve viewer.

Save the active set of ``DiagramSpec`` records (plus active stage / step)
when the user closes a viewer and offer to restore them next time the
same Results file is opened. The serialized form is plain JSON next to
the Results file:

    <results-file>.viewer-session.json

Style subclasses (``ContourStyle``, ``LineForceStyle`` etc.) are
discriminated by ``DiagramSpec.kind`` — same convention the Add Diagram
dialog uses. The session record carries a copy of
``fem.snapshot_id`` so a later open against a re-meshed model can warn
and refuse to restore stale specs.

Public surface::

    serialize_spec(spec)              -> dict
    deserialize_spec(data)            -> DiagramSpec
    serialize_session(specs, ...)     -> dict
    deserialize_session(data)         -> ViewerSession
    save_session(...)                 -> Path
    load_session(path)                -> ViewerSession
    default_session_path(results_path) -> Path
"""
from __future__ import annotations

import dataclasses
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ._base import DiagramSpec
from ._selectors import SlabSelector
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


# Discriminator: kind_id -> Style class. Mirrors the dialog's _KINDS
# without depending on it (no Qt import).
_KIND_TO_STYLE: dict[str, type[DiagramStyle]] = {
    "contour":        ContourStyle,
    "deformed_shape": DeformedShapeStyle,
    "vector_glyph":   VectorGlyphStyle,
    "line_force":     LineForceStyle,
    "fiber_section":  FiberSectionStyle,
    "layer_stack":    LayerStackStyle,
    "gauss_marker":   GaussMarkerStyle,
    "spring_force":   SpringForceStyle,
}


SESSION_SCHEMA_VERSION = 2


# =====================================================================
# Records
# =====================================================================

@dataclass(frozen=True)
class CompositionSnapshot:
    """One composition: name + layer-index references."""
    id: Optional[str]
    name: str
    layer_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class GeometrySnapshot:
    """One geometry: deformation state + child compositions."""
    id: Optional[str]
    name: str
    deform_enabled: bool = False
    deform_field: Optional[str] = None
    deform_scale: float = 1.0
    active_composition_id: Optional[str] = None
    compositions: tuple[CompositionSnapshot, ...] = ()


@dataclass(frozen=True)
class ViewerSession:
    """Persisted viewer state for one Results file.

    Attributes
    ----------
    schema_version
        Bumps when the on-disk shape changes incompatibly.
    results_path
        Absolute path to the Results file the session was saved against.
    fem_snapshot_id
        ``fem.snapshot_id`` at save time. Stored as metadata; no longer
        enforced on restore.
    saved_at
        ISO-8601 timestamp.
    diagrams
        Tuple of ``DiagramSpec`` records (flat list across every
        geometry / composition). Compositions reference them by index.
    geometries
        Tuple of :class:`GeometrySnapshot` describing the
        Geometry → Composition → Layer hierarchy. Empty for legacy
        (v1) sessions, in which case all diagrams load into a single
        "Restored" composition under the active Geometry.
    active_geometry_id
        UUID of the geometry that was active at save time, or None.
    active_stage_id
    active_step
    """
    schema_version: int
    results_path: str
    fem_snapshot_id: Optional[str]
    saved_at: str
    diagrams: tuple[DiagramSpec, ...]
    geometries: tuple[GeometrySnapshot, ...] = ()
    active_geometry_id: Optional[str] = None
    active_stage_id: Optional[str] = None
    active_step: int = 0


# =====================================================================
# DiagramSpec ↔ dict
# =====================================================================

def serialize_spec(spec: DiagramSpec) -> dict[str, Any]:
    """Convert a :class:`DiagramSpec` to a JSON-friendly dict."""
    return {
        "kind":     spec.kind,
        "selector": dataclasses.asdict(spec.selector),
        "style":    dataclasses.asdict(spec.style),
        "stage_id": spec.stage_id,
        "visible":  spec.visible,
        "label":    spec.label,
    }


def deserialize_spec(data: dict[str, Any]) -> DiagramSpec:
    """Reconstruct a :class:`DiagramSpec` from :func:`serialize_spec`'s output.

    Raises
    ------
    KeyError
        If ``data["kind"]`` doesn't map to a known Style class. Callers
        should catch and surface this so the user knows which spec was
        skipped.
    """
    kind = data["kind"]
    style_cls = _KIND_TO_STYLE.get(kind)
    if style_cls is None:
        raise KeyError(
            f"Unknown diagram kind {kind!r}. Known kinds: "
            f"{sorted(_KIND_TO_STYLE)}."
        )

    selector_data = dict(data.get("selector") or {})
    # Tuples come back as lists from JSON — normalize.
    for key in ("pg", "label", "selection", "ids"):
        v = selector_data.get(key)
        if isinstance(v, list):
            selector_data[key] = tuple(v)
    selector = SlabSelector(**selector_data)

    style_data = dict(data.get("style") or {})
    # Some style fields (components, clim) are tuples; coerce lists back.
    for key, value in list(style_data.items()):
        if isinstance(value, list):
            style_data[key] = tuple(value)
    style = style_cls(**style_data)

    return DiagramSpec(
        kind=kind,
        selector=selector,
        style=style,
        stage_id=data.get("stage_id"),
        visible=bool(data.get("visible", True)),
        label=data.get("label"),
    )


# =====================================================================
# Session ↔ dict
# =====================================================================

def serialize_session(
    *,
    specs: "list[DiagramSpec] | tuple[DiagramSpec, ...]",
    results_path: str | Path,
    fem_snapshot_id: Optional[str],
    geometries: "list[GeometrySnapshot] | tuple[GeometrySnapshot, ...] | None" = None,
    active_geometry_id: Optional[str] = None,
    active_stage_id: Optional[str] = None,
    active_step: int = 0,
) -> dict[str, Any]:
    """Build the JSON-friendly dict for one viewer session.

    ``geometries`` is the Geometry → Composition tree captured from
    the live ``GeometryManager``; compositions reference layers by
    their position in ``specs``. When ``None`` or empty we still emit
    a v2 envelope (the restore path falls back to a single Geometry).
    """
    return {
        "schema_version":   SESSION_SCHEMA_VERSION,
        "results_path":     str(Path(results_path).resolve()),
        "fem_snapshot_id":  fem_snapshot_id,
        "saved_at":         datetime.datetime.now(
            datetime.timezone.utc,
        ).isoformat(),
        "active_geometry_id": active_geometry_id,
        "active_stage_id":  active_stage_id,
        "active_step":      int(active_step),
        "geometries":       [
            _serialize_geometry(g) for g in (geometries or ())
        ],
        "diagrams":         [serialize_spec(s) for s in specs],
    }


def _serialize_geometry(g: "GeometrySnapshot") -> dict[str, Any]:
    return {
        "id":    g.id,
        "name":  g.name,
        "deform_enabled":        bool(g.deform_enabled),
        "deform_field":          g.deform_field,
        "deform_scale":          float(g.deform_scale),
        "active_composition_id": g.active_composition_id,
        "compositions": [
            {
                "id":             c.id,
                "name":           c.name,
                "layer_indices":  list(c.layer_indices),
            }
            for c in g.compositions
        ],
    }


def _deserialize_geometry(raw: dict[str, Any]) -> GeometrySnapshot:
    comps: list[CompositionSnapshot] = []
    for craw in raw.get("compositions") or []:
        try:
            comps.append(CompositionSnapshot(
                id=craw.get("id"),
                name=str(craw.get("name", "Diagram")),
                layer_indices=tuple(
                    int(i) for i in (craw.get("layer_indices") or [])
                ),
            ))
        except Exception:
            continue
    return GeometrySnapshot(
        id=raw.get("id"),
        name=str(raw.get("name", "Geometry")),
        deform_enabled=bool(raw.get("deform_enabled", False)),
        deform_field=raw.get("deform_field"),
        deform_scale=float(raw.get("deform_scale", 1.0) or 1.0),
        active_composition_id=raw.get("active_composition_id"),
        compositions=tuple(comps),
    )


def deserialize_session(data: dict[str, Any]) -> ViewerSession:
    """Reconstruct a :class:`ViewerSession` from :func:`serialize_session`'s output.

    Diagram specs that fail to deserialize (unknown kind, bad fields)
    are skipped; the resulting session simply contains fewer specs.
    Legacy v1 sessions (no ``geometries`` block) deserialize with an
    empty geometries tuple — :class:`ResultsViewer._apply_session`
    bundles them into one "Restored" composition for back-compat.
    """
    diagrams: list[DiagramSpec] = []
    for raw in data.get("diagrams") or []:
        try:
            diagrams.append(deserialize_spec(raw))
        except Exception:
            continue
    geometries: list[GeometrySnapshot] = []
    for raw in data.get("geometries") or []:
        try:
            geometries.append(_deserialize_geometry(raw))
        except Exception:
            continue
    return ViewerSession(
        schema_version=int(
            data.get("schema_version", SESSION_SCHEMA_VERSION),
        ),
        results_path=str(data.get("results_path", "")),
        fem_snapshot_id=data.get("fem_snapshot_id"),
        saved_at=str(data.get("saved_at", "")),
        diagrams=tuple(diagrams),
        geometries=tuple(geometries),
        active_geometry_id=data.get("active_geometry_id"),
        active_stage_id=data.get("active_stage_id"),
        active_step=int(data.get("active_step", 0) or 0),
    )


# =====================================================================
# Disk I/O
# =====================================================================

def default_session_path(results_path: str | Path) -> Path:
    """Convention: ``<results>.viewer-session.json`` next to the file."""
    p = Path(results_path)
    return p.with_suffix(p.suffix + ".viewer-session.json")


def save_session(
    *,
    specs: "list[DiagramSpec] | tuple[DiagramSpec, ...]",
    results_path: str | Path,
    fem_snapshot_id: Optional[str],
    geometries: "list[GeometrySnapshot] | tuple[GeometrySnapshot, ...] | None" = None,
    active_geometry_id: Optional[str] = None,
    target_path: str | Path | None = None,
    active_stage_id: Optional[str] = None,
    active_step: int = 0,
) -> Path:
    """Write a session JSON next to (or at) the given path.

    Returns the path actually written.
    """
    payload = serialize_session(
        specs=specs,
        results_path=results_path,
        fem_snapshot_id=fem_snapshot_id,
        geometries=geometries,
        active_geometry_id=active_geometry_id,
        active_stage_id=active_stage_id,
        active_step=active_step,
    )
    out = Path(target_path) if target_path else default_session_path(
        results_path,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def load_session(path: str | Path) -> ViewerSession:
    """Load and deserialize a session JSON from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return deserialize_session(raw)


__all__ = [
    "SESSION_SCHEMA_VERSION",
    "CompositionSnapshot",
    "GeometrySnapshot",
    "ViewerSession",
    "default_session_path",
    "deserialize_session",
    "deserialize_spec",
    "load_session",
    "save_session",
    "serialize_session",
    "serialize_spec",
]
