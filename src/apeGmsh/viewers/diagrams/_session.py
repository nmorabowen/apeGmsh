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
from ._kinds import kind_ids, style_class_for
from ._selectors import SlabSelector


# Bumped to 4 in the cuts v2.2 viewer overlay: ``ViewerSession`` gained a
# ``model_h5`` field so a restored session can rebuild the
# ``FemToOpsTagMap`` needed by ``SectionCutDiagram`` layers. Bumped to 5
# for ADR 0058 S2b: ``GeometrySnapshot`` gained ``visible`` (concurrent
# rendering; absent = legacy, restored as "visible iff active"). Bumped
# to 6 for ADR 0058 S3a: ``GeometrySnapshot`` gained ``offset`` (per-
# geometry spatial offset; absent = legacy, restored as zero). Bumped
# to 7 for ADR 0058 S3b: ``GeometrySnapshot`` gained ``stage_id``
# (per-geometry stage pin; absent = legacy, restored as None = follow
# the active stage). The on-disk format stays forward/back compatible
# — missing fields read as defaults.
SESSION_SCHEMA_VERSION = 7


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
    """One geometry: deformation + display state + child compositions.

    The ``show_mesh / show_nodes / display_opacity`` triple was added
    in schema v3 to persist per-geometry substrate visibility. v2
    snapshots load with the v3 defaults (mesh + nodes on, full alpha).

    ``visible`` was added in schema v5 (ADR 0058 S2b — concurrent
    rendering). ``None`` marks a legacy session that predates the
    flag; the restore path maps it to "visible iff this geometry is
    the active one", reproducing the old active-only rendering.

    ``offset`` was added in schema v6 (ADR 0058 S3a — per-geometry
    spatial offset). Legacy sessions (no field) read ``(0, 0, 0)``.

    ``stage_id`` was added in schema v7 (ADR 0058 S3b — per-geometry
    stage pin). Legacy sessions (no field) read ``None`` = follow the
    active stage.
    """
    id: Optional[str]
    name: str
    deform_enabled: bool = False
    deform_field: Optional[str] = None
    deform_scale: float = 1.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stage_id: Optional[str] = None
    visible: Optional[bool] = None
    show_mesh: bool = True
    show_nodes: bool = True
    display_opacity: float = 1.0
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
    # Added in schema v4 (cuts v2.2 viewer overlay). Absolute path to
    # the ``model.h5`` the SectionCutDiagram layers were built against;
    # the restore path sets it on the director so the FemToOpsTagMap
    # can rebuild before cut layers attach. None for sessions that
    # don't carry any cuts.
    model_h5: Optional[str] = None


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
    style_cls = style_class_for(kind)
    if style_cls is None:
        raise KeyError(
            f"Unknown diagram kind {kind!r}. Known kinds: "
            f"{sorted(kind_ids())}."
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
    # ``SectionCutStyle.cut`` is a nested SectionCutDef dataclass —
    # rehydrate it from the dict that ``asdict`` produced. The
    # SectionCutDef constructor coerces tuple-like fields so we can
    # just hand it the dict's values directly.
    if kind == "section_cut":
        cut_raw = style_data.get("cut")
        if isinstance(cut_raw, dict):
            from apeGmsh.cuts import SectionCutDef
            style_data["cut"] = SectionCutDef(
                plane_point=cut_raw["plane_point"],
                plane_normal=cut_raw["plane_normal"],
                element_ids=cut_raw["element_ids"],
                side=cut_raw.get("side", "positive"),
                label=cut_raw.get("label"),
                bounding_polygon=cut_raw.get("bounding_polygon"),
            )
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
    model_h5: "Optional[str | Path]" = None,
) -> dict[str, Any]:
    """Build the JSON-friendly dict for one viewer session.

    ``geometries`` is the Geometry → Composition tree captured from
    the live ``GeometryManager``; compositions reference layers by
    their position in ``specs``. When ``None`` or empty we still emit
    a v2 envelope (the restore path falls back to a single Geometry).

    ``model_h5`` is the path the director was pointed at for the
    section-cut tag map. Only emitted when present.
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
        "model_h5":         str(model_h5) if model_h5 is not None else None,
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
        "offset":                [float(c) for c in g.offset],
        "stage_id":              g.stage_id,
        "visible":               None if g.visible is None else bool(g.visible),
        "show_mesh":             bool(g.show_mesh),
        "show_nodes":            bool(g.show_nodes),
        "display_opacity":       float(g.display_opacity),
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
    # v2 sessions don't carry display fields — the dataclass defaults
    # (mesh + nodes on, full opacity) match the historical global
    # behavior so old saves restore unchanged.
    # ``visible`` (schema v5, ADR 0058 S2b) stays None when absent so
    # the restore path can apply the legacy "visible iff active"
    # mapping instead of a blanket default.
    visible_raw = raw.get("visible")
    # ``offset`` (schema v6, ADR 0058 S3a) — legacy sessions carry no
    # key; anything malformed also degrades to the zero offset.
    try:
        offset = tuple(float(c) for c in (raw.get("offset") or ()))
    except (TypeError, ValueError):
        offset = ()
    if len(offset) != 3:
        offset = (0.0, 0.0, 0.0)
    # ``stage_id`` (schema v7, ADR 0058 S3b) — legacy sessions carry
    # no key; None = follow the active stage.
    stage_id_raw = raw.get("stage_id")
    return GeometrySnapshot(
        id=raw.get("id"),
        name=str(raw.get("name", "Geometry")),
        deform_enabled=bool(raw.get("deform_enabled", False)),
        deform_field=raw.get("deform_field"),
        deform_scale=float(raw.get("deform_scale", 1.0) or 1.0),
        offset=offset,
        stage_id=str(stage_id_raw) if stage_id_raw else None,
        visible=None if visible_raw is None else bool(visible_raw),
        show_mesh=bool(raw.get("show_mesh", True)),
        show_nodes=bool(raw.get("show_nodes", True)),
        display_opacity=float(raw.get("display_opacity", 1.0) or 1.0),
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
    model_h5_raw = data.get("model_h5")
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
        model_h5=str(model_h5_raw) if model_h5_raw else None,
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
    model_h5: "Optional[str | Path]" = None,
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
        model_h5=model_h5,
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
