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

from .._log import log_action
from ._base import DiagramSpec
from ._kinds import kind_ids, style_class_for
from ._selectors import SlabSelector


# Diagram kinds removed from the live registry but still recognized on
# load so a legacy session that carries one drops it with a clear log
# line instead of a bare "unknown kind". Maps the retired kind id to
# the migration hint shown to the user.
_RETIRED_KINDS: dict[str, str] = {
    # ADR 0058 S4 — DeformedShapeDiagram retired: deformation is now
    # per-geometry state (a deform-on geometry), and the undeformed
    # reference is the S3c `add_reference_ghost` preset.
    "deformed_shape":
        "ADR 0058 S4 — deformation is now per-geometry "
        "(enable Deform on a geometry); the undeformed reference is "
        "the 'Add reference ghost' preset.",
}


# Bumped to 4 in the cuts v2.2 viewer overlay: ``ViewerSession`` gained a
# ``model_h5`` field so a restored session can rebuild the
# ``FemToOpsTagMap`` needed by ``SectionCutDiagram`` layers. Bumped to 5
# for ADR 0058 S2b: ``GeometrySnapshot`` gained ``visible`` (concurrent
# rendering; absent = legacy, restored as "visible iff active"). Bumped
# to 6 for ADR 0058 S3a: ``GeometrySnapshot`` gained ``offset`` (per-
# geometry spatial offset; absent = legacy, restored as zero). Bumped
# to 7 for ADR 0058 S3b: ``GeometrySnapshot`` gained ``stage_id``
# (per-geometry stage pin; absent = legacy, restored as None = follow
# the active stage). Bumped to 8 for ADR 0081 L3: sessions carry a
# ``legends`` block so a colour scale the user placed, resized or hid
# comes back where they left it (absent = legacy, restored as "every
# legend docked at its default"). Bumped to 9 for ADR 0083 S1: sessions
# carry a ``clip_planes`` block plus the two set-level toggles, so the
# section planes the user cut with come back (absent = legacy, restored
# as "no planes"). The on-disk format stays forward/back compatible —
# missing fields read as defaults.
SESSION_SCHEMA_VERSION = 9


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
class LegendSnapshot:
    """One colour scale's placement (ADR 0081 L3, schema v8).

    Keyed by ``(geometry, component)`` — the ``LegendController`` entry
    key — because that is what survives a re-attach; layer ids do not.
    ``slot`` is ``None`` for a legend the user placed by hand, in which
    case ``anchor`` is authoritative; a docked legend restores to its
    slot and lets the layout place it, so its saved anchor is advisory
    only and a window resized between sessions still lays out correctly.
    """
    geometry: str
    component: str
    vertical: bool = True
    visible: bool = True
    fmt: str = "%.3g"
    font_scale: Optional[float] = None
    slot: Optional[int] = 0
    anchor: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ClipPlaneSnapshot:
    """One section plane (ADR 0083 Part 5, schema v9).

    A straight record of the :class:`ClipPlane` the controller owns —
    planes are world-space and belong to no diagram, so unlike a legend
    there is nothing to key them to and nothing to wait for on restore.
    """
    plane_id: str
    name: str
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0)
    offset: float = 0.0
    active: bool = True
    flipped: bool = False
    gizmo_visible: bool = True


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
    # Added in schema v8 (ADR 0081 L3). Colour-scale placement, keyed by
    # legend key rather than by layer so it survives a re-attach.
    legends: tuple["LegendSnapshot", ...] = ()
    # Added in schema v9 (ADR 0083 S1). The section-plane set plus its
    # two display toggles. A missing block reads as "no planes", so
    # every older session loads clean.
    clip_planes: tuple["ClipPlaneSnapshot", ...] = ()
    clip_apply_cuts: bool = True
    clip_show_gizmos: bool = True


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
        If ``data["kind"]`` doesn't map to a known Style class —
        including a recognized-but-retired kind (see
        :data:`_RETIRED_KINDS`). Callers should catch and surface this
        so the user knows which spec was skipped;
        :func:`deserialize_session` already does (catch-and-skip).
    """
    kind = data["kind"]
    if kind in _RETIRED_KINDS:
        # Recognized-but-retired (ADR 0058 S4): log a clear migration
        # line, then raise so the existing catch-and-skip in
        # deserialize_session drops just this spec — the rest of the
        # session's hierarchy loads intact.
        log_action(
            "viewer.session", "retired_diagram_dropped",
            kind=kind, migration=_RETIRED_KINDS[kind],
        )
        raise KeyError(
            f"Diagram kind {kind!r} was retired and is no longer "
            f"loadable. {_RETIRED_KINDS[kind]}"
        )
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
    legends: "list[LegendSnapshot] | tuple[LegendSnapshot, ...] | None" = None,
    clip_planes: "list[ClipPlaneSnapshot] | tuple[ClipPlaneSnapshot, ...] | None" = None,
    clip_apply_cuts: bool = True,
    clip_show_gizmos: bool = True,
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
        "legends":          [
            dataclasses.asdict(lg) for lg in (legends or ())
        ],
        "clip_planes":      [
            dataclasses.asdict(cp) for cp in (clip_planes or ())
        ],
        "clip_apply_cuts":  bool(clip_apply_cuts),
        "clip_show_gizmos": bool(clip_show_gizmos),
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
    legends: list[LegendSnapshot] = []
    for raw in data.get("legends") or []:
        try:
            legends.append(_deserialize_legend(raw))
        except Exception:
            continue
    clip_planes: list[ClipPlaneSnapshot] = []
    for raw in data.get("clip_planes") or []:
        try:
            clip_planes.append(_deserialize_clip_plane(raw))
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
        legends=tuple(legends),
        clip_planes=tuple(clip_planes),
        clip_apply_cuts=bool(data.get("clip_apply_cuts", True)),
        clip_show_gizmos=bool(data.get("clip_show_gizmos", True)),
    )


def _deserialize_clip_plane(raw: dict[str, Any]) -> ClipPlaneSnapshot:
    normal = raw.get("normal") or (1.0, 0.0, 0.0)
    return ClipPlaneSnapshot(
        plane_id=str(raw["plane_id"]),
        name=str(raw.get("name", "Plane")),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        offset=float(raw.get("offset", 0.0)),
        active=bool(raw.get("active", True)),
        flipped=bool(raw.get("flipped", False)),
        gizmo_visible=bool(raw.get("gizmo_visible", True)),
    )


def _deserialize_legend(raw: dict[str, Any]) -> LegendSnapshot:
    anchor = raw.get("anchor") or (0.0, 0.0)
    scale = raw.get("font_scale")
    return LegendSnapshot(
        geometry=str(raw.get("geometry", "")),
        component=str(raw["component"]),
        vertical=bool(raw.get("vertical", True)),
        visible=bool(raw.get("visible", True)),
        fmt=str(raw.get("fmt", "%.3g")),
        font_scale=None if scale is None else float(scale),
        slot=raw.get("slot"),
        anchor=(float(anchor[0]), float(anchor[1])),
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
    legends: "list[LegendSnapshot] | tuple[LegendSnapshot, ...] | None" = None,
    clip_planes: "list[ClipPlaneSnapshot] | tuple[ClipPlaneSnapshot, ...] | None" = None,
    clip_apply_cuts: bool = True,
    clip_show_gizmos: bool = True,
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
        legends=legends,
        clip_planes=clip_planes,
        clip_apply_cuts=clip_apply_cuts,
        clip_show_gizmos=clip_show_gizmos,
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


def session_restorable(session: ViewerSession) -> bool:
    """Whether ``session`` carries anything worth restoring.

    Diagrams were the only restorable payload until ADR 0083 added
    section planes, which exist without any diagram — a planes-only
    session (cut substrate, no contours) must not be dropped by the
    restore gate (S1 review finding B2). Legends stay off this
    predicate: they cannot exist without the diagram they annotate.
    """
    if session.diagrams:
        return True
    return bool(getattr(session, "clip_planes", ()))


__all__ = [
    "SESSION_SCHEMA_VERSION",
    "ClipPlaneSnapshot",
    "CompositionSnapshot",
    "GeometrySnapshot",
    "LegendSnapshot",
    "ViewerSession",
    "default_session_path",
    "deserialize_session",
    "deserialize_spec",
    "load_session",
    "save_session",
    "serialize_session",
    "serialize_spec",
    "session_restorable",
]
