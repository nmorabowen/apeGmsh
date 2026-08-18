"""Session snapshot — the JSON of what the human built (ADR 0098 §11 S5).

One file, one session: panes, their slots, the pose, the time link and
each pane's own instant, the one selection set. An agent can then draw
a still of what a human arranged (``render`` of a snapshot, S5c) and a
pin can carry it (``results_pin``, S5b, under the record key
``session_snapshot``).

**The schema is frozen at version 1** against exactly what S0 shipped.
S4 never widened the time surface — plan decision 9 landed on a
one-stage-at-a-time scrubber, which is a WIDGET choice: ``Instant`` is
``(stage, step)`` under either traversal — so S5b's contract can
publish without paying a second version bump.

Two failure families, deliberately not alike (plan decision 15):

* **Schema / ontology violations refuse loudly.** An unknown slot
  category is the loudest of them: the §4 catalog is CLOSED (amended
  ADR 0094 INV-10), and this refusal is that amendment's enforcement
  point, not a nicety. Restore builds the real frozen records, so every
  law S0 wrote — the closed catalog, the scope axes, the deform fields,
  the plot kinds, no negative steps — is enforced on the way in by the
  same validator a script hits. There is no second, weaker copy of the
  laws here.
* **Data mismatches degrade with a notice** on
  ``RestoredSession.notices``. An instant naming a stage these results
  no longer have drops to ``None`` (realize's documented "last stage,
  last step"); a stage rename must not cost the human every pane, slot
  and scope in the file. Silence is the only forbidden option.

What is NOT snapshot state: the ``SelectionLog`` op history (nothing
realizes from it, and a replayed gesture has no model left to hit — the
set restores as ONE honest ``SET`` write), the derived legends (§5:
``legends = f(occupied colour-mapped slots)``; only the per-field
*hidden* chrome is state), and every widget geometry — the window is a
projection, never truth.

The legacy gate: the S6a flip ADOPTED ``<results>.viewer-session.json``
(plan decision 11), so save-on-close now writes exactly where a v13
file from the retired window lives. A v13-shaped file therefore gets a
notice and a ``.legacy`` rename-aside — and **never** an overwrite, of
the original or of an existing ``.legacy``. :func:`legacy_shape` is the
bare predicate so S5c's MCP verb can refuse an old file without
renaming anything (the rename belongs to the human flow), and
:mod:`apeGmsh.results.session._boot` is where the window's open policy
turns a refused rename into a notice plus a disarmed auto-save instead
of a window that will not open.
"""
from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from apeGmsh._atomic_io import atomic_write_text, replace_with_retry

from ._session import ResultsSession
from ._slots import SLOT_CATALOG, Slot
from ._time import Instant
from ._views import (
    Deform,
    MeshStyle,
    MeshView,
    PlotSeries,
    PlotSource,
    PlotView,
    Scope,
    ViewClip,
)

if TYPE_CHECKING:  # pragma: no cover
    from apeGmsh.results.Results import Results

#: Marker key + value. The OLD viewer session carries an int
#: ``schema_version`` and no ``kind``; this carries ``kind`` and no
#: ``schema_version``. The two shapes are told apart by a key that
#: exists, never by comparing 1 against 13.
SNAPSHOT_KIND = "apegmsh.results.session"

#: Frozen against S0's IR (see the module docstring). A new slot
#: category, a new pane kind or a new *stored* field is an ADR 0098
#: amendment AND a bump here.
SNAPSHOT_VERSION = 1

#: Suffix appended (never substituted) when an old viewer session is
#: moved out of the new session's way.
LEGACY_SUFFIX = ".legacy"

#: The snapshot filename, ADOPTED from the old viewer at the S6a flip
#: (plan decision 11). Through S2–S5 the new session wrote
#: ``<results>.session.json`` beside the old window's
#: ``<results>.viewer-session.json``; the flip takes the old name over,
#: which is why :func:`rename_legacy_aside` exists — save-on-close now
#: writes exactly where a v13 file lives.
_SNAPSHOT_SUFFIX = ".viewer-session.json"


class SnapshotError(ValueError):
    """The file is not a session this ontology has — a schema or
    ontology violation (unknown slot category, unknown pane kind,
    missing marker, unknown version). Loud by design."""


class LegacySessionFile(SnapshotError):
    """The file is the OLD viewer's v13 session (``viewers.diagrams``).

    Raised by :func:`load_snapshot` when ``rename_legacy=False`` — the
    contract S5c's MCP verb needs: refuse an old-schema file, never
    rename it. The human flow renames it aside instead.
    """

    def __init__(self, path: "str | Path", schema_version: Any = None) -> None:
        self.path = Path(path)
        self.schema_version = schema_version
        super().__init__(
            f"{self.path} is an old viewer session"
            + (
                f" (schema v{schema_version})"
                if schema_version is not None else ""
            )
            + " from the retired diagram ontology; ADR 0098 does not "
            "restore it. Open it in the results window to have it "
            "renamed aside, or point at a "
            f"'{_SNAPSHOT_SUFFIX}' snapshot."
        )


@dataclass(frozen=True)
class RestoredSession:
    """A restored session plus every degradation it survived.

    ``notices`` is empty for a clean restore. It is a RETURN value, not
    a log line, because the caller decides how loud to be: the window
    shows them, the MCP verb reports them, a test asserts them. What no
    caller may do is not know.
    """

    session: ResultsSession
    notices: tuple[str, ...] = ()


# ======================================================================
# Serialize
# ======================================================================

def _dump_instant(instant: Optional[Instant]) -> Optional[dict]:
    if instant is None:
        return None
    return {"stage": instant.stage, "step": int(instant.step)}


def _dump_slot(record: Slot) -> dict:
    # The seven records are frozen dataclasses of plain tokens, so
    # asdict IS the payload — Reactions correctly yields {}.
    return asdict(record)  # type: ignore[call-overload]


def _dump_source(source: PlotSource) -> dict:
    key: Any = source.key
    if source.kind == "gauss":
        key = [int(key[0]), int(key[1])]  # type: ignore[index]
    elif source.kind == "node":
        key = int(key)  # type: ignore[arg-type]
    return {"kind": source.kind, "key": key}


def _dump_mesh_view(view: MeshView) -> dict:
    return {
        "pane": "mesh",
        "id": view.id,
        "name": view.name,
        "scope": (
            None if view.scope is None else {
                "axis": view.scope.axis,
                "names": (
                    None if view.scope.names is None
                    else list(view.scope.names)
                ),
            }
        ),
        "deform": (
            None if view.deform is None else {
                "field": view.deform.field,
                "scale": view.deform.scale,
                "mode": view.deform.mode,
            }
        ),
        "time": _dump_instant(view.time),
        "style": {
            "mesh": view.style.mesh,
            "outlines": view.style.outlines,
            "nodes": view.style.nodes,
            "gauss": view.style.gauss,
        },
        "overlay": view.overlay,
        "pick_target": view.pick_target,
        "slots": {
            category: _dump_slot(record)
            for category, record in view.slots.items()
        },
        # Derived from legends(), not from the private flag dict: only a
        # field an occupied colour-mapped slot actually causes can be
        # hidden (INV-LEGEND-2/-3), so the file cannot carry chrome for
        # a legend that does not exist.
        "legend_hidden": [
            legend.field for legend in view.legends() if legend.hidden
        ],
        "clips": [
            {
                "plane_id": clip.plane_id,
                "name": clip.name,
                "normal": list(clip.normal),
                "offset": clip.offset,
                "active": clip.active,
                "flipped": clip.flipped,
                "gizmo_visible": clip.gizmo_visible,
            }
            for clip in view.clips
        ],
    }


def _dump_plot_view(plot: PlotView) -> dict:
    return {
        "pane": "plot",
        "id": plot.id,
        "kind": plot.kind,
        "name": plot.name,
        "series": [
            {
                "source": _dump_source(series.source),
                "quantity": series.quantity,
            }
            for series in plot.series
        ],
        "cursor": _dump_instant(plot.cursor),
    }


def _dump_selection(session: ResultsSession) -> dict:
    # `.kind` raises on a store some other writer corrupted — a session
    # whose set broke the §8 XOR law must not be quietly serialised as
    # a half-truth.
    kind = session.selection.kind
    if kind is None:
        return {"kind": None}
    if kind == "nodes":
        return {
            "kind": "nodes",
            "nodes": [int(n) for n in session.selection.nodes],
        }
    return {
        "kind": "gauss",
        "gauss": [
            [int(eid), int(gp)] for eid, gp in session.selection.gauss
        ],
    }


def snapshot(session: ResultsSession) -> dict:
    """This session as a JSON-safe dict (schema
    :data:`SNAPSHOT_VERSION`).

    Panes in creation order; nothing derived is stored (legends are a
    function of the slots, §5) and nothing about a window is stored.
    """
    return {
        "kind": SNAPSHOT_KIND,
        "version": SNAPSHOT_VERSION,
        "saved_at": datetime.datetime.now(
            datetime.timezone.utc,
        ).isoformat(),
        "results_path": _results_path_str(session.results),
        "time": _dump_instant(session.time),
        "time_linked": session.time_linked,
        # Both halves of the §7 state, always: `time_linked` alone would
        # restore an unlinked session with every pane silently relinked
        # to one instant (plan decision 15). Each pane's own instant
        # rides on the pane, above.
        "selection": _dump_selection(session),
        "panes": [
            _dump_mesh_view(pane) if isinstance(pane, MeshView)
            else _dump_plot_view(pane)
            for pane in session.panes
        ],
    }


# ======================================================================
# Restore
# ======================================================================

def _require_dict(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise SnapshotError(
            f"{what} must be a JSON object; got "
            f"{type(value).__name__}."
        )
    return value


def _read_instant(raw: Any, what: str) -> Optional[Instant]:
    if raw is None:
        return None
    data = _require_dict(raw, what)
    try:
        return Instant(stage=data["stage"], step=data["step"])
    except KeyError as exc:
        raise SnapshotError(
            f"{what} needs both 'stage' and 'step'; missing {exc}."
        ) from None


def _read_slot(category: str, raw: Any, pane_id: str) -> Slot:
    record_type = SLOT_CATALOG.get(category)
    if record_type is None:
        # THE amended-0094-INV-10 enforcement point (§4). The catalog is
        # closed: a category this build does not have means the file was
        # written by an ontology this one does not share, and guessing
        # would draw a picture nobody authorised. A new category is an
        # ADR 0098 amendment, not a silent drop.
        raise SnapshotError(
            f"Pane {pane_id!r} carries an unknown result-slot category "
            f"{category!r}. The ADR 0098 §4 catalog is CLOSED (amended "
            f"ADR 0094 INV-10) — the seven categories are "
            f"{sorted(SLOT_CATALOG)}. A new slot is an ADR 0098 "
            f"amendment, not a snapshot field."
        )
    fields = _require_dict(raw, f"pane {pane_id!r} slot {category!r}")
    try:
        return record_type(**fields)
    except TypeError as exc:
        raise SnapshotError(
            f"Pane {pane_id!r} slot {category!r} does not fit "
            f"{record_type.__name__}: {exc}."
        ) from None


def _restore_mesh_view(raw: dict, notices: list[str]) -> MeshView:
    pane_id = raw.get("id")
    if not isinstance(pane_id, str) or not pane_id:
        raise SnapshotError(
            f"A mesh pane needs a non-empty string 'id'; got "
            f"{pane_id!r}."
        )
    view = MeshView(pane_id=pane_id, name=raw.get("name"))

    scope_raw = raw.get("scope")
    if scope_raw is not None:
        scope = _require_dict(scope_raw, f"pane {pane_id!r} scope")
        names = scope.get("names")
        view.scope = Scope(
            axis=scope.get("axis"),
            names=None if names is None else tuple(names),
        )

    deform_raw = raw.get("deform")
    if deform_raw is not None:
        deform = _require_dict(deform_raw, f"pane {pane_id!r} deform")
        view.deform = Deform(
            field=deform.get("field", "displacement"),
            scale=deform.get("scale"),
            mode=deform.get("mode"),
        )

    view.time = _read_instant(raw.get("time"), f"pane {pane_id!r} time")

    style_raw = raw.get("style")
    if style_raw is not None:
        style = _require_dict(style_raw, f"pane {pane_id!r} style")
        view.style = MeshStyle(
            mesh=style.get("mesh", True),
            outlines=style.get("outlines", True),
            nodes=style.get("nodes", False),
            gauss=style.get("gauss", False),
        )

    view.overlay = bool(raw.get("overlay", False))
    if "pick_target" in raw:
        view.pick_target = raw["pick_target"]

    # Slots BEFORE legend chrome: a hidden flag exists only as long as
    # the legend its slot causes (INV-LEGEND-2), and set_legend_hidden
    # refuses a field no occupied slot causes.
    for category, payload in _require_dict(
        raw.get("slots") or {}, f"pane {pane_id!r} slots",
    ).items():
        setattr(view, category, _read_slot(category, payload, pane_id))

    live = {legend.field for legend in view.legends()}
    for field in raw.get("legend_hidden") or ():
        if field not in live:
            # Only reachable by hand-editing: the writer derives these
            # from legends(). A data inconsistency, so it degrades.
            notices.append(
                f"Pane {pane_id!r}: dropped a hidden-legend flag for "
                f"field {field!r} — no occupied colour-mapped slot on "
                f"this view causes that legend (ADR 0098 §5)."
            )
            continue
        view.set_legend_hidden(field, True)

    clips = []
    for raw_clip in raw.get("clips") or ():
        clip = _require_dict(raw_clip, f"pane {pane_id!r} clip")
        normal = clip.get("normal", (1.0, 0.0, 0.0))
        if len(normal) != 3:
            raise SnapshotError(
                f"Pane {pane_id!r} clip {clip.get('plane_id')!r} has a "
                f"{len(normal)}-component normal; a plane normal is "
                f"three numbers."
            )
        clips.append(ViewClip(
            plane_id=clip.get("plane_id"),
            name=clip.get("name", ""),
            normal=(normal[0], normal[1], normal[2]),
            offset=clip.get("offset", 0.0),
            active=clip.get("active", True),
            flipped=clip.get("flipped", False),
            gizmo_visible=clip.get("gizmo_visible", True),
        ))
    # Verbatim, keeping each plane_id — it is identity (the outline and
    # the gizmos address planes by it), so add_clip() cannot be used: it
    # mints a fresh one. This also re-seeds the clip-id counter.
    view._adopt_clips(clips)
    return view


def _restore_plot_view(raw: dict) -> PlotView:
    pane_id = raw.get("id")
    if not isinstance(pane_id, str) or not pane_id:
        raise SnapshotError(
            f"A plot pane needs a non-empty string 'id'; got "
            f"{pane_id!r}."
        )
    series = []
    for raw_series in raw.get("series") or ():
        item = _require_dict(raw_series, f"pane {pane_id!r} series")
        source = _require_dict(
            item.get("source"), f"pane {pane_id!r} series source",
        )
        key = source.get("key")
        series.append(PlotSeries(
            source=PlotSource(
                source.get("kind"),
                tuple(key) if isinstance(key, list) else key,
            ),
            quantity=item.get("quantity"),
        ))
    plot = PlotView(
        pane_id=pane_id,
        kind=raw.get("kind", "history"),
        series=tuple(series),
        name=raw.get("name"),
    )
    plot.cursor = _read_instant(
        raw.get("cursor"), f"pane {pane_id!r} cursor",
    )
    return plot


def _restore_selection(session: ResultsSession, raw: Any) -> None:
    if raw is None:
        return
    data = _require_dict(raw, "selection")
    kind = data.get("kind")
    if kind is None:
        return
    if kind == "nodes":
        session.selection.set_nodes(data.get("nodes") or ())
    elif kind == "gauss":
        session.selection.set_gauss(
            [(int(e), int(gp)) for e, gp in (data.get("gauss") or ())]
        )
    else:
        raise SnapshotError(
            f"Selection kind must be 'nodes', 'gauss' or null (ADR "
            f"0098 §8 — nodes XOR Gauss, never elements); got "
            f"{kind!r}."
        )


def _validate_instants(
    session: ResultsSession, results: "Results", notices: list[str],
) -> None:
    """Drop instants these results cannot answer, with a notice each.

    Resolved through the SAME ``results.stage(...)`` lookup realize
    applies, so restore and realize cannot disagree about which stages
    exist. Doing it HERE and not at first paint is the point: an
    unvalidated dead stage id surfaces as a traceback inside a repaint,
    naming a stage, for a fault that belongs to a file.
    """
    def keep(instant: Optional[Instant], where: str) -> Optional[Instant]:
        if instant is None:
            return None
        try:
            scoped = results.stage(instant.stage)
        except KeyError:
            available = sorted({s.id for s in results.stages})
            notices.append(
                f"{where}: stage {instant.stage!r} is not in these "
                f"results (have {available}) — instant dropped; the "
                f"pane falls back to the last recorded step."
            )
            return None
        n_steps = int(scoped.n_steps)
        if instant.step >= n_steps:
            notices.append(
                f"{where}: step {instant.step} is past stage "
                f"{instant.stage!r}'s {n_steps} recorded step(s) — "
                f"instant dropped; the pane falls back to the last "
                f"recorded step."
            )
            return None
        return instant

    session.time = keep(session.time, "Session time")
    for pane in session.panes:
        if isinstance(pane, MeshView):
            pane.time = keep(pane.time, f"Pane {pane.id!r} time")
        else:
            pane.cursor = keep(pane.cursor, f"Plot {pane.id!r} cursor")


def restore_snapshot(
    data: Any, results: "Optional[Results]" = None,
) -> RestoredSession:
    """Build a session from a snapshot dict.

    ``results=`` binds the broker the restored session presents; pass
    ``None`` for an IR-only restore (nothing to validate instants
    against, so they restore verbatim).
    """
    payload = _require_dict(data, "A session snapshot")
    kind = payload.get("kind")
    if kind != SNAPSHOT_KIND:
        raise SnapshotError(
            f"Not an apeGmsh session snapshot: expected "
            f"kind={SNAPSHOT_KIND!r}, got {kind!r}."
            + (
                " This looks like the OLD viewer's session (it carries "
                "'schema_version'); ADR 0098 does not restore it."
                if legacy_shape(payload) else ""
            )
        )
    version = payload.get("version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotError(
            f"Session snapshot schema v{version!r} — this build reads "
            f"v{SNAPSHOT_VERSION}."
        )

    notices: list[str] = []
    session = ResultsSession(results=results)
    for raw_pane in payload.get("panes") or ():
        pane_raw = _require_dict(raw_pane, "A pane")
        pane_kind = pane_raw.get("pane")
        if pane_kind == "mesh":
            pane = _restore_mesh_view(pane_raw, notices)
        elif pane_kind == "plot":
            pane = _restore_plot_view(pane_raw)
        else:
            raise SnapshotError(
                f"Unknown pane kind {pane_kind!r} — a session has mesh "
                f"views and plot views (ADR 0098 §3/§6)."
            )
        session._adopt_pane(pane)

    session.time_linked = bool(payload.get("time_linked", True))
    session.time = _read_instant(payload.get("time"), "Session time")
    _restore_selection(session, payload.get("selection"))

    # TWO counters, not one (S0's runway note): the session's pane ids
    # here, and each view's own clip ids in _adopt_clips above. Restore
    # mesh-3, add a view, and without this you get a SECOND mesh-1.
    session._reseed_ids()

    if results is not None:
        _validate_instants(session, results, notices)
    return RestoredSession(session=session, notices=tuple(notices))


# ======================================================================
# Disk — paths, the legacy gate, atomic write
# ======================================================================

def default_snapshot_path(results_path: "str | Path") -> Path:
    """``<results>.viewer-session.json`` beside the results file.

    The old viewer's name, adopted at the S6a flip (plan decision 11)
    now that nothing else writes it. A file already at this path may
    therefore be a v13 session from the retired window — which is what
    :func:`legacy_shape` and :func:`rename_legacy_aside` below are for.
    """
    p = Path(results_path)
    return p.with_suffix(p.suffix + _SNAPSHOT_SUFFIX)


def _results_path_str(results: "Optional[Results]") -> Optional[str]:
    path = getattr(results, "_path", None)
    return None if path is None else str(Path(path).resolve())


def legacy_shape(data: Any) -> bool:
    """Whether ``data`` is the OLD viewer's session (v13 and friends).

    The bare predicate, so S5c's MCP verb can refuse an old-schema file
    while renaming nothing. Told apart by keys, never by version
    arithmetic: the old envelope carries an int ``schema_version`` plus
    the retired ontology's ``diagrams`` / ``geometries`` blocks, and no
    :data:`SNAPSHOT_KIND` marker.
    """
    if not isinstance(data, dict) or data.get("kind") == SNAPSHOT_KIND:
        return False
    version = data.get("schema_version")
    return (
        isinstance(version, int)
        and not isinstance(version, bool)
        and ("diagrams" in data or "geometries" in data)
    )


def rename_legacy_aside(path: "str | Path") -> Path:
    """Move an old viewer session to ``<path>.legacy``. Never destroys.

    The ADR keeps the old file so a one-shot importer would still have
    its input, which is worth nothing if the second open overwrites the
    first rename. So an existing ``.legacy`` is REFUSED, not replaced —
    and the destination is reserved with an exclusive create before the
    replace, making that guarantee atomic rather than advisory.
    """
    src = Path(path)
    dest = Path(str(src) + LEGACY_SUFFIX)
    try:
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise FileExistsError(
            f"Refusing to move {src.name} aside: {dest.name} already "
            f"exists. An older viewer session was renamed aside here "
            f"before, and overwriting it would destroy it. Move or "
            f"delete {dest} yourself, then reopen."
        ) from None
    os.close(fd)
    try:
        # Atomic, and the only thing it can overwrite is the empty
        # placeholder we just proved we were the ones to create. Through
        # the shared retry because this is the same syscall, on the same
        # platform, with the same transient-denial behaviour that
        # atomic_write_text has always guarded against — an asymmetry
        # that only shows up on a loaded machine, which is the worst
        # kind of thing to leave to chance on a file a human cares
        # about.
        replace_with_retry(src, dest)
    except OSError:
        try:
            os.unlink(dest)  # never leave a 0-byte file blocking a retry
        except OSError:
            pass
        raise
    return dest


def save_snapshot(
    session: ResultsSession, path: "str | Path | None" = None,
) -> Path:
    """Write this session's snapshot; returns the path written.

    Atomically (ADR 0095 INV-16, via the shared
    :func:`apeGmsh._atomic_io.atomic_write_text`): a reader may see the
    file missing mid-replace, never a truncated JSON body. ``path=None``
    uses :func:`default_snapshot_path`, which needs a Results opened
    from disk.
    """
    if path is None:
        results_path = _results_path_str(session.results)
        if results_path is None:
            raise ValueError(
                "This session's Results was not opened from a file, so "
                "there is no <results>.viewer-session.json to default "
                "to — "
                "pass an explicit path."
            )
        path = default_snapshot_path(results_path)
    text = json.dumps(
        snapshot(session), indent=2, ensure_ascii=False,
    ) + "\n"
    return atomic_write_text(path, text)


def load_snapshot(
    path: "str | Path",
    results: "Optional[Results]" = None,
    *,
    rename_legacy: bool = True,
) -> Optional[RestoredSession]:
    """Read a snapshot from disk. ``None`` when the legacy gate fired.

    An old v13-shaped viewer session is not restorable (ADR 0098
    Consequences). In the human flow (``rename_legacy=True``) it earns
    a notice and a ``.legacy`` rename-aside, and this returns ``None``
    — the caller boots a fresh session. With ``rename_legacy=False``
    (S5c's MCP verb) it raises :class:`LegacySessionFile` and touches
    nothing on disk.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if legacy_shape(data):
        version = data.get("schema_version")
        if not rename_legacy:
            raise LegacySessionFile(p, version)
        dest = rename_legacy_aside(p)
        print(
            f"[session] {p.name} is an old viewer session (schema "
            f"v{version}) from the retired diagram ontology; ADR 0098 "
            f"does not restore it. Renamed aside to {dest.name} — "
            f"nothing was overwritten, and a fresh session is used."
        )
        return None
    return restore_snapshot(data, results=results)


__all__ = [
    "SNAPSHOT_KIND",
    "SNAPSHOT_VERSION",
    "LEGACY_SUFFIX",
    "SnapshotError",
    "LegacySessionFile",
    "RestoredSession",
    "snapshot",
    "restore_snapshot",
    "save_snapshot",
    "load_snapshot",
    "default_snapshot_path",
    "legacy_shape",
    "rename_legacy_aside",
]
