"""Point tools: ``highlight`` payload + ``promote_selection`` (ADR 0095 S4d).

Pure: no ``gmsh``, no Qt, no viewers. MCP writes
``.apegmsh/highlight.json`` (names-first request). The Qt host consumes
that file via ``SelectionState.select_batch`` (INV-7) after a mtime
change — file poll is the INV-5 adapter, not a Qt
``viewer.highlight(names)`` mutator. ``highlight`` does not write
``selection.json`` (the envelope stays the human pick until the host
applies). ``promote_selection`` returns suggested ``.to_label()`` /
``.to_physical()`` edits; it never writes the script, OCC, or
``model.h5`` (INV-2 / INV-9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ._envelope import PickEvidence, read_envelope
from ._paths import envelope_path, highlight_path, names_path, resolve_root

HIGHLIGHT_SCHEMA = 1
_DIM_WORD = {0: "point", 1: "curve", 2: "face", 3: "solid"}


def _root(root: Path | str | None) -> Path:
    return resolve_root(root)


def _unique_names(names: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _load_names(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _records_from_manifest(
    manifest: dict[str, Any],
) -> dict[tuple[int, int], Any]:
    records: dict[tuple[int, int], Any] = {}
    for raw in manifest.get("entities") or ():
        if not isinstance(raw, dict):
            continue
        try:
            dim, tag = int(raw["dim"]), int(raw["tag"])
        except (KeyError, TypeError, ValueError):
            continue
        bbox = raw.get("bbox")
        if bbox is not None:
            bbox = tuple(float(x) for x in bbox)
            if len(bbox) != 6:
                bbox = None
        records[(dim, tag)] = {
            "labels": tuple(raw.get("labels") or ()),
            "physical_groups": tuple(raw.get("physical_groups") or ()),
            "bbox": bbox,
        }
    return records


def resolve_names(
    names: Sequence[str],
    manifest: dict[str, Any] | None,
) -> tuple[list[tuple[int, int]], list[str], dict[tuple[int, int], Any]]:
    """Map label / PG names to dimtags using ``names.json`` entities."""
    wanted = _unique_names(names)
    records = _records_from_manifest(manifest or {})
    matched: list[tuple[int, int]] = []
    found: set[str] = set()
    for dimtag, rec in records.items():
        if any(n in rec["labels"] or n in rec["physical_groups"] for n in wanted):
            matched.append(dimtag)
            found.update(rec["labels"])
            found.update(rec["physical_groups"])
    missing = [n for n in wanted if n not in found]
    return matched, missing, records


def dumps_highlight(
    *,
    names: Sequence[str],
    dimtags: Sequence[tuple[int, int]],
) -> str:
    payload = {
        "schema": HIGHLIGHT_SCHEMA,
        "names": list(names),
        "dimtags": [[int(d), int(t)] for d, t in dimtags],
    }
    return json.dumps(payload, indent=2) + "\n"


def write_highlight(
    path: Path,
    *,
    names: Sequence[str],
    dimtags: Sequence[tuple[int, int]],
) -> Path:
    from ._paths import atomic_write_text

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(
        path, dumps_highlight(names=names, dimtags=dimtags),
    )


def read_highlight(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("highlight JSON must be an object")
    names = [str(n) for n in (data.get("names") or ()) if str(n).strip()]
    dimtags: list[tuple[int, int]] = []
    for item in data.get("dimtags") or ():
        if isinstance(item, (list, tuple)) and len(item) == 2:
            dimtags.append((int(item[0]), int(item[1])))
    return {
        "schema": int(data.get("schema") or HIGHLIGHT_SCHEMA),
        "names": names,
        "dimtags": dimtags,
    }


def seed_highlight_mtime(path: Path) -> float | None:
    """Current mtime, or None if missing. Does not read the payload."""
    if not path.is_file():
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def consume_highlight_update(
    path: Path,
    last_mtime: float | None,
) -> tuple[dict[str, Any] | None, float | None]:
    """Return a new payload only when ``highlight.json`` mtime advanced.

    Seed ``last_mtime`` with :func:`seed_highlight_mtime` at host open so
    a leftover file from a previous session is not applied.
    """
    if not path.is_file():
        return None, last_mtime
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None, last_mtime
    if mtime == last_mtime:
        return None, last_mtime
    try:
        return read_highlight(path), mtime
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
        return None, mtime


def apply_highlight_to_state(
    sel: object,
    *,
    names: Sequence[str] | None = None,
    resolve: Any = None,
) -> list[tuple[int, int]]:
    """Owner mutator: ``select_batch(..., replace=True)`` (INV-7).

    ``resolve(names)`` is the live-kernel lookup. Dimtags in the JSON
    are evidence, never identity (INV-3): a miss is a no-op so the
    human pick is not cleared.
    """
    wanted = _unique_names(names or ())
    if not wanted or resolve is None:
        return []
    found = resolve(wanted)
    if not found:
        return []
    targets = [(int(d), int(t)) for d, t in found]
    batch = getattr(sel, "select_batch", None)
    if batch is None:
        raise TypeError("selection state has no select_batch")
    batch(targets, replace=True)
    return targets


def highlight(
    names: Sequence[str],
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Write ``highlight.json`` only. Does not write ``selection.json``."""
    wanted = _unique_names(names)
    if not wanted:
        raise ValueError("highlight requires at least one name")
    base = _root(root)
    manifest = _load_names(names_path(base))
    dimtags, missing, _records = resolve_names(wanted, manifest)
    dest = write_highlight(
        highlight_path(base), names=wanted, dimtags=dimtags,
    )
    return {
        "ok": not missing,
        "path": str(dest),
        "names": wanted,
        "matched": [[d, t] for d, t in dimtags],
        "missing": missing,
        "note": (
            "Wrote highlight.json (request). Does not write "
            "selection.json; the host applies names via select_batch "
            "when the file mtime changes."
        ),
    }


def promote_selection(*, root: Path | str | None = None) -> dict[str, Any]:
    """Suggested ``.to_label()`` / ``.to_physical()`` edits for unnamed picks."""
    base = _root(root)
    path = envelope_path(base)
    if not path.is_file():
        return {
            "ok": True,
            "present": False,
            "unnamed": [],
            "suggestions": [],
            "applied": False,
            "note": (
                "No selection envelope. Pick in the host, then call "
                "promote_selection. The host does not write the .py."
            ),
        }
    try:
        env = read_envelope(path)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {
            "ok": False,
            "error": {"code": "UNREADABLE", "message": str(exc)},
            "present": False,
            "path": str(path),
            "unnamed": [],
            "suggestions": [],
            "applied": False,
            "note": "selection.json is present but unreadable.",
        }
    suggestions = [_suggestion(item) for item in env.unnamed]
    return {
        "ok": True,
        "present": True,
        "unnamed": env.to_dict().get("unnamed") or [],
        "suggestions": suggestions,
        "applied": False,
        "note": (
            "Suggested script edits (INV-2 / INV-9). "
            "The host does not write the .py, OCC, or model.h5."
        ),
    }


def _suggestion(item: PickEvidence) -> dict[str, Any]:
    word = _DIM_WORD.get(item.dim, f"dim{item.dim}")
    label = f"{word}_{item.key}"
    bbox = item.bbox
    if bbox is None:
        return {
            "dim": item.dim,
            "key": item.key,
            "suggested_name": label,
            "to_label": None,
            "to_physical": None,
            "note": (
                f"unnamed {word} tag={item.key} has no bbox; pick again "
                "after names.json has entity bboxes. Do not author tags "
                "(INV-3)."
            ),
        }
    lo = ", ".join(_fmt_num(x) for x in bbox[:3])
    hi = ", ".join(_fmt_num(x) for x in bbox[3:])
    chain = (
        f"g.model.select(None, dim={item.dim})"
        f".in_box(({lo}), ({hi}))"
    )
    return {
        "dim": item.dim,
        "key": item.key,
        "suggested_name": label,
        "to_label": f"{chain}.to_label({label!r})",
        "to_physical": f"{chain}.to_physical({label!r})",
        "note": (
            "in_box is BRep bbox containment (closed). dim= keeps the "
            "dimension; smaller entities whose bbox sits inside may "
            "also match."
        ),
    }


def _fmt_num(value: float) -> str:
    text = f"{float(value):.6g}"
    return text
