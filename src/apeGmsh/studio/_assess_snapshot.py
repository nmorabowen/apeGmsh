"""Assess snapshot: ``.apegmsh/assess.json`` (ADR 0095 Amendment 5).

Written by the assess verb (CLI ``--assess``; MCP ``assess`` shells to
the same CLI, INV-5) so the verdict reaches :mod:`._bundle` without
re-running assess. Same atomic-replace discipline as ``names.json``
(INV-16, ``_paths.atomic_write_text``). Readers tolerate a torn /
unparseable file (the ``names_error`` degrade pattern in ``_status.py``)
and never raise.

``results_pin`` copies the live snapshot into
``.apegmsh/pins/<id>/assess.json`` so the archived chapter keeps the
verdict that was current at pin time even after the live file changes
or is deleted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._paths import assess_path, atomic_write_text, display_path

ASSESS_SCHEMA = 1


def build_snapshot(
    payload: dict[str, Any],
    *,
    target: Path | str,
    model_h5: Path | str | None,
    root: Path,
    ts: str,
) -> dict[str, Any]:
    """``assess_artifact`` payload -> the ``.apegmsh/assess.json`` shape."""
    return {
        "schema": ASSESS_SCHEMA,
        "ts": ts,
        "targets": {
            "path": display_path(target, root),
            "model_h5": (
                display_path(model_h5, root) if model_h5 is not None else None
            ),
        },
        "findings": payload.get("findings") or [],
        "skipped": payload.get("skipped") or [],
        "figures": [display_path(p, root) for p in (payload.get("figures") or [])],
        "text": payload.get("text") or "",
    }


def write_snapshot(root: Path, snapshot: dict[str, Any]) -> Path:
    return atomic_write_text(assess_path(root), json.dumps(snapshot, indent=2) + "\n")


def read_snapshot_safe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Torn / unparseable JSON degrades to ``(None, reason)``; never raises."""
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "assess snapshot JSON must be an object"
    return data, None
