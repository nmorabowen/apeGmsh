"""Append-only run ledger (ADR 0095 S2c).

``names.json`` is the current snapshot. ``.apegmsh/runs.jsonl`` is the
history of ``run_until`` stops — enough for a later session to reconstruct
phase, counts, and names without the chat. Assess findings and visor
paths are reserved keys; they stay empty until those artifacts exist.

Pure: no ``gmsh``, no Qt. Imported only by the host.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEDGER_SCHEMA = 1
_ERROR_CAP = 8000


def make_record(
    *,
    script: Path,
    phase: str,
    ok: bool,
    hash: str | None = None,
    stopped_at: str | None = None,
    session: str | None = None,
    manifest: dict[str, Any] | None = None,
    error: str | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """One JSONL object. ``assess`` / ``visors`` reserved, not invented."""
    labels: list[str] = []
    pgs: list[str] = []
    counts: dict[str, Any] | None = None
    if manifest is not None:
        labels = list(manifest.get("labels") or [])
        pgs = list(manifest.get("physical_groups") or [])
        raw_counts = manifest.get("counts")
        counts = raw_counts if isinstance(raw_counts, dict) else None
    err = error
    if err is not None and len(err) > _ERROR_CAP:
        err = err[:_ERROR_CAP]
    return {
        "schema": LEDGER_SCHEMA,
        "ts": ts if ts is not None else datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "script": str(Path(script)),
        "hash": hash,
        "phase": phase,
        "stopped_at": stopped_at,
        "ok": ok,
        "session": session,
        "labels": labels,
        "physical_groups": pgs,
        "counts": counts,
        "assess": None,
        "visors": [],
        "error": err,
    }


def append_run(path: Path, record: dict[str, Any]) -> Path | None:
    """Append one compact JSON line. ``OSError`` is swallowed (preview still ok)."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        return None
    return path


def read_runs(path: Path) -> list[dict[str, Any]]:
    """Load every JSONL record. Missing file → empty list."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def last_run(path: Path) -> dict[str, Any] | None:
    rows = read_runs(path)
    return rows[-1] if rows else None
