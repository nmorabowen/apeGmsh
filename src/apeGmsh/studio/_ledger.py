"""Append-only run ledger (ADR 0095 S2c / S5f).

``names.json`` is the current snapshot. ``.apegmsh/runs.jsonl`` is the
history of ``run_until`` stops — enough for a later session to reconstruct
phase, counts, and names without the chat. Assess findings and visor
paths are reserved keys; they stay empty until those artifacts exist.

Append is best-effort (not atomic replace). Readers skip torn lines and
never raise (INV-16). Pure: no ``gmsh``, no Qt.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._paths import display_path

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
    root: Path | str | None = None,
    cwd: Path | str | None = None,
) -> dict[str, Any]:
    """One JSONL object. ``assess`` / ``visors`` reserved, not invented.

    ``root`` is the habitat project root (INV-15), stored absolute.
    ``script`` / ``cwd`` are root-relative posix paths when they fall
    under that root (S5i); absolute otherwise.
    """
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
    base = Path(root).expanduser().resolve() if root is not None else None
    rec: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "ts": ts if ts is not None else datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "script": display_path(script, base),
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
    if base is not None:
        rec["root"] = str(base)
    if cwd is not None:
        rec["cwd"] = display_path(cwd, base)
    return rec


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


def read_runs_safe(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Load JSONL records, skipping bad lines (INV-16 / S5f).

    Missing file → ``([], None)``. Any skipped line (bad JSON, non-object,
    or a UTF-8 tear mid-line) → ``(good_rows, error_message)``. Never
    raises. Bytes are decoded with ``errors="replace"`` so a partial
    multibyte append does not discard earlier good rows.
    """
    path = Path(path)
    if not path.is_file():
        return [], None
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return [], str(exc)
    text = raw_bytes.decode("utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    skipped = 0
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(obj, dict):
            skipped += 1
            continue
        rows.append(obj)
    if skipped:
        return rows, f"skipped {skipped} unparseable ledger line(s)"
    return rows, None


def read_runs(path: Path) -> list[dict[str, Any]]:
    """Load every parseable JSONL record. Missing / torn → best-effort list."""
    rows, _err = read_runs_safe(path)
    return rows


def last_run(path: Path) -> dict[str, Any] | None:
    rows = read_runs(path)
    return rows[-1] if rows else None
