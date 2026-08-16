"""Read studio artifacts without replaying (ADR 0095 S2d / S5b).

``run_until`` writes the files. ``--status`` is how a later session
extracts the basic information — last phase, names, counts, pick —
without exec. Pure: no ``gmsh``, no Qt.

Torn / unparseable snapshot JSON degrades to ``names_error`` /
``selection_error``; torn ``runs.jsonl`` lines degrade to
``ledger_error`` (INV-16 / S5f). Never raises out of ``collect_status``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._envelope import read_envelope
from ._contract import CONTRACT_VERSION
from ._ledger import read_runs_safe
from ._paths import envelope_path, ledger_path, names_path, resolve_root

STATUS_SCHEMA = 1


def collect_status(root: Path | str | None = None) -> dict[str, Any]:
    """Snapshot of ``.apegmsh/`` under the resolved project root (INV-15)."""
    base = resolve_root(root)
    names, names_error = _load_json_safe(names_path(base))
    runs, ledger_error = read_runs_safe(ledger_path(base))
    pins = [row for row in runs if row.get("kind") == "pin"]
    execs = [row for row in runs if row.get("kind") != "pin"]
    selection: dict[str, Any] | None = None
    selection_error: str | None = None
    env_file = envelope_path(base)
    if env_file.is_file():
        try:
            selection = read_envelope(env_file).to_dict()
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            selection_error = str(exc)
    return {
        "schema": STATUS_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "root": str(base),
        "names": names,
        "names_error": names_error,
        "last_run": execs[-1] if execs else None,
        "n_runs": len(execs),
        "last_pin": pins[-1] if pins else None,
        "n_pins": len(pins),
        "selection": selection,
        "selection_error": selection_error,
        "ledger_error": ledger_error,
    }


def has_studio_state(payload: dict[str, Any]) -> bool:
    return payload.get("names") is not None or payload.get("last_run") is not None


def format_status(payload: dict[str, Any]) -> str:
    """Short agent-readable dump. Not a substitute for the JSON files."""
    if (
        not has_studio_state(payload)
        and not payload.get("names_error")
        and not payload.get("ledger_error")
    ):
        return "studio status: no .apegmsh state (run a script first)"
    last = payload.get("last_run") or {}
    names = payload.get("names") or {}
    labels = last.get("labels") or names.get("labels") or []
    pgs = last.get("physical_groups") or names.get("physical_groups") or []
    counts = last.get("counts") or names.get("counts") or {}
    lines = [
        "studio status",
        f"  last: {last.get('ts') or '(no ledger)'}  "
        f"ok={_fmt(last.get('ok'))}  "
        f"phase={last.get('phase') or names.get('phase') or '(unknown)'}  "
        f"stopped_at={last.get('stopped_at')!r}",
        f"  session: {last.get('session') or '(none)'}",
        f"  script: {last.get('script') or '(none)'}",
        f"  n_runs: {payload.get('n_runs') or 0}",
        f"  labels: {_join(labels)}",
        f"  physical_groups: {_join(pgs)}",
        f"  entities: {_fmt_counts(counts.get('entities'))}",
        f"  elements: {_fmt_counts(counts.get('elements'))}",
        f"  pick: {_fmt_pick(payload)}",
    ]
    names_err = payload.get("names_error")
    if names_err:
        lines.append(f"  names: (unreadable: {names_err})")
    ledger_err = payload.get("ledger_error")
    if ledger_err:
        lines.append(f"  ledger: (degraded: {ledger_err})")
    err = last.get("error")
    if err:
        first = str(err).strip().splitlines()[-1]
        lines.append(f"  error: {first}")
    return "\n".join(lines)


def _load_json_safe(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, str(exc)
    if isinstance(data, dict):
        return data, None
    return None, "names JSON must be an object"


def _join(items: Any) -> str:
    seq = [str(x) for x in items if x]
    return ", ".join(seq) if seq else "(none)"


def _fmt(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _fmt_counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "(none)"
    return " ".join(f"{k}={v}" for k, v in counts.items())


def _fmt_pick(payload: dict[str, Any]) -> str:
    err = payload.get("selection_error")
    if err:
        return f"(unreadable: {err})"
    sel = payload.get("selection")
    if not sel:
        return "(none)"
    labels = _join(sel.get("labels") or [])
    pgs = _join(sel.get("physical_groups") or [])
    unnamed = sel.get("unnamed") or []
    n_unnamed = len(unnamed) if isinstance(unnamed, list) else 0
    return f"labels={labels}  pgs={pgs}  unnamed={n_unnamed}"
