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

from ._busy import read_busy
from ._envelope import read_envelope
from ._contract import CONTRACT_VERSION
from ._host_state import read_host
from ._ledger import read_runs_safe
from ._paths import envelope_path, ledger_path, names_path, resolve_root
from ._project import read_project

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
        "host": read_host(base),
        "project": read_project(base),
        "busy": read_busy(base),
    }


def has_studio_state(payload: dict[str, Any]) -> bool:
    return payload.get("names") is not None or payload.get("last_run") is not None


def brief_status(
    payload: dict[str, Any] | None = None,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Slim agent-facing status (token budget; MCP ``mode="brief"``).

    Keeps root / last-run intent / label+PG lists / counts summary / pick
    summary. Omits ``names.entities`` and the full selection evidence
    dump. ``text`` mirrors CLI ``format_status``.
    """
    full = collect_status(root) if payload is None else payload
    last = full.get("last_run") or {}
    names = full.get("names") or {}
    labels = list(last.get("labels") or names.get("labels") or [])
    pgs = list(last.get("physical_groups") or names.get("physical_groups") or [])
    raw_counts = last.get("counts") or names.get("counts") or {}
    counts: dict[str, Any] = {}
    if isinstance(raw_counts, dict):
        for key in ("entities", "elements"):
            block = raw_counts.get(key)
            if isinstance(block, dict):
                counts[key] = block

    pick: dict[str, Any] | None = None
    sel = full.get("selection")
    if isinstance(sel, dict):
        unnamed = sel.get("unnamed") or []
        pick = {
            "labels": list(sel.get("labels") or []),
            "physical_groups": list(sel.get("physical_groups") or []),
            "n_unnamed": len(unnamed) if isinstance(unnamed, (list, tuple)) else 0,
            "phase": sel.get("phase"),
        }

    proj = full.get("project")
    project = None
    if isinstance(proj, dict) and proj.get("script"):
        project = {"script": proj.get("script")}

    host = full.get("host") if isinstance(full.get("host"), dict) else {}
    busy = full.get("busy") if isinstance(full.get("busy"), dict) else {}

    last_brief = None
    if last:
        last_brief = {
            "ok": last.get("ok"),
            "phase": last.get("phase"),
            "script": last.get("script"),
            "stopped_at": last.get("stopped_at"),
            "session": last.get("session"),
            "ts": last.get("ts"),
        }
        err = last.get("error")
        if err:
            last_brief["error"] = str(err).strip().splitlines()[-1]

    return {
        "schema": STATUS_SCHEMA,
        "contract_version": full.get("contract_version") or CONTRACT_VERSION,
        "mode": "brief",
        "root": full.get("root"),
        "empty": not has_studio_state(full),
        "phase": (last_brief or {}).get("phase") or names.get("phase"),
        "last": last_brief,
        "n_runs": full.get("n_runs") or 0,
        "n_pins": full.get("n_pins") or 0,
        "labels": labels,
        "physical_groups": pgs,
        "counts": counts,
        "pick": pick,
        "host": {
            "running": bool(host.get("running")),
            "pid": host.get("pid"),
            "phase": host.get("phase"),
            "stale": bool(host.get("stale")),
        },
        "busy": {
            "busy": bool(busy.get("busy")),
            "pid": busy.get("pid"),
            "op": busy.get("op"),
            "phase": busy.get("phase"),
            "stale": bool(busy.get("stale")),
        },
        "project": project,
        "names_error": full.get("names_error"),
        "selection_error": full.get("selection_error"),
        "ledger_error": full.get("ledger_error"),
        "text": format_status(full),
    }


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
        f"  root: {payload.get('root') or '(none)'}",
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
    host = payload.get("host") or {}
    if host.get("running"):
        lines.append(
            f"  host: running pid={host.get('pid')} phase={host.get('phase')}"
        )
    elif host.get("stale"):
        lines.append(
            f"  host: stale (dead pid={host.get('pid')})"
        )
    else:
        lines.append("  host: (not running)")
    busy = payload.get("busy") or {}
    if busy.get("busy"):
        lines.append(
            f"  busy: pid={busy.get('pid')} op={busy.get('op')} "
            f"phase={busy.get('phase')}"
        )
    elif busy.get("stale"):
        lines.append(f"  busy: stale (dead pid={busy.get('pid')})")
    proj = payload.get("project") or {}
    if proj.get("script"):
        lines.append(f"  project.script: {proj.get('script')}")
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
