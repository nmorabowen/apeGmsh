"""Append-only MCP call ledger (ADR 0096 S4).

Side effect of the stdio adapter, not an MCP tool. Records tool name
and payload byte counts — not arguments, not source. ``OSError`` is
swallowed so a full disk does not fail the habitat verb (same as the
run ledger).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ._paths import mcp_calls_path

MCP_CALL_SCHEMA = 1


def _byte_len(obj: Any) -> int:
    try:
        return len(
            json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))
        )
    except (TypeError, ValueError):
        return len(str(obj))


def make_call_record(
    tool: str,
    payload_in: Any,
    payload_out: Any,
    *,
    ts: str | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema": MCP_CALL_SCHEMA,
        "ts": ts if ts is not None else datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "tool": tool,
        "bytes_in": _byte_len(payload_in),
        "bytes_out": _byte_len(payload_out),
    }
    if isinstance(payload_out, dict) and "ok" in payload_out:
        rec["ok"] = bool(payload_out["ok"])
    if tool == "lookup":
        symbol = _lookup_symbol(payload_in)
        if symbol is not None:
            rec["symbol"] = symbol
        kind = _lookup_kind(payload_out)
        if kind is not None:
            rec["kind"] = kind
        if isinstance(payload_out, dict) and "code" in payload_out:
            try:
                rec["code"] = int(payload_out["code"])
            except (TypeError, ValueError):
                pass
    return rec


def _lookup_kind(payload_out: Any) -> str | None:
    if not isinstance(payload_out, dict):
        return None
    raw = payload_out.get("kind")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    text = payload_out.get("text")
    if isinstance(text, str):
        if text.startswith("miss:"):
            return "miss"
        if text.startswith("ambiguous:"):
            return "ambiguous"
        if text.startswith("error:"):
            return "error"
    if payload_out.get("ok") is True or payload_out.get("code") == 0:
        return "hit"
    return None


def _lookup_symbol(payload_in: Any) -> str | None:
    if not isinstance(payload_in, dict):
        return None
    raw = payload_in.get("symbol")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    args = payload_in.get("args")
    if isinstance(args, list) and args and isinstance(args[0], str) and args[0].strip():
        return args[0].strip()
    return None


def append_mcp_call(
    tool: str,
    payload_in: Any,
    payload_out: Any,
    *,
    root: Path | str | None = None,
) -> Path | None:
    """Append one compact JSON line under ``.apegmsh/mcp_calls.jsonl``."""
    path = mcp_calls_path(root)
    record = make_call_record(tool, payload_in, payload_out)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        return None
    return path


def logged(
    tool: str,
    fn: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run *fn* then append a ledger line. Never raises from the log.

    When *kwargs* include ``root``, it is passed through to *fn* and used
    as the mcp_calls habitat root (INV-15).
    """
    result = fn(*args, **kwargs)
    payload = kwargs if not args else {"args": list(args), **kwargs}
    append_mcp_call(tool, payload, result, root=kwargs.get("root"))
    return result
