"""``.apegmsh/host.json`` — Qt host presence (ADR 0095 S5g).

The Qt host claims the file on open and clears it when ``show()``
returns. ``status`` re-checks the recorded PID so a crashed host does
not look alive. No MCP verb opens or closes the host (INV-6).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._paths import atomic_write_text, host_path, resolve_root

HOST_SCHEMA = 1


def pid_alive(pid: int) -> bool:
    """True if *pid* appears to be a live process."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def claim_host(
    root: Path | str | None,
    *,
    phase: str,
    pid: int | None = None,
) -> Path:
    """Mark the Qt host as running under *root* (atomic write)."""
    base = resolve_root(root)
    payload = {
        "schema": HOST_SCHEMA,
        "running": True,
        "pid": int(os.getpid() if pid is None else pid),
        "phase": phase,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return atomic_write_text(
        host_path(base),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def clear_host(root: Path | str | None = None) -> Path | None:
    """Mark the host as not running (or remove a torn file)."""
    base = resolve_root(root)
    path = host_path(base)
    payload = {
        "schema": HOST_SCHEMA,
        "running": False,
        "pid": None,
        "phase": None,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        return atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
    except OSError:
        return None


def read_host(root: Path | str | None = None) -> dict[str, Any]:
    """Host block for ``status``. Always returns a dict; never raises.

    If the file says ``running`` but the PID is dead, report
    ``running=False`` (stale claim after a crash).
    """
    path = host_path(root)
    empty = {
        "running": False,
        "pid": None,
        "phase": None,
        "stale": False,
    }
    if not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {**empty, "stale": True}
    if not isinstance(data, dict):
        return {**empty, "stale": True}
    claimed = bool(data.get("running"))
    raw_pid = data.get("pid")
    try:
        pid = int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        pid = None
    phase = data.get("phase")
    if phase is not None:
        phase = str(phase)
    if claimed and pid is not None and not pid_alive(pid):
        return {
            "running": False,
            "pid": pid,
            "phase": phase,
            "stale": True,
        }
    return {
        "running": claimed and (pid is None or pid_alive(pid)),
        "pid": pid,
        "phase": phase if claimed else None,
        "stale": False,
    }
