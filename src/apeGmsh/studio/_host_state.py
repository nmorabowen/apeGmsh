"""``.apegmsh/host.json`` — Qt host presence (ADR 0095 S5g / S5j).

The Qt host claims the file on open and clears it when ``show()``
returns. ``status`` re-checks the recorded PID so a crashed host does
not look alive. ``clear_host`` only clears *this* process's claim (S5j).
No MCP verb opens or closes the host (INV-6).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._paths import atomic_write_text, host_path, resolve_root

HOST_SCHEMA = 1

# Windows: OpenProcess fails with ACCESS_DENIED for live processes we
# cannot query; INVALID_PARAMETER for a non-existent PID.
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87


def pid_alive(pid: int) -> bool:
    """True if *pid* appears to be a live process."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetLastError.restype = wintypes.DWORD

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32.SetLastError(0)
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if handle:
            kernel32.CloseHandle(handle)
            return True
        err = int(kernel32.GetLastError())
        if err == _ERROR_ACCESS_DENIED:
            return True
        # INVALID_PARAMETER (87) and friends → treat as dead.
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
) -> Path | None:
    """Mark the Qt host as running under *root* (atomic write).

    If another *live* process already holds the claim, leave it alone
    and return ``None``. ``OSError`` is swallowed — presence is metadata.
    """
    base = resolve_root(root)
    existing = read_host(base)
    if (
        existing.get("running")
        and existing.get("pid") is not None
        and int(existing["pid"]) != int(os.getpid() if pid is None else pid)
        and not existing.get("stale")
    ):
        return None
    payload = {
        "schema": HOST_SCHEMA,
        "running": True,
        "pid": int(os.getpid() if pid is None else pid),
        "phase": phase,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        return atomic_write_text(
            host_path(base),
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
    except OSError:
        return None


def clear_host(root: Path | str | None = None) -> Path | None:
    """Clear *this* process's host claim (no-op for a foreign live PID)."""
    base = resolve_root(root)
    path = host_path(base)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            data = None
        if isinstance(data, dict):
            raw_pid = data.get("pid")
            try:
                owner = int(raw_pid) if raw_pid is not None else None
            except (TypeError, ValueError):
                owner = None
            if owner is not None and owner != os.getpid():
                if pid_alive(owner):
                    return None
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
