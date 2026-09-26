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
# GetExitCodeProcess reports STILL_ACTIVE until the process exits.
_STILL_ACTIVE = 259
# Largest value either platform treats as a PID (a Windows DWORD; POSIX
# pid_t is narrower still). A bigger number came from a corrupt file, and
# ctypes would silently wrap it onto an unrelated PID.
_PID_MAX = 0xFFFFFFFF

_kernel32: Any = None


def _win_kernel32() -> Any:
    """A private ``kernel32`` loaded with ``use_last_error=True``.

    Not ``ctypes.windll.kernel32``: that object is shared by the whole
    process, so its ``argtypes`` are not ours to set, and a raw
    ``GetLastError`` call can read an error left by ctypes' own
    intervening calls instead of the one ``OpenProcess`` set.
    """
    global _kernel32
    if _kernel32 is None:
        import ctypes
        from ctypes import wintypes

        lib = ctypes.WinDLL(  # type: ignore[attr-defined]
            "kernel32", use_last_error=True
        )
        lib.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        lib.OpenProcess.restype = wintypes.HANDLE
        lib.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        lib.GetExitCodeProcess.restype = wintypes.BOOL
        lib.CloseHandle.argtypes = [wintypes.HANDLE]
        lib.CloseHandle.restype = wintypes.BOOL
        _kernel32 = lib
    return _kernel32


def pid_alive(pid: int) -> bool:
    """True if *pid* appears to be a live process."""
    if pid <= 0 or pid > _PID_MAX:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = _win_kernel32()
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            err = ctypes.get_last_error()  # type: ignore[attr-defined]
            if err == _ERROR_ACCESS_DENIED:
                return True
            # INVALID_PARAMETER (87) and friends → treat as dead.
            return False
        # An exited process whose handle someone still holds (a parent's
        # ``Popen``) can still be opened, so the exit code decides. One
        # that exited with code 259 (STILL_ACTIVE) still reads as alive.
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
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
