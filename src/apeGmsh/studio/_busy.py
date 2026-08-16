"""``.apegmsh/busy.json`` — habitat concurrency lock (ADR 0095 S5h / S5j).

``run_until`` / replay acquires the lock for the duration of an exec.
A second concurrent replay returns ``BUSY`` instead of interleaving
``names.json`` / ``runs.jsonl``. Stale locks (dead PID) are stolen;
empty / unparseable mid-write files are *not* stolen until a short
grace window elapses (S5j).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._host_state import pid_alive
from ._paths import atomic_write_text, busy_path, resolve_root

BUSY_SCHEMA = 1
# Do not steal empty/torn busy.json during another process's write.
_TORN_GRACE_S = 2.0


class BusyError(RuntimeError):
    """Habitat is locked by another live process."""

    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info
        pid = info.get("pid")
        op = info.get("op") or "run_until"
        super().__init__(f"BUSY: habitat locked by pid={pid} op={op}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_busy_raw(path: Path) -> dict[str, Any] | None:
    """Parse busy.json. ``None`` = missing; empty dict = torn/unparseable."""
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return {}
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_busy(root: Path | str | None = None) -> dict[str, Any]:
    """Busy block for ``status``. Never raises."""
    empty = {
        "busy": False,
        "pid": None,
        "op": None,
        "phase": None,
        "script": None,
        "stale": False,
    }
    path = busy_path(root)
    data = _load_busy_raw(path)
    if data is None:
        return empty
    if not data:
        return {**empty, "stale": True}
    claimed = bool(data.get("busy"))
    raw_pid = data.get("pid")
    try:
        pid = int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        pid = None
    if claimed and pid is not None and not pid_alive(pid):
        return {
            "busy": False,
            "pid": pid,
            "op": data.get("op"),
            "phase": data.get("phase"),
            "script": data.get("script"),
            "stale": True,
        }
    if claimed and data.get("busy") is False:
        return empty
    return {
        "busy": claimed and (pid is None or pid_alive(pid)),
        "pid": pid if claimed else None,
        "op": data.get("op") if claimed else None,
        "phase": data.get("phase") if claimed else None,
        "script": data.get("script") if claimed else None,
        "stale": False,
    }


def _may_steal(path: Path, data: dict[str, Any] | None) -> bool:
    """True when an existing busy.json may be removed and reclaimed."""
    if data is None:
        return True
    if not data:
        # Torn / empty mid-write — wait for grace before stealing.
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return False
        return age >= _TORN_GRACE_S
    if data.get("busy") is False:
        return True
    raw_pid = data.get("pid")
    try:
        pid = int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        pid = None
    if pid is None:
        return True
    return not pid_alive(pid)


def try_acquire_busy(
    root: Path | str | None,
    *,
    op: str = "run_until",
    phase: str | None = None,
    script: Path | str | None = None,
    pid: int | None = None,
) -> bool:
    """Claim the habitat lock. ``True`` if acquired; ``False`` if busy.

    Uses ``O_CREAT|O_EXCL`` so two processes cannot both win. A stale
    (dead-PID) or cleared claim is removed and retried. Empty mid-write
    files are only stolen after :data:`_TORN_GRACE_S`.
    """
    base = resolve_root(root)
    path = busy_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BUSY_SCHEMA,
        "busy": True,
        "pid": int(os.getpid() if pid is None else pid),
        "op": op,
        "phase": phase,
        "script": str(script) if script is not None else None,
        "ts": _utc_now(),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    for _attempt in range(3):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            data = _load_busy_raw(path)
            info = read_busy(base)
            if info.get("busy"):
                return False
            if not _may_steal(path, data):
                return False
            # Cleared marker: overwrite in place when unlink is denied.
            if data is not None and data.get("busy") is False:
                try:
                    atomic_write_text(path, text)
                    return True
                except OSError:
                    return False
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                return False
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                path.unlink()
            except OSError:
                pass
            raise
        return True
    return False


def release_busy(root: Path | str | None = None) -> None:
    """Drop *this* process's habitat lock (best-effort)."""
    path = busy_path(root)
    data = _load_busy_raw(path)
    if data is None:
        return
    raw_pid = data.get("pid") if data else None
    try:
        owner = int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        owner = None
    if owner is not None and owner != os.getpid():
        return
    try:
        path.unlink()
    except OSError:
        try:
            atomic_write_text(
                path,
                json.dumps(
                    {
                        "schema": BUSY_SCHEMA,
                        "busy": False,
                        "pid": None,
                        "op": None,
                        "phase": None,
                        "script": None,
                        "ts": _utc_now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        except OSError:
            pass
