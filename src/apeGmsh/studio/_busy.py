"""``.apegmsh/busy.json`` — habitat concurrency lock (ADR 0095 S5h).

``run_until`` / replay acquires the lock for the duration of an exec.
A second concurrent replay returns ``BUSY`` instead of interleaving
``names.json`` / ``runs.jsonl``. Stale locks (dead PID) are stolen.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._host_state import pid_alive
from ._paths import atomic_write_text, busy_path, resolve_root

BUSY_SCHEMA = 1


class BusyError(RuntimeError):
    """Habitat is locked by another live process."""

    def __init__(self, info: dict[str, Any]) -> None:
        self.info = info
        pid = info.get("pid")
        op = info.get("op") or "run_until"
        super().__init__(f"BUSY: habitat locked by pid={pid} op={op}")


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
    if not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {**empty, "stale": True}
    if not isinstance(data, dict):
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
    return {
        "busy": claimed and (pid is None or pid_alive(pid)),
        "pid": pid if claimed else None,
        "op": data.get("op") if claimed else None,
        "phase": data.get("phase") if claimed else None,
        "script": data.get("script") if claimed else None,
        "stale": False,
    }


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
    (dead-PID) claim is removed and retried once.
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
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    for _attempt in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = read_busy(base)
            if info.get("busy"):
                return False
            # Stale or cleared — remove and retry exclusive create.
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
    """Drop the habitat lock (best-effort)."""
    path = busy_path(root)
    try:
        path.unlink()
    except OSError:
        # Fall back to a clear marker if unlink fails (Windows locks).
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
                        "ts": datetime.now(timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        except OSError:
            pass
