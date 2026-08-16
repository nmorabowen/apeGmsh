"""``.apegmsh/project.json`` — entry script for the habitat (ADR 0095 S5g).

Written on successful ``run_until`` / replay. Out-of-process consumers
(and MCP ``run_until`` with an empty script) read it so agents need not
re-invent the entry path every turn.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._paths import (
    atomic_write_text,
    is_under_root,
    project_path,
    resolve_root,
)

PROJECT_SCHEMA = 1


class OutsideRootError(ValueError):
    """``project.json`` entry script lies outside the habitat root."""


def read_project(root: Path | str | None = None) -> dict[str, Any] | None:
    """Load project.json. Missing / unreadable → ``None``."""
    path = project_path(root)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_project(
    root: Path | str | None,
    *,
    script: Path | str,
) -> Path:
    """Atomically write the entry-script pointer under the habitat root."""
    base = resolve_root(root)
    script_path = Path(script).expanduser().resolve()
    try:
        rel = script_path.relative_to(base)
        script_field = rel.as_posix()
    except ValueError:
        script_field = str(script_path)
    payload = {
        "schema": PROJECT_SCHEMA,
        "script": script_field,
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return atomic_write_text(
        project_path(base),
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def resolve_entry_script(
    script: Path | str | None,
    *,
    root: Path | str | None = None,
) -> Path | None:
    """Resolve an explicit script, or fall back to ``project.json``.

    Returns ``None`` when neither is available. Raises
    :class:`OutsideRootError` when the *project.json* default resolves
    outside the habitat root (S5j) — explicit ``script=`` may still
    point outside.
    """
    base = resolve_root(root)
    if script is not None and str(script).strip():
        p = Path(str(script).strip())
        if not p.is_absolute():
            return (base / p).resolve()
        return p.expanduser().resolve()
    proj = read_project(base)
    if proj is None:
        return None
    raw = proj.get("script")
    if raw is None or not str(raw).strip():
        return None
    p = Path(str(raw).strip())
    if not p.is_absolute():
        resolved = (base / p).resolve()
    else:
        resolved = p.expanduser().resolve()
    if not is_under_root(resolved, base):
        raise OutsideRootError(
            f"project.json script lies outside habitat root: {resolved}"
        )
    return resolved
