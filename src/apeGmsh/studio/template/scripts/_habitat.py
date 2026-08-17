"""Shared habitat paths for session start/finish scripts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
HABITAT = SCRIPTS.parent
APEGMSH = HABITAT / ".apegmsh"
SESSION_FILE = APEGMSH / "session.json"
MCP_JSON = HABITAT / ".cursor" / "mcp.json"

# Soft contract MUST/SHOULD directories (template seed).
REQUIRED_DIRS = [
    "APE",
    "APE/instructions",
    "APE/memory",
    "APE/skills",
    "APE/libraries",
    "tools",
    "tools/apeCAD",
    "tools/apeSketch",
    "models",
    "reports",
    "reports/technical_briefs",
    "reports/model_ledgers",
    "reports/model_reports",
    "reports/figures",
    "postmortem",
    "postmortem/sessions",
    "postmortem/backlog",
    "postmortem/templates",
    "references",
    "scripts",
]

REQUIRED_FILES = [
    "ape.project.yaml",
    "APE/README.md",
    "APE/instructions/how-we-work.md",
    "APE/instructions/reporting.md",
    "APE/instructions/session-postmortem.md",
    "APE/instructions/studio-mcp.md",
    "APE/memory/brief.md",
    "reports/README.md",
    "postmortem/README.md",
    "postmortem/backlog/open.md",
    "postmortem/templates/01_metrics.md",
    "postmortem/templates/02_friction.md",
    "tools/README.md",
    "scripts/start_session.py",
    "scripts/finish_session.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_pythonpath() -> Path:
    """Prepend apeGmsh src from mcp.json launcher or default worktree."""
    default_src = (
        Path.home()
        / ".cursor"
        / "worktrees"
        / "apeGmsh"
        / "mcp-ledger"
        / "src"
    )
    src = Path(os.environ.get("APEGMSH_SRC", default_src)).expanduser()
    if src.is_dir():
        s = str(src.resolve())
        cur = os.environ.get("PYTHONPATH", "")
        parts = [p for p in cur.split(os.pathsep) if p]
        if s not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([s, *parts])
    os.environ["APEGMSH_STUDIO_ROOT"] = str(HABITAT)
    os.environ.setdefault("APEGMSH_QUIET", "1")
    os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    return src


def read_session() -> dict:
    if SESSION_FILE.is_file():
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    return {}


def write_session(payload: dict) -> None:
    APEGMSH.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def mcp_json_studio_root() -> str | None:
    """Prefer habitat server id, else first APEGMSH_STUDIO_ROOT in mcp.json."""
    if not MCP_JSON.is_file():
        return None
    try:
        data = json.loads(MCP_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    servers = data.get("mcpServers") or {}
    prefer = ("apegmsh-studio-habitat", "apegmsh-studio")
    for name in prefer:
        cfg = servers.get(name) or {}
        env = cfg.get("env") or {}
        root = env.get("APEGMSH_STUDIO_ROOT")
        if root:
            return str(Path(root).resolve())
    for cfg in servers.values():
        env = (cfg or {}).get("env") or {}
        root = env.get("APEGMSH_STUDIO_ROOT")
        if root:
            return str(Path(root).resolve())
    return None
