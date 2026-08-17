"""Shared habitat paths for session start/finish scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
    "APE/instructions/checkpoints.md",
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
    "scripts/run_case.py",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _apegmsh_src() -> Path | None:
    """Where apeGmsh lives, for this process and for the subprocesses it
    spawns. Probe order, first hit wins:

    1. an already-importable apeGmsh — the honest answer, and it already
       reflects any PYTHONPATH the launcher pinned;
    2. ``APEGMSH_SRC`` — the explicit override for a checkout that is not
       installed anywhere;
    3. nothing. A habitat is cloned between machines, so there is no
       default checkout path worth guessing.
    """
    try:
        import apeGmsh

        return Path(apeGmsh.__file__).resolve().parent.parent
    except Exception:  # noqa: BLE001 - any import failure means "not here"
        pass
    override = os.environ.get("APEGMSH_SRC")
    if override:
        src = Path(override).expanduser()
        if src.is_dir():
            return src.resolve()
    return None


def ensure_pythonpath() -> Path | None:
    """Put apeGmsh src on this process's ``sys.path`` and on the
    ``PYTHONPATH`` its subprocesses inherit."""
    src = _apegmsh_src()
    if src is not None:
        s = str(src)
        cur = os.environ.get("PYTHONPATH", "")
        parts = [p for p in cur.split(os.pathsep) if p]
        if s not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([s, *parts])
        if s not in sys.path:
            sys.path.insert(0, s)
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


def git_info(path: Path = HABITAT) -> dict | None:
    """Read-only git snapshot: branch, HEAD sha, dirty file count.
    None without git on PATH, outside a repo, or before the first
    commit. Never writes (INV-24, ADR 0095 Amendment 8)."""
    if shutil.which("git") is None:
        return None

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True
        )

    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return None
    branch = run("branch", "--show-current").stdout.strip() or "(detached)"
    status = run("status", "--porcelain")
    dirty = [ln for ln in status.stdout.splitlines() if ln.strip()]
    return {"branch": branch, "sha": head.stdout.strip(), "dirty_files": len(dirty)}


def git_provenance(path: Path = HABITAT) -> dict | None:
    """The ``run.json`` provenance fields (models/README.md contract):
    which source produced a case. None when un-versioned — omit the
    fields rather than writing null."""
    info = git_info(path)
    if info is None:
        return None
    return {"model_sha": info["sha"], "git_dirty": info["dirty_files"] > 0}


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
