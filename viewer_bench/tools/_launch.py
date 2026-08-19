"""Shared helpers for tools/*/open_interface.py launchers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
HABITAT = TOOLS.parent
HOME = Path.home()
GITHUB = Path(os.environ.get("APE_GITHUB", HOME / "Documents" / "Github"))


def ensure_src(env_var: str, repo_name: str) -> Path:
    """Put package ``src`` on sys.path; return that src path."""
    override = os.environ.get(env_var)
    if override:
        src = Path(override).expanduser().resolve()
    else:
        src = (GITHUB / repo_name / "src").resolve()
    if not src.is_dir():
        raise SystemExit(
            f"Missing {repo_name} sources at {src}. "
            f"Clone under {GITHUB / repo_name} or set {env_var} to its src/."
        )
    inserted = str(src)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    return src


def sketch_dirs(tool_dir: Path) -> tuple[Path, Path]:
    human = tool_dir / "human_sketches"
    agentic = tool_dir / "agentic_sketches"
    human.mkdir(parents=True, exist_ok=True)
    agentic.mkdir(parents=True, exist_ok=True)
    return human, agentic


def role_root(tool_dir: Path, role: str) -> Path:
    human, agentic = sketch_dirs(tool_dir)
    if role == "human":
        return human
    if role == "agentic":
        return agentic
    raise SystemExit("--role must be 'human' or 'agentic'")
