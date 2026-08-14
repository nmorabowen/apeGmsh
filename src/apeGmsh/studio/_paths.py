"""Generated studio files (ADR 0095).

Workspace-relative ``.apegmsh/selection.json`` (picks),
``names.json`` (what exists now), ``runs.jsonl`` (run_until
history), ``highlight.json`` (Point-tool request), and ``visors/``
(working stills / clips). Stable for the skill: one path regardless
of which script is in the host.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_ENVELOPE_REL = Path(".apegmsh") / "selection.json"
DEFAULT_NAMES_REL = Path(".apegmsh") / "names.json"
DEFAULT_LEDGER_REL = Path(".apegmsh") / "runs.jsonl"
DEFAULT_VISORS_REL = Path(".apegmsh") / "visors"
DEFAULT_HIGHLIGHT_REL = Path(".apegmsh") / "highlight.json"
DEFAULT_MCP_CALLS_REL = Path(".apegmsh") / "mcp_calls.jsonl"


def envelope_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/selection.json`` (``root`` defaults to cwd)."""
    base = Path.cwd() if root is None else Path(root)
    return base / DEFAULT_ENVELOPE_REL


def names_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/names.json`` (``root`` defaults to cwd)."""
    base = Path.cwd() if root is None else Path(root)
    return base / DEFAULT_NAMES_REL


def ledger_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/runs.jsonl`` (``root`` defaults to cwd)."""
    base = Path.cwd() if root is None else Path(root)
    return base / DEFAULT_LEDGER_REL


def visors_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/visors`` (``root`` defaults to cwd)."""
    base = Path.cwd() if root is None else Path(root)
    return base / DEFAULT_VISORS_REL


def highlight_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/highlight.json`` (``root`` defaults to cwd)."""
    base = Path.cwd() if root is None else Path(root)
    return base / DEFAULT_HIGHLIGHT_REL


def mcp_calls_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/mcp_calls.jsonl`` (``root`` defaults to cwd)."""
    base = Path.cwd() if root is None else Path(root)
    return base / DEFAULT_MCP_CALLS_REL
