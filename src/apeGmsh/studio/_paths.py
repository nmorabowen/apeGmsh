"""Envelope file location (ADR 0095 open question 1 — decided at S1).

Workspace-relative ``.apegmsh/selection.json`` (cwd). Stable for the
skill: one path regardless of which script is in the host.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_ENVELOPE_REL = Path(".apegmsh") / "selection.json"


def envelope_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/selection.json`` (``root`` defaults to cwd)."""
    base = Path.cwd() if root is None else Path(root)
    return base / DEFAULT_ENVELOPE_REL
