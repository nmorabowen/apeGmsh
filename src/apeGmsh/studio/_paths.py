"""Generated studio files (ADR 0095).

Workspace-relative ``.apegmsh/selection.json`` (picks),
``names.json`` (what exists now), ``runs.jsonl`` (run_until
history), ``highlight.json`` (Point-tool request), and ``visors/``
(working stills / clips). Stable for the skill: one path under a
resolved **project root** (INV-15), not "wherever the process cwd
happened to be."
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Mapping

DEFAULT_ENVELOPE_REL = Path(".apegmsh") / "selection.json"
DEFAULT_NAMES_REL = Path(".apegmsh") / "names.json"
DEFAULT_LEDGER_REL = Path(".apegmsh") / "runs.jsonl"
DEFAULT_VISORS_REL = Path(".apegmsh") / "visors"
DEFAULT_HIGHLIGHT_REL = Path(".apegmsh") / "highlight.json"
DEFAULT_MCP_CALLS_REL = Path(".apegmsh") / "mcp_calls.jsonl"
DEFAULT_HOST_REL = Path(".apegmsh") / "host.json"
DEFAULT_PROJECT_REL = Path(".apegmsh") / "project.json"
DEFAULT_BUSY_REL = Path(".apegmsh") / "busy.json"
DEFAULT_ASSESS_REL = Path(".apegmsh") / "assess.json"
DEFAULT_PINS_REL = Path(".apegmsh") / "pins"

ROOT_ENV = "APEGMSH_ROOT"


def display_path(path: Path | str, root: Path | str | None = None) -> str:
    """Return a root-relative posix path when *path* is under *root*.

    Absolute OS paths otherwise. The habitat ``root`` field itself stays
    absolute so consumers have an anchor (ADR 0095 S5i).
    """
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = Path(os.path.abspath(str(p)))
    if root is None:
        return str(p)
    base = Path(root).expanduser()
    try:
        base = base.resolve()
    except OSError:
        base = Path(os.path.abspath(str(base)))
    try:
        return p.relative_to(base).as_posix()
    except ValueError:
        return str(p)


def is_under_root(path: Path | str, root: Path | str | None = None) -> bool:
    """True when *path* resolves under the habitat *root*."""
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = Path(os.path.abspath(str(p)))
    base = resolve_root(root)
    try:
        p.relative_to(base)
        return True
    except ValueError:
        return False


def resolve_under(root: Path | str, path: Path | str) -> Path:
    """Resolve *path* against *root* when relative; absolute paths as-is."""
    base = Path(root).expanduser().resolve()
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write *text* via temp file + ``os.replace`` (ADR 0095 INV-16).

    Same-directory temp so replace is atomic on the habitat volume.
    Readers may still see a missing file mid-replace; they must not see
    a truncated JSON body.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent),
        prefix=f".{dest.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        # Windows: a concurrent reader may briefly deny replace.
        last_err: OSError | None = None
        for attempt in range(20):
            try:
                os.replace(tmp_name, dest)
                last_err = None
                break
            except PermissionError as exc:
                last_err = exc
                time.sleep(0.01 * (attempt + 1))
        if last_err is not None:
            raise last_err
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return dest


def resolve_root(
    explicit: Path | str | None = None,
    *,
    start: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the studio project root (ADR 0095 INV-15).

    Order: explicit ``root=`` / ``--root`` → ``APEGMSH_ROOT`` → nearest
    ancestor of *start* (default: cwd) that contains a ``.apegmsh/``
    directory → cwd (or *start*'s directory when *start* was given and
    no ``.apegmsh/`` was found).

    A ``.apegmsh/`` under the user home directory is ignored unless
    *start* itself is home — otherwise every process inherits a stale
    habitat from ``~/``.

    Cwd is the last fallback, never the definition of the habitat.
    Empty / whitespace-only ``root=`` or ``APEGMSH_ROOT`` is treated as
    unset (S5f) — agents must not silently bind process cwd via ``""``.
    """
    if explicit is not None and str(explicit).strip():
        return Path(str(explicit).strip()).expanduser().resolve()
    environ = os.environ if env is None else env
    raw = environ.get(ROOT_ENV)
    if raw is not None and str(raw).strip():
        return Path(str(raw).strip()).expanduser().resolve()
    origin = Path.cwd() if start is None else Path(start)
    origin = origin.expanduser().resolve()
    if origin.is_file():
        origin = origin.parent
    home = Path.home().resolve()
    for candidate in (origin, *origin.parents):
        if not (candidate / ".apegmsh").is_dir():
            continue
        if candidate == home and origin != home:
            continue
        return candidate
    if start is None:
        return Path.cwd().resolve()
    return origin


def envelope_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/selection.json`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_ENVELOPE_REL


def names_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/names.json`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_NAMES_REL


def ledger_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/runs.jsonl`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_LEDGER_REL


def visors_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/visors`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_VISORS_REL


def highlight_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/highlight.json`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_HIGHLIGHT_REL


def mcp_calls_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/mcp_calls.jsonl`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_MCP_CALLS_REL


def host_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/host.json`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_HOST_REL


def project_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/project.json`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_PROJECT_REL


def busy_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/busy.json`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_BUSY_REL


def assess_path(root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/assess.json`` (``root`` via :func:`resolve_root`)."""
    base = resolve_root(root)
    return base / DEFAULT_ASSESS_REL


def pin_assess_path(pin_id: str, root: Path | None = None) -> Path:
    """Return ``<root>/.apegmsh/pins/<pin_id>/assess.json`` (ADR 0095 Amendment 5).

    Where ``--pin`` copies the live assess snapshot so a later emit
    against that pin keeps the verdict current at pin time.
    """
    base = resolve_root(root)
    return base / DEFAULT_PINS_REL / pin_id / "assess.json"
