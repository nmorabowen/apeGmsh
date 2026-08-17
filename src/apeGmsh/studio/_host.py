"""Qt host: ModelViewer or MeshViewer + envelope publish on pick (ADR 0095 S2/S3)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from ._envelope import Phase

# QTimer refs so the highlight watch is not garbage-collected.
_HIGHLIGHT_WATCHERS: list[Any] = []


def session_has_mesh() -> bool:
    """True when the live Gmsh kernel already has 1-D/2-D/3-D elements."""
    import gmsh

    try:
        for dim in (3, 2, 1):
            _etypes, etags, _enodes = gmsh.model.mesh.getElements(dim)
            if any(len(t) > 0 for t in etags):
                return True
    except Exception:
        return False
    return False


def resolve_names_from_gmsh(names: list[str]) -> list[tuple[int, int]]:
    """Live-kernel name → dimtag lookup (host only; MCP never imports gmsh)."""
    import gmsh

    from ._names import lookup_from_gmsh

    wanted = set(names)
    if not wanted:
        return []
    hits: list[tuple[int, int]] = []
    try:
        dimtags = gmsh.model.getEntities()
    except Exception:
        return []
    for dim, tag in dimtags:
        rec = lookup_from_gmsh(int(dim), int(tag))
        if wanted.intersection(rec.labels) or wanted.intersection(
            rec.physical_groups
        ):
            hits.append((int(dim), int(tag)))
    return hits


def open_host(
    session: Any,
    *,
    envelope_path: Path,
    title: str | None = None,
    root: Path | str | None = None,
    script: Path | str | None = None,
    runner: Any = None,
) -> None:
    """Open the geometry or mesh viewer and write the envelope on pick.

    MeshViewer when the replayed script already generated elements
    (INV-8: mesh mode once a mesh exists); otherwise ModelViewer.
    The first pick callback (fired at wire time) starts a QTimer that
    applies a sibling ``highlight.json`` through ``select_batch``, and
    installs the manual Refresh action (ADR 0095 S6a) — a toolbar
    button and F5 shortcut on the viewer's window. A leftover
    highlight file from a previous session is ignored until its mtime
    changes.

    ``script`` is the entry script this host was opened with (used as
    the refresh fallback when ``.apegmsh/project.json`` has none yet).
    ``runner`` is the :class:`~apeGmsh.studio._replay.ReplayRunner`
    that performed the initial replay; reusing it lets a refresh with
    an unchanged file skip immediately (same skip-hash cache as the
    open that got here). A fresh one is built when omitted.

    Claims ``.apegmsh/host.json`` for the duration of ``show()`` (S5g).
    """
    from ._envelope import project_state, write_envelope
    from ._host_state import claim_host, clear_host
    from ._names import lookup_from_gmsh
    from ._paths import resolve_root

    phase: Phase = "mesh" if session_has_mesh() else "model"
    watch_started = False
    refresh_installed = False
    request_path = Path(envelope_path).with_name("highlight.json")
    if root is not None:
        habitat = resolve_root(root)
    else:
        env_path = Path(envelope_path).resolve()
        # Production: <root>/.apegmsh/selection.json → parent.parent is root.
        # Tests / odd layouts may pass a bare file next to the project.
        if env_path.parent.name == ".apegmsh":
            habitat = env_path.parent.parent
        else:
            habitat = env_path.parent

    def on_sel(sel: Any) -> None:
        nonlocal watch_started, refresh_installed
        env = project_state(sel, phase=phase, names=lookup_from_gmsh)
        write_envelope(envelope_path, env)
        if not watch_started:
            watch_started = True
            _watch_highlight(sel, request_path)
        if not refresh_installed:
            refresh_installed = True
            win = getattr(viewer, "_win", None)
            if win is not None:
                _install_refresh_action(
                    viewer,
                    win=win,
                    habitat=habitat,
                    script=script,
                    phase=phase,
                    runner=runner,
                )

    if phase == "mesh":
        from apeGmsh.viewers.mesh_viewer import MeshViewer

        viewer: Any = MeshViewer(
            parent=session,
            on_selection_changed=on_sel,
        )
    else:
        from apeGmsh.viewers.model_viewer import ModelViewer

        viewer = ModelViewer(
            parent=session,
            model=session.model,
            on_selection_changed=on_sel,
            annotate=True,
        )
    claim_host(habitat, phase=phase)
    try:
        viewer.show(title=title or "apeGmsh.studio")
    finally:
        clear_host(habitat)


def _install_refresh_action(
    viewer: Any,
    *,
    win: Any,
    habitat: Path,
    script: Path | str | None,
    phase: str,
    runner: Any,
) -> None:
    """Wire the Refresh toolbar action + F5 shortcut (ADR 0095 S6a).

    Thin Qt glue: the decision logic lives in :func:`refresh_host`, so
    this closure only resolves the entry script, picks the
    scene-rebuild callback the open viewer exposes (if any), and
    reflects the outcome in the status bar.
    """
    if runner is None:
        from ._replay import ReplayRunner

        runner = ReplayRunner(root=habitat)

    def _do_refresh() -> None:
        from ._project import OutsideRootError, resolve_entry_script

        try:
            entry = resolve_entry_script(None, root=habitat)
        except OutsideRootError:
            entry = None
        if entry is None and script is not None:
            entry = Path(script)
        if entry is None:
            win.set_status("Refresh: no entry script")
            return
        rebuild = getattr(viewer, "_rebuild_scene", None)
        outcome = refresh_host(runner, entry, phase=phase, on_success=rebuild)
        win.set_status(outcome["message"])

    win.add_toolbar_action("Refresh (F5)", "⟲", _do_refresh)
    win.add_shortcut("F5", _do_refresh, application=True)


def refresh_host(
    runner: Any,
    script: Path,
    *,
    phase: str,
    on_success: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Manual-refresh decision core (ADR 0095 S6a) — no Qt.

    Re-runs *script* at *phase* through ``runner.run_until`` with
    ``trigger="refresh"`` and returns a small status dict for the host
    status bar. Never raises.

    - Busy (INV-18): the runner's existing busy outcome is surfaced
      as-is; *on_success* is not called. No steal logic here — that
      stays inside the runner.
    - Skip-hash (unchanged ``(hash, phase)``): no-op, "Up to date";
      *on_success* is not called, nothing is appended to the ledger.
    - Failure (INV-4): the runner's ``ok: false`` ledger append
      stands; *on_success* is not called, so the previous scene and
      camera stay untouched.
    - Success: *on_success* fires once (the scene-rebuild callback);
      ``names.json`` / ``runs.jsonl`` are already written by
      ``run_until`` itself.
    """
    result = runner.run_until(script, phase=phase, trigger="refresh")
    if not result.ok:
        error = result.error or ""
        if error.startswith("BUSY:"):
            return {
                "status": "busy",
                "message": _format_busy_status(error),
                "result": result,
            }
        tail = error.strip().splitlines()[-1] if error.strip() else "replay failed"
        return {
            "status": "failed",
            "message": f"Refresh failed: {tail}",
            "result": result,
        }
    if result.skipped:
        return {"status": "up_to_date", "message": "Up to date", "result": result}
    if on_success is not None:
        on_success()
    return {"status": "refreshed", "message": "Refreshed", "result": result}


def _format_busy_status(error: str) -> str:
    """``BUSY: habitat locked by pid=123 op=run_until`` -> ``Busy: run_until pid 123``."""
    m = re.search(r"pid=(\S+)\s+op=(\S+)", error)
    if not m:
        return "Busy"
    pid, op = m.group(1), m.group(2)
    return f"Busy: {op} pid {pid}"


def _watch_highlight(sel: Any, path: Path) -> None:
    """Poll highlight.json and apply via owner mutator (INV-7)."""
    from qtpy.QtCore import QTimer

    from ._highlight import (
        apply_highlight_to_state,
        consume_highlight_update,
        seed_highlight_mtime,
    )

    last_mtime = seed_highlight_mtime(path)

    def tick() -> None:
        nonlocal last_mtime
        request, last_mtime = consume_highlight_update(path, last_mtime)
        if request is None:
            return
        apply_highlight_to_state(
            sel,
            names=request["names"],
            resolve=resolve_names_from_gmsh,
        )

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(250)
    _HIGHLIGHT_WATCHERS.append(timer)
