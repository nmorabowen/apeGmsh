"""Qt host: ModelViewer or MeshViewer + envelope publish on pick (ADR 0095 S2/S3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
) -> None:
    """Open the geometry or mesh viewer and write the envelope on pick.

    MeshViewer when the replayed script already generated elements
    (INV-8: mesh mode once a mesh exists); otherwise ModelViewer.
    The first pick callback (fired at wire time) starts a QTimer that
    applies a sibling ``highlight.json`` through ``select_batch``.
    A leftover file from a previous session is ignored until its mtime
    changes.
    """
    from ._envelope import project_state, write_envelope
    from ._names import lookup_from_gmsh

    phase: Phase = "mesh" if session_has_mesh() else "model"
    watch_started = False
    request_path = Path(envelope_path).with_name("highlight.json")

    def on_sel(sel: Any) -> None:
        nonlocal watch_started
        env = project_state(sel, phase=phase, names=lookup_from_gmsh)
        write_envelope(envelope_path, env)
        if not watch_started:
            watch_started = True
            _watch_highlight(sel, request_path)

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
        )
    viewer.show(title=title or "apeGmsh.studio")


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
