"""Qt host: ModelViewer or MeshViewer + envelope publish on pick (ADR 0095 S2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._envelope import Phase


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


def open_host(
    session: Any,
    *,
    envelope_path: Path,
    title: str | None = None,
) -> None:
    """Open the geometry or mesh viewer and write the envelope on pick.

    MeshViewer when the replayed script already generated elements
    (INV-8: mesh mode once a mesh exists); otherwise ModelViewer.
    """
    from ._envelope import project_state, write_envelope
    from ._names import lookup_from_gmsh

    phase: Phase = "mesh" if session_has_mesh() else "model"

    def on_sel(sel: Any) -> None:
        env = project_state(sel, phase=phase, names=lookup_from_gmsh)
        write_envelope(envelope_path, env)

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
