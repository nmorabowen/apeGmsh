"""
glyph_helpers — Shared glyph rebuild functions.
=================================================

Point entities (ModelViewer) and node clouds (MeshViewer) are both
rendered as sphere glyphs with a fixed radius.  ``SetPointSize()``
does nothing on glyphs — the only way to resize is to rebuild.

These helpers encapsulate the remove-old / build-new / swap pattern
so both viewers call the same code.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def rebuild_brep_point_glyphs(
    plotter: Any,
    registry: Any,
) -> None:
    """Rebuild dim=0 BRep point glyphs from stored kwargs.

    Reads ``_add_mesh_kwargs[0]`` for centers, tags, diagonal, and
    the current ``point_size``.  Removes the old actor, builds new
    glyphs, and swaps via ``registry.swap_dim()``.

    Call this from the ModelViewer's ``on_point_size`` callback after
    updating ``point_size`` in the stored kwargs.
    """
    from ..scene.glyph_points import build_point_glyphs
    from ..core.color_manager import IDLE_COLORS

    if 0 not in registry.dim_actors:
        return

    kw = registry._add_mesh_kwargs.get(0, {})
    centers = kw.get('_centers_d0')
    tags = kw.get('_tags_d0', [])
    diag = kw.get('model_diagonal', 1.0)
    size = kw.get('point_size', 10.0)

    if centers is None or not len(tags):
        return

    # Remove old
    old_actor = registry.dim_actors[0]
    try:
        plotter.remove_actor(old_actor)
    except Exception:
        pass

    # Build new
    new_mesh, new_actor, _, _ = build_point_glyphs(
        plotter, centers, tags,
        model_diagonal=diag,
        point_size=size,
        idle_color=np.array(IDLE_COLORS[0], dtype=np.uint8),
    )
    registry.swap_dim(0, new_mesh, new_actor)


def rebuild_node_cloud(
    plotter: Any,
    scene: Any,
    new_size: float,
) -> None:
    """Rebuild the mesh-viewer node cloud with a new marker size.

    Parameters
    ----------
    plotter : pv.Plotter
    scene : MeshSceneData
        Must have ``.node_cloud``, ``.node_actor``, ``.model_diagonal``.
    new_size : float
        New marker size value (passed to ``build_node_cloud``).
    """
    from ..scene.glyph_points import build_node_cloud

    if scene.node_cloud is None or scene.node_cloud.n_points == 0:
        return

    if scene.node_actor is not None:
        try:
            plotter.remove_actor(scene.node_actor)
        except Exception:
            pass

    new_cloud, new_actor = build_node_cloud(
        plotter, scene.node_cloud.points,
        model_diagonal=scene.model_diagonal,
        marker_size=new_size,
    )
    scene.node_cloud = new_cloud
    scene.node_actor = new_actor
