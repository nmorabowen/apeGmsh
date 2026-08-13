"""
Mesh — top-level meshing composite attached to an ``apeGmsh`` session
as ``g.mesh``.

``g.mesh`` is a thin composition container.  Every action lives in a
focused sub-composite:

* ``g.mesh.generation``   — generate, set_order, refine, optimize,
                            set_algorithm (+ ``_by_physical``)
* ``g.mesh.sizing``       — global / per-entity element size
                            (``set_global_size``, ``set_size``,
                            ``set_size_callback``, …)
* ``g.mesh.field``        — fluent ``FieldHelper`` around
                            ``gmsh.model.mesh.field`` (Distance,
                            Threshold, Box, etc.)
* ``g.mesh.structured``   — transfinite constraints, recombine,
                            smoothing, compound
* ``g.mesh.editing``      — clear, reverse, relocate, duplicate
                            removal, affine transform, embed,
                            periodic, STL -> discrete pipeline
* ``g.mesh.queries``      — get_nodes, get_elements, get_fem_data,
                            quality_report
* ``g.mesh.partitioning`` — partition / unpartition / renumbering
* ``g.mesh.recipe``       — one-call unstructured / structured meshing
                            (ADR 0059) layered over the verbs above

Plus a flat top-level entry point that opens an interactive window:

* ``g.mesh.viewer(**kw)``
* ``g.mesh.render(path)`` — offscreen undeformed mesh still (ADR 0094 S4)

Example
-------
::

    (g.mesh.sizing
       .set_size_sources(from_points=False)
       .set_global_size(6000))
    g.mesh.generation.generate(3)
    g.mesh.partitioning.renumber_mesh(method="rcm", base=1)
    fem = g.mesh.queries.get_fem_data(dim=3)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export algorithm constants for backwards-compatible imports from
# ``apeGmsh.mesh.Mesh`` and from the top-level ``apeGmsh`` package.
from ._mesh_algorithms import (
    ALGORITHM_2D,
    ALGORITHM_3D,
    Algorithm2D,
    Algorithm3D,
    MeshAlgorithm2D,
    MeshAlgorithm3D,
    OptimizeMethod,
)
from ._mesh_editing import _Editing
from ._mesh_field import FieldHelper
from ._mesh_generation import _Generation
from ._mesh_partitioning import _Partitioning
from ._mesh_options import _Options
from ._mesh_queries import _Queries
from ._mesh_recipe import _Recipe
from ._mesh_sizing import _Sizing
from ._mesh_structured import _Structured

if TYPE_CHECKING:
    from pathlib import Path

    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh._types import DimTag, TagsLike


__all__ = [
    "Mesh",
    "Algorithm2D",
    "Algorithm3D",
    "ALGORITHM_2D",
    "ALGORITHM_3D",
    "MeshAlgorithm2D",
    "MeshAlgorithm3D",
    "OptimizeMethod",
]


from apeGmsh._logging import _HasLogging


class Mesh(_HasLogging):
    """
    Thin composition container for meshing.  Every action lives in a
    focused sub-composite — see the module docstring for the full map.

    Parameters
    ----------
    parent : _SessionBase
        Owning session — used to read ``_verbose`` and access the
        ``physical`` composite during ``_by_physical`` helpers.
    """

    _log_prefix = "Mesh"

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent

        # Directive log — records every write-only mesh setting that
        # cannot be read back from gmsh (transfinite, setSize, recombine,
        # fields, per-entity algorithm, smoothing).  Used by
        # ``Inspect.print_summary()`` to show what's been applied.
        self._directives: list[dict] = []

        # gmsh has no getter for the background field — tracked here by
        # FieldHelper.set_background so g.mesh.recipe can fold a
        # user-set background into its Min combiner (ADR 0059 §3).
        self._background_field_tag: int | None = None

        # Sub-composites — each keeps a reference to self.
        self.field        = FieldHelper(self)
        self.generation   = _Generation(self)
        self.sizing       = _Sizing(self)
        self.structured   = _Structured(self)
        self.editing      = _Editing(self)
        self.queries      = _Queries(self)
        self.partitioning = _Partitioning(self)
        self.options      = _Options(self)
        self.recipe       = _Recipe(self)

    # ------------------------------------------------------------------
    # Internal helpers (used by sub-composites via self._mesh._*)
    # ------------------------------------------------------------------


    def _as_dimtags(self, tags: TagsLike, default_dim: int = 0) -> list[DimTag]:
        """Normalize tag input to [(dim, tag), ...]. See :func:`core._helpers.as_dimtags`."""
        from apeGmsh.core._helpers import as_dimtags
        return as_dimtags(tags, default_dim)

    def _resolve_physical(self, name: str, dim: int) -> list[int]:
        """Look up the entity tags behind a physical group name."""
        return self._parent.physical.entities(name, dim=dim)

    # ------------------------------------------------------------------
    # Interactive viewers (flat — single entry points, no sub-composite)
    # ------------------------------------------------------------------

    def viewer(self, **kwargs):
        """Open the interactive mesh viewer.

        The viewer supports picking (BRep entities, elements, nodes),
        color modes, and load/constraint/mass overlays.

        Parameters are forwarded to :class:`MeshViewer`.
        """
        from ..viewers.mesh_viewer import MeshViewer
        mv = MeshViewer(self._parent, **kwargs)
        return mv.show()

    def render(
        self,
        path: "str | Path",
        *,
        camera: str = "iso",
        window_size: tuple[int, int] = (1280, 720),
    ) -> "Path | None":
        """Write one offscreen undeformed mesh still (ADR 0094 S4).

        Live session only. Uses ``build_mesh_scene`` (the same live
        Gmsh scene as :meth:`viewer`), not ``get_fem_data`` +
        ``build_fem_scene``. VTK offscreen — no Qt window, no event
        loop.

        Returns the written :class:`~pathlib.Path`, or ``None`` (and
        prints the ``[skip viewer]`` notice) under
        ``APEGMSH_SKIP_VIEWER=1`` or with no GL. Raises
        ``RuntimeError`` on a ``from_h5`` / closed session, or when
        no mesh has been generated. Snapshot stills stay on
        ``fem.render``.
        """
        from apeGmsh.viewers.render import render_mesh

        return render_mesh(
            self._parent, path, camera=camera, window_size=window_size,
        )

    def preview(
        self,
        *,
        dims: list[int] | None = None,
        show_nodes: bool = True,
        browser: bool = False,
        return_fig: bool = False,
    ):
        """Interactive WebGL preview of the mesh.

        Zero Qt dependency — works inline in Jupyter / VS Code / Colab,
        or in a dedicated browser tab when ``browser=True``. Hover over
        an element to see its BRep ``dim`` and ``tag``; hover over a
        node to see its ``node=N`` id. Single-click a legend entry to
        hide a trace, double-click to isolate it.

        Parameters
        ----------
        dims : list of int, optional
            Mesh dimensions to render. Defaults to ``[1, 2, 3]`` —
            surface / volume / 1D curve elements.
        show_nodes : bool
            Render the full mesh-node cloud as a separate trace
            (default ``True``). Matches the Qt mesh viewer, which
            always shows the node cloud. Disable for very large meshes
            where the nodes overwhelm the element rendering.
        browser : bool
            If ``True``, open in a new browser tab (temp HTML file)
            instead of rendering inline.
        return_fig : bool
            If ``True``, skip display and return the raw
            :class:`plotly.graph_objects.Figure`.
        """
        from ..viz.NotebookPreview import preview_mesh
        return preview_mesh(
            self._parent,
            dims=dims,
            show_nodes=show_nodes,
            browser=browser,
            return_fig=return_fig,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Mesh(parent={self._parent.name!r}, directives={len(self._directives)})"
