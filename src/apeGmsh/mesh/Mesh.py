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

Plus two flat top-level entry points that open interactive windows:

* ``g.mesh.viewer(**kw)``
* ``g.mesh.results_viewer(results=..., point_data=..., cell_data=...)``

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
from ._mesh_queries import _Queries
from ._mesh_sizing import _Sizing
from ._mesh_structured import _Structured

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh._types import Tag, DimTag, TagsLike


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

        # Sub-composites — each keeps a reference to self.
        self.field        = FieldHelper(self)
        self.generation   = _Generation(self)
        self.sizing       = _Sizing(self)
        self.structured   = _Structured(self)
        self.editing      = _Editing(self)
        self.queries      = _Queries(self)
        self.partitioning = _Partitioning(self)

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

    def _get_raw_fem_data(self, dim: int = 2) -> dict:
        """
        Internal helper — extracts raw FEM data as a plain dict.

        .. deprecated::
            No longer called internally.  Use
            ``_fem_extract.extract_raw()`` directly.  Kept for backward
            compatibility — will be removed in a future release.
        """
        import gmsh
        import numpy as np
        from numpy import ndarray

        # --- nodes (full mesh) ---
        nodes       = self.queries.get_nodes()
        node_tags   = nodes['tags']
        node_coords = nodes['coords']

        # --- elements of requested dimension ---
        elems = self.queries.get_elements(dim=dim)

        conn_blocks: list[ndarray] = []
        elem_tags: list[int] = []
        elem_type_codes: list[int] = []
        elem_type_info: dict[int, tuple] = {}
        for etype, etags, enodes in zip(
            elems['types'], elems['tags'], elems['node_tags']
        ):
            props  = self.queries.get_element_properties(etype)
            npe    = props['n_nodes']
            conn_blocks.append(enodes.reshape(-1, npe).astype(int))
            n_this = len(etags)
            elem_tags.extend(etags.astype(int).tolist())
            elem_type_codes.extend([int(etype)] * n_this)
            elem_type_info[int(etype)] = (
                props['name'], props['dim'], props['n_nodes'],
            )

        connectivity = np.vstack(conn_blocks) if conn_blocks else np.empty(
            (0, 0), dtype=int
        )

        # --- used_tags from ALL dimensions (not just target dim) ---
        # Nodes on lower-dim entities (columns, supports) are connected
        # to line/point elements even when they don't appear in the
        # target-dim connectivity.
        _, _, all_node_tags = gmsh.model.mesh.getElements(dim=-1, tag=-1)
        used_tags: set[int] = set()
        for enodes in all_node_tags:
            used_tags.update(int(n) for n in enodes)

        return {
            'node_tags'      : node_tags,
            'node_coords'    : node_coords,
            'connectivity'   : connectivity,
            'elem_tags'      : elem_tags,
            'elem_type_codes': elem_type_codes,
            'elem_type_info' : elem_type_info,
            'used_tags'      : used_tags,
        }

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

    def results_viewer(
        self,
        results: str | None = None,
        *,
        point_data: dict | None = None,
        cell_data: dict | None = None,
        blocking: bool = False,
    ) -> None:
        """Open the results viewer (apeGmshViewer).

        Parameters
        ----------
        results : str, optional
            Path to a ``.vtu``, ``.vtk``, or ``.pvd`` file.
        point_data : dict, optional
            Nodal fields as numpy arrays: ``{name: ndarray}``.
        cell_data : dict, optional
            Element fields as numpy arrays: ``{name: ndarray}``.
        blocking : bool
            If False (default), the viewer runs non-blocking.
        """
        if results is not None:
            from apeGmshViewer import show
            show(results, blocking=blocking)
        elif point_data is not None or cell_data is not None:
            from ..results.Results import Results
            fem = self.queries.get_fem_data()
            r = Results.from_fem(
                fem,
                point_data=point_data,
                cell_data=cell_data,
                name=self._parent.name,
            )
            r.viewer(blocking=blocking)
        else:
            import tempfile

            from apeGmshViewer import show

            from ..viz.VTKExport import VTKExport
            vtk_export = VTKExport(self._parent)
            tmp = tempfile.NamedTemporaryFile(suffix=".vtu", delete=False)
            vtk_export.write(tmp.name)
            tmp.close()
            show(tmp.name, blocking=blocking)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Mesh(parent={self._parent.name!r}, directives={len(self._directives)})"
