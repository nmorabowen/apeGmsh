"""
_Sizing — global & per-entity element size control.

Accessed via ``g.mesh.sizing``.  Controls every size source *except*
Fields — those are owned by the sibling :class:`FieldHelper` at
``g.mesh.field``.
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from .Mesh import Mesh


# Re-export the same type aliases used by the old Mesh.py
Tag      = int
DimTag   = tuple[int, int]
TagsLike = Tag | list[Tag] | DimTag | list[DimTag]


class _Sizing:
    """Global and per-entity element size control."""

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    # ------------------------------------------------------------------
    # Global size
    # ------------------------------------------------------------------

    def set_global_size(
        self,
        max_size: float,
        min_size: float = 0.0,
    ) -> "_Sizing":
        """
        Set the global element-size band.

        Parameters
        ----------
        max_size : upper bound assigned to ``Mesh.MeshSizeMax``.  Acts
                   as a ceiling on every size source.
        min_size : lower bound assigned to ``Mesh.MeshSizeMin``.
                   Defaults to ``0.0`` — i.e. no floor, so per-point
                   refinements (``set_size(..., 100)``) are free to
                   produce elements smaller than ``max_size``.

        Example
        -------
        ::

            m1.mesh.sizing.set_global_size(6000)            # ceiling only
            m1.mesh.sizing.set_global_size(6000, 200)       # band [200, 6000]
        """
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        self._mesh._log(f"set_global_size(max={max_size}, min={min_size})")
        return self

    def set_size_sources(
        self,
        *,
        from_points          : bool | None = None,
        from_curvature       : bool | None = None,
        extend_from_boundary : bool | None = None,
    ) -> "_Sizing":
        """
        Control *which* size sources Gmsh consults when meshing.

        Gmsh combines several size sources at each node and takes the
        minimum.  When ``from_points`` is on (Gmsh default), every
        BRep point carries its own characteristic length — imported
        CAD files typically bake in small ``lc`` values that silently
        override :meth:`set_global_size`.

        Example
        -------
        ::

            (g.mesh.sizing
               .set_size_sources(from_points=False,
                                 from_curvature=False,
                                 extend_from_boundary=False)
               .set_global_size(6000))
        """
        if from_points is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", int(bool(from_points)))
        if from_curvature is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(bool(from_curvature)))
        if extend_from_boundary is not None:
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",
                                  int(bool(extend_from_boundary)))
        self._mesh._log(
            f"set_size_sources(from_points={from_points}, "
            f"from_curvature={from_curvature}, "
            f"extend_from_boundary={extend_from_boundary})"
        )
        return self

    def set_size_global(
        self,
        *,
        min_size: float | None = None,
        max_size: float | None = None,
    ) -> "_Sizing":
        """
        Set the global mesh-size bounds independently.

        Example
        -------
        ::

            g.mesh.sizing.set_size_global(min_size=15, max_size=25)
            g.mesh.sizing.set_size_global(max_size=50)
        """
        if min_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        if max_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        self._mesh._log(f"set_size_global(min={min_size}, max={max_size})")
        return self

    # ------------------------------------------------------------------
    # Per-entity size
    # ------------------------------------------------------------------

    def set_size(
        self,
        tags,
        size: float,
        *,
        dim : int = 0,
    ) -> "_Sizing":
        """
        Assign a target element size to specific points.

        ``tags`` accepts int, str (label/PG), (dim, tag), or a list.

        Example
        -------
        ::

            g.mesh.sizing.set_size([p1, p2, p3], 0.05)
            g.mesh.sizing.set_size("col.start_face", 10, dim=0)
        """
        if isinstance(tags, str):
            from apeGmsh.core._helpers import resolve_to_tags
            resolved = resolve_to_tags(tags, dim=dim, session=self._mesh._parent)
            dimtags = [(dim, t) for t in resolved]
        else:
            dimtags = self._mesh._as_dimtags(tags, dim)
        gmsh.model.mesh.setSize(dimtags, size)
        self._mesh._directives.append({
            'kind': 'set_size', 'dim': dim,
            'tags': [t for _, t in dimtags], 'size': size,
        })
        self._mesh._log(f"set_size(dim={dim}, size={size})")
        return self

    def set_size_all_points(self, size: float) -> "_Sizing":
        """
        Assign the same characteristic length to every BRep point in
        the model.

        Typical use — normalising per-point ``lc`` values after an
        IGES/STEP/DXF import.

        Example
        -------
        ::

            (g.mesh.sizing
               .set_size_all_points(6000)
               .set_global_size(6000))
        """
        pts = gmsh.model.getEntities(dim=0)
        if pts:
            gmsh.model.mesh.setSize(pts, size)
        self._mesh._directives.append({
            'kind': 'set_size_all_points', 'size': size,
            'n_points': len(pts),
        })
        self._mesh._log(f"set_size_all_points(size={size}, n={len(pts)})")
        return self

    def set_size_callback(
        self,
        func: Callable[[float, float, float, float, int, int], float],
    ) -> "_Sizing":
        """
        Register a Python callback that returns the desired element
        size at any point in the model.

        Callback signature ``func(dim, tag, x, y, z, lc) -> float``.
        """
        gmsh.model.mesh.setSizeCallback(func)
        self._mesh._directives.append({
            'kind': 'set_size_callback',
            'func_name': getattr(func, '__name__', '<callable>'),
        })
        self._mesh._log("set_size_callback(<callable>)")
        return self

    def set_size_by_physical(
        self,
        name : str,
        size : float,
        *,
        dim  : int = 0,
    ) -> "_Sizing":
        """
        Apply :meth:`set_size` to every entity in a physical group.

        Only effective for ``dim=0`` (points) — Gmsh ignores
        characteristic lengths set on higher-dimensional entities.
        For surface or volume size control prefer a mesh field
        (``g.mesh.field``).
        """
        tags = self._mesh._resolve_physical(name, dim)
        self._mesh._log(
            f"set_size_by_physical(name={name!r}, dim={dim}, "
            f"tags={tags}, size={size})"
        )
        self.set_size(tags, size, dim=dim)
        return self
