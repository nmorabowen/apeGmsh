from __future__ import annotations

import math
from enum import IntEnum
from typing import Callable, TYPE_CHECKING

import gmsh
import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase

# ---------------------------------------------------------------------------
# Type aliases  (mirror Model.py for consistency)
# ---------------------------------------------------------------------------
Tag      = int
DimTag   = tuple[int, int]
TagsLike = Tag | list[Tag] | DimTag | list[DimTag]


# ---------------------------------------------------------------------------
# Algorithm constants
# ---------------------------------------------------------------------------

class Algorithm2D(IntEnum):
    """2-D meshing algorithm selector — use with ``Mesh.set_algorithm``."""
    MESH_ADAPT             = 1
    AUTOMATIC              = 2
    INITIAL_MESH_ONLY      = 3
    DELAUNAY               = 5
    FRONTAL_DELAUNAY       = 6
    BAMG                   = 7
    FRONTAL_DELAUNAY_QUADS = 8
    PACKING_PARALLELOGRAMS = 9
    QUASI_STRUCTURED_QUAD  = 11


class Algorithm3D(IntEnum):
    """3-D meshing algorithm selector — use with ``Mesh.set_algorithm``."""
    DELAUNAY          = 1
    INITIAL_MESH_ONLY = 3
    FRONTAL           = 4
    MMG3D             = 7
    R_TREE            = 9
    HXT               = 10


class OptimizeMethod:
    """Mesh optimisation method names — use with ``Mesh.optimize``."""
    DEFAULT                    = ""
    NETGEN                     = "Netgen"
    HIGH_ORDER                 = "HighOrder"
    HIGH_ORDER_ELASTIC         = "HighOrderElastic"
    HIGH_ORDER_FAST_CURVING    = "HighOrderFastCurving"
    LAPLACE_2D                 = "Laplace2D"
    RELOCATE_2D                = "Relocate2D"
    RELOCATE_3D                = "Relocate3D"
    QUAD_QUASI_STRUCTURED      = "QuadQuasiStructured"
    UNTANGLE_MESH_GEOMETRY     = "UntangleMeshGeometry"


from ._mesh_field import FieldHelper


# ---------------------------------------------------------------------------
# Mesh  — main composite class
# ---------------------------------------------------------------------------

class Mesh:
    """
    Meshing composite attached to a ``pyGmsh`` instance as ``self.mesh``.

    Wraps ``gmsh.model.mesh`` with a clean, method-chaining API organised
    into logical sections:

    * **Generation**      — ``generate``, ``set_order``, ``refine``,
      ``optimize``
    * **Size control**    — ``set_global_size``, ``set_size``,
      ``set_size_callback``; and ``self.field`` (``FieldHelper``)
    * **Structured**      — ``set_transfinite_curve``,
      ``set_transfinite_surface``, ``set_transfinite_volume``,
      ``set_transfinite_automatic``
    * **Quad / recombine**— ``set_recombine``, ``recombine``,
      ``set_smoothing``, ``set_algorithm``
    * **Embedding**       — ``embed``
    * **Periodicity**     — ``set_periodic``
    * **STL / discrete**  — ``import_stl``, ``classify_surfaces``,
      ``create_geometry``
    * **Editing**         — ``clear``, ``reverse``, ``relocate_nodes``,
      ``remove_duplicate_nodes``, ``remove_duplicate_elements``,
      ``affine_transform``
    * **Partitioning**    — ``partition``, ``unpartition``,
      ``compute_renumbering``, ``renumber_nodes``, ``renumber_elements``,
      ``renumber_mesh``
    * **Queries**         — ``get_nodes``, ``get_elements``,
      ``get_fem_data``, ``get_element_qualities``,
      ``get_element_properties``, ``quality_report``
    * **IO**              — ``save``

    All methods that do not return data return ``self`` so they can be
    chained::

        (g.mesh
           .set_global_size(0.5)
           .generate(3)
           .optimize(OptimizeMethod.NETGEN, niter=3)
           .save("result.msh"))

    Parameters
    ----------
    parent : _SessionBase
        Owning instance — used to read ``_verbose``.
    """

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent
        self.field   = FieldHelper(self)

        # Directive log — records every write-only mesh setting that
        # cannot be read back from gmsh (transfinite, setSize, recombine,
        # fields, per-entity algorithm, smoothing).  Used by
        # ``Inspect.print_summary()`` to show what's been applied.
        self._directives: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[Mesh] {msg}")

    def _as_dimtags(self, tags: TagsLike, default_dim: int = 0) -> list[DimTag]:
        """Normalize tag input to [(dim, tag), ...]. See :func:`core._helpers.as_dimtags`."""
        from pyGmsh.core._helpers import as_dimtags
        return as_dimtags(tags, default_dim)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, dim: int = 3) -> Mesh:
        """
        Generate a mesh up to the given dimension.

        Parameters
        ----------
        dim : 1 = edges only, 2 = surface mesh, 3 = volume mesh (default)
        """
        gmsh.model.mesh.generate(dim)
        self._log(f"generate(dim={dim})")
        return self

    def set_order(self, order: int) -> Mesh:
        """
        Elevate elements to high order.

        Parameters
        ----------
        order : 1 = linear, 2 = quadratic, 3 = cubic, …
        """
        gmsh.model.mesh.setOrder(order)
        self._log(f"set_order({order})")
        return self

    def refine(self) -> Mesh:
        """Uniformly refine by splitting every element once."""
        gmsh.model.mesh.refine()
        self._log("refine()")
        return self

    def optimize(
        self,
        method  : str  = OptimizeMethod.DEFAULT,
        *,
        force   : bool = False,
        niter   : int  = 1,
        dim_tags: list[DimTag] | None = None,
    ) -> Mesh:
        """
        Optimise mesh quality.

        Parameters
        ----------
        method   : one of ``OptimizeMethod.*`` constants
        force    : apply optimisation even to already-valid elements
        niter    : number of passes
        dim_tags : limit to specific entities (``None`` = all)

        Example
        -------
        ::

            g.mesh.generate(3).optimize(OptimizeMethod.NETGEN, niter=5)
        """
        gmsh.model.mesh.optimize(method, force=force, niter=niter,
                                  dimTags=dim_tags or [])
        self._log(f"optimize(method={method!r}, niter={niter})")
        return self

    # ------------------------------------------------------------------
    # Global / point size control
    # ------------------------------------------------------------------

    def set_global_size(
        self,
        max_size: float,
        min_size: float = 0.0,
    ) -> Mesh:
        """
        Set the global element-size band.

        Parameters
        ----------
        max_size : upper bound assigned to ``Mesh.MeshSizeMax``.  Acts as a
                   ceiling on every size source (per-point ``lc`` values,
                   fields, extend-from-boundary, …).
        min_size : lower bound assigned to ``Mesh.MeshSizeMin``.  Defaults
                   to ``0.0`` — i.e. no floor, so per-point refinements
                   (``set_size(..., 100)``) are free to produce elements
                   smaller than ``max_size``.  Raise it only when you want
                   to prevent Gmsh from going finer than a given threshold.

        Notes
        -----
        Earlier versions of this method clamped both bounds to the same
        value, which silently *overrode* per-point refinements.  The new
        default (``min_size=0``) avoids that trap.  Call with two args when
        you truly want a band, e.g. ``set_global_size(6000, 200)``.

        Example
        -------
        ::

            m1.mesh.set_global_size(6000)            # ceiling only
            m1.mesh.set_global_size(6000, 200)       # band [200, 6000]
        """
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        self._log(f"set_global_size(max={max_size}, min={min_size})")
        return self

    def set_size_sources(
        self,
        *,
        from_points          : bool | None = None,
        from_curvature       : bool | None = None,
        extend_from_boundary : bool | None = None,
    ) -> Mesh:
        """
        Control *which* size sources Gmsh consults when meshing.

        Gmsh combines several size sources at each node and takes the
        minimum.  When ``from_points`` is on (Gmsh default), every BRep
        point carries its own characteristic length — imported CAD files
        typically bake in small ``lc`` values that silently override
        :meth:`set_global_size`.  Disable the sources you do not want.

        Parameters
        ----------
        from_points : when ``True`` (default in Gmsh), per-point ``lc``
                      values affect the mesh.  Set ``False`` to ignore
                      them — useful after IGES/DXF import.
        from_curvature : when ``True``, Gmsh adapts element size to
                         curvature.  Default off.
        extend_from_boundary : when ``True`` (default in Gmsh), sizes on
                               boundary entities propagate inward.

        Returns
        -------
        self — for method chaining

        Example
        -------
        ::

            # Make set_global_size(6000) actually honoured after IGES import
            (g.mesh
               .set_size_sources(from_points=False,
                                 from_curvature=False,
                                 extend_from_boundary=False)
               .set_global_size(6000)
               .generate(dim=2))
        """
        if from_points is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", int(bool(from_points)))
        if from_curvature is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(bool(from_curvature)))
        if extend_from_boundary is not None:
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",
                                  int(bool(extend_from_boundary)))
        self._log(
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
    ) -> Mesh:
        """
        Set the global mesh-size bounds.

        Unlike :meth:`set_global_size` (which sets *both* limits to the
        same value), this method lets you control the minimum and maximum
        independently — useful when you want Gmsh's size field to
        interpolate between a range.

        Parameters
        ----------
        min_size : lower bound on element size.  ``None`` leaves the
                   current Gmsh default unchanged.
        max_size : upper bound on element size.  ``None`` leaves the
                   current Gmsh default unchanged.

        Example
        -------
        ::

            # fiber-section mesh with a size band
            g.mesh.set_size_global(min_size=15, max_size=25)

            # only cap the maximum — leave minimum at Gmsh default
            g.mesh.set_size_global(max_size=50)
        """
        if min_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        if max_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        self._log(f"set_size_global(min={min_size}, max={max_size})")
        return self

    def set_size(
        self,
        tags: TagsLike,
        size: float,
        *,
        dim : int = 0,
    ) -> Mesh:
        """
        Assign a target element size to specific points.

        ``gmsh.model.mesh.setSize`` is currently effective only for
        dimension-0 entities (points).  Passing ``dim > 0`` will not raise,
        but Gmsh will silently ignore the call.  Use a ``MathEval`` or
        ``Distance`` mesh field (``g.mesh.distance()`` / ``g.mesh.threshold()``)
        to control sizing on curves, surfaces, or volumes.

        Parameters
        ----------
        tags : point tags (dim=0)
        size : target element size
        dim  : entity dimension — keep at 0 (default)

        Example
        -------
        ::

            # refine around a set of corner points
            g.mesh.set_size([p1, p2, p3], 0.05)
        """
        dimtags = self._as_dimtags(tags, dim)
        gmsh.model.mesh.setSize(dimtags, size)
        self._directives.append({
            'kind': 'set_size', 'dim': dim,
            'tags': [t for _, t in dimtags], 'size': size,
        })
        self._log(f"set_size(dim={dim}, size={size})")
        return self

    def set_size_all_points(self, size: float) -> Mesh:
        """
        Assign the same characteristic length to *every* BRep point in
        the model.

        Typical use — normalising per-point ``lc`` values after an
        IGES/STEP/DXF import, where the imported points carry small
        characteristic lengths that would otherwise override
        :meth:`set_global_size`.

        Note
        ----
        This method only has an effect if ``Mesh.MeshSizeFromPoints`` is
        enabled (the Gmsh default).  If you've called
        ``set_size_sources(from_points=False)`` the per-point sizes set
        here will be ignored at mesh time.

        Parameters
        ----------
        size : characteristic length assigned to every point

        Returns
        -------
        self — for method chaining

        Example
        -------
        ::

            (g.mesh
               .set_size_all_points(6000)   # overwrite imported IGES lc
               .set_global_size(6000)
               .generate(dim=2))
        """
        pts = gmsh.model.getEntities(dim=0)
        if pts:
            gmsh.model.mesh.setSize(pts, size)
        self._directives.append({
            'kind': 'set_size_all_points', 'size': size,
            'n_points': len(pts),
        })
        self._log(f"set_size_all_points(size={size}, n={len(pts)})")
        return self

    def set_size_callback(
        self,
        func: Callable[[float, float, float, float, int, int], float],
    ) -> Mesh:
        """
        Register a Python callback that returns the desired element size at
        any point in the model.

        Callback signature
        ------------------
        ``func(dim, tag, x, y, z, lc) -> float``

        Gmsh passes ``dim``/``tag`` of the entity being sampled first,
        then the XYZ coordinates, then ``lc`` (current size estimate).
        Return a positive float.

        Example
        -------
        ::

            def my_size(dim, tag, x, y, z, lc):
                return max(0.05, 0.01 * math.sqrt(x**2 + y**2))

            g.mesh.set_size_callback(my_size)
        """
        gmsh.model.mesh.setSizeCallback(func)
        self._directives.append({
            'kind': 'set_size_callback',
            'func_name': getattr(func, '__name__', '<callable>'),
        })
        self._log("set_size_callback(<callable>)")
        return self

    # ------------------------------------------------------------------
    # Structured / transfinite meshing
    # ------------------------------------------------------------------

    def set_transfinite_curve(
        self,
        tag      : int,
        n_nodes  : int,
        *,
        mesh_type: str   = "Progression",
        coef     : float = 1.0,
    ) -> Mesh:
        """
        Set a transfinite constraint on a curve.

        Parameters
        ----------
        tag       : curve tag
        n_nodes   : number of nodes along the curve (endpoints included)
        mesh_type : ``"Progression"`` (geometric) or ``"Bump"`` (clustered
                    at both ends)
        coef      : progression ratio; >1 clusters nodes toward the start

        Example
        -------
        ::

            # 20 nodes, mildly clustered at start
            g.mesh.set_transfinite_curve(edge_tag, 20, coef=1.3)
        """
        gmsh.model.mesh.setTransfiniteCurve(tag, n_nodes,
                                             meshType=mesh_type, coef=coef)
        self._directives.append({
            'kind': 'transfinite_curve', 'tag': tag,
            'n_nodes': n_nodes, 'mesh_type': mesh_type, 'coef': coef,
        })
        self._log(
            f"set_transfinite_curve(tag={tag}, n={n_nodes}, "
            f"type={mesh_type!r}, coef={coef})"
        )
        return self

    def set_transfinite_surface(
        self,
        tag        : int,
        *,
        arrangement: str            = "Left",
        corners    : list[int] | None = None,
    ) -> Mesh:
        """
        Set a transfinite constraint on a surface (mapped/structured quad).

        Parameters
        ----------
        tag         : surface tag
        arrangement : triangle split pattern — ``"Left"``, ``"Right"``,
                      ``"AlternateLeft"``, or ``"AlternateRight"``
        corners     : 3 or 4 corner point tags (needed for non-trivial
                      topology, e.g. triangular faces)

        Note
        ----
        All four bounding curves must also have transfinite constraints
        (via ``set_transfinite_curve``) and compatible node counts.
        """
        gmsh.model.mesh.setTransfiniteSurface(tag, arrangement=arrangement,
                                               cornerTags=corners or [])
        self._directives.append({
            'kind': 'transfinite_surface', 'tag': tag,
            'arrangement': arrangement,
            'corners': corners or [],
        })
        self._log(
            f"set_transfinite_surface(tag={tag}, "
            f"arrangement={arrangement!r})"
        )
        return self

    def set_transfinite_volume(
        self,
        tag    : int,
        *,
        corners: list[int] | None = None,
    ) -> Mesh:
        """
        Set a transfinite constraint on a volume (mapped hex-style).

        Parameters
        ----------
        tag     : volume tag
        corners : optional list of 8 corner point tags
        """
        gmsh.model.mesh.setTransfiniteVolume(tag, cornerTags=corners or [])
        self._directives.append({
            'kind': 'transfinite_volume', 'tag': tag,
            'corners': corners or [],
        })
        self._log(f"set_transfinite_volume(tag={tag})")
        return self

    def set_transfinite_automatic(
        self,
        dim_tags    : list[DimTag] | None = None,
        *,
        corner_angle: float = 2.35,
        recombine   : bool  = True,
    ) -> Mesh:
        """
        Let gmsh automatically detect and set transfinite constraints on
        compatible 3- and 4-sided surfaces/volumes.

        Parameters
        ----------
        dim_tags     : entities to consider (``[]`` = all)
        corner_angle : maximum angle (radians) to classify a vertex as a
                       transfinite corner (default ≈ 135°)
        recombine    : also set recombination on detected surfaces
        """
        gmsh.model.mesh.setTransfiniteAutomatic(
            dimTags=dim_tags or [],
            cornerAngle=corner_angle,
            recombine=recombine,
        )
        self._directives.append({
            'kind': 'transfinite_automatic',
            'dim_tags': dim_tags or [],
            'corner_angle': corner_angle,
            'recombine': recombine,
        })
        self._log(
            f"set_transfinite_automatic("
            f"corner_angle={math.degrees(corner_angle):.1f}°, "
            f"recombine={recombine})"
        )
        return self

    # ------------------------------------------------------------------
    # Recombination / quad meshing
    # ------------------------------------------------------------------

    def set_recombine(
        self,
        tag  : int,
        *,
        dim  : int   = 2,
        angle: float = 45.0,
    ) -> Mesh:
        """
        Request quad recombination for a surface (dim=2) or volume (dim=3).

        Parameters
        ----------
        tag   : surface or volume tag
        dim   : entity dimension
        angle : template angle threshold (degrees) for the recombination
                algorithm — lower values are more strict
        """
        gmsh.model.mesh.setRecombine(dim, tag, angle)
        self._directives.append({
            'kind': 'recombine', 'dim': dim, 'tag': tag, 'angle': angle,
        })
        self._log(f"set_recombine(dim={dim}, tag={tag}, angle={angle}°)")
        return self

    def recombine(self) -> Mesh:
        """Globally recombine all triangular elements into quads."""
        gmsh.model.mesh.recombine()
        self._log("recombine()")
        return self

    def set_smoothing(self, tag: int, val: int, *, dim: int = 2) -> Mesh:
        """
        Set the number of Laplacian smoothing passes for a surface.

        Parameters
        ----------
        tag : surface tag
        val : number of smoothing iterations
        """
        gmsh.model.mesh.setSmoothing(dim, tag, val)
        self._directives.append({
            'kind': 'smoothing', 'dim': dim, 'tag': tag, 'val': val,
        })
        self._log(f"set_smoothing(dim={dim}, tag={tag}, val={val})")
        return self

    def set_algorithm(
        self,
        tag      : int,
        algorithm: int,
        *,
        dim      : int = 2,
    ) -> Mesh:
        """
        Choose the meshing algorithm for a surface (dim=2) or globally for
        all volumes (dim=3).

        Parameters
        ----------
        tag       : surface tag (dim=2); ignored for dim=3
        algorithm : use ``Algorithm2D.*`` (dim=2) or ``Algorithm3D.*`` (dim=3)
        dim       : entity dimension — must be 2 or 3

        Note
        ----
        ``gmsh.model.mesh.setAlgorithm`` only supports dim=2 (per-surface
        selection).  For dim=3, the algorithm is applied globally via
        ``gmsh.option.setNumber("Mesh.Algorithm3D", ...)``.

        Example
        -------
        ::

            g.mesh.set_algorithm(surf_tag, Algorithm2D.FRONTAL_DELAUNAY_QUADS)
            g.mesh.set_algorithm(0, Algorithm3D.HXT, dim=3)
        """
        if dim == 2:
            gmsh.model.mesh.setAlgorithm(2, tag, int(algorithm))
            self._directives.append({
                'kind': 'algorithm', 'dim': 2, 'tag': tag,
                'algorithm': int(algorithm),
            })
            self._log(f"set_algorithm(dim=2, tag={tag}, alg={algorithm!r})")
        elif dim == 3:
            gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm))
            self._directives.append({
                'kind': 'algorithm', 'dim': 3, 'tag': tag,
                'algorithm': int(algorithm),
            })
            self._log(f"set_algorithm(dim=3, alg={algorithm!r})  [global option]")
        else:
            raise ValueError(f"set_algorithm: dim must be 2 or 3, got {dim!r}")
        return self

    def set_compound(self, dim: int, tags: list[int]) -> Mesh:
        """Merge entities so they are meshed together as a single compound."""
        gmsh.model.mesh.setCompound(dim, tags)
        self._log(f"set_compound(dim={dim}, tags={tags})")
        return self

    def remove_constraints(
        self,
        dim_tags: list[DimTag] | None = None,
    ) -> Mesh:
        """Remove all meshing constraints from the given (or all) entities."""
        gmsh.model.mesh.removeConstraints(dimTags=dim_tags or [])
        self._log("remove_constraints()")
        return self

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        tags  : int | list[int],
        in_tag: int,
        *,
        dim   : int = 0,
        in_dim: int = 3,
    ) -> Mesh:
        """
        Embed lower-dimensional entities inside a higher-dimensional entity
        so the mesh is conforming along them.

        Parameters
        ----------
        tags   : tags of entities to embed
        in_tag : tag of the containing entity
        dim    : dimension of entities to embed (0=pts, 1=curves, 2=surfs)
        in_dim : dimension of the container (must be > dim)

        Example
        -------
        ::

            # force mesh nodes along a crack surface inside a volume
            g.mesh.embed(crack_surf_tag, body_tag, dim=2, in_dim=3)

            # force mesh nodes on a point cloud inside a surface
            g.mesh.embed([p1, p2, p3], surf_tag, dim=0, in_dim=2)
        """
        tag_list = tags if isinstance(tags, list) else [tags]
        gmsh.model.mesh.embed(dim, tag_list, in_dim, in_tag)
        self._log(
            f"embed(dim={dim}, tags={tag_list}, "
            f"in_dim={in_dim}, in_tag={in_tag})"
        )
        return self

    # ------------------------------------------------------------------
    # Periodicity
    # ------------------------------------------------------------------

    def set_periodic(
        self,
        tags       : list[int],
        master_tags: list[int],
        transform  : list[float],
        *,
        dim        : int = 2,
    ) -> Mesh:
        """
        Declare periodic mesh correspondence between entities (typically
        opposite faces of a periodic domain).

        Parameters
        ----------
        tags        : slave entity tags
        master_tags : master entity tags
        transform   : 16-element row-major 4×4 affine matrix mapping
                      master → slave coordinates
        dim         : entity dimension (1 = curves, 2 = surfaces)

        Example
        -------
        ::

            # translate surface 2 to surface 1 by Δx = 10
            g.mesh.set_periodic(
                [2], [1],
                [1,0,0,10,  0,1,0,0,  0,0,1,0,  0,0,0,1]
            )
        """
        gmsh.model.mesh.setPeriodic(dim, tags, master_tags, transform)
        self._log(
            f"set_periodic(dim={dim}, tags={tags}, master={master_tags})"
        )
        return self

    # ------------------------------------------------------------------
    # STL / discrete geometry
    # ------------------------------------------------------------------

    def import_stl(self) -> Mesh:
        """
        Classify an STL mesh previously loaded into the gmsh model via
        ``gmsh.merge`` as a discrete surface mesh.

        Typically followed by ``classify_surfaces`` and ``create_geometry``.
        """
        gmsh.model.mesh.importStl()
        self._log("import_stl()")
        return self

    def classify_surfaces(
        self,
        angle              : float,
        *,
        boundary           : bool  = True,
        for_reparametrization: bool = False,
        curve_angle        : float = math.pi,
        export_discrete    : bool  = True,
    ) -> Mesh:
        """
        Partition a discrete STL mesh into surface patches based on
        dihedral angle.  First step in the STL → geometry → mesh pipeline.

        Parameters
        ----------
        angle                : dihedral angle threshold (radians) for edge
                               classification — edges sharper than this are
                               treated as patch boundaries
        boundary             : also reconstruct bounding curves
        for_reparametrization: prepare the classification for surface
                               reparametrization
        curve_angle          : threshold for curve classification (default π)
        export_discrete      : export the resulting discrete geometry

        Example
        -------
        ::

            gmsh.merge("scan.stl")
            (g.mesh
               .import_stl()
               .classify_surfaces(math.radians(30))
               .create_geometry()
               .generate(2))
        """
        gmsh.model.mesh.classifySurfaces(
            angle,
            boundary=boundary,
            forReparametrization=for_reparametrization,
            curveAngle=curve_angle,
            exportDiscrete=export_discrete,
        )
        self._log(
            f"classify_surfaces(angle={math.degrees(angle):.1f}°, "
            f"boundary={boundary})"
        )
        return self

    def create_geometry(
        self,
        dim_tags: list[DimTag] | None = None,
    ) -> Mesh:
        """
        Create a proper CAD-like geometry from classified discrete surfaces.
        Must be called after ``classify_surfaces``.
        """
        gmsh.model.mesh.createGeometry(dimTags=dim_tags or [])
        self._log("create_geometry()")
        return self

    # ------------------------------------------------------------------
    # Mesh editing
    # ------------------------------------------------------------------

    def clear(self, dim_tags: list[DimTag] | None = None) -> Mesh:
        """
        Clear mesh data (nodes + elements).

        Parameters
        ----------
        dim_tags : limit deletion to specific entities (``None`` = all)
        """
        gmsh.model.mesh.clear(dimTags=dim_tags or [])
        self._log(f"clear(dim_tags={dim_tags})")
        return self

    def reverse(self, dim_tags: list[DimTag] | None = None) -> Mesh:
        """Reverse the orientation of mesh elements in the given entities."""
        gmsh.model.mesh.reverse(dimTags=dim_tags or [])
        self._log("reverse()")
        return self

    def relocate_nodes(self, *, dim: int = -1, tag: int = -1) -> Mesh:
        """
        Project mesh nodes back onto their underlying geometry.  Useful
        after high-order elevation or after modifying geometry post-mesh.
        """
        gmsh.model.mesh.relocateNodes(dim=dim, tag=tag)
        self._log(f"relocate_nodes(dim={dim}, tag={tag})")
        return self

    def remove_duplicate_nodes(self, verbose: bool = True) -> Mesh:
        """
        Merge nodes that share the same position within tolerance.

        Parameters
        ----------
        verbose : if True (default), print how many nodes were merged
                  (or a 'nothing to do' notice when the mesh was already clean).
        """
        before = len(gmsh.model.mesh.getNodes()[0])
        gmsh.model.mesh.removeDuplicateNodes()
        after  = len(gmsh.model.mesh.getNodes()[0])
        removed = before - after
        if verbose:
            if removed > 0:
                print(f"remove_duplicate_nodes: merged {removed} "
                      f"node(s) ({before} → {after})")
            else:
                print(f"remove_duplicate_nodes: no duplicates found "
                      f"({before} nodes unchanged)")
        self._log(f"remove_duplicate_nodes() removed={removed}")
        return self

    def remove_duplicate_elements(self, verbose: bool = True) -> Mesh:
        """
        Remove elements with identical node connectivity.

        Parameters
        ----------
        verbose : if True (default), print how many elements were removed
                  (or a 'nothing to do' notice when the mesh was already clean).
        """
        def _count() -> int:
            _, tags, _ = gmsh.model.mesh.getElements()
            return sum(len(t) for t in tags)

        before = _count()
        gmsh.model.mesh.removeDuplicateElements()
        after  = _count()
        removed = before - after
        if verbose:
            if removed > 0:
                print(f"remove_duplicate_elements: removed {removed} "
                      f"element(s) ({before} → {after})")
            else:
                print(f"remove_duplicate_elements: no duplicates found "
                      f"({before} elements unchanged)")
        self._log(f"remove_duplicate_elements() removed={removed}")
        return self

    def affine_transform(
        self,
        matrix  : list[float],
        dim_tags: list[DimTag] | None = None,
    ) -> Mesh:
        """
        Apply an affine transformation to mesh nodes (12 coefficients,
        row-major 4×3 matrix — translation in last column).

        Parameters
        ----------
        matrix   : 12 floats ``[a b c d  e f g h  i j k l]``
                   representing ``x' = ax+by+cz+d``, etc.
        dim_tags : limit to specific entities (``None`` = all)
        """
        gmsh.model.mesh.affineTransform(matrix, dimTags=dim_tags or [])
        self._log("affine_transform()")
        return self

    # ------------------------------------------------------------------
    # Partitioning / renumbering
    # ------------------------------------------------------------------

    def partition(
        self,
        n_parts     : int,
        element_tags: list[int] | None = None,
        partitions  : list[int] | None = None,
    ) -> Mesh:
        """
        Partition the mesh into ``n_parts`` sub-domains (e.g. for MPI runs).

        Parameters
        ----------
        n_parts      : number of partitions
        element_tags : elements to partition (``None`` = all)
        partitions   : pre-assigned partition IDs per element (``None`` = auto)
        """
        gmsh.model.mesh.partition(
            n_parts,
            elementTags=element_tags or [],
            partitions=partitions    or [],
        )
        self._log(f"partition(n_parts={n_parts})")
        return self

    def unpartition(self) -> Mesh:
        """Remove the partition structure and restore a monolithic mesh."""
        gmsh.model.mesh.unpartition()
        self._log("unpartition()")
        return self

    def compute_renumbering(
        self,
        method      : str            = "RCMK",
        element_tags: list[int] | None = None,
    ) -> tuple[ndarray, ndarray]:
        """
        Compute an optimised node renumbering (e.g. RCMK bandwidth reduction).

        Returns
        -------
        (old_tags, new_tags) : two int ndarrays of equal length
        """
        old, new = gmsh.model.mesh.computeRenumbering(
            method=method, elementTags=element_tags or []
        )
        self._log(
            f"compute_renumbering(method={method!r}) → {len(old)} nodes"
        )
        return np.array(old, dtype=np.int64), np.array(new, dtype=np.int64)

    def renumber_nodes(
        self,
        old_tags: list[int],
        new_tags: list[int],
    ) -> Mesh:
        """Apply a pre-computed node renumbering."""
        gmsh.model.mesh.renumberNodes(oldTags=old_tags, newTags=new_tags)
        self._log(f"renumber_nodes({len(old_tags)} nodes)")
        return self

    def renumber_elements(
        self,
        old_tags: list[int],
        new_tags: list[int],
    ) -> Mesh:
        """Apply a pre-computed element renumbering."""
        gmsh.model.mesh.renumberElements(oldTags=old_tags, newTags=new_tags)
        self._log(f"renumber_elements({len(old_tags)} elements)")
        return self

    def renumber_mesh(
        self,
        dim: int = 2,
        *,
        method: str = "simple",
        base: int = 1,
        used_only: bool = True,
    ) -> Mesh:
        """
        Renumber nodes and elements in the Gmsh model to contiguous IDs.

        After this call, **all** Gmsh queries (``get_nodes``,
        ``get_elements``, ``getNodesForPhysicalGroup``, etc.) return
        solver-ready contiguous IDs directly.

        This is a mutation of the Gmsh model — call it **once**, before
        extracting FEM data with :meth:`get_fem_data`.

        Parameters
        ----------
        dim : int
            Element dimension used to build adjacency for RCM
            (2 = shells/quads, 3 = solids).
        method : ``"simple"`` or ``"rcm"``
            ``"simple"``  — contiguous IDs preserving relative order.
            ``"rcm"``  — Reverse Cuthill-McKee bandwidth minimisation.
        base : int
            Starting ID (default 1 = OpenSees/Abaqus convention).
        used_only : bool
            If True (default), only renumber nodes connected to at
            least one element (orphan nodes are skipped).

        Returns
        -------
        Mesh
            ``self``, for method chaining.

        Example
        -------
        ::

            g.mesh.renumber_mesh(method="rcm", base=1)
            fem = g.mesh.get_fem_data(dim=2)

            for i in range(fem.info.n_nodes):
                ops.node(int(fem.node_ids[i]), *fem.node_coords[i])

        Note
        ----
        This operation is **irreversible** within the current Gmsh
        session.
        """
        from ..solvers.Numberer import Numberer
        raw = self._get_raw_fem_data(dim=dim)
        numb = Numberer(raw)
        info = numb.renumber(method=method, base=base, used_only=used_only)
        self._log(
            f"renumber_mesh(method={method!r}): "
            f"{info.n_nodes} nodes, {info.n_elems} elements, "
            f"bandwidth={info.bandwidth}"
        )
        return self

    # ------------------------------------------------------------------
    # Queries  (return data — no chaining)
    # ------------------------------------------------------------------

    def get_nodes(
        self,
        *,
        dim              : int  = -1,
        tag              : int  = -1,
        include_boundary : bool = False,
        return_parametric: bool = False,
    ) -> dict:
        """
        Query mesh nodes.

        Parameters
        ----------
        dim              : restrict to entities of this dimension (-1 = all)
        tag              : restrict to a single entity (-1 = all)
        include_boundary : include nodes on the boundary of the entity
        return_parametric: also return parametric (UV) coordinates

        Returns
        -------
        dict
            ``'tags'``              : ndarray(N,)   — node tags
            ``'coords'``            : ndarray(N, 3) — XYZ coordinates
            ``'parametric_coords'`` : ndarray       — only if requested
        """
        node_tags, coords, param = gmsh.model.mesh.getNodes(
            dim=dim, tag=tag,
            includeBoundary=include_boundary,
            returnParametricCoord=return_parametric,
        )
        result: dict = {
            'tags'  : np.array(node_tags, dtype=np.int64),
            'coords': np.array(coords).reshape(-1, 3),
        }
        if return_parametric and len(param):
            result['parametric_coords'] = np.array(param)
        self._log(f"get_nodes → {len(node_tags)} nodes")
        return result

    def get_elements(
        self,
        *,
        dim: int = -1,
        tag: int = -1,
    ) -> dict:
        """
        Query mesh elements.

        Parameters
        ----------
        dim : restrict to this dimension (-1 = all)
        tag : restrict to a single entity  (-1 = all)

        Returns
        -------
        dict
            ``'types'``     : list[int]         — gmsh element type codes
            ``'tags'``      : list[ndarray]     — element tags per type
            ``'node_tags'`` : list[ndarray]     — connectivity per type
        """
        elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(
            dim=dim, tag=tag
        )
        result = {
            'types'    : list(elem_types),
            'tags'     : [np.array(t, dtype=np.int64) for t in elem_tags],
            'node_tags': [np.array(n, dtype=np.int64) for n in node_tags],
        }
        total = sum(len(t) for t in result['tags'])
        self._log(
            f"get_elements → {total} elements "
            f"({len(elem_types)} types)"
        )
        return result

    def get_element_properties(self, element_type: int) -> dict:
        """
        Return metadata for a given gmsh element type code.

        Returns
        -------
        dict
            ``'name'``            : str   — human-readable type name
            ``'dim'``             : int   — topological dimension
            ``'order'``           : int   — polynomial order
            ``'n_nodes'``         : int   — nodes per element
            ``'n_primary_nodes'`` : int   — corner nodes (excluding mid-side)
            ``'local_coords'``    : ndarray — node positions in ref element
        """
        name, dim, order, n_nodes, local_coords, n_primary = \
            gmsh.model.mesh.getElementProperties(element_type)
        d = max(dim, 1)
        return {
            'name'           : name,
            'dim'            : dim,
            'order'          : order,
            'n_nodes'        : n_nodes,
            'n_primary_nodes': n_primary,
            'local_coords'   : np.array(local_coords).reshape(-1, d),
        }

    def _get_raw_fem_data(self, dim: int = 2) -> dict:
        """
        Internal helper — extracts raw FEM data as a plain dict.

        Used by :meth:`renumber_mesh` (which needs raw data before
        the Gmsh model is renumbered) and by :meth:`get_fem_data`
        (which packages it into a :class:`FEMData` object).
        """
        # --- nodes (full mesh) ---
        nodes       = self.get_nodes()
        node_tags   = nodes['tags']
        node_coords = nodes['coords']

        # --- elements of requested dimension ---
        elems = self.get_elements(dim=dim)

        conn_blocks: list[ndarray] = []
        elem_tags: list[int] = []
        elem_type_codes: list[int] = []
        elem_type_info: dict[int, tuple] = {}
        for etype, etags, enodes in zip(
            elems['types'], elems['tags'], elems['node_tags']
        ):
            props  = self.get_element_properties(etype)
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
        # target-dim connectivity.  Counting only target-dim elements
        # would incorrectly classify them as orphans and drop them
        # during renumbering / FEM extraction.
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

    def get_fem_data(self, dim: int = 2):
        """
        Extract solver-ready FEM data as a :class:`FEMData` object.

        Returns node IDs, coordinates, element IDs, connectivity,
        mesh statistics (``.info``), and physical group data
        (``.physical``) — everything needed to build a solver model.

        If :meth:`renumber_mesh` was called first, all IDs are
        contiguous and solver-ready.  Otherwise they are the raw
        (potentially non-contiguous) Gmsh tags.

        Must be called **after** ``generate()``.

        Parameters
        ----------
        dim : int
            Element dimension to extract (2 = triangles/quads,
            3 = tets/hexes).

        Returns
        -------
        FEMData

        Example
        -------
        ::

            g.mesh.renumber_mesh(method="rcm", base=1)
            fem = g.mesh.get_fem_data(dim=2)

            # Mesh stats
            print(fem.info)
            print(fem.info.n_nodes, fem.info.bandwidth)

            # Physical groups (mirrors g.physical API)
            fem.physical.get_all()
            base = fem.physical.get_nodes(0, 1)  # {'tags': ..., 'coords': ...}

            # Build solver model
            for i in range(fem.info.n_nodes):
                ops.node(int(fem.node_ids[i]),
                         *fem.node_coords[i])
        """
        from ._fem_extract import build_fem_data

        ms_composite = getattr(self._parent, 'mesh_selection', None)
        result = build_fem_data(dim=dim, mesh_selection_composite=ms_composite)

        self._log(
            f"get_fem_data(dim={dim}) → "
            f"{result.info.n_nodes} nodes, "
            f"{result.info.n_elems} elements, "
            f"bw={result.info.bandwidth}"
        )

        return result

    def get_element_qualities(
        self,
        element_tags: list[int] | ndarray,
        quality_name: str = "minSICN",
    ) -> ndarray:
        """
        Compute quality metrics for the given elements.

        Parameters
        ----------
        element_tags : element tags to evaluate
        quality_name : metric — ``"minSICN"`` (signed inverse condition
                       number), ``"minSIGE"`` (signed inverse gradient
                       error), ``"gamma"`` (inscribed/circumscribed ratio),
                       ``"minSJ"`` (minimum scaled Jacobian)

        Returns
        -------
        ndarray
            Quality values, one per element.
        """
        tags = list(element_tags) if not isinstance(element_tags, list) else element_tags
        q = gmsh.model.mesh.getElementQualities(tags, qualityName=quality_name)
        return np.asarray(q)

    def quality_report(
        self,
        *,
        dim: int = -1,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute a summary quality report for all mesh elements.

        For each element type and quality metric, reports count, min, max,
        mean, std, and the percentage of elements below common thresholds.

        Must be called **after** ``generate()``.

        Parameters
        ----------
        dim : element dimension to report (-1 = all dimensions present)
        metrics : quality names to evaluate.  Defaults to
                  ``["minSICN", "minSIGE", "gamma", "minSJ"]``.

                  * ``minSICN`` — signed inverse condition number
                  * ``minSIGE`` — signed inverse gradient error
                  * ``gamma``   — inscribed / circumscribed radius ratio
                  * ``minSJ``   — minimum scaled Jacobian

        Returns
        -------
        pd.DataFrame
            One row per (element_type, metric) combination with summary
            statistics.

        Example
        -------
        ::

            g.mesh.generate(2)
            print(g.mesh.quality_report().to_string())
        """
        import pandas as pd

        if metrics is None:
            metrics = ["minSICN", "minSIGE", "gamma", "minSJ"]

        elems = self.get_elements(dim=dim)

        rows: list[dict] = []
        for etype, etags in zip(elems['types'], elems['tags']):
            if len(etags) == 0:
                continue
            props = self.get_element_properties(etype)
            etype_name = props.get('name', str(etype))

            for metric in metrics:
                try:
                    q = gmsh.model.mesh.getElementQualities(
                        list(etags.astype(int)), qualityName=metric,
                    )
                    q = np.asarray(q)
                except Exception:
                    continue  # metric not supported for this element type

                if len(q) == 0:
                    continue

                row: dict = {
                    'element_type' : etype_name,
                    'gmsh_code'    : int(etype),
                    'metric'       : metric,
                    'count'        : len(q),
                    'min'          : float(q.min()),
                    'max'          : float(q.max()),
                    'mean'         : float(q.mean()),
                    'std'          : float(q.std()),
                    'pct_below_0.1': float((q < 0.1).sum() / len(q) * 100),
                    'pct_below_0.3': float((q < 0.3).sum() / len(q) * 100),
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index(['element_type', 'metric']).sort_index()

        self._log(
            f"quality_report(dim={dim}) → "
            f"{len(rows)} metric rows across "
            f"{df.index.get_level_values('element_type').nunique() if not df.empty else 0} "
            f"element types"
        )

        if self._parent._verbose and not df.empty:
            print("\n--- Mesh Quality Report ---")
            print(df.to_string())

        return df

    # ------------------------------------------------------------------
    # Interactive mesh viewer
    # ------------------------------------------------------------------

    def viewer(self, **kwargs):
        """Open the interactive mesh viewer.

        The viewer supports picking (BRep entities, elements, nodes),
        color modes, and physical group visualization.

        Parameters are forwarded to :class:`MeshViewer`.
        """
        from ..viewers._mesh_viewer import MeshViewer
        mv = MeshViewer(self._parent, self, **kwargs)
        return mv.show()

    def viewer_fast(self, **kwargs):
        """Open the mesh viewer (fast).

        Same as :meth:`viewer` — always uses batched actors.
        Kept for backward compatibility.

        Parameters are forwarded to :class:`MeshViewer`.
        """
        from ..viewers._mesh_viewer import MeshViewer
        mv = MeshViewer(self._parent, self, fast=True, **kwargs)
        return mv.show()

    def results_viewer(
        self,
        results: str | None = None,
        *,
        point_data: dict | None = None,
        cell_data: dict | None = None,
        blocking: bool = False,
    ) -> None:
        """Open the results viewer (pyGmshViewer).

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
            If True, blocks until viewer is closed.
        """
        if results is not None:
            from pyGmshViewer import show
            show(results, blocking=blocking)
        elif point_data is not None or cell_data is not None:
            from ..results.Results import Results
            fem = self.get_fem_data()
            r = Results.from_fem(
                fem,
                point_data=point_data,
                cell_data=cell_data,
                name=self._parent.model_name,
            )
            r.viewer(blocking=blocking)
        else:
            from ..viz.VTKExport import VTKExport
            import tempfile
            from pyGmshViewer import show
            vtk_export = VTKExport(self._parent)
            tmp = tempfile.NamedTemporaryFile(suffix=".vtu", delete=False)
            vtk_export.write(tmp.name)
            tmp.close()
            show(tmp.name, blocking=blocking)

    # -----------------------------