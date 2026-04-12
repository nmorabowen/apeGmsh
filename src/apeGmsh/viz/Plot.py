from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D              # noqa: F401 – registers projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from numpy import ndarray

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
DimTag = tuple[int, int]

# ---------------------------------------------------------------------------
# Gmsh element-type -> corner-node count
# Used to stride the flat node-tag arrays returned by getElements.
# Higher-order variants (9, 10, …) are not listed; those fall back to
# getElementProperties() at runtime.
# ---------------------------------------------------------------------------
_CORNER_NODES: dict[int, int] = {
    1:  2,   # 2-node line
    2:  3,   # 3-node triangle
    3:  4,   # 4-node quad
    4:  4,   # 4-node tet
    5:  8,   # 8-node hex
    6:  6,   # 6-node prism
    7:  5,   # 5-node pyramid
    15: 1,   # 1-node point
}

# Dimension labels / colours used for per-entity annotations
_DIM_PREFIX = {0: 'P', 1: 'C', 2: 'S', 3: 'V'}
_DIM_LABEL  = {0: 'points', 1: 'curves', 2: 'surfaces', 3: 'volumes'}


# ---------------------------------------------------------------------------
# Plot — plotting composite
# ---------------------------------------------------------------------------

class Plot:
    """
    Plotting composite attached to a ``apeGmsh`` instance as ``g.plot``.

    Provides matplotlib-based visualisation of both BRep geometry and mesh,
    with optional entity-tag labels for introspection.  All methods return
    ``self`` so they can be chained::

        (g.plot
           .geometry(label_tags=True, show=False)
           .mesh(alpha=0.6, show=False)
           .label_entities(dims=[2])
           .show())

    The figure is created on the first drawing call and reused for
    subsequent layering calls.  Call ``clear()`` to start a new figure.

    Parameters
    ----------
    parent : _SessionBase
        The owning instance — used for ``name`` and ``_verbose``.
    """

    # Default colour palette — override at instance level if needed
    COLOR_POINTS   = '#E05C00'
    COLOR_CURVES   = '#2C4A6E'
    COLOR_SURFACES = '#5B8DB8'
    COLOR_MESH     = '#3A7CB8'

    def __init__(self, parent: _SessionBase) -> None:
        self._parent = parent
        self._fig: plt.Figure | None = None
        self._ax:  Axes3D | None     = None
        self._figsize: tuple[float, float] = (9, 7)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self._parent._verbose:
            print(f"[Plot] {msg}")

    def figsize(self, size: tuple[float, float]) -> Plot:
        """
        Set the matplotlib figure size (width, height in inches).

        Call this *before* any drawing method to control the size of the
        figure when it's first created.  If a figure already exists, it is
        resized in place.

        Parameters
        ----------
        size : (width, height) tuple in inches

        Example
        -------
        ::

            (g.plot
               .figsize((12, 9))
               .geometry(show=False)
               .mesh()
               .show())
        """
        self._figsize = (float(size[0]), float(size[1]))
        if self._fig is not None:
            self._fig.set_size_inches(*self._figsize, forward=True)
        self._log(f"figsize({self._figsize})")
        return self

    def _ensure_axes(self) -> tuple[plt.Figure, Axes3D]:
        """Return ``(fig, ax)``, creating them on the first call."""
        if self._fig is None or self._ax is None:
            self._fig = plt.figure(figsize=self._figsize)
            self._ax  = self._fig.add_subplot(111, projection='3d')
            self._ax.set_xlabel('X')
            self._ax.set_ylabel('Y')
            self._ax.set_zlabel('Z')
            self._ax.set_title(self._parent.name)
        return self._fig, self._ax

    def _autoscale(self, pts: ndarray) -> None:
        """
        Set equal-aspect axes limits from an (N, 3) point cloud.
        Called automatically whenever a drawing method adds geometry.
        """
        _, ax = self._ensure_axes()
        if pts.shape[0] == 0:
            return
        x, y, z  = pts[:, 0], pts[:, 1], pts[:, 2]
        x_mid = 0.5 * (x.min() + x.max())
        y_mid = 0.5 * (y.min() + y.max())
        z_mid = 0.5 * (z.min() + z.max())
        half  = 0.5 * max(
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min(),
            1e-9,           # guard against zero-span models
        )
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_zlim(z_mid - half, z_mid + half)

    @staticmethod
    def _element_node_counts(etype: int) -> tuple[int, int]:
        """
        Return ``(n_total, n_corner)`` for a gmsh element type.

        ``n_total``  — total nodes per element as packed in getElements()
                       flat arrays; used as the stride when slicing.
        ``n_corner`` — first-order corner nodes; used for polygon vertices.

        For first-order elements both values are identical.  For second-order
        elements (e.g. 10-node tet, type 11) n_total > n_corner.
        """
        _, _, _, n_total, _, n_corner = gmsh.model.mesh.getElementProperties(etype)
        return int(n_total), int(n_corner)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def geometry(
        self,
        *,
        n_curve_samples : int   = 60,
        n_surf_samples  : int   = 20,
        show_points     : bool  = True,
        show_curves     : bool  = True,
        show_surfaces   : bool  = True,
        label_tags      : bool  = False,
        color_points    : str   = COLOR_POINTS,
        color_curves    : str   = COLOR_CURVES,
        color_surfaces  : str   = COLOR_SURFACES,
        surface_alpha   : float = 0.25,
        show            : bool  = True,
    ) -> Plot:
        """
        Plot BRep geometry by parametric sampling.  No mesh required.

        Parameters
        ----------
        n_curve_samples  : points sampled along each curve
        n_surf_samples   : UV grid resolution per surface (n × n)
        show_points      : scatter-plot geometric vertices
        show_curves      : draw sampled curve polylines
        show_surfaces    : draw shaded surface patches via UV grid
        label_tags       : annotate each entity with its tag (e.g. ``C3``)
        color_points/curves/surfaces : matplotlib colour specs
        surface_alpha    : opacity of surface patches
        show             : call ``plt.show()`` at the end

        Returns
        -------
        self — for method chaining
        """
        _, ax = self._ensure_axes()
        collected: list[ndarray] = []

        # --- Vertices (dim=0) ---
        if show_points:
            for _, tag in gmsh.model.getEntities(dim=0):
                xyz = np.array(gmsh.model.getValue(0, tag, []))
                ax.scatter(*xyz, color=color_points, s=20, zorder=5,
                           depthshade=False)
                if label_tags:
                    ax.text(*xyz, f'  P{tag}', fontsize=6,
                            color=color_points, va='bottom')
                collected.append(xyz.reshape(1, 3))

        # --- Curves (dim=1): sample parametrically ---
        if show_curves:
            for _, tag in gmsh.model.getEntities(dim=1):
                try:
                    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
                    u  = np.linspace(lo[0], hi[0], n_curve_samples)
                    pts = np.array(
                        gmsh.model.getValue(1, tag, u.tolist())
                    ).reshape(-1, 3)
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                            color=color_curves, linewidth=0.9)
                    if label_tags:
                        mid = pts[len(pts) // 2]
                        ax.text(*mid, f'  C{tag}', fontsize=6,
                                color=color_curves, va='bottom')
                    collected.append(pts)
                except Exception:
                    pass

        # --- Surfaces (dim=2): boundary-curve sampling -> Poly3DCollection ---
        # Previous approach used a uniform UV grid which fails badly for
        # trimmed surfaces (IGES/STEP imports) because the parametric
        # rectangle extends beyond the trim boundary.  Instead we sample
        # each surface's boundary curves, chain the points into a closed
        # polygon, and fan-triangulate from the centroid.
        if show_surfaces:
            tris: list[ndarray] = []
            for _, tag in gmsh.model.getEntities(dim=2):
                try:
                    bnd = gmsh.model.getBoundary(
                        [(2, tag)], combined=False, oriented=True,
                    )
                    if not bnd:
                        continue

                    # Sample each boundary curve and chain into polygon
                    boundary_pts: list[ndarray] = []
                    for _, ctag in bnd:
                        abs_ctag = abs(ctag)
                        lo, hi = gmsh.model.getParametrizationBounds(
                            1, abs_ctag,
                        )
                        u = np.linspace(lo[0], hi[0], n_curve_samples)
                        cpts = np.array(
                            gmsh.model.getValue(1, abs_ctag, u.tolist())
                        ).reshape(-1, 3)
                        if ctag < 0:          # reversed orientation
                            cpts = cpts[::-1]
                        # Drop last point (== first of next curve)
                        boundary_pts.append(cpts[:-1])

                    if not boundary_pts:
                        continue
                    polygon = np.vstack(boundary_pts)

                    # Delaunay triangulation (handles concave surfaces)
                    try:
                        from scipy.spatial import Delaunay
                        tri_obj = Delaunay(polygon[:, :2])
                        for simplex in tri_obj.simplices:
                            tris.append(polygon[simplex])
                    except Exception:
                        # Fallback: centroid fan (convex only)
                        centroid = polygon.mean(axis=0)
                        n_bnd = len(polygon)
                        for i in range(n_bnd):
                            j = (i + 1) % n_bnd
                            tris.append(
                                np.array([centroid, polygon[i], polygon[j]])
                            )

                    if label_tags:
                        ax.text(*centroid, f'  S{tag}', fontsize=6,
                                color=color_surfaces)
                    collected.append(polygon)
                except Exception:
                    pass

            if tris:
                ax.add_collection3d(
                    Poly3DCollection(tris, alpha=surface_alpha,
                                     facecolor=color_surfaces,
                                     edgecolor='none')
                )

        if collected:
            self._autoscale(np.vstack(collected))

        n_pts  = len(gmsh.model.getEntities(dim=0))
        n_crv  = len(gmsh.model.getEntities(dim=1))
        n_srf  = len(gmsh.model.getEntities(dim=2))
        self._log(
            f"geometry(pts={n_pts}, curves={n_crv}, "
            f"surfaces={n_srf}, label_tags={label_tags})"
        )
        if show:
            self.show()
        return self

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------

    def mesh(
        self,
        *,
        color      : str   = COLOR_MESH,
        edge_color : str   = 'white',
        alpha      : float = 0.70,
        linewidth  : float = 0.30,
        show       : bool  = True,
    ) -> Plot:
        """
        Plot the surface mesh as filled polygons.

        For a 3-D volume mesh, only the dim=2 (surface) elements are drawn;
        interior tetrahedra are not individually shown.  If no surface mesh
        exists, falls back to dim=1 edge elements (useful for 2-D models).

        Parameters
        ----------
        color      : face colour
        edge_color : element edge colour
        alpha      : face opacity
        linewidth  : element outline width
        show       : call ``plt.show()`` at the end

        Returns
        -------
        self — for method chaining
        """
        _, ax = self._ensure_axes()

        # Collect every entity whose mesh elements we can draw.  Both
        # dim=2 (surface polygons) and dim=1 (line segments) are included
        # so mixed models — e.g. slabs + column curves — render both.
        surf_entities = gmsh.model.getEntities(dim=2)
        edge_entities = gmsh.model.getEntities(dim=1)
        plot_entities: list[DimTag] = list(surf_entities) + list(edge_entities)

        if not plot_entities:
            self._log("mesh(): no surface or edge entities found")
            if show:
                self.show()
            return self

        # Build node-tag -> XYZ lookup
        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes(
            dim=-1, tag=-1, includeBoundary=True
        )
        if len(node_tags) == 0:
            self._log("mesh(): no nodes — has the mesh been generated?")
            if show:
                self.show()
            return self

        coords: dict[int, ndarray] = {
            int(t): coords_flat[i * 3:(i + 1) * 3]
            for i, t in enumerate(node_tags)
        }

        polys: list[ndarray] = []
        segments: list[ndarray] = []
        for ent_dim, ent_tag in plot_entities:
            etypes, _, enodes_list = gmsh.model.mesh.getElements(
                dim=ent_dim, tag=ent_tag
            )
            for etype, enodes in zip(etypes, enodes_list):
                n_total, n_corner = self._element_node_counts(etype)
                if n_corner < 2:
                    continue
                elem_count = len(enodes) // n_total
                for k in range(elem_count):
                    all_ns = enodes[k * n_total:(k + 1) * n_total]
                    ns     = all_ns[:n_corner]   # corner nodes only
                    try:
                        verts = np.array([coords[int(ni)] for ni in ns])
                    except KeyError:
                        continue
                    if n_corner == 2:
                        segments.append(verts)
                    else:
                        polys.append(verts)

        all_pts: list[ndarray] = []

        if polys:
            ax.add_collection3d(
                Poly3DCollection(polys,
                                 alpha=alpha,
                                 facecolor=color,
                                 edgecolor=edge_color,
                                 linewidth=linewidth)
            )
            all_pts.append(np.vstack(polys))

        if segments:
            ax.add_collection3d(
                Line3DCollection(segments,
                                 colors=color,
                                 linewidths=max(linewidth, 1.0))
            )
            all_pts.append(np.vstack(segments))

        if all_pts:
            combined = np.vstack(all_pts)
            self._autoscale(combined)
            # Show nodes as scatter points for 1D meshes
            if not polys and segments:
                unique_pts = np.unique(combined, axis=0)
                ax.scatter(
                    unique_pts[:, 0], unique_pts[:, 1], unique_pts[:, 2],
                    c=edge_color, s=12, zorder=5, depthshade=False,
                )

        self._log(f"mesh(): {len(polys)} polygons, {len(segments)} segments")
        if show:
            self.show()
        return self

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def quality(
        self,
        *,
        quality_name : str        = "minSICN",
        cmap         : str        = "RdYlGn",
        vmin         : float | None = None,
        vmax         : float | None = None,
        alpha        : float      = 0.85,
        linewidth    : float      = 0.20,
        show_colorbar: bool       = True,
        show         : bool       = True,
    ) -> Plot:
        """
        Plot surface mesh elements coloured by a quality metric.

        Parameters
        ----------
        quality_name  : gmsh quality metric — ``"minSICN"`` (default),
                        ``"minSIGE"``, ``"gamma"``, ``"minEdge"``,
                        ``"maxEdge"``, ``"minAngle"``, ``"maxAngle"``
        cmap          : matplotlib colormap name
        vmin / vmax   : colormap range clamp (``None`` = data min / max)
        alpha         : face opacity
        linewidth     : element edge width
        show_colorbar : add a colour bar
        show          : call ``plt.show()`` at the end

        Returns
        -------
        self — for method chaining
        """
        fig, ax = self._ensure_axes()

        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes(
            dim=-1, tag=-1, includeBoundary=True
        )
        if len(node_tags) == 0:
            self._log("quality(): no nodes found")
            if show:
                self.show()
            return self

        coords: dict[int, ndarray] = {
            int(t): coords_flat[i * 3:(i + 1) * 3]
            for i, t in enumerate(node_tags)
        }

        polys:     list[ndarray] = []
        qualities: list[float]   = []

        for ent_dim, ent_tag in gmsh.model.getEntities(dim=2):
            etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(
                dim=ent_dim, tag=ent_tag
            )
            for etype, elem_tags_arr, enodes in zip(etypes, etags_list, enodes_list):
                n_total, n_corner = self._element_node_counts(etype)
                if n_corner < 3:
                    continue
                q_vals = gmsh.model.mesh.getElementQualities(
                    list(elem_tags_arr), qualityName=quality_name
                )
                elem_count = len(elem_tags_arr)
                for k in range(elem_count):
                    all_ns = enodes[k * n_total:(k + 1) * n_total]
                    ns     = all_ns[:n_corner]   # corner nodes only
                    try:
                        polys.append(
                            np.array([coords[int(ni)] for ni in ns])
                        )
                        qualities.append(float(q_vals[k]))
                    except KeyError:
                        pass

        if not polys:
            self._log("quality(): no surface elements found")
            if show:
                self.show()
            return self

        q_arr  = np.array(qualities)
        q_min  = float(q_arr.min()) if vmin is None else vmin
        q_max  = float(q_arr.max()) if vmax is None else vmax
        norm   = mcolors.Normalize(vmin=q_min, vmax=q_max)
        cmap_f = cm.get_cmap(cmap)
        face_colors = [cmap_f(norm(q)) for q in qualities]

        ax.add_collection3d(
            Poly3DCollection(polys,
                             alpha=alpha,
                             facecolors=face_colors,
                             edgecolor='k',
                             linewidth=linewidth)
        )
        self._autoscale(np.vstack(polys))

        if show_colorbar:
            sm = cm.ScalarMappable(cmap=cmap_f, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10,
                         label=quality_name)

        self._log(
            f"quality(metric={quality_name!r}, "
            f"range=[{q_min:.4f}, {q_max:.4f}], "
            f"n={len(polys)} elements)"
        )
        if show:
            self.show()
        return self

    # ------------------------------------------------------------------
    # Entity-tag overlay
    # ------------------------------------------------------------------

    def label_entities(
        self,
        *,
        dims    : list[int] | None = None,
        show_dim: bool = True,
        fontsize: int  = 7,
        show    : bool = True,
    ) -> Plot:
        """
        Annotate the current axes with entity tags positioned at each
        entity's geometric centroid.  Works without a mesh.

        Parameters
        ----------
        dims     : dimensions to label (default: all present in model)
        show_dim : prefix label with dimension indicator (P / C / S / V)
        fontsize : annotation font size
        show     : call ``plt.show()`` at the end

        Returns
        -------
        self — for method chaining
        """
        _, ax = self._ensure_axes()
        dim_colors = {
            0: self.COLOR_POINTS,
            1: self.COLOR_CURVES,
            2: self.COLOR_SURFACES,
            3: '#888888',
        }
        target_dims = dims if dims is not None else [0, 1, 2, 3]
        n_labeled = 0

        for d in target_dims:
            for _, tag in gmsh.model.getEntities(dim=d):
                try:
                    bounds = gmsh.model.getParametrizationBounds(d, tag)
                    if d == 0:
                        xyz = np.array(gmsh.model.getValue(0, tag, []))
                    elif d == 1:
                        u_mid = [0.5 * (bounds[0][0] + bounds[1][0])]
                        xyz = np.array(gmsh.model.getValue(1, tag, u_mid))
                    elif d == 2:
                        uv_mid = [
                            0.5 * (bounds[0][i] + bounds[1][i])
                            for i in range(2)
                        ]
                        xyz = np.array(gmsh.model.getValue(2, tag, uv_mid))
                    else:
                        xyz = np.array(gmsh.model.occ.getCenterOfMass(3, tag))

                    label = (f"{_DIM_PREFIX[d]}{tag}"
                             if show_dim else str(tag))
                    ax.text(*xyz, f'  {label}',
                            fontsize=fontsize,
                            color=dim_colors.get(d, 'k'),
                            va='bottom')
                    n_labeled += 1
                except Exception:
                    pass

        self._log(f"label_entities(dims={target_dims}, n={n_labeled})")
        if show:
            self.show()
        return self

    # ------------------------------------------------------------------
    # Mesh node / element labelling
    # ------------------------------------------------------------------

    def label_nodes(
        self,
        *,
        dim     : int  = -1,
        tag     : int  = -1,
        stride  : int  = 1,
        fontsize: int  = 5,
        color   : str  = 'black',
        prefix  : str  = 'n',
        offset  : tuple[float, float, float] | None = None,
        show    : bool = True,
    ) -> Plot:
        """
        Annotate mesh nodes with their tag ids.

        Parameters
        ----------
        dim, tag : restrict to nodes on the given entity
                   (``dim=-1`` returns all nodes; see ``gmsh.model.mesh.getNodes``)
        stride   : label every Nth node (useful for dense meshes)
        fontsize : annotation font size
        color    : text colour
        prefix   : string prepended to each node tag (e.g. ``'n'`` -> ``n17``)
        offset   : (dx, dy, dz) text offset applied to each label; defaults
                   to a small positive x offset so labels do not overlap markers
        show     : call ``plt.show()`` at the end

        Returns
        -------
        self — for method chaining
        """
        _, ax = self._ensure_axes()

        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes(
            dim=dim, tag=tag, includeBoundary=True,
        )
        if len(node_tags) == 0:
            self._log("label_nodes(): no nodes — has the mesh been generated?")
            return self

        xyz = np.asarray(coords_flat).reshape(-1, 3)
        dx, dy, dz = offset if offset is not None else (0.0, 0.0, 0.0)
        step = max(int(stride), 1)
        n_labeled = 0
        for t, p in zip(node_tags[::step], xyz[::step]):
            ax.text(p[0] + dx, p[1] + dy, p[2] + dz,
                    f"  {prefix}{t}", fontsize=fontsize, color=color,
                    va='bottom')
            n_labeled += 1

        self._autoscale(xyz)
        self._log(f"label_nodes(dim={dim}, tag={tag}, stride={step}, "
                  f"n={n_labeled}/{len(node_tags)})")
        if show:
            self.show()
        return self

    def label_elements(
        self,
        *,
        dim     : int  = -1,
        tag     : int  = -1,
        stride  : int  = 1,
        fontsize: int  = 5,
        color   : str  = 'darkred',
        prefix  : str  = 'e',
        show    : bool = True,
    ) -> Plot:
        """
        Annotate mesh elements with their tag ids at element centroids.

        Parameters
        ----------
        dim, tag : restrict to elements on the given entity
                   (``dim=-1`` returns elements of all dimensions)
        stride   : label every Nth element
        fontsize : annotation font size
        color    : text colour
        prefix   : string prepended to each element tag (e.g. ``'e'`` -> ``e42``)
        show     : call ``plt.show()`` at the end

        Returns
        -------
        self — for method chaining
        """
        _, ax = self._ensure_axes()

        # Build node-tag -> XYZ lookup (needed for centroid computation)
        n_tags, n_coords_flat, _ = gmsh.model.mesh.getNodes(
            dim=-1, tag=-1, includeBoundary=True,
        )
        if len(n_tags) == 0:
            self._log("label_elements(): no nodes — mesh not generated?")
            return self
        n_xyz = np.asarray(n_coords_flat).reshape(-1, 3)
        node_lookup = {int(t): n_xyz[i] for i, t in enumerate(n_tags)}

        step = max(int(stride), 1)
        n_labeled = 0

        # Iterate over dims (or just the requested one) and element types
        dims_iter = (dim,) if dim >= 0 else (0, 1, 2, 3)
        for d in dims_iter:
            etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(
                dim=d, tag=tag,
            )
            for etype, elem_tags_arr, enodes in zip(
                    etypes, etags_list, enodes_list):
                n_total, n_corner = self._element_node_counts(etype)
                n_elem = len(elem_tags_arr)
                for k in range(0, n_elem, step):
                    all_ns = enodes[k * n_total:(k + 1) * n_total]
                    corner = all_ns[:n_corner]
                    pts = np.array([node_lookup[int(t)] for t in corner])
                    c = pts.mean(axis=0)
                    etag = int(elem_tags_arr[k])
                    ax.text(c[0], c[1], c[2],
                            f"{prefix}{etag}", fontsize=fontsize,
                            color=color, ha='center', va='center')
                    n_labeled += 1

        self._log(
            f"label_elements(dim={dim}, tag={tag}, "
            f"stride={step}) -> {n_labeled} labels"
        )
        if show:
            self.show()
        return self

    # ------------------------------------------------------------------
    # Figure lifecycle
    # ------------------------------------------------------------------

    def show(self) -> Plot:
        """Flush the current figure to the screen and reset handles."""
        plt.tight_layout()
        plt.show()
        self._fig = None
        self._ax  = None
        return self

    def clear(self) -> Plot:
        """Discard the current figure without showing it."""
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax  = None
        return self

    # ------------------------------------------------------------------
    # Physical groups
    # ------------------------------------------------------------------

    def physical_groups(
        self,
        *,
        dims            : list[int] | None = None,
        names           : list[str] | None = None,
        cmap            : str   = "tab20",
        n_curve_samples : int   = 40,
        n_surf_samples  : int   = 12,
        point_size      : int   = 60,
        linewidth       : float = 2.5,
        surface_alpha   : float = 0.35,
        label_groups    : bool  = True,
        show_legend     : bool  = True,
        show            : bool  = True,
    ) -> Plot:
        """
        Colour BRep entities by the physical group they belong to.

        Iterates over ``gmsh.model.getPhysicalGroups()`` and draws every
        entity in each group with a distinct colour.  Works at the
        geometry level — no mesh required.

        Parameters
        ----------
        dims             : physical-group dimensions to draw (default: all)
        names            : only draw groups whose name is in this list
                           (``None`` draws all groups)
        cmap             : matplotlib colormap used to cycle group colours
        n_curve_samples  : points sampled per curve
        n_surf_samples   : UV grid resolution per surface
        point_size       : marker size for dim=0 groups
        linewidth        : width for dim=1 groups
        surface_alpha    : opacity for dim=2 group patches
        label_groups     : annotate each group at its centroid
        show_legend      : draw a legend mapping colour -> group name
        show             : call ``plt.show()`` at the end

        Returns
        -------
        self — for method chaining
        """
        gmsh.model.occ.synchronize()
        _, ax = self._ensure_axes()

        groups = list(gmsh.model.getPhysicalGroups())
        if dims is not None:
            groups = [(d, t) for (d, t) in groups if d in dims]
        if names is not None:
            name_set = set(names)
            groups = [
                (d, t) for (d, t) in groups
                if gmsh.model.getPhysicalName(d, t) in name_set
            ]

        if not groups:
            self._log("physical_groups(): no groups found")
            if show:
                self.show()
            return self

        cmap_f = cm.get_cmap(cmap)
        collected: list[ndarray] = []
        legend_handles: list = []

        for i, (pg_dim, pg_tag) in enumerate(groups):
            name = gmsh.model.getPhysicalName(pg_dim, pg_tag) or f"PG{pg_tag}"
            color = cmap_f(i % cmap_f.N)
            ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
            centroid_pts: list[ndarray] = []

            # --- dim=0 ---
            if pg_dim == 0:
                for t in ents:
                    xyz = np.array(gmsh.model.getValue(0, int(t), []))
                    ax.scatter(*xyz, color=color, s=point_size,
                               zorder=6, depthshade=False,
                               edgecolors='k', linewidths=0.4)
                    collected.append(xyz.reshape(1, 3))
                    centroid_pts.append(xyz)

            # --- dim=1 ---
            elif pg_dim == 1:
                segs: list[ndarray] = []
                for t in ents:
                    try:
                        lo, hi = gmsh.model.getParametrizationBounds(1, int(t))
                        u = np.linspace(lo[0], hi[0], n_curve_samples)
                        pts = np.array(
                            gmsh.model.getValue(1, int(t), u.tolist())
                        ).reshape(-1, 3)
                        for k in range(len(pts) - 1):
                            segs.append(pts[k:k + 2])
                        collected.append(pts)
                        centroid_pts.append(pts.mean(axis=0))
                    except Exception:
                        pass
                if segs:
                    ax.add_collection3d(
                        Line3DCollection(segs, colors=[color],
                                         linewidths=linewidth)
                    )

            # --- dim=2 ---
            elif pg_dim == 2:
                tris: list[ndarray] = []
                for t in ents:
                    try:
                        # Sample boundary curves (handles trimmed/concave)
                        bnd = gmsh.model.getBoundary(
                            [(2, int(t))], combined=False, oriented=True,
                        )
                        if not bnd:
                            continue
                        bnd_pts: list[ndarray] = []
                        for _, ctag in bnd:
                            abs_ctag = abs(ctag)
                            lo, hi = gmsh.model.getParametrizationBounds(
                                1, abs_ctag,
                            )
                            u = np.linspace(lo[0], hi[0], n_curve_samples)
                            cpts = np.array(
                                gmsh.model.getValue(1, abs_ctag, u.tolist())
                            ).reshape(-1, 3)
                            if ctag < 0:
                                cpts = cpts[::-1]
                            bnd_pts.append(cpts[:-1])
                        if not bnd_pts:
                            continue
                        pts = np.vstack(bnd_pts)
                        # Delaunay triangulation
                        try:
                            from scipy.spatial import Delaunay
                            tri_obj = Delaunay(pts[:, :2])
                            for simplex in tri_obj.simplices:
                                tris.append(pts[simplex])
                        except Exception:
                            centroid = pts.mean(axis=0)
                            n_bnd = len(pts)
                            for i in range(n_bnd):
                                j = (i + 1) % n_bnd
                                tris.append(
                                    np.array([centroid, pts[i], pts[j]])
                                )
                        collected.append(pts)
                        centroid_pts.append(pts.mean(axis=0))
                    except Exception:
                        pass
                if tris:
                    ax.add_collection3d(
                        Poly3DCollection(tris, alpha=surface_alpha,
                                         facecolor=color,
                                         edgecolor='none')
                    )

            # --- dim=3 (centroids only, too expensive to render volumes) ---
            elif pg_dim == 3:
                for t in ents:
                    try:
                        c = np.array(gmsh.model.occ.getCenterOfMass(3, int(t)))
                        ax.scatter(*c, color=color, s=point_size * 2,
                                   marker='X', zorder=6, depthshade=False)
                        centroid_pts.append(c)
                        collected.append(c.reshape(1, 3))
                    except Exception:
                        pass

            # --- label + legend entry ---
            if label_groups and centroid_pts:
                c = np.mean(np.vstack(centroid_pts), axis=0)
                ax.text(c[0], c[1], c[2], f"  {name}",
                        fontsize=7, color='black',
                        ha='left', va='bottom')
            legend_handles.append(
                plt.Line2D([0], [0], marker='s', color='w',
                           markerfacecolor=color, markersize=9,
                           label=f"[{_DIM_PREFIX[pg_dim]}] {name}")
            )

        if collected:
            self._autoscale(np.vstack(collected))

        if show_legend and legend_handles:
            ax.legend(handles=legend_handles, loc='upper right',
                      fontsize=7, framealpha=0.85)

        self._log(f"physical_groups(): drew {len(groups)} group(s)")
        if show:
            self.show()
        return self

    def physical_groups_mesh(
        self,
        *,
        dims         : list[int] | None = None,
        names        : list[str] | None = None,
        cmap         : str   = "tab20",
        alpha        : float = 0.80,
        linewidth    : float = 0.30,
        edge_color   : str   = 'white',
        point_size   : int   = 50,
        seg_width    : float = 2.5,
        show_legend  : bool  = True,
        show         : bool  = True,
    ) -> Plot:
        """
        Colour mesh entities by the physical group they belong to.

        For each physical group, collects the mesh nodes/elements on
        every member entity and renders them in the group's colour.

        Parameters
        ----------
        dims         : physical-group dimensions to draw (default: all)
        names        : only draw groups whose name is in this list
        cmap         : matplotlib colormap used to cycle group colours
        alpha        : face opacity for dim=2 group polygons
        linewidth    : element outline width for dim=2 polygons
        edge_color   : edge colour for dim=2 polygons
        point_size   : scatter size for dim=0 group nodes
        seg_width    : line width for dim=1 group segments
        show_legend  : draw legend mapping colour -> group name
        show         : call ``plt.show()`` at the end

        Returns
        -------
        self — for method chaining
        """
        _, ax = self._ensure_axes()

        groups = list(gmsh.model.getPhysicalGroups())
        if dims is not None:
            groups = [(d, t) for (d, t) in groups if d in dims]
        if names is not None:
            name_set = set(names)
            groups = [
                (d, t) for (d, t) in groups
                if gmsh.model.getPhysicalName(d, t) in name_set
            ]

        if not groups:
            self._log("physical_groups_mesh(): no groups found")
            if show:
                self.show()
            return self

        # Node-tag -> XYZ lookup (global)
        n_tags, n_coords_flat, _ = gmsh.model.mesh.getNodes(
            dim=-1, tag=-1, includeBoundary=True,
        )
        if len(n_tags) == 0:
            self._log("physical_groups_mesh(): no nodes — mesh generated?")
            if show:
                self.show()
            return self
        node_lookup = {
            int(t): np.asarray(n_coords_flat[i * 3:(i + 1) * 3])
            for i, t in enumerate(n_tags)
        }

        cmap_f = cm.get_cmap(cmap)
        collected: list[ndarray] = []
        legend_handles: list = []

        for i, (pg_dim, pg_tag) in enumerate(groups):
            name = gmsh.model.getPhysicalName(pg_dim, pg_tag) or f"PG{pg_tag}"
            color = cmap_f(i % cmap_f.N)
            ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)

            # --- dim=0: scatter the tagged nodes ---
            if pg_dim == 0:
                pts_list: list[ndarray] = []
                for t in ents:
                    nt, ncoord, _ = gmsh.model.mesh.getNodes(
                        dim=0, tag=int(t), includeBoundary=True,
                    )
                    if len(nt) == 0:
                        # fall back to BRep coord if no mesh node
                        xyz = np.array(gmsh.model.getValue(0, int(t), []))
                        pts_list.append(xyz.reshape(1, 3))
                    else:
                        pts_list.append(
                            np.asarray(ncoord).reshape(-1, 3)
                        )
                if pts_list:
                    pts = np.vstack(pts_list)
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                               color=color, s=point_size, zorder=6,
                               depthshade=False, edgecolors='k',
                               linewidths=0.4)
                    collected.append(pts)

            # --- dim=1: line segments per element ---
            elif pg_dim == 1:
                segs: list[ndarray] = []
                for t in ents:
                    etypes, _, enodes_list = gmsh.model.mesh.getElements(
                        dim=1, tag=int(t),
                    )
                    for etype, enodes in zip(etypes, enodes_list):
                        n_total, n_corner = self._element_node_counts(etype)
                        if n_corner < 2:
                            continue
                        n_elem = len(enodes) // n_total
                        for k in range(n_elem):
                            all_ns = enodes[k * n_total:(k + 1) * n_total]
                            ns = all_ns[:n_corner]
                            try:
                                verts = np.array(
                                    [node_lookup[int(ni)] for ni in ns]
                                )
                            except KeyError:
                                continue
                            for s in range(len(verts) - 1):
                                segs.append(verts[s:s + 2])
                if segs:
                    ax.add_collection3d(
                        Line3DCollection(segs, colors=[color],
                                         linewidths=seg_width)
                    )
                    collected.append(np.vstack(segs))

            # --- dim=2: polygon faces per element ---
            elif pg_dim == 2:
                polys: list[ndarray] = []
                for t in ents:
                    etypes, _, enodes_list = gmsh.model.mesh.getElements(
                        dim=2, tag=int(t),
                    )
                    for etype, enodes in zip(etypes, enodes_list):
                        n_total, n_corner = self._element_node_counts(etype)
                        if n_corner < 3:
                            continue
                        n_elem = len(enodes) // n_total
                        for k in range(n_elem):
                            all_ns = enodes[k * n_total:(k + 1) * n_total]
                            ns = all_ns[:n_corner]
                            try:
                                verts = np.array(
                                    [node_lookup[int(ni)] for ni in ns]
                                )
                            except KeyError:
                                continue
                            polys.append(verts)
                if polys:
                    ax.add_collection3d(
                        Poly3DCollection(polys, alpha=alpha,
                                         facecolor=color,
                                         edgecolor=edge_color,
                                         linewidth=linewidth)
                    )
                    collected.append(np.vstack(polys))

            # --- dim=3: skip rendering interior tets; mark COM ---
            elif pg_dim == 3:
                for t in ents:
                    try:
                        c = np.array(gmsh.model.occ.getCenterOfMass(3, int(t)))
                        ax.scatter(*c, color=color, s=point_size * 2,
                                   marker='X', zorder=6, depthshade=False)
                        collected.append(c.reshape(1, 3))
                    except Exception:
                        pass

            legend_handles.append(
                plt.Line2D([0], [0], marker='s', color='w',
                           markerfacecolor=color, markersize=9,
                           label=f"[{_DIM_PREFIX[pg_dim]}] {name}")
            )

        if collected:
            self._autoscale(np.vstack(collected))

        if show_legend and legend_handles:
            ax.legend(handles=legend_handles, loc='upper right',
                      fontsize=7, framealpha=0.85)

        self._log(
            f"physical_groups_mesh(): drew {len(groups)} group(s)"
        )
        if show:
            self.show()
        return self
       