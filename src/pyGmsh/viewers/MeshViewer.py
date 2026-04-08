"""
MeshViewer
==========

Interactive mesh viewer inheriting from :class:`SelectionPicker`.

Renders a Gmsh mesh (nodes + elements) using one VTK actor per BRep entity
(grouping all elements on that entity into a single
``pyvista.UnstructuredGrid``), plus a node-cloud glyph overlay.

SelectionPicker provides all BRep-level picking, hover highlight, recolor,
group management, box-select, and hide/isolate/reveal.  MeshViewer adds
mesh-level picking (element and node modes), multiple coloring modes, label
overlays, renumbering, and partitioning.

Usage
-----
::

    with pyGmsh(model_name="Example") as g:
        g.model.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.generate(3)
        g.mesh.viewer()

Keyboard
--------
    [E]  switch to element picking mode
    [N]  switch to node picking mode
    [Esc] deselect all / return to BRep mode
    [H]  hide selected
    [I]  isolate selected
    [R]  reveal all
    [Q]  close window
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np
import pyvista as pv

from .SelectionPicker import SelectionPicker
from .BaseViewer import DimTag, _hex_to_rgb

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase
    from pyGmsh.mesh.Mesh import Mesh


# ======================================================================
# Gmsh element type -> VTK cell type mapping
# ======================================================================

_GMSH_TO_VTK = {
    1:  3,    # 2-node line          -> VTK_LINE
    2:  5,    # 3-node triangle      -> VTK_TRIANGLE
    3:  9,    # 4-node quad          -> VTK_QUAD
    4:  10,   # 4-node tet           -> VTK_TETRA
    5:  12,   # 8-node hex           -> VTK_HEXAHEDRON
    6:  13,   # 6-node prism         -> VTK_WEDGE
    7:  14,   # 5-node pyramid       -> VTK_PYRAMID
    8:  21,   # 6-node tri (2nd)     -> VTK_QUADRATIC_TRIANGLE
    9:  23,   # 8-node quad (2nd)    -> VTK_QUADRATIC_QUAD
    11: 24,   # 10-node tet (2nd)    -> VTK_QUADRATIC_TETRA
    15: 1,    # 1-node point         -> VTK_VERTEX
}

# Maximum number of labels rendered before camera-distance culling kicks in.
MAX_LABELS = 2000

# ======================================================================
# Colour palettes
# ======================================================================

_PARTITION_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

_ELEM_TYPE_COLORS = {
    "Triangle":      "#4363d8",
    "Quadrilateral": "#3cb44b",
    "Tetrahedron":   "#e6194b",
    "Hexahedron":    "#f58231",
    "Prism":         "#911eb4",
    "Pyramid":       "#42d4f4",
    "Line":          "#aaaaaa",
    "Point":         "#ffffff",
}

_DEFAULT_MESH_COLOR = "#5B8DB8"
_PICK_COLOR         = "#E74C3C"
_HOVER_COLOR        = "#FFD700"
_NODE_COLOR         = "#FF6600"
_EDGE_COLOR         = "#2C4A6E"


# ======================================================================
# Element-type name normalisation
# ======================================================================

def _elem_type_category(name: str) -> str:
    """Map a Gmsh element type name to a colour-palette key."""
    low = name.lower()
    if "triangle" in low:
        return "Triangle"
    if "quadrilateral" in low or "quad" in low:
        return "Quadrilateral"
    if "tetrahedron" in low:
        return "Tetrahedron"
    if "hexahedron" in low:
        return "Hexahedron"
    if "prism" in low:
        return "Prism"
    if "pyramid" in low:
        return "Pyramid"
    if "line" in low:
        return "Line"
    if "point" in low:
        return "Point"
    return "Line"


# ======================================================================
# MeshViewer
# ======================================================================

class MeshViewer(SelectionPicker):
    """
    Interactive mesh viewer inheriting from SelectionPicker.

    Builds one VTK actor per BRep entity (grouping all elements on that
    entity into a single ``pyvista.UnstructuredGrid``), plus a node-cloud
    glyph overlay.

    SelectionPicker's BRep-level picking, hover highlight, recolor, group
    management, box-select, and hide/isolate/reveal all work unchanged on
    mesh actors.  MeshViewer adds element- and node-level picking modes.

    Parameters
    ----------
    parent : _SessionBase
        Owning session instance.
    mesh_composite : Mesh
        The ``Mesh`` composite -- used for renumbering access.
    dims : list[int] or None
        Dimensions to render.  Default ``[1, 2, 3]``.
    point_size : float
        Visual size for the node glyph cloud.
    line_width : float
        VTK line width for 1-D element actors.
    surface_opacity : float
        Opacity for 2-D / 3-D element actors.
    show_surface_edges : bool
        Whether to draw mesh edges on surface actors.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        mesh_composite: "Mesh",
        *,
        dims: list[int] | None = None,
        point_size: float = 1.0,
        line_width: float = 3.0,
        surface_opacity: float = 1.0,
        show_surface_edges: bool = True,
        fast: bool = False,
    ) -> None:
        # SelectionPicker.__init__ expects (parent, model, ...).
        # We pass physical_group=None so no group is auto-written on close.
        super().__init__(
            parent, parent.model,
            physical_group=None,
            dims=dims if dims is not None else [1, 2, 3],
            point_size=point_size,
            line_width=line_width,
            surface_opacity=surface_opacity,
            show_surface_edges=show_surface_edges,
            fast=fast,
        )
        self._mesh = mesh_composite

        # ---- Mesh-level pick mode ----
        # "off" -> BRep picking (super), "element" -> element, "node" -> node
        self._mesh_pick_mode: str = "off"
        self._picked_elems: list[int] = []
        self._picked_nodes: list[int] = []

        # Actor for highlighting picked nodes (sphere glyphs)
        self._node_pick_actor: Any = None

        # ---- Mesh data (populated in _build_scene) ----
        self._node_tags: np.ndarray | None = None       # (N,) node tags
        self._node_coords: np.ndarray | None = None      # (N, 3) xyz
        self._node_tag_to_idx: dict[int, int] = {}       # tag -> row index

        # Element bookkeeping
        self._elem_data: dict[int, dict[str, Any]] = {}  # elem_tag -> info dict
        # {type_name, nodes, dim, brep_dt}

        # ---- BRep-level grids and element mappings ----
        self._brep_grids: dict[DimTag, Any] = {}
        self._brep_to_elems: dict[DimTag, list[int]] = {}  # BRep -> elem tags
        self._elem_to_brep: dict[int, DimTag] = {}          # elem -> parent BRep

        # Per-BRep dominant element type (for type-coloring)
        self._brep_dominant_type: dict[DimTag, str] = {}

        # ---- Node cloud actor (visual overlay, not pickable) ----
        self._node_actor: Any = None
        self._node_cloud: pv.PolyData | None = None

        # ---- Visual properties for mesh ----
        self._node_marker_size: float = point_size
        self._edge_color: str = _EDGE_COLOR

        # ---- Element-level picking ----
        # Maps (brep_dt, cell_idx_in_grid) -> gmsh elem tag
        self._grid_cell_to_elem: dict[tuple[DimTag, int], int] = {}

        # ---- Physical-group <-> BRep mappings ----
        self._group_to_breps: dict[str, list[DimTag]] = {}
        self._brep_to_group: dict[DimTag, str] = {}

        # ---- Color mode ----
        self._color_mode: str = "default"
        # "default", "partition", "quality", "type", "group"

        # ---- Label actors ----
        self._node_label_actor: object | None = None
        self._elem_label_actor: object | None = None

        # ---- Own cell picker ----
        import vtk
        self._click_picker = vtk.vtkCellPicker()
        self._click_picker.SetTolerance(0.005)

        # ---- Opt-1: vectorized tag-to-idx lookup array ----
        self._tag_to_idx: np.ndarray | None = None  # dense lookup, O(max_tag)

        # ---- Opt-2: KD-tree for nearest-node picking ----
        self._node_tree: Any = None  # scipy cKDTree (or None)

        # ---- Opt-7: cached gmsh API calls ----
        self._cached_elem_props: dict[int, tuple] = {}
        self._cached_qualities: dict[DimTag, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Gmsh element-property cache (Opt-7)
    # ------------------------------------------------------------------

    def _get_elem_props(self, etype: int) -> tuple:
        """Return cached ``gmsh.model.mesh.getElementProperties(etype)``."""
        cached = self._cached_elem_props.get(etype)
        if cached is not None:
            return cached
        props = gmsh.model.mesh.getElementProperties(int(etype))
        self._cached_elem_props[etype] = props
        return props

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:  # noqa: C901
        """Populate the plotter with mesh actors plus a node glyph cloud.

        When ``self._fast`` is True, merges all entities per dimension
        into one ``UnstructuredGrid`` (one ``add_mesh`` per dim).
        Otherwise creates one actor per BRep entity (original path).
        """
        import time
        t0 = time.perf_counter()
        plotter = self._plotter

        # 1. Get all mesh nodes
        node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        node_coords = np.asarray(node_coords_flat, dtype=np.float64).reshape(-1, 3)

        self._node_tags = node_tags
        self._node_coords = node_coords
        self._node_tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # --- Opt-1: build dense tag_to_idx lookup array ---
        if len(node_tags) > 0:
            max_tag = int(node_tags.max())
            tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
            tag_to_idx[node_tags] = np.arange(len(node_tags), dtype=np.int64)
            self._tag_to_idx = tag_to_idx
        else:
            self._tag_to_idx = np.array([], dtype=np.int64)

        _default_rgb_u8 = np.array(
            [int(c * 255) for c in _hex_to_rgb(_DEFAULT_MESH_COLOR)],
            dtype=np.uint8,
        )

        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm(
                [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
            ))
            if diag <= 0.0:
                diag = 1.0
        except Exception:
            diag = 1.0
        self._model_diagonal = diag
        t_setup = time.perf_counter() - t0

        # 2. Build actors — dispatch to fast or original path
        t_actors = time.perf_counter()

        if self._fast:
            n_actors = self._build_mesh_actors_batched(
                plotter, node_coords, _default_rgb_u8,
            )
        else:
            n_actors = self._build_mesh_actors_per_entity(
                plotter, node_coords, _default_rgb_u8,
            )

        t_actors_elapsed = time.perf_counter() - t_actors

        # 3. Node cloud
        self._build_node_cloud(plotter, diag)

        # 4. Build physical-group <-> BRep mappings
        self._group_to_breps.clear()
        self._brep_to_group.clear()
        for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                pg_name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
            except Exception:
                pg_name = f"Group_{pg_dim}_{pg_tag}"
            ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
            breps = [(pg_dim, int(t)) for t in ents]
            self._group_to_breps[pg_name] = breps
            for dt in breps:
                if dt not in self._brep_to_group:
                    self._brep_to_group[dt] = pg_name

        # 5. Apply initial coloring
        if self._color_mode != "default":
            self._apply_coloring()

        # ── Profiling ───────────────────────────────────────────────
        total = time.perf_counter() - t0
        n_nodes = len(self._node_tags) if self._node_tags is not None else 0
        entities = {d: len(gmsh.model.getEntities(d)) for d in self._dims}
        mode = "batched" if self._fast else "per-entity"
        print(f"\n[mesh.viewer] Scene built in {total:.2f}s  "
              f"({n_actors} actors, {n_nodes} nodes, {mode})")
        print(f"  Entities: {entities}")
        print(f"  Node setup    : {t_setup:.3f}s")
        print(f"  Actor creation: {t_actors_elapsed:.3f}s")
        print(f"  Remainder     : {total - t_setup - t_actors_elapsed:.3f}s")

    # ------------------------------------------------------------------
    # Actor construction paths
    # ------------------------------------------------------------------

    def _collect_entity_cells(
        self, dim: int, tag: int, node_coords: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int], str] | None:
        """Extract VTK cells for one BRep entity. Returns
        (cell_parts, type_parts, elem_tags, dominant_type_name) or None.
        """
        dt: DimTag = (dim, tag)
        try:
            elem_types, elem_tags_list, elem_node_tags_list = (
                gmsh.model.mesh.getElements(dim, tag)
            )
        except Exception:
            return None
        if not elem_types:
            return None

        all_cells: list[np.ndarray] = []
        all_types: list[np.ndarray] = []
        brep_elem_tags: list[int] = []
        dominant_type_name = ""
        dominant_count = 0

        for etype, etags, enodes in zip(
            elem_types, elem_tags_list, elem_node_tags_list,
        ):
            vtk_type = _GMSH_TO_VTK.get(int(etype))
            if vtk_type is None:
                continue
            props = self._get_elem_props(int(etype))
            type_name: str = props[0]
            n_nodes: int = props[3]

            etags_arr = np.asarray(etags, dtype=np.int64)
            enodes_arr = np.asarray(enodes, dtype=np.int64)
            n_elems = len(etags_arr)
            if n_elems == 0:
                continue
            if n_elems > dominant_count:
                dominant_count = n_elems
                dominant_type_name = type_name

            node_rows = enodes_arr.reshape(n_elems, n_nodes)
            if self._tag_to_idx is not None and len(self._tag_to_idx) > 0:
                max_allowed = len(self._tag_to_idx) - 1
                in_range = (
                    np.all(node_rows <= max_allowed)
                    and np.all(node_rows >= 0)
                )
                if in_range:
                    idx_arr = self._tag_to_idx[node_rows]
                    valid_mask = np.all(idx_arr >= 0, axis=1)
                else:
                    clipped = np.clip(node_rows, 0, max_allowed)
                    idx_arr = self._tag_to_idx[clipped]
                    valid_mask = (
                        np.all(idx_arr >= 0, axis=1)
                        & np.all(node_rows >= 0, axis=1)
                        & np.all(node_rows <= max_allowed, axis=1)
                    )
            else:
                idx_arr = np.zeros_like(node_rows)
                valid_mask = np.zeros(n_elems, dtype=bool)

            valid_idx_arr = idx_arr[valid_mask]
            valid_etags = etags_arr[valid_mask]
            n_valid = int(valid_mask.sum())
            if n_valid == 0:
                continue

            prefix = np.full((n_valid, 1), n_nodes, dtype=np.int64)
            all_cells.append(np.hstack([prefix, valid_idx_arr]).ravel())
            all_types.append(np.full(n_valid, vtk_type, dtype=np.uint8))

            valid_node_rows = node_rows[valid_mask]
            for ri in range(n_valid):
                elem_tag = int(valid_etags[ri])
                brep_elem_tags.append(elem_tag)
                self._elem_to_brep[elem_tag] = dt
                self._elem_data[elem_tag] = {
                    "type_name": type_name,
                    "nodes": valid_node_rows[ri].tolist(),
                    "dim": dim,
                    "brep_dt": dt,
                }

        if not all_cells:
            return None
        return all_cells, all_types, brep_elem_tags, dominant_type_name

    def _build_mesh_actors_per_entity(
        self, plotter, node_coords, default_rgb,
    ) -> int:
        """Original path: one add_mesh per BRep entity."""
        n_actors = 0
        for dim in sorted(self._dims):
            for _, tag in gmsh.model.getEntities(dim=dim):
                dt: DimTag = (dim, tag)
                result = self._collect_entity_cells(dim, tag, node_coords)
                if result is None:
                    continue
                all_cells, all_types, brep_elem_tags, dom_type = result

                for cell_idx, etag in enumerate(brep_elem_tags):
                    self._grid_cell_to_elem[(dt, cell_idx)] = etag
                self._brep_to_elems[dt] = brep_elem_tags
                self._brep_dominant_type[dt] = _elem_type_category(dom_type)

                cells_flat = np.concatenate(all_cells)
                cell_types_flat = np.concatenate(all_types)
                grid = pv.UnstructuredGrid(
                    cells_flat, cell_types_flat, node_coords,
                )
                self._brep_grids[dt] = grid
                grid["colors"] = np.tile(default_rgb, (grid.n_cells, 1))

                show_edges = self._show_surface_edges and dim >= 2
                opacity = self._surface_opacity if dim >= 2 else 1.0
                actor = plotter.add_mesh(
                    grid, scalars="colors", rgb=True,
                    opacity=opacity,
                    show_edges=show_edges,
                    edge_color=_EDGE_COLOR,
                    line_width=self._line_width if dim == 1 else 0.5,
                    render_lines_as_tubes=(dim == 1),
                    smooth_shading=False, pickable=True,
                )
                self._register_actor(actor, dt)
                n_actors += 1
        return n_actors

    def _build_mesh_actors_batched(
        self, plotter, node_coords, default_rgb,
    ) -> int:
        """Fast path: merge all entities per dim into one grid."""
        self._batched = True
        self._batch_actors = {}
        self._batch_meshes = {}
        self._batch_cell_to_dt = {}
        self._batch_dt_to_cells = {}
        self._batch_centroids = {}
        # For mesh-level element picking in batched mode
        self._batch_cell_to_elem: dict[int, dict[int, int]] = {}  # dim → {cell_idx: elem_tag}

        n_actors = 0
        for dim in sorted(self._dims):
            all_cells_parts: list[np.ndarray] = []
            all_types_parts: list[np.ndarray] = []
            all_entity_tags: list[int] = []  # per-cell entity tag
            all_elem_tags_flat: list[int] = []  # per-cell elem tag
            cell_to_dt: dict[int, DimTag] = {}
            cell_offset = 0

            for _, tag in gmsh.model.getEntities(dim=dim):
                dt: DimTag = (dim, tag)
                result = self._collect_entity_cells(dim, tag, node_coords)
                if result is None:
                    continue
                cell_parts, type_parts, brep_elem_tags, dom_type = result

                n_entity_cells = len(brep_elem_tags)
                self._brep_to_elems[dt] = brep_elem_tags
                self._brep_dominant_type[dt] = _elem_type_category(dom_type)

                cell_indices = list(range(cell_offset, cell_offset + n_entity_cells))
                self._batch_dt_to_cells[dt] = cell_indices
                for ci in cell_indices:
                    cell_to_dt[ci] = dt

                all_cells_parts.extend(cell_parts)
                all_types_parts.extend(type_parts)
                all_entity_tags.extend([tag] * n_entity_cells)
                all_elem_tags_flat.extend(brep_elem_tags)

                # Compute centroid from element node positions
                try:
                    ntags, ncoords, _ = gmsh.model.mesh.getNodes(
                        dim=dim, tag=tag, includeBoundary=True,
                    )
                    if len(ntags) > 0:
                        pts = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
                        self._batch_centroids[dt] = pts.mean(axis=0)
                except Exception:
                    pass

                cell_offset += n_entity_cells

            if not all_cells_parts:
                continue

            cells_flat = np.concatenate(all_cells_parts)
            cell_types_flat = np.concatenate(all_types_parts)
            grid = pv.UnstructuredGrid(
                cells_flat, cell_types_flat, node_coords,
            )
            colors = np.tile(default_rgb, (grid.n_cells, 1))
            grid["colors"] = colors
            grid.cell_data["entity_tag"] = np.array(all_entity_tags, dtype=np.int64)

            show_edges = self._show_surface_edges and dim >= 2
            opacity = self._surface_opacity if dim >= 2 else 1.0
            actor = plotter.add_mesh(
                grid, scalars="colors", rgb=True,
                opacity=opacity,
                show_edges=show_edges,
                edge_color=_EDGE_COLOR,
                line_width=self._line_width if dim == 1 else 0.5,
                render_lines_as_tubes=(dim == 1),
                smooth_shading=False, pickable=True,
                reset_camera=False,
            )

            self._batch_actors[dim] = actor
            self._batch_meshes[dim] = grid
            self._batch_cell_to_dt[dim] = cell_to_dt
            self._batch_cell_to_elem[dim] = {
                i: etag for i, etag in enumerate(all_elem_tags_flat)
            }

            # Register entities for picking
            for dt in self._batch_dt_to_cells:
                if dt[0] == dim:
                    self._id_to_actor[dt] = actor
            self._actor_to_id[id(actor)] = (dim, -1)
            n_actors += 1

        plotter.reset_camera()
        return n_actors

    def _build_node_cloud(self, plotter, diag: float) -> None:
        """Build filtered node cloud with glyph overlay and KD-tree."""
        used_node_tags: set[int] = set()
        for elem_tags_list in self._brep_to_elems.values():
            for etag in elem_tags_list:
                info = self._elem_data.get(etag)
                if info is not None:
                    used_node_tags.update(info["nodes"])

        if self._node_tags is not None and len(self._node_tags) > 0:
            mask = np.isin(self._node_tags, list(used_node_tags))
            filtered_tags = self._node_tags[mask]
            filtered_coords = self._node_coords[mask]
            self._node_tags = filtered_tags
            self._node_coords = filtered_coords
            self._node_tag_to_idx = {
                int(t): i for i, t in enumerate(filtered_tags)
            }

        if self._node_coords is not None and len(self._node_coords) > 0:
            node_cloud = pv.PolyData(self._node_coords)
            self._node_cloud = node_cloud
            glyph_radius = 0.004 * diag * self._node_marker_size
            sphere_src = pv.Sphere(
                radius=glyph_radius,
                theta_resolution=10, phi_resolution=10,
            )
            glyphs = node_cloud.glyph(
                geom=sphere_src, orient=False, scale=False,
            )
            self._node_actor = plotter.add_mesh(
                glyphs, color=_NODE_COLOR,
                smooth_shading=True, pickable=False, opacity=1.0,
            )

        if self._node_coords is not None and len(self._node_coords) > 0:
            try:
                from scipy.spatial import cKDTree
                self._node_tree = cKDTree(self._node_coords)
            except ImportError:
                self._node_tree = None

    # ------------------------------------------------------------------
    # Mesh-level pick mode
    # ------------------------------------------------------------------

    def _set_mesh_pick_mode(self, mode: str) -> None:
        """Switch mesh pick mode: ``"off"`` | ``"element"`` | ``"node"``."""
        self._mesh_pick_mode = mode
        self._update_status()
        self._fire_pick_changed()

    # ------------------------------------------------------------------
    # Keybindings
    # ------------------------------------------------------------------

    def _install_keybindings(self) -> None:
        super()._install_keybindings()
        plotter = self._plotter
        plotter.add_key_event("e", lambda: self._set_mesh_pick_mode("element"))
        plotter.add_key_event("E", lambda: self._set_mesh_pick_mode("element"))
        plotter.add_key_event("n", lambda: self._set_mesh_pick_mode("node"))
        plotter.add_key_event("N", lambda: self._set_mesh_pick_mode("node"))

    # ------------------------------------------------------------------
    # Picker hooks (override SelectionPicker)
    # ------------------------------------------------------------------

    def _on_lmb_click(self, x: int, y: int, ctrl: bool) -> None:
        """Handle a left-mouse-button click at display coords *(x, y)*.

        If mesh-pick mode is ``"off"``, delegate to SelectionPicker for
        BRep-level picking.  Otherwise handle element or node picking.
        """
        if self._mesh_pick_mode == "off":
            super()._on_lmb_click(x, y, ctrl)
            return

        renderer = self._plotter.renderer
        self._click_picker.Pick(x, y, 0, renderer)
        prop = self._click_picker.GetViewProp()

        if self._mesh_pick_mode == "element":
            if prop is None:
                return
            entity_id = self._actor_to_id.get(id(prop))
            if entity_id is None:
                return
            cell_id = self._click_picker.GetCellId()
            if cell_id < 0:
                return
            # entity_id is a DimTag from _register_actor
            if not isinstance(entity_id, tuple) or len(entity_id) != 2:
                return
            brep_dt: DimTag = entity_id
            elem_tag = self._grid_cell_to_elem.get((brep_dt, cell_id))
            if elem_tag is None:
                return
            if ctrl:
                if elem_tag in self._picked_elems:
                    self._picked_elems.remove(elem_tag)
            else:
                if elem_tag in self._picked_elems:
                    self._picked_elems.remove(elem_tag)
                else:
                    self._picked_elems.append(elem_tag)
            self._highlight_picked_elems()
            self._update_status()
            self._fire_pick_changed()

        elif self._mesh_pick_mode == "node":
            pos = self._click_picker.GetPickPosition()
            nearest = self._find_nearest_node(pos)
            if nearest is None:
                return
            if ctrl:
                if nearest in self._picked_nodes:
                    self._picked_nodes.remove(nearest)
            else:
                if nearest in self._picked_nodes:
                    self._picked_nodes.remove(nearest)
                else:
                    self._picked_nodes.append(nearest)
            self._highlight_picked_nodes()
            self._update_status()
            self._fire_pick_changed()

    def _on_box_select(
        self,
        x0: int, y0: int, x1: int, y1: int,
        crossing: bool,
        ctrl: bool,
    ) -> None:
        """Box-select override: node mode uses batch matrix projection,
        otherwise delegate to SelectionPicker."""
        if self._mesh_pick_mode == "node":
            # DPI scaling
            try:
                rw = self._plotter.render_window
                vw, vh = rw.GetSize()
                aw, ah = rw.GetActualSize()
                sx_ratio = aw / vw if vw else 1.0
                sy_ratio = ah / vh if vh else 1.0
            except Exception:
                sx_ratio = sy_ratio = 1.0
            bx0, bx1 = x0 * sx_ratio, x1 * sx_ratio
            by0, by1 = y0 * sy_ratio, y1 * sy_ratio
            self._box_select_nodes(bx0, by0, bx1, by1, ctrl)
        else:
            super()._on_box_select(x0, y0, x1, y1, crossing, ctrl)

    def _box_select_nodes(
        self,
        x0: float, y0: float, x1: float, y1: float,
        ctrl: bool,
    ) -> None:
        """Select/unselect nodes whose screen projection falls in the box.

        Uses batch matrix projection via camera composite matrix (Opt-5).
        """
        if self._node_coords is None or self._node_tags is None:
            return

        renderer = self._plotter.renderer
        rw = self._plotter.render_window
        cam = renderer.GetActiveCamera()
        proj_matrix = cam.GetCompositeProjectionTransformMatrix(
            renderer.GetTiledAspectRatio(), 0, 1,
        )
        M = np.array(
            [[proj_matrix.GetElement(i, j) for j in range(4)] for i in range(4)],
            dtype=np.float64,
        )

        n = len(self._node_coords)
        ones = np.ones((n, 1), dtype=np.float64)
        world4 = np.hstack([self._node_coords, ones])
        clip = world4 @ M.T
        w_clip = clip[:, 3].copy()
        w_clip[w_clip == 0] = 1e-12  # avoid div by zero
        ndc = clip[:, :2] / w_clip[:, None]

        w, h = rw.GetSize()
        sx = (ndc[:, 0] + 1.0) * 0.5 * w
        sy = (ndc[:, 1] + 1.0) * 0.5 * h

        mask = (x0 <= sx) & (sx <= x1) & (y0 <= sy) & (sy <= y1)
        hit_indices = np.nonzero(mask)[0]

        changed = False
        selected_set = set(self._picked_nodes)

        for i in hit_indices:
            tag = int(self._node_tags[i])
            if ctrl:
                if tag in selected_set:
                    self._picked_nodes.remove(tag)
                    selected_set.discard(tag)
                    changed = True
            else:
                if tag not in selected_set:
                    self._picked_nodes.append(tag)
                    selected_set.add(tag)
                    changed = True

        if changed:
            self._highlight_picked_nodes()
            self._update_status()
            self._fire_pick_changed()

    # ------------------------------------------------------------------
    # Hover
    # ------------------------------------------------------------------

    def _on_hover_changed_internal(self, old_id: Any, new_id: Any) -> None:
        """Recolor actors on hover transition."""
        if old_id is not None:
            self._recolor(old_id)
        if new_id is not None and new_id in self._id_to_actor:
            if new_id not in self._picks:
                actor = self._id_to_actor[new_id]
                actor.GetProperty().SetColor(*_hex_to_rgb(_HOVER_COLOR))
                self._plotter.render()

    # ------------------------------------------------------------------
    # Node utilities
    # ------------------------------------------------------------------

    def _find_nearest_node(
        self, world_pos: tuple[float, float, float],
    ) -> int | None:
        """Return the node tag closest to *world_pos*, or ``None``."""
        if self._node_coords is None or len(self._node_coords) == 0:
            return None

        pos = np.asarray(world_pos, dtype=np.float64)

        # --- Opt-2: KD-tree query ---
        if self._node_tree is not None:
            _, idx = self._node_tree.query(pos)
            return int(self._node_tags[idx])

        # Fallback: brute-force numpy
        dists = np.linalg.norm(self._node_coords - pos, axis=1)
        idx = int(np.argmin(dists))
        return int(self._node_tags[idx])

    # ------------------------------------------------------------------
    # Mesh-level highlighting
    # ------------------------------------------------------------------

    def _highlight_picked_elems(self) -> None:
        """Refresh BRep grid colors to highlight picked elements.

        In-place per-cell color update on BRep grids (Opt-4).
        """
        picked_set = set(self._picked_elems)
        pick_rgb = np.array(
            [int(c * 255) for c in _hex_to_rgb(_PICK_COLOR)], dtype=np.uint8,
        )

        for dt, grid in self._brep_grids.items():
            elem_tags = self._brep_to_elems.get(dt, [])
            if not elem_tags:
                continue
            n_cells = grid.n_cells
            # Start from the current color-mode base color
            base_color = self._get_idle_color_u8(dt)
            colors = np.tile(base_color, (n_cells, 1))
            for ci, etag in enumerate(elem_tags):
                if ci < n_cells and etag in picked_set:
                    colors[ci] = pick_rgb

            grid["colors"] = colors
            grid.Modified()

        self._plotter.render()

    def _highlight_picked_nodes(self) -> None:
        """Sphere glyph overlay at picked node positions."""
        # Remove previous pick actor
        if self._node_pick_actor is not None:
            try:
                self._plotter.remove_actor(self._node_pick_actor)
            except Exception:
                pass
            self._node_pick_actor = None

        if not self._picked_nodes or self._node_coords is None:
            self._plotter.render()
            return

        # Gather picked node positions
        picked_positions = []
        for tag in self._picked_nodes:
            idx = self._node_tag_to_idx.get(tag)
            if idx is not None:
                picked_positions.append(self._node_coords[idx])

        if not picked_positions:
            self._plotter.render()
            return

        pts = np.array(picked_positions)
        cloud = pv.PolyData(pts)
        glyph_radius = 0.006 * self._model_diagonal * self._node_marker_size
        sphere_src = pv.Sphere(
            radius=glyph_radius,
            theta_resolution=12,
            phi_resolution=12,
        )
        glyphs = cloud.glyph(geom=sphere_src, orient=False, scale=False)
        self._node_pick_actor = self._plotter.add_mesh(
            glyphs,
            color=_PICK_COLOR,
            smooth_shading=True,
            pickable=False,
            opacity=1.0,
        )
        self._plotter.render()

    def _get_idle_color_u8(self, dt: DimTag) -> np.ndarray:
        """Return the uint8 RGB for an idle (unpicked) BRep entity
        under the current color mode."""
        if self._color_mode == "type":
            cat = self._brep_dominant_type.get(dt, "Line")
            hex_color = _ELEM_TYPE_COLORS.get(cat, _DEFAULT_MESH_COLOR)
        elif self._color_mode == "partition":
            try:
                parts = gmsh.model.mesh.getPartitions(dt[0], dt[1])
                if parts:
                    idx = int(parts[0]) % len(_PARTITION_COLORS)
                    hex_color = _PARTITION_COLORS[idx]
                else:
                    hex_color = _DEFAULT_MESH_COLOR
            except Exception:
                hex_color = _DEFAULT_MESH_COLOR
        elif self._color_mode == "group":
            hex_color = self._get_group_color_hex(dt)
        else:
            hex_color = _DEFAULT_MESH_COLOR
        return np.array(
            [int(c * 255) for c in _hex_to_rgb(hex_color)], dtype=np.uint8,
        )

    def _get_group_color_hex(self, dt: DimTag) -> str:
        """Return the hex color for a BRep entity based on its physical
        group membership."""
        try:
            groups = gmsh.model.getPhysicalGroups()
        except Exception:
            return _DEFAULT_MESH_COLOR
        for pg_dim, pg_tag in groups:
            if pg_dim != dt[0]:
                continue
            try:
                ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
                if dt[1] in ents:
                    idx = pg_tag % len(_PARTITION_COLORS)
                    return _PARTITION_COLORS[idx]
            except Exception:
                continue
        return _DEFAULT_MESH_COLOR

    # ------------------------------------------------------------------
    # _recolor override (SelectionPicker virtual)
    # ------------------------------------------------------------------

    def _recolor(self, dt: DimTag) -> None:
        """Recolor a single BRep actor based on pick / hover / color-mode
        state.  Picked elements take priority over hover; hover takes
        priority over idle color-mode color."""
        actor = self._id_to_actor.get(dt)
        if actor is None:
            return

        if dt == self._hover_id and dt not in self._picks:
            actor.GetProperty().SetColor(*_hex_to_rgb(_HOVER_COLOR))
        elif dt in self._picks:
            actor.GetProperty().SetColor(*_hex_to_rgb(_PICK_COLOR))
        else:
            self._apply_coloring_single(dt, actor)

        self._plotter.render()

    # ------------------------------------------------------------------
    # Clear override
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all picks and mesh-level selections."""
        # Clear BRep picks via parent
        super().clear()
        # Clear mesh-level picks
        self._picked_elems.clear()
        self._picked_nodes.clear()
        # Remove node pick overlay
        if self._node_pick_actor is not None:
            try:
                self._plotter.remove_actor(self._node_pick_actor)
            except Exception:
                pass
            self._node_pick_actor = None
        # Reset mesh pick mode
        self._mesh_pick_mode = "off"
        # Reapply coloring to clear element highlights
        self._apply_coloring()
        self._plotter.render()

    # ------------------------------------------------------------------
    # Color mode
    # ------------------------------------------------------------------

    def _set_color_mode(self, mode: str) -> None:
        """Switch the color mode and reapply."""
        self._color_mode = mode
        self._apply_coloring()

    def _apply_coloring(self) -> None:
        """Recolor all BRep actors according to the current color mode."""
        if self._color_mode == "partition":
            self._color_by_partition()
        elif self._color_mode == "quality":
            self._color_by_quality()
        elif self._color_mode == "type":
            self._color_by_type()
        elif self._color_mode == "group":
            self._color_by_group()
        else:
            self._color_default()
        self._plotter.render()

    def _apply_coloring_single(self, dt: DimTag, actor: object) -> None:
        """Apply the current color mode to a single actor (used after
        un-picking to restore the correct base color)."""
        if self._color_mode == "partition":
            self._color_single_partition(dt, actor)
        elif self._color_mode == "type":
            self._color_single_type(dt, actor)
        elif self._color_mode == "group":
            self._color_single_group(dt, actor)
        else:
            actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))
            actor.GetProperty().SetOpacity(self._surface_opacity)

    # ---- default ----

    def _color_default(self) -> None:
        for dt, actor in self._id_to_actor.items():
            if dt not in self._picks:
                actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))
                actor.GetProperty().SetOpacity(self._surface_opacity)

    # ---- partition ----

    def _color_by_partition(self) -> None:
        for dt, actor in self._id_to_actor.items():
            if dt not in self._picks:
                self._color_single_partition(dt, actor)

    def _color_single_partition(self, dt: DimTag, actor: object) -> None:
        try:
            parts = gmsh.model.mesh.getPartitions(dt[0], dt[1])
            if parts:
                idx = int(parts[0]) % len(_PARTITION_COLORS)
                actor.GetProperty().SetColor(
                    *_hex_to_rgb(_PARTITION_COLORS[idx]),
                )
            else:
                actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))
        except Exception:
            actor.GetProperty().SetColor(*_hex_to_rgb(_DEFAULT_MESH_COLOR))

    # ---- quality ----

    def _color_by_quality(self) -> None:
        """Scalar-field coloring using element quality."""
        for dt in list(self._id_to_actor.keys()):
            actor = self._id_to_actor.get(dt)
            if actor is None or dt in self._picks:
                continue
            grid = self._brep_grids.get(dt)
            if grid is None:
                continue
            etags = self._brep_to_elems.get(dt, [])
            if not etags:
                continue
            try:
                # --- Opt-7: lazy quality cache ---
                q_arr = self._cached_qualities.get(dt)
                if q_arr is None:
                    qualities = gmsh.model.mesh.getElementQualities(
                        np.array(etags, dtype=np.int64),
                    )
                    q_arr = np.asarray(qualities, dtype=np.float64)
                    self._cached_qualities[dt] = q_arr

                if len(q_arr) == grid.n_cells:
                    grid.cell_data["quality"] = q_arr
                    # Remove old actor and re-add with scalar coloring
                    try:
                        self._plotter.remove_actor(actor)
                    except Exception:
                        pass
                    new_actor = self._plotter.add_mesh(
                        grid,
                        scalars="quality",
                        cmap="RdYlGn",
                        clim=[0.0, 1.0],
                        show_edges=self._show_surface_edges,
                        edge_color=_EDGE_COLOR,
                        line_width=0.5,
                        pickable=True,
                    )
                    self._register_actor(new_actor, dt)
            except Exception:
                pass

    # ---- type ----

    def _color_by_type(self) -> None:
        for dt, actor in self._id_to_actor.items():
            if dt not in self._picks:
                self._color_single_type(dt, actor)

    def _color_single_type(self, dt: DimTag, actor: object) -> None:
        cat = self._brep_dominant_type.get(dt, "Line")
        hex_color = _ELEM_TYPE_COLORS.get(cat, _DEFAULT_MESH_COLOR)
        actor.GetProperty().SetColor(*_hex_to_rgb(hex_color))

    # ---- group (physical group membership) ----

    def _color_by_group(self) -> None:
        for dt, actor in self._id_to_actor.items():
            if dt not in self._picks:
                self._color_single_group(dt, actor)

    def _color_single_group(self, dt: DimTag, actor: object) -> None:
        """Color a BRep actor by physical-group membership.  Entities
        belonging to a physical group get a palette colour; others stay
        at the default."""
        hex_color = self._get_group_color_hex(dt)
        actor.GetProperty().SetColor(*_hex_to_rgb(hex_color))

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def _toggle_node_labels(self, show: bool) -> None:
        """Show or hide node-tag labels at every mesh node."""
        if self._node_label_actor is not None:
            try:
                self._plotter.remove_actor(self._node_label_actor)
            except Exception:
                pass
            self._node_label_actor = None

        if show and self._node_tags is not None and len(self._node_tags) > 0:
            labels = [str(int(t)) for t in self._node_tags]
            pts = self._node_coords

            # --- Opt-8: label cap ---
            if len(labels) > MAX_LABELS:
                cam_pos = np.array(
                    self._plotter.camera_position[0], dtype=np.float64,
                )
                dists = np.linalg.norm(self._node_coords - cam_pos, axis=1)
                closest = np.argsort(dists)[:MAX_LABELS]
                labels = [labels[i] for i in closest]
                pts = self._node_coords[closest]

            self._node_label_actor = self._plotter.add_point_labels(
                pts,
                labels,
                font_size=8,
                font_family="arial",
                text_color="white",
                show_points=False,
                always_visible=True,
                name="_mesh_node_labels",
            )
        self._plotter.render()

    def _toggle_elem_labels(self, show: bool) -> None:
        """Show or hide element-tag labels at element centroids."""
        if self._elem_label_actor is not None:
            try:
                self._plotter.remove_actor(self._elem_label_actor)
            except Exception:
                pass
            self._elem_label_actor = None

        if show and self._elem_data:
            centroids = []
            labels = []
            for etag, info in self._elem_data.items():
                node_tags = info["nodes"]
                pts = []
                for nt in node_tags:
                    idx = self._node_tag_to_idx.get(nt)
                    if idx is not None:
                        pts.append(self._node_coords[idx])
                if pts:
                    centroid = np.mean(pts, axis=0)
                    centroids.append(centroid)
                    labels.append(str(etag))

            if centroids:
                centroids_arr = np.array(centroids)

                # --- Opt-8: label cap ---
                if len(labels) > MAX_LABELS:
                    cam_pos = np.array(
                        self._plotter.camera_position[0], dtype=np.float64,
                    )
                    dists = np.linalg.norm(
                        centroids_arr - cam_pos, axis=1,
                    )
                    closest = np.argsort(dists)[:MAX_LABELS]
                    labels = [labels[i] for i in closest]
                    centroids_arr = centroids_arr[closest]

                self._elem_label_actor = self._plotter.add_point_labels(
                    centroids_arr,
                    labels,
                    font_size=7,
                    font_family="arial",
                    text_color="yellow",
                    show_points=False,
                    always_visible=True,
                    name="_mesh_elem_labels",
                )
        self._plotter.render()

    # ------------------------------------------------------------------
    # Renumbering
    # ------------------------------------------------------------------

    def _apply_renumbering(self, method: str = "rcmk") -> None:
        """Renumber the mesh and rebuild the scene.

        Delegates to the Mesh composite's renumber methods, then fully
        rebuilds the viewer scene with the new numbering.
        """
        # Use Numberer if available for advanced methods
        try:
            from pyGmsh.solvers.Numberer import Numberer
            fem = self._mesh.get_fem_data()
            numb = Numberer(fem)
            numb.renumber(method=method)
        except Exception:
            pass

        # Rebuild scene
        self._actor_to_id.clear()
        self._id_to_actor.clear()
        self._brep_grids.clear()
        self._brep_to_elems.clear()
        self._elem_to_brep.clear()
        self._elem_data.clear()
        self._brep_dominant_type.clear()
        self._grid_cell_to_elem.clear()
        self._node_actor = None
        self._node_cloud = None
        self._node_tree = None
        self._cached_elem_props.clear()
        self._cached_qualities.clear()
        self._tag_to_idx = None
        self._node_label_actor = None
        self._elem_label_actor = None
        self._picked_nodes.clear()
        self._picked_elems.clear()
        if self._node_pick_actor is not None:
            try:
                self._plotter.remove_actor(self._node_pick_actor)
            except Exception:
                pass
            self._node_pick_actor = None
        self._picks.clear()
        self._plotter.clear()
        self._build_scene()
        self._plotter.render()

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def _apply_partitioning(self, n_partitions: int) -> None:
        """Partition the mesh and switch to partition coloring."""
        if hasattr(self._parent, "partition") and self._parent.partition:
            self._parent.partition.auto(n_partitions)
            self._set_color_mode("partition")

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        """Refresh the HUD text on the plotter."""
        try:
            self._plotter.remove_actor("_picker_hud")
        except Exception:
            pass

        if self._mesh_pick_mode == "element":
            n = len(self._picked_elems)
            text = f"Mode: Element | Elements picked: {n}"
        elif self._mesh_pick_mode == "node":
            n = len(self._picked_nodes)
            text = f"Mode: Node | Nodes picked: {n}"
        else:
            n = len(self._picks)
            text = f"Mode: BRep | Patches selected: {n}"

        try:
            self._plotter.add_text(
                text,
                position="lower_left",
                font_size=9,
                color="white",
                name="_picker_hud",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def _create_window(self, *, title: str, maximized: bool):
        from .MeshViewerUI import MeshViewerWindow
        return MeshViewerWindow(self, title=title, maximized=maximized)

    def show(
        self, *, title: str | None = None, maximized: bool = True,
    ) -> "MeshViewer":
        """Open the mesh viewer window.  Blocks until closed."""
        gmsh.model.occ.synchronize()
        default_title = (
            f"MeshViewer \u2014 {self._parent.model_name} - Ladru\u00f1o"
        )
        window = self._create_window(
            title=title or default_title, maximized=maximized,
        )
        window.exec()
        self._on_window_closed()
        return self

    def _on_window_closed(self) -> None:
        """Cleanup after the viewer window closes."""
        if self._parent._verbose:
            print(
                f"[MeshViewer] closed \u2014 "
                f"{len(self._picked_nodes)} nodes picked, "
                f"{len(self._picked_elems)} elements picked, "
                f"{len(self._picks)} BRep patches selected"
            )
