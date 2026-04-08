"""
SelectionPicker
===============

Headless core of the interactive BRep picker.  Manages the VTK scene
(one actor per entity), pick state, hide/isolate, box-select, hover
highlight and keybindings -- but does **not** own the window.  The Qt
UI lives in ``SelectionPickerUI`` and is opened by ``.show()``.

Usage
-----
    # General workflow -- open the viewer, create / rename / delete
    # groups from the toolbar or tree right-click menu, done.
    m1.model.selection.picker()

    # Shortcut: pre-activate an existing group for editing its members.
    m1.model.selection.picker(physical_group="base_supports")

    # Programmatic access after close:
    p = m1.model.selection.picker()
    sel = p.selection

Inside the window
-----------------
    Mouse:        LEFT click        pick entity (pixel-accurate)
                  LEFT drag         box-select (L->R window, R->L crossing)
                  Ctrl+LEFT click   unpick entity under cursor
                  Ctrl+LEFT drag    box-UNselect
                  MIDDLE drag       pan camera
                  Shift+MIDDLE drag rotate camera (orbit)
                  RIGHT drag        pan camera
                  WHEEL             zoom
                  hover             highlight entity under cursor (gold)
    Pick filter:  [1] points  [2] curves  [3] surfaces  [4] volumes  [0] all
    Visibility:   [H] hide picks  [I] isolate picks  [R] reveal all
    Edit:         [U] undo    [Tab] cycle overlapping entities
                  [Esc] deselect all    [Q] close window

The right-side panel shows a model tree (physical groups + unassigned
entities with labels from ``Model._registry``), a preferences dock, and
a toolbar with New/Rename/Delete group and a parallel / perspective
projection toggle.  On close, every group staged during the session
is flushed to Gmsh (existing groups with the same name are replaced).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh
import numpy as np

from .BaseViewer import BaseViewer, DimTag, _hex_to_rgb

if TYPE_CHECKING:
    from pyGmsh._session import _SessionBase
    from pyGmsh.core.Model import Model
    from pyGmsh.viz.Selection import Selection


# ======================================================================
# Module-level BRep colour constants
# ======================================================================

# Picked entities get this highlight color
# Palette is protanopia-safe: states separate on the blue / yellow
# axis (fully preserved by protanopia) plus luminance contrast.
_PICK_COLOR       = "#E74C3C"   # red -- all entity types
_PICK_COLOR_POINT = "#E74C3C"   # red -- same as above (unified)
_HOVER_COLOR      = "#FFD700"   # gold  -- yellow axis, distinct from cyan

# Base (unpicked) colors -- muted, protanopia-friendly
_IDLE_POINT   = "#E8D5B7"   # warm white  -- low-sat, luminance vs hover
_IDLE_CURVE   = "#AAAAAA"   # mid grey    -- luminance only, universal
_IDLE_SURFACE = "#5B8DB8"   # steel blue  -- blue axis
_IDLE_VOLUME  = "#5A6E82"   # slate       -- darker blue-grey, distinct from surface
_EDGE_COLOR   = "#2C4A6E"   # dark navy

# Opacity applied to actors whose dim is not in the pick filter
_FILTER_DIMMED_ALPHA = 0.12


# ======================================================================
# Boundary-curve surface tessellation helper
# ======================================================================

def _surface_is_planar(tag: int) -> bool:
    """Heuristic: sample the surface normal at several UV points.
    If all normals are (nearly) parallel, the surface is planar."""
    try:
        lo, hi = gmsh.model.getParametrizationBounds(2, tag)
        u_mid = 0.5 * (lo[0] + hi[0])
        v_mid = 0.5 * (lo[1] + hi[1])
        n0 = np.array(gmsh.model.getNormal(tag, [u_mid, v_mid]))
        # Sample 4 corners
        for u, v in [(lo[0], lo[1]), (hi[0], lo[1]),
                     (lo[0], hi[1]), (hi[0], hi[1])]:
            try:
                ni = np.array(gmsh.model.getNormal(tag, [u, v]))
                if abs(np.dot(n0, ni)) < 0.95:
                    return False
            except Exception:
                pass
        return True
    except Exception:
        return True  # assume planar if we can't check


def _uv_grid_polydata(
    tag: int, n_samples: int = 16,
) -> tuple[np.ndarray, list[int]]:
    """Tessellate a surface via its UV parametric grid.

    Works well for **curved** surfaces (cylinders, spheres, cones)
    where the UV rectangle maps cleanly to the 3D surface.
    """
    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
    u_vals = np.linspace(lo[0], hi[0], n_samples)
    v_vals = np.linspace(lo[1], hi[1], n_samples)
    uu, vv = np.meshgrid(u_vals, v_vals)
    uv_flat = np.column_stack([uu.ravel(), vv.ravel()]).ravel().tolist()
    pts = np.array(gmsh.model.getValue(2, tag, uv_flat)).reshape(-1, 3)

    n = n_samples
    faces: list[int] = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = (i + 1) * n + j
            v3 = v2 + 1
            faces.extend([3, v0, v1, v3])
            faces.extend([3, v0, v3, v2])
    return pts, faces


def _boundary_surface_polydata(
    tag: int,
    n_curve_samples: int = 60,
) -> tuple[np.ndarray, list[int]]:
    """Return ``(pts, faces)`` for a surface.

    Strategy:
    - **Curved** surfaces (cylinders, spheres): use UV parametric grid.
    - **Planar** surfaces (trimmed, irregular): sample boundary curves
      and triangulate with Delaunay 2D.

    Parameters
    ----------
    tag : int
        Gmsh surface tag.
    n_curve_samples : int
        Number of sample points per boundary curve (for planar path).

    Returns
    -------
    pts : ndarray, shape (N, 3)
    faces : list[int]  VTK-style ``[3, i, j, k, ...]``.
    """
    # Curved surfaces → UV grid (handles cylinders, spheres correctly)
    if not _surface_is_planar(tag):
        try:
            return _uv_grid_polydata(tag, n_samples=n_curve_samples // 4)
        except Exception:
            pass  # fall through to boundary approach

    # Planar / trimmed surfaces → boundary curve sampling + Delaunay
    bnd = gmsh.model.getBoundary(
        [(2, tag)], combined=False, oriented=True,
    )
    if not bnd:
        raise ValueError(f"Surface {tag} has no boundary curves")

    boundary_pts: list[np.ndarray] = []
    for _, ctag in bnd:
        abs_ctag = abs(ctag)
        lo, hi = gmsh.model.getParametrizationBounds(1, abs_ctag)
        u = np.linspace(lo[0], hi[0], n_curve_samples)
        cpts = np.array(
            gmsh.model.getValue(1, abs_ctag, u.tolist())
        ).reshape(-1, 3)
        if ctag < 0:
            cpts = cpts[::-1]
        boundary_pts.append(cpts[:-1])

    polygon = np.vstack(boundary_pts)

    # Centroid fan triangulation — works for any star-shaped polygon
    # (convex or mildly concave). Respects boundary edges exactly,
    # unlike Delaunay which can add triangles outside the boundary.
    centroid = polygon.mean(axis=0)
    pts = np.vstack([polygon, centroid.reshape(1, 3)])
    c_idx = len(polygon)
    faces: list[int] = []
    n_bnd = len(polygon)
    for i in range(n_bnd):
        j = (i + 1) % n_bnd
        faces.extend([3, c_idx, i, j])

    return pts, faces


# ======================================================================
# Physical-group I/O helpers
# ======================================================================

_SUFFIX_BY_DIM = {0: "_p", 1: "_c", 2: "_s", 3: "_v"}
_DIM_BY_SUFFIX = {v: k for k, v in _SUFFIX_BY_DIM.items()}


def _load_physical_group_members(name: str) -> list[DimTag]:
    """
    Return all DimTags belonging to the physical group(s) associated with
    *name*.  If *name* itself matches a Gmsh physical group name, that
    group's entities are returned.  Otherwise, look for mixed-dim groups
    written as ``name_p / name_c / name_s / name_v``.
    """
    out: list[DimTag] = []
    for d, pg_tag in gmsh.model.getPhysicalGroups():
        try:
            pg_name = gmsh.model.getPhysicalName(d, pg_tag)
        except Exception:
            continue
        if pg_name == name:
            for t in gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag):
                out.append((d, int(t)))
            continue
        # Match suffixed variants
        for suf, sdim in _DIM_BY_SUFFIX.items():
            if pg_name == f"{name}{suf}" and d == sdim:
                for t in gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag):
                    out.append((d, int(t)))
    return out


def _delete_physical_group_by_name(name: str) -> None:
    """
    Remove any physical group matching *name* or its suffixed per-dim
    variants (``name_p / name_c / name_s / name_v``).
    """
    to_remove: list[tuple[int, int]] = []
    for d, pg_tag in gmsh.model.getPhysicalGroups():
        try:
            pg_name = gmsh.model.getPhysicalName(d, pg_tag)
        except Exception:
            continue
        if pg_name == name:
            to_remove.append((d, pg_tag))
            continue
        for suf, sdim in _DIM_BY_SUFFIX.items():
            if pg_name == f"{name}{suf}" and d == sdim:
                to_remove.append((d, pg_tag))
    if to_remove:
        try:
            gmsh.model.removePhysicalGroups(to_remove)
        except Exception:
            pass


def _write_physical_group(name: str, members: list[DimTag]) -> None:
    """
    Create a physical group named *name* from *members*.  If the members
    span multiple dims, write one group per dim with suffix
    ``_p / _c / _s / _v``.
    """
    if not members:
        return
    by_dim: dict[int, list[int]] = {}
    for d, t in members:
        by_dim.setdefault(d, []).append(int(t))
    if len(by_dim) == 1:
        d = next(iter(by_dim))
        pg = gmsh.model.addPhysicalGroup(d, by_dim[d])
        gmsh.model.setPhysicalName(d, pg, name)
        return
    for d, tags in by_dim.items():
        full = f"{name}{_SUFFIX_BY_DIM[d]}"
        pg = gmsh.model.addPhysicalGroup(d, tags)
        gmsh.model.setPhysicalName(d, pg, full)


# ======================================================================
# SelectionPicker
# ======================================================================

class SelectionPicker(BaseViewer):
    """
    Interactive BRep picker backed by PyVista.

    One picker targets one physical group.  Opening the window,
    left-clicking entities, and closing the window is enough to
    create the physical group.

    Parameters
    ----------
    parent : _SessionBase
        Owning session instance (pyGmsh, Assembly, or Part).
    model  : Model
        The Model composite -- used for sync.
    physical_group : str or None
        Name of the Gmsh physical group to create from the picks
        when the window closes.  If ``None``, no group is written
        and the picks are available on ``self.selection`` as a
        :class:`Selection` object.
    dims : list of int
        BRep dimensions to render & make pickable.  Default ``[0, 1, 2]``.
    n_curve_samples : int
        Polyline samples per curve.
    n_surf_samples : int
        UV grid resolution per surface (n x n).
    point_size : float
        Marker size for dim=0 entities.
    line_width : float
        Width for dim=1 entities.
    """

    def __init__(
        self,
        parent: "_SessionBase",
        model: "Model",
        *,
        physical_group: str | None = None,
        dims: list[int] | None = None,
        # --- visual properties ---
        point_size: float = 10.0,
        line_width: float = 6.0,
        surface_opacity: float = 0.35,
        show_surface_edges: bool = False,
        # --- sampling ---
        n_curve_samples: int = 60,
        n_surf_samples: int = 16,
        # --- performance ---
        fast: bool = False,
    ) -> None:
        super().__init__(
            parent, model,
            dims=dims,
            point_size=point_size,
            line_width=line_width,
            surface_opacity=surface_opacity,
            show_surface_edges=show_surface_edges,
        )
        self._group_name = physical_group
        self._n_curve_samples = n_curve_samples
        self._n_surf_samples = n_surf_samples
        self._fast = fast
        self._pick_color: str = _PICK_COLOR
        self._pick_history: list[DimTag] = []

        # Tab-cycling: Revit-style pick cycling through overlapping entities
        self._tab_candidates: list[DimTag] = []
        self._tab_index: int = 0
        self._tab_pos: tuple[int, int] | None = None

        # Active physical-group state.  The picker supports editing multiple
        # groups in one session -- the tree UI switches the active group via
        # set_active_group(); each switch stages the current picks to
        # _staged_groups and loads the new group's members into _picks.
        self._active_group: str | None = physical_group
        self._staged_groups: dict[str, list[DimTag]] = {}

        # Own cell picker for pixel-picking and peel-picking (Tab cycling)
        import vtk
        self._click_picker = vtk.vtkCellPicker()
        self._click_picker.SetTolerance(0.005)

    # ------------------------------------------------------------------
    # Backward-compat properties (SelectionPickerUI uses old names)
    # ------------------------------------------------------------------

    @property
    def _actor_to_dimtag(self):
        return self._actor_to_id

    @property
    def _dimtag_to_actor(self):
        return self._id_to_actor

    @property
    def _hover_dt(self):
        return self._hover_id

    @_hover_dt.setter
    def _hover_dt(self, value):
        self._hover_id = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selection(self) -> "Selection":
        """The current working set as a :class:`Selection` object."""
        from pyGmsh.viz.Selection import Selection
        return Selection(list(self._picks), self._parent)

    @property
    def tags(self) -> list[DimTag]:
        """The current working set as a list of DimTags."""
        return list(self._picks)

    @property
    def active_group(self) -> str | None:
        """The name of the physical group currently receiving picks."""
        return self._active_group

    def set_active_group(self, name: str | None) -> None:
        """
        Switch which physical group is currently being edited.

        Stages the current picks to the outgoing group (if any), then loads
        the new group's members into the working set.  If *name* does not
        correspond to an existing group (in Gmsh or in this session's
        staged groups), the working set becomes empty.

        Passing ``None`` stages the current picks and leaves no active
        group (picks will accumulate in ``.selection`` only).
        """
        # Stage outgoing picks
        if self._active_group is not None:
            self._staged_groups[self._active_group] = list(self._picks)

        self._active_group = name

        # Load incoming group members into the working set
        if name is None:
            self._picks = []
        elif name in self._staged_groups:
            self._picks = list(self._staged_groups[name])
        else:
            self._picks = _load_physical_group_members(name)

        # Refresh history + recolor the whole scene
        self._pick_history = list(self._picks)
        self._recolor_all()
        self._fire_pick_changed()

    def _commit_active_group(self) -> None:
        """Snapshot the current picks into the active group's staging slot."""
        if self._active_group is not None:
            self._staged_groups[self._active_group] = list(self._picks)

    def _flush_staged_to_gmsh(self) -> None:
        """
        Write all staged groups to Gmsh.  For each group in
        ``_staged_groups``: if a physical group with that name already
        exists, remove it first, then recreate it from the staged members.
        Empty groups are removed (not recreated).
        """
        self._commit_active_group()
        for gname, members in self._staged_groups.items():
            _delete_physical_group_by_name(gname)
            if not members:
                continue
            _write_physical_group(gname, members)

    # ------------------------------------------------------------------
    # Selection state machine
    # ------------------------------------------------------------------
    # These methods encapsulate every pick-state transition.  The UI
    # should call ONLY these — never mutate _picks / _active_group /
    # _staged_groups / _pick_history directly.

    def clear(self) -> None:
        """Clear all picks and deactivate the active group."""
        old = list(self._picks)
        self._picks.clear()
        self._pick_history.clear()
        self._active_group = None
        for dt in old:
            self._recolor(dt)
        self._update_status()
        self._fire_pick_changed()

    def apply_group(self, name: str) -> None:
        """Stage the current picks as *name* and set it as active."""
        self._staged_groups[name] = list(self._picks)
        self._active_group = name
        self._fire_pick_changed()

    def revert(self) -> bool:
        """Revert to the last-applied group members.

        Returns ``True`` if the revert changed something (there were
        uncommitted edits).  Returns ``False`` if there's nothing to
        revert (no active group, or picks already match the applied
        state) — the caller can then decide to ``clear()`` instead.
        """
        if not self._active_group:
            return False
        saved = self._staged_groups.get(self._active_group)
        if not saved or self._picks == list(saved):
            return False
        old = list(self._picks)
        self._picks = list(saved)
        self._pick_history = list(saved)
        for dt in set(old) ^ set(self._picks):
            self._recolor(dt)
        self._update_status()
        self._fire_pick_changed()
        return True

    def rename_group(self, old_name: str, new_name: str) -> None:
        """Rename a staged group.  Keeps current picks if it's active."""
        members = self._staged_groups.pop(old_name, [])
        if not members:
            members = _load_physical_group_members(old_name)
        self._staged_groups[new_name] = members
        # Mark old name for deletion on flush
        self._staged_groups[old_name] = []
        if self._active_group == old_name:
            self._active_group = new_name

    def delete_group(self, name: str) -> None:
        """Mark a group for deletion.  Clears picks if it was active."""
        self._staged_groups[name] = []
        if self._active_group == name:
            self.clear()

    def group_exists(self, name: str) -> bool:
        """Check whether *name* is already taken (staged or in Gmsh)."""
        if name in self._staged_groups and self._staged_groups[name]:
            return True
        for d, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                if gmsh.model.getPhysicalName(d, pg_tag) == name:
                    return True
            except Exception:
                pass
        return False

    @property
    def is_editing(self) -> bool:
        """True when a group is set as the active editing target."""
        return self._active_group is not None

    @property
    def staged_groups(self) -> dict[str, list[DimTag]]:
        """Snapshot of all staged groups (read-only copy)."""
        return dict(self._staged_groups)

    # ------------------------------------------------------------------

    def to_physical(self, name: str | None = None) -> int | None:
        """
        Write the current picks as a Gmsh physical group.

        If ``name`` is given it overrides the one set at construction.
        Mixed-dim picks are split into per-dim groups with suffixes
        ``_p / _c / _s / _v``.

        Returns the physical-group tag, or ``None`` if picks are empty.
        """
        group_name = name or self._group_name
        if group_name is None:
            raise ValueError(
                "No physical_group name provided at construction "
                "or in to_physical(name=...)"
            )
        if not self._picks:
            return None

        by_dim: dict[int, list[int]] = {}
        for d, t in self._picks:
            by_dim.setdefault(d, []).append(t)

        if len(by_dim) == 1:
            d = next(iter(by_dim))
            pg = gmsh.model.addPhysicalGroup(d, by_dim[d])
            gmsh.model.setPhysicalName(d, pg, group_name)
            return pg

        last_pg = None
        for d, ts in by_dim.items():
            full = f"{group_name}{_SUFFIX_BY_DIM[d]}"
            last_pg = gmsh.model.addPhysicalGroup(d, ts)
            gmsh.model.setPhysicalName(d, last_pg, full)
        return last_pg

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def show(
        self, *, title: str | None = None, maximized: bool = True,
    ) -> "SelectionPicker":
        """
        Open the picker window (Qt + pyvistaqt).  Blocks until closed.
        On close, flushes every staged physical group (including the
        active one) to Gmsh.
        """
        gmsh.model.occ.synchronize()

        # If we're starting with an active group that already exists,
        # preload its members as the working set (edit mode).
        if self._active_group is not None:
            existing = _load_physical_group_members(self._active_group)
            if existing:
                self._picks = existing
                self._pick_history = list(existing)

        default_title = (
            f"SelectionPicker — {self._parent.model_name}"
            + (f" → {self._active_group}" if self._active_group else "")
            + " - Ladruño"
        )
        window = self._create_window(
            title=title or default_title, maximized=maximized,
        )
        window.exec()
        self._on_window_closed()
        return self

    def _create_window(self, *, title: str, maximized: bool):
        from pyGmsh.viewers.SelectionPickerUI import SelectionPickerWindow
        return SelectionPickerWindow(self, title=title, maximized=maximized)

    def _on_window_closed(self) -> None:
        self._flush_staged_to_gmsh()
        if self._parent._verbose:
            n_groups = sum(1 for m in self._staged_groups.values() if m)
            print(
                f"[picker] closed — {n_groups} physical group(s) written, "
                f"{len(self._picks)} picks in working set"
            )

    # ------------------------------------------------------------------
    # Scene construction (BRep-specific)
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        if self._fast:
            self._build_scene_from_mesh()
            return
        self._build_scene_parametric()

    def _build_scene_from_mesh(self) -> None:
        """Build the VTK scene using batched actors (one per dimension).

        Generates a coarse 2D mesh, merges all entities of each
        dimension into a single PolyData with ``cell_data["entity_tag"]``
        for pick resolution.  Typically 3 ``add_mesh`` calls instead of
        1000+.
        """
        import time
        import pyvista as pv
        plotter = self._plotter
        t0 = time.perf_counter()

        # ── 1. Model diagonal ───────────────────────────────────────
        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm(
                [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
            ))
            if diag <= 0.0:
                diag = 1.0
        except Exception:
            diag = 1.0

        # ── 2. Generate temp mesh if needed ─────────────────────────
        t_mesh = time.perf_counter()
        had_mesh = False
        try:
            existing_tags, _, _ = gmsh.model.mesh.getNodes()
            had_mesh = len(existing_tags) > 0
        except Exception:
            pass

        if not had_mesh:
            # Save current mesh settings
            _saved = {}
            for key in ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax",
                        "Mesh.Algorithm", "Mesh.MeshSizeFromCurvature",
                        "Mesh.MinimumElementsPerTwoPi"):
                try:
                    _saved[key] = gmsh.option.getNumber(key)
                except Exception:
                    pass
            try:
                # Coarse mesh that still resolves small features:
                # - MeshSizeFromCurvature auto-refines around holes/curves
                # - MeshSizeMax caps element size for large flat areas
                # - MinimumElementsPerTwoPi ensures small circles get
                #   enough elements to be visible
                gmsh.option.setNumber("Mesh.MeshSizeMin", diag * 0.001)
                gmsh.option.setNumber("Mesh.MeshSizeMax", diag * 0.03)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
                gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 8)
                gmsh.option.setNumber("Mesh.Algorithm", 6)
                gmsh.model.mesh.generate(2)
            except Exception:
                pass
            finally:
                for key, val in _saved.items():
                    try:
                        gmsh.option.setNumber(key, val)
                    except Exception:
                        pass
        t_mesh_elapsed = time.perf_counter() - t_mesh

        # ── 3. Check mesh was generated ─────────────────────────────
        try:
            all_tags, _, _ = gmsh.model.mesh.getNodes()
        except Exception:
            self._build_scene_parametric()
            return
        if len(all_tags) == 0:
            self._build_scene_parametric()
            return

        # ── 4. Enable batched pick/hover/recolor mode ───────────────
        self._batched = True
        self._batch_actors: dict[int, object] = {}     # dim → VTK actor
        self._batch_meshes: dict[int, pv.PolyData] = {}
        self._batch_cell_to_dt: dict[int, dict[int, DimTag]] = {}  # dim → {cell_idx: dt}
        self._batch_dt_to_cells: dict[DimTag, list[int]] = {}      # dt → [cell_indices]
        self._batch_centroids: dict[DimTag, np.ndarray] = {}       # dt → xyz centroid

        idle_rgb = {
            0: np.array([int(c * 255) for c in _hex_to_rgb(_IDLE_POINT)], dtype=np.uint8),
            1: np.array([int(c * 255) for c in _hex_to_rgb(_IDLE_CURVE)], dtype=np.uint8),
            2: np.array([int(c * 255) for c in _hex_to_rgb(_IDLE_SURFACE)], dtype=np.uint8),
            3: np.array([int(c * 255) for c in _hex_to_rgb(_IDLE_VOLUME)], dtype=np.uint8),
        }

        t_build = time.perf_counter()
        n_entities = 0

        # ── dim=0: single glyph actor for all points ───────────────
        t_dim = time.perf_counter()
        n_d0 = 0
        if 0 in self._dims:
            centers = []
            tags_d0 = []
            for _, tag in gmsh.model.getEntities(dim=0):
                try:
                    ntags, ncoords, _ = gmsh.model.mesh.getNodes(dim=0, tag=tag)
                    if len(ntags) == 0:
                        continue
                    xyz = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)[0]
                    centers.append(xyz)
                    tags_d0.append(tag)
                    self._batch_centroids[(0, tag)] = xyz
                    n_d0 += 1
                except Exception:
                    pass
            if centers:
                centers_arr = np.array(centers)
                cloud = pv.PolyData(centers_arr)
                base_r = diag * 0.003 * max(0.1, self._point_size / 10.0)
                sphere_src = pv.Sphere(
                    radius=base_r,
                    theta_resolution=8, phi_resolution=8,
                )
                glyphs = cloud.glyph(geom=sphere_src, orient=False, scale=False)
                n_cells_per_pt = glyphs.n_cells // len(centers) if len(centers) else 1
                entity_tags = np.empty(glyphs.n_cells, dtype=np.int64)
                colors = np.tile(idle_rgb[0], (glyphs.n_cells, 1))
                cell_to_dt: dict[int, DimTag] = {}
                for i, tag in enumerate(tags_d0):
                    start = i * n_cells_per_pt
                    end = start + n_cells_per_pt
                    entity_tags[start:end] = tag
                    dt = (0, tag)
                    self._batch_dt_to_cells[dt] = list(range(start, end))
                    for ci in range(start, end):
                        cell_to_dt[ci] = dt
                glyphs.cell_data["entity_tag"] = entity_tags
                glyphs.cell_data["colors"] = colors
                actor = plotter.add_mesh(
                    glyphs, scalars="colors", rgb=True,
                    smooth_shading=True,
                    pickable=True,
                    reset_camera=False,
                )
                self._batch_actors[0] = actor
                self._batch_meshes[0] = glyphs
                self._batch_cell_to_dt[0] = cell_to_dt
                for tag in tags_d0:
                    self._id_to_actor[(0, tag)] = actor
                self._actor_to_id[id(actor)] = (0, -1)
        t_d0 = time.perf_counter() - t_dim
        n_entities += n_d0

        # ── dim=1: merge all curves into one PolyData ───────────────
        t_dim = time.perf_counter()
        n_d1 = 0
        if 1 in self._dims:
            all_pts_parts: list[np.ndarray] = []
            all_lines_parts: list[np.ndarray] = []
            all_etags: list[int] = []
            all_dt_cells: dict[DimTag, list[int]] = {}
            cell_offset = 0
            pt_offset = 0
            for _, tag in gmsh.model.getEntities(dim=1):
                try:
                    ntags, ncoords, _ = gmsh.model.mesh.getNodes(
                        dim=1, tag=tag, includeBoundary=True,
                    )
                    if len(ntags) < 2:
                        continue
                    pts = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
                    n = len(pts)
                    n_lines = n - 1
                    idx = np.arange(n_lines, dtype=np.int64)
                    lines = np.empty(n_lines * 3, dtype=np.int64)
                    lines[0::3] = 2
                    lines[1::3] = idx + pt_offset
                    lines[2::3] = idx + pt_offset + 1
                    all_pts_parts.append(pts)
                    all_lines_parts.append(lines)
                    cell_indices = list(range(cell_offset, cell_offset + n_lines))
                    dt = (1, tag)
                    all_dt_cells[dt] = cell_indices
                    all_etags.extend([tag] * n_lines)
                    self._batch_centroids[dt] = pts.mean(axis=0)
                    cell_offset += n_lines
                    pt_offset += n
                    n_d1 += 1
                except Exception:
                    pass
            if all_pts_parts:
                merged_pts = np.vstack(all_pts_parts)
                merged_lines = np.concatenate(all_lines_parts)
                poly = pv.PolyData()
                poly.points = merged_pts
                poly.lines = merged_lines
                entity_tags = np.array(all_etags, dtype=np.int64)
                colors = np.tile(idle_rgb[1], (len(all_etags), 1))
                poly.cell_data["entity_tag"] = entity_tags
                poly.cell_data["colors"] = colors
                actor = plotter.add_mesh(
                    poly, scalars="colors", rgb=True,
                    line_width=self._line_width,
                    render_lines_as_tubes=True,
                    pickable=True, reset_camera=False,
                )
                self._batch_actors[1] = actor
                self._batch_meshes[1] = poly
                cell_to_dt = {}
                for dt, cells in all_dt_cells.items():
                    self._batch_dt_to_cells[dt] = cells
                    self._id_to_actor[dt] = actor
                    for ci in cells:
                        cell_to_dt[ci] = dt
                self._batch_cell_to_dt[1] = cell_to_dt
                self._actor_to_id[id(actor)] = (1, -1)
        t_d1 = time.perf_counter() - t_dim
        n_entities += n_d1

        # ── dim=2: merge all surfaces into one PolyData ─────────────
        t_dim = time.perf_counter()
        n_d2 = 0
        if 2 in self._dims:
            all_pts_parts = []
            all_faces_parts: list[np.ndarray] = []
            all_etags = []
            all_dt_cells = {}
            cell_offset = 0
            pt_offset = 0
            for _, tag in gmsh.model.getEntities(dim=2):
                try:
                    ntags, ncoords, _ = gmsh.model.mesh.getNodes(
                        dim=2, tag=tag, includeBoundary=True,
                    )
                    if len(ntags) == 0:
                        continue
                    ltags = np.asarray(ntags, dtype=np.int64)
                    lpts = np.asarray(ncoords, dtype=np.float64).reshape(-1, 3)
                    lmax = int(ltags.max())
                    lidx = np.full(lmax + 1, -1, dtype=np.int64)
                    lidx[ltags] = np.arange(len(ltags), dtype=np.int64)

                    etypes, _, enodes_list = gmsh.model.mesh.getElements(2, tag)
                    n_cells_entity = 0
                    for etype, enodes in zip(etypes, enodes_list):
                        etype = int(etype)
                        if etype == 2:
                            npe = 3
                        elif etype == 3:
                            npe = 4
                        else:
                            continue
                        enodes = np.asarray(enodes, dtype=np.int64)
                        n_elems = len(enodes) // npe
                        node_mat = enodes.reshape(n_elems, npe)
                        idx_mat = lidx[node_mat]
                        valid = np.all(idx_mat >= 0, axis=1)
                        idx_mat = idx_mat[valid]
                        if len(idx_mat) == 0:
                            continue
                        # Offset indices
                        idx_mat = idx_mat + pt_offset
                        prefix = np.full((len(idx_mat), 1), npe, dtype=np.int64)
                        all_faces_parts.append(
                            np.hstack([prefix, idx_mat]).ravel()
                        )
                        n_cells_entity += len(idx_mat)
                        all_etags.extend([tag] * len(idx_mat))

                    if n_cells_entity > 0:
                        dt = (2, tag)
                        cell_indices = list(range(cell_offset, cell_offset + n_cells_entity))
                        all_dt_cells[dt] = cell_indices
                        self._batch_centroids[dt] = lpts.mean(axis=0)
                        all_pts_parts.append(lpts)
                        cell_offset += n_cells_entity
                        pt_offset += len(lpts)
                        n_d2 += 1
                except Exception:
                    pass
            if all_pts_parts:
                merged_pts = np.vstack(all_pts_parts)
                merged_faces = np.concatenate(all_faces_parts)
                poly = pv.PolyData(merged_pts, faces=merged_faces)
                entity_tags = np.array(all_etags, dtype=np.int64)
                colors = np.tile(idle_rgb[2], (len(all_etags), 1))
                poly.cell_data["entity_tag"] = entity_tags
                poly.cell_data["colors"] = colors
                actor = plotter.add_mesh(
                    poly, scalars="colors", rgb=True,
                    opacity=self._surface_opacity,
                    show_edges=self._show_surface_edges,
                    edge_color=_EDGE_COLOR,
                    line_width=0.5,
                    smooth_shading=True,
                    pickable=True, reset_camera=False,
                )
                self._batch_actors[2] = actor
                self._batch_meshes[2] = poly
                cell_to_dt = {}
                for dt, cells in all_dt_cells.items():
                    self._batch_dt_to_cells[dt] = cells
                    self._id_to_actor[dt] = actor
                    for ci in cells:
                        cell_to_dt[ci] = dt
                self._batch_cell_to_dt[2] = cell_to_dt
                self._actor_to_id[id(actor)] = (2, -1)
        t_d2 = time.perf_counter() - t_dim
        n_entities += n_d2

        # dim=3 skipped for now (same merge pattern on boundary surfaces)

        plotter.reset_camera()
        t_total_build = time.perf_counter() - t_build

        # ── 5. Cleanup temp mesh ────────────────────────────────────
        if not had_mesh:
            try:
                gmsh.model.mesh.clear()
            except Exception:
                pass

        # ── 6. Print profiling summary ──────────────────────────────
        total = time.perf_counter() - t0
        n_add_mesh = len(self._batch_actors)
        entities = {d: len(gmsh.model.getEntities(d)) for d in self._dims}
        print(f"\n[viewer_fast] Scene built in {total:.2f}s  "
              f"({n_add_mesh} add_mesh calls, {n_entities} entities)")
        print(f"  Entities: {entities}")
        print(f"  Mesh generate : {t_mesh_elapsed:.3f}s"
              f"  {'(used existing)' if had_mesh else '(temp coarse)'}")
        print(f"  dim0 points   : {t_d0:.3f}s  ({n_d0} entities → 1 glyph actor)")
        print(f"  dim1 curves   : {t_d1:.3f}s  ({n_d1} entities → 1 merged actor)")
        print(f"  dim2 surfaces : {t_d2:.3f}s  ({n_d2} entities → 1 merged actor)")
        print(f"  Data build    : {t_total_build:.3f}s")

    # ── Batched pick / hover / recolor overrides ────────────────────

    def _resolve_pick_cell(self, prop, cell_id: int) -> DimTag | None:
        """Resolve a VTK pick to a DimTag using cell_data."""
        if not getattr(self, '_batched', False):
            return self._actor_to_id.get(id(prop))
        # Find which dim-batch this actor belongs to
        for dim, actor in self._batch_actors.items():
            if id(actor) == id(prop):
                cell_map = self._batch_cell_to_dt.get(dim, {})
                return cell_map.get(cell_id)
        return None

    def _on_lmb_click(self, x: int, y: int, ctrl: bool) -> None:
        """Pixel-pick or Ctrl+unpick at display coords (x, y)."""
        renderer = self._plotter.renderer
        self._click_picker.Pick(x, y, 0, renderer)
        prop = self._click_picker.GetViewProp()
        if prop is None:
            return

        if getattr(self, '_batched', False):
            cell_id = self._click_picker.GetCellId()
            dt = self._resolve_pick_cell(prop, cell_id)
        else:
            dt = self._actor_to_id.get(id(prop))

        if dt is None:
            return

        if ctrl:
            if dt in self._picks:
                self._picks.remove(dt)
                self._pick_history = [
                    d for d in self._pick_history if d != dt
                ]
                self._recolor(dt)
                self._update_status()
                self._fire_pick_changed()
        else:
            self._tab_candidates = [dt]  # simplified for batched mode
            self._tab_pos = (x, y)
            self._tab_index = 0
            if dt not in self._picks:
                self._picks.append(dt)
                self._pick_history.append(dt)
            self._recolor(dt)
            self._update_status()
            self._fire_pick_changed()

    def _update_hover(self, x: int, y: int) -> None:
        """Hover highlight with batched cell-data resolution."""
        import vtk

        if self._plotter is None:
            return

        self._hover_throttle = (self._hover_throttle + 1) % 3
        if self._hover_throttle != 0:
            return

        if self._hover_picker is None:
            p = vtk.vtkCellPicker()
            p.SetTolerance(0.005)
            self._hover_picker = p

        self._hover_picker.Pick(x, y, 0, self._plotter.renderer)
        prop = self._hover_picker.GetViewProp()

        new_dt: DimTag | None = None
        if prop is not None:
            if getattr(self, '_batched', False):
                cell_id = self._hover_picker.GetCellId()
                candidate = self._resolve_pick_cell(prop, cell_id)
            else:
                candidate = self._actor_to_id.get(id(prop))
            if candidate is not None:
                if candidate[0] in self._pickable_dims and candidate not in self._hidden:
                    new_dt = candidate

        if new_dt == self._hover_id:
            return

        old_dt = self._hover_id
        self._hover_id = new_dt
        self._on_hover_changed_internal(old_dt, new_dt)

        for cb in self._on_hover_changed:
            try:
                cb()
            except Exception:
                pass

    def _on_hover_changed_internal(self, old_id, new_id) -> None:
        """Recolor old hover back to idle, recolor new hover to gold."""
        if old_id is not None:
            self._recolor(old_id)
        if new_id is not None:
            self._recolor(new_id)

    def _recolor(self, dt: DimTag, *, _render: bool = True) -> None:
        if not getattr(self, '_batched', False):
            return self._recolor_per_actor(dt, _render=_render)

        # Batched: update cell colors in the merged mesh
        dim = dt[0]
        mesh = self._batch_meshes.get(dim)
        if mesh is None:
            return
        cells = self._batch_dt_to_cells.get(dt)
        if not cells:
            return

        if dt == self._hover_id and dt not in self._picks:
            rgb = np.array([int(c * 255) for c in _hex_to_rgb(_HOVER_COLOR)], dtype=np.uint8)
        elif dt in self._picks:
            rgb = np.array([int(c * 255) for c in _hex_to_rgb(self._pick_color)], dtype=np.uint8)
        else:
            idle = {
                0: _IDLE_POINT, 1: _IDLE_CURVE,
                2: _IDLE_SURFACE, 3: _IDLE_VOLUME,
            }[dim]
            rgb = np.array([int(c * 255) for c in _hex_to_rgb(idle)], dtype=np.uint8)

        colors = mesh.cell_data["colors"]
        for ci in cells:
            colors[ci] = rgb
        mesh.cell_data["colors"] = colors  # trigger VTK update

        if _render:
            self._plotter.render()

    def _recolor_per_actor(self, dt: DimTag, *, _render: bool = True) -> None:
        """Original per-actor recolor (non-batched mode)."""
        actor = self._id_to_actor.get(dt)
        if actor is None:
            return
        if dt == self._hover_id and dt not in self._picks:
            actor.GetProperty().SetColor(_hex_to_rgb(_HOVER_COLOR))
        elif dt in self._picks:
            actor.GetProperty().SetColor(_hex_to_rgb(self._pick_color))
        else:
            idle = {
                0: _IDLE_POINT, 1: _IDLE_CURVE,
                2: _IDLE_SURFACE, 3: _IDLE_VOLUME,
            }[dt[0]]
            actor.GetProperty().SetColor(_hex_to_rgb(idle))
        if _render:
            self._plotter.render()

    def _recolor_all(self) -> None:
        """Recolor every entity, render once at the end."""
        if getattr(self, '_batched', False):
            for dt in self._batch_dt_to_cells:
                self._recolor(dt, _render=False)
        else:
            for dt in self._id_to_actor:
                self._recolor(dt, _render=False)
        self._plotter.render()

    def _project_centroid(self, actor_or_dt):
        """Project centroid to display coords.

        Accepts either a VTK actor (non-batched) or a DimTag (batched).
        """
        if getattr(self, '_batched', False) and isinstance(actor_or_dt, tuple):
            xyz = self._batch_centroids.get(actor_or_dt)
            if xyz is None:
                return None
            renderer = self._plotter.renderer
            renderer.SetWorldPoint(xyz[0], xyz[1], xyz[2], 1.0)
            renderer.WorldToDisplay()
            dx, dy, _ = renderer.GetDisplayPoint()
            return (dx, dy)
        # Non-batched: actor path
        return super()._project_centroid(actor_or_dt)

    def _do_box_select(
        self, x0: float, y0: float, x1: float, y1: float,
        *, crossing: bool,
    ) -> None:
        if not getattr(self, '_batched', False):
            return super()._do_box_select(x0, y0, x1, y1, crossing=crossing)

        # Batched box-select using precomputed centroids
        try:
            rw = self._plotter.render_window
            vw, vh = rw.GetSize()
            aw, ah = rw.GetActualSize()
            sx_ratio = aw / vw if vw else 1.0
            sy_ratio = ah / vh if vh else 1.0
        except Exception:
            sx_ratio = sy_ratio = 1.0

        bx0 = x0 * sx_ratio
        bx1 = x1 * sx_ratio
        by0 = y0 * sy_ratio
        by1 = y1 * sy_ratio

        added = 0
        for dt in self._batch_dt_to_cells:
            if dt[0] not in self._pickable_dims:
                continue
            if dt in self._hidden or dt in self._picks:
                continue
            pt = self._project_centroid(dt)
            if pt is None:
                continue
            sx, sy = pt
            if bx0 <= sx <= bx1 and by0 <= sy <= by1:
                self._picks.append(dt)
                self._pick_history.append(dt)
                self._recolor(dt, _render=False)
                added += 1
        self._plotter.render()

    def _build_scene_parametric(self) -> None:
        """Original parametric-sampling scene builder."""
        import time
        import pyvista as pv
        plotter = self._plotter
        t0 = time.perf_counter()

        t_dim = time.perf_counter()
        n_dim0 = 0
        if 0 in self._dims:
            try:
                bb = gmsh.model.getBoundingBox(-1, -1)
                diag = float(np.linalg.norm(
                    [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
                ))
                if diag <= 0.0:
                    diag = 1.0
            except Exception:
                diag = 1.0
            centers = []
            tags_d0 = []
            for _, tag in gmsh.model.getEntities(dim=0):
                try:
                    xyz = np.array(gmsh.model.getValue(0, tag, []))
                    centers.append(xyz)
                    tags_d0.append(tag)
                    n_dim0 += 1
                except Exception:
                    pass
            if centers:
                cloud = pv.PolyData(np.array(centers))
                base_r = diag * 0.003 * max(0.1, self._point_size / 10.0)
                sphere_src = pv.Sphere(
                    radius=base_r,
                    theta_resolution=8, phi_resolution=8,
                )
                glyphs = cloud.glyph(geom=sphere_src, orient=False, scale=False)
                actor = plotter.add_mesh(
                    glyphs, color=_IDLE_POINT,
                    smooth_shading=True,
                    pickable=True,
                )
                # Register each point entity to the single actor
                for tag in tags_d0:
                    self._register_actor(actor, (0, tag))
        t_dim0 = time.perf_counter() - t_dim

        t_dim = time.perf_counter()
        n_dim1 = 0
        if 1 in self._dims:
            for _, tag in gmsh.model.getEntities(dim=1):
                try:
                    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
                    u = np.linspace(lo[0], hi[0], self._n_curve_samples)
                    pts = np.array(
                        gmsh.model.getValue(1, tag, u.tolist())
                    ).reshape(-1, 3)
                    lines = np.hstack(
                        [[2, i, i + 1] for i in range(len(pts) - 1)]
                    )
                    poly = pv.PolyData(pts)
                    poly.lines = lines
                    actor = plotter.add_mesh(
                        poly, color=_IDLE_CURVE,
                        line_width=self._line_width,
                        render_lines_as_tubes=True,
                        pickable=True,
                    )
                    self._register_actor(actor, (1, tag))
                    n_dim1 += 1
                except Exception:
                    pass
        t_dim1 = time.perf_counter() - t_dim

        t_dim = time.perf_counter()
        n_dim2 = 0
        if 2 in self._dims:
            for _, tag in gmsh.model.getEntities(dim=2):
                try:
                    pts, faces = _boundary_surface_polydata(
                        tag, self._n_curve_samples,
                    )
                    mesh = pv.PolyData(pts, faces=np.array(faces))
                    actor = plotter.add_mesh(
                        mesh, color=_IDLE_SURFACE,
                        opacity=self._surface_opacity,
                        show_edges=self._show_surface_edges,
                        edge_color=_EDGE_COLOR,
                        line_width=0.5,
                        smooth_shading=True,
                        pickable=True,
                    )
                    self._register_actor(actor, (2, tag))
                    n_dim2 += 1
                except Exception:
                    pass
        t_dim2 = time.perf_counter() - t_dim

        t_dim = time.perf_counter()
        n_dim3 = 0
        if 3 in self._dims:
            # Render each volume as the combined mesh of its boundary
            # surfaces -- one actor per volume, so a click anywhere on
            # the hull picks the volume.  Uses a slightly-lower opacity
            # than dim=2 to reduce z-fighting when surfaces are also
            # shown (depth peeling handles the rest).
            vol_alpha = max(0.05, self._surface_opacity * 0.6)
            for _, vtag in gmsh.model.getEntities(dim=3):
                try:
                    boundary = gmsh.model.getBoundary(
                        [(3, vtag)], combined=False,
                        oriented=False, recursive=False,
                    )
                    all_pts: list[np.ndarray] = []
                    all_faces: list[int] = []
                    offset = 0
                    for bd, btag in boundary:
                        if bd != 2:
                            continue
                        try:
                            pts, faces_local = _boundary_surface_polydata(
                                btag, self._n_curve_samples,
                            )
                        except Exception:
                            continue
                        all_pts.append(pts)
                        # Shift face indices by the running offset
                        for k in range(0, len(faces_local), 4):
                            all_faces.extend([
                                faces_local[k],
                                faces_local[k + 1] + offset,
                                faces_local[k + 2] + offset,
                                faces_local[k + 3] + offset,
                            ])
                        offset += len(pts)
                    if not all_pts:
                        continue
                    combined = np.vstack(all_pts)
                    mesh = pv.PolyData(
                        combined, faces=np.array(all_faces),
                    )
                    actor = plotter.add_mesh(
                        mesh, color=_IDLE_VOLUME,
                        opacity=vol_alpha,
                        smooth_shading=True,
                        pickable=True,
                    )
                    self._register_actor(actor, (3, vtag))
                    n_dim3 += 1
                except Exception:
                    pass
        t_dim3 = time.perf_counter() - t_dim

        # Apply initial pick colours (for edit-mode preload)
        t_rc = time.perf_counter()
        self._recolor_all()
        t_recolor = time.perf_counter() - t_rc

        # ── Profiling summary ───────────────────────────────────────
        total = time.perf_counter() - t0
        n_actors = len(self._actor_to_id)
        print(f"\n[viewer] Scene built in {total:.2f}s  "
              f"({n_actors} actors, parametric)")
        for label, t, n in [
            ("dim0_points", t_dim0, n_dim0),
            ("dim1_curves", t_dim1, n_dim1),
            ("dim2_surfaces", t_dim2, n_dim2),
            ("dim3_volumes", t_dim3, n_dim3),
        ]:
            if t > 0.001:
                print(f"  {label:16s}: {t:.3f}s  ({n} actors)")
        print(f"  Recolor       : {t_recolor:.3f}s")

    # ------------------------------------------------------------------
    # Keybindings
    # ------------------------------------------------------------------

    def _install_keybindings(self) -> None:
        plotter = self._plotter
        plotter.add_key_event("q", lambda: plotter.close())
        plotter.add_key_event("Q", lambda: plotter.close())
        plotter.add_key_event("h", self._hide_selected)
        plotter.add_key_event("H", self._hide_selected)
        plotter.add_key_event("i", self._isolate_selected)
        plotter.add_key_event("I", self._isolate_selected)
        plotter.add_key_event("r", self._show_all)
        plotter.add_key_event("R", self._show_all)
        plotter.add_key_event("u", self._undo)
        plotter.add_key_event("U", self._undo)
        # Dim-filter (which dims respond to clicks)
        plotter.add_key_event("1", lambda: self._set_pickable_dims({0}))
        plotter.add_key_event("2", lambda: self._set_pickable_dims({1}))
        plotter.add_key_event("3", lambda: self._set_pickable_dims({2}))
        plotter.add_key_event("4", lambda: self._set_pickable_dims({3}))
        plotter.add_key_event("0", lambda: self._set_pickable_dims(set(self._dims)))
        # Tab-cycle is handled by a Qt shortcut in SelectionPickerUI
        # (Qt eats Tab for focus-cycling before VTK sees it).

    # ------------------------------------------------------------------
    # Picker hooks (called by BaseViewer._install_picker)
    # ------------------------------------------------------------------

    def _peel_candidates_at(self, x: int, y: int) -> list[DimTag]:
        """Peel-pick: repeatedly pick at (x, y), hiding each hit
        actor before re-picking, to find ALL entities stacked under
        the cursor at any depth.  Restores visibility afterwards.
        Pixel-precise -- no radius or bbox projection needed."""
        renderer = self._plotter.renderer
        candidates: list[DimTag] = []
        hidden_during_peel: list = []
        for _ in range(50):               # safety cap
            self._click_picker.Pick(x, y, 0, renderer)
            prop = self._click_picker.GetViewProp()
            if prop is None:
                break
            dt = self._actor_to_id.get(id(prop))
            if dt is None:
                break
            if dt[0] not in self._pickable_dims:
                break
            if dt in self._hidden:
                break
            candidates.append(dt)
            # Temporarily hide so next pick peels through
            prop.VisibilityOff()
            hidden_during_peel.append(prop)
        # Restore all peeled actors
        for a in hidden_during_peel:
            a.VisibilityOn()
        return candidates

    def _on_box_select(
        self,
        x0: int, y0: int, x1: int, y1: int,
        crossing: bool,
        ctrl: bool,
    ) -> None:
        """Box-select or Ctrl+box-unselect using centroid-in-box test."""
        if ctrl:
            self._do_box_unselect(x0, y0, x1, y1, crossing=crossing)
        else:
            if self._parent._verbose:
                print(
                    f"[picker] LMB drag → box-select "
                    f"({x0},{y0})→({x1},{y1}) "
                    f"crossing={crossing}"
                )
            self._do_box_select(x0, y0, x1, y1, crossing=crossing)

    # ------------------------------------------------------------------
    # Pick management
    # ------------------------------------------------------------------

    def _toggle_pick(self, dt: DimTag) -> None:
        # Respect current dim-filter -- picks on disabled dims are ignored
        if dt[0] not in self._pickable_dims:
            return
        if dt in self._hidden:
            return
        if dt in self._picks:
            self._picks.remove(dt)
            self._pick_history = [d for d in self._pick_history if d != dt]
        else:
            self._picks.append(dt)
            self._pick_history.append(dt)
        self._recolor(dt)
        self._update_status()
        self._fire_pick_changed()

    def _cycle_pick(self) -> None:
        """Tab -- cycle through overlapping entities at the last click
        position.  Unpicks the current candidate, advances to the next
        in the candidate ring, and picks it.  If the candidate ring is
        empty or exhausted, does nothing."""
        cands = self._tab_candidates
        if len(cands) < 2:
            return
        # Unpick the current candidate (if it was picked)
        cur = cands[self._tab_index]
        if cur in self._picks:
            self._picks.remove(cur)
            self._pick_history = [d for d in self._pick_history if d != cur]
            self._recolor(cur)
        # Advance
        self._tab_index = (self._tab_index + 1) % len(cands)
        nxt = cands[self._tab_index]
        # Pick the next candidate
        if nxt not in self._picks:
            self._picks.append(nxt)
            self._pick_history.append(nxt)
            self._recolor(nxt)
        self._update_status()
        self._fire_pick_changed()
        if self._parent._verbose:
            print(
                f"[picker] Tab cycle → {nxt}  "
                f"({self._tab_index + 1}/{len(cands)})"
            )

    # Scale multiplier for dim=0 spheres when picked -- makes the
    # selection clearly visible even at small point_size values.
    _PICK_SCALE_BOOST = 1.6

    def _undo(self) -> None:
        """Undo the last pick (pop from history)."""
        if not self._pick_history:
            return
        dt = self._pick_history.pop()
        if dt in self._picks:
            self._picks.remove(dt)
            self._recolor(dt)
        self._update_status()
        self._fire_pick_changed()

    def _set_pickable_dims(self, dims_set: set[int]) -> None:
        """Change which dims respond to clicks.  Entities on non-pickable
        dims are dimmed (reduced opacity) as a visual cue."""
        self._pickable_dims = set(dims_set)
        vol_alpha = max(0.05, self._surface_opacity * 0.6)
        for dt, actor in self._id_to_actor.items():
            prop = actor.GetProperty()
            if dt[0] in self._pickable_dims:
                # Restore normal opacity for this dim
                if dt[0] == 2:
                    prop.SetOpacity(self._surface_opacity)
                elif dt[0] == 3:
                    prop.SetOpacity(vol_alpha)
                else:
                    prop.SetOpacity(1.0)
                actor.SetPickable(True)
            else:
                prop.SetOpacity(_FILTER_DIMMED_ALPHA)
                actor.SetPickable(False)
        self._plotter.render()
        self._update_status()
        self._fire_visibility_changed()

    # ------------------------------------------------------------------
    # Visibility overrides
    # ------------------------------------------------------------------

    def _show_all(self) -> None:
        """Reveal every hidden entity and restore pickability."""
        for entity_id in self._hidden:
            actor = self._id_to_actor.get(entity_id)
            if actor is not None:
                actor.VisibilityOn()
                if entity_id[0] in self._pickable_dims:
                    actor.SetPickable(True)
        self._hidden.clear()
        self._plotter.render()
        self._update_status()
        self._fire_visibility_changed()

    # ------------------------------------------------------------------
    # Batch-select API (used by the Qt tree)
    # ------------------------------------------------------------------

    def select_dimtags(
        self, dts: "list[DimTag] | tuple[DimTag, ...]",
        *, replace: bool = True,
    ) -> None:
        """
        Set the working-set picks to *dts* (or extend them if
        ``replace=False``).  Unknown DimTags (entities not in the current
        scene) are silently skipped.  Fires ``_on_pick_changed`` exactly
        once for the whole batch.
        """
        if replace:
            old = list(self._picks)
            self._picks.clear()
            self._pick_history.clear()
            for dt in old:
                self._recolor(dt)
        added = 0
        for dt in dts:
            if dt in self._hidden:
                continue
            if dt[0] not in self._pickable_dims:
                continue
            if dt not in self._id_to_actor:
                continue
            if dt in self._picks:
                continue
            self._picks.append(dt)
            self._pick_history.append(dt)
            self._recolor(dt)
            added += 1
        if self._plotter is not None:
            self._plotter.render()
        self._update_status()
        self._fire_pick_changed()

    # Alias
    set_picks = select_dimtags
