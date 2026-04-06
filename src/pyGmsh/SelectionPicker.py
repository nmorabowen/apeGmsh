"""
SelectionPicker
===============

Headless core of the interactive BRep picker.  Manages the VTK scene
(one actor per entity), pick state, hide/isolate, box-select, hover
highlight and keybindings — but does **not** own the window.  The Qt
UI lives in ``SelectionPickerUI`` and is opened by ``.show()``.

Usage
-----
    # General workflow — open the viewer, create / rename / delete
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
                  LEFT drag         box-select (L→R window, R→L crossing)
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
a toolbar with New/Rename/Delete group and a parallel ↔ perspective
projection toggle.  On close, every group staged during the session
is flushed to Gmsh (existing groups with the same name are replaced).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import gmsh
import numpy as np

if TYPE_CHECKING:
    from pyGmsh._core import pyGmsh
    from pyGmsh.Model import Model
    from pyGmsh.Selection import Selection


DimTag = tuple[int, int]

# Picked entities get this highlight color
# Palette is protanopia-safe: states separate on the blue ↔ yellow
# axis (fully preserved by protanopia) plus luminance contrast.
_PICK_COLOR       = "#E74C3C"   # red — all entity types
_PICK_COLOR_POINT = "#E74C3C"   # red — same as above (unified)
_HOVER_COLOR      = "#FFD700"   # gold  — yellow axis, distinct from cyan

# Base (unpicked) colors — muted, protanopia-friendly
_IDLE_POINT   = "#E8D5B7"   # warm white  — low-sat, luminance vs hover
_IDLE_CURVE   = "#AAAAAA"   # mid grey    — luminance only, universal
_IDLE_SURFACE = "#5B8DB8"   # steel blue  — blue axis
_IDLE_VOLUME  = "#5A6E82"   # slate       — darker blue-grey, distinct from surface
_EDGE_COLOR   = "#2C4A6E"   # dark navy

# Opacity applied to actors whose dim is not in the pick filter
_FILTER_DIMMED_ALPHA = 0.12
_BG_TOP       = "#1a1a2e"   # viewer dark gradient top
_BG_BOTTOM    = "#16213e"   # viewer dark gradient bottom
_HUD_COLOR    = "white"


class SelectionPicker:
    """
    Interactive BRep picker backed by PyVista.

    One picker targets one physical group.  Opening the window,
    left-clicking entities, and closing the window is enough to
    create the physical group.

    Parameters
    ----------
    parent : pyGmsh
        Owning pyGmsh instance.
    model  : Model
        The Model composite — used for sync.
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
        parent: "pyGmsh",
        model : "Model",
        *,
        physical_group  : str | None = None,
        dims            : list[int] | None = None,
        # --- visual properties ---
        point_size      : float = 2.0,    # dim=0 sphere scale
        line_width      : float = 6.0,    # dim=1 tube width
        surface_opacity : float = 0.35,   # dim=2 face alpha
        show_surface_edges : bool = False, # outline UV grid on surfaces
        # --- sampling ---
        n_curve_samples : int   = 60,
        n_surf_samples  : int   = 16,
    ) -> None:
        self._parent = parent
        self._model  = model

        self._group_name = physical_group
        self._dims       = list(dims) if dims is not None else [0, 1, 2, 3]

        self._n_curve_samples = n_curve_samples
        self._n_surf_samples  = n_surf_samples
        self._point_size      = point_size
        self._line_width      = line_width
        self._surface_opacity = surface_opacity
        self._show_surface_edges = show_surface_edges

        # Selection highlight colour — mutable so the prefs panel can
        # change it at runtime.
        self._pick_color: str = _PICK_COLOR

        # Working set — the entities currently selected
        self._picks: list[DimTag] = []

        # Active pick-filter dims (which dims respond to clicks)
        self._pickable_dims: set[int] = set(self._dims)

        # Hidden entities (manually hidden via H / isolated via I)
        self._hidden: set[DimTag] = set()

        # Populated on .show()
        self._plotter = None
        self._actor_to_dimtag: dict[int, DimTag] = {}
        self._dimtag_to_actor: dict[DimTag, object] = {}
        self._pick_history: list[DimTag] = []

        # Hover-highlight state (modern-GUI pixel-precise feedback)
        self._hover_dt: DimTag | None = None
        self._hover_picker = None   # lazy vtkCellPicker
        self._hover_throttle: int = 0   # skip N-1 of N move events

        # Tab-cycling: Revit-style pick cycling through overlapping entities
        self._tab_candidates: list[DimTag] = []
        self._tab_index: int = 0
        self._tab_pos: tuple[int, int] | None = None  # screen pos of last click
        self._drag_threshold: int = 8  # px before LMB-move becomes a drag

        # Observer callbacks — the Qt UI wires these to refresh the tree/HUD.
        # Fired after any change to _picks or to visibility state.
        self._on_pick_changed: list[Callable[[], None]] = []
        self._on_visibility_changed: list[Callable[[], None]] = []
        self._on_hover_changed: list[Callable[[], None]] = []

        # Active physical-group state.  The picker supports editing multiple
        # groups in one session — the tree UI switches the active group via
        # set_active_group(); each switch stages the current picks to
        # _staged_groups and loads the new group's members into _picks.
        self._active_group: str | None = physical_group
        self._staged_groups: dict[str, list[DimTag]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selection(self) -> "Selection":
        """The current working set as a :class:`Selection` object."""
        from pyGmsh.Selection import Selection
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

    def _fire_pick_changed(self) -> None:
        for cb in self._on_pick_changed:
            try:
                cb()
            except Exception:
                pass

    def _fire_visibility_changed(self) -> None:
        for cb in self._on_visibility_changed:
            try:
                cb()
            except Exception:
                pass

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

        _suffix = {0: "p", 1: "c", 2: "s", 3: "v"}
        if len(by_dim) == 1:
            d = next(iter(by_dim))
            pg = gmsh.model.addPhysicalGroup(d, by_dim[d])
            gmsh.model.setPhysicalName(d, pg, group_name)
            return pg

        last_pg = None
        for d, ts in by_dim.items():
            full = f"{group_name}_{_suffix[d]}"
            last_pg = gmsh.model.addPhysicalGroup(d, ts)
            gmsh.model.setPhysicalName(d, last_pg, full)
        return last_pg

    def show(
        self, *, title: str | None = None, maximized: bool = True,
    ) -> "SelectionPicker":
        """
        Open the picker window (Qt + pyvistaqt).  Blocks until closed.
        On close, flushes every staged physical group (including the
        active one) to Gmsh.
        """
        from pyGmsh.SelectionPickerUI import SelectionPickerWindow

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
        window = SelectionPickerWindow(
            self, title=title or default_title, maximized=maximized,
        )
        window.exec()

        # Commit all staged groups to Gmsh
        self._flush_staged_to_gmsh()

        if self._parent._verbose:
            n_groups = sum(1 for m in self._staged_groups.values() if m)
            n_picks = len(self._picks)
            print(
                f"[picker] closed — {n_groups} physical group(s) written, "
                f"{n_picks} picks in working set"
            )
        return self

    # ------------------------------------------------------------------
    # Plotter configuration (called by the Qt window)
    # ------------------------------------------------------------------

    def _setup_on(self, plotter) -> None:
        """
        Configure a PyVista ``BasePlotter`` (either a ``pv.Plotter`` or a
        ``pyvistaqt.QtInteractor``) with background, axes, AA, depth
        peeling, then build the scene, install the custom interactor
        style (LMB pick, LMB drag box-select, MMB rotate, RMB pan, hover
        highlight), install keybindings, and render the HUD.

        Also wires ``_active_group``'s existing members (if any) as the
        current working set — this is the "edit mode" preload.
        """
        self._plotter = plotter
        # Sharpening knobs set directly on the VTK RenderWindow
        # (supported across pyvista versions that don't expose these
        # as Plotter kwargs).
        try:
            rwin = plotter.render_window
            rwin.SetMultiSamples(8)          # 8x MSAA
            rwin.SetLineSmoothing(True)
            rwin.SetPointSmoothing(True)
            rwin.SetPolygonSmoothing(True)
        except Exception:
            pass
        plotter.set_background(_BG_TOP, top=_BG_BOTTOM)
        # Interactive axes orientation widget (Blender-style) —
        # coloured X/Y/Z axes with labels, clickable to snap camera.
        try:
            import vtk as _vtk
            axes = _vtk.vtkAxesActor()
            axes.SetShaftTypeToCylinder()
            axes.SetCylinderRadius(0.05)
            axes.SetTotalLength(1.2, 1.2, 1.2)
            axes.SetNormalizedLabelPosition(1.2, 1.2, 1.2)
            # Axis colours: X=red, Y=green, Z=blue (Blender convention)
            axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(
                1.0, 0.2, 0.2,
            )
            axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(
                0.2, 0.8, 0.2,
            )
            axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(
                0.3, 0.5, 1.0,
            )
            for cap in (axes.GetXAxisCaptionActor2D(),
                        axes.GetYAxisCaptionActor2D(),
                        axes.GetZAxisCaptionActor2D()):
                cap.GetCaptionTextProperty().SetFontSize(14)
                cap.GetCaptionTextProperty().SetBold(True)
                cap.GetCaptionTextProperty().SetShadow(False)
            plotter.add_orientation_widget(
                axes, interactive=True, color=None, opacity=1.0,
            )
        except Exception:
            plotter.add_axes(
                interactive=False, line_width=2, color="white",
                xlabel="X", ylabel="Y", zlabel="Z",
            )
        # SSAA is the highest-quality AA pyvista offers — stacks nicely
        # with the MSAA from multi_samples above.
        try:
            plotter.enable_anti_aliasing("ssaa")
        except Exception:
            try:
                plotter.enable_anti_aliasing()
            except Exception:
                pass
        # Depth peeling renders transparent surfaces correctly (no
        # stacking artifacts when multiple translucent patches overlap).
        try:
            plotter.enable_depth_peeling(number_of_peels=8)
        except Exception:
            pass

        self._actor_to_dimtag.clear()
        self._dimtag_to_actor.clear()

        self._build_scene()
        self._install_keybindings()
        self._install_picker()
        self._recolor_all()
        self._update_status()

        # Force a render so the background gradient and scene are
        # painted before the window becomes visible — prevents a
        # flash of default-theme grey on re-open.
        try:
            plotter.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        import pyvista as pv
        plotter = self._plotter

        if 0 in self._dims:
            # Compute an auto sphere radius as a fraction of the model's
            # bounding-box diagonal.  Spheres are built at the unit
            # ``base_r`` and the user's ``point_size`` is applied via
            # ``actor.SetScale`` — that keeps the slider in sync with the
            # displayed size regardless of the initial ``point_size``.
            try:
                bb = gmsh.model.getBoundingBox(-1, -1)
                diag = float(np.linalg.norm(
                    [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
                ))
                if diag <= 0.0:
                    diag = 1.0
            except Exception:
                diag = 1.0
            base_r = 0.005 * diag
            scale = float(self._point_size)
            for _, tag in gmsh.model.getEntities(dim=0):
                try:
                    xyz = np.array(gmsh.model.getValue(0, tag, []))
                    sphere = pv.Sphere(
                        radius=base_r, center=xyz,
                        theta_resolution=14, phi_resolution=14,
                    )
                    actor = plotter.add_mesh(
                        sphere, color=_IDLE_POINT,
                        smooth_shading=True,
                        pickable=True,
                    )
                    # SetScale() scales the actor about its origin; our
                    # spheres are centred at xyz so we need to pin the
                    # origin to the centre before scaling.
                    actor.SetOrigin(xyz[0], xyz[1], xyz[2])
                    actor.SetScale(scale, scale, scale)
                    self._register_actor(actor, (0, tag))
                except Exception:
                    pass

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
                except Exception:
                    pass

        if 2 in self._dims:
            for _, tag in gmsh.model.getEntities(dim=2):
                try:
                    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
                    n = self._n_surf_samples
                    u_vals = np.linspace(lo[0], hi[0], n)
                    v_vals = np.linspace(lo[1], hi[1], n)
                    uu, vv = np.meshgrid(u_vals, v_vals)
                    uv_flat = np.column_stack(
                        [uu.ravel(), vv.ravel()]
                    ).ravel().tolist()
                    pts = np.array(
                        gmsh.model.getValue(2, tag, uv_flat)
                    ).reshape(-1, 3)
                    faces = []
                    for ii in range(n - 1):
                        for jj in range(n - 1):
                            a = ii * n + jj
                            b = a + 1
                            c = (ii + 1) * n + jj
                            d = c + 1
                            faces.extend([3, a, b, d])
                            faces.extend([3, a, d, c])
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
                except Exception:
                    pass

        if 3 in self._dims:
            # Render each volume as the combined mesh of its boundary
            # surfaces — one actor per volume, so a click anywhere on
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
                    n = self._n_surf_samples
                    for bd, btag in boundary:
                        if bd != 2:
                            continue
                        try:
                            lo, hi = gmsh.model.getParametrizationBounds(
                                2, btag,
                            )
                            u_vals = np.linspace(lo[0], hi[0], n)
                            v_vals = np.linspace(lo[1], hi[1], n)
                            uu, vv = np.meshgrid(u_vals, v_vals)
                            uv_flat = np.column_stack(
                                [uu.ravel(), vv.ravel()]
                            ).ravel().tolist()
                            pts = np.array(
                                gmsh.model.getValue(2, btag, uv_flat)
                            ).reshape(-1, 3)
                        except Exception:
                            continue
                        all_pts.append(pts)
                        for ii in range(n - 1):
                            for jj in range(n - 1):
                                a = offset + ii * n + jj
                                b = a + 1
                                c = offset + (ii + 1) * n + jj
                                d = c + 1
                                all_faces.extend([3, a, b, d])
                                all_faces.extend([3, a, d, c])
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
                except Exception:
                    pass

    def _register_actor(self, actor, dt: DimTag) -> None:
        self._actor_to_dimtag[id(actor)] = dt
        self._dimtag_to_actor[dt] = actor

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def _install_picker(self) -> None:
        """Install camera-control style + high-priority observers for
        click picking, rubber-band box-select, and hover highlight.

            LEFT click           : pick / toggle entity (pixel-accurate)
            LEFT drag            : rubber-band box-select (additive)
                                       L→R = window (enclosed)
                                       R→L = crossing (overlap)
            Ctrl+LEFT click      : unpick entity under cursor
            Ctrl+LEFT drag       : rubber-band box-UNselect
            MIDDLE drag          : pan camera
            Shift+MIDDLE drag    : rotate camera (orbit)
            WHEEL                : zoom
            hover (no button)    : highlight entity under cursor (gold)

        The observer pattern (with AbortFlagOn) is used instead of a
        subclassed interactor style — it composes reliably with pyvistaqt's
        QtInteractor, which otherwise can swallow or re-install styles.
        """
        import vtk
        iren_wrap = self._plotter.iren
        iren = iren_wrap.interactor
        renderer = self._plotter.renderer

        # Base trackball style — we override ALL button handling via
        # observers (Python subclass overrides of C++ virtual methods
        # don't reliably dispatch in vtkmodules).  The style is only
        # used for its OnMouseMove camera-update logic (reads state set
        # by StartRotate/StartPan and moves the camera).
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
        self._interactor_style = style

        # Per-gesture state
        self._press_pos: tuple[int, int] | None = None
        self._dragging: bool = False
        self._ctrl_held: bool = False
        self._rubberband_actor = None
        self._rubberband_pts = None
        self._rubberband_poly = None

        # Drag threshold is read from picker_self._drag_threshold at
        # runtime so the preferences slider can adjust it live.

        picker_self = self

        def _ensure_rubberband() -> None:
            if picker_self._rubberband_actor is not None:
                return
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(4)
            for i in range(4):
                pts.SetPoint(i, 0.0, 0.0, 0.0)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(5)
            for i in (0, 1, 2, 3, 0):
                lines.InsertCellPoint(i)
            poly = vtk.vtkPolyData()
            poly.SetPoints(pts)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputData(poly)
            coord = vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            mapper.SetTransformCoordinate(coord)
            actor = vtk.vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            actor.GetProperty().SetLineWidth(1.5)
            actor.VisibilityOff()
            renderer.AddActor2D(actor)
            picker_self._rubberband_actor = actor
            picker_self._rubberband_pts = pts
            picker_self._rubberband_poly = poly

        def _update_rubberband(x0, y0, x1, y1) -> None:
            _ensure_rubberband()
            pts = picker_self._rubberband_pts
            pts.SetPoint(0, x0, y0, 0.0)
            pts.SetPoint(1, x1, y0, 0.0)
            pts.SetPoint(2, x1, y1, 0.0)
            pts.SetPoint(3, x0, y1, 0.0)
            pts.Modified()
            picker_self._rubberband_poly.Modified()
            crossing = x1 < x0
            prop = picker_self._rubberband_actor.GetProperty()
            try:
                prop.SetLineStipplePattern(0x3333 if crossing else 0xFFFF)
                prop.SetLineStippleRepeatFactor(1)
            except Exception:
                pass
            picker_self._rubberband_actor.VisibilityOn()
            picker_self._plotter.render()

        def _hide_rubberband() -> None:
            if picker_self._rubberband_actor is not None:
                picker_self._rubberband_actor.VisibilityOff()
                picker_self._plotter.render()

        # Software cell picker — avoids vtkPropPicker's hardware-selection
        # path, which is unreliable under pyvistaqt on Windows and spams
        # "Too many props" OpenGL errors on every hover.
        click_picker = vtk.vtkCellPicker()
        click_picker.SetTolerance(0.005)

        def _peel_candidates_at(x: int, y: int) -> list[DimTag]:
            """Peel-pick: repeatedly pick at (x, y), hiding each hit
            actor before re-picking, to find ALL entities stacked under
            the cursor at any depth.  Restores visibility afterwards.
            Pixel-precise — no radius or bbox projection needed."""
            candidates: list[DimTag] = []
            hidden_during_peel: list = []     # (actor, DimTag)
            for _ in range(50):               # safety cap
                click_picker.Pick(x, y, 0, renderer)
                prop = click_picker.GetViewProp()
                if prop is None:
                    break
                dt = picker_self._actor_to_dimtag.get(id(prop))
                if dt is None:
                    break
                if dt[0] not in picker_self._pickable_dims:
                    break
                if dt in picker_self._hidden:
                    break
                candidates.append(dt)
                # Temporarily hide so next pick peels through
                prop.VisibilityOff()
                hidden_during_peel.append(prop)
            # Restore all peeled actors
            for a in hidden_during_peel:
                a.VisibilityOn()
            return candidates

        def _pixel_pick(x: int, y: int) -> None:
            click_picker.Pick(x, y, 0, renderer)
            prop = click_picker.GetViewProp()
            if prop is None:
                return
            dt = picker_self._actor_to_dimtag.get(id(prop))
            if dt is not None:
                # Build Tab-cycle candidate list via peel-picking —
                # pixel-precise at any depth, no radius needed.
                picker_self._tab_candidates = _peel_candidates_at(x, y)
                picker_self._tab_pos = (x, y)
                picker_self._tab_index = 0
                # Make sure the actually-clicked entity is first
                if dt in picker_self._tab_candidates:
                    picker_self._tab_candidates.remove(dt)
                picker_self._tab_candidates.insert(0, dt)
                picker_self._toggle_pick(dt)

        def _pixel_unpick(x: int, y: int) -> None:
            """Ctrl+click: remove the entity under cursor from picks."""
            click_picker.Pick(x, y, 0, renderer)
            prop = click_picker.GetViewProp()
            if prop is None:
                return
            dt = picker_self._actor_to_dimtag.get(id(prop))
            if dt is not None and dt in picker_self._picks:
                picker_self._picks.remove(dt)
                picker_self._pick_history = [
                    d for d in picker_self._pick_history if d != dt
                ]
                picker_self._recolor(dt)
                picker_self._update_status()
                picker_self._fire_pick_changed()

        # ---- Observers ----
        # NOTE: `caller` (the interactor) does NOT have SetAbortFlag().
        # The abort flag lives on the vtkCommand, which we retrieve via
        # the observer tag stored in closure.  Setting cmd.SetAbortFlag(1)
        # stops subsequent observers (including the style's) from running
        # on this event dispatch.

        _tag_press: list[int] = [0]
        _tag_move:  list[int] = [0]
        _tag_release: list[int] = [0]

        def _abort(caller, tag: int) -> None:
            cmd = caller.GetCommand(tag)
            if cmd is not None:
                cmd.SetAbortFlag(1)

        def on_lmb_press(caller, _event):
            # All LMB gestures are ours — pick, box-select, or
            # Ctrl variants for unpick / box-unselect.
            picker_self._press_pos = caller.GetEventPosition()
            picker_self._dragging = False
            picker_self._ctrl_held = bool(caller.GetControlKey())
            _abort(caller, _tag_press[0])

        def on_mouse_move(caller, _event):
            # LMB-drag in our custom mode → rubber-band
            if picker_self._press_pos is not None:
                px, py = caller.GetEventPosition()
                sx, sy = picker_self._press_pos
                if not picker_self._dragging:
                    if (px - sx) ** 2 + (py - sy) ** 2 > picker_self._drag_threshold ** 2:
                        picker_self._dragging = True
                if picker_self._dragging:
                    _update_rubberband(sx, sy, px, py)
                _abort(caller, _tag_move[0])
                return
            # Idle hover
            px, py = caller.GetEventPosition()
            picker_self._update_hover(px, py)
            # Do NOT abort — allow trackball to see move events for MMB

        def on_lmb_release(caller, _event):
            if picker_self._press_pos is None:
                return
            x0, y0 = picker_self._press_pos
            x1, y1 = caller.GetEventPosition()
            ctrl = picker_self._ctrl_held
            _hide_rubberband()
            if picker_self._dragging:
                rx0, rx1 = min(x0, x1), max(x0, x1)
                ry0, ry1 = min(y0, y1), max(y0, y1)
                crossing = x1 < x0
                if ctrl:
                    # Ctrl+drag → box-UNselect
                    picker_self._do_box_unselect(
                        rx0, ry0, rx1, ry1, crossing=crossing,
                    )
                else:
                    if picker_self._parent._verbose:
                        print(
                            f"[picker] LMB drag → box-select "
                            f"({x0},{y0})→({x1},{y1}) "
                            f"crossing={crossing}"
                        )
                    picker_self._do_box_select(
                        rx0, ry0, rx1, ry1, crossing=crossing,
                    )
                picker_self._update_status()
            else:
                if ctrl:
                    _pixel_unpick(x1, y1)
                else:
                    _pixel_pick(x1, y1)
            picker_self._press_pos = None
            picker_self._dragging = False
            picker_self._ctrl_held = False
            _abort(caller, _tag_release[0])

        # ---- MMB / RMB observers (camera pan + rotate) ----
        _tag_mmb_press: list[int] = [0]
        _tag_mmb_release: list[int] = [0]
        _tag_rmb_press: list[int] = [0]
        _tag_rmb_release: list[int] = [0]

        def on_mmb_press(caller, _event):
            if caller.GetShiftKey():
                picker_self._interactor_style.StartRotate()
            else:
                picker_self._interactor_style.StartPan()
            _abort(caller, _tag_mmb_press[0])

        def on_mmb_release(caller, _event):
            state = picker_self._interactor_style.GetState()
            if state == 1:      # VTKIS_ROTATE
                picker_self._interactor_style.EndRotate()
            elif state == 2:    # VTKIS_PAN
                picker_self._interactor_style.EndPan()
            _abort(caller, _tag_mmb_release[0])

        def on_rmb_press(caller, _event):
            picker_self._interactor_style.StartPan()
            _abort(caller, _tag_rmb_press[0])

        def on_rmb_release(caller, _event):
            state = picker_self._interactor_style.GetState()
            if state == 2:
                picker_self._interactor_style.EndPan()
            _abort(caller, _tag_rmb_release[0])

        # High priority (runs before style's default handlers)
        _tag_press[0] = iren.AddObserver(
            "LeftButtonPressEvent", on_lmb_press, 10.0,
        )
        _tag_move[0] = iren.AddObserver(
            "MouseMoveEvent", on_mouse_move, 10.0,
        )
        _tag_release[0] = iren.AddObserver(
            "LeftButtonReleaseEvent", on_lmb_release, 10.0,
        )
        _tag_mmb_press[0] = iren.AddObserver(
            "MiddleButtonPressEvent", on_mmb_press, 10.0,
        )
        _tag_mmb_release[0] = iren.AddObserver(
            "MiddleButtonReleaseEvent", on_mmb_release, 10.0,
        )
        _tag_rmb_press[0] = iren.AddObserver(
            "RightButtonPressEvent", on_rmb_press, 10.0,
        )
        _tag_rmb_release[0] = iren.AddObserver(
            "RightButtonReleaseEvent", on_rmb_release, 10.0,
        )

    # ------------------------------------------------------------------
    # Box-select (AutoCAD style) — invoked by LEFT-drag in _install_picker
    # ------------------------------------------------------------------

    def _project_centroid(self, actor) -> tuple[float, float] | None:
        """Project an actor's 3D bounding-box centre to screen coords.
        Returns ``(sx, sy)`` or ``None`` if bounds are degenerate."""
        bx0, bx1, by0, by1, bz0, bz1 = actor.GetBounds()
        if bx0 > bx1:
            return None
        cx = (bx0 + bx1) * 0.5
        cy = (by0 + by1) * 0.5
        cz = (bz0 + bz1) * 0.5
        renderer = self._plotter.renderer
        renderer.SetWorldPoint(cx, cy, cz, 1.0)
        renderer.WorldToDisplay()
        dx, dy, _dz = renderer.GetDisplayPoint()
        return (dx, dy)

    def _do_box_select(
        self, x0: float, y0: float, x1: float, y1: float,
        *, crossing: bool,
    ) -> None:
        """Select entities whose projected centroid falls inside the
        rubber-band rectangle.  Uses a single-point test (centroid)
        instead of full bbox projection — much more precise, especially
        for surfaces at grazing angles."""
        added = 0
        # VTK display coords have Y=0 at bottom; GetEventPosition()
        # already flips Y through SetEventInformationFlipY, so both
        # the rubber-band box and WorldToDisplay are in the same space.
        # However on high-DPI Windows, Qt logical-pixels may differ
        # from VTK physical-pixels.  Detect and apply the DPI ratio.
        try:
            rw = self._plotter.render_window
            vw, vh = rw.GetSize()          # VTK physical pixels
            aw, ah = rw.GetActualSize()    # may differ on HiDPI
            # If actual vs logical differ, scale the box to physical
            sx_ratio = aw / vw if vw else 1.0
            sy_ratio = ah / vh if vh else 1.0
        except Exception:
            sx_ratio = sy_ratio = 1.0
        bx0 = x0 * sx_ratio
        bx1 = x1 * sx_ratio
        by0 = y0 * sy_ratio
        by1 = y1 * sy_ratio

        if self._parent._verbose:
            print(f"[picker] box raw=({x0},{y0})-({x1},{y1}) "
                  f"scaled=({bx0:.0f},{by0:.0f})-({bx1:.0f},{by1:.0f}) "
                  f"ratio=({sx_ratio:.2f},{sy_ratio:.2f})")

        candidates = 0
        for dt, actor in list(self._dimtag_to_actor.items()):
            if dt[0] not in self._pickable_dims:
                continue
            if dt in self._hidden:
                continue
            if dt in self._picks:
                continue   # box-select is additive
            pt = self._project_centroid(actor)
            if pt is None:
                continue
            sx, sy = pt
            candidates += 1
            # Print first few centroids for debugging
            if self._parent._verbose and candidates <= 5:
                hit_dbg = bx0 <= sx <= bx1 and by0 <= sy <= by1
                print(f"  centroid {dt}: display=({sx:.0f},{sy:.0f}) hit={hit_dbg}")
            hit = bx0 <= sx <= bx1 and by0 <= sy <= by1
            if hit:
                self._picks.append(dt)
                self._pick_history.append(dt)
                self._recolor(dt)
                added += 1
        self._plotter.render()
        if added:
            self._fire_pick_changed()
        if self._parent._verbose:
            mode = "crossing" if crossing else "window"
            print(f"[picker] box-select ({mode}) {candidates} candidates, added {added} entities")

    def _do_box_unselect(
        self, x0: float, y0: float, x1: float, y1: float,
        *, crossing: bool,
    ) -> None:
        """Ctrl+drag counterpart to ``_do_box_select``: REMOVES every
        currently-picked entity whose projected centroid falls inside
        the rubber-band rectangle."""
        removed = 0
        for dt in list(self._picks):
            actor = self._dimtag_to_actor.get(dt)
            if actor is None:
                continue
            pt = self._project_centroid(actor)
            if pt is None:
                continue
            sx, sy = pt
            hit = x0 <= sx <= x1 and y0 <= sy <= y1
            if hit:
                self._picks.remove(dt)
                self._pick_history = [
                    d for d in self._pick_history if d != dt
                ]
                self._recolor(dt)
                removed += 1
        self._plotter.render()
        if removed:
            self._fire_pick_changed()
        if self._parent._verbose:
            mode = "crossing" if crossing else "window"
            print(f"[picker] box-UNselect ({mode}) removed {removed} entities")

    def _toggle_pick(self, dt: DimTag) -> None:
        # Respect current dim-filter — picks on disabled dims are ignored
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
        """Tab — cycle through overlapping entities at the last click
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

    # Scale multiplier for dim=0 spheres when picked — makes the
    # selection clearly visible even at small point_size values.
    _PICK_SCALE_BOOST = 1.6

    def _recolor(self, dt: DimTag) -> None:
        actor = self._dimtag_to_actor.get(dt)
        if actor is None:
            return
        if dt == self._hover_dt and dt not in self._picks:
            actor.GetProperty().SetColor(_hex_to_rgb(_HOVER_COLOR))
        elif dt in self._picks:
            actor.GetProperty().SetColor(_hex_to_rgb(self._pick_color))
        else:
            idle = {
                0: _IDLE_POINT, 1: _IDLE_CURVE,
                2: _IDLE_SURFACE, 3: _IDLE_VOLUME,
            }[dt[0]]
            actor.GetProperty().SetColor(_hex_to_rgb(idle))
        # For dim=0 (point spheres), scale up when picked so the
        # selection is unmissable — especially useful for protanopia
        # where subtle colour shifts may not be enough.
        if dt[0] == 0:
            base = float(self._point_size)
            if dt in self._picks:
                s = base * self._PICK_SCALE_BOOST
            else:
                s = base
            actor.SetScale(s, s, s)
        self._plotter.render()

    def _recolor_all(self) -> None:
        """Recolor every actor based on current hover / picks state.
        Called after set_active_group() reloads the working set."""
        for dt in self._dimtag_to_actor:
            self._recolor(dt)

    def _update_hover(self, x: int, y: int) -> None:
        """Pixel-precise hover highlight using a software cell picker
        (vtkCellPicker).  Intentionally avoids vtkPropPicker whose
        hardware-selection path is unreliable under pyvistaqt on Windows.

        Throttled via ``_hover_throttle`` — only every Nth call runs the
        picker, to keep idle-move event dispatch cheap.
        """
        import vtk
        if self._plotter is None:
            return
        # Throttle: only pick every Nth move event
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
            candidate = self._actor_to_dimtag.get(id(prop))
            if candidate is not None:
                if candidate[0] in self._pickable_dims and candidate not in self._hidden:
                    new_dt = candidate
        if new_dt == self._hover_dt:
            return
        old_dt = self._hover_dt
        self._hover_dt = new_dt
        if old_dt is not None:
            self._recolor(old_dt)
        if new_dt is not None:
            self._recolor(new_dt)
        # Notify UI (Entity Info tab)
        for cb in self._on_hover_changed:
            try:
                cb()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Keybindings + HUD
    # ------------------------------------------------------------------

    def _install_keybindings(self) -> None:
        plotter = self._plotter
        # Undo
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

        # Visibility
        plotter.add_key_event("h", self._hide_selected)
        plotter.add_key_event("H", self._hide_selected)
        plotter.add_key_event("i", self._isolate_selected)
        plotter.add_key_event("I", self._isolate_selected)
        plotter.add_key_event("r", self._show_all)
        plotter.add_key_event("R", self._show_all)

    def _update_status(self) -> None:
        """Redraw the HUD text overlay summarising pick state."""
        if self._plotter is None:
            return
        by_dim: dict[int, int] = {}
        for d, _ in self._picks:
            by_dim[d] = by_dim.get(d, 0) + 1
        parts = []
        names = {0: "pt", 1: "crv", 2: "srf", 3: "vol"}
        for d in sorted(by_dim):
            parts.append(f"{by_dim[d]} {names[d]}")
        pick_text = ", ".join(parts) if parts else "none"
        dim_filter = "".join(
            str(d) for d in sorted(self._pickable_dims)
        ) or "-"
        active = self._active_group or "(none)"
        text = (
            f"picks: {pick_text}   "
            f"filter: dim[{dim_filter}]   "
            f"active: {active}"
        )
        try:
            self._plotter.remove_actor("_picker_hud")
        except Exception:
            pass
        try:
            self._plotter.add_text(
                text, name="_picker_hud",
                position="lower_left", font_size=10, color=_HUD_COLOR,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Undo / dim filter / visibility
    # ------------------------------------------------------------------

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
        for dt, actor in self._dimtag_to_actor.items():
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

    def _hide_selected(self) -> None:
        """Hide every currently-picked entity."""
        if not self._picks:
            return
        for dt in list(self._picks):
            actor = self._dimtag_to_actor.get(dt)
            if actor is not None:
                actor.VisibilityOff()
                actor.SetPickable(False)
            self._hidden.add(dt)
        self._picks.clear()
        self._pick_history.clear()
        self._plotter.render()
        self._update_status()
        self._fire_pick_changed()
        self._fire_visibility_changed()

    def _isolate_selected(self) -> None:
        """Hide every entity that is NOT currently picked."""
        if not self._picks:
            return
        picks = set(self._picks)
        for dt, actor in self._dimtag_to_actor.items():
            if dt in picks:
                continue
            actor.VisibilityOff()
            actor.SetPickable(False)
            self._hidden.add(dt)
        self._plotter.render()
        self._update_status()
        self._fire_visibility_changed()

    def _show_all(self) -> None:
        """Reveal every hidden entity and restore pickability."""
        if not self._hidden:
            return
        for dt in list(self._hidden):
            actor = self._dimtag_to_actor.get(dt)
            if actor is not None:
                actor.VisibilityOn()
                if dt[0] in self._pickable_dims:
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
            # Deselect every currently-picked entity first
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
            if dt not in self._dimtag_to_actor:
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


# ======================================================================
# Module-level helpers
# ======================================================================

def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Convert ``"#RRGGBB"`` to an (R, G, B) tuple of floats in [0, 1]."""
    s = hex_str.lstrip("#")
    if len(s) != 6:
        return (1.0, 1.0, 1.0)
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


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
