"""GaussPointDiagram — sphere markers at GP world positions.

Reads :class:`GaussSlab`, calls ``slab.global_coords(fem)`` to map
natural coords to world coords (proper shape fns for hex8 / quad4;
centroid + bbox approximation otherwise), and renders one sphere glyph
per GP, coloured by the GP value.

Render seam (ADR 0042, R-B Wave 2). Emits one sphere
:class:`GlyphLayer` via the backend; holds no VTK objects. Sphere
**size** is fixed (a constant ``scales`` array derived from the model
diagonal); glyph **colour** uses the raw GP value (``color_scalar`` +
``ColorSpec(by_array)``). The Qt LUT mirror stays diagram-side — its
``changed`` signal is translated to a plain ``ColorSpec`` / ``LutSpec``
and pushed through ``backend.set_layer_color`` (the backend never sees
Qt).

We deliberately use real sphere geometry rather than
``render_points_as_spheres=True``: on co-planar (z=0) 2-D models that
flag loses its billboards to z-fighting with the substrate fill, and
the only mitigation is global VTK state that also disturbs the
wireframe overlay. Real spheres sit a finite radius above / below the
plane and render unambiguously.

Picking stays on the legacy ``PickInventory`` path (deferred to R-D): the
diagram registers the backend's glyph actor and reverse-maps a picked
cell index back to a GP. Because the backend rebuilds the glyph actor
on each ``update_layer``, the registration is refreshed whenever the
actor identity changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._kinds import register_diagram_kind
from ._scalar_color_support import ScalarColorSupport
from ._styles import GaussMarkerStyle
from ..scene_ir import ColorSpec, GlyphLayer, LutSpec, PointSet

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


@register_diagram_kind(
    label="Gauss point markers", style_class=GaussMarkerStyle, order=70,
)
class GaussPointDiagram(ScalarColorSupport, Diagram):
    """Sphere markers at Gauss-point world positions, colored by value."""

    kind = "gauss_marker"
    topology = "gauss"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, GaussMarkerStyle):
            raise TypeError(
                "GaussPointDiagram requires a GaussMarkerStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        self._layer: Optional[GlyphLayer] = None
        self._handle: Any = None
        self._coords: Optional[ndarray] = None        # GP world centers (n, 3)
        self._scales: Optional[ndarray] = None         # fixed per-glyph size
        self._gp_values: Optional[ndarray] = None      # per-center scalar
        self._element_ids_to_read: tuple[int, ...] = ()
        # Cached at attach so deformation sync can re-evaluate shape
        # functions against deformed substrate coords without
        # re-reading the slab from disk.
        self._gp_element_index: Optional[ndarray] = None
        self._gp_natural_coords: Optional[ndarray] = None
        # Reference world coords captured at attach; used by
        # sync_substrate_points as the deformation baseline.
        self._gp_centers_at_build: Optional[ndarray] = None
        # Glyph cell→center mapping: each GP center contributes
        # ``_glyph_cells_per_center`` consecutive cells in the backend's
        # output glyph PolyData. Lets ``resolve_picked_cell`` invert the
        # picker's cell index back to a GP center. Derived from the
        # backend layer handle's dataset after the glyph is added.
        self._glyph_cells_per_center: int = 0
        # Scalar-bar + runtime colour state + LUT mirror (mixin).
        self._init_scalar_color_state()
        # The glyph actor currently registered on the scene PickInventory.
        self._registered_actor: Any = None

    # ------------------------------------------------------------------
    # Attach / detach / update
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "GaussPointDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, view, scene)
        style: GaussMarkerStyle = self.spec.style    # type: ignore[assignment]

        # ── Resolve element IDs (continuum: 2-D + 3-D) ──────────────
        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_continuum_element_ids(view)
        if element_ids.size == 0:
            return
        self._element_ids_to_read = tuple(int(e) for e in element_ids)

        # ── Step 0 read ─────────────────────────────────────────────
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.gauss.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[0],
            )
        except Exception as exc:
            raise RuntimeError(
                f"GaussPointDiagram could not read gauss slab: {exc}"
            )
        if slab.values.size == 0 or slab.element_index.size == 0:
            return

        # ── World positions via the slab's shape-fn helper ──────────
        # Cache the per-GP arrays so deformation sync can recompute
        # world coords later against deformed substrate points without
        # re-reading the slab from disk.
        self._gp_element_index = np.asarray(
            slab.element_index, dtype=np.int64,
        ).copy()
        self._gp_natural_coords = np.asarray(
            slab.natural_coords, dtype=np.float64,
        ).copy()

        try:
            world = slab.global_coords(view)
        except Exception:
            # Defensive: fall back to a zero array — we'd rather show
            # something than crash the viewer.
            world = np.zeros((slab.element_index.size, 3), dtype=np.float64)

        self._coords = np.asarray(world, dtype=np.float64).copy()
        self._gp_centers_at_build = self._coords.copy()
        self._gp_values = np.asarray(slab.values[0], dtype=np.float64).copy()

        # Initial clim
        if style.clim is not None:
            self._initial_clim = (
                float(style.clim[0]), float(style.clim[1]),
            )
        else:
            data = self._gp_values
            finite = data[np.isfinite(data)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                self._initial_clim = (lo, hi)
            else:
                self._initial_clim = (0.0, 1.0)

        # ── Fixed sphere size off the model diagonal ────────────────
        # ``style.point_size`` keeps its prior semantic (bigger = more
        # visible) but scales a world-space sphere size off the model
        # diagonal — same convention as the pre-solve mesh viewer's node
        # cloud. The backend's unit sphere geometry (radius 0.5) is
        # scaled by this value, so we double the desired world radius.
        diag = float(getattr(scene, "model_diagonal", 0.0)) or 1.0
        radius = 0.003 * diag * max(0.1, float(style.point_size) / 10.0)
        n = self._coords.shape[0]
        self._scales = np.full(n, 2.0 * radius, dtype=np.float64)

        self._layer = self._build_layer(self._gp_values)
        self._handle = self._backend.add_layer(self._layer)

        # Picking + LUT mirror.
        self._update_glyph_cells_per_center()
        self._register_pick()
        self._init_lut()
        if self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle, self._make_scalar_bar_spec(),
            )

    def update_to_step(self, step_index: int) -> None:
        if (
            self._layer is None
            or self._handle is None
            or self._gp_values is None
        ):
            return
        results = self._scoped_results()
        if results is None:
            return
        try:
            slab = results.elements.gauss.get(
                ids=self._element_ids_to_read,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return
        if slab.values.size == 0:
            return
        slab_values = np.asarray(slab.values[0], dtype=np.float64)
        if slab_values.size != self._gp_values.size:
            return
        self._gp_values = slab_values
        self._layer = self._build_layer(self._gp_values)
        self._backend.update_layer(self._handle, self._layer)
        self._register_pick()

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        """Translate every GP sphere to follow the deformed substrate.

        Re-evaluates the per-GP world coords against ``deformed_pts``
        (or ``fem.nodes.coords`` when ``None``) and re-emits the glyph
        layer at the new centers (colours / sizes unchanged).
        """
        if (
            self._layer is None
            or self._handle is None
            or self._view is None
            or self._gp_element_index is None
            or self._gp_natural_coords is None
            or self._gp_values is None
        ):
            return
        try:
            from apeGmsh.results._gauss_world_coords import (
                compute_global_coords_from_arrays,
            )
            new_centers = compute_global_coords_from_arrays(
                self._gp_element_index,
                self._gp_natural_coords,
                self._view,  # type: ignore[arg-type]
                node_coords_override=deformed_pts,
            )
        except Exception:
            return
        self._coords = np.asarray(new_centers, dtype=np.float64)
        self._layer = self._build_layer(self._gp_values)
        self._backend.update_layer(self._handle, self._layer)
        self._register_pick()

    def detach(self) -> None:
        self._remove_scalar_bar(self._scalar_bar_title())
        self._teardown_lut()
        # Drop the GP actor from the PickInventory inventory before
        # clearing local state.
        self._unregister_pick()
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._layer = None
        self._handle = None
        self._coords = None
        self._scales = None
        self._gp_values = None
        self._element_ids_to_read = ()
        self._initial_clim = None
        self._gp_element_index = None
        self._gp_natural_coords = None
        self._gp_centers_at_build = None
        self._glyph_cells_per_center = 0
        super().detach()

    # ------------------------------------------------------------------
    # Visibility (backend-routed)
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    # ------------------------------------------------------------------
    # Picking — invert glyph cell index to GP center
    # ------------------------------------------------------------------

    def resolve_picked_cell(
        self, cell_id: int,
    ) -> Optional[tuple[int, int, "ndarray"]]:
        """Map a picker cell index back to ``(element_id, gp_index, world)``.

        Returns
        -------
        ``(element_id, gp_index, world_xyz)`` if ``cell_id`` lies within
        this diagram's glyph cells, where ``gp_index`` is the row in
        the slab that this GP came from (also doubles as the input
        center index in the glyph PolyData). Returns ``None`` when
        the cell index is out of range, the diagram isn't attached,
        or the necessary metadata is missing — callers should treat a
        ``None`` as "not my pick" and fall through.
        """
        if (
            self._glyph_cells_per_center <= 0
            or self._gp_element_index is None
            or self._coords is None
        ):
            return None
        try:
            cell_id = int(cell_id)
        except Exception:
            return None
        if cell_id < 0:
            return None
        center_idx = cell_id // self._glyph_cells_per_center
        eidx = self._gp_element_index
        if center_idx < 0 or center_idx >= eidx.size:
            return None
        try:
            element_id = int(eidx[center_idx])
        except Exception:
            return None
        # World coords come from the live centers (kept in sync by
        # ``sync_substrate_points``) so the highlight follows
        # deformation if the active geometry is warped.
        try:
            world = np.asarray(
                self._coords[center_idx], dtype=np.float64,
            ).copy()
        except Exception:
            return None
        return (element_id, int(center_idx), world)

    def _register_pick(self) -> None:
        """(Re)register the current backend glyph actor with the
        scene's :class:`PickInventory`.

        The backend rebuilds the glyph actor on every ``update_layer``,
        so the registration is refreshed whenever the actor identity
        changes. No-op when the scene has no pick engine (headless) or
        the backend exposes no actor (recording stub).
        """
        scene = getattr(self, "_scene", None)
        pick_engine = getattr(scene, "pick_engine", None) if scene else None
        if pick_engine is None:
            return
        actor = getattr(self._handle, "actor", None)
        if actor is None or actor is self._registered_actor:
            return
        if self._registered_actor is not None:
            pick_engine.unregister_actor(self._registered_actor)
        pick_engine.register_actor(actor, "gp", self.resolve_picked_cell)
        self._registered_actor = actor

    def _unregister_pick(self) -> None:
        scene = getattr(self, "_scene", None)
        pick_engine = getattr(scene, "pick_engine", None) if scene else None
        if pick_engine is not None and self._registered_actor is not None:
            pick_engine.unregister_actor(self._registered_actor)
        self._registered_actor = None

    def _update_glyph_cells_per_center(self) -> None:
        """Derive cells-per-sphere from the backend handle's dataset.

        The sphere geometry (and thus its cell count) is the backend's
        choice; we read it back off the glyphed dataset so picking is
        independent of the backend's sphere resolution. No-op for the
        recording stub backend (no dataset).
        """
        dataset = getattr(self._handle, "dataset", None)
        n_centers = self._coords.shape[0] if self._coords is not None else 0
        if dataset is None or n_centers == 0:
            return
        try:
            self._glyph_cells_per_center = int(dataset.n_cells // n_centers)
        except Exception:
            self._glyph_cells_per_center = 0

    # ------------------------------------------------------------------
    # Runtime style — clim/cmap/LUT from ScalarColorSupport
    # ------------------------------------------------------------------

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        return self._gp_values

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"gauss_{id(self):x}"

    def _color_array_name(self) -> str:
        return self.spec.selector.component or "_gp_value"

    def _build_layer(self, gp_values: ndarray) -> GlyphLayer:
        """Sphere glyph: fixed size; colour = raw GP value through the LUT."""
        style: GaussMarkerStyle = self.spec.style    # type: ignore[assignment]
        assert self._coords is not None
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        cmap = self._runtime_cmap or style.cmap
        color = ColorSpec(
            mode="by_array",
            array_name=self._color_array_name(),
            lut=LutSpec(name=cmap, vmin=float(clim[0]), vmax=float(clim[1])),
        )
        return GlyphLayer(
            layer_id=self._layer_id(),
            positions=PointSet(self._coords),
            kind="sphere",
            scales=self._scales,
            color_scalar=gp_values,
            color=color,
            opacity=style.opacity,
        )

    @staticmethod
    def _collect_continuum_element_ids(view: "ViewerData") -> ndarray:
        """All 2-D and 3-D element IDs (continuum types)."""
        ids: list[int] = []
        for group in view.elements:
            if group.element_type.dim in (2, 3):
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)
