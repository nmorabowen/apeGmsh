"""ContourDiagram — paint scalar values on the substrate.

Five rendering paths share one diagram, dispatched at attach time
based on ``topology`` × ``averaging`` × per-element GP count:

* **nodes** — ``topology="nodes"``. Values from
  ``results.nodes.get(...)``, painted as point data on a submesh
  extracted by node IDs. The ``averaging`` field is ignored — nodal
  slabs already carry one value per global node.
* **gauss_cell** — ``topology="gauss"``, ``averaging="discrete"``,
  ``n_gp == 1`` per element (CST / tri31, hex8 with one-point
  integration). Painted as cell data on a submesh extracted by
  element IDs.
* **gauss_cell_averaged** — ``topology="gauss"``,
  ``averaging="averaged"``, ``n_gp == 1``. Same data as gauss_cell,
  but spread to corner nodes and averaged across elements sharing
  each node. Smooths boundaries between adjacent CSTs.
* **gauss_node** — ``topology="gauss"``, ``averaging="averaged"``,
  ``n_gp > 1``. Values extrapolated to corner nodes via the inverse
  of the element shape-function matrix, then averaged across
  elements that share a node. Painted as point data.
* **gauss_node_discrete** — ``topology="gauss"``,
  ``averaging="discrete"``, ``n_gp > 1``. Per-element extrapolation
  with **no cross-element averaging**: each element keeps its own
  corner values, rendered on a shattered submesh built via
  ``UnstructuredGrid.separate_cells()``. Boundary jumps are visible.

Render seam (ADR 0042, R-B Wave 2 #4 — final colour diagram). Each
path emits one substrate-submesh :class:`MeshLayer` (point- or
cell-located :class:`ScalarField`) through ``self._backend``; the
diagram holds no VTK objects. Submeshes are extracted via the handed
scene grid (``extract_points`` / ``extract_cells`` / ``separate_cells``
— method calls, no pyvista import) and re-expressed as neutral IR via
``cellblocks_from_grid``.

Performance contract:

* Selector resolves to FEM IDs **once at attach**. The submesh is
  extracted once; per-step reads scatter into a persistent scalar
  buffer in place, then emit through ``backend.update_layer`` — its
  mesh fast path mutates the bound dataset's scalar array without
  re-adding the actor (actor + dataset + mapper identity stable).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._kinds import register_diagram_kind
from ._scalar_color_support import ScalarColorSupport
from ._styles import ContourStyle
from ..scene_ir import (
    CellBlocks,
    ColorSpec,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarBarSpec,
    ScalarField,
)

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


# Style.topology values (user-facing)
_TOPO_NODES = "nodes"
_TOPO_GAUSS = "gauss"

# Style.averaging values (user-facing) — only used when topology="gauss"
_AVG_AVERAGED = "averaged"
_AVG_DISCRETE = "discrete"

# Internal effective-path values — gauss splits four ways depending on
# n_gp per element and the averaging choice.
_EFFECTIVE_NODES = "nodes"
_EFFECTIVE_GAUSS_CELL = "gauss_cell"                # n_gp==1, discrete
_EFFECTIVE_GAUSS_CELL_AVERAGED = "gauss_cell_averaged"   # n_gp==1, averaged
_EFFECTIVE_GAUSS_NODE = "gauss_node"                # n_gp>1, averaged
_EFFECTIVE_GAUSS_NODE_DISCRETE = "gauss_node_discrete"   # n_gp>1, discrete


@register_diagram_kind(label="Contour", style_class=ContourStyle, order=10)
class ContourDiagram(ScalarColorSupport, Diagram):
    """Scalar contour painted on a slice of the substrate mesh.

    The selector picks which nodes / elements carry data; everything
    else stays as the gray substrate. Multiple contour diagrams compose
    naturally — each owns its own submesh layer.

    The class-level ``topology = "nodes"`` declaration informs the Add-
    Diagram dialog which composite to enumerate components from. Per-
    instance Gauss contour is opted into via
    ``ContourStyle.topology = "gauss"``; the user must pick explicitly
    (no auto-resolution against component availability).
    """

    kind = "contour"
    topology = "nodes"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, ContourStyle):
            raise TypeError(
                "ContourDiagram requires a ContourStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        # Backend handle + last-emitted layer + cached geometry.
        self._handle: Any = None
        self._layer: Optional[MeshLayer] = None
        self._points: Optional[PointSet] = None
        self._cells: Optional[CellBlocks] = None
        # Persistent per-point / per-cell scalar buffer, mutated in place
        # by the scatter helpers (no per-step reallocation).
        self._scalar_values: Optional[ndarray] = None
        self._scalar_location: str = "point"

        # Effective topology after resolution at attach.
        self._effective_topology: Optional[str] = None

        # Nodes / extrapolated-node path lookup state.
        self._submesh_pos_of_id: Optional[ndarray] = None
        self._fem_ids_to_read: Optional[ndarray] = None

        # Gauss cell-data path lookup state.
        self._submesh_cell_pos_of_eid: Optional[ndarray] = None
        self._fem_eids_to_read: Optional[ndarray] = None

        # Discrete shattered-submesh runtime state (n_gp>1 + discrete).
        self._discrete_cell_point_offsets: Optional[ndarray] = None

        # Submesh-point -> substrate-row map (vtkOriginalPointIds),
        # cached at attach so sync_substrate_points can re-sample the
        # deformed substrate. None when the extraction carried no map.
        self._substrate_rows: Optional[ndarray] = None

        # Mutable runtime overrides (style is frozen). The clim/cmap
        # halves + scalar bar + LUT mirror come from the mixin.
        self._init_scalar_color_state()
        self._runtime_opacity: Optional[float] = None

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
                "ContourDiagram.attach requires a FEMSceneData (the "
                "viewer's substrate mesh). The Director must call "
                "bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, view, scene)

        topology = self._resolve_topology()
        if topology == _TOPO_GAUSS:
            # Peek at step 0 to choose cell-vs-node sub-path.
            self._attach_gauss(scene)
        else:
            self._effective_topology = _EFFECTIVE_NODES
            self._attach_nodes(scene)

        # Build the LUT mirror once the layer exists, then add the scalar
        # bar (all paths converge here).
        self._init_lut()
        if self._handle is not None and self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle,
                ScalarBarSpec(
                    layer_id=self._handle.layer_id,
                    title=self._scalar_bar_title(),
                    lut=self._current_lutspec(),
                    fmt=self._runtime_fmt or self._scalar_bar_default_fmt(),
                ),
            )

    def update_to_step(self, step_index: int) -> None:
        if self._handle is None or self._scalar_values is None:
            return
        topo = self._effective_topology
        if topo == _EFFECTIVE_GAUSS_CELL:
            fetched = self._fetch_step_values_gauss(step_index)
            if fetched is None:
                return
            slab_eids, slab_values = fetched
            self._scatter_into_cell_scalar(slab_eids, slab_values)
        elif topo in (_EFFECTIVE_GAUSS_NODE, _EFFECTIVE_GAUSS_CELL_AVERAGED):
            fetched = self._fetch_step_values_gauss_node(step_index)
            if fetched is None:
                return
            node_ids, nodal_values = fetched
            self._scatter_into_scalar(node_ids, nodal_values)
        elif topo == _EFFECTIVE_GAUSS_NODE_DISCRETE:
            self._refresh_shattered_at_step(step_index)
        else:    # _EFFECTIVE_NODES
            fetched = self._fetch_step_values(step_index)
            if fetched is None:
                return
            slab_node_ids, slab_values = fetched
            self._scatter_into_scalar(slab_node_ids, slab_values)
        self._push_scalar_update()

    def sync_substrate_points(
        self,
        deformed_pts: "ndarray | None",
        scene: "FEMSceneData",
    ) -> None:
        """Re-sample the submesh points from the (deformed) substrate.

        The submesh is an extracted COPY of the scene grid (the diagram
        emits IR through the backend and holds no shared VTK dataset),
        so mutating ``scene.grid.points`` alone leaves the contour at
        the reference configuration — this hook moves it along via the
        cached ``vtkOriginalPointIds`` rows.
        """
        if (
            self._handle is None
            or self._points is None
            or self._substrate_rows is None
        ):
            return
        try:
            target = (
                np.asarray(deformed_pts, dtype=np.float64)
                if deformed_pts is not None
                else np.asarray(scene.grid.points, dtype=np.float64)
            )
        except Exception:
            return
        rows = self._substrate_rows
        if rows.size == 0 or int(rows.max()) >= target.shape[0]:
            return
        self._points = PointSet(target[rows])
        # Same fast path as a step change: topology unchanged, the
        # backend swaps the bound dataset's points in place.
        self._push_scalar_update()

    def detach(self) -> None:
        # Drop the scalar bar before tearing the layer down.
        self._remove_scalar_bar(self._scalar_bar_title())
        self._teardown_lut()
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._handle = None
        self._layer = None
        self._points = None
        self._cells = None
        self._scalar_values = None
        self._submesh_pos_of_id = None
        self._fem_ids_to_read = None
        self._submesh_cell_pos_of_eid = None
        self._fem_eids_to_read = None
        self._discrete_cell_point_offsets = None
        self._substrate_rows = None
        self._initial_clim = None
        self._effective_topology = None
        super().detach()

    # ------------------------------------------------------------------
    # Visibility (backend-routed)
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    # ------------------------------------------------------------------
    # Runtime style adjustments (used by the settings tab)
    # ------------------------------------------------------------------

    # clim/cmap/LUT handling comes from ScalarColorSupport.

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        return self._scalar_values

    def set_opacity(self, opacity: float) -> None:
        self._runtime_opacity = float(opacity)
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_opacity(self._handle, float(opacity))

    # ------------------------------------------------------------------
    # Topology resolution
    # ------------------------------------------------------------------

    def _resolve_topology(self) -> str:
        """Validate and return the user-selected topology."""
        style: ContourStyle = self.spec.style    # type: ignore[assignment]
        requested = getattr(style, "topology", _TOPO_NODES) or _TOPO_NODES
        if requested == _TOPO_NODES:
            return _TOPO_NODES
        if requested == _TOPO_GAUSS:
            return _TOPO_GAUSS
        raise ValueError(
            f"ContourStyle.topology must be one of "
            f"{{'nodes', 'gauss'}}; got {requested!r}."
        )

    def _resolve_averaging(self) -> str:
        """Validate and return the user-selected averaging mode."""
        style: ContourStyle = self.spec.style    # type: ignore[assignment]
        requested = getattr(style, "averaging", _AVG_AVERAGED) or _AVG_AVERAGED
        if requested in (_AVG_AVERAGED, _AVG_DISCRETE):
            return requested
        raise ValueError(
            f"ContourStyle.averaging must be one of "
            f"{{'averaged', 'discrete'}}; got {requested!r}."
        )

    # ------------------------------------------------------------------
    # Attach — nodes path
    # ------------------------------------------------------------------

    def _attach_nodes(self, scene: "FEMSceneData") -> None:
        # ── Resolve selector to substrate point indices ─────────────
        node_ids = self._resolved_node_ids
        if node_ids is None:
            point_indices = np.arange(scene.grid.n_points, dtype=np.int64)
        else:
            point_indices = self._fem_ids_to_substrate_indices(scene, node_ids)
            if point_indices.size == 0:
                from ._base import NoDataError
                raise NoDataError(
                    f"ContourDiagram: selector resolved to {node_ids.size} "
                    f"node(s) but none of them are in the substrate mesh "
                    f"(selector={self.spec.selector!r})."
                )

        submesh = scene.grid.extract_points(
            point_indices, adjacent_cells=False,
        )
        if submesh.n_points == 0:
            from ._base import NoDataError
            raise NoDataError(
                "ContourDiagram: substrate submesh is empty for this "
                "selector — nothing to color."
            )

        orig_indices = np.asarray(
            submesh.point_data["vtkOriginalPointIds"], dtype=np.int64,
        )
        fem_ids_in_submesh = scene.node_ids[orig_indices]

        max_id = int(fem_ids_in_submesh.max()) + 1
        submesh_pos = np.full(max_id + 1, -1, dtype=np.int64)
        submesh_pos[fem_ids_in_submesh] = np.arange(
            fem_ids_in_submesh.size, dtype=np.int64,
        )

        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = fem_ids_in_submesh
        self._set_point_geometry(submesh)

        values_at_step_0 = self._fetch_step_values(0)
        if values_at_step_0 is None:
            from ._base import NoDataError
            raise NoDataError(
                f"ContourDiagram: no nodal data for component "
                f"{self.spec.selector.component!r} at step 0. Use "
                f"`results.inspect.diagnose("
                f"{self.spec.selector.component!r})` to see which "
                f"buckets were checked."
            )
        self._scatter_into_scalar(values_at_step_0[0], values_at_step_0[1])
        self._initial_clim = self._compute_initial_clim(self._scalar_values)
        self._finalize_layer()

    # ------------------------------------------------------------------
    # Attach — gauss (element-constant) path
    # ------------------------------------------------------------------

    def _attach_gauss(self, scene: "FEMSceneData") -> None:
        """Read step 0, decide cell-vs-node sub-path, dispatch."""
        from ._base import NoDataError
        from apeGmsh.results._gauss_extrapolation import (
            per_element_max_gp_count,
        )

        eids = self._resolved_element_ids
        results = self._scoped_results()
        if results is None:
            raise NoDataError(
                "ContourDiagram (gauss): could not scope Results to a "
                "stage — diagram needs a stage_id on the spec."
            )
        slab_step0 = results.elements.gauss.get(
            ids=eids,
            component=self.spec.selector.component,
            time=[0],
        )
        if slab_step0.values.size == 0:
            raise NoDataError(
                f"ContourDiagram (gauss): no element data for component "
                f"{self.spec.selector.component!r} at step 0."
            )
        max_n_gp = per_element_max_gp_count(slab_step0)
        averaging = self._resolve_averaging()
        if max_n_gp <= 1:
            if averaging == _AVG_DISCRETE:
                self._attach_gauss_cell(scene, slab_step0)
            else:
                self._attach_gauss_node(scene, slab_step0,
                                        _EFFECTIVE_GAUSS_CELL_AVERAGED)
        else:
            if averaging == _AVG_DISCRETE:
                self._attach_gauss_node_discrete(scene, slab_step0)
            else:
                self._attach_gauss_node(scene, slab_step0,
                                        _EFFECTIVE_GAUSS_NODE)

    # ------------------------------------------------------------------
    # Attach — gauss cell-data path (n_gp == 1)
    # ------------------------------------------------------------------

    def _attach_gauss_cell(
        self, scene: "FEMSceneData", slab_step0: Any,
    ) -> None:
        from ._base import NoDataError

        # ── Resolve selector to substrate cell indices ──────────────
        eids = self._resolved_element_ids
        if eids is None:
            cell_indices = np.arange(scene.grid.n_cells, dtype=np.int64)
        else:
            cell_indices = np.fromiter(
                (
                    scene.element_id_to_cell.get(int(e), -1)
                    for e in eids
                ),
                dtype=np.int64,
                count=len(eids),
            )
            cell_indices = cell_indices[cell_indices >= 0]
            if cell_indices.size == 0:
                raise NoDataError(
                    f"ContourDiagram (gauss): selector resolved to "
                    f"{eids.size} element(s) but none are in the substrate "
                    f"mesh (selector={self.spec.selector!r})."
                )

        submesh = scene.grid.extract_cells(cell_indices)
        if submesh.n_cells == 0:
            raise NoDataError(
                "ContourDiagram (gauss): substrate submesh has no cells "
                "for this selector — nothing to color."
            )

        try:
            orig_cells = np.asarray(
                submesh.cell_data["vtkOriginalCellIds"], dtype=np.int64,
            )
        except KeyError as exc:
            raise NoDataError(
                "ContourDiagram (gauss): extract_cells did not provide "
                "vtkOriginalCellIds — cannot map cells back to FEM "
                "element IDs."
            ) from exc
        fem_eids_in_submesh = scene.cell_to_element_id[orig_cells]

        # CellBlocks groups cells by VTK type, so the rebuilt grid's cell
        # order differs from the extracted submesh order for a mixed
        # tri+quad mesh. Reorder the per-cell metadata into grouped order
        # once so the per-cell ScalarField stays aligned with CellBlocks
        # (identity for the homogeneous common case).
        fem_eids_in_submesh = fem_eids_in_submesh[
            self._cellblocks_group_to_orig(submesh)
        ]

        max_eid = int(fem_eids_in_submesh.max()) + 1
        submesh_cell_pos = np.full(max_eid + 1, -1, dtype=np.int64)
        submesh_cell_pos[fem_eids_in_submesh] = np.arange(
            fem_eids_in_submesh.size, dtype=np.int64,
        )

        self._effective_topology = _EFFECTIVE_GAUSS_CELL
        self._submesh_cell_pos_of_eid = submesh_cell_pos
        self._fem_eids_to_read = fem_eids_in_submesh
        self._set_cell_geometry(submesh)

        # Step 0 values are already in slab_step0 — flatten and scatter
        # directly rather than re-reading.
        slab_eids = np.asarray(slab_step0.element_index, dtype=np.int64)
        slab_vals = np.asarray(slab_step0.values[0], dtype=np.float64)
        self._scatter_into_cell_scalar(slab_eids, slab_vals)
        self._initial_clim = self._compute_initial_clim(self._scalar_values)
        self._finalize_layer()

    # ------------------------------------------------------------------
    # Attach — gauss extrapolated point-data path (n_gp > 1)
    # ------------------------------------------------------------------

    def _attach_gauss_node(
        self,
        scene: "FEMSceneData",
        slab_step0: Any,
        effective_topology: str = _EFFECTIVE_GAUSS_NODE,
    ) -> None:
        """Extrapolate to nodes + average across elements; paint as point data."""
        from ._base import NoDataError
        from apeGmsh.results._gauss_extrapolation import (
            extrapolate_gauss_slab_to_nodes,
        )

        node_ids, nodal_values = extrapolate_gauss_slab_to_nodes(
            slab_step0, self._view,
        )
        if node_ids.size == 0:
            raise NoDataError(
                f"ContourDiagram (gauss-extrapolated): no nodal "
                f"contributions for component "
                f"{self.spec.selector.component!r} at step 0 (no "
                f"selected element matches the bound FEM)."
            )

        point_indices = self._fem_ids_to_substrate_indices(scene, node_ids)
        if point_indices.size == 0:
            raise NoDataError(
                f"ContourDiagram (gauss-extrapolated): "
                f"{node_ids.size} node(s) received contributions but "
                f"none are in the substrate mesh."
            )

        submesh = scene.grid.extract_points(
            point_indices, adjacent_cells=False,
        )
        if submesh.n_points == 0:
            raise NoDataError(
                "ContourDiagram (gauss-extrapolated): substrate submesh "
                "is empty — nothing to color."
            )

        orig_indices = np.asarray(
            submesh.point_data["vtkOriginalPointIds"], dtype=np.int64,
        )
        fem_ids_in_submesh = scene.node_ids[orig_indices]

        max_id = int(fem_ids_in_submesh.max()) + 1
        submesh_pos = np.full(max_id + 1, -1, dtype=np.int64)
        submesh_pos[fem_ids_in_submesh] = np.arange(
            fem_ids_in_submesh.size, dtype=np.int64,
        )

        self._effective_topology = effective_topology
        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = fem_ids_in_submesh
        self._set_point_geometry(submesh)

        self._scatter_into_scalar(node_ids, nodal_values[0])
        self._initial_clim = self._compute_initial_clim(self._scalar_values)
        self._finalize_layer()

    # ------------------------------------------------------------------
    # Attach — gauss discrete path (n_gp > 1, no cross-element averaging)
    # ------------------------------------------------------------------

    def _attach_gauss_node_discrete(
        self, scene: "FEMSceneData", slab_step0: Any,
    ) -> None:
        """Extrapolate per-element with no averaging; paint on shattered submesh.

        Each cell owns its own copy of its corner points
        (``UnstructuredGrid.separate_cells``), so neighbouring cells
        can carry different values at "the same" geometric corner —
        boundary jumps are visible. The corner ordering produced by
        ``build_fem_scene`` (``conn[:, :n_corner]``) matches the
        ordering used by the extrapolation primitive, so cell c's k-th
        point in the shattered submesh receives the k-th corner value
        from element ``cell_to_element_id[c]``.
        """
        from ._base import NoDataError
        from apeGmsh.results._gauss_extrapolation import (
            extrapolate_gauss_slab_per_element,
        )

        per_elem = extrapolate_gauss_slab_per_element(slab_step0, self._view)
        if per_elem.element_ids.size == 0:
            raise NoDataError(
                f"ContourDiagram (gauss-discrete): no element "
                f"contributions for component "
                f"{self.spec.selector.component!r} at step 0."
            )

        # Build {element_id: (n_corner,) values_at_step0} for fast lookup.
        per_elem_step0: dict[int, ndarray] = {}
        for eid, vals in zip(per_elem.element_ids, per_elem.values):
            per_elem_step0[int(eid)] = np.asarray(vals[0], dtype=np.float64)

        # Resolve selector to substrate cell indices (same as gauss_cell).
        eids = self._resolved_element_ids
        if eids is None:
            cell_indices = np.arange(scene.grid.n_cells, dtype=np.int64)
        else:
            cell_indices = np.fromiter(
                (
                    scene.element_id_to_cell.get(int(e), -1)
                    for e in eids
                ),
                dtype=np.int64,
                count=len(eids),
            )
            cell_indices = cell_indices[cell_indices >= 0]
            if cell_indices.size == 0:
                raise NoDataError(
                    f"ContourDiagram (gauss-discrete): selector "
                    f"resolved to {eids.size} element(s) but none are "
                    f"in the substrate mesh "
                    f"(selector={self.spec.selector!r})."
                )

        extracted = scene.grid.extract_cells(cell_indices)
        if extracted.n_cells == 0:
            raise NoDataError(
                "ContourDiagram (gauss-discrete): substrate submesh "
                "has no cells for this selector — nothing to color."
            )

        try:
            orig_cells = np.asarray(
                extracted.cell_data["vtkOriginalCellIds"], dtype=np.int64,
            )
        except KeyError as exc:
            raise NoDataError(
                "ContourDiagram (gauss-discrete): extract_cells did "
                "not provide vtkOriginalCellIds — cannot map cells "
                "back to FEM element IDs."
            ) from exc
        fem_eids_in_submesh = scene.cell_to_element_id[orig_cells]

        # Shatter: each cell now owns its own copies of its corner points.
        submesh = extracted.separate_cells()
        if submesh.n_points == 0:
            raise NoDataError(
                "ContourDiagram (gauss-discrete): shattered submesh "
                "is empty — nothing to color."
            )

        # Cache per-cell point offsets so per-step updates don't have
        # to walk the connectivity again.
        cells_arr = np.asarray(submesh.cells, dtype=np.int64)
        n_cells_sub = int(submesh.n_cells)
        cell_point_offsets = np.empty(n_cells_sub + 1, dtype=np.int64)
        cell_point_offsets[0] = 0
        i = 0
        for c in range(n_cells_sub):
            npe_c = int(cells_arr[i])
            cell_point_offsets[c + 1] = cell_point_offsets[c] + npe_c
            i += 1 + npe_c

        self._effective_topology = _EFFECTIVE_GAUSS_NODE_DISCRETE
        self._fem_eids_to_read = fem_eids_in_submesh
        self._discrete_cell_point_offsets = cell_point_offsets
        # separate_cells produces one cell type per original cell type; the
        # point order is per-cell-contiguous (the offsets above index it),
        # so point scalars stay aligned regardless of CellBlocks regrouping.
        self._set_point_geometry(submesh)

        self._scatter_per_element_into_shattered(per_elem_step0)
        self._initial_clim = self._compute_initial_clim(self._scalar_values)
        self._finalize_layer()

    def _scatter_per_element_into_shattered(
        self, per_elem_step: dict,
    ) -> None:
        """Write per-element corner values into the shattered submesh array."""
        if (self._scalar_values is None
                or self._fem_eids_to_read is None
                or self._discrete_cell_point_offsets is None):
            return
        offsets = self._discrete_cell_point_offsets
        eids = self._fem_eids_to_read
        for c, eid in enumerate(eids):
            vals = per_elem_step.get(int(eid))
            if vals is None:
                continue
            lo = int(offsets[c])
            hi = int(offsets[c + 1])
            n = hi - lo
            if vals.size < n:
                continue
            self._scalar_values[lo:hi] = vals[:n]

    def _refresh_shattered_at_step(self, step_index: int) -> None:
        """Re-read the slab at ``step_index`` and rescatter into the
        shattered submesh."""
        if self._fem_eids_to_read is None:
            return
        results = self._scoped_results()
        if results is None:
            return
        from apeGmsh.results._gauss_extrapolation import (
            extrapolate_gauss_slab_per_element,
        )
        eids = self._resolved_element_ids
        slab = results.elements.gauss.get(
            ids=eids,
            component=self.spec.selector.component,
            time=[int(step_index)],
        )
        if slab.values.size == 0:
            return
        per_elem = extrapolate_gauss_slab_per_element(slab, self._view)
        if per_elem.element_ids.size == 0:
            return
        per_elem_step: dict[int, ndarray] = {}
        for eid, vals in zip(per_elem.element_ids, per_elem.values):
            per_elem_step[int(eid)] = np.asarray(vals[0], dtype=np.float64)
        self._scatter_per_element_into_shattered(per_elem_step)

    # ------------------------------------------------------------------
    # Layer build / emit
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"contour_{id(self):x}"

    def _color_array_name(self) -> str:
        return self.spec.selector.component or "_contour"

    def _effective_opacity(self) -> float:
        style: ContourStyle = self.spec.style    # type: ignore[assignment]
        return (
            self._runtime_opacity
            if self._runtime_opacity is not None else style.opacity
        )

    def _set_point_geometry(self, submesh: Any) -> None:
        from ..backends.pyvista_qt import cellblocks_from_grid
        self._points = PointSet(np.asarray(submesh.points))
        self._cells = cellblocks_from_grid(submesh)
        self._scalar_values = np.zeros(submesh.n_points, dtype=np.float64)
        self._scalar_location = "point"
        self._substrate_rows = self._opid_rows(submesh)

    def _set_cell_geometry(self, submesh: Any) -> None:
        from ..backends.pyvista_qt import cellblocks_from_grid
        self._points = PointSet(np.asarray(submesh.points))
        self._cells = cellblocks_from_grid(submesh)
        self._scalar_values = np.zeros(submesh.n_cells, dtype=np.float64)
        self._scalar_location = "cell"
        self._substrate_rows = self._opid_rows(submesh)

    @staticmethod
    def _opid_rows(submesh: Any) -> "Optional[ndarray]":
        """Per-submesh-point substrate row from ``vtkOriginalPointIds``.

        ``extract_points`` / ``extract_cells`` produce the map directly;
        ``separate_cells`` (the shattered discrete path) inherits it as
        carried point data from the extracted input. ``None`` (no map)
        leaves the diagram pinned to the reference configuration.
        """
        try:
            opid = submesh.point_data["vtkOriginalPointIds"]
        except KeyError:
            return None
        return np.asarray(opid, dtype=np.int64)

    @staticmethod
    def _cellblocks_group_to_orig(submesh: Any) -> ndarray:
        """Permutation from extracted-cell order → ``cellblocks_from_grid``
        grouped order (cells grouped by VTK type). Identity when the
        submesh is a single cell type."""
        if submesh.n_cells == 0:
            return np.empty(0, dtype=np.int64)
        celltypes = np.asarray(submesh.celltypes, dtype=np.int64)
        return np.concatenate(
            [np.where(celltypes == t)[0] for t in submesh.cells_dict]
        ).astype(np.int64)

    def _build_layer(self) -> MeshLayer:
        style: ContourStyle = self.spec.style    # type: ignore[assignment]
        assert (
            self._points is not None
            and self._cells is not None
            and self._scalar_values is not None
        )
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        cmap = self._runtime_cmap or style.cmap
        name = self._color_array_name()
        color = ColorSpec(
            mode="by_array",
            array_name=name,
            lut=LutSpec(name=cmap, vmin=float(clim[0]), vmax=float(clim[1])),
        )
        return MeshLayer(
            layer_id=self._layer_id(),
            points=self._points,
            cells=self._cells,
            fields=(ScalarField(name, self._scalar_values, self._scalar_location),),
            color=color,
            opacity=self._effective_opacity(),
            show_edges=style.show_edges,
            pickable=False,
        )

    def _finalize_layer(self) -> None:
        self._layer = self._build_layer()
        self._handle = self._backend.add_layer(self._layer)

    def _push_scalar_update(self) -> None:
        if self._handle is None:
            return
        self._layer = self._build_layer()
        # Topology unchanged across steps → the backend's in-place mesh
        # fast path recolours the bound dataset without re-adding the actor.
        self._backend.update_layer(self._handle, self._layer)

    # ------------------------------------------------------------------
    # Scatter helpers (write the persistent buffer in place)
    # ------------------------------------------------------------------

    def _scatter_into_scalar(
        self, slab_node_ids: ndarray, slab_values: ndarray,
    ) -> None:
        if self._submesh_pos_of_id is None or self._scalar_values is None:
            return
        positions = self._submesh_pos_of_id[slab_node_ids]
        valid = positions >= 0
        self._scalar_values[positions[valid]] = slab_values[valid]

    def _scatter_into_cell_scalar(
        self, slab_eids: ndarray, slab_values: ndarray,
    ) -> None:
        if (self._submesh_cell_pos_of_eid is None
                or self._scalar_values is None):
            return
        max_known = self._submesh_cell_pos_of_eid.size - 1
        in_range = (slab_eids >= 0) & (slab_eids <= max_known)
        positions = np.full_like(slab_eids, -1)
        positions[in_range] = self._submesh_cell_pos_of_eid[
            slab_eids[in_range]
        ]
        valid = positions >= 0
        self._scalar_values[positions[valid]] = slab_values[valid]

    # ------------------------------------------------------------------
    # Step reads
    # ------------------------------------------------------------------

    def _compute_initial_clim(
        self, data: ndarray,
    ) -> tuple[float, float]:
        style: ContourStyle = self.spec.style    # type: ignore[assignment]
        if style.clim is not None:
            return (float(style.clim[0]), float(style.clim[1]))
        finite = data[np.isfinite(data)]
        if finite.size:
            lo = float(finite.min())
            hi = float(finite.max())
            if lo == hi:
                hi = lo + 1.0
            return (lo, hi)
        return (0.0, 1.0)

    def _fetch_step_values(
        self, step_index: int,
    ) -> Optional[tuple[ndarray, ndarray]]:
        """Read one step's slab. Returns ``(node_ids, values)`` or None."""
        if self._fem_ids_to_read is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        ids = self._fem_ids_to_read
        slab = results.nodes.get(
            ids=ids,
            component=self.spec.selector.component,
            time=[int(step_index)],
        )
        if slab.values.size == 0:
            return None
        return (np.asarray(slab.node_ids, dtype=np.int64),
                np.asarray(slab.values[0], dtype=np.float64))

    def _fetch_step_values_gauss(
        self, step_index: int,
    ) -> Optional[tuple[ndarray, ndarray]]:
        """Read one step's cell-constant Gauss slab (n_gp == 1)."""
        if self._fem_eids_to_read is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        slab = results.elements.gauss.get(
            ids=self._fem_eids_to_read,
            component=self.spec.selector.component,
            time=[int(step_index)],
        )
        if slab.values.size == 0:
            return None
        slab_eids = np.asarray(slab.element_index, dtype=np.int64)
        slab_vals = np.asarray(slab.values[0], dtype=np.float64)
        # Defensive: collapse to per-element mean if multi-GP slipped
        # through (shouldn't happen — gauss_cell is a 1-GP path).
        if slab_eids.size != np.unique(slab_eids).size:
            uniq, inv = np.unique(slab_eids, return_inverse=True)
            sums = np.zeros(uniq.size, dtype=np.float64)
            counts = np.zeros(uniq.size, dtype=np.int64)
            np.add.at(sums, inv, slab_vals)
            np.add.at(counts, inv, 1)
            return (uniq, sums / counts)
        return (slab_eids, slab_vals)

    def _fetch_step_values_gauss_node(
        self, step_index: int,
    ) -> Optional[tuple[ndarray, ndarray]]:
        """Read one step's GP slab and extrapolate to nodal values."""
        if self._fem_eids_to_read is None and self._fem_ids_to_read is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        from apeGmsh.results._gauss_extrapolation import (
            extrapolate_gauss_slab_to_nodes,
        )
        # The slab fetch is keyed by element IDs (via the selector),
        # not the node IDs we cached for the submesh.
        eids = self._resolved_element_ids
        slab = results.elements.gauss.get(
            ids=eids,
            component=self.spec.selector.component,
            time=[int(step_index)],
        )
        if slab.values.size == 0:
            return None
        node_ids, nodal = extrapolate_gauss_slab_to_nodes(slab, self._view)
        if node_ids.size == 0:
            return None
        return (node_ids, nodal[0])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _fem_ids_to_substrate_indices(
        scene: "FEMSceneData", fem_ids: ndarray,
    ) -> ndarray:
        """Map a FEM-id array to substrate point indices, dropping misses."""
        max_id = max(int(fem_ids.max()), int(scene.node_ids.max())) + 1
        lookup = np.full(max_id + 1, -1, dtype=np.int64)
        lookup[scene.node_ids] = np.arange(scene.node_ids.size, dtype=np.int64)
        idx = lookup[fem_ids]
        return idx[idx >= 0]
