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

The dispatch in ``_resolve_path`` and ``_attach_gauss`` peeks at the
slab at step 0 to count GPs per element and picks the right gauss
sub-path.

Performance contract (locked in Phase 0, validated here):

* Selector resolves to FEM IDs **once at attach**. The submesh is
  extracted from the substrate grid once; per-step reads only refresh
  the relevant scalar array in place.
* Per-step update is one h5py read for the selected IDs at the active
  step, one numpy scatter into the submesh array, one ``Modified()``
  mark. No actor re-creation.
* The mapper id seen by ``actor.GetMapper()`` is stable across step
  changes — the in-place mutation test asserts this.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._base import Diagram, DiagramSpec
from ._scalar_bar_support import ScalarBarSupport
from ._styles import ContourStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


_SCALAR_NAME = "_contour"

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


class ContourDiagram(ScalarBarSupport, Diagram):
    """Scalar contour painted on a slice of the substrate mesh.

    The selector picks which nodes / elements carry data; everything
    else stays as the gray substrate. Multiple contour diagrams compose
    naturally — each owns its own submesh actor.

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

        # Runtime state populated by attach()
        self._submesh: Any = None              # pv.UnstructuredGrid slice
        self._actor: Any = None
        self._initial_clim: Optional[tuple[float, float]] = None

        # Effective topology after resolution at attach (one of
        # _TOPO_NODES / _TOPO_GAUSS).
        self._effective_topology: Optional[str] = None

        # Nodes-path runtime state
        self._scalar_array: Optional[ndarray] = None
        self._submesh_pos_of_id: Optional[ndarray] = None
        self._fem_ids_to_read: Optional[ndarray] = None

        # Gauss-path runtime state
        self._cell_scalar_array: Optional[ndarray] = None
        self._submesh_cell_pos_of_eid: Optional[ndarray] = None
        self._fem_eids_to_read: Optional[ndarray] = None

        # Discrete shattered-submesh runtime state (n_gp>1 + discrete)
        self._discrete_cell_point_offsets: Optional[ndarray] = None

        # Mutable runtime overrides (style is frozen)
        self._runtime_clim: Optional[tuple[float, float]] = None
        self._runtime_opacity: Optional[float] = None
        self._runtime_cmap: Optional[str] = None
        self._init_scalar_bar_state()

    # ------------------------------------------------------------------
    # Attach / detach / update
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        fem: "FEMData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "ContourDiagram.attach requires a FEMSceneData (the "
                "viewer's substrate mesh). The Director must call "
                "bind_plotter(plotter, scene=scene)."
            )
        super().attach(plotter, fem, scene)

        topology = self._resolve_topology()
        if topology == _TOPO_GAUSS:
            # Peek at step 0 to choose cell-vs-node sub-path.
            self._attach_gauss(plotter, scene)
        else:
            self._effective_topology = _EFFECTIVE_NODES
            self._attach_nodes(plotter, scene)

    def update_to_step(self, step_index: int) -> None:
        if self._submesh is None:
            return
        topo = self._effective_topology
        if topo == _EFFECTIVE_GAUSS_CELL:
            if self._cell_scalar_array is None:
                return
            fetched = self._fetch_step_values_gauss(step_index)
            if fetched is None:
                return
            slab_eids, slab_values = fetched
            self._scatter_into_cell_scalar(slab_eids, slab_values)
        elif topo in (_EFFECTIVE_GAUSS_NODE, _EFFECTIVE_GAUSS_CELL_AVERAGED):
            if self._scalar_array is None:
                return
            fetched = self._fetch_step_values_gauss_node(step_index)
            if fetched is None:
                return
            node_ids, nodal_values = fetched
            self._scatter_into_scalar(node_ids, nodal_values)
        elif topo == _EFFECTIVE_GAUSS_NODE_DISCRETE:
            if self._scalar_array is None:
                return
            self._refresh_shattered_at_step(step_index)
        else:    # _EFFECTIVE_NODES
            if self._scalar_array is None:
                return
            fetched = self._fetch_step_values(step_index)
            if fetched is None:
                return
            slab_node_ids, slab_values = fetched
            self._scatter_into_scalar(slab_node_ids, slab_values)

    def detach(self) -> None:
        # Drop the scalar bar before tearing the actor down, otherwise
        # PyVista leaves the bar's actor and registry entry in the
        # plotter — every detach/re-attach cycle would otherwise leak
        # one bar onto the screen.
        self._remove_scalar_bar(self._scalar_bar_title())
        self._submesh = None
        self._actor = None
        self._scalar_array = None
        self._submesh_pos_of_id = None
        self._fem_ids_to_read = None
        self._cell_scalar_array = None
        self._submesh_cell_pos_of_eid = None
        self._fem_eids_to_read = None
        self._discrete_cell_point_offsets = None
        self._initial_clim = None
        self._effective_topology = None
        super().detach()

    # ------------------------------------------------------------------
    # Runtime style adjustments (used by the settings tab)
    # ------------------------------------------------------------------

    def set_clim(self, vmin: float, vmax: float) -> None:
        """Override the colormap range. Live update."""
        if vmin == vmax:
            vmax = vmin + 1.0
        self._runtime_clim = (float(vmin), float(vmax))
        self._apply_clim()

    def autofit_clim_at_current_step(self) -> Optional[tuple[float, float]]:
        """Re-fit clim to the current step's value range."""
        active = (
            self._cell_scalar_array
            if self._effective_topology == _EFFECTIVE_GAUSS_CELL
            else self._scalar_array
        )
        if active is None:
            return None
        data = np.asarray(active)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None
        lo = float(finite.min())
        hi = float(finite.max())
        if lo == hi:
            hi = lo + 1.0
        self.set_clim(lo, hi)
        return (lo, hi)

    def set_opacity(self, opacity: float) -> None:
        self._runtime_opacity = float(opacity)
        if self._actor is not None:
            try:
                self._actor.GetProperty().SetOpacity(float(opacity))
            except Exception:
                pass

    def set_cmap(self, cmap: str) -> None:
        """Switch the colormap. Mutates the lookup table on the active actor."""
        self._runtime_cmap = cmap
        if self._actor is None:
            return
        # PyVista's add_mesh creates a vtkScalarsToColors — the cleanest
        # way to swap cmap without re-adding the actor is to rebuild
        # the lookup table via PyVista's helper.
        try:
            import pyvista as pv
            lut = pv.LookupTable(cmap)
            clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
            lut.scalar_range = clim
            mapper = self._actor.GetMapper()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(*clim)
        except Exception:
            pass

    def current_clim(self) -> Optional[tuple[float, float]]:
        return self._runtime_clim or self._initial_clim

    # ------------------------------------------------------------------
    # Topology resolution
    # ------------------------------------------------------------------

    def _resolve_topology(self) -> str:
        """Validate and return the user-selected topology.

        Only ``"nodes"`` and ``"gauss"`` are accepted — there is no
        ``"auto"`` mode. The user picks the source explicitly.
        """
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

    def _attach_nodes(
        self, plotter: Any, scene: "FEMSceneData",
    ) -> None:
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

        self._submesh = submesh
        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = fem_ids_in_submesh

        scalars = np.zeros(submesh.n_points, dtype=np.float64)
        submesh.point_data[_SCALAR_NAME] = scalars
        self._scalar_array = submesh.point_data[_SCALAR_NAME]

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
        self._initial_clim = self._compute_initial_clim(
            np.asarray(self._scalar_array),
        )
        self._add_actor(submesh, _SCALAR_NAME, preference="point")

    # ------------------------------------------------------------------
    # Attach — gauss (element-constant) path
    # ------------------------------------------------------------------

    def _attach_gauss(
        self, plotter: Any, scene: "FEMSceneData",
    ) -> None:
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
                self._attach_gauss_cell(plotter, scene, slab_step0)
            else:
                self._attach_gauss_node(plotter, scene, slab_step0,
                                       _EFFECTIVE_GAUSS_CELL_AVERAGED)
        else:
            if averaging == _AVG_DISCRETE:
                self._attach_gauss_node_discrete(plotter, scene, slab_step0)
            else:
                self._attach_gauss_node(plotter, scene, slab_step0,
                                       _EFFECTIVE_GAUSS_NODE)

    # ------------------------------------------------------------------
    # Attach — gauss cell-data path (n_gp == 1)
    # ------------------------------------------------------------------

    def _attach_gauss_cell(
        self, plotter: Any, scene: "FEMSceneData", slab_step0: Any,
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

        max_eid = int(fem_eids_in_submesh.max()) + 1
        submesh_cell_pos = np.full(max_eid + 1, -1, dtype=np.int64)
        submesh_cell_pos[fem_eids_in_submesh] = np.arange(
            fem_eids_in_submesh.size, dtype=np.int64,
        )

        self._effective_topology = _EFFECTIVE_GAUSS_CELL
        self._submesh = submesh
        self._submesh_cell_pos_of_eid = submesh_cell_pos
        self._fem_eids_to_read = fem_eids_in_submesh

        cell_scalars = np.zeros(submesh.n_cells, dtype=np.float64)
        submesh.cell_data[_SCALAR_NAME] = cell_scalars
        self._cell_scalar_array = submesh.cell_data[_SCALAR_NAME]

        # Step 0 values are already in slab_step0 — flatten and scatter
        # directly rather than re-reading.
        slab_eids = np.asarray(slab_step0.element_index, dtype=np.int64)
        slab_vals = np.asarray(slab_step0.values[0], dtype=np.float64)
        self._scatter_into_cell_scalar(slab_eids, slab_vals)
        self._initial_clim = self._compute_initial_clim(
            np.asarray(self._cell_scalar_array),
        )
        self._add_actor(submesh, _SCALAR_NAME, preference="cell")

    # ------------------------------------------------------------------
    # Attach — gauss extrapolated point-data path (n_gp > 1)
    # ------------------------------------------------------------------

    def _attach_gauss_node(
        self,
        plotter: Any,
        scene: "FEMSceneData",
        slab_step0: Any,
        effective_topology: str = _EFFECTIVE_GAUSS_NODE,
    ) -> None:
        """Extrapolate to nodes + average across elements; paint as point data.

        Used by both the canonical ``gauss_node`` path (n_gp > 1) and
        the ``gauss_cell_averaged`` path (n_gp == 1, smoothed). The
        extrapolation primitive handles both cases — for n_gp == 1 it
        broadcasts the cell value to each corner, then averaging yields
        the mean of neighbouring cells at shared nodes.
        """
        from ._base import NoDataError
        from apeGmsh.results._gauss_extrapolation import (
            extrapolate_gauss_slab_to_nodes,
        )

        node_ids, nodal_values = extrapolate_gauss_slab_to_nodes(
            slab_step0, self._fem,
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
        self._submesh = submesh
        self._submesh_pos_of_id = submesh_pos
        self._fem_ids_to_read = fem_ids_in_submesh

        scalars = np.zeros(submesh.n_points, dtype=np.float64)
        submesh.point_data[_SCALAR_NAME] = scalars
        self._scalar_array = submesh.point_data[_SCALAR_NAME]

        self._scatter_into_scalar(node_ids, nodal_values[0])
        self._initial_clim = self._compute_initial_clim(
            np.asarray(self._scalar_array),
        )
        self._add_actor(submesh, _SCALAR_NAME, preference="point")

    # ------------------------------------------------------------------
    # Attach — gauss discrete path (n_gp > 1, no cross-element averaging)
    # ------------------------------------------------------------------

    def _attach_gauss_node_discrete(
        self, plotter: Any, scene: "FEMSceneData", slab_step0: Any,
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

        per_elem = extrapolate_gauss_slab_per_element(slab_step0, self._fem)
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
        self._submesh = submesh
        self._fem_eids_to_read = fem_eids_in_submesh
        self._discrete_cell_point_offsets = cell_point_offsets

        scalars = np.zeros(submesh.n_points, dtype=np.float64)
        submesh.point_data[_SCALAR_NAME] = scalars
        self._scalar_array = submesh.point_data[_SCALAR_NAME]

        self._scatter_per_element_into_shattered(per_elem_step0)
        self._initial_clim = self._compute_initial_clim(
            np.asarray(self._scalar_array),
        )
        self._add_actor(submesh, _SCALAR_NAME, preference="point")

    def _scatter_per_element_into_shattered(
        self, per_elem_step: dict,
    ) -> None:
        """Write per-element corner values into the shattered submesh array."""
        if (self._scalar_array is None
                or self._fem_eids_to_read is None
                or getattr(self, "_discrete_cell_point_offsets", None) is None):
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
            self._scalar_array[lo:hi] = vals[:n]
        try:
            self._submesh.Modified()
        except Exception:
            pass

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
        per_elem = extrapolate_gauss_slab_per_element(slab, self._fem)
        if per_elem.element_ids.size == 0:
            return
        per_elem_step: dict[int, ndarray] = {}
        for eid, vals in zip(per_elem.element_ids, per_elem.values):
            per_elem_step[int(eid)] = np.asarray(vals[0], dtype=np.float64)
        self._scatter_per_element_into_shattered(per_elem_step)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _actor_name(self) -> str:
        return f"diagram_contour_{id(self):x}"

    def _add_actor(
        self, submesh: Any, scalar_name: str, *, preference: str,
    ) -> None:
        """Common ``add_mesh`` call for both topology paths."""
        style: ContourStyle = self.spec.style    # type: ignore[assignment]
        cmap = self._runtime_cmap or style.cmap
        opacity = (
            self._runtime_opacity
            if self._runtime_opacity is not None else style.opacity
        )
        clim = self._runtime_clim or self._initial_clim
        bar_args = self._scalar_bar_args()

        actor = self._plotter.add_mesh(
            submesh,
            scalars=scalar_name,
            preference=preference,
            cmap=cmap,
            clim=clim,
            opacity=opacity,
            show_edges=style.show_edges,
            show_scalar_bar=bar_args is not None,
            scalar_bar_args=bar_args,
            name=self._actor_name(),
            reset_camera=False,
            lighting=True,
            smooth_shading=False,
            # Picks fall through to the substrate. Cell IDs on this
            # submesh would index a different topology than
            # ``scene.cell_to_element_id``, so resolving an element
            # off this actor would point to the wrong cell anyway.
            pickable=False,
        )
        self._actor = actor
        self._actors = [actor]

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
        # values shape (1, M)
        return (np.asarray(slab.node_ids, dtype=np.int64),
                np.asarray(slab.values[0], dtype=np.float64))

    def _scatter_into_scalar(
        self, slab_node_ids: ndarray, slab_values: ndarray,
    ) -> None:
        if self._submesh_pos_of_id is None or self._scalar_array is None:
            return
        positions = self._submesh_pos_of_id[slab_node_ids]
        valid = positions >= 0
        # In-place assignment — no array re-allocation.
        self._scalar_array[positions[valid]] = slab_values[valid]
        try:
            self._submesh.Modified()
        except Exception:
            pass

    def _fetch_step_values_gauss(
        self, step_index: int,
    ) -> Optional[tuple[ndarray, ndarray]]:
        """Read one step's cell-constant Gauss slab (n_gp == 1).

        Returns ``(element_ids, values)`` or ``None``. The dispatch in
        ``_attach_gauss`` should never route a multi-GP slab here; if
        n_gp > 1 we silently take the per-element mean as a defensive
        fallback rather than crash mid-animation.
        """
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
        """Read one step's GP slab and extrapolate to nodal values.

        Returns ``(node_ids, values_for_step)`` — values is a 1-D
        array of length ``node_ids.size``. The submesh's point order
        is preserved by ``_scatter_into_scalar`` via
        ``_submesh_pos_of_id``.
        """
        if self._fem_eids_to_read is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        from apeGmsh.results._gauss_extrapolation import (
            extrapolate_gauss_slab_to_nodes,
        )
        # The slab fetch is keyed by element IDs (via the selector),
        # not the node IDs we cached for the submesh. Re-derive from
        # the original element-id resolution.
        eids = self._resolved_element_ids
        slab = results.elements.gauss.get(
            ids=eids,
            component=self.spec.selector.component,
            time=[int(step_index)],
        )
        if slab.values.size == 0:
            return None
        node_ids, nodal = extrapolate_gauss_slab_to_nodes(slab, self._fem)
        if node_ids.size == 0:
            return None
        return (node_ids, nodal[0])

    def _scatter_into_cell_scalar(
        self, slab_eids: ndarray, slab_values: ndarray,
    ) -> None:
        if (self._submesh_cell_pos_of_eid is None
                or self._cell_scalar_array is None):
            return
        # An element ID outside the lookup range is a miss; clip into
        # range and then mask via the -1 sentinel.
        max_known = self._submesh_cell_pos_of_eid.size - 1
        in_range = (slab_eids >= 0) & (slab_eids <= max_known)
        positions = np.full_like(slab_eids, -1)
        positions[in_range] = self._submesh_cell_pos_of_eid[
            slab_eids[in_range]
        ]
        valid = positions >= 0
        self._cell_scalar_array[positions[valid]] = slab_values[valid]
        try:
            self._submesh.Modified()
        except Exception:
            pass

    def _apply_clim(self) -> None:
        if self._actor is None:
            return
        clim = self._runtime_clim or self._initial_clim
        if clim is None:
            return
        try:
            mapper = self._actor.GetMapper()
            mapper.SetScalarRange(*clim)
        except Exception:
            pass


    def _scoped_results(self) -> "Optional[Results]":
        """Return a Results scoped to the diagram's stage (or the spec's)."""
        if self.spec.stage_id is not None:
            return self._results.stage(self.spec.stage_id)
        # No stage on spec — let Results auto-resolve (works for
        # single-stage files; raises for multi-stage). The Director
        # is responsible for setting the spec.stage_id when adding.
        try:
            return self._results
        except Exception:
            return None

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
