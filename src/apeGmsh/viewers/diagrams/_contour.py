"""ContourDiagram — paint scalar values on the substrate.

Two topology paths share one diagram:

* **nodes** — values come from ``results.nodes.get(...)`` and are
  painted as point data on a submesh extracted by node IDs.
* **gauss** — values come from ``results.elements.gauss.get(...)``
  and are painted as cell data on a submesh extracted by element IDs.
  Element-constant only (``n_gp == 1`` per element); higher-order
  integration would need shape-function interpolation, which is a
  future extension.

The active path is chosen by ``ContourStyle.topology`` (``"auto"``,
``"nodes"``, ``"gauss"``). Auto inspects the available components on
each composite at attach time and prefers nodal data when both have
the component.

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
from ._styles import ContourStyle

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


_SCALAR_NAME = "_contour"

_TOPO_NODES = "nodes"
_TOPO_GAUSS = "gauss"
_TOPO_AUTO = "auto"


class ContourDiagram(Diagram):
    """Scalar contour painted on a slice of the substrate mesh.

    The selector picks which nodes / elements carry data; everything
    else stays as the gray substrate. Multiple contour diagrams compose
    naturally — each owns its own submesh actor.

    The class-level ``topology = "nodes"`` declaration informs the Add-
    Diagram dialog which composite to enumerate components from. Per-
    instance Gauss contour is opted into via
    ``ContourStyle.topology = "gauss"``; ``"auto"`` picks based on
    which composite has the requested component.
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

        # Mutable runtime overrides (style is frozen)
        self._runtime_clim: Optional[tuple[float, float]] = None
        self._runtime_opacity: Optional[float] = None
        self._runtime_cmap: Optional[str] = None

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
        self._effective_topology = topology
        if topology == _TOPO_GAUSS:
            self._attach_gauss(plotter, scene)
        else:
            self._attach_nodes(plotter, scene)

    def update_to_step(self, step_index: int) -> None:
        if self._submesh is None:
            return
        if self._effective_topology == _TOPO_GAUSS:
            if self._cell_scalar_array is None:
                return
            fetched = self._fetch_step_values_gauss(step_index)
            if fetched is None:
                return
            slab_eids, slab_values = fetched
            self._scatter_into_cell_scalar(slab_eids, slab_values)
        else:
            if self._scalar_array is None:
                return
            fetched = self._fetch_step_values(step_index)
            if fetched is None:
                return
            slab_node_ids, slab_values = fetched
            self._scatter_into_scalar(slab_node_ids, slab_values)

    def detach(self) -> None:
        self._submesh = None
        self._actor = None
        self._scalar_array = None
        self._submesh_pos_of_id = None
        self._fem_ids_to_read = None
        self._cell_scalar_array = None
        self._submesh_cell_pos_of_eid = None
        self._fem_eids_to_read = None
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
            if self._effective_topology == _TOPO_GAUSS
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
        """Pick the effective topology based on style + availability.

        ``"nodes"`` and ``"gauss"`` are returned verbatim. ``"auto"``
        prefers nodal data when both composites carry the requested
        component; falls back to gauss; finally defaults to nodes
        (the existing path) so the caller hits the original NoDataError
        with its diagnose hint when neither has the component.
        """
        style: ContourStyle = self.spec.style    # type: ignore[assignment]
        requested = getattr(style, "topology", _TOPO_AUTO) or _TOPO_AUTO
        if requested == _TOPO_NODES:
            return _TOPO_NODES
        if requested == _TOPO_GAUSS:
            return _TOPO_GAUSS
        if requested != _TOPO_AUTO:
            raise ValueError(
                f"ContourStyle.topology must be one of "
                f"{{'auto', 'nodes', 'gauss'}}; got {requested!r}."
            )
        comp = self.spec.selector.component
        results = self._scoped_results()
        if results is None:
            return _TOPO_NODES
        try:
            if comp in results.nodes.available_components():
                return _TOPO_NODES
        except Exception:
            pass
        try:
            if comp in results.elements.gauss.available_components():
                return _TOPO_GAUSS
        except Exception:
            pass
        return _TOPO_NODES

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

        # vtkOriginalCellIds maps submesh cell index -> substrate cell
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

        # FEM element ID -> submesh cell row
        max_eid = int(fem_eids_in_submesh.max()) + 1
        submesh_cell_pos = np.full(max_eid + 1, -1, dtype=np.int64)
        submesh_cell_pos[fem_eids_in_submesh] = np.arange(
            fem_eids_in_submesh.size, dtype=np.int64,
        )

        self._submesh = submesh
        self._submesh_cell_pos_of_eid = submesh_cell_pos
        self._fem_eids_to_read = fem_eids_in_submesh

        cell_scalars = np.zeros(submesh.n_cells, dtype=np.float64)
        submesh.cell_data[_SCALAR_NAME] = cell_scalars
        self._cell_scalar_array = submesh.cell_data[_SCALAR_NAME]

        values_at_step_0 = self._fetch_step_values_gauss(0)
        if values_at_step_0 is None:
            raise NoDataError(
                f"ContourDiagram (gauss): no element data for component "
                f"{self.spec.selector.component!r} at step 0."
            )
        self._scatter_into_cell_scalar(
            values_at_step_0[0], values_at_step_0[1],
        )
        self._initial_clim = self._compute_initial_clim(
            np.asarray(self._cell_scalar_array),
        )
        self._add_actor(submesh, _SCALAR_NAME, preference="cell")

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

        actor = self._plotter.add_mesh(
            submesh,
            scalars=scalar_name,
            preference=preference,
            cmap=cmap,
            clim=clim,
            opacity=opacity,
            show_edges=style.show_edges,
            show_scalar_bar=style.show_scalar_bar,
            scalar_bar_args={
                "title": self.spec.selector.component,
            } if style.show_scalar_bar else None,
            name=self._actor_name(),
            reset_camera=False,
            lighting=True,
            smooth_shading=False,
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
        """Read one step's element-constant Gauss slab.

        Returns ``(element_ids, values)`` or ``None``. Raises
        ``NoDataError`` if the slab carries more than one Gauss point
        per element — Gauss-contour with higher-order integration
        needs shape-function interpolation, which this diagram does
        not yet implement.
        """
        if self._fem_eids_to_read is None:
            return None
        results = self._scoped_results()
        if results is None:
            return None
        eids = self._fem_eids_to_read
        slab = results.elements.gauss.get(
            ids=eids,
            component=self.spec.selector.component,
            time=[int(step_index)],
        )
        if slab.values.size == 0:
            return None
        slab_eids = np.asarray(slab.element_index, dtype=np.int64)
        # Reject n_gp > 1 with a clear hint.
        if slab_eids.size != np.unique(slab_eids).size:
            from ._base import NoDataError
            counts = np.bincount(slab_eids - int(slab_eids.min()))
            n_gp = int(counts[counts > 0].max())
            raise NoDataError(
                f"ContourDiagram (gauss): component "
                f"{self.spec.selector.component!r} has {n_gp} Gauss "
                f"points per element. Element-constant rendering "
                f"requires n_gp == 1 (e.g. CST / tri31). For higher-"
                f"order integration, use a diagram that interpolates "
                f"through the shape functions."
            )
        # values shape (1, E)
        return (slab_eids,
                np.asarray(slab.values[0], dtype=np.float64))

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
