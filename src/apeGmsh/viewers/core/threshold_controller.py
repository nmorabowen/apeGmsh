"""ThresholdController — scalar threshold as an ElementVisibility layer.

The catalog's first region tool (ADR 0084 D1, unblocked once the
cascade freeze closed). A threshold is **not a diagram kind**: it hides
cells whose scalar values fall outside ``[lo, hi]``, which is exactly
what :class:`~.element_visibility.ElementVisibility` already does for
the manual hide and the ``0/1/2/3/4`` dim filter. So it takes a layer
of its own — :data:`~.element_visibility.LAYER_THRESHOLD` — and rides
the existing mechanism instead of adding a second writer of
``vtkGhostType``:

* the ghost array is mutated in place, so there is no polydata rebuild
  and no re-extraction on a scrub tick;
* hidden cells become non-pickable for free (VTK skips HIDDENCELL and
  the box-pick path ANDs the mask);
* it COMPOSES — the effective hidden set is the OR of manual ∪ dim ∪
  stage ∪ threshold, so clearing the threshold cannot reveal a cell the
  user hid by hand.

Per geometry, not per viewer. Each geometry owns its own
:class:`FEMSceneData` (ADR 0058 S1/S2a) and therefore its own
``ElementVisibility``, so keying this controller's state by geometry id
makes the feature per-geometry by construction.

Two ratified behaviours
-----------------------
**Cell rule — ALL VALUES IN RANGE.** A cell survives only if *every*
value recorded on it is inside ``[lo, hi]`` (ParaView's "All Points"):
every NODE for a nodal component, every INTEGRATION POINT for a gauss
one. One value outside hides the cell, so a cell that straddles the
boundary is hidden, not clipped — nothing here cuts geometry. The same
rule on both topologies is deliberate: reducing a cell to one number
BEFORE the range is known (a mean, say) lets a cell whose every value
is outside the range still test inside, and the visible set stops
meaning "these values are in range" — von Mises gauss points reading
400/100/100/100 average to 175 and land inside a ``[150, 200]`` band
although not one of them does.

**LIVE — the mask follows the time step.** :meth:`refresh` recomputes
from the values at the *current* step, and ``PumpSet.pump_step`` calls
it on every STEP, so scrubbing moves the thresholded region. Step
resolution is pin-aware and combined-mode-aware — see :meth:`refresh`.

No VTK, no render backend, no Qt: the writes go through
``ElementVisibility``, the sanctioned ghost writer, and the values
arrive through an injected ``read_values`` callback (the same reason
``PumpSet`` takes ``read_deform_field`` as a callback — the reader owns
the viewer's visual-store cache and the bound ``Results`` handle). That
keeps the whole thing unit-testable headless.
:class:`SceneValueReader` is that callback's production implementation
— row alignment included, and headless-constructible for the same
reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Union

import numpy as np

from .element_visibility import LAYER_THRESHOLD

#: Component topologies a threshold understands. ``"nodes"`` values are
#: aligned to grid POINT rows and reduced to cells by the all-values
#: rule over the grid connectivity; ``"gauss"`` values are one per
#: INTEGRATION POINT and carry the grid CELL row each landed on
#: (:class:`GaussValues`), reduced by the same rule over those rows.
TOPOLOGY_NODES: str = "nodes"
TOPOLOGY_GAUSS: str = "gauss"


class GaussValues(NamedTuple):
    """A gauss read: one value per INTEGRATION POINT, plus its cell row.

    Deliberately NOT pre-reduced to one value per cell. "All gauss
    points in range" cannot be collapsed into a single per-cell number
    before ``lo`` / ``hi`` are known — which is exactly why the nodal
    path also reduces AFTER :func:`values_in_range`. So the reader hands
    the scatter index over and the reduction happens where the range
    lives, in :func:`compute_hidden_mask`.

    ``values`` and ``cell_rows`` are parallel: row ``i`` of ``values``
    was recorded on grid cell ``cell_rows[i]``. Several rows landing on
    the same cell is the normal case (one per integration point).
    """

    values: "np.ndarray"
    cell_rows: "np.ndarray"


@dataclass(frozen=True)
class ThresholdSettings:
    """One geometry's threshold state.

    ``lo > hi`` is an empty range — legal, and it hides everything
    rather than raising (a slider pair can cross mid-drag). ``lo == hi``
    keeps only cells whose every value — node or integration point, per
    ``topology`` — equals that value exactly.
    """

    component: str
    lo: float
    hi: float
    topology: str = TOPOLOGY_NODES


def values_in_range(
    values: "np.ndarray", lo: float, hi: float,
) -> "np.ndarray":
    """Boolean "inside ``[lo, hi]``" mask, NaN-safe.

    NaN compares False against both bounds, so a NaN value lands
    OUTSIDE the range and its cell is hidden. That is the deliberate
    reading: a node with no value is not evidence that the cell belongs
    in the kept set, and silently treating it as in-range would leave
    un-analysed cells sitting inside the thresholded region.
    """
    vals = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (vals >= float(lo)) & (vals <= float(hi))


def cells_with_all_nodes_in_range(
    grid: Any, point_in_range: "np.ndarray",
) -> "np.ndarray":
    """Reduce a per-POINT boolean mask to per-CELL by logical AND.

    The all-nodes rule, vectorised — these are FE meshes with 100k+
    cells, so a Python loop over cells is not an option.

    ``grid.cell_connectivity`` is the flat node-index stream and
    ``grid.offset`` is the ``n_cells + 1`` array of start positions into
    it (VTK 9's split representation, which pyvista surfaces directly).
    Gathering ``point_in_range[connectivity]`` gives one boolean per
    (cell, node) incidence; ``np.logical_and.reduceat`` at the offsets
    ANDs each cell's run in one pass. Mixed cell types need no special
    case — a tri contributes a run of 3 and a hex a run of 8, and the
    offsets already say so.

    ``reduceat`` reduces ``a[idx[i]:idx[i+1]]`` for each index and
    ``a[idx[-1]:]`` for the last, which is why the trailing offset is
    dropped: ``offset[:-1]`` has length ``n_cells`` and yields exactly
    ``n_cells`` results. (The degenerate ``idx[i] == idx[i+1]`` case
    that would make ``reduceat`` return an element instead of the
    identity cannot arise — every VTK cell has at least one point.)

    Returns a boolean array of length ``n_cells``: True = keep.
    """
    n_cells = int(grid.n_cells)
    if n_cells == 0:
        return np.zeros(0, dtype=bool)
    conn = np.asarray(grid.cell_connectivity)
    offsets = np.asarray(grid.offset)
    if conn.size == 0 or offsets.size != n_cells + 1:
        # Nothing sane to reduce (an empty or non-standard grid) — keep
        # every cell rather than blanking the viewport.
        return np.ones(n_cells, dtype=bool)
    per_incidence = np.asarray(point_in_range, dtype=bool)[conn]
    return np.logical_and.reduceat(per_incidence, offsets[:-1])


def cells_with_all_gauss_points_in_range(
    n_cells: int, cell_rows: "np.ndarray", gp_in_range: "np.ndarray",
) -> "np.ndarray":
    """Reduce a per-GAUSS-POINT boolean mask to per-CELL by logical AND.

    The gauss twin of :func:`cells_with_all_nodes_in_range`, and the
    same rule — but the incidences arrive as an explicit ``cell_rows``
    scatter index instead of through the grid's connectivity, because a
    slab's gauss rows are neither sorted nor contiguous by cell and
    ``reduceat`` would have no runs to reduce. Two ``bincount`` passes
    do it vectorised: ``counts`` is how many integration points reached
    each cell and ``in_counts`` how many of those were inside, so a cell
    is kept only when the two agree.

    A cell NO row reached has ``counts == 0`` and is hidden — an
    un-recorded cell is not evidence that it belongs in the kept set,
    the same reading :func:`values_in_range` gives a NaN.

    Returns a boolean array of length ``n_cells``: True = keep.
    """
    n = int(n_cells)
    rows = np.asarray(cell_rows, dtype=np.int64)
    counts = np.bincount(rows, minlength=n)
    in_counts = np.bincount(
        rows,
        weights=np.asarray(gp_in_range, dtype=bool).astype(np.float64),
        minlength=n,
    )
    return (counts > 0) & (in_counts == counts)


def compute_hidden_mask(
    grid: Any, values: Union["np.ndarray", GaussValues],
    settings: ThresholdSettings,
) -> "np.ndarray":
    """The per-cell hide mask for ``values`` under ``settings``.

    ``True`` = hide. For :data:`TOPOLOGY_NODES` ``values`` is per-point
    and reduces through :func:`cells_with_all_nodes_in_range`; for
    :data:`TOPOLOGY_GAUSS` it is a :class:`GaussValues` and reduces
    through :func:`cells_with_all_gauss_points_in_range`. Both apply
    :func:`values_in_range` FIRST and AND afterwards — the all-values
    rule cannot be evaluated on a value pre-reduced per cell.
    """
    if settings.topology == TOPOLOGY_GAUSS:
        keep = cells_with_all_gauss_points_in_range(
            int(grid.n_cells), values.cell_rows,
            values_in_range(values.values, settings.lo, settings.hi),
        )
    else:
        keep = cells_with_all_nodes_in_range(
            grid, values_in_range(values, settings.lo, settings.hi),
        )
    return ~np.asarray(keep, dtype=bool)


def _dense_row_map(ids: "np.ndarray", rows: "np.ndarray") -> "np.ndarray":
    """Dense ``fem id -> scene row`` lookup; ``-1`` = not in the scene.

    The same device ``results_viewer._read_deform_field`` builds for the
    substrate (``_deform_id_to_idx``): FEM ids are neither contiguous
    nor sorted in general, so a dense table turns the per-step scatter
    into one fancy-index instead of a dict walk.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    table = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    table[ids] = np.asarray(rows, dtype=np.int64)
    return table


class SceneValueReader:
    """``read_values`` over a bound ``Results`` + one ``FEMSceneData``.

    The risky half of the feature, and the reason it is a class here
    rather than a closure in ``results_viewer._show_impl``: **a slab's
    row order is not the scene's row order**, and getting that wrong
    paints a plausible-looking picture of the wrong cells rather than
    raising. Two rules, both mechanical:

    * **Topology is told, never guessed.** The same component name can
      exist under ``results.nodes`` AND ``results.elements.gauss``;
      there is no ``is_nodal`` flag to consult. The user picks, the
      setting carries it, and this reader honours it.
    * **Values are scattered by FEM id, never zipped by position.**
      Nodal rows land on grid POINT rows through ``scene.node_ids``;
      gauss rows land on grid CELL rows through
      ``scene.element_id_to_cell``, and the read hands back BOTH the
      per-integration-point values and those rows
      (:class:`GaussValues`) so the all-values reduction can run once
      the range is known. Same mechanism as
      ``results_viewer._read_deform_field`` and
      ``ContourDiagram._scatter_into_cell_scalar``, not a parallel one.

    Reads go through the director's :class:`VisualDataStore` slabs
    (``nodes_slab`` / ``gauss_slab``) so a scrub slices a cached row
    instead of re-reading HDF5, falling back to a per-step read when no
    store is bound or the store could not load that component. The
    id→row index is memoized per (stage, component, topology) against
    the slab it was built from, so the scatter costs one fancy-index
    per tick.

    Failures are **loud**: a stage that will not scope, a step outside
    the slab, a malformed read — all propagate to
    :meth:`ThresholdController.refresh`, which reports them through
    ``on_failure`` and takes the layer down (ADR 0084 D4). Only a
    component genuinely absent from the stage returns ``None``.
    """

    __slots__ = (
        "director", "scene", "_node_row", "_cell_row", "_n_points",
        "_maps",
    )

    def __init__(self, *, director: Any, scene: Any) -> None:
        self.director = director
        self.scene = scene
        node_ids = np.asarray(scene.node_ids, dtype=np.int64)
        self._n_points = int(node_ids.size)
        self._node_row = _dense_row_map(
            node_ids, np.arange(node_ids.size, dtype=np.int64),
        )
        id_to_cell = scene.element_id_to_cell or {}
        self._cell_row = _dense_row_map(
            np.fromiter(id_to_cell.keys(), dtype=np.int64, count=len(id_to_cell)),
            np.fromiter(id_to_cell.values(), dtype=np.int64, count=len(id_to_cell)),
        )
        self._maps: dict = {}

    def __call__(
        self, component: str, step: int,
        *, stage_id: Optional[str] = None,
        topology: str = TOPOLOGY_NODES,
    ) -> Optional[Union["np.ndarray", GaussValues]]:
        director = self.director
        # ``stage_id=None`` is an UNPINNED geometry: the pump already
        # translated the cursor onto the active stage, so read that one.
        sid = stage_id if stage_id is not None else director.stage_id
        if not component or sid is None:
            return None
        results = director.results.stage(sid)
        gauss = topology == TOPOLOGY_GAUSS
        got = self._slab_row(results, sid, str(component), gauss, int(step))
        if got is None:
            return None
        ids, values = got
        if gauss:
            cols, rows = self._index(
                (sid, component, "gauss"), ids, self._cell_row,
            )
            return GaussValues(values[cols], rows)
        cols, rows = self._index(
            (sid, component, "nodes"), ids, self._node_row,
        )
        out = np.full(self._n_points, np.nan, dtype=np.float64)
        out[rows] = values[cols]
        return out

    def _slab_row(
        self, results: Any, sid: str, component: str, gauss: bool, step: int,
    ) -> Optional[tuple]:
        """``(ids, values_at_step)`` for one component, or ``None``.

        ``ids`` is the slab's own location index (``node_ids`` /
        ``element_index``) — deliberately NOT re-sorted: the caller
        scatters by id, so slab order is irrelevant and re-ordering
        here would only invite a zip-by-position bug.
        """
        store = getattr(self.director, "visual_store", None)
        full = None
        if store is not None:
            full = (
                store.gauss_slab(results, sid, component) if gauss
                else store.nodes_slab(results, sid, component)
            )
        if full is not None:
            return (
                full.element_index if gauss else full.node_ids,
                np.asarray(full.values[step], dtype=np.float64),
            )
        if gauss:
            slab = results.elements.gauss.get(
                component=component, time=[step],
            )
        else:
            slab = results.nodes.get(
                ids=self.scene.node_ids, component=component, time=[step],
            )
        if np.asarray(slab.values).size == 0:
            return None
        return (
            slab.element_index if gauss else slab.node_ids,
            np.asarray(slab.values[0], dtype=np.float64),
        )

    def _index(
        self, key: tuple, ids: Any, table: "np.ndarray",
    ) -> tuple:
        """Memoized ``(slab columns, scene rows)`` for one slab's ids.

        Keyed by (stage, component, topology) but re-derived whenever
        the ids ARRAY changes identity, so a cached index can never
        outlive the slab it describes.
        """
        cached = self._maps.get(key)
        if cached is not None and cached[0] is ids:
            return cached[1], cached[2]
        arr = np.asarray(ids, dtype=np.int64)
        rows = np.full(arr.shape, -1, dtype=np.int64)
        known = (arr >= 0) & (arr < table.size)
        rows[known] = table[arr[known]]
        cols = np.flatnonzero(rows >= 0)
        self._maps[key] = (ids, cols, rows[cols])
        return cols, rows[cols]


class ThresholdController:
    """Owns the per-geometry threshold state and applies it.

    Parameters
    ----------
    read_values
        ``fn(component, step, *, stage_id=None, topology="nodes")``
        returning the scalar values for that (stage, step) — aligned to
        grid POINT rows for ``"nodes"``, a :class:`GaussValues` of
        per-integration-point values plus their grid CELL rows for
        ``"gauss"`` — or ``None`` when the component is not recorded in
        that stage. A callback for the same reason
        ``PumpSet.read_deform_field`` is one: the reader owns the
        viewer's visual-store column cache and the bound ``Results``.
    on_failure
        ``fn(action, exc, **payload)`` — ADR 0084 D4, the loud-failure
        sink. A component that vanished from the current stage, or a
        slab read that raised, must not kill the scrub, but it must not
        be swallowed either (the shell reader's own bare ``except`` is
        what hid the combined-mode DEFORM bug). Defaults to
        ``_pump_set._pump_failed``, the same registry the strict pytest
        fixture collects from.
    """

    __slots__ = ("read_values", "on_failure", "_by_geometry", "_applied")

    def __init__(
        self,
        *,
        read_values: Callable[..., Any],
        on_failure: Optional[Callable[..., None]] = None,
    ) -> None:
        if on_failure is None:
            from .._pump_set import _pump_failed
            on_failure = _pump_failed
        self.read_values = read_values
        self.on_failure = on_failure
        self._by_geometry: dict[str, ThresholdSettings] = {}
        # Geometry ids whose scene currently CARRIES the layer. Tracked
        # so that turning the last threshold off still gets one more
        # refresh pass to take the layer back down — see
        # :meth:`needs_refresh`.
        self._applied: set[str] = set()

    # ------------------------------------------------------------------
    # State (the programmatic API — usable from a script, no UI)
    # ------------------------------------------------------------------

    def set_threshold(
        self,
        geometry_id: str,
        *,
        component: str,
        lo: float,
        hi: float,
        topology: str = TOPOLOGY_NODES,
    ) -> None:
        """Enable (or re-aim) the threshold on one geometry.

        Records intent only — the mask lands on the next
        :meth:`refresh`, which the STEP pump runs. Callers outside a
        pump tick should fire ``STEP_CHANGED`` (or call
        :meth:`refresh`) to see it immediately.
        """
        self._by_geometry[str(geometry_id)] = ThresholdSettings(
            component=str(component),
            lo=float(lo),
            hi=float(hi),
            topology=str(topology),
        )

    def clear_threshold(self, geometry_id: str) -> None:
        """Disable the threshold on one geometry.

        Records intent only; the next :meth:`refresh` takes the layer
        down (and :meth:`needs_refresh` stays True until it has), so the
        hidden cells come back on the following STEP.
        """
        self._by_geometry.pop(str(geometry_id), None)

    def settings_for(
        self, geometry_id: str,
    ) -> Optional[ThresholdSettings]:
        """The geometry's threshold, or ``None`` when disabled."""
        return self._by_geometry.get(str(geometry_id))

    def needs_refresh(self) -> bool:
        """The STEP pump's zero-cost gate.

        True while any threshold is configured OR any layer is still
        applied. The second half matters: after the last
        :meth:`clear_threshold` the configured set is empty but a layer
        is still hiding cells, so the pump must run once more to take
        it down. With no threshold ever set this is False and the STEP
        path skips the refresh loop entirely.
        """
        return bool(self._by_geometry) or bool(self._applied)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def refresh(
        self, geometry: Any, scene: Any, step: int,
        *, stage_id: Optional[str] = None,
    ) -> None:
        """Recompute and apply ``geometry``'s threshold layer on ``scene``.

        Called once per visible geometry per STEP. Disabled, or a scene
        with no ``element_visibility``, is a no-op; a disabled geometry
        whose layer is still on the scene gets it cleared, so turning
        the threshold off restores exactly the cells it hid and nothing
        else.

        ``step`` must already be the LOCAL step for the stage being read
        — the caller resolves it (pinned geometry →
        ``director.local_step_for_stage(pin)``, unpinned →
        ``director.local_step_for_active_stage()``). Passing the raw
        global cursor in combined mode is the defect ADR 0084 fixed
        twice, for STEP and then for DEFORM.

        A read that raises, or a component missing from this stage, is
        reported through :attr:`on_failure` and leaves the layer
        CLEARED rather than stale — a value-visible outcome (every cell
        shown) beats a frozen mask that silently claims to describe the
        current step.
        """
        ev = getattr(scene, "element_visibility", None)
        if ev is None:
            return
        gid = str(getattr(geometry, "id", geometry))
        settings = self._by_geometry.get(gid)
        if settings is None:
            self._drop(ev, gid)
            return
        try:
            values = self.read_values(
                settings.component, int(step),
                stage_id=stage_id, topology=settings.topology,
            )
        except Exception as exc:
            self.on_failure(
                "threshold", exc,
                geometry=getattr(geometry, "id", None),
                component=settings.component,
                step=int(step),
            )
            self._drop(ev, gid)
            return
        if values is None:
            # Not recorded in this stage. Loud (D4) but not fatal, and
            # the layer goes rather than lying about the current step.
            self.on_failure(
                "threshold",
                KeyError(
                    f"component {settings.component!r} not available "
                    f"at step {int(step)}"
                    + (f" of stage {stage_id!r}" if stage_id else ""),
                ),
                geometry=getattr(geometry, "id", None),
                component=settings.component,
                step=int(step),
            )
            self._drop(ev, gid)
            return
        ev.set_layer(
            LAYER_THRESHOLD, compute_hidden_mask(
                scene.grid, values, settings,
            ),
        )
        self._applied.add(gid)

    def _drop(self, ev: Any, geometry_id: str) -> None:
        """Take the layer down and forget it was applied."""
        ev.clear_layer(LAYER_THRESHOLD)
        self._applied.discard(geometry_id)


__all__ = [
    "LAYER_THRESHOLD",
    "TOPOLOGY_NODES",
    "TOPOLOGY_GAUSS",
    "GaussValues",
    "SceneValueReader",
    "ThresholdSettings",
    "ThresholdController",
    "cells_with_all_gauss_points_in_range",
    "cells_with_all_nodes_in_range",
    "compute_hidden_mask",
    "values_in_range",
]
