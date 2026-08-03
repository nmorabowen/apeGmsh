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
**Cell rule — ALL NODES IN RANGE.** For a NODAL component a cell
survives only if *every* one of its nodes is inside ``[lo, hi]``
(ParaView's "All Points"); one node outside hides the cell. A cell that
straddles the boundary is therefore hidden, not clipped — nothing here
cuts geometry. For a CELL/gauss component the test is direct on the
cell's own value.

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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .element_visibility import LAYER_THRESHOLD

#: Component topologies a threshold understands. ``"nodes"`` values are
#: aligned to grid POINT rows and reduced to cells by the all-nodes
#: rule; ``"gauss"`` values are already aligned to grid CELL rows.
TOPOLOGY_NODES: str = "nodes"
TOPOLOGY_GAUSS: str = "gauss"


@dataclass(frozen=True)
class ThresholdSettings:
    """One geometry's threshold state.

    ``lo > hi`` is an empty range — legal, and it hides everything
    rather than raising (a slider pair can cross mid-drag). ``lo == hi``
    keeps only cells whose every node equals that value exactly.
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


def compute_hidden_mask(
    grid: Any, values: "np.ndarray", settings: ThresholdSettings,
) -> "np.ndarray":
    """The per-cell hide mask for ``values`` under ``settings``.

    ``True`` = hide. For :data:`TOPOLOGY_NODES` the values are per-point
    and go through :func:`cells_with_all_nodes_in_range`; for
    :data:`TOPOLOGY_GAUSS` they are per-cell and the test is direct.
    """
    in_range = values_in_range(values, settings.lo, settings.hi)
    if settings.topology == TOPOLOGY_NODES:
        keep = cells_with_all_nodes_in_range(grid, in_range)
    else:
        keep = in_range
    return ~np.asarray(keep, dtype=bool)


class ThresholdController:
    """Owns the per-geometry threshold state and applies it.

    Parameters
    ----------
    read_values
        ``fn(component, step, *, stage_id=None, topology="nodes")``
        returning the scalar values for that (stage, step) — aligned to
        grid POINT rows for ``"nodes"``, to grid CELL rows for
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
    "ThresholdSettings",
    "ThresholdController",
    "cells_with_all_nodes_in_range",
    "compute_hidden_mask",
    "values_in_range",
]
