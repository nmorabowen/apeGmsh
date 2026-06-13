"""Stage-activation hidden-cell masks (ADR 0055 viewer-consume, V1).

Maps the staged-analysis PROGRAM (``OpenSeesModel.stages()`` →
``StageRecordRO``, persisted under ``/opensees/stages``) onto the
results substrate grid: which cells exist while each analysis stage is
selected in the viewer.

Semantics (mirrors the bridge's ``_emit_stages_flat`` order)::

    active(K) = global ∪ owned(stages 1..K) − removed(stages 1..K)

A stage's ``remove_elements`` emit at stage start (before its analyze
loop), so a removed element is hidden IN its removing stage, not after
it.  Elements owned by no stage (global topology) are visible in every
stage unless removed.

The program records carry RESOLVED OpenSees element tags; the grid is
keyed by broker FEM element ids (``cell_data["element_id"]`` /
``FEMSceneData.element_id_to_cell``).  The ops-tag → fem-eid join
comes from the model's :class:`ElementRecord` rows (``fem_eid``,
rehydrated from ``/opensees/element_meta``).  Tags that cannot be
mapped (``fem_eid`` sentinel ``-1``, unknown tag, eid not in the grid)
are skipped and counted — the viewer is a read-only consumer, so the
filter degrades open (fail-soft) instead of raising.

Capture stages (the results-zone ``/stages``, ``StageInfo.name``) and
program stages are linked BY NAME (user convention; nothing enforces
a match), with a POSITIONAL fallback via
:func:`pair_capture_to_program` for MPCO/Ladruno captures (their
stage names are ``MODEL_STAGE[<stamp>]``, never equal to program
names — applies only when no name matches and the counts line up).
:meth:`StageActivationMap.mask_for` returns ``None`` for unmatched
names — callers clear the layer so the stage renders unfiltered.
Duplicate program-stage names keep the LAST stage's mask
(registration order), matching how a name-keyed lookup must collapse
them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy import ndarray

#: ElementVisibility layer owned by the stage-activation filter.  Kept
#: here (not element_visibility.py) so the pure data module is the
#: single home of the feature's constants.
LAYER_STAGE: str = "stage"


@dataclass(frozen=True)
class StageActivationMap:
    """Per-stage hidden-cell masks over the substrate grid.

    ``hidden_by_name`` maps each program-stage name to a boolean mask
    of length ``n_cells`` (``True`` = hide).  ``final_hidden`` is the
    configuration after the last stage (everything activated, all
    removals applied) — applied on combined-stage ENTRY; while the
    user scrubs the combined timeline the director re-fires real
    stage ids on every boundary cross, so the view follows the cursor
    with per-stage masks (construction playback).  ``unmapped_tags``
    counts owned/removed tags that could not be joined onto a grid
    cell (diagnostic; node-pair elements like zeroLength carry the
    ``fem_eid`` sentinel and land here by design).
    """

    hidden_by_name: Mapping[str, "ndarray"]
    final_hidden: "ndarray"
    n_cells: int
    unmapped_tags: int

    def mask_for(self, stage_name: Optional[str]) -> "Optional[ndarray]":
        """Mask for a capture-stage name; ``None`` when unmatched."""
        if stage_name is None:
            return None
        return self.hidden_by_name.get(stage_name)


def build_stage_activation(
    stages: "Sequence[Any]",
    element_tag_to_fem_eid: Mapping[int, int],
    element_id_to_cell: Mapping[int, int],
    n_cells: int,
) -> Optional[StageActivationMap]:
    """Build per-stage hidden masks from the staged-analysis program.

    Parameters
    ----------
    stages
        ``OpenSeesModel.stages()`` — :class:`StageRecordRO` rows in
        registration order.  Empty → returns ``None`` (vanilla model;
        callers skip the feature entirely).
    element_tag_to_fem_eid
        ``{ops_element_tag: fem_eid}`` from the model's
        :class:`ElementRecord` rows.  Sentinel/unknown eids are
        skipped (counted in ``unmapped_tags``).
    element_id_to_cell
        ``FEMSceneData.element_id_to_cell`` — broker fem eid → grid
        cell row.
    n_cells
        ``scene.grid.n_cells`` — mask length.
    """
    if not stages:
        return None

    unmapped = 0

    def _cells(tags: "Iterable[int]") -> "ndarray":
        nonlocal unmapped
        rows: list[int] = []
        for tag in tags:
            eid = element_tag_to_fem_eid.get(int(tag))
            if eid is None or int(eid) <= 0:
                unmapped += 1
                continue
            cell = element_id_to_cell.get(int(eid))
            if cell is None:
                unmapped += 1
                continue
            rows.append(int(cell))
        return np.asarray(rows, dtype=np.int64)

    # Owned cells per stage, in registration order.
    owned_per_stage = [_cells(s.owned_element_ids) for s in stages]
    removed_per_stage = [_cells(s.remove_elements) for s in stages]

    # Union of every stage-owned cell: anything in here is hidden
    # until its owning stage arrives.  Global topology never appears.
    all_owned = np.zeros(n_cells, dtype=bool)
    for rows in owned_per_stage:
        all_owned[rows] = True

    hidden_by_name: dict[str, "ndarray"] = {}
    activated = np.zeros(n_cells, dtype=bool)
    removed = np.zeros(n_cells, dtype=bool)
    for stage, owned_rows, removed_rows in zip(
        stages, owned_per_stage, removed_per_stage,
    ):
        activated[owned_rows] = True
        removed[removed_rows] = True
        # hidden(K) = not-yet-activated stage-owned ∪ removed-so-far
        hidden = (all_owned & ~activated) | removed
        hidden_by_name[str(stage.name)] = hidden.copy()

    final_hidden = (all_owned & ~activated) | removed  # == removed
    return StageActivationMap(
        hidden_by_name=hidden_by_name,
        final_hidden=final_hidden.copy(),
        n_cells=int(n_cells),
        unmapped_tags=int(unmapped),
    )


def build_from_model(
    model: Any,
    element_id_to_cell: Mapping[int, int],
    n_cells: int,
) -> Optional[StageActivationMap]:
    """Build the activation map straight from a bound ``OpenSeesModel``.

    Convenience over :func:`build_stage_activation` for the viewer
    wiring (and its tests): reads the staged program via
    ``model.stages()`` and the ops-tag → fem-eid join via
    ``model.elements()``.  Returns ``None`` for vanilla models (no
    program stages) and for models whose surface lacks the accessors
    (fail-soft — read-only consumer).
    """
    try:
        stages = model.stages()
        if not stages:
            return None
        tag_to_eid = {
            int(r.tag): int(r.fem_eid) for r in model.elements()
        }
    except Exception:
        return None
    return build_stage_activation(
        stages, tag_to_eid, element_id_to_cell, n_cells,
    )


def pair_capture_to_program(
    capture: "Sequence[tuple[str, str]]",
    program_names: "Sequence[str]",
) -> dict[str, str]:
    """Map director (capture) stage ids to program-stage names.

    Pairs BY NAME when at least one capture-stage name matches a
    program stage (native capture writes real stage names).  MPCO /
    Ladruno capture stages are named ``MODEL_STAGE[<stamp>]`` — never
    equal to user program names — so when NO name matches and the
    counts line up, pair BY POSITION instead (capture enumeration
    order vs program registration order).  Counts that don't line up
    keep the name mapping (fail-soft: unmatched stages render
    unfiltered).  ``capture`` must exclude the synthetic combined
    entry — the caller handles it via ``combined_stage_id``.
    """
    prog = set(program_names)
    by_name = {sid: name for sid, name in capture}
    if any(name in prog for _sid, name in capture):
        return by_name
    if len(capture) == len(program_names):
        return {
            sid: pname
            for (sid, _name), pname in zip(capture, program_names)
        }
    return by_name


class StageActivationController:
    """Applies stage-activation masks to an :class:`ElementVisibility`.

    Pure-Python glue between the director's stage observer and the
    substrate's layered ghost masks — no Qt, testable headless.  The
    controller owns the :data:`LAYER_STAGE` layer the same way the dim
    filter owns ``LAYER_DIM``: enabled + matched stage → ``set_layer``;
    disabled or unmatched → ``clear_layer`` (stage renders unfiltered).

    ``stage_name_for_id`` resolves a director stage id to the capture
    stage's display name (built from ``director.stages()``);
    ``combined_stage_id`` (the synthetic "All stages" entry) maps to
    the final configuration.
    """

    def __init__(
        self,
        element_visibility: Any,
        activation: StageActivationMap,
        *,
        stage_name_for_id: Callable[[str], Optional[str]],
        combined_stage_id: Optional[str] = None,
    ) -> None:
        self._ev = element_visibility
        self._map = activation
        self._name_for_id = stage_name_for_id
        self._combined_id = combined_stage_id
        self._enabled = True
        self._stage_id: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, flag: bool) -> None:
        """Toggle the filter; re-applies (or clears) immediately."""
        self._enabled = bool(flag)
        self._apply()

    def on_stage_changed(self, stage_id: Optional[str]) -> None:
        """Director ``subscribe_stage`` observer (also called once at
        wiring time with the initial ``director.stage_id``)."""
        self._stage_id = stage_id
        self._apply()

    def current_mask(self) -> "Optional[ndarray]":
        """Mask for the remembered stage (``None`` = no filter)."""
        return self.mask_for_stage_id(self._stage_id)

    def mask_for_stage_id(
        self, stage_id: Optional[str],
    ) -> "Optional[ndarray]":
        """Mask for an explicit stage id (``None`` = no filter).

        Parameterized sibling of :meth:`current_mask` (ADR 0058 S3b —
        a stage-PINNED geometry's scene wears its pinned stage's mask
        while the active stage scrubs). Respects the enabled toggle;
        the combined id maps to the final configuration; an unmatched
        id renders unfiltered (fail-soft).
        """
        if not self._enabled or stage_id is None:
            return None
        if self._combined_id is not None and stage_id == self._combined_id:
            return self._map.final_hidden
        return self._map.mask_for(self._name_for_id(stage_id))

    def _apply(self) -> None:
        mask = self.current_mask()
        if mask is None:
            self._ev.clear_layer(LAYER_STAGE)
        else:
            self._ev.set_layer(LAYER_STAGE, mask)


__all__ = [
    "LAYER_STAGE",
    "StageActivationMap",
    "build_stage_activation",
    "build_from_model",
    "pair_capture_to_program",
    "StageActivationController",
]
