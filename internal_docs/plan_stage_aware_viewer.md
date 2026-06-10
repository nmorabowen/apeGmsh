# Plan — stage-aware results viewer (ADR 0055 consume slice)

Status: V1 IMPLEMENTED (2026-06-10; this PR). V2/V3 not started.

V1 deviations from the proposal below, found during gate-2 review:
the toggle shipped as a checkable toolbar button (the Plan-02
extensibility hook), not a View-menu item; multi-stage files open
stage-LESS (``director.stage_id`` is None until a stage is picked) so
the filter idles until a stage is selected; and during combined-mode
scrubbing the director re-fires real stage ids on boundary crossings,
so the combined view follows the cursor with per-stage masks
(construction playback) rather than pinning the final configuration —
``final_hidden`` applies only at combined-stage entry.

## Goal

When a results file carries `/opensees/stages` (ADR 0055 P2, schema
2.18.0), `results.viewer()` renders the model **per analysis stage**:

* elements owned by a later stage are hidden while an earlier stage is
  selected (construction sequence playback);
* elements removed in a stage disappear from that stage onward;
* stage-bound supports (`s.fix`, HOLD `sp` records) render as glyph
  overlays scoped to their stage;
* optionally, a "Stage" color mode paints each element by its owning
  stage.

This is the user-facing payoff of the P2 read side (`.stages()` was
built for exactly this; unblocked since P2.2, reachable from capture
run files since #590/#591).

## Source-verified inputs (2026-06-10, post #591)

* **Stage program records** — `OpenSeesModel.stages()` →
  `StageRecordRO` (`_internal/typed_records.py:663`), registration
  order. Rendering-relevant fields: `name`, `activated_pgs`,
  `owned_node_ids` / `owned_element_ids` (**resolved ops tags** of
  topology CREATED in that stage; global topology is owned by no
  stage and active from the start), `fixes` (`FixRecord(tag, dofs)`),
  `patterns` (resolved `loads`/`sps`/`sp_holds`), `remove_elements`,
  `remove_sps`, `activate_absorbing`.
* **Active-set semantics**: `active(K) = global ∪ owned(stages 1..K)
  − removed(stages 1..K)`. Removals emit at stage-start (before the
  stage's analyze), so a removed element is hidden IN its removing
  stage, not after it.
* **Tag→cell join**: ops element tags map to broker fem eids via
  `/opensees/element_meta/{token}/{ids, fem_eids}`
  (`h5_reader.element_meta_arrays`, h5_reader.py:419) — the viewer
  data layer already performs this join (`viewers/data/_elements.py:201`).
  Node tags ARE fem node ids (bridge emits broker ids).
* **Capture-stage ↔ program-stage linkage**: name convention ONLY.
  `/stages/stage_NNN` (results zone, `NativeReader.stages()` →
  `StageInfo.name`) and `/opensees/stages` names are both
  user-chosen; nothing enforces a match.
* **Viewer hooks** (all existing, no new seams needed):
  * `ResultsDirector.set_stage` / `on_stage_changed`
    (`viewers/diagrams/_director.py:671-826`) + outline-tree stage
    list (`ui/_outline_tree.py:154,323`);
  * `ElementVisibility.set_layer(name, mask)` layered ghost-cell
    model (`viewers/core/element_visibility.py:123`, ADR 0045) —
    layers OR together, pickers skip hidden cells natively;
  * `DiagramRegistry.reattach_all()` already re-attaches every
    diagram on stage change (`_registry.py:219`);
  * `GlyphLayer` IR for overlays (`scene_ir/_layers.py:238`),
    `LoadsDiagram` (`diagrams/_loads.py`) as the pattern to mirror;
  * `ColorModeController.set_idle_fn` strategy
    (`core/color_mode_controller.py:189`) for a new color mode.

## Non-goals (defer)

* Partitioned staged files — no `/opensees/stages` exists for them
  (ADR 0055 Phase 5; `ops.h5` raises).
* Web viewer (`results.show_web`) parity — separate slice, after the
  Qt viewer proves the model.
* Per-stage modal damping / per-stage modal results.
* Animating mid-stage activation (activation is a stage-boundary
  event in the program model; no sub-stage timeline exists).

## Slices

### V1 — stage activation filter (the core)

1. **Pure mapping module** `viewers/data/_stage_activation.py`:
   `build_stage_masks(stages, element_meta_join, grid_fem_eids) ->
   {stage_name: hidden_cell_mask}` implementing the active-set
   semantics above. Pure numpy, no Qt — unit-testable without GPU.
   Inputs come from surfaces the data layer already loads.
2. **Name matcher**: capture `StageInfo.name` ↔ program
   `StageRecordRO.name`, exact match. **Fail-soft**: any capture
   stage without a program match → no filter for that stage (mask =
   show-all) and a one-line status hint. No positional guessing.
   "All stages" synthetic stage → FINAL configuration (everything
   activated, removals applied).
3. **Wiring**: in `results_viewer._show_impl`, when
   `results.model.stages()` is non-empty, precompute masks once and
   subscribe `director.on_stage_changed` →
   `element_visibility.set_layer(LAYER_STAGE, mask)`. New constant
   `LAYER_STAGE = "stage"` beside `LAYER_DIM`/`LAYER_MANUAL`.
4. **UI**: a "Stage activation" toggle (View menu, default ON when
   masks resolved; absent when the file has no program stages).
5. **Tests**: unit tests for mask math (ownership, removal, global
   carry-through, name mismatch, empty stages); offscreen smoke test
   that a staged Composed file drives `set_layer` with the right
   mask per stage (mirror existing viewer panel-test idiom — no GPU
   here, final eyeball is the user's).

### V2 — stage-bound support overlays

`StageSupportsDiagram` mirroring `LoadsDiagram`: for the current
stage render `fixes` (axes/cone glyph per constrained node, label by
dof mask) and pattern `sps`/`sp_holds` (prescribed-dof glyph).
`update_to_step` is a no-op; stage change already re-attaches via
the registry. Node positions via fem node ids (= tags). Off by
default in the Diagrams tab, like other overlays.

### V3 — "Stage" color mode

`ColorModeController` idle-fn mapping cell → owning-stage palette
color (global topology = neutral grey). Reuses V1's ownership map.

## Decision points (defaults chosen; flag at review)

1. Removed elements hidden IN the removing stage (matches emit order:
   removal precedes the stage's analyze).
2. No-match fallback is silent-permissive (viewer is a read-only
   consumer — fail-loud belongs to writers), with a status hint.
3. Stage-owned NODES are not separately hidden (substrate is
   cell-keyed); V2 glyphs filter themselves by the active node set.

## Gates

Standard loop: scope→implement→gate-2 adversarial diff review→PR
`--base main` per slice. ruff `src/apeGmsh/opensees` clean, mypy 0,
curated suite green. Viewer slices additionally need a live-GPU
eyeball by the user before merge (no GPU on the dev machine —
parity/smoke/offscreen only here).
