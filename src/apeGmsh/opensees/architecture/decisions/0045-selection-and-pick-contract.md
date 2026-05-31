# ADR 0045 — Selection & Pick Contract: `SelectionTarget` IR + `FilterController` + `SelectionLog` over an additively-widened `PickBackend`

**Status:** Accepted (2026-05-30, head-engineer scoping session; born
from a 13-agent design workflow — 6 grounding readers, a ParaView +
Blender prior-art pass, 3 independent contract proposals, an
adversarial critique round, and a synthesis). The four design forks
were **ratified the same day** (see §Resolved decisions) — mirroring
how ADR 0042 was drafted Proposed and flipped to Accepted once its
three questions were chosen. This is the **domain layer above
the pick boundary**: it sits on top of ADR 0047's `PickBackend`
(geometric screen→cell seam) and reshapes 0047's sequencing — the
foundational selection slices here land *before* 0047's web/export
slices. It mirrors ADR 0042's IR + structural-`Protocol` discipline on
the selection side.

## Context

### Three viewers, three pick substrates, and only one shared spine

apeGmsh has three viewers, each picking a fundamentally different
substrate:

- **Model viewer** (`viewers/model_viewer.py`, `scene/brep_scene.py`)
  — **BREP geometry**: OCC vertices / edges / faces / volumes, keyed
  `(dim, occ_tag)`.
- **Mesh viewer** (`viewers/mesh_viewer.py`) — **mesh topology**: FE
  nodes and elements.
- **Results viewer** (`viewers/results_viewer.py`,
  `scene/fem_scene.py`) — **topology driven by the OpenSees bridge +
  Results**: elements, gauss points, fibers, keyed `element_id` /
  `(element_id, gp_index)`.

The model and mesh viewers **share a spine** that already works well:
`core/entity_registry.EntityRegistry` maps `(dim, tag)` → VTK cells,
`core/pick_engine.PickEngine` does the `vtkCellPicker` click / hover /
rubber-band, `core/selection.SelectionState` holds the picks, and
`PickEngine._pickable_dims` gates which dimensions are pickable. The
**results viewer is the odd one out**: an *entirely separate* pick
path (`core/results_pick.install_results_pick` +
`core/results_pick_engine.PickEngine`) with no `EntityRegistry`, keyed
on `element_id` / `gp` / `fiber`, and **no dimension concept at all**.

### Five concrete failures (the repo owner's brief)

1. **No canonical bounding box.** There are *six* competing notions:
   `EntityRegistry._bboxes` (8-corner AABB), the
   `np.tile(centroid, (8, 1))` **degenerate zero-size fallback** at
   `entity_registry.py:155-158`, the `entity_points` 64-point
   subsample, BREP `instance.bbox`, `_world_bbox()`
   (`model_viewer.py:767`), and `_compute_model_diagonal`. They
   disagree on frame (shifted vs. world) and on what "the box" even is.
2. **Picking volumes doesn't select them.** dim=3 picking is broken.
3. **The `0/1/2/3/4` keystroke contract is not honoured.** The keys
   exist *only* inline in `model_viewer.py:1657-1668`; the mesh viewer
   monkeypatches `filter_tab._on_filter` (`mesh_viewer.py:594`) and has
   no key bindings; results has no dimensional filter whatsoever.
4. **(same family as 2/3)** — dimensional pickability is expressed
   three different ways across the viewers and is the root of both the
   keystroke gap and the volume-pick failure.
5. **Operations are not serialized.** `SelectionState._history`
   (`selection.py:105-156`) is a flat, single-level, BREP-only LIFO
   with no redo; mesh element/node picks are *bare lists*
   (`mesh_viewer.py:1417-1442`) with no undo at all; results picks are
   stateless highlights; and live-Gmsh group writes
   (`selection.py:252-259`) are immediate and irreversible.

### What the render seam already gives us to build on

ADR 0042 (render seam) and ADR 0047 (pick boundary) established the
disciplines this contract reuses verbatim: **VTK-free frozen IR value
types**, **structural `Protocol`s (not ABCs)**, **semantics live in the
IR, not in backend tricks**, **additive widening**, **a capability
probe** (`supports_picking()`, already in `scene_ir/_backend.py:132`),
and **one reference backend whose IR output is the parity oracle**. The
selection contract is the next layer up: where ADR 0047's `PickBackend`
resolves *screen → cell geometry*, this ADR governs *cell → selection
semantics* and the cross-viewer state machine around it.

### Prior art: ParaView and Blender solved this

- **ParaView / VTK** — `vtkSelection` decouples *what* is selected from
  *how* it was picked: a `vtkSelectionNode` is `(FieldType, ContentType,
  SelectionList)` — point-vs-cell × indices-vs-frustum-vs-query. Through
  selection uses `vtkExtractSelectedFrustum` (unproject the screen rect
  to six world clip planes, exact plane-side test) — which selects
  *interior solids*, the missing volume-pick mechanism. Hardware
  (G-buffer) picking exists but needs a GPU.
- **Blender** — the vertex / edge / face select-mode keys (`1`/`2`/`3`)
  are the *spine* of selection, not a filter bolted on; X-ray /
  select-through toggles window-vs-crossing; and **every selection
  action is a serialized, replayable operator on one undo stack** — the
  exact shape requirement 5 asks for.

The synthesis below grafts ParaView's typed-decoupled selection node
and frustum volume-select onto Blender's mode-machine spine and
serialized operator log, and lands both on the existing apeGmsh spine.

### Why this is an ADR, not a refactor

It introduces value-type IR + two structural `Protocol`s that all three
viewers and the (future) web/export backends will depend on, unifies
three divergent state machines, and re-sequences ADR 0047. That is "an
architecture event" by the ADR 0024/0025/0026 precedent — and doing it
*before* touching the three pick paths is what makes the migration a
parity-gated sequence instead of a third brittle rewrite.

## Decision

Lock one selection/pick contract by **extraction from the spine that
already works on two of three viewers**, not a green-field rewrite.
Seven abstractions, adopted across slices S0–S6.

### Part 1 — Selection IR (`apeGmsh.viewers.scene_ir`, VTK-free)

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class Substrate(Enum):
    """Which of the three pick substrates a target refers to.
    Keeps BREP / mesh / results identities DISTINCT — never merged
    into one id space (INV-3)."""
    MODEL_BREP = "model_brep"      # (dim, occ_tag) — today's DimTag
    MESH_TOPO = "mesh_topo"        # FE node_id / element_id
    RESULTS_TOPO = "results_topo"  # element_id / (element_id, gp_index) / fiber


@dataclass(frozen=True)
class SelectionTarget:
    """The vtkSelectionNode analog — one selected thing, VTK-free.

    ``dim`` is the dimensional class (0/1/2/3) the 0/1/2/3/4 contract
    filters on. ``key`` is the substrate-native identity (occ_tag for
    BREP, node_id/element_id for mesh, element_id for results).
    ``sub`` carries a secondary index (gp_index for a gauss target).
    ``parent`` links a gauss target to its element / a boundary face to
    its volume — the cross-dim flush channel (additive, INV-8).

    Content-agnostic by design so ONE SelectionState serves all three
    viewers. Each substrate validates its own (dim, key) at __post_init__."""
    substrate: Substrate
    dim: int
    key: int
    sub: Optional[int] = None
    parent: Optional["SelectionTarget"] = None


@dataclass(frozen=True)
class BBox:
    """THE one canonical CAD-like bounding box. World frame, always.

    Replaces the six competing notions as a TYPE (INV-2). Derived
    helpers below; ``origin_shift`` is applied only at projection time,
    never baked into the stored min/max — ending the shifted-vs-world
    disagreement. Shared with the render seam: ADR 0042 reset_camera /
    fit and this contract's frustum builder consume the SAME BBox."""
    min: np.ndarray   # (3,) world
    max: np.ndarray   # (3,) world

    @property
    def corners8(self) -> np.ndarray: ...   # (8, 3)
    @property
    def center(self) -> np.ndarray: ...     # (3,)
    @property
    def diagonal(self) -> float: ...
    def union(self, other: "BBox") -> "BBox": ...
    def contains(self, pt: np.ndarray) -> bool: ...
```

### Part 2 — `SelectableSubstrate` Protocol (generalises `EntityRegistry`)

```python
from typing import Protocol, runtime_checkable, Sequence


@runtime_checkable
class SelectableSubstrate(Protocol):
    """The cell↔target contract each viewer's pickable model implements.

    Structural (not an ABC), like ADR 0026's H5ModelReader and ADR 0042's
    RenderBackend. EntityRegistry implements it for MODEL_BREP + MESH_TOPO;
    a new ResultsSubstrate wraps FEMSceneData (with a new cell_dim array)
    for RESULTS_TOPO. Adding a fourth (foreign) substrate = implement this
    + add a Substrate enum value, with ZERO change to SelectionState /
    SelectionLog / FilterController (INV-8)."""

    @property
    def substrate(self) -> Substrate: ...
    @property
    def dims(self) -> Sequence[int]: ...
    def resolve_cell(self, actor_id: int, cell_id: int) -> Optional[SelectionTarget]: ...
    def targets_by_dim(self, dim: int) -> Sequence[SelectionTarget]: ...
    def bbox(self, target: SelectionTarget) -> Optional[BBox]: ...
    def representative_points(self, target: SelectionTarget, max_points: int = 64) -> Optional[np.ndarray]: ...
```

### Part 3 — `FilterController` (the `0/1/2/3/4` spine)

A single, sticky owner of the dimensional select-mode `frozenset`, plus
the select-through / X-ray and window-vs-crossing toggles. **One per
viewer**, bound identically in all three. The `0/1/2/3/4` keystrokes and
both `FilterTab` variants are two *front-ends* writing the same
`frozenset`; `PickEngine._pickable_dims` becomes a *derived mirror*, not
a source of truth.

`set_mode` fans out through **one** code path to three coupled effects:
the pick-resolution gate (replacing the scattered `dt[0] not in
_pickable_dims` guards at `pick_engine.py:307,324,382`), per-dim actor
`SetPickable`, and visibility feedback (one policy, replacing model's
"dim to 0.1 opacity" vs. mesh's "hide" split). Every keypress emits a
`MODE_SET` op (Part 5) for replay.

**Ratified keystroke semantics (§Resolved decisions 1):** bare
`0/1/2/3` **toggle/add** a dim bit — the dimensional filter is
**multi-select by default**, so the keys and the checkbox panel are
symmetric front-ends onto the same `frozenset` (a key flips its dim
in/out of the active set). `4` selects all dims; a dedicated clear
(`Esc`-class) resets to empty. There is no separate "single-dim
replace" key mode — REPLACE was the rejected alternative.

### Part 4 — widened `PickBackend` (from ADR 0047)

ADR 0047's `PickBackend` is widened **additively**: it keeps
`resolve_pick(request) -> PickHit` (the existing `vtkCellPicker` ray,
the no-GPU-safe reference path) and gains a sibling
`resolve_frustum(corners8) -> Sequence[cell hits]` (a true viewing
frustum via `vtkExtractSelectedFrustum`). `PickHit` stays the thin
geometric carrier (`actor_id`, `cell_id`, `world`, `crossing`); the
domain maps it to a `SelectionTarget` via
`SelectableSubstrate.resolve_cell`. This answers ADR 0047's open
questions: one geometric Protocol (Q4), types live in `scene_ir` (Q2),
both faces kept (Q1). `vtkHardwareSelector` stays **off** the critical
click path (no GPU here).

### Part 5 — `SelectionOp` + `SelectionLog` (serialization)

```python
class OpKind(Enum):
    ADD = "add"; REMOVE = "remove"; TOGGLE = "toggle"
    BOX_ADD = "box_add"; BOX_REMOVE = "box_remove"; CLEAR = "clear"
    MODE_SET = "mode_set"
    GROUP_ACTIVATE = "group_activate"; GROUP_COMMIT = "group_commit"


@dataclass(frozen=True)
class SelectionOp:
    kind: OpKind
    targets: Tuple[SelectionTarget, ...] = ()   # one op per GESTURE
    payload: Optional[object] = None            # mode frozenset, group name, ...
```

**One** process-wide ordered, append-only `SelectionLog` with a cursor,
shared across all three viewers (not per-domain — a whole-session replay
needs one deterministic order; per-viewer undo is a *cursor filter*).
Every mutation routes through `SelectionState.apply(op)`. Properties:
**ordered** (append + cursor), **gesture-grouped** (a box of N targets is
*one* op — fixes the per-entity undo granularity at
`selection.py:170-180`), **undoable + redoable** (cursor back/forward),
**replayable** (re-apply ops `0..k`; targets are content-agnostic VTK-free
values, so the log serialises). `MODE_SET` is in the serialized/replay
log but **off the user-facing undo cursor**. Live-Gmsh group writes
become `GROUP_ACTIVATE`/`GROUP_COMMIT`; staging is in-memory and
reversible; `flush_to_gmsh` is the single explicit freeze boundary. No
races: all ops on the one Qt/VTK main thread through one `apply`;
`on_changed` fires once per committed op; the VTK observer priority
abort-chain (10/9/11) is preserved untouched *below* the op layer.

### Substrate mapping

| Substrate | Target identity | How it lands |
|---|---|---|
| **MODEL_BREP** | `dim` = OCC dim, `key` = occ_tag (**= today's DimTag verbatim**) | Volume pick is **already grounded**: `brep_scene.py:461-555` builds dim=3 as merged volume boundary surfaces, registers a **separate dim=3 actor**, and at `:547-550` maps every boundary-face cell to its owning volume tag with real per-volume bboxes (`:551`). The breakage is *only* the coincident dim=2/dim=3 actors (both `pickable=True`) losing the depth-sort tie — fixed by the `FilterController` (Part 6 below). |
| **MESH_TOPO** | element: `dim`=element dim, `key`=element_id; node: `dim`=0, `key`=node_id | Gives mesh element/node picks (today bare lists, `mesh_viewer.py:1417-1442`) first-class identity in `SelectionState`, so undo / box / staging / filter all apply. The brep-mode vs element/node-mode split collapses. BBox from mesh node coords the registry already holds; the `np.tile` fallback is **deleted**. |
| **RESULTS_TOPO** | element: `dim`=`cell_dim`, `key`=element_id; node: `dim`=0; gauss: `key`=element_id+`sub`=gp_index, `parent`=element; fiber: `parent`=gauss | `fem_scene.build_fem_scene` has the element dim in hand (`fem_scene.py:248-257`) but discards it — store a parallel `cell_dim` ndarray + split the merged grid into per-dim sub-grids (fixes hex pick ambiguity). `PickMode` (node/element/gp/fiber) stays an **orthogonal semantic axis** composed with dim: dim scopes which cells are pickable, a post-hit resolver interprets the survivor. *(Note: `PickMode`
node/element/gp/fiber here is the results-only semantic axis — distinct
from the dimensional `0/1/2/3/4` filter, which the `FilterController`
owns.)* `set_pick_mode` is **kept** (it gates the GP/FIBER overlay actors — the `MODE_ALLOW` sets for NODE/ELEMENT are both empty, so node-vs-element is post-hit); only the controller-vs-engine desync is fixed. |

### Volume pick — the mechanism (verified, not asserted)

**CLICK = highest-active-dim-wins** over the *existing* boundary→volume
map (no new dual map — all three proposals wrongly assumed it was
missing; `brep_scene.py:547-550` proves otherwise). When only dim=3 is
active, the `FilterController` sets the dim=2 actor non-pickable, so the
picker can *only* hit the dim=3 actor and `resolve_cell` returns the
volume; in dim=4 ("all"), a documented highest-active-dim-whose-actor-was-hit
tie-break in `resolve_cell`. **Headless-verifiable** on a synthetic
registry — no render, no GPU.

**BOX = true viewing frustum.** `resolve_frustum` unprojects the 8
rubber-band corners to 6 world clip planes (`vtkExtractSelectedFrustum`)
and does an exact plane-side test against real cell geometry — selecting
*interior* solids. This is the only piece needing a live camera/GPU, so
it is gated behind `supports_picking()`, **parity-asserted against the
shipped 64-point `entity_points` subsample** (kept as both the headless
oracle *and* the fallback box path — preserving the measured ~20×
optimisation in `test_core_perf.py`), and guarded by a perf-regression
test. The mesh-substrate cross-dim down-flush (pick volume ⇒ its faces)
is **BREP-only in v1** (gmsh gives adjacency for free); mesh/results
dim=4 means all dims pickable without down-flush.

## Invariants

- **INV-1** — `SelectionTarget`, `Substrate`, `BBox`, `PickRequest`,
  `PickHit` import **neither `vtk` nor `pyvista`**. Enforced by
  extending the ADR 0042 AST purity guard
  (`tests/test_scene_ir_pure.py`).
- **INV-2** — Exactly **one** bounding-box value type (`scene_ir.BBox`),
  world-frame; all 8-corner and 6-tuple consumers *derive* from it. No
  `np.tile` centroid-tile, no projected-AABB stored as truth. Providers
  may differ per substrate (gmsh for BREP/mesh, grid sub-extents for
  results — there is **no** gmsh model behind results, so a single
  producer is impossible) but all return `BBox`. INV-2 fixes the
  **type**, not a single producer. `origin_shift` applied only at
  projection.
- **INV-3** — The three substrates keep **distinct identities**. The
  *type* is unified (`SelectionTarget`), the *values* are not;
  `SelectionTarget.substrate` keeps them separate and each validates its
  own `(dim, key)`.
- **INV-4** — **One `FilterController` per viewer** owns dimensional
  pickability; the `0/1/2/3/4` keys and the `FilterTab`/`MeshFilterTab`
  checkboxes are two front-ends on the same `frozenset`, consumed
  identically by all three viewers. Pickability = resolution-gate AND
  actor `SetPickable` AND visibility — one concept.
  `PickEngine._pickable_dims` is a derived mirror.
- **INV-5** — All selection mutations across all three viewers route
  through **one** `SelectionState` applying `SelectionOp`s to **one**
  ordered `SelectionLog`. No bare pick lists, no stateless results
  highlights, no second store. One op per gesture; `on_changed` fires
  once per committed op. `MODE_SET` is serialized but off the
  user-facing undo cursor.
- **INV-6** — Picking **semantics** (target mapping, dim filtering,
  `PickMode` routing, highlight, group staging) live in the domain
  *above* the backend — re-affirming ADR 0047 INV-3. `PickBackend`
  resolves only screen→cell geometry (`vtkCellPicker` ray +
  `vtkExtractSelectedFrustum`). The reference `vtkCellPicker` path is
  the parity oracle, assertable headlessly.
- **INV-7** — Volume **CLICK** uses highest-active-dim-wins over the
  *existing* boundary→volume map, gated by the `FilterController`;
  volume **BOX** uses a true world-frustum plane-side test, gated by
  `supports_picking()` and parity-asserted against the 64-point
  subsample (retained as fallback + oracle). Never a projected-AABB or
  centroid-tile heuristic for volumes.
- **INV-8** — The IR **widens additively** (hover channel, X-ray,
  cross-dim flush via `SelectionTarget.parent`, multi-hit ray, a fourth
  foreign substrate) as new fields / enum members / Protocol methods —
  never a VTK object across the seam. Adding a substrate = implement
  `SelectableSubstrate` + a `Substrate` value, with zero change to
  `SelectionState` / `SelectionLog` / `FilterController`.

## Phased adoption (parity-gated, mirrors ADR 0042's R-A→R-C staging)

| Slice | Scope | Headless-verifiable? |
|---|---|---|
| **S0** | `scene_ir._targets` (`SelectionTarget`, `Substrate`) + `scene_ir._bbox` (`BBox`) value types; extend the AST purity guard. No behaviour change. | **Fully** — value-type + purity unit tests; no VTK/GPU. |
| **S1** | `BBox` provider; `EntityRegistry.bbox` returns `BBox` from real geometry; **delete** the `np.tile` degenerate fallback (`entity_registry.py:155-158`); `mesh_scene` passes real AABBs into `register_dim`; fold `instance.bbox` / `_world_bbox` / `_compute_model_diagonal` onto `BBox.union`. **Per §Resolved decisions 4, also migrate the remaining `getBoundingBox` sites in v1** — `overlays/tangent_normal_overlay.py`, `overlays/mesh_tangent_normal_overlay.py`, and `_model_info_panel.py:98` — so the sweep is complete (no follow-up bbox debt). | **Mostly** — `BBox` value parity on golden registries; orbit-pivot/clip-range asserted without render. Camera-fit pixels = eyeball. |
| **S2** | `FilterController`; move the inline `0/1/2/3/4` keys (`model_viewer.py:1657-1668`) into it; bind it in the mesh viewer (closes HARD REQ 2); route both `FilterTab` variants through `set_mode` (removes the `mesh_viewer.py:594` monkeypatch; adds the missing dim=0 row). **The cheapest user-visible win.** | **State headless** — assert `frozenset` transitions + the pick-resolution gate on a synthetic registry. Visual feedback = eyeball. |
| **S3** | `SelectionState` holds `SelectionTarget`s + a `SelectionLog`; adapt `DimTag` → `SelectionTarget(MODEL_BREP)` for one release; migrate mesh element/node picks OFF bare lists INTO `SelectionState`; box pushes ONE op; add redo; `GROUP_ACTIVATE`/`GROUP_COMMIT` through the log with `flush_to_gmsh` as the freeze boundary. | **Fully** — replaying the op-log from empty reproduces the `SelectionState`; one-op-per-gesture undo/redo + staging deferral, no interactor. |
| **S4** | Add `cell_dim` to `fem_scene` (the discarded dim at `fem_scene.py:248-257`) + split the results grid per-dim; write `ResultsSubstrate` implementing `SelectableSubstrate`; give results a `SelectionState` + `FilterController` + post-hit node/element resolver composed with the dim filter; fix the controller-vs-engine mode desync without deleting `set_pick_mode`. | **Mostly** — `cell_dim` tagging, `resolve_cell`, per-element `BBox`, dim-masked candidates on a golden FEM scene. Highlight pixels = eyeball. |
| **S5** | Add `resolve_frustum` to the `PickBackend` Protocol + reference backend (`vtkExtractSelectedFrustum`); switch `_do_box` for volumes to the frustum path behind `supports_picking()`; keep the 64-point subsample as fallback + oracle; add the highest-active-dim-wins click tie-break. Fixes HARD REQ 4 box path. | **Partial** — frustum plane-math + the click resolver are headless unit tests; result-set parity vs the subsample on synthetic scenes; real over/under-select on angled volumes = GPU + eyeball. |
| **S6** | Recast ADR 0047's `PickRequest`/`PickHit`/`PickMode` as the wire-level subset the domain maps from (adapters for one release); re-sequence 0047's web/export slices to consume the canonical `PickHit`→`SelectionTarget` mapping. ADR 0047 amended + Accepted after S0–S5 prove the layer. | **Headless** for the adapter mapping; web/export verification is downstream / out of this environment. |

## Relation to ADR 0042 and ADR 0047

ADR 0042 (`RenderBackend` + `SceneLayer` IR) is **untouched** and is the
discipline template. The **one shared type across both seams is `BBox`**
(render `reset_camera`/fit and pick frustum/fit consume it) — one bbox
truth across the render and pick boundaries.

ADR 0047 (`PickBackend`, still Proposed) is **reshaped, not
superseded** — it is correctly the lower geometric seam; this contract
is the domain layer above it, re-affirming 0047 INV-3. Concretely:
0047's `PickHit` stays the thin geometric carrier; the domain maps it to
a `SelectionTarget` via `resolve_cell` (answers 0047 Q2: types live in
`scene_ir`, plus `SelectionTarget`/`BBox`). 0047's `PickBackend` gains
`resolve_frustum` as an additive sibling (answers 0047 Q4: **one**
geometric Protocol, **three** `SelectableSubstrate` implementers, not
three pick paths — honouring 0047 §Rejected C). 0047 Q1 = keep both
faces (stateless `resolve` core + `install` desktop observer); the
`SelectionLog` is the new layer between resolved hits and
`SelectionState`. 0047's `vtkHardwareSelector` ambition is explicitly
off the critical click path. 0047's R-D is **re-sequenced**: this
contract's foundational slices (S0–S4, all headless) land *before*
0047's web/export slices, which then consume the now-canonical
`PickHit`→`SelectionTarget` mapping unchanged. The mapping honours
ADR 0027 per-rank cross-partition identity.

## Rejected alternatives

- **`vtkHardwareSelector` G-buffer as the reference CLICK path**
  (Proposal 1). No GPU here — the most important fix (volumes) would be
  unverifiable until run on a GPU box. Keep `vtkCellPicker` for click;
  reserve the frustum strictly for box/through.
- **Full op-log + multi-select + undo bolted onto the results viewer in
  v1** (Proposal 1). Unrequested scope, the riskiest single slice;
  silently changes the picks-don't-accumulate UX. HARD REQ 1 only asks
  results to be *pickable*. Defer accumulation/undo behind a toggle
  (Open Q2).
- **Per-domain / per-viewer op-logs** (Proposal 2). Three independent
  logs lose cross-viewer serial order, undercutting HARD REQ 5. One
  process-wide ordered log; per-viewer undo is a cursor filter.
- **`MODE_SET` on the user-facing undo stack** (Proposal 2). Ctrl+Z
  stepping back through dimension-mode changes surprises users. Keep
  `MODE_SET` serialized but off the undo cursor.
- **`BBox` sourced uniformly from `gmsh.getBoundingBox`, or exactly one
  producer** (Proposal 3 INV-2). Results topology has no gmsh model —
  uniform sourcing is false for one of three substrates. Scope the INV
  to one bbox *type* with per-substrate providers.
- **Building a brand-new boundary-cell→volume dual map as unscoped late
  work** (all three proposals assumed it was missing). **Verified
  false**: `brep_scene.py:547-550` already registers it. The volume
  CLICK fix is `FilterController` disambiguation over the existing map.
- **Discarding the 64-point `entity_points` subsample for frustum-only
  box-select** (Proposal 3). Trades a measured ~20× path
  (`test_core_perf.py`) for an unbenchmarked O(cells) plane test that
  can't be perf-validated here. Keep the subsample as oracle + fallback.
- **Deleting `results_pick_engine.set_pick_mode` as dead routing**
  (Proposal 3). Misread of `MODE_ALLOW`: NODE/ELEMENT allow-sets are
  empty (node-vs-element is post-hit), but `set_pick_mode` actively
  gates the GP/FIBER overlay actors. Keep it; fix only the desync.

## Resolved decisions (ratified 2026-05-30)

The four design forks the synthesis left open were ratified the same day
(two confirmed the synthesis lean, two overrode it). Recorded inline, as
ADR 0042 recorded its three.

1. **Keystroke semantics — multi-select by default** *(overrode the
   single-dim-replace lean).* Bare `0/1/2/3` **toggle/add** a dim bit;
   the dimensional filter is multi-select, so the keystrokes and the
   checkbox panel are symmetric front-ends onto one `frozenset`. `4` =
   all. There is no separate `Shift`-to-add mode and no single-dim
   REPLACE — wired into Part 3. Rationale: matches the existing
   checkbox-panel muscle memory and keeps one mental model for "which
   dims are active."
2. **Results viewer scope — pickable + dim-filter + store now;
   accumulation/undo behind a toggle** *(confirmed the lean).* Results
   gains pickability, the `0/1/2/3/4` filter, and a `SelectionState`
   store in S4; multi-select accumulation + undo go behind an explicit
   toggle so the existing picks-replace UX
   (`_highlight_element_cells`, `results_viewer.py:2251`) is not silently
   changed.
3. **`MODE_SET` is off the user-facing undo stack** *(confirmed the
   lean).* It stays in the serialized/replay log (for whole-session
   replay) but `Ctrl+Z` does not step back through dimension-mode
   changes — undo touches selections; mode history is a separate
   replayable channel. Wired into Part 5 / INV-5.
4. **Canonical `BBox` migration is complete in v1** *(overrode the
   follow-up-debt lean).* S1 migrates **every** `getBoundingBox` site,
   including `overlays/tangent_normal_overlay.py`,
   `overlays/mesh_tangent_normal_overlay.py`, and
   `_model_info_panel.py:98` — no residual bbox debt. INV-2 still fixes
   the *type* (per-substrate providers remain), but every producer lands
   on `BBox` in the first pass.

## References

- [ADR 0042](0042-render-backend-seam.md) — `SceneLayer` IR +
  `RenderBackend` Protocol. The discipline template (VTK-free IR,
  structural Protocol, semantics-in-IR, additive widening, capability
  probe, IR parity oracle). `BBox` is the one shared type across both
  seams.
- [ADR 0047](0047-pick-backend-and-export.md) — `PickBackend` Protocol +
  web picking + ParaView export. This contract is the domain layer above
  it; 0047 is reshaped (additive `resolve_frustum`, `PickHit`→
  `SelectionTarget` mapping) and re-sequenced (its web/export slices land
  after S0–S5). Answers 0047's four open questions.
- [ADR 0026](0026-h5modelreader-protocol-contract.md) — the read-side
  structural `Protocol`; `SelectableSubstrate` is its selection-side
  sibling.
- [ADR 0024](0024-emitter-protocol-widen-region.md) /
  [ADR 0025](0025-emitter-protocol-widen-eigen.md) — Protocol
  introduction/widening as an architecture event; the precedent for
  INV-8's additive-widening rule.
- [ADR 0027](0027-cross-partition-mp-constraints.md) — per-rank
  cross-partition identity the `PickHit`→`SelectionTarget` mapping
  honours.
- `viewers/core/entity_registry.py`, `core/pick_engine.py`,
  `core/selection.py`, `core/results_pick.py`,
  `core/results_pick_engine.py`, `scene/brep_scene.py`,
  `scene/fem_scene.py` — the current three-path reality S1–S5 unify.
- `tests/test_scene_ir_pure.py` — the INV-1 AST guard S0 extends.
- ParaView `vtkSelection` / `vtkExtractSelectedFrustum` and Blender's
  select-mode + operator-undo model — the prior art the 13-agent design
  workflow mined for the typed-decoupled selection node, frustum
  volume-select, dim-mode spine, and serialized operator log.
