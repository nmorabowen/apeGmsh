# ADR 0044 — `PickBackend` Protocol + web picking + `ParaViewExportBackend` (Phase R-D)

**Status:** Accepted (2026-05-30). Drafted Proposed in the head-engineer
scoping session with four open questions; **ratified the same day** after
a discovery pass over the two live pick engines resolved all four (see
§Resolved decisions) and surfaced two refinements to the draft IR (a
`prop_id` registry-key on `PickHit`; box picking as a shared *projection
primitive* rather than resolved ids in the IR). R-D.1 (the IR +
`PickBackend` Protocol) landed against this ratified contract. Phase R-D
of the render-backend seam. Successor to **ADR 0042** (`SceneLayer`
IR + `RenderBackend` Protocol): R-D fills the one capability ADR 0042
*deliberately* deferred — picking — and adds the first non-interactive
backend behind the seam (`ParaViewExportBackend`). It does not amend
ADR 0042; it consumes the `supports_picking()` capability probe that
ADR 0042 designed in for exactly this purpose.

**Scope ratified for this land:** R-D.1 + R-D.2 (the uniform desktop
implementation — pick IR + Protocol, then both desktop engines extracted
behind it). R-D.3 (server/hybrid web picking) and R-D.4
(`ParaViewExportBackend`) remain deferred follow-ons. The R-D.2 extraction
also **removes the dead results-side mode machinery** (`PickEngine.set_pick_mode`
has zero production callers — the controller-string mode and the engine
allow-list are collapsed into one path, not carried forward).

## Context

### What ADR 0042 finished, and the one thing it left open on purpose

ADR 0042's render seam is **complete through R-C**:

- **R-A/R-B** — all 11 diagram types emit `SceneLayer` value types and
  call only `RenderBackend`; `viewers/diagrams/` imports no `vtk` /
  `pyvista` (INV-2, guarded by `tests/test_diagrams_pure_no_pyvista.py`).
- **R-C** — `TrameBackend` ships (web / Jupyter), `results.show_web()`
  (inline Jupyter, ipywidgets controls, client/server/hybrid render
  modes) and `results.serve_web()` (standalone vuetify3 browser app)
  are live alongside the desktop `results.viewer()`.

One capability was held back by design. ADR 0042 §Decision Part 2 put
picking **off** the base `RenderBackend` Protocol and made
`supports_picking()` a capability probe:

> Picking is **not** on the base Protocol. Ray-casting is the most
> deeply VTK-bound surface and the least essential for the first
> web/Jupyter target […]. It gets its own `PickBackend` Protocol in
> Phase R-D (future ADR).

`PyVistaQtBackend.supports_picking()` returns `True`
(`backends/pyvista_qt.py:412`); `TrameBackend.supports_picking()`
returns `False` (`backends/trame.py:37`). The web viewer is
**view-only**: "look at my model" and "animate my steps" work; "click
a node to read its displacement" does not. That gap is *explicit*, not
silent — but it is still a gap, and it is the last unstarted chunk of
the original render-seam plan.

### Picking is the last domain surface that bypasses the seam

Everything the viewer *draws* now goes through `SceneLayer` →
`RenderBackend`. Everything it *picks* still talks straight to VTK.
There are **two independent VTK-observer pick engines** today, neither
behind any seam:

1. **Mesh-side** — `viewers/core/pick_engine.PickEngine`. Constructs
   `vtk.vtkCellPicker`s, installs `iren.AddObserver` on the plotter's
   interactor for LMB press/move/release (`pick_engine.py:160-291`),
   does click + hover + rubber-band box-select, resolves hits through
   `EntityRegistry.resolve_pick(id(prop), cell_id)` to a `(dim, tag)`.
   Wired in `mesh_viewer.py:430,644`.
2. **Results-side** — `viewers/core/results_pick.install_results_pick`
   (the observer controller) + `viewers/core/results_pick_engine.PickEngine`
   (the per-actor inventory + `PickMode` allow-list routing for
   node / element / gp / fiber, with the `Alt`-pick-through gesture).
   Resolves to a `PickResult` / `BoxPickResult` (`element_id`,
   `gp_index`, world coords). Wired in `results_viewer.py:1217,308`.

Both reach raw `vtk.vtkCellPicker`, both install interactor observers,
both project world→display via the shared `_project_points_to_display`
helper, both draw a `vtkActor2D` rubber-band. They are entirely
Qt/VTK-desktop-bound. This is the **structural mirror** of the
pre-ADR-0042 render scatter: 11 diagrams once called `plotter.add_mesh`
directly; two pick engines now call `vtkCellPicker.Pick` directly.

### ParaView was kept as a future export adapter and never started

ADR 0042 §Rejected B assessed ParaView, rejected a *wholesale* engine
swap (it has zero concept of OpenSees fiber/layer/gauss/partition
domain logic), and kept it as a **future export/automation adapter
behind `RenderBackend`** — never a hard dependency. The seam now has
two interactive backends (`PyVistaQtBackend`, `TrameBackend`) but no
non-interactive one. A `ParaViewExportBackend` is the cheapest way to
prove the seam holds for a *third*, fundamentally different consumer
(export, not display) — and it delivers a real power-user feature
(hand the model to ParaView / a `.vtm` / `.pvd` for offline analysis).

### Why this is an ADR, not a refactor

Same reasoning as ADR 0024/0025/0026: introducing a Protocol that
multiple call sites will depend on is "an architecture event." A
`PickBackend` will be implemented by the desktop backend, the trame
backend, and consumed by both viewers' domain layers plus the web
shell. Codifying the contract — and the desktop↔web *asymmetry* it has
to absorb — **before** refactoring the two existing engines is what
makes R-D a controlled, parity-gated sequence instead of a second
brittle big-bang. The picking surface also carries a desktop/web split
the render seam never had (event-driven observers vs.
request/response over a socket), which is precisely the kind of design
tension an ADR exists to record.

## Decision

Introduce a third seam element — an **optional** `PickBackend`
Protocol, sibling to `RenderBackend`, gated by the existing
`supports_picking()` capability probe — plus a VTK-free pick IR, and
adopt it in four parity-gated slices. Add `ParaViewExportBackend` as
the first non-interactive `RenderBackend` implementer.

### Part 1 — pick IR (`apeGmsh.viewers.scene_ir`)

The screen-coordinate input and the resolved-entity output are plain
value types — **no `vtk`, no `pyvista`** (extends ADR 0042 INV-1). The
results-side `PickResult` / `BoxPickResult` dataclasses
(`core/results_pick.py:55-109`) are *already* VTK-free; R-D promotes
their shape into the shared `scene_ir` vocabulary so both backends and
the web shell speak one language.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence
import numpy as np


class PickMode(str, Enum):
    """What a click is *for*. Domain-level routing, not a backend concern.

    The backend resolves a screen hit to a cell + world point; the mode
    decides how the domain layer interprets it (snap-to-node, resolve to
    element id, route to a gauss / fiber overlay). Mirrors the existing
    results_pick_engine.PickMode allow-list machinery."""
    NODE = "node"
    ELEMENT = "element"
    GP = "gp"
    FIBER = "fiber"


@dataclass(frozen=True)
class PickRequest:
    """A single click-pick query in display-pixel space. The stateless
    core's only input.

    ``(x, y)`` is a display-pixel coordinate. ``mode`` and ``modifiers``
    travel so the backend can run the correct picker, but the
    *interpretation* of the result stays in the domain layer (INV-3).
    Box picking is **not** a ``PickRequest`` — see ``BoxGesture`` and the
    projection primitives below."""
    x: int
    y: int
    mode: PickMode = PickMode.NODE
    modifiers: PickModifiers = PickModifiers()   # ctrl / alt (no shift)


@dataclass(frozen=True)
class PickHit:
    """A resolved screen→scene hit, VTK-free — the geometric click result.

    ``prop_id`` is ``id()`` of the picked VTK prop — a **plain integer**,
    the key both domain registries already use (``EntityRegistry``'s
    ``_actor_id_to_dim`` for mesh ``(dim, tag)``; the results pick
    inventory's ``id(vtkProp) → (kind, …)``). Returning the integer, not
    the actor, is what lets the hit cross the VTK-free seam (INV-2) while
    each viewer keeps its own entity vocabulary (INV-3). A miss is
    reported as ``None`` from ``resolve_pick``, not an empty ``PickHit``."""
    world: tuple[float, float, float]
    cell_id: Optional[int] = None
    prop_id: Optional[int] = None
```

**Box picking** is the one place the two engines' candidate sourcing
genuinely diverges (mesh vertices vs. GP centers vs. node coords), so the
*resolved ids* never enter the IR. Instead the seam exposes the shared
geometric primitive both engines already use — world→display projection
(today's `_project_points_to_display`, which mesh literally imports from
results) and the optional 3-D `frustum_planes` — and the domain runs the
in-box test against its own candidates. A completed drag reaches the
domain as a `BoxGesture` (box rect + `crossing` + mode + modifiers); the
domain resolves it. This corrects the draft's `PickHit.entity_ids`, which
would have leaked domain resolution into the IR (§Rejected C caveat).

The IR is **additive** (ADR 0042 INV-6 discipline): a future hover
channel or a multi-hit ray result is a new field / type, never a VTK
object smuggled back across the seam.

### Part 2 — `PickBackend` Protocol (`apeGmsh.viewers.scene_ir`)

```python
from typing import Protocol, runtime_checkable, Callable


@runtime_checkable
class PickBackend(Protocol):
    """Optional capability: resolve a screen pick to a scene hit.

    A RenderBackend exposes a PickBackend only when
    ``supports_picking() == True`` (ADR 0042 Part 2). View-only
    backends (trame client-only, ParaViewExportBackend) expose none.

    The Protocol must absorb TWO faces (see §Open question 1):

    * **desktop, event-driven** — the backend installs interactor
      observers and fires callbacks (the current PickEngine model;
      preserves the priority-10/11 abort chain shared with navigation).
    * **web, request/response** — a browser click forwards display
      coordinates to the server, which resolves and returns a PickHit.

    ``resolve_pick`` is the stateless core both faces share; ``install``
    is the desktop event face layered on top of it.
    """

    # -- stateless geometric core (web reuses this verbatim in R-D.3) --

    def resolve_pick(self, request: "PickRequest") -> "Optional[PickHit]":
        """Ray-cast one click; return the geometric hit, or None on a miss.

        Pure screen→scene geometry: vtkCellPicker.Pick → prop id + cell
        id + world. No mode routing, no FEM-id resolution, no highlight —
        those are domain logic (INV-3)."""
        ...

    def project_points(self, world: "np.ndarray") -> "np.ndarray":
        """Project (N,3) world points to (N,2) display pixels — the shared
        box-pick core. The domain feeds its own candidates and runs the
        in-box test itself (INV-3)."""
        ...

    def frustum_planes(
        self, box: "tuple[int, int, int, int]"
    ) -> "Optional[np.ndarray]":
        """Un-project a display box to its 6 world frustum planes for an
        exact 3-D test, or None when the camera can't un-project (domain
        falls back to the 2-D project test). Optional capability."""
        ...

    # -- desktop event face (layered over the stateless core) --

    def install(
        self,
        *,
        on_pick: "OnPick",
        on_hover: "Optional[OnHover]" = None,
        on_box: "Optional[OnBox]" = None,
    ) -> None:
        """Install the event-driven desktop face (interactor observers).

        The backend runs the click/hover/drag state machine + rubber-band
        overlay and fires the callbacks with the *geometric* result. A
        request/response backend (web) may leave this a no-op and be
        driven purely through ``resolve_pick``."""
        ...

    def uninstall(self) -> None:
        """Remove any installed observers + overlay actors. Idempotent.

        Closes the observer leak both legacy engines carry."""
        ...
```

The live contract is `viewers/scene_ir/_pick.py`; the bodies above are
illustrative. `OnPick` / `OnHover` / `OnBox` are the callback aliases
defined there (`(PickHit, PickModifiers)`, `(Optional[PickHit])`,
`(BoxGesture)`).

`RenderBackend` is **not** widened. A backend that supports picking
exposes the `PickBackend` via a `picking() -> Optional[PickBackend]`
accessor (or simply *is* one — structural, the consumer probes
`supports_picking()` then narrows). The exact attachment point is
§Open question 1; what is fixed is that picking stays **off the base
Protocol**, exactly as ADR 0042 INV-3 chose.

### Part 3 — web (server-side) picking

In `server` / `hybrid` render mode the trame view has a *live VTK
render window on the server*. A browser click forwards display
coordinates over the trame/wslink socket; a server-side `@state.change`
(or trame mouse-event) handler calls `resolve_pick(PickRequest(...))`
against that server render window — **reusing the same `vtkCellPicker`
resolution the desktop uses**. The resolved `PickHit` flows back to the
domain layer (node-snap / element-id / gp), and the web shell renders a
read-out (the inline-Jupyter and standalone-app counterparts of the
desktop HUD).

`client`-only render mode (vtk.js, the snappy default chosen at R-C)
has **no server geometry** to ray-cast against — picking there needs a
vtk.js-side pick + a JS→Python callback bridge, which is a separate,
heavier slice (§Rejected B). The first web-picking cut is
**server/hybrid only**, and the gap is explicit: a `client`-mode web
viewer reports `supports_picking() == False`, mirroring ADR 0042's
hybrid/explicit-gap discipline (INV-4).

### Part 4 — `ParaViewExportBackend`

A `RenderBackend` implementer that translates `SceneLayer`s to a
ParaView-consumable export — a `.vtm` / `.pvd` dataset (per-step for
animations) or a `paraview.simple` automation script — instead of
on-screen pixels. `supports_picking() == False`; `render()` /
`reset_camera()` are no-ops or flush-to-disk; `screenshot()` may render
offline via ParaView's offscreen pipeline. It is **export-only and
behind the seam** — never a hard dependency, loaded lazily like trame
(ADR 0042 R-C packaging). It validates the seam under a third backend
whose output is not a live window at all.

### Backends after R-D

| Backend | Target | `supports_picking()` | Picking face |
|---|---|---|---|
| `PyVistaQtBackend` | Native Qt desktop | `True` | Event-driven (observers); reference |
| `TrameBackend` (server/hybrid) | Web / Jupyter | `True` (R-D) | Request/response (server resolve) |
| `TrameBackend` (client-only) | Web / Jupyter | `False` | View-only (vtk.js client picking = later slice) |
| `ParaViewExportBackend` | Power-user export / automation | `False` | None (export-only) |

### Phased adoption (parity-gated, mirrors ADR 0042's R-A→R-C staging)

| Phase | Scope | Effect |
|---|---|---|
| **R-D.1** | Define the pick IR (`PickMode` / `PickRequest` / `PickHit`) + `PickBackend` Protocol in `scene_ir`. Extend the INV-1 AST guard (`tests/test_scene_ir_pure.py`) — `scene_ir` still imports no vtk/pyvista. | No behaviour change. New value types + Protocol only. |
| **R-D.2** | Extract the desktop pick engines behind `PickBackend`: the `vtkCellPicker` / projection / observer-install core moves *into* the backend; mode routing, registry resolution, highlight, the `PickMode` allow-list, and `Alt`-pick-through stay in the domain layer (INV-3). | Parity-gated, zero behaviour change. **Side benefit:** screen→scene resolution becomes headlessly testable by asserting on `PickHit` for a synthetic `PickRequest` — picking logic currently needs a live interactor. |
| **R-D.3** | `TrameBackend` (server/hybrid) picking: server-side `resolve_pick` wired to forwarded browser click events; read-out in `show_web` / `serve_web`. | Web picking delivered for server/hybrid. `client`-only stays `supports_picking() == False` (INV-4). |
| **R-D.4** | `ParaViewExportBackend` (export-only). | Validates the seam under a non-interactive backend; ships ParaView export as a power-user feature. |

Each slice is one small PR off `main`, independently revertible, behind
the existing `tests/viewers/` suite. R-D.3's *pixels* and the web
read-out need a browser eyeball; R-D.1/R-D.2/R-D.4 are fully
headless-verifiable (IR assertions, `PickHit` parity, export-file
shape).

## Invariants

- **INV-1** — Picking is an **optional capability**, gated by
  `RenderBackend.supports_picking()`. `PickBackend` is a *separate*
  Protocol, never folded onto the base `RenderBackend`. A view-only
  backend (trame client-only, `ParaViewExportBackend`) exposes no
  `PickBackend` and is legal. This preserves ADR 0042 INV-3 verbatim —
  R-D consumes that decision, does not revisit it.
- **INV-2** — The pick IR (`PickRequest` / `PickHit` / `PickMode`)
  imports **neither `vtk` nor `pyvista`**. Screen coordinates in, FEM
  ids + world point out. Enforced by the same AST guard that holds the
  render IR pure (`tests/test_scene_ir_pure.py`), extended to the new
  types. Mirrors ADR 0042 INV-1.
- **INV-3** — Picking **semantics** live above the backend. The backend
  resolves only screen→scene geometry (`vtkCellPicker.Pick` →
  `cell_id` + world point, or a projected box mask). Mode routing
  (node / element / gp / fiber), `EntityRegistry` / `cell_to_element_id`
  resolution, the `PickMode` allow-list, `Alt`-pick-through, and
  highlight overlays stay in `viewers/core/` + the viewers' domain
  layer — exactly as ADR 0042 INV-5 kept *visibility* semantics in the
  IR and out of backend tricks. The clean line: `vtkCellPicker` is the
  backend's; `resolve_pick`'s entity *meaning* is the domain's.
- **INV-4** — Web picking is **server-side** in the first cut (it needs
  the server render window to ray-cast). `client`-only render mode is
  view-only and reports `supports_picking() == False` — an explicit,
  not silent, gap. vtk.js client-side picking is a deliberate later
  slice. Mirrors ADR 0042's explicit-hybrid-gap discipline.
- **INV-5** — `ParaViewExportBackend` is **export-only**:
  `supports_picking() == False`, no interactive loop, behind the seam,
  never a hard dependency (lazily imported like trame). Re-affirms ADR
  0042 §Rejected B — ParaView is an adapter, not the engine.
- **INV-6** — `PyVistaQtBackend` remains the **reference** picking
  backend. R-D.2 is parity-gated: the extracted `PickBackend` resolves
  the *same* `PickHit` for the same `(PickRequest, scene)` as the
  current engines. The equivalence oracle is the `PickHit`, not pixels
  — the same IR-as-oracle rule ADR 0042 INV-4 set for rendering.
- **INV-7** — The pick IR **widens additively** (hover channel,
  multi-hit ray, per-vertex pick) the way ADR 0042 INV-6 widened the
  render IR — a new field / type, never a VTK object across the seam.

## Rejected alternatives

### A — Put picking back on the base `RenderBackend` Protocol

Add `resolve_pick` / `install` directly to `RenderBackend`. Rejected:
it forces every view-only backend (`ParaViewExportBackend`, trame
client-only) to implement a dead no-op, and it re-opens a decision ADR
0042 made deliberately (INV-3 / the `supports_picking()` probe). The
capability-probe split is the established pattern; R-D should *use* it,
not undo it.

### B — Client-side (vtk.js) web picking from the start

Do web picking in the browser via vtk.js's local pickers, with a
JS→Python callback bridge, instead of a server-side resolve. Attractive
because it keeps the snappy `client` render mode pickable. Rejected
**for the first cut**: vtk.js picking requires the full geometry
client-side plus a wslink callback bridge back to Python to resolve FEM
ids — materially more surface than reusing the server's existing
`vtkCellPicker`. The server/hybrid round-trip ships web picking sooner
and reuses the desktop resolution path. Client picking stays a
deliberate later slice (INV-4); it is not foreclosed.

### C — Keep the two pick engines, no `PickBackend` Protocol

Leave `core/pick_engine.py` and `core/results_pick.py` as-is and just
add a bespoke web pick path. Rejected: it perpetuates the
VTK-direct-call scatter (the very thing ADR 0042's render seam
eliminated for drawing) and gives the web path no shared resolution
core — a third hand-rolled `vtkCellPicker` site. The Protocol is the
chance to share the screen→scene core across desktop + web. *Caveat
honoured by INV-3:* the two engines resolve to **different** entity
vocabularies (mesh `(dim, tag)` vs. results `element_id` / `gp`); R-D
shares the *geometric* core, not the domain routing — it does not
force-merge them.

### D — ParaView as the interactive web backend instead of trame

Use ParaView Web / `pvserver` for the browser viewer rather than
trame. Rejected: re-litigates ADR 0042 §Rejected B. ParaView has no
concept of the OpenSees domain logic; trame is already shipped (R-C)
and drives the same `pyvista.Plotter` the desktop uses. ParaView's
role is **offline export/automation**, not the live web engine.

### E — Ship picking inside R-C, no separate ADR

Fold picking into the R-C trame work. Rejected: that is exactly what
ADR 0042 deferred, and why — picking is a new Protocol with a
desktop/web asymmetry, i.e. an architecture event (ADR 0024/0025/0026
precedent). It earns its own record.

## Consequences

**Positive:**

- Closes the last interactivity gap in the web viewer (server/hybrid):
  click-to-read-displacement, box-select, gp/fiber selection on the web.
- Brings the **last VTK-bound domain surface** behind the seam. After
  R-D, no `viewers/` domain code calls `vtkCellPicker` directly — the
  render *and* pick boundaries are both Protocol-mediated.
- Makes pick resolution **headlessly testable** for the first time: a
  `PickRequest` → `PickHit` assertion needs no live interactor. Today
  pick logic can only be exercised with a real Qt event loop — the same
  pain ADR 0042 R-B retired for rendering, now retired for picking.
- Delivers ParaView export as a power-user feature and proves the seam
  holds for a third, non-interactive backend.
- Symmetry: the viewer now has documented structural contracts on its
  **read** boundary (ADR 0026 `H5ModelReader`), its **render** boundary
  (ADR 0042 `RenderBackend`), and its **pick** boundary (this ADR).

**Negative:**

- Server-side web picking constrains the web render mode: picking needs
  a live server render window, so a pickable web viewer cannot be
  `client`-only. The snappy default and pickability pull in opposite
  directions — `hybrid` is the reconciling mode, at its known
  complexity cost (ADR 0042 §Consequences).
- Refactoring two mature, subtle pick engines (priority-10/11 abort
  chains shared with navigation, rubber-band overlays, hidden-cell
  ghost-mask exclusion, `Alt`-pick-through) behind one Protocol without
  behaviour change is real, careful work. Mitigated by R-D.2 being
  parity-gated on the `PickHit` oracle and shipping ahead of any web
  work.
- `client`-only web picking remains unavailable after R-D — a
  documented gap (INV-4), not a silent one. The notebook user who wants
  both snappy interaction *and* picking must choose `hybrid`.
- One more optional backend dependency surface (ParaView), lazily
  loaded; mitigated by INV-5 (export-only, behind the seam, never base).

## Resolved decisions

Drafted open; all four chosen at ratification (2026-05-30) after the
discovery pass over the two live pick engines. Each took its draft lean,
with one sharpened by what discovery found.

1. **`PickBackend` attachment point → keep both faces.** The backend
   exposes `resolve_pick` (stateless core, reused verbatim by the web
   face in R-D.3) *and* `install`/`uninstall` (the desktop event face
   that runs the observer state machine). Desktop observer priorities
   (the 10/11 abort chain shared with navigation) are load-bearing —
   discovery confirmed both engines depend on them — and are preserved
   untouched, not migrated to polling.
2. **Types live in `scene_ir` → yes, promoted.** `PickMode` /
   `PickModifiers` / `PickRequest` / `PickHit` / `BoxGesture` are in
   `scene_ir/_pick.py`. `core/results_pick.PickResult` / `BoxPickResult`
   are re-expressed in terms of them during R-D.2 (one shared vocabulary
   across both backends + the future web shell). `PickMode` subclasses
   `str` so the results controller's existing `"node"`/`"element"`/`"gp"`
   strings compare equal through the migration.
3. **First web-picking render mode → server/hybrid only** (INV-4 /
   §Rejected B). Deferred to R-D.3; client (vtk.js) picking is a later
   slice. Out of scope for this land.
4. **One `PickBackend`, shared geometric core, two domain consumers.**
   The mesh viewer resolves the `PickHit` to `(dim, tag)`; the results
   viewer to `element_id` / `gp`. They share only the geometric core —
   and discovery sharpened *how*: the shared key is the `prop_id` integer
   (`id(actor)`), which **both** registries already index on, so neither
   the geometry nor a VTK object is duplicated and the entity routing is
   **not** force-merged (INV-3 / §Rejected C caveat).

Two refinements to the draft IR fell out of the same pass and are now
in the ratified contract (§Decision Part 1): `PickHit` carries a
`prop_id` registry key (the draft had no prop field — resolution would
have been impossible), and box picking is a shared *projection primitive*
plus a domain-side in-box test rather than `PickHit.entity_ids` (which
would have leaked domain resolution into the IR).

## References

- [ADR 0042](0042-render-backend-seam.md) — `SceneLayer` IR +
  `RenderBackend` Protocol. This ADR is Phase R-D: it fills the picking
  capability ADR 0042 deferred and consumes its `supports_picking()`
  probe. `backends/pyvista_qt.py:412` (`True`) and `backends/trame.py:37`
  (`False`) are the live capability gates R-D builds on.
- [ADR 0026](0026-h5modelreader-protocol-contract.md) — the
  `H5ModelReader` read-side Protocol; the structural-`Protocol` +
  AST-guard + capability-probe (`has_opensees_orientation`) pattern this
  ADR re-uses on the pick boundary.
- [ADR 0024](0024-emitter-protocol-widen-region.md) /
  [ADR 0025](0025-emitter-protocol-widen-eigen.md) — Protocol
  introduction/widening as an architecture event; the precedent for
  treating `PickBackend` as ADR-worthy and for INV-7's additive-widening
  rule.
- [ADR 0027](0027-cross-partition-mp-constraints.md) — cross-partition
  partition colouring. A pick must resolve a hit to the *right* per-rank
  entity; the `PickHit` → FEM-id mapping (INV-3, domain side) honours
  the same per-entity identity the `per_entity_rgb` colour mode does.
- `viewers/core/pick_engine.py`, `viewers/core/results_pick.py`,
  `viewers/core/results_pick_engine.py` — the two desktop pick engines
  R-D.2 extracts behind `PickBackend`.
- `tests/test_scene_ir_pure.py` — the INV-1 AST guard R-D.1 extends to
  the pick IR.
- ADR 0042 §Rejected B — the ParaView-as-future-export-adapter
  assessment Part 4 / INV-5 implement.
