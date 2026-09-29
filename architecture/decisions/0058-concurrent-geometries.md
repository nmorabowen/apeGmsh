# ADR 0058 — Concurrent geometries: a Geometry is a scene instance, deform-follow becomes universal

**Status:** Accepted (2026-06-12; sequenced S0–S4 all shipped —
S0 #623, S1 #625, S2a #629, S2b #636, S2c #647, S3a #649, S3b #650,
S3c #651, S4 retires `DeformedShapeDiagram` and makes the deform-follow
contract unconditional. Builds on the deform-follow contract guard
shipped in PR #620 and the event contract of ADR 0056.)

## Context

### What a "Geometry" is today: an exclusive view preset

The results viewer's organizing hierarchy is

```
GeometryManager            ← director.geometries
└── Geometry               ← outline first level
    └── CompositionManager
        └── Composition    ← outline second level
            └── Diagram    ← layer (Contour, LineForce, VectorGlyph, …)
```

A `Geometry` (`viewers/diagrams/_geometries.py`) carries deformation
state (`deform_enabled` / `deform_field` / `deform_scale`) and substrate
display state (`show_mesh` / `show_nodes` / `display_opacity`) plus its
compositions. But **the viewport renders only the active geometry** —
its own module docstring says so. Multiple geometries exist as
*switchable presets*, not as concurrent scene objects: switching the
active geometry re-applies that geometry's deformation to the one
shared substrate.

That substrate is a single `FEMSceneData` (`viewers/scene/fem_scene.py`)
whose `grid.points` the DEFORM pump mutates in place
(`results_viewer.py:_pump_deform`), then fans out to every diagram
through exactly one hook:

```python
Diagram.sync_substrate_points(deformed_pts, scene)
```

Post-ADR-0042 this hook is the **only** deformation fan-out, and since
PR #620 it is machine-enforced: `tests/viewers/test_deform_follow_contract.py`
walks every `Diagram` subclass and fails if a rendering diagram inherits
the base no-op. The hook already takes `scene` as a *parameter* — no
diagram reaches for a substrate singleton.

### The three frictions

**1. No side-by-side states.** The one-active-geometry model cannot
show two configurations at once: deformed vs undeformed reference,
stage A vs stage B of a staged model (ADR 0055), mode 1 vs mode 2, the
same state from two load cases. Every comparison is a toggle, never a
view.

**2. Two deformation mechanisms — and the contract exemption that
proves it.** There are two ways to "see deformation":

* *Geometry-level deform* — the substrate warps; every diagram follows
  via the sync hook.
* *`DeformedShapeDiagram`* — a layer that warps **its own copy** of the
  substrate while the shared substrate stays at reference.

`DeformedShapeDiagram` is the **sole exemption** in the contract-guard
test ("renders its own warp; following the global substrate would
double-deform"). The exemption is not an implementation wart — it is
the symptom of a missing capability. "Deformed overlay on top of the
reference mesh" is something users genuinely want, and the exclusive
preset model can only express it as a special diagram that smuggles a
second configuration into the scene.

**3. Conceptual / extension friction.** The vocabulary is split — code
says Geometry / Composition / Diagram, the UI labels a Composition
"Diagram" and a Diagram "layer" — so every conversation about the
viewer needs a translation table. And adding a new diagram kind
currently touches ~6 places: the class file, the `_KINDS` tuple in
`ui/_add_diagram_dialog.py`, a style class in `_styles.py`, package
exports, topology wiring, and tests.

### Why the timing is right

PR #620's guard is exactly the safety net this change needs: every
rendering diagram has *proven* it can re-position itself against any
substrate it is handed. The hook signature is already
scene-parameterized. Diagrams already belong to exactly one geometry
(via its compositions; `GeometryManager.geometry_for_layer`). The
dispatcher (ADR 0056) already has granular `geometry_*` rows carrying a
`geom_id` payload — the pumps just ignore the scope today. The distance
from "exclusive preset" to "scene instance" is an argument change, not
a redesign.

## Decision

Six parts. Parts 1–3 are the model; Parts 4–6 are the policies that
make it shippable.

### Part 1 — A Geometry is a scene instance

Each `Geometry` owns:

* **its own `FEMSceneData`** — a substrate grid copy built once from
  the shared `ViewerData`;
* its existing deformation state (`deform_enabled` / `deform_field` /
  `deform_scale`) and display state (`show_mesh` / `show_nodes` /
  `display_opacity`);
* **new** — a `visible: bool` flag (rendered or not, independent of
  which geometry is being edited);
* **new** — an optional spatial `offset` (3-vector, default zero),
  applied as a rigid translation after the warp, so two instances can
  sit beside each other;
* **new** — an optional `stage_id` pin (`None` = follow the active
  stage), mirroring `DiagramSpec.stage_id`. A geometry pinned to a
  stage reads its deformation field from that stage; combined with
  per-diagram stage pinning this is what makes *stage A vs stage B
  side-by-side* expressible.

All **visible** geometries render concurrently. "Active" is demoted
from *the one that renders* to *the one the settings panels edit and
the default target for Add Diagram* — a selection concept, never a
render gate.

The **time cursor stays director-global**: one step index for the whole
viewport. Per-geometry step cursors are an explicit non-goal (one time
axis; comparing time points is a different feature with different UI
demands).

### Part 2 — The DEFORM pump becomes per-geometry

`pump_deform` iterates the visible geometries; for each one it computes
that geometry's `deformed_pts` (field × scale at the global step, read
from the geometry's pinned-or-active stage, plus offset), mutates **that
geometry's** `scene.grid.points`, and fans out to **that geometry's
diagrams only**:

```python
for geom in visible_geometries:
    pts = compute_deformed_pts(geom, step)        # None = reference (+ offset)
    geom.scene.grid.points = pts if pts is not None else geom.reference_pts
    for d in diagrams_of(geom):
        d.sync_substrate_points(pts, geom.scene)
```

The `deformed_pts` semantics are unchanged from the PR #620 docstring
(row-aligned with `scene.node_ids`; `None` = reference). The granular
dispatcher rows (`geometry_deform_changed` with `geom_id` payload,
ADR 0056) stop being advisory: the pump honors the scope and re-pumps
one geometry instead of all of them.

### Part 3 — Deform-follow becomes universal; `DeformedShapeDiagram` retires

With concurrent instances, "deformed shape over the reference mesh" is
just **two geometries** — one deform-on, one deform-off dimmed
wireframe. Therefore:

* `DeformedShapeDiagram` is retired. The Add-Diagram kind becomes sugar
  that creates a new geometry with deform enabled; session files
  containing the old kind migrate the same way on load.
* A **reference-ghost preset** ("Add reference ghost") is the inverse
  sugar: duplicate the geometry, deform off, wireframe, dimmed.
* The exemption list in `test_deform_follow_contract.py` empties and is
  **deleted**. The contract becomes unconditional: *every* rendering
  diagram overrides `sync_substrate_points`, no exemption mechanism
  exists to grow back.

### Part 4 — Declarative diagram registration

One registry (extend `viewers/diagrams/_kind_catalog.py`) becomes the
single source of truth for diagram kinds. An entry declares:

```
kind_id, ui_label, diagram_class, style_factory, topology
```

Consumers: the Add-Diagram dialog (the hand-maintained `_KINDS` tuple
is deleted), the kind-availability catalog, the session loader, and the
contract-guard test's discovery walk. Adding a diagram kind drops from
~6 touch points to 2: the class file (which registers itself) and its
tests.

### Part 5 — Concurrency policies

* **Scalar bars** stay per-diagram (existing `_scalar_bar_support`
  path, `ScalarBarSpec` through the backend). When more than one
  geometry is visible, bar titles are prefixed with the geometry name
  so two contour bars are distinguishable. No shared-range coupling
  across geometries in v1 — two instances of the same field may show
  different ranges, exactly as two separate diagrams do today.
* **Picking** maps the hit actor → owning geometry → that geometry's
  scene for cell→element / node resolution. The pick IR (ADR 0045/0047)
  gains an additive `geometry_id` field so pick reporting and the
  selection log stay unambiguous.
* **Memory**: plain per-geometry grid copies first — simplicity over
  speculation. S1 below measures the real cost on a representative
  solid model. If N× `grid.points`+cells copies bite, the escape hatch
  is copy-on-write (geometries at reference with zero offset share the
  reference grid; the first deform/offset materializes the copy) — an
  internal optimization that changes no contract.

### Part 6 — Vocabulary: the UI adopts the code names

Outline levels read **Geometry → Composition → Diagram**. UI strings
that label a Composition "Diagram" change to "Composition"; strings
that call a Diagram a "layer" change to "diagram". Code identifiers do
not churn — the documented, tested names win and the UI is the cheap
side to move.

## Adoption slices

* **S0 — Declarative registration** (Part 4). Standalone value, no
  behavior change; shrinks the surface every later slice touches.
* **S1 — Scene-per-geometry plumbing, active-only rendering.** Pure
  refactor: `FEMSceneData` instantiated per geometry, attach/pump paths
  resolve a diagram's scene through its owning geometry, viewport still
  renders only the active one. Memory measurement happens here.
* **S2 — Concurrent rendering.** The `visible` flag is honored by the
  pumps and the gate (geometry ∧ composition ∧ layer visibility); the
  outline eye on geometry rows becomes the flag's owner (ADR 0056
  single-owner rule); scalar-bar title prefixes; pick IR `geometry_id`.
* **S3 — Offsets, per-geometry stage pin, reference-ghost preset.**
  Also upgrades `GeometryManager.duplicate()` from "deform state only"
  to duplicate-with-layers (rebuild diagrams from their serializable
  `DiagramSpec`s — the same mechanism session restore already uses).
* **S4 — Retire `DeformedShapeDiagram`.** Kind becomes
  create-geometry sugar; session migration; exemption list deleted;
  contract universal.

Each slice lands independently green; S2 is the first user-visible
change.

## Consequences

**Gains.** Side-by-side comparison (deformed/reference, stage A/B,
mode shapes, load cases) becomes a first-class view instead of a
toggle. The deformation model collapses to one mechanism with zero
exemptions — the PR #620 contract becomes the *whole* story. The
hierarchy's names become honest ("Geometry" finally denotes a geometry
in the scene). New diagram kinds cost 2 touch points. The reference
ghost — the most common FEM-viewer ask — is a preset, not a special
diagram.

**Costs.** Memory scales with visible geometries (bounded, measured at
S1, COW escape hatch reserved). DEFORM pump cost scales with visible
geometries — each iteration is the same O(N) resample that runs today.
The gate pump composes a third visibility term. Session schema gains
additive fields (`visible`, `offset`, `stage_id`); **old sessions load
with `visible = (geometry is active)`**, reproducing their previous
rendering exactly. The pick IR widens additively (ADR 0047's
established pattern).

**Relations.** ADR 0042 is untouched — diagrams still emit IR through
the backend seam, and per-geometry scenes ride entirely on the domain
side. ADR 0056 extends naturally: `visible` gets a single owner
(`GeometryManager`) and owner-fired events; the matrix gains a
`geometry_visibility_changed` row. ADR 0055's stage records combine
with the per-geometry stage pin to deliver construction-sequence
side-by-side. The mesh viewer is explicitly out of scope — no parity
rider.

**Rejected alternative: backend-side warp (pull model).** Keeping the
substrate at reference and warping downstream (VTK warp-by-vector /
shader) was considered and rejected: it only works for layers whose
geometry is a pure translation of substrate points. Line-force
diagrams rebuild local axes from the deformed chord; fiber clouds and
glyph anchors recompute their geometry *from* the deformed
configuration. A downstream filter cannot express that, so the hook
survives regardless — and a hybrid would split the contract PR #620
just unified into two diagram classes with different rules.

## Open questions resolved

1. **Scalar-bar policy across instances** → per-diagram bars (status
   quo), geometry-name title prefix when >1 geometry is visible; no
   cross-geometry range coupling in v1.
2. **What does "active" mean once everything renders?** → editing
   target only (panels + Add Diagram default). Render participation is
   `visible`, owned by `GeometryManager`, shown as the outline eye.
3. **Memory strategy** → plain copies first, measure at S1, COW as the
   reserved escape hatch.
4. **Vocabulary direction** → UI adopts code names
   (Geometry / Composition / Diagram); no code-identifier churn.
5. **Per-geometry time step?** → No. One global time cursor; comparing
   time points is out of scope for this ADR.
