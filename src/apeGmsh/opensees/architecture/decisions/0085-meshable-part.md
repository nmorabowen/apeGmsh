# ADR 0085 — `Part` stays geometry-only; independently-meshed parts are the compose seam, and `Assembly` learns to `tie`

**Status:** Proposed (2026-08-04) — awaiting sign-off on the chosen
option before implementation.

The question posed: `Part` is "the abstraction literally named for
this", yet it cannot mesh, cannot `get_fem_data`, cannot save a
`model.h5` — so any model that needs per-part element order (hex20
ribs next to hex8 covers) abandons `Part` entirely and hand-rolls one
full `apeGmsh` session per part. Should `Part` become a first-class
independently-meshable module?

**The answer this ADR defends: no.** The capability gap is real, but
it is not `Part`'s gap to close — it was already closed, deliberately,
by `g.compose` (ADR 0038) and its declarative wrapper
`apeGmsh.assembly.Assembly` (ADR 0043 slice 1.4). What is actually
missing is smaller and cheaper: the `Assembly` helper cannot declare
the one interface kind the production model needs (`tie` with
`enforce="equation"`), the workflow is not documented as *the* route
for mixed-order parts, and `Part`'s docstring lies about what `Part`
is. This ADR fixes those three things and leaves `Part`'s contract
alone.

## Context

### What `Part` genuinely supports today (vs. its docstring)

Verified against the source, not the docstring — the docstring is
stale twice over:

* `Part._COMPOSITES` ([core/Part.py:130-137](../../../core/Part.py))
  is `model`, `labels`, `physical`, `inspect`, `plot`, `edit`. The
  module docstring ("no meshing, **no physical groups**, no mesh
  settings", Part.py:7-11) and the class docstring ("no meshing, no
  physical groups", Part.py:106) both deny the `physical` composite
  that has been there since `ba4c6e45` (2026-04-11, "physical groups
  as the naming system, not label_to_tag").
* `_auto_pg_from_label = True` (Part.py:158) auto-creates a PG for
  every `label=`-named entity; those names travel through the STEP
  sidecar (`{name}.step.apegmsh.json`, written by
  `_write_anchors`) and are rebound in the assembly as Tier-1 labels
  `"{instance}.{name}"` by `PartsRegistry._import_cad`
  ([core/_parts_registry.py:1439-1493](../../../core/_parts_registry.py)).
* The module docstring also still describes an "**Assembly** =
  imports Parts, positions them, …" object that was deleted in
  `f87c9584` (2026-04-09, "the session IS the assembly").

What `Part` does **not** have is equally definite: no `mesh`
composite, no `constraints`/`loads`/`displacements`/`masses`/
`sections`, no `node_ndf`, no FEMData cache (`_fem`,
`_fem_is_fresh`), no `save_to`/h5 persistence. `Part.save()` writes
CAD only (`_VALID_EXT = {.step,.stp,.iges,.igs}`, Part.py:123), and
`end()` auto-persists STEP to a tempfile because `g.parts.add(part)`
**re-imports the geometry** through OCC (`importShapes`,
_parts_registry.py:1361) — the STEP file *is* the Part's canonical
form.

### Why geometry-only is deliberate, with receipts

This is not an accident of history; it is stated policy in three
places and was **re-litigated and rejected once already**:

* **ADR 0038 §Alternatives** rejects `Part.bake()` by name: "Baking
  exposes only a subset of apeGmsh and forces callers into a
  constrained authoring model. The full apeGmsh session is the
  natural unit of saved-and-loaded; making the H5 the bake target
  lets the same `g.save()` machinery serve both straight-line
  persistence and module export." It likewise rejects a
  `g.save_as_module(...)` variant — "module-ness is a load-time
  interpretation."
* **docs/design/parts-assembly.md:35-41**: the `_COMPOSITES`
  allow-list "is the enforcement of the invariant that a Part is
  geometry only; adding a mesh composite there would quietly break
  the Part-as-CAD-artifact model, which is why contributor rule
  number one is don't."
* **docs/concepts/parts-and-assembly.md:171-192** draws the two-seam
  map explicitly: "Parts compose geometry before meshing inside one
  session; `g.compose` composes finished models across sessions."
  A Part is a geometry *template*; a module is a *finished
  sub-model*.

The root cause behind the policy (docs/design/parts-assembly.md:10-24)
is physical, not stylistic: gmsh keeps **one process-global OCC
kernel**, so a Part and an assembly cannot coexist in memory as
independent geometries — something must serialize, and apeGmsh
serializes the Part as STEP. A *meshed* part faces the same
constraint with a different payload: what compose consumes is a
`model.h5` broker snapshot, i.e. exactly what a full session's
`g.save()` already writes.

### The driver, and what it proves

The Cerro Lindo rung-4 solid fuse (5 parts, per-part element
formulation *and* order) hand-rolled the route: one full `apeGmsh`
session per part → `g.save()` each to `part_*.h5` →
`apeGmsh.from_h5(host)` → `g.compose(...)` × 4 →
`g.constraints.tie(..., enforce="equation")` × 6. It is in
production and measured (−0.01 % against a closed form on the tie).

Two details of that script are load-bearing for this decision:

1. **The host part needed `g.displacements`.** The prescribed-push
   case is declared on the topcover *part session* — deliberately,
   because a composed module's load patterns are dropped
   (ComposeFilterWarning) while host records survive. A meshable
   `Part` has no `displacements` composite, so **option (a)/(b) could
   not have authored the production host part**. Any "meshable Part"
   is a strict subset of a session and re-creates exactly the
   "constrained authoring model" ADR 0038 rejected. This is the
   decisive argument, and it is empirical, not aesthetic.
2. The script's two hand-rolled safety rails — disjoint PG tag bands
   and the "tie made 0 records" assertion — are now library
   concerns: the PG-tag collision is fixed on
   `fix/compose-pg-tag-collision` (`7b63d67a`, this branch's base),
   and the zero-record tie check is precisely what
   `Assembly.materialize` already does
   ([assembly.py:301-310](../../../assembly.py)).

### The infrastructure already built for this

`apeGmsh.assembly.Assembly` (ADR 0043 slice 1.4, PR #433, v2.0.0) is
the declarative form of the hand-rolled pattern: `.add(label, h5)`
(first = host via `from_h5`, rest via `compose`), `.couple(a, b,
kind=, ports=)` with bare per-part PG names namespaced automatically,
`.materialize()` → composed session, **fail-loud** — a couple that
produces zero constraint records raises `AssemblyError` instead of
leaving a part floating. It closes both hazards the prompt warns
about (PG survival is checked implicitly by the zero-record guard;
the caller never touches `try_chain_phase_route`'s KeyError swallow).

What it cannot do today: `_COUPLE_KINDS = ("equal_dof",
"tied_contact")` ([assembly.py:72](../../../assembly.py)). The
calibrated production interface is `g.constraints.tie(...,
enforce="equation")` — measured −0.01 % vs. penalty's −1.3 % at 1e10
and outright failure ≥ 1e12 in N/mm units. So the one helper built
for this workflow cannot express the one constraint the workflow is
calibrated on.

## Options

**(a) Give `Part` the `mesh` composite + h5 save.** Technically
feasible — `FEMData.from_gmsh` reads session composites via
`getattr(session, X, None)` throughout
([mesh/_fem_factory.py](../../../mesh/_fem_factory.py)), and
`get_fem_data`'s cache path degrades gracefully when the parent lacks
`_fem_is_fresh` (_mesh_queries.py:173). But it (i) reverses two
explicit ADR 0038 rejections with no new information, (ii) forks the
persistence contract — `end()` auto-persists STEP for `parts.add`,
so a meshed Part either auto-persists a payload that silently drops
its mesh or grows a second mode switch, (iii) still cannot author
real parts (no loads/displacements/sections — see driver detail 1),
so users would graduate to sessions anyway, now with two half-routes
to document, and (iv) breaks the docs' stated invariant and the
Abaqus-mirroring mental model (loads live on the assembly, but
Abaqus *meshes* on the part — the analogy is imperfect either way;
apeGmsh's seam is persistence, and its "meshed part" artifact is the
h5, authored by a session).

**(b) `MeshablePart` / `Part.to_module(path, mesh=...)`.** Same
subset-authoring trap as (a) minus the docstring clash. A
`to_module()` that must express recipes, order, size fields, PGs on
mesh entities, and displacement cases through keyword arguments is a
worse API than the session it wraps. Rejected for the same reason
ADR 0038 rejected `Part.bake()` — it is `Part.bake()` with a new
name.

**(c) Leave `Part` alone; make the session-per-part route
first-class.** The session already *is* the meshable part (full API,
`save_to=` autosave, proven at production scale); `compose` already
is the assembler; `Assembly` already is the declarative helper with
the fail-loud guards. The gap reduces to: `tie` support in
`Assembly`, documentation that names this route as the answer to
"different element types/orders per part", and honest docstrings.

**Decision: (c).**

## Decision

Four work items, in order:

**D1 — Fix the `Part` docstrings** (module + class,
[core/Part.py](../../../core/Part.py)). State the real composite
set (including `physical` and `_auto_pg_from_label` Tier-1 naming),
delete the ghost "Assembly" object, and add a signpost paragraph:
*"A Part is a geometry template for stamping into ONE session's
mesh. If parts need independent meshes — different element types,
sizes, or orders — the unit of authorship is a full `apeGmsh`
session per part saved to `model.h5`, assembled with `g.compose` or
`apeGmsh.assembly.Assembly`; see docs/how-to/compose-modules.md."*

**D2 — `Assembly` learns `kind="tie"`.** Add `"tie"` to
`_COUPLE_KINDS`; `materialize` already dispatches by
`getattr(g.constraints, kind)` and `tie(master_label, slave_label,
...)` accepts the same positional shape, with `enforce=` /
`stiffness=` / `tolerance=` flowing through the existing `**options`
/ `tolerance=` plumbing. The zero-record guard applies unchanged.
Docstring gains the calibration note: `enforce="equation"` is exact
but requires the Lagrange constraint handler and an unsymmetric
system; penalty at 1e18 (the default) does not converge in N/mm.

**D3 — Documentation.** In `docs/how-to/compose-modules.md` (and a
paragraph in `docs/concepts/parts-and-assembly.md`), add a "parts
with independent meshes" section: when to choose this seam (mixed
order is *impossible* in one session — `set_order` is global),
the two known hazards as rules (extract each part with
`get_fem_data(dim=None)` because the tie resolver needs dim-2
groups; always assert PG survival / rely on `Assembly`'s guard), and
the copy-paste snippet from the acceptance test below.

**D4 — Acceptance test** (`tests/test_meshable_part_route.py`):
three parts meshed independently as **hex20, tet10, hex8**, each
saved to h5; assembled via `Assembly` (one host + two composed, one
interface genuinely non-matching); asserts (i) every part's PGs
survive by explicit name, (ii) the assembled FEMData reports mixed
element types, (iii) `openseespy`-gated (`pytest.importorskip`):
two stacked blocks, `nu = 0`, tie with `enforce="equation"`,
axial stiffness within 0.1 % of the series closed form
`K = (EA/L)/2`. Runs on top of the PG-collision fix (`7b63d67a`),
which this branch is based on.

## Consequences

* `Part`'s contract, `g.parts.add`, the STEP sidecar, and the
  auto-persist lifecycle are untouched — zero blast radius on the
  single-mesh workflow and its tests.
* The mixed-order workflow becomes documented, guarded, and
  one-abstraction (`Assembly`) instead of hand-rolled; the Cerro
  Lindo script's tag-band workaround and manual assertions become
  unnecessary (the assertions remain good hygiene).
* The "Part cannot mesh" asymmetry with Abaqus remains, now stated
  honestly in the docstring instead of hidden behind a stale one. If
  a future need arises for a *geometry-template* that also carries
  meshing intent (not a meshed snapshot), that is a new decision —
  this ADR does not foreclose it, it only declines to make the
  meshed-module artifact a `Part` responsibility.
* `Assembly` inches toward ADR 0043's fuller couple vocabulary
  (`embedded` / `rigid_link` remain deferred — nothing here blocks
  them).

## Not chosen — summary table

| Option | Fatal objection |
|---|---|
| (a) `mesh` composite on `Part` | Reverses ADR 0038's explicit rejection; dual STEP/h5 persistence contract on one object; still can't author parts that need loads/BCs (production host part required `g.displacements`) |
| (b) `MeshablePart` / `to_module()` | `Part.bake()` renamed; subset-authoring API strictly worse than the session it would wrap |
| Do nothing | Workflow stays hand-rolled; `Assembly` can't express the calibrated `tie`; docstring keeps lying |
