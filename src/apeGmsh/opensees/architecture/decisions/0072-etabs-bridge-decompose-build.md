# ADR 0072 — Decompose `build_opensees`: the ETABS bridge hands off labeled geometry; apeSees is the modeling-control surface

**Status:** Proposed (2026-06-25). Interop-side refactor
(`apeGmsh.interop.etabs_import`) over the existing `apeSees` bridge — no
Emitter Protocol change, no schema change, byte-identical default output.
Successor framing to the ADR 0009 (apeETABS) round-trip; sibling of
ADR 0051 (bridge load consumption).

## Context

The ETABS → apeGmsh interop turns a neutral `StructuralModel` (`.sm.json`)
into a solving OpenSees deck through one call:

```python
ops = build_opensees(fem, model, result)   # the WHOLE elastic deck, atomically
```

`build_opensees` hardcodes a fixed elastic idealization: `elasticBeamColumn`
for every frame, `ElasticMembranePlateSection` + `ASDShellT3` for every area,
isotropic materials from `(E, ν)`, plus the support fixities, grounded springs,
loads, self-mass, and rigid diaphragms. It is the right *convenience* — a
one-liner that solves — and the solve cross-check (ADR 0009, `live_compare.py`)
leans on it.

But the cross-check also clarified what the import **is**. On Casa 17B the
reactions matched while the displacement field diverged, and digging in showed
the divergence is modeling (mesh density, stiffness), not a bug. The real
lesson is upstream of any single number:

> **ETABS is a regenerable *scaffold*, not the target model.** The engineer's
> model = the scaffold (geometry, loads, masses, supports) **plus their own
> modeling decisions** (fiber columns, plastic-damage concrete, force-based
> elements, custom walls, cracked-section modifiers, refined mesh). The elastic
> deck `build_opensees` emits is one profile over the scaffold; the research
> model is another.

apeGmsh **already is** the environment for those decisions: `apeSees` has typed
materials, fiber sections, force/displacement beam-columns, layered shells,
constraints, staged analysis. The neutral document deliberately preserves the
ETABS **section names**, **material names**, and **joint ids** as keys, and the
importer already builds physical groups named by section (`"COL400"`,
`"SLAB200"`, with a `__v`/`__h` orientation suffix when a section spans both)
plus restraint / load / diaphragm PGs. That is exactly the handle `pg=`
selectors consume. The handoff contract the user needs already exists — it is
**labeled geometry**.

So the temptation — build a declarative `ModelProfile` / overlay DSL inside the
interop that maps section names to elements and materials — is **re-implementing
apeGmsh's modeling surface behind a second, weaker grammar**. Rejected (see
Alternatives).

The one thing actually blocking "import the base, then build the model I want"
is that `build_opensees` is **all-or-nothing**. The user can *add* to the deck
after it (the `apeSees` object is returned), but cannot *replace* its defaults
for a subset — there is no way to say "your elastic deck for everything **except**
`COL400` and `W1`, which I will model myself." Any customization fights the
monolith or hand-edits the generated `.py`/`.tcl` (which dies on the next
ETABS re-export).

## Decision

### 1. The seam: *mechanical* bridge-owned vs *modeling* user-owned

Split the deck's contents by who decides them:

- **Mechanical** (objective consequences of the ETABS model, not modeling
  choices): support fixities, grounded springs, applied loads, self-mass,
  rigid diaphragms. The bridge owns these.
- **Modeling** (where the engineering judgement lives): which element, section,
  and material each group gets. The bridge provides only a documented **elastic
  default**; the user owns the rest via native `apeSees`.

The contract between them is the **labeled geometry** the importer already
produces — PGs named by ETABS section (+ orientation), restraint/load/diaphragm
PGs, and joint ids preserved on `FEMData`. `ImportResult` enumerates the handles
(`frame_groups`, `area_groups`, `restraint_groups`, `load_patterns`,
`diaphragms`, `spring_grounds`, `subgrade_pgs`).

### 2. Decompose `build_opensees` into public, composable emit steps

Each current phase becomes a public function operating on `(ops, fem, model,
result)`; none is privileged:

```python
inject_diaphragms(fem, result)                 # fem-mutation; BEFORE apeSees(fem)
ops = apeSees(fem); ops.model(ndm=3, ndf=6)
emit_elements(ops, fem, model, result, *, skip=(), only=None,
              shell_element="ASDShellT3")       # the MODELING default (elastic)
emit_supports(ops, fem, result)                # fixities + grounded springs
emit_loads(ops, result)                        # load patterns (from_model)
emit_mass(ops, model, result)                  # self-mass
```

`inject_diaphragms` stays a `fem`-mutation (rigid-diaphragm records must land on
the snapshot before `apeSees(fem)` reads it — unchanged from today's private
`_inject_diaphragms`), surfaced as public because the composing user calls it.

### 3. `emit_elements(skip=/only=)` — the opt-out that unlocks modeling

`emit_elements` accepts a **section-name** set:

- `skip={"COL400", "W1"}` — emit the elastic default for every section *except*
  these; the user models them with native `apeSees` against the same PGs.
- `only={"BEAM300"}` — the inverse, for incremental adoption.

```python
inject_diaphragms(fem, result)
ops = apeSees(fem); ops.model(ndm=3, ndf=6)

emit_elements(ops, fem, model, result, skip={"COL400", "W1"})   # elastic for the rest
emit_supports(ops, fem, result)
emit_loads(ops, result)
emit_mass(ops, model, result)

# my model, native apeSees, targeting the bridge's PGs:
conc = ops.nDMaterial.PlasticDamageConcrete3d(...)
ops.element.forceBeamColumn(pg="COL400", section=my_fiber, transf=..., integration=...)
ops.element.MVLEM(pg="W1", ...)
```

No new grammar: control *is* `apeSees` with `pg=` selectors. The bridge stepped
out of the way for two groups; it still owns the mechanical deck and the
elastic default for everything else.

### 4. `build_opensees` becomes a thin wrapper — byte-identical default

```python
def build_opensees(fem, model, result, *, ndm=3, ndf=6, shell_element="ASDShellT3"):
    inject_diaphragms(fem, result)
    ops = apeSees(fem); ops.model(ndm=ndm, ndf=ndf)
    emit_elements(ops, fem, model, result, shell_element=shell_element)  # skip empty
    emit_supports(ops, fem, result)
    emit_mass(ops, model, result)
    emit_loads(ops, result)
    return ops
```

Empty `skip` ⇒ today's output, line-for-line. The cross-check, the runnable-deck
tests, and every quick run are unchanged. The decomposition is purely additive.

### 5. Mesh + constraints already have a seam — document, don't build

Geometry-level control (per-section mesh size, extra `g.constraints.*`,
embedded/decoupled nodes, added loads) is **already available**: it is the
native `g` API, between `import_structural_model` and `get_fem_data`. This ADR
does not add a mesh/constraint surface; it documents the existing seam in the
worked example so "use apeGmsh" is the explicit answer there too.

### 6. Stiffness modifiers are *faithfulness*, out of scope here

The cross-check's displacement gap points at ETABS property modifiers (cracked-
section `f`/`m` factors) that the `.sm.json` does not carry. That is a **base-
fidelity** concern, not modeling control: the fix is to *export* the modifiers
(apeETABS side, a `modifiers` field on sections) and have the **elastic default**
apply them — so the scaffold itself matches ETABS. It is an orthogonal
workstream, called out here only to keep it out of the decomposition's scope.

## Rationale

- **Do not rebuild apeGmsh.** The modeling surface (materials/sections/elements/
  constraints/staged analysis) exists and is typed, tested, and documented. The
  interop's job ends at handing off labeled geometry; re-expressing modeling
  behind an overlay grammar would be a second, weaker, perpetually-trailing API.
- **The contract already exists.** Section/material names and joint ids are
  preserved end-to-end *by design* (ADR 0009 traceability keys), and the
  importer already groups by them. `skip=`/`only=` just stops the monolith from
  claiming groups the user wants — minimal surface, maximal reuse.
- **The mechanical/modeling split is the natural seam.** Fixities, springs,
  loads, mass, diaphragms are determined by the ETABS model; elements and
  materials are where judgement enters. Bridge owns the former, user owns the
  latter — and the elastic default is a *starting point*, not a wall.
- **Survives re-export.** Decisions live in the user's `apeSees` script keyed on
  stable names; the base regenerates from ETABS underneath. New members of a
  known section inherit the user's element automatically; a new section falls to
  the elastic default. Nothing is hand-edited in the deck.
- **Backward-compatible by construction.** The wrapper preserves byte-identical
  output, so nothing downstream moves.

**Alternatives rejected:**

- *A declarative `ModelProfile` / overlay DSL* (section-name → element/material/
  modifier spec, serialized) — re-implements apeGmsh modeling behind a second
  grammar that will always trail the real API; turns "use the element you want"
  into "wait for the spec to support it." The hybrid-with-hooks variant still
  carries the declarative core's maintenance and the conceptual second surface.
  Composition over a framework.
- *Hooks/callbacks into the monolith* (`build_opensees(frame_element=lambda …)`)
  — scatters modeling across callback signatures, forces the user to learn build
  internals, and is not a durable artifact. `skip=` + native `apeSees` is
  strictly simpler and more powerful.
- *Hand-edit the generated deck* — lost on every ETABS re-export; the failure
  mode this whole ADR exists to prevent.
- *Post-process only* (add after `build_opensees`) — can *add* elements but
  cannot *replace* the defaults the monolith already emitted for a group, so
  the elastic and custom elements would double up on the same PG.

## Consequences

- New public functions in `apeGmsh.interop`: `inject_diaphragms`,
  `emit_elements`, `emit_supports`, `emit_loads`, `emit_mass` (the current
  private bodies, lifted and given `skip=`/`only=`). `build_opensees` and
  `apply_subgrade_springs` keep their signatures.
- **No schema change, no Emitter Protocol change, no `FEMData` change.** Pure
  interop reorganization over existing bridge verbs (the ADR 0059 shape).
- `emit_elements` must resolve a **section name → its PG(s)** (the `__v`/`__h`
  split means one section can own two groups); `ImportResult.frame_groups`/
  `area_groups` already carry `(pg, section, orient)`, so `skip` filters on
  `group.section`.
- Docs: a new worked example ("ETABS base, then override two sections with
  nonlinear apeSees") in the interop guide + the skill cheatsheet; the example
  also documents the §5 mesh/constraint seam. `build_opensees` is presented as
  the convenience, the explicit steps as the control path — neither deprecated.
- `live_compare.py` / the cross-check continue to call `build_opensees`
  unchanged (they want the faithful elastic profile).

## Open questions

1. **Where do the steps live** — module-level functions in `etabs_import`
   (proposed, mirrors today's private functions) vs methods on a small
   `Bridge`/deck object returned by the importer? Lean: functions — no new
   object, matches the existing `build_opensees(fem, model, result)` shape.
2. **`skip`/`only` granularity** — section name only (proposed) vs also by
   `kind` (`skip_kinds={"shell"}`) vs per-instance id. Lean: section + a kind
   convenience; per-instance deferred (ids are the least stable key across
   ETABS edits, and per-section covers the overwhelming majority).
3. **Do `emit_supports`/`emit_loads`/`emit_mass` need `skip` too?** They are
   mechanical, but a user replacing a support region (e.g. a custom SSI macro at
   some joints) might want to opt those joints out. Lean: add `skip=` to
   `emit_supports` only (springs/fixities are the realistic override target),
   keep loads/mass whole.
4. **`emit_elements` return value** — nothing (mutate `ops`) vs a small report
   of `{section: pg(s) emitted / skipped}` for a `describe()`-style trust print.
   Lean: return the report; it is the cheap analogue of the rejected profile's
   one genuinely useful feature (seeing the plan before solving).
5. **Should the worked example live as a runnable `examples/` script** (like
   `live_compare.py`) so it is CI-exercised, not just prose? Lean: yes.

## Related

- `src/apeGmsh/interop/etabs_import.py` — `build_opensees`, the private
  `_inject_diaphragms` / `_emit_springs`, `ImportResult`, the PG naming in
  `import_structural_model` (§ the `<section>__<orient>` split).
- ADR 0009 (apeETABS) — the round-trip + traceability-key contract this builds
  on; `scripts/live_compare.py` solve cross-check that surfaced the scaffold
  framing.
- ADR 0051 — bridge load consumption: loads are opt-in (`from_model`), the same
  "bridge offers, user composes" philosophy applied to load cases.
- ADR 0059 — mesh recipes: the precedent for a pure-orchestration tier over
  existing verbs with no schema change and escape hatches preserved.
- Orthogonal workstream (separate ADR/PR): export ETABS stiffness modifiers into
  the `.sm.json` so the elastic default matches ETABS (§6).
