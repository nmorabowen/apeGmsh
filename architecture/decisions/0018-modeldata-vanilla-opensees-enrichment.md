# ADR 0018 — `ModelData` — declarative `model.h5` enrichment for hand-written OpenSees

**Status:** Accepted (May 2026). Complements ADR 0011; does not supersede it.
**Amended (May 2026):** scope widened from orientation-only to
orientation **plus recorders** — see [Amendment — recorders](#amendment--recorders-may-2026).

## Context

ADR 0011 made `model.h5` the model-definition archive and tabulated
the data that lives **only** in the bridge — including, verbatim,
*"GeomTransf vecxz (per element when orientation is used)"*. ADR 0014
made `apeGmsh.viewers` a pure `model.h5` consumer: beam / line
orientation in the results viewer is driven entirely by the
`/opensees/transforms` + `/opensees/element_meta` join in
`h5_reader.element_local_axes_vecxz()`
(`opensees/emitter/h5_reader.py:284`), keyed by **FEM element id**.

Both ADRs assume the enrichment zone is produced by the **bridge**
(`apeSees(fem).h5()`, `opensees/apesees.py:704-793`). A user who
writes their OpenSees model **by hand** in vanilla openseespy never
instantiates the bridge, so `/opensees/transforms` is never written.
`ViewerData.from_fem` carries no vecxz by design — FEMData is the
solver-agnostic neutral zone (`viewers/data/_elements.py:189-206`) —
and MPCO cannot carry it (`project_mpco_no_vecxz`). The failure is
silent: every beam renders at the structural-default orientation with
no diagnostic.

The need: let a hand-written-OpenSees user produce the same
`/opensees/transforms` + `element_meta` zone **without** being forced
through the bridge's typed-primitive API, the project owner having
explicitly chosen an *explicit declarative side-channel* over a
monkeypatch tap.

**Relationship to ADR 0011's "never an input" mitigation.** ADR 0011
mitigates "H5 becomes the primary contract" by *"treat the H5 strictly
as an emit output … the bridge does not read its own H5 back."*
`ModelData` is **not the bridge**. It is a separate authoring façade.
Its `from_h5` is a narrow enrich round-trip of the broker zone plus
the two orientation record lists; it never reads H5 back to *drive
emission*. The bridge still never reads its own output. `ModelData`
is an additional *writer front-door* to the same target, not a bridge
input path — 0011's invariant is preserved.

## Decision

### The class — orientation-only

Add `apeGmsh.opensees.ModelData`: a free-standing, read-mostly
authoring façade (symmetric with `FEMData` / `ViewerData`; not an
attribute of `g` or of `apeSees`).

```python
md = ModelData(fem, *, ndm, model_name=None)      # fem AND ndm mandatory
md.oriented_elements(pg=, ele_type=, vecxz=)      # the ONE inject method
md.write(path)
ModelData.from_h5(path)                            # rehydrate + enrich
```

`fem` is mandatory: the neutral zone is therefore always whole, which
eliminates the orphaned-orientation hazard by construction rather than
by a runtime guard. `ndm` is mandatory: the reader's transf-arg slot
index is `ndm`-dependent (`h5_reader.py:329`); a defaulted `ndm=0`
silently mis-reads the join. `oriented_elements` resolves `pg=` against
the broker **at inject time** into `(fem_eid, connectivity)` — the user
never types a tag, so `fem_eid` is correct by construction. The
surface is locked to orientation: no materials / sections / patterns /
recorders / analysis / constraints / loads / masses (charter scope;
keeps `ModelData` from becoming a second model-authoring API).

> **Amended (May 2026):** recorders are now in scope — see
> [Amendment — recorders](#amendment--recorders-may-2026). The
> remaining list (materials / sections / patterns / analysis /
> constraints / loads / masses) stays out: those *define* model state,
> while recorders only *observe* it.

### Schema authority stays single — one new `H5Emitter` method

`ModelData` owns **zero** HDF5 bytes and **zero** schema knowledge. It
holds a private `H5Emitter` and a mandatory `FEMData`. Record
construction for the orientation pair is done by **one new public
method on the schema-owning `H5Emitter`** that appends a
`_TransformRecord` + the per-element `_ElementRecord`s
(`opensees/emitter/h5.py:323,337`), placing the transf tag at the slot
the reader's join expects via the **same** `_ELEM_REGISTRY` the reader
consults (`h5_reader.py:334-339`). `ModelData` does **not** drive the
emitter through the bridge's `_internal` tag-resolution side-channels.

### One shared composer — internal extraction, bridge public API unchanged

The composition body of `apeSees.h5()` (`apesees.py:775-793`: broker
neutral zone via `_try_write_broker_zone` with the stub-fallback +
schema-stamp + teardown, then `emitter.write_opensees_into(f)`, then
cuts) is extracted into one shared
`_compose_model_h5(fem, emitter, path, *, snapshot_id=None, cuts=(),
sweeps=())`. Both `apeSees.h5` and `ModelData.write` call it. This is
an **internal** extraction: `apeSees.h5`'s public signature is
untouched. There is exactly one composer, one schema-stamp rule, one
teardown.

### Fail-loud contract

| Condition | Behavior |
|---|---|
| `pg=` resolves to no elements / no broker | **raise** at inject time |
| `ele_type` not a `_ELEM_REGISTRY` beam token | **raise** at inject time, list valid tokens |
| `ndm` unset or 0 | **raise** at construction |
| element-meta injected but its transform group missing | **raise** (partial-correctness trap) |
| nothing injected | **write**, no raise — graceful default-orientation degrade |
| crash mid-write | reuse `_try_write_broker_zone` teardown — no half-file |

`fem_eid = -1` (`MISSING_FEM_ELEMENT_ID`) is a valid sentinel for the
bridge fan-out test path; it is **never** valid for a declarative
inject and must raise, never be written.

### The one accepted coupling

`oriented_elements` ↔ `_ELEM_REGISTRY` for the transf-tag slot. This
is unavoidable for *any* design — writer and reader must agree on the
slot — and is contained by pointing both at the one vocabulary module
so they evolve together. It is **vocabulary** coupling, not
bridge-evolution coupling.

### Round-trip / `from_h5`

`ModelData.from_h5` loads the broker zone (`FEMData.from_h5`) and
rehydrates only the two orientation record lists via the reader's
public accessors. `snapshot_id` is carried **opaque** (read from
`/meta`, never recomputed). `fem_eids ↔ per_element_emitted_tag ↔
args` row correspondence is preserved byte-for-byte (no re-bin / no
re-order). Optional `/opensees` children are probed with `in`
(H5Lexists), never `Group.get()` (`project_h5py_optional_child_get_hazard`,
PR #261).

Output is byte-equivalent (modulo `created_iso`) to
`apeSees(fem).h5()` for the same model: no `ModelData` marker, so the
viewer / future P2 read path needs zero `ModelData` awareness.

## Alternatives considered

1. **Monkeypatch / tap interception** of `ops.geomTransf` / `element`.
   Rejected — magical; in-process-only; fragile to call style
   (`from … import geomTransf` escapes a module-attribute patch); the
   project owner explicitly chose a declarative side-channel.
2. **`ModelData` drives `H5Emitter` via `_internal` side-channels**
   (`set_current_fem_element_id` / `set_element_nodes`,
   `tag_resolution.py:126,164`). Rejected — exports bridge-internal
   coupling into a public class; the emitter reads those attrs at
   `h5.py:655,666` and a bridge refactor would silently break the
   public façade.
3. **Inject a populated emitter into `apeSees.h5()`** (Blue Variant C).
   Rejected — widens the **primary** bridge's public contract for a
   **secondary** feature, the exact lock-in the owner forbade; and it
   reads as "using the bridge" while the premise is *not* using it.
4. **PG-keyed spec container + stateless `write_model_h5()`**
   (declarative Variant B). Rejected — the spec layer is PG-keyed
   while the stored layer is eid-keyed, so `from_h5` round-trip is
   lossy / asymmetric, violating the "symmetric with
   `FEMData.from_h5`" requirement; also two public concepts for a lean
   feature.
5. **Use the bridge for the whole model** (declare the structural
   model via typed primitives, hand-roll only the analysis driver).
   Rejected as the *general* answer — the owner's premise is users who
   will not use the bridge for model construction. It remains the
   recommended path for users who *are* willing and is unaffected by
   this ADR.
6. **Extend FEMData to hold vecxz.** Rejected for the same reason ADR
   0011 alternative 2 rejected it — FEM is geometry-only by design;
   `geomTransf` is solver-specific and would leak into every FEM
   consumer.

## Consequences

**Positive:**

- One schema authority. `ModelData` holds no HDF5 / no schema;
  schema-version bumps propagate to it for free. Enforced by an
  AST/import guard test (the ADR 0014 / `test_viewer_data.py`
  precedent) asserting `ModelData` has no `h5py` write surface.
- Non-locking. The shared composer is an internal extraction;
  `apeSees.h5`'s public signature is unchanged, so future bridge
  refactors are unconstrained by this secondary feature.
- Byte-equivalent output → `ModelData` is invisible to the viewer and
  to any future P2 consumer; no read-path branching on provenance.
- **P2 shipped (May 2026).** The viewer-side consumer ratifies this
  ADR's INV-16 (byte-equivalent output) by needing zero `ModelData`
  awareness. `ResultsViewer` auto-resolves `results._path` into a
  branched `ViewerData.from_h5` scene whenever the file carries the
  `/opensees/transforms` + `/opensees/element_meta` pair (probe at
  `viewers/data/_h5_probe.py::has_opensees_orientation`, using
  H5Lexists per `project_h5py_optional_child_get_hazard` / PR #261).
  Explicit `Results.viewer(model_h5=)` overrides the auto-resolve and
  is forwarded into the spawned subprocess via `--model-h5` on
  `blocking=False`. See [modeldata-enrichment-scope.md](../modeldata-enrichment-scope.md)
  §4 (P2 row flipped to shipped) and the merge PR for the four-commit
  decomposition.
- Lean: one public class, one new `H5Emitter` method, one bounded
  refactor, one inject method.
- Eliminates a duplicate of the already-subtle stub-teardown +
  schema-stamp logic (`apesees.py:782-788,896-898`) — the single most
  likely origin of silent wrong-version files.

**Negative:**

- One accepted vocabulary coupling (`oriented_elements` ↔
  `_ELEM_REGISTRY`), reviewed and contained.
- One tag-correspondence caveat lives **outside** `ModelData`'s
  enforcement reach: the user's hand-written OpenSees element tags
  must equal the broker FEM eids (the "drive `ops.element` from
  `fem.elements`" pattern). `ModelData` makes `model.h5` internally
  correct; it cannot police the tags the user's recorder wrote. This
  is documented loudly in the class docstring, not faked.
- A new public class to maintain, though deliberately minimal.

## Amendment — recorders (May 2026)

### Context

The original scope locked `ModelData` to orientation and explicitly
excluded recorders. In practice the hand-written-deck user who needs
orientation *also* needs recorders — the results viewer binds two
things: the orientation join (`model.h5`, which `ModelData` already
writes) **and** the result files (`.out` / `.mpco`, which recorders
produce). Without a recorder surface the user must hand-type raw
`ops.recorder("Node", "-file", …, "-node", …, "-dof", …, "disp")` /
`ops.recorder("mpco", …)` lines — verbose, and a fresh chance to
mistype the very tags the orientation join depends on
(`feedback_no_private_emit_vanilla`: pointing the bridge's whole-model
`LiveOpsEmitter` at recorders mid-session crashed the kernel because
its `__init__` calls `ops.wipe()`).

### Decision

Admit recorders to `ModelData` via three methods. They reuse the
**existing** recorder fan-out (`_internal.build._emit_recorder_declaration`)
and the **shared** declaration builder (`recorder.build_recorder_declaration`,
extracted so `ops.recorder.declare` and `ModelData.recorders` have one
source of truth for shorthand expansion + per-category record
construction).

```python
md.recorders(nodes="displacement", pg="Base", file_root="out")  # declare
md.attach_recorders(ops)                  # forward into a LIVE openseespy session
lines = md.recorder_commands(target="py") # OR render script lines (py | tcl)
```

- **`recorders(...)`** — same canonical vocabulary and selectors as
  `ops.recorder.declare`, bound to `ModelData`'s `ndm`/`ndf`; stores a
  `RecorderDeclaration`. Calls accumulate.
- **`attach_recorders(ops)`** — forwards each declaration into the
  caller's already-imported `openseespy.opensees` module via a
  recorder-only `_LiveRecorderSink` that implements exactly the three
  fan-out methods (`recorder` / `recorder_declaration_begin` /
  `recorder_declaration_end`). It cannot wipe the domain or re-declare
  the model, so it is safe to call mid-session — the failure mode the
  bridge's `LiveOpsEmitter` has by design.
- **`recorder_commands(target="py"|"tcl")`** — for the file /
  subprocess workflow where there is no live session to attach to.
  Returns recorder lines only: the deck preamble the `PyEmitter` seeds
  (`import openseespy …` + `ops.wipe()`) is stripped by capturing the
  buffer length before fan-out, so pasted lines cannot erase the user's
  model.

### Why recorders but still not materials / analysis

The original scope lock guarded against `ModelData` becoming a second
**model-authoring** API. Recorders do not author model state — they
*observe* a domain that already exists. They declare no nodes,
elements, materials, sections, patterns, constraints, loads, masses,
or analysis chain. Admitting them therefore does not reopen the risk
the lock was protecting against. The excluded list is unchanged for
everything that *defines* the model.

### Invariants preserved

- **Schema authority single / byte-equivalent output (INV-3 / INV-16).**
  Recorder declarations are **not** written to `model.h5` by
  `write()`. They describe runtime observation, not model definition,
  so `write()` is byte-identical to before for any model and the
  `H5Emitter` schema is untouched. (Persisting recorder intent into
  `/opensees/recorders` so the archive is self-describing is a
  coherent future step, deliberately deferred — it would be the first
  thing this amendment did *not* do.)
- **No bridge `_internal` tag side-channels.** `attach_recorders` /
  `recorder_commands` drive the fan-out with `fem_eid_to_ops_tag=None`,
  whose documented fallback resolves selectors to raw FEM eids — exactly
  correct under the `tag == fem_eid` convention below. No `TagAllocator`,
  no `BuiltModel`, no bridge state.
- **No new h5py write surface.** The AST guard
  (`test_model_data_ast_guard.py`) still passes — the new methods touch
  no HDF5.

### Fail-loud / tag-correspondence

The tag-correspondence caveat in Consequences (negative #2) is now
**load-bearing for recorders, not just orientation**: recorder
selectors resolve to FEM ids, so the emitted commands target the user's
`ops.element` / `ops.node` tags only if those equal the broker fem ids.
A mismatch *silently* records the wrong entities. As with orientation,
`ModelData` cannot police the tags the user's live deck used; it is
documented loudly — a `.. warning::` on the class plus a `# recorders
assume … tags == fem eids` banner prepended to every
`recorder_commands` output — not faked into a false guarantee.

### Alternatives considered (amendment)

1. **Live-attach via the bridge's `LiveOpsEmitter`.** Rejected — its
   constructor calls `ops.wipe()`, erasing the user's hand-built model;
   it is a whole-model emitter, not a recorder forwarder. The 3-method
   `_LiveRecorderSink` is the surgical fit.
2. **String-rendering only (paste into a notebook).** Rejected as the
   *primary* surface — in a py/notebook session the `ops` module is
   already live in-process, so copy-paste is pointless friction.
   Retained as the secondary surface for the file/subprocess workflow,
   where there is no live session.
3. **Persist recorders into `model.h5` and re-emit via
   `OpenSeesModel.build`.** Rejected for now — the hand-written-deck
   user does not re-emit through `OpenSeesModel`, so it would not serve
   the live workflow; and it would touch the schema. Deferred, not
   refused.

## References

- [decisions/0011-h5-as-fourth-emit-target.md](0011-h5-as-fourth-emit-target.md)
  — `model.h5` as emit target; this ADR adds a second writer
  front-door to the same target without making H5 a bridge input.
- [decisions/0014-viewer-is-pure-h5-consumer.md](0014-viewer-is-pure-h5-consumer.md)
  — the consumer this feature serves; the AST/parity-test precedent.
- [decisions/0010-csys-for-frame-orientation.md](0010-csys-for-frame-orientation.md)
  — orientation fields; vecxz is the datum this ADR makes
  vanilla-authorable.
- [../modeldata-enrichment-scope.md](../modeldata-enrichment-scope.md)
  — implementation scope, phased commits, the invariant checklist
  (Red-team adversarial review, May 2026).
- [../h5-schema.md](../h5-schema.md) — the on-disk contract
  `ModelData` produces but does not own.
- [../charter.md](../charter.md) — P3 (bridge takes a `FEMData`), P8
  (Emitter abstraction), P9 (`apeGmsh.opensees` does not depend on
  `apeGmsh.mesh` internals).
