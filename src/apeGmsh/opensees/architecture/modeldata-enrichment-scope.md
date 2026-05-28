# ModelData — vanilla-OpenSees `model.h5` orientation enrichment

**Status:** **Shipped May 2026.** P1 (the `ModelData` writer) merged
in PR [#262](https://github.com/nmorabowen/apeGmsh/pull/262); post-merge
selection-v2 migration + AST guard in PR
[#264](https://github.com/nmorabowen/apeGmsh/pull/264); **P2** (the
consumer-side wiring — `Results.viewer(model_h5=)` as the unified
pointer, auto-resolve of `results._path`, branched scene builder,
subprocess `--model-h5` forwarding) merged on branch
`feat/viewer-model-h5-enrichment`. Full
H5/femdata/viewer/cuts gate passes (1302 tests; +13 new across P1+P2).
Design ratified by ADR 0018 after a blue/red adversarial exercise
(1× BLUE Opus + 1× RED Opus + head-engineer synthesis); the P2
subprocess decision (forward vs refuse) ran the same red/blue
exercise — see the merge PR body for that synthesis. Every
load-bearing claim below is cited at source so it can be re-checked.

This phase lets a user who writes their OpenSees model **by hand**
(vanilla openseespy, no `apeSees` typed primitives) produce the
`/opensees/transforms` + `/opensees/element_meta` zone of `model.h5`,
so the results viewer orients beam / line diagrams correctly. It is
the side-feeder counterpart to ADR 0011 (bridge writes the enrichment
zone) for users who never touch the bridge.

```
hand-written OpenSees ──► ModelData(fem) ──► model.h5  ─► Viewer / P2
                                              ▲
                              apeSees(fem).h5 ┘  (same target, same shape)
```

Rationale, alternatives, and the "never an input" relationship to ADR
0011 are in
[decisions/0018-modeldata-vanilla-opensees-enrichment.md](decisions/0018-modeldata-vanilla-opensees-enrichment.md).
This document is the implementation scope and the acceptance ledger.

## 1. The change

### 1a. New public class `apeGmsh.opensees.ModelData`

Free-standing, orientation-only. `fem` and `ndm` mandatory.

```python
from apeGmsh import FEMData
from apeGmsh.opensees import ModelData

fem = FEMData.from_h5("frame.h5")
md  = ModelData(fem, ndm=3, model_name="frame")
md.oriented_elements(pg="columns",     ele_type="forceBeamColumn", vecxz=(0, 1, 0))
md.oriented_elements(pg="floor_beams", ele_type="forceBeamColumn", vecxz=(0, 0, 1))
md.write("frame.h5")

# enrich-in-place later (symmetric with FEMData.from_h5)
md = ModelData.from_h5("frame.h5")
md.oriented_elements(pg="braces", ele_type="dispBeamColumn", vecxz=(0, 0, 1))
md.write("frame.h5")
```

`oriented_elements` resolves `pg=` against the broker **at inject
time** into `(fem_eid, connectivity)`; the user never types a tag.
Partial recording = which PGs you inject; un-injected PGs degrade to
the structural-default orientation (the documented
`ViewerData.from_fem` path, `viewers/data/_elements.py:189-206`).

### 1b. One new public method on `H5Emitter`

`ModelData` owns no HDF5. Record construction for the orientation pair
is a single new public method on the schema-owning `H5Emitter`
(`opensees/emitter/h5.py`) that appends a `_TransformRecord` + the
per-element `_ElementRecord`s (`h5.py:323,337`), placing the transf
tag at the slot the reader's join expects via the same `_ELEM_REGISTRY`
the reader uses (`h5_reader.py:334-339`). No `_internal`
tag-resolution side-channel use from `ModelData`.

### 1c. Bounded internal refactor — one shared composer

Extract the composition body of `apeSees.h5()`
(`opensees/apesees.py:775-793`) into
`_compose_model_h5(fem, emitter, path, *, snapshot_id=None, cuts=(),
sweeps=())`. `apeSees.h5` and `ModelData.write` both call it.
`apeSees.h5`'s public signature is **unchanged**. No new schema
module; no rework of `_write_transforms` / `_write_element_meta`.

## 2. Acceptance criteria — the invariant ledger

From the RED-team dossier. Each: invariant → failure prevented →
citation. **P0 = the design fails without it. P1 = wrong-in-corners
without it.** These are the test plan's assertions.

### P0

1. **No HDF5 writer in `ModelData`.** Delegates serialization to
   `H5Emitter.write_opensees_into` / `_write_*` (`h5.py:866,1177,1219`)
   and `_femdata_h5_io.write_neutral_zone` / `write_meta`
   (`mesh/_femdata_h5_io.py:144,120`). → blocks a third drifting copy
   of the `/opensees/transforms` layout.
2. **`schema_version` never a literal in `ModelData`.** Obtained from
   the owning code path. → blocks the file lying about its layout past
   a schema bump (`h5.py:137`; `_femdata_h5_io.py:85`;
   `apesees.py:787-788`).
3. **One module owns the `/opensees` schema; AST/import guard test
   asserts `ModelData` has no `h5py` write surface** (ADR 0014 /
   `test_viewer_data.py` precedent). → blocks the schema-deviation
   rationale (`h5.py:39-66`) going stale via a second writer.
4. **Surface uses only the already-buffered primitives**
   (`_TransformRecord` / `_ElementRecord`, `h5.py:323,337`); no richer
   element/transform model. → blocks `ModelData` becoming a parallel
   model object that freezes `_internal/build.py` + `_ELEM_REGISTRY`.
5. **Raise (never write `-1`) on missing/negative/unknown fem_eid.**
   → blocks a successfully-written file the viewer renders fully
   un-oriented with zero diagnostics (`tag_resolution.py:74,177-180`;
   `h5_reader.py:354-356`).
6. **Validate `ele_type` against `_ELEM_REGISTRY` at inject time.** →
   blocks silent whole-type orientation loss from a typo/casing
   (`h5_reader.py:338-339`; `h5.py:1257`).
7. **No `/opensees` orientation pair without the neutral-zone element
   ids it keys against** — enforced structurally by making `fem`
   mandatory. → blocks orphaned orientation mapping onto nothing
   (`h5_reader.py:336-363`; `viewers/data/_elements.py:192-195`).
8. **`from_h5` + rewrite treats `snapshot_id` as opaque
   carry-through; never recompute.** → blocks silent
   binding-identity mutation to `""` (bind does not raise —
   `project_bind_contract`) (`mesh/_femdata_hash.py:41-43`;
   `apesees.py:759-766`).
9. **`fem_eids ↔ per_element_emitted_tag ↔ args` row correspondence
   invariant across round-trip; no re-bin / no re-order.** → blocks
   element A inheriting element B's orientation after a harmless
   load-and-resave (`h5.py:1251-1271`; `h5_reader.py:353`).

### P1

10. **Materials / sections / patterns / analysis / constraints /
    loads / masses ABSENT from the surface.** → blocks scope
    metastasis into a second authoring API. *(Amended May 2026:
    recorders were removed from this exclusion and added to the
    surface — they observe the domain rather than define it, so they
    don't reopen the authoring-API risk. See ADR 0018 "Amendment —
    recorders". The on-disk invariant at §4 still holds: `ModelData`
    writes no recorder zone to `model.h5`; the recorder surface emits
    live / as script lines only.)*
11. **Explicit write-vs-raise split** (raise on the
    element-meta-without-its-transform partial-correctness trap;
    write+degrade on nothing-injected). → blocks "80% oriented looks
    like a viewer bug" (`h5_reader.py:360-361`).
12. **`ndm` required and non-zero whenever a transform is injected.**
    → blocks the vocabulary slot index computed for the wrong
    dimension (`h5_reader.py:329`; `h5.py:1526`).
13. **Schema-version stamping under partial zones reuses the one
    existing rule** (`_override_schema_version` semantics,
    `apesees.py:902`). → blocks a third stamping rule defeating the
    (major-only) version gate (`h5_reader.py:101-106`).
14. **Half-written / torn-down behavior reuses
    `_try_write_broker_zone`'s teardown** (`apesees.py:896-898`). →
    blocks a crash leaving a silently mis-rendering half-file.
15. **`from_h5` probes optional `/opensees` children with `in`
    (H5Lexists), not `Group.get()`** (`project_h5py_optional_child_get_hazard`,
    PR #261). → blocks propagating the known-hazardous probe idiom
    into a new loader (`h5_reader.py:309-310`).
16. **Output byte-equivalent (modulo `created_iso`) to
    `apeSees(fem).h5()`; no `ModelData` marker attr/group.** → blocks
    `ModelData` leaking into the future P2 read path
    (`results/Results.py:441-443`; `h5_reader.py:65,284`;
    `h5.py:1525`).

## 3. Commit decomposition

Each commit is independently green; the feature is usable after C3.

- **C1 — shared composer (refactor, behavior-invariant).** Extract
  `_compose_model_h5` from `apesees.py:775-793`; `apeSees.h5` calls
  it. Verify: full existing `apeSees.h5` / H5 round-trip suite passes
  unchanged; output byte-identical (modulo `created_iso`) to pre-refactor
  (INV-13/16).
- **C2 — `H5Emitter` orientation method.** One public method building
  `_TransformRecord` + `_ElementRecord`s, transf slot via
  `_ELEM_REGISTRY`. Verify: unit test that a constructed pair round-trips
  through `h5_reader.element_local_axes_vecxz()` to the injected vecxz
  keyed by fem_eid (INV-4/9).
- **C3 — `ModelData` class.** `__init__(fem, *, ndm, model_name)`,
  `oriented_elements`, `write` (via C1 composer). Fail-loud contract
  (INV-5/6/7/11/12). Verify: parity test — `ModelData` file vs
  `apeSees(fem).h5()` file for the same model are byte-equivalent
  modulo `created_iso` (INV-16); fail-loud tests for each raise row.
- **C4 — `ModelData.from_h5` + round-trip.** Rehydrate broker + the
  two record lists via reader public accessors; opaque `snapshot_id`;
  `in`-probe optional children. Verify: load→inject→write→reload is
  fixed-point on the orientation pair and on `snapshot_id`
  (INV-8/9/15).
- **C5 — AST/import guard test.** Assert `ModelData` imports no
  `h5py` write surface (INV-1/3), mirroring the ADR-0014 pattern.
- **C6 — docs.** Class docstring carries the tag-correspondence
  caveat (ADR 0018 Consequences/Negative) loudly; `apegmsh-helper`
  skill + `h5-schema.md` cross-reference; flip this doc's Status to
  Shipped with PR/commit refs.

## 4. Out of scope (explicit)

- **P2 — non-native consumption. Shipped May 2026** on branch
  `feat/viewer-model-h5-enrichment`. The viewer wires the
  `/opensees/transforms` + `/opensees/element_meta` pair into the
  post-solve scene via `ResultsViewer`'s auto-resolve of
  `results._path` (`viewers/data/_h5_probe.py::has_opensees_orientation`
  → branched `ViewerData.from_h5` at `results_viewer.py:268-287`),
  generalising the existing `Results.viewer(model_h5=)` from a
  cuts-only kwarg to the unified consumer-side pointer for both
  cuts and orientation. INV-16's byte-equivalence is the guarantee
  that lets P2 consume `ModelData` output with zero `ModelData`
  awareness — exactly as predicted in P1's design. See the merge PR
  and ADR 0018 Consequences/Positive for the four-commit
  decomposition (probe + resolver, scene-builder branch + subprocess
  forwarding, parity test + docs, ADR/scope flip).
- Any `/opensees` zone other than transforms + element_meta
  (materials, sections, patterns, recorders, analysis) — INV-10.
- Tap / interception authoring — rejected in ADR 0018 (alt 1).
- Enforcing user OpenSees-tag ↔ FEM-eid equality — documented caveat,
  not enforceable from `ModelData` (ADR 0018 Consequences/Negative).

## 5. Verification posture

No GPU/OpenGL here (`feedback_viewer_no_gpu`): every criterion is
data-level — `h5_reader` joins, byte-equivalence diffs, fail-loud
assertions, AST guard. No viewer window is required to prove
correctness; the existing `ViewerData.from_h5` join is the contract
under test. Use the project's standard `opensees_venv` python for
pytest (`feedback_opensees_venv`).
