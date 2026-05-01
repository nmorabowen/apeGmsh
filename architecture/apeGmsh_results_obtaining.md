---
title: apeGmsh Results — Obtaining the Database
aliases: [results-obtaining, recorder-strategies, spec-as-seam, emit_recorders, emit_mpco]
tags: [apeGmsh, architecture, results, recorders, OpenSees, MPCO, capture]
---

# apeGmsh Results — Obtaining the Database

> [!note] Companion document
> This file documents the **declare → resolve → execute** pattern at
> the core of apeGmsh's results pipeline: how a *single* declarative
> recorder spec can drive five distinct execution strategies, all
> producing files that the same `Results` reader can consume. For the
> full reader / slab / FEMData-binding architecture, see
> [[Results_architecture]] §Layer 4–5. For the analysis-loop side
> (what the user is responsible for between OpenSees commands), see
> [[apeGmsh_architecture]] §OpenSees pipeline.

The user's intent — *"record displacement on the top surface, stress
on the body, every step"* — is stable across every way of running
OpenSees. **What changes** between a notebook session, a cluster Tcl
job, and an STKO MPCO run is *who writes the file*, *what format*,
and *how it gets back to Python*. apeGmsh keeps the intent in one
object and lets each execution strategy consume it differently.

The result is **five strategies, one spec, one read API**. This
document explains the seam that makes that work.

---

## 1. The seam in one diagram

```
              g.opensees.recorders.nodes(...)        ┐
              g.opensees.recorders.gauss(...)        │  Layer 1
              g.opensees.recorders.line_stations(...)│  Declarative
              g.opensees.recorders.modal(...)        ┘  intent
                            │
                            ▼
                  recorders.resolve(fem)
                            │
                            ▼
              ┌──────────────────────────────┐
              │  ResolvedRecorderSpec        │      ◀─ Layer 2: Seam
              │  • fem.snapshot_id (binding) │         (frozen, gmsh-
              │  • tuple of records          │          and OpenSees-
              │  • numpy + dataclasses       │          independent)
              └──────────────┬───────────────┘
                             │
       ┌──────────┬──────────┼──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼
   export.tcl  export.py  emit_      capture     emit_
                          recorders              mpco
       │          │          │          │          │  Layer 3:
       │          │          │          │          │  Execution
       ▼          ▼          ▼          ▼          ▼  strategies
   model.tcl  model.py    *.out/    run.h5     run.mpco
   (run later)(run later) *.xml     (apeGmsh   (STKO HDF5)
                          (live or  native)
                           later)
       │          │          │          │          │
       └─────┬────┴────┬─────┘          │          │
             ▼         ▼                ▼          ▼
   Results.from_recorders()    Results.    Results.    ◀─ Layer 4:
   (transcode + cache)         from_native from_mpco     Same reader
                                                         API for all
                       │
                       ▼
                  results.nodes.get(component=..., pg=...)
                  results.elements.gauss.get(...)
                  results.viewer()                                ◀─ Layer 5
```

Every arrow above is in the codebase today. The five strategies are
the cells of column three.

---

## 2. Layer 1 — Declarative intent

```python
g.opensees.recorders.nodes(
    components=["displacement", "reaction"], pg="Top", dt=0.01)
g.opensees.recorders.gauss(
    components=["stress", "strain"], pg="Body")
g.opensees.recorders.line_stations(
    components=["force"], label="frame")
g.opensees.recorders.modal(n_modes=10)
```

Each call stores a frozen `RecorderRecord` on the `Recorders`
composite. **No FEMData. No OpenSees. No IDs.** Just a dataclass that
says *"I want X on whatever PG/label resolves to, at this cadence."*
You can declare against names that don't even exist in the mesh yet —
declarations are pure data.

Selectors mirror the read-side composite vocabulary:
`pg=` (physical groups), `label=` (apeGmsh labels), `selection=`
(post-mesh `g.mesh_selection` sets), `ids=` (raw IDs). Categories
match topology levels: `nodes`, `elements`, `gauss`, `line_stations`,
`fibers`, `layers`, `modal`. Cadence is at most one of `dt=` or
`n_steps=`; missing both means every step.

> Pre-resolution, the records know **what** the user wants — not
> **where** it lives in the mesh.

---

## 3. Layer 2 — The seam: `ResolvedRecorderSpec`

```python
spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=6)
```

`resolve(fem)` is the bridge from intent to geometry. It:

1. Looks each selector up against the bound `FEMData` — `pg="Top"` →
   concrete node ID array.
2. Expands shorthand components (`"displacement"` →
   `displacement_x/y/z`).
3. Pulls section metadata for layered shells / fiber sections from
   `g.opensees._sections` so downstream strategies have what they
   need without re-reading the session.
4. Tags the result with `fem.snapshot_id` so any consumer can verify
   the binding is still consistent.

The output is `ResolvedRecorderSpec` — **gmsh-independent and
OpenSees-independent at the type level**, just numpy and dataclasses.

This is the seam. **Anything that knows how to read a spec is a valid
execution strategy.** Adding a new strategy means writing a new
consumer of this object — no changes to the declaration side, no
changes to the readers.

> [!tip] What lives on the spec
> - `fem_snapshot_id` — drift-detection contract.
> - `records` — tuple of `ResolvedRecorderRecord` (one per declaration).
> - Per-record: category, components, resolved IDs, cadence, optional
>   section metadata.
> - Methods on the spec correspond to **execution strategies**:
>   `to_tcl_commands()`, `to_python_commands()`, `capture(...)`,
>   `emit_recorders(...)`, `emit_mpco(...)`.

---

## 4. Layer 3 — The five execution strategies

All five consume the spec and produce a file. They differ in **who
runs OpenSees**, **what format gets written**, and **what coverage
each path supports**.

### 4.1  Strategy A₁ — Export Tcl (`export.tcl`)

```python
g.opensees.export.tcl("model.tcl", recorders=spec)
```

Translates each spec record into `recorder Node …` / `recorder
Element …` text and inlines it into the Tcl model script. The user
runs OpenSees externally; output `.out` / `.xml` files land in the
recorder output directory. The spec is also serialized as an HDF5
sidecar manifest so `Results.from_recorders` can decode column
layouts later.

**Use when:** running on a cluster, integrating with non-Python
tooling, reproducibility from a checked-in script.

### 4.2  Strategy A₂ — Export Python (`export.py`)

```python
g.opensees.export.py("model.py", recorders=spec)
```

Same as A₁ but emits `ops.recorder(...)` source code instead of Tcl.
Same output files, same reader. Useful when the rest of the
production pipeline is Python.

### 4.3  Strategy A₃ — Live recorders (`emit_recorders`) — *new*

```python
spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

with spec.emit_recorders("out/") as live:
    live.begin_stage("gravity", kind="static")
    for _ in range(n_grav):
        ops.analyze(1, 1.0)
    live.end_stage()

    live.begin_stage("dynamic", kind="transient")
    for _ in range(n_dyn):
        ops.analyze(1, dt)
    live.end_stage()

grav = Results.from_recorders(spec, "out/", fem=fem, stage_id="gravity")
dyn  = Results.from_recorders(spec, "out/", fem=fem, stage_id="dynamic")
```

Same recorder commands as A₁/A₂, but pushed **directly into the
running openseespy domain** via `ops.recorder(*args)` calls. No
script written, no subprocess. Output filenames carry a per-stage
prefix (`<stage>__<record>_<token>.{out,xml}`) so multi-stage runs
don't collide. `Results.from_recorders(stage_id=...)` loads one stage
at a time.

**Use when:** notebook-driven analysis, interactive debugging,
production runs that need recorder behaviour but stay in-process.

**Coverage:** `nodes`, `elements`, `gauss`, `line_stations`. `fibers`
and `layers` warn-and-skip (classic recorders don't carry fiber/layer
metadata cleanly). `modal` raises on `__enter__` (needs `ops.eigen()`,
which lives on `capture`).

### 4.4  Strategy B — Domain capture (`capture`)

```python
with spec.capture(path="run.h5", fem=fem, ndm=3, ndf=3) as cap:
    cap.begin_stage("gravity", kind="static")
    for _ in range(n_grav):
        ops.analyze(1, 1.0)
        cap.step(t=ops.getTime())
    cap.end_stage()

    cap.capture_modes(n_modes=10)

results = Results.from_native("run.h5", fem=fem)
```

The native path. apeGmsh queries the live ops domain itself —
`ops.nodeDisp(...)`, `ops.nodeReaction(...)`, `ops.eleResponse(...)`
— translating each spec record's canonical names into the right call
for that step. Per-step values buffer in RAM; chunked writes go to
HDF5 at `end_stage()` via `NativeWriter`.

**Use when:** broadest coverage including modal stages, interactive
work, when you want apeGmsh to own the file format.

**Coverage:** all seven topology levels (nodes, elements, gauss,
line_stations, fibers, layers, springs) plus modal — the most
complete strategy.

### 4.5  Strategy C₁ — Export with MPCO line

```python
g.opensees.export.tcl("model.tcl", recorders=spec, mpco=True)
# … run with STKO loaded …
results = Results.from_mpco("run.mpco")
```

apeGmsh emits a single `recorder mpco …` line into the script;
STKO's MPCO recorder writes the HDF5 itself. apeGmsh's `MPCOReader`
synthesizes a partial `FEMData` from `/MODEL/` so reads work even
without re-binding a session-side snapshot.

**Use when:** big parallel runs, the STKO ecosystem, when you want
the recorder's C++ implementation doing the heavy lifting.

### 4.6  Strategy C₂ — Live MPCO (`emit_mpco`) — *new*

```python
spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

with spec.emit_mpco("run.mpco"):
    for _ in range(n_steps):
        ops.analyze(1, dt)

results = Results.from_mpco("run.mpco")
```

Same MPCO file as C₁, but emitted in-process via
`ops.recorder("mpco", ...)`. **No `begin_stage`/`end_stage` ceremony**
— MPCO writes one file containing all stages with `pseudoTime`
encoding stage boundaries internally. Build-gate at `__enter__`
raises with a remediation pointer if the active openseespy build
doesn't include the MPCO recorder.

**Use when:** notebook-driven analyses on STKO's bundled Python
distribution.

**Coverage:** all categories including `fibers`, `layers`, and
`modal` — the MPCO recorder handles them natively via
`section.fiber.stress`, layered-section tokens, and
`modesOfVibration`.

---

## 5. Strategy comparison

| | **A₁ Export Tcl** | **A₂ Export Py** | **A₃ Live recorders** | **B Capture** | **C₁ Export MPCO** | **C₂ Live MPCO** |
|---|---|---|---|---|---|---|
| Who runs OpenSees | You (separate proc) | You (separate proc) | **Your notebook** | **Your notebook** | You (separate proc, STKO) | **Your notebook** (STKO build) |
| File written by | OpenSees | OpenSees | OpenSees recorder | apeGmsh `NativeWriter` | STKO MPCO recorder | STKO MPCO recorder |
| Format | `.out`/`.xml` | `.out`/`.xml` | `.out`/`.xml` | apeGmsh native `.h5` | `.mpco` HDF5 | `.mpco` HDF5 |
| Multi-stage | Manual (per-stage scripts) | Manual | **`begin/end_stage`** | **`begin/end_stage`** | Manual | All stages, single recorder |
| Modal records | Skipped | Skipped | **Raises** | **Native (`capture_modes`)** | OK | **Native** |
| Fibers / layers | Skipped | Skipped | Warn-and-skip | **Native (Phase 11e/f)** | OK | **Native** |
| Read via | `from_recorders` | `from_recorders` | `from_recorders(stage_id=)` | `from_native` | `from_mpco` | `from_mpco` |
| Cache | Yes | Yes | Yes | N/A — direct write | N/A | N/A |
| Build dependence | None | None | None | None | STKO required | **STKO required** (in-proc) |

The bold cells are why each strategy exists. None of them subsume
another — every one has a workflow it owns.

---

## 6. The shared infrastructure

The five strategies look distinct at the user level, but underneath
they share a small core that's worth knowing about.

### 6.1  `LogicalRecorder` — backend-agnostic intermediate

`emit_logical(record, output_dir, file_format, stage_id)` takes a
resolved record and returns a list of `LogicalRecorder` dataclasses:
*one classic-recorder command in structured form*. From there:

- `format_tcl(logical)` → Tcl source line (used by A₁).
- `format_python(logical)` → Python source line (used by A₂).
- `to_ops_args(logical)` → tuple ready to splat into
  `ops.recorder(*args)` (used by A₃).

**Single source of truth.** Whatever changes the recorder argument
shape change here, and all three formatters update together. The
file paths, the DOF lists, the response tokens — same code path on
every strategy.

### 6.2  `mpco_ops_args` — MPCO equivalent

For MPCO there's only one recorder per spec, so the shape is
flatter:

- `emit_mpco_tcl(records, ...)` → Tcl line (used by C₁).
- `emit_mpco_python(records, ...)` → Python source (used by C₁ Py
  variant).
- `mpco_ops_args(records, ...)` → tuple for
  `ops.recorder("mpco", *args)` (used by C₂).

Same single-source-of-truth principle.

### 6.3  Stage filename plumbing

Live recorders need per-stage files; export.tcl doesn't. The seam:
an optional `stage_id` parameter on `emit_logical` and
`_build_file_path` that prepends `<stage_id>__` to the basename.

```python
emit_logical(rec, output_dir="out/", stage_id=None)   # → out/r_disp.out
emit_logical(rec, output_dir="out/", stage_id="grav") # → out/grav__r_disp.out
```

The same parameter threads through:

- `_cache.list_source_files(spec, dir, stage_id=...)` — what
  `from_recorders` looks for.
- `RecorderTranscoder(spec, dir, ..., stage_id=...)` — how it
  recovers per-record file paths.

Default `None` keeps export.tcl/py byte-for-byte compatible with
pre-stage emission. Multi-stage live emission opts into the prefix.

### 6.4  Bind contract

Every read-side `Results.from_*` call walks the same protocol:

1. The file embeds (native) or synthesizes (MPCO) a `FEMData`
   snapshot tagged with a `snapshot_id`.
2. If the user passed a `fem=` snapshot, hashes must match —
   otherwise `BindError`.
3. Without `fem=`, the embedded/synthesized snapshot is used
   directly.

So a spec resolved against `fem` and a results file produced from
that run are **provably consistent** — the snapshot ID is the
contract. This is what lets `pg="Top"` work on `results.nodes.get`
even after the gmsh session has closed.

---

## 7. Why this layering matters

Three properties fall out of the spec-as-seam pattern, none of which
were obvious until they did:

**1. Adding strategies is cheap.** When the new "live recorders" and
"live MPCO" strategies were added (Phase 1c, Phase 2), the
declaration side and the read side did not change — five files of
new code, all in the `live/` subpackage. Anything that knows how to
walk a `ResolvedRecorderSpec` is a valid strategy.

**2. The reader is source-agnostic.** A notebook fragment like

```python
results = open_results(...)
slab = results.elements.gauss.get(component="stress_xx", pg="Body")
```

doesn't know whether `open_results` was `from_native`,
`from_recorders`, or `from_mpco`. The composite layer sees only the
`ResultsReader` protocol. Three formats, one mental model.

**3. Verification via shared helpers.** Because text emission and
live emission both flow through `LogicalRecorder` /
`emit_logical`, the cross-check tests assert that the source code
form (`format_python`) and the live arg form (`to_ops_args`) agree.
They cannot drift silently.

---

## 8. What's not in this picture

Two things deliberately stay outside this seam:

- **The analysis loop.** apeGmsh emits the model (nodes, elements,
  fix, mass, patterns, ties), the recorders, and reads back the
  results. It does **not** drive `ops.analysis`, `ops.algorithm`,
  `ops.integrator`, `ops.numberer`, `ops.system`, `ops.test`,
  `ops.analyze`. That's the user's responsibility (or
  [[opensees-expert]]'s domain). The execution strategies all
  *expect* the user is calling `ops.analyze` themselves between
  begin/end markers.

- **The on-disk schema details.** `Results_architecture.md` is the
  comprehensive reference for the native HDF5 schema, MPCO
  translation rules, the canonical naming vocabulary, and the slab
  shape conventions. This document scopes itself to the
  *strategy-level* architecture; the schema-level details belong
  there.

---

## 9. Cross-references

- [[Results_architecture]] — comprehensive reference for the reader
  protocol, native HDF5 schema, slab dataclasses, and FEMData
  embedding.
- [[apeGmsh_architecture]] — overall apeGmsh architecture; OpenSees
  pipeline section explains where the analysis loop lives.
- [[apeGmsh_broker]] — `FEMData` snapshot semantics and `snapshot_id`
  drift-detection contract.
- [[apeGmsh_results_viewer]] — the post-solve viewer, one of many
  consumers of `Results`.
- `apegmsh-helper` skill — the apeGmsh skill that knows the user-
  facing API for declaring recorders and building models.
- `mpco-recorder` skill — for what MPCO actually writes to disk and
  the GP_X / integration-rule metadata.
- `stko-to-python` skill — for reading `.mpco` files in Python after
  a run.
