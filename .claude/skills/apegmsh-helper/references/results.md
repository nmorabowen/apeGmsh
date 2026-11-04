# Results — post-processing OpenSees output
<!-- skill-freshness: verified against apeGmsh main@fe910e8d (2026-11-03) · signatures: python -m apeGmsh.studio.lookup SYMBOL (ADR 0096); src/ is not the authoring lookup -->

`Results` reads an OpenSees run back into apeGmsh's label/query world.
All signatures below are read from `src/apeGmsh/results/Results.py`
(and `_composites.py`, `demo.py`, `viewers/web_viewer.py`).

Top-level import: `from apeGmsh import Results` (also
`from apeGmsh.results import Results, make_demo_results`).

## 1. Constructors — read a `.ladruno`

apeGmsh is converging on one results format, the `.ladruno` (ADR 0131).
Open a run with `Results.from_ladruno`. A stock OpenSees `.mpco` gets
there through `convert_mpco`. The native `results.h5` path
(`from_native`, `to_native`) and `from_mpco` still work today, but
both are being retired: new code should not write or read them.
apeGmshViewer is `.ladruno`-only by decision (ADR 0131 D1); its
native `<stem>.results.h5` slot goes at slice V1d (D2).

| Constructor | Required kwarg | Source |
|---|---|---|
| `Results.from_ladruno(path, *, fem=None, merge_partitions=True, model_h5=None)` | *(none — self-sufficient)* | a `.ladruno` (fork recorder, `convert_mpco`, `EigenResult.results()`, `Results.demo()`), single or `part-N`; pass `model_h5=` for stage names, PGs and labels |
| `Results.from_recorders(spec, output_dir, *, fem, model=...)` | `fem=` **and** `model=` | `.out`/`.xml` from a Tcl/Py recorder run; transcodes + caches: nodal records over the program's first `ops.stage` blocks (same names and kinds, in order) → a `.ladruno`; anything else → native |
| `Results.from_mpco(path, *, fem=None, merge_partitions=True, model_h5=...)` | `model_h5=<path>` | STKO `.mpco` (single or `part-N`). **Being retired** (ADR 0131 D5): convert instead |
| `Results.from_native(path, *, fem=None, model=...)` | `model=OpenSeesModel` | apeGmsh native `results.h5`. **Being retired** with `to_native` (ADR 0131 D10) |

```python
from apeGmsh import Results
from apeGmsh.results import convert_mpco

# A .ladruno is SELF-SUFFICIENT (geometry, regions, local axes are in the
# file), so model_h5= is optional. Pass it anyway when you have it: the
# ops.stage names, physical groups and labels come from the archive.
results = Results.from_ladruno("run.ladruno", model_h5="model.h5")
results = Results.from_ladruno("run.part-0.ladruno")          # parts merged

# Stock OpenSees: the deck records an .mpco; convert it once, then read
# the .ladruno. model_h5= (the ops.h5 archive) is required: the stage
# kinds come from its program. The .mpco is never modified.
convert_mpco("run.mpco", "run.ladruno", model_h5="model.h5")   # -> [Path]
results = Results.from_ladruno("run.ladruno", model_h5="model.h5")

# Recorder .out/.xml run — needs BOTH fem= and model=.
results = Results.from_recorders(spec, "out/", fem=fem, model=model)
# results.path is <cache>/<key>.ladruno (stamped against <key>.model.h5,
# opensees_backend="unknown", stage_names = the plan's) for nodal records
# whose every stage (stage_id= / stages=) is one of the program's ops.stage
# blocks with its kind, in any order or subset: a reopen binds the stamped
# names. results.model is the model= you passed. Otherwise <key>.h5
# (native), which keeps the names: a stage the program does not declare
# or of another kind, a program with no ops.stage, a FEMData archive, any
# element record, displacement_increment.
# verified: tests/test_results_transcoder.py::test_a_nodal_run_transcodes_to_ladruno_read_as_the_native_file
# verified: tests/results/test_ladruno_stamped_stage_names.py::test_a_reordered_plan_is_a_ladruno_bound_by_name
```

**Stage names on a `.ladruno` come from its stamp, never from position**
(ADR 0122 D3, #2823). With `model_h5=`, a file stamped with
`INFO/PROVENANCE@stage_names` binds each stage to the program stage its
stamp names, in any order; a stage stamped with its own `MODEL_STAGE[k]`
name (a modes stage) keeps it. A stamped name the program does not
declare, or a program stage of another kind than the recorded `KIND`,
raises `StageStampRefused`. An unstamped file pairs one to one with the
program when it holds as many stages with steps; with fewer, it pairs
with the program's first stages only when the `KIND`s fit no other run of
program stages, and otherwise keeps its `MODEL_STAGE[k]` names with
`UnpairedStageWarning` (a recorder opened in a later stage). A modes-only
stage (`KIND="eigen"`, no step) is never a program stage. Every Python
writer stamps the names; a writer that knows which program stages a file
holds passes them, `stamp_provenance(path, "model.h5",
recorded_stages=["pushover"])`.
(`tests/results/test_ladruno_stamped_stage_names.py`)

A `.ladruno` that Python writes (`from_recorders`, `EigenResult.to_ladruno`,
`ParallelModalResult.to_ladruno`, `Results.demo()`) stamps
`INFO/PROVENANCE@opensees_backend="unknown"`: it names the fork's engine
only where the fork provably solved the values (`capture_live_modes` on
the fork), and never `"stock"`, which is kept for a converted `.mpco`.

`convert_mpco(src, dst, *, model_h5, overwrite=False)` converts a
partitioned set too (`run.part-0.mpco` writes `run.part-0.ladruno` …),
returns the written paths, and is atomic: either the whole new set is
in place, or nothing changed. It refuses with `MPCOConversionRefused`
when the stages cannot be paired with the program's `ops.stage` blocks
(an unstaged program refuses), when the archive is another model's, or
when a target exists without `overwrite=True`. What it cannot carry is
listed in the file's `INFO/PROVENANCE@unsupported`: `local_axes`,
`springs` (copied, but not read by `from_ladruno`), and
`quadrature:<group>` for an element group with no known rule.
(`tests/test_results_mpco_to_ladruno.py`)

Modes come back the same way: `EigenResult.results()` writes the modes
to a stamped `.ladruno` and opens it (`results.modes` lists one
mode-scoped `Results` per eigenvalue); `EigenResult.to_ladruno(path)`
keeps the file. `to_native` remains only as an explicit native export.

**Pairing.** A `.ladruno` written by apeGmsh is stamped in
`INFO/PROVENANCE` with the hashes of the model archive it came from. A
bridge archive (`ops.h5`) stamps both `model_hash` and `fem_hash`, and
pairs only with an archive that matches both. A bare FEMData archive
(`fem.to_h5`) has no `model_hash`; ADR 0133 lets it pair by `fem_hash`
alone, which reaches the stamp with W1b (#2807). Until then the stamp
refuses a FEMData archive. `from_ladruno(model_h5=<another model>)`
warns `ModelFemMismatchWarning` and binds the file's own mesh instead.

The retiring constructors still require their model broker; omitting
it raises `TypeError`:

```python
Results.from_native("run.h5")        # TypeError: model= is required
Results.from_mpco("run.mpco")        # TypeError: model_h5= is required
Results.from_recorders(spec, "out/", fem=fem)   # TypeError: model= is required
# verified: tests/test_results_bind.py::test_from_native_without_model_raises_typeerror
# verified: tests/test_results_bind.py::test_from_mpco_without_model_h5_raises_typeerror
# verified: tests/test_results_bind.py::test_from_recorders_without_model_raises_typeerror
```

Notes:
- In-process runs: call `ops.wipe()` before opening the files. It flushes and closes every
  recorder, so a text file holds every row and an `.mpco` is complete. Recorders take
  `precision=<int>` (significant digits per value). A deck run with `ops.py/tcl(run=True)`
  returns a `RunReport`, and a failed run raises.
- `from_recorders` also requires `fem=` (a separate `TypeError`) — the
  spec's `snapshot_id` is part of the transcode cache key. Nodal,
  Gauss-point, line-station, nodal-force and fibre records are read;
  a layers record refuses by name, and a modal record is skipped (no
  `.out` emission).
- `from_mpco` over a **composed** model (ADR 0038/0043) auto-attaches
  an `ElementTagTranslator` so element queries speak `fem_eid`, not the
  raw OpenSees ops tag. (`tests/test_results_mpco_composed_join.py::test_query_by_offset_fem_eid_returns_the_row`)
- `from_ladruno` is the **exception** to the model-required rule: a
  `.ladruno` is the fork's *canonical* recorder and self-describes its
  geometry, so the broker is built from the file (`model_h5=` optional,
  only for richer bridge records / lineage). Reader specifics:
  `results.elements.gauss`/`line_stations` speak the **neutral** vocabulary
  (`stress_xx`, `axial_force`, …) and work cross-backend; `results.elements`
  is **token-driven** — the component is the file's `ON_ELEMENTS/<token>`
  key (`basicForce`/`localForce`/`force`/`globalForce`) returning the raw
  block. Beam diagrams orient from the recorder's `MODEL/LOCAL_AXES`
  (`results.elements.local_axes(...)`; `line_force` uses it for true
  cross-section roll). Energy: `results.energy(region=)`. ASDPlasticMaterial3D's
  material-level responses (`material.pstrain`, `material.PStress`, …) land on
  `results.elements.gauss` under canonical names too (ADR 0105 Amendment 1) — see
  `references/ladruno.md` for the fork-only details.
- Zero-setup: `Results.demo()` / `make_demo_results(...)` — see §6.

## 2. The three-broker chain — `model`, `model.fem`, `fem`

ADR 0019–0023. A constructed `Results` always carries an
`OpenSeesModel` broker; reach the neutral `FEMData` *through* it.

```python
osm   = results.model        # OpenSeesModel broker — ALWAYS non-None      # verified: tests/test_results_bind.py::test_lineage_propagates_from_model
fem   = results.model.fem    # FEMData neutral zone (chain-forward path)
local = results.fem          # locally-bound snapshot (Optional; may differ after .bind())
```

- `results.model` is non-None on any constructed `Results` (ADR 0020
  INV-1). It is the chain-forward handle.
- `results.model.fem` is the canonical FEMData. Prefer this for
  labels/PGs.
- `results.fem` is the *bound* snapshot — `None` until bound, and may
  diverge from `results.model.fem` after a `.bind()`.

## 3. Lineage — warns, never raises (BindError is gone)

`BindError` was **deleted** in Phase 8. There is no construction-time
hash rejection — pairing the right `fem`/model with a run is the
**user's** responsibility. Provenance surfaces through `results.lineage`.

```python
lin = results.lineage        # Lineage(fem_hash, model_hash, results_hash, warnings)
lin.warnings                 # tuple[str, ...] — "[lineage] ..." drift strings; () = clean
lin.assert_clean()           # opt-in: raises LineageError if warnings non-empty
# verified: tests/test_results_bind.py::test_lineage_propagates_from_model
```

- `results.lineage` **never raises** (ADR 0021 INV-2). A stale FEMData
  paired with a fresh results file shows up as a `"[lineage] ..."`
  string in `lineage.warnings`, not an exception.
- `Lineage` is a dataclass:
  `Lineage(fem_hash: str = "", model_hash: Optional[str] = None, results_hash: Optional[str] = None, warnings: tuple[str, ...] = ())`
  (`src/apeGmsh/opensees/_internal/lineage.py:117`).
- `LineageError` (subclass of `ValueError`,
  `src/apeGmsh/opensees/_internal/lineage.py:105`) is raised **only**
  by the explicit `lineage.assert_clean()` opt-in check.

Re-bind a fresh-session FEMData (to recover labels/Parts the embedded
snapshot lacks) — returns a NEW `Results`, no hash validation:

```python
r2 = results.bind(fem)       # no validation; pairing is your job   # verified: tests/test_results_bind.py::test_bind_accepts_mismatched_fem
disp = r2.nodes.get(pg="Top", component="displacement_z")           # verified: tests/test_results_bind.py::test_pg_query_works_after_bind
```

If results look wrong / PG selectors come back empty, suspect a stale
FEMData bound to a fresh results file — the deleted `BindError` would
have caught it; now it's on you.

### 3b. After-solve check — assess (not Qt)

Agents do **not** open `results.viewer()` / `show_web()` to decide
whether a run is sane. Those windows are for humans who asked to look.
After `get_fem_data` / `Results.from_*`:

```python
report = results.assess(figures=True)   # verdict + stills; figures=False is default
print(report.text)
for path in report.figures:             # () under SKIP_VIEWER / no GL
    pass                                # read_file each PNG
# act on finding.severity == "error"; do not proceed to a human memo
```

Catalog, skip ≠ pass, and code → next action: **`assess.md`**.
`inspect` / `diagnose` stay inventory (tables, one empty query).
`results.lineage.warnings` still never raises; assess surfaces them as
`RES.LINEAGE`. Open a Qt / web viewer only when the human asked.

## 4. Querying — stages, nodes, elements, modes

```python
results.stages               # list[StageInfo]
s = results.stage("dynamic") # scope to one stage -> a new Results    # verified: tests/test_results_modes.py::test_modes_accessor_returns_scoped_results
```

Node / element reads return a typed slab. Selectors are by **label /
PG / selection / id** (never raw tags); `component=` is keyword-only:

```python
slab = results.nodes.get(pg="Body", component="displacement_x")      # verified: tests/test_results_modes.py::test_mode_shape_is_single_step
slab.values, slab.time, slab.node_ids
```

`nodes.get` signature (`_composites.py:883`) — all selectors keyword-only:

```
get(*, pg=None, label=None, selection=None, ids=None,
    component, time=None, stage=None) -> NodeSlab
```

`results.elements.get(...)` mirrors it (→ `ElementSlab`) and additionally
owns the sub-composites
`results.elements.{gauss, fibers, layers, line_stations, springs}`
(each a `.get(...)` returning `GaussSlab` / `FiberSlab` / `LayerSlab` /
`LineStationSlab` / `SpringSlab`; `gauss`/`fibers`/`layers` also take
`gp_indices=` / `layer_indices=`). For exact piecewise-quadratic shear
$V(x)$ and piecewise-cubic moment $M(x)$ beam span diagrams under uniform,
partial-span, or trapezoidal loads (including `LadrunoMembrane` tributary
sub-segment loads), use `from apeGmsh.results import reconstruct_beam_span_diagrams, reconstruct_beam_element_diagram`
(inserts analytical breakpoints $a_L, b_L$ and interior zero-shear extrema $V(x)=0$).

**Modes / eigen:**

```python
results.modes                # list[Results], one per kind="mode" stage   # verified: tests/test_results_modes.py::test_modes_accessor_returns_scoped_results
sorted(results.modes, key=lambda m: m.mode_index)   # stable index order
results.eigen_modes          # list[EigenMode] lightweight snapshots
```

Component names expand DOF-aware (`"displacement"` → `displacement_x/y/z`
clipped to `ndm`; `"stress"` → 6 in 3D / 3 in 2D); unknown names raise
`ValueError`.

### Custom scalar expressions — `define` (ADR 0076)

Register a **custom scalar** as a restricted arithmetic expression over a
composite's own `available_components()`. It becomes a first-class
`component=` everywhere that composite reads (computed on read, like the
`von_mises_stress` derived scalars) **and** shows up in the desktop /
web viewer's scalar picker — no viewer-side setup.

```python
# node domain — combine node fields (v^2 + u, a kinetic-energy-like scalar)
results.nodes.define("kinetic_ish",
                     "velocity_x**2 + velocity_y**2 + displacement_x")
results.nodes.get(component="kinetic_ish", pg="deck", time=-1)      # NodeSlab

# gauss domain — compose on an existing DERIVED scalar (demand/capacity)
results.elements.gauss.define("dcr", "von_mises_stress / 250.0", units="-")
results.elements.gauss.get(component="dcr", pg="wall")

# vector magnitude helper — Euclidean norm of a nodal vector family
results.nodes.define("speed", "mag(velocity)")     # sqrt(vx^2+vy^2+vz^2), 2-D → in-plane

results.nodes.definitions          # {name: ExprDef}
results.nodes.undefine("kinetic_ish")
"kinetic_ish" in results.nodes.available_components()   # -> True (feeds the picker)
```

**Registration is per-composite → the domain is implied.** `define`
lives on `results.nodes` and `results.elements.gauss` (v1). A node
expression can only reference node components; you cannot mix a node
velocity with a Gauss stress (different point sets). Operands resolve
against that composite's `available_components()` — stored columns,
derived scalars (`von_mises_stress`, `principal_stress_1`, …), and
earlier custom names all in scope.

**Grammar** (restricted AST over numpy — never `eval`): `+ - * / // % **`,
comparisons, bitwise `& |` (combine comparison results), numeric
literals, the operand names, `mag(<vector-family>)`, and the elementwise
functions `sqrt abs exp log sin cos tan sign hypot minimum maximum clip
where`. **Gotchas that raise at `define()`:**

- `and` / `or` / `a if c else b` are **rejected, not lowered** (they
  dispatch on array truthiness → ambiguous-truth error). Use
  `where(cond, a, b)` and `& / |`, e.g.
  `(von_mises_stress > 250) * von_mises_stress`.
- `min` / `max` are **not** exposed (numpy's collapse the field; Python's
  are variadic). Use 2-arg `minimum` / `maximum`.
- `mag(x)` takes a **bare nodal vector family** (`velocity`,
  `displacement`, `acceleration`, `force`, `rotation`, `moment`, …) — not
  a component (`mag(velocity_x)`), not the combined `reaction` shorthand,
  not an expression (`mag(velocity + 1)`).
- Fail-loud at `define()`: parse errors, unknown operands, and a name
  that shadows a stored / derived / already-registered component all
  raise. `undefine` refuses to strand a dependent definition. Operand
  coverage mismatch (two fields on different point sets) raises a legible
  `ExprError` at read.

**Persistence + reach:** `results.save_definitions()` writes a
`<results>.defs.json` sidecar; `from_native` / `from_mpco` /
`from_ladruno` **auto-load** it at open (best-effort — a stale operand
warns and is skipped). The same sidecar carries definitions to the
`viewer(blocking=False)` subprocess. Definitions also **follow
`.stage()` / `.modes` derivation**.

```python
results.save_definitions()          # -> <results>.defs.json (also persists on viewer spawn)
# ... later session ...
r = Results.from_ladruno("run.ladruno", model_h5="model.h5")   # auto-loads the sidecar
r.stage("dynamic").nodes.definitions             # carried across derivation
# verified: tests/test_custom_scalar_expressions.py
```

### Combinations and the user file — `<stem>.user.h5` (ADR 0130)

A results set is the model archive, the recorded run, and one file of
what you derive from them:

```text
<stem>.h5        model archive (ops.h5), + /opensees/combinations
<stem>.ladruno   the recorded run
<stem>.user.h5   /combinations, /results, /modes, /spectra, /entities
<stem>.workspace.h5   named sets, plots, saved views (below; ADR 0132)
<study>.variants.json lists several sets of one study (below)
```

You write `<stem>.user.h5` from Python with one call. apeGmshViewer reads
it and does not write it today. ADR 0130 plans for the app to write the
`/combinations` zone and nothing else.

```python
import numpy as np
from apeGmsh import Results
from apeGmsh.results.user import (
    Combination, StateAddress, Term, UserFile, UserResult, write_user_file,
)

results = Results.from_ladruno("run.ladruno", model_h5="run.h5")
ids = np.asarray(results.model.fem.nodes.ids, dtype=np.int64)
ux = np.stack([results.stage(s).nodes.get(component="displacement_x", time=-1).values[-1]
               for s in ("gravity", "live")])                     # (states, nodes)

write_user_file(
    "run.user.h5", "run.h5",              # name must end .user.h5; hashes come from run.h5
    results_file="run.ladruno",           # its fingerprint stamps the derived zones
    combinations=[
        Combination("ULS", "linear", (Term("stage", "gravity", factor=1.2),
                                      Term("stage", "live", factor=1.6))),
        Combination("ENV", "envelope", (Term("combination", "ULS"),
                                        Term("stage", "wind")), envelope="absmax"),
    ],
    results=[UserResult(
        name="drift_ratio", target="nodes", basis="results", components=("drift_ratio",),
        states=(StateAddress("stage", "gravity"), StateAddress("stage", "live")),
        linear=True, data=(ux / 3.0)[:, :, None], ids=ids)],     # DATA (states, nodes, comps)
)

combos = results.combinations()           # finds run.user.h5 beside the set
combos.names                              # .user.h5 rows, then model.h5 rows
uls = combos.evaluate("ULS", "displacement_z")             # a recorded quantity
uls.field.values, uls.field.keys["id"], uls.signed, uls.badges
combos.evaluate("ULS", "axial_force", level="line_stations")
combos.evaluate("ULS", user_result="drift_ratio")           # a user result
UserFile("run.user.h5").results().entries                   # what the file holds
# verified: tests/results/user/test_user_evaluate_results.py
```

- **A combination is a descriptor, evaluated on read, per quantity.**
  Rules: `linear` (factored sum, signed), `srss`, `cqc` (`damping=`),
  `abs_sum`, `directional` (`fraction=`, e.g. 0.3), `envelope`
  (`envelope="max"|"min"|"absmax"`). Only `linear` is signed, so only
  it draws as a deformed shape; `linear` over a stage carries the badge
  `"superposition assumes linearity"`.
- **A term** is `Term(kind, source, step=-1, mode=0, factor=1.0)`, with
  `kind` one of `stage`, `mode`, `spectrum`, `reference`, `combination`.
  A stage is named by its `ops.stage` name, else `stage_<i>`, never
  `MODEL_STAGE[k]`; open the results with `model_h5=` so the names are
  bound. `step=-1` is the last step.
- **Refusals.** `write_user_file` refuses the whole write
  (`UserWriteRefused`, nothing written) when a row is bad: a duplicate
  name, a cycle, an unknown rule, a bad term. Reading a file someone else
  wrote is gentler: a bad row refuses alone and the file still opens.
  `evaluate` raises `CombinationRefused` with a `.reason`: `unresolved`
  for a term naming a stage that is not there, or a stage with no
  recorded steps (an `eigen` stage); `not-linear` when a summing rule
  meets a user result without `linear=True` (A5; a `linear=True` result
  must also store a row for every state the combination names, or that
  term is `unresolved`), a derived scalar such
  as von Mises, or a design field stored in a native `results.h5`
  (`wood_armer_*`); `state-free` for a user result with `states=None`;
  `stale` for a stale input; `foreign-handle` for another run's state.
- **Declared in the bridge.** `ops.combination(name, terms, rule=...)`
  archives the row in `model.h5` `/opensees/combinations`; a term is
  `(stage_name, factor)`, `(combination, factor)`, or a `CombinationTerm`
  (`from apeGmsh.opensees.apesees import CombinationTerm`) for a step or
  a mode. A `spectrum` or `reference` term is refused there: its case
  lives in `.user.h5`, so declare that combination after the run. A
  `.user.h5` row may name a `model.h5` row, never the reverse; names are
  unique across both.
- **Atomic and checked.** Each write builds a private `.partial`, reads it
  back, and replaces the file only if nobody else changed it meanwhile
  (the second writer gets `UserWriteRefused`). `combinations=` adds a
  generation and keeps the earlier ones as the backup; `results=` adds or
  replaces entries by name; `None` leaves a zone untouched.
- **Staleness.** `staleness("run.user.h5", model_h5=..., results_file=...)`
  reports each zone. `/modes`, `/spectra` and `/entities` go stale when
  the model hashes change; a `/results` entry follows its `basis`
  (`"results"`: the `.ladruno` fingerprint; `"model"`: the model hashes).
  Combinations never go stale; a stale input is refused unless
  `evaluate(..., allow_stale=True)`.

The other zones are filled from what the analysis already returns:

```python
from apeGmsh.results.user import Entities, mode_table, spectrum_cases, write_user_file

modal = ops.modal_properties(6)                        # ModalPropertiesResult
sp = ops.spectrum_after_gravity(...)                   # SpectrumModalResult (opensees-bridge.md)
write_user_file(
    "frame.user.h5", "frame.h5",
    modes=mode_table(modal, host="stage_0"),           # period, participation, mass ratios
    spectra=spectrum_cases(sp, ["E_x"], damping=0.05, nodal=["disp"]),
    entities=Entities(cuts=(cut.to_entity(name="pier A"),)),   # cut: apeGmsh.cuts.SectionCutDef
)
# verified: tests/results/user/test_user_modes_spectra_live.py, test_user_entities.py
```

A spectrum term is `Term("spectrum", "E_x")` for the whole case (under
`srss` / `cqc`) or `Term("spectrum", "E_x", mode=k)` for one mode's
increment; `Term("reference", "E_x")` is the case's preloaded state. So
`1.0 D + E` is a `linear` row over a stage and a `cqc` row. Spectrum
terms read `/spectra` through `UserSpectra`, the default of
`results.combinations()`.

### What the user makes — `<stem>.workspace.h5`, and variants (ADR 0132)

Named sets, plot definitions and saved views live in a fourth file of the
set, `<stem>.workspace.h5`, never in `.user.h5`. It holds definitions
only, never a result value. Today you write and read it from Python;
ADR 0132 plans for the app to write it when the human saves:

```python
from apeGmsh.results.user import (
    SetDef, VariantDef, ViewDef, WorkspaceFile,
    read_variants, write_variants, write_workspace,
)

write_workspace(
    "frame.workspace.h5",                       # pairs with frame.h5 beside it
    sets=[SetDef("roof", "nodes", roof_node_ids),
          SetDef("piers", "elements", pier_ops_tags)],     # OpenSees element tags
    views=[ViewDef("roof iso", {"name": "roof iso", "camera": "iso"})],  # one apegmsh-view/1 shot
)
ws = WorkspaceFile("frame.workspace.h5")
ws.set("roof").ids                              # the members, in your order
[(v.name, v.verdict) for v in ws.verdicts().verdicts]   # current | moved | partial | lost
write_workspace("frame.workspace.h5", remove={"views": ["roof iso"]})
```

- A given row replaces the row of the same name; every other row is
  copied verbatim. Like `.user.h5`, the write is atomic and refuses when
  another writer got there first. A tag the model lacks refuses the
  write (`WorkspaceWriteRefused`). An unreadable old file is renamed
  aside with a `WorkspaceRenamedAside` warning, never overwritten.
- **Verdicts.** Each member's coordinates are stored with the set (a
  node's position, an element's centroid). When the model is re-meshed
  or re-declared, `verdicts()` looks for each member where it was:
  `current` (the hashes still match), `moved` (every member found under
  a new tag), `partial`, or `lost`. A position that two items fit counts
  as not found.

A study of stiffness states (uncracked, cracked, Bischoff) is several
sets listed in one `<study>.variants.json`. Each row points at a
variant's model file. ADR 0132 plans for the app to open a variant,
only from a manifest the user opened; today the manifest is written and
read from Python:

```python
write_variants("slab.variants.json", [
    VariantDef("Uncracked", "slab_uncracked.h5", {"state": "uncracked", "Ie/Ig": 1.0}),
    VariantDef("Bischoff", "bischoff/slab_pass2.h5", {"pass": 2}),
])
m = read_variants("slab.variants.json")
m.by_label("Bischoff").resolved                 # absolute Path inside the study folder
m.refused                                       # rows that refuse alone, with reasons
```

`path` is relative to the manifest's folder, uses `/`, and never leaves
it (`..` and absolute paths refuse). It names a `<stem>.h5`, never a
sibling such as `.ladruno` or `.user.h5`. A row may name a file that a
later run will write: the write accepts it, and each read refuses that
row with reason `missing` until the file is there. A combination
never crosses variants (a term holds no path).
(`tests/results/user/test_workspace_file.py`, `test_variants_manifest.py`)

### Footfall vibration map — `FootfallResult.to_results` (ADR 0109, ADR 0131 D6)

`apeSees.footfall_walking(num_modes=..., body_weight=..., g=..., response_nodes=...)`
(both `body_weight` and `g` are required, model units — no defaults)
runs a walking-vibration check per AISC Design Guide 11 2nd ed. Chapter 7
and returns a `FootfallResult`. `result.to_results(fem, "<stem>.user.h5", model_h5=...)`
writes it into the model's user-results file (ADR 0130) as one
**state-free** user result named `"footfall"` on the nodes: components
`footfall_ap` (fraction of g), `footfall_ratio`, `footfall_fdom` (Hz),
`DATA (1, N, 3)` over every node of `fem`, `NaN` off the response nodes.
No native `results.h5` is written. `model_h5` is required (no default) and
must be the archive of the same `fem` (`ops.h5(...)`, whose `fem_hash` is
`fem.snapshot_id`); another fem's archive, a `fem` with no nodes, a
response node outside `fem`, or a name not ending `.user.h5` raise
`UserWriteRefused` and write nothing. An existing `.user.h5` keeps its
other entries. `/meta@writer` names the driver and the OpenSees backend.

```python
from apeGmsh.results.user import UserFile
ops.h5("bay.h5")
result.to_results(fem, "bay.user.h5", model_h5="bay.h5")
ff = UserFile("bay.user.h5").read_result("footfall")   # UserResult; ff.states is None
ratio = ff.data[0, :, ff.components.index("footfall_ratio")]   # per fem.nodes.ids
# verified: tests/opensees/unit/test_footfall_user_h5.py
```

Evaluate `response_nodes=ops.nodes.get(pg="Slab")` (self excitation at
every slab node) when the map is the point. The driver, its rules and the figure accessors
(`frf`, `mode_table`, `to_dataframe`) are in
[opensees-bridge.md](opensees-bridge.md) §Footfall; worked floor
`examples/footfall_two_bay_shell.py`; how-to
<https://nmorabowen.github.io/apeGmsh/how-to/footfall-vibration/>.

## 5. Plots and the desktop viewer

**Stills for an agent.** Write them under `.apegmsh/visors/` and read
the PNGs. On Windows take them with `render_stills`
(`apeGmsh.agent_view`; apeGmshViewer's batch stills, ADR 0128;
`assess.md`). The VTK stills (`results.render` / `fem.render`,
ADR 0094) are for Linux, where the app has no batch stills yet.
Labeled `results.plot.*` works anywhere.

### Static matplotlib (`[plot]` extra)

```python
results.plot.contour("displacement_z", step=-1)          # ResultsPlot.contour, _plot.py:159
results.plot.deformed(step=-1, scale=50, component="stress_xx")
results.plot.history(node=412, component="displacement_x")
```

`results.plot` is the headless renderer. Methods: `mesh / contour /
deformed / history / vector_glyph / reactions / loads / line_force`.
`ImportError` if the `[plot]` extra (matplotlib) is absent.

### Interactive desktop viewer — `blocking=None` auto-detect

Qt viewers are for **humans who asked to look**. After a solve, an
agent's check is `results.assess()` (§3b / `assess.md`), not a
window.

`viewer()` is sugar for `results.session().show()`: the window projects
a **`ResultsSession`** — docked mesh and plot panes, result slots per
pane.

```python
def viewer(self, *, blocking=None, title=None,
           restore_session="prompt", save_session=True)
```

- The **default is `blocking=None` (auto)**: `True` (in-process Qt) in
  scripts / the plain CLI; `False` (subprocess) inside a Jupyter /
  IPython ZMQ kernel. A hand-built, in-memory Results (no backing file)
  in a notebook raises `RuntimeError` naming the `from_*` constructors;
  open it from a file first (no web fallback, no temp `.h5`). The
  notebook spawn prints one `[viewer]` line.
- An **explicit** `blocking=True` still runs the VTK+Qt event loop
  in-process and **native-crashes a Jupyter / VS Code kernel**.
- **Returns differ by path, on purpose**: blocking returns the
  `ResultsSession` (still live after the window closes — query it,
  `render()` off it, `snapshot()` it); `blocking=False` returns the
  `subprocess.Popen`, as does auto mode in a notebook.
- `viewer(blocking=False)` always spawns a subprocess; needs a Results
  opened from disk, raises `RuntimeError` for in-memory.
- `viewer()` already shows the window — never chain
  `results.viewer().show()`; that would open a second one on the
  returned session.
- `save_session=True` (default) writes `<results>.viewer-session.json`
  on close; `restore_session` is `True` / `False` / `"prompt"`
  (default). A session file that cannot be read never blocks the
  window: it is announced, the default picture boots, and auto-save is
  disarmed for that window so the file survives. A v13 file from the
  retired window is renamed to `…​.viewer-session.json.legacy`, never
  overwritten.
- `cuts=` is **retired** (ADR 0098 §1). A cut plane is clip state on a
  view: build defs with `apeGmsh.cuts`, then
  `results.session()` → `view.add_clip(normal, offset=…)`.
- `APEGMSH_SKIP_VIEWER=1` makes the viewer call return `None` — lets a
  cell survive `jupyter nbconvert --execute` / CI. Checked before
  anything touches disk, so a skipped call never renames a session file.
- Higher-order solids (tet10/20/27, tri6, quad8/9, prism15, Bézier fork
  elements) render as their linear corner cell (mid-side nodes dropped).
- **Not in the session window:** there is no **File** menu — no *Open
  results…*, *Save screenshot…*, *Export animation…* or *Preferences…*
  entry (screenshot survives as a toolbar action on the active pane;
  `apeGmsh.viewers.settings()` is the preferences door). Also absent:
  the draggable / right-click colour-scale chrome, the probe overlay,
  the threshold UI, and the outline's eye-cascade. Scales are per-pane
  and caused by slots — `INV-LEGEND-1..5` — but the legend interactor
  is not installed; `view.set_legend_hidden(field)` is the gesture.

### apeGmshViewer from the command line (ADR 0128)

apeGmshViewer is the installed desktop app (`apeGmsh.agent_view.find_viewer()`
returns its path: `$APEGMSH_VIEWER`, else the per-user Windows install).
Besides opening a set for the human, it takes four agent commands. None
of them touches the human's window except `--present`, which only adds
an offer:

| Command | What you get | Exit |
|---|---|---|
| `apeGmshViewer run.h5` | the human's window on the set (any file of the set opens it) | — |
| `--render spec.json --out <dir>` | one PNG per shot of an `apegmsh-view/1` spec, drawn hidden, plus `manifest.json` (written last) | 0 all drawn, 1 a shot refused or the run failed, 2 spec or `--out` refused |
| `--describe run.h5 --out <dir>` | `describe.json`: the stages, steps, fields and groups a spec may name | |
| `--present spec.json` | an offer listed in the running window; the view changes only when the human clicks it | 0 acknowledged, 1 no window answered, 2 spec refused |
| `--snapshot --out <dir>` | the running window's current view: `view.json`, `view.png`, `manifest.json` | 0 written, 1 no window or no read-back, 2 `--out` refused |

`--out` must be absent or empty. From Python, `render_stills(spec, out)`
runs `--render` and returns the manifest, raising `AgentViewError` with
every refusal (`assess.md`). Quote numbers from the manifest, not from
the pixels.

### Headless animation export — `results.export_animation(...)` (#755)

```python
results.export_animation("run.mp4", fps=30, step_stride=2,
    deform=("displacement", 50.0),        # or a bare number → "displacement" at that scale
    stage=None, camera=None, window_size=(1280, 720))
```

Renders the time history to **`.mp4`** (H.264 — needs the
`apegmsh[animation]` extra) or **`.gif`** (Pillow, no extra); the path
suffix picks the format. It re-steps one `ResultsSession` mesh view
(the `results.session()` picture) through an off-screen plotter — no
Qt window, no event loop: the pane is realized once, then each captured
step moves the session instant and re-steps it in place.
`step_stride=N` captures every N-th step plus the last; `deform=None`
(default) renders undeformed; `stage=None` animates the last stage;
`camera=None` is `xy` for a planar model, `iso` otherwise. `setup=`
takes a `callback(plotter, view)` invoked once before capture, where
`view` is the session `MeshView` — fill a slot there
(`view.contour = Contour("displacement_x")`). Before X2-b (#1663) the
second argument was a `ResultsDirector`; a callback still written
against it raises `TypeError`. A failed frame leaves no file at the
path. It is script-only — the session window has no File menu and no
export path. `# src/apeGmsh/results/Results.py:1544`

### Per-pane state — the `MeshView` (ADR 0098 §3–§7)

The picture belongs to the pane. A session is docked panes; each mesh
pane is a `MeshView` carrying its own **scope** (ONE composition axis —
`physical_groups` / `materials` / `element_types` — plus the names
checked on it, never a boolean `concrete AND hex`), **pose**
(`view.deform`, a field + scale; a pose, not a slot, and it causes no
legend), **time** (`view.time`, an `Instant(stage, step)`), style, clips
and slots. Comparing two states is two panes — a second deform state is
not a second scene instance. On a **staged** model (`ops.stage(...)`
with `activate` / `remove_element`) a mesh pane draws only the elements
active in the stage of its instant, intersected with its scope: stills,
`export_animation`, the CLI `render` and the window alike (#1722). A
scope with no active element in that stage raises rather than drawing
the full mesh.

```python
from apeGmsh.results.session import Contour, Deform, Instant, Scope

s = results.session()                             # the document, no window
v = s.panes[0]                                    # the booted empty mesh view
v.scope = Scope("physical_groups", ("wall",))     # one axis + names
v.deform = Deform("displacement", 50.0)           # pose (Deform(mode=3) = mode pose)
v.contour = Contour("stress_von_mises")           # fills the contour slot
v.time = Instant("dynamic", 12)                   # this pane's instant
v.add_clip((1.0, 0.0, 0.0), offset=2.5)           # section plane, view-owned

s.add_view()                                      # a second pane
s.time_linked = False                             # each pane scrubs alone
s.render("wall.png", v.id)                        # a still, no Qt
```

- **Slots are a closed catalog of seven** — `contour`, `vector`,
  `gauss`, `line`, `sand`, `loads`, `reactions`. At most one occupant
  per category per pane; different categories stack; filling an occupied
  slot **replaces** the occupant and `None` clears it. `v.slots` reads
  the occupied ones. A new category is an ADR 0098 amendment, not a
  subclass.
- **Legends are derived, never set** — `v.legends()` is a function of
  the occupied colour-mapped slots (`contour` / `vector` / `gauss` /
  `sand`); two slots on the same quantity share one scale
  (`INV-LEGEND-1..5`). Deform on with every slot empty draws no scale.
- **Colour-mapped slots take `clim=(lo, hi)` and `cmap=`** (ADR 0098
  A7.3) — `None` (default) keeps the kind's default; `cmap` is any
  matplotlib name (`'jet'` included), validated when the record is
  built. Two slots of the same quantity with different explicit values
  raise (one field, one scale); `v.legends()` reports the set values.
  Every realize applies them (stills, the window, `export_animation`);
  a same-quantity slot left at `None` draws with its neighbour's values.
- **`v.overlay = True` draws the undeformed ghost** (A7.5) — the view's
  own cells at scale 0 in the Outlines style, under the same scope,
  stage activation and clip planes; independent of the Outlines button,
  never colour-mapped, no legend.
- **Plot kinds are `history` and `xy`** — `path` was removed (A7.7).
- **Snapshot is v2** — `SNAPSHOT_VERSION = 2`, `kind:
  "apegmsh.results.session"` (`results/session/_snapshot.py`).
  `s.snapshot()` returns the dict, `s.save_snapshot()` writes it; the
  reader takes v1 and v2 (a v1 `path` plot pane drops with a notice).
  `restore_snapshot` refuses an unknown slot category. A v13 file from
  the retired window is renamed `.legacy`, never restored.

## 6. Web / Jupyter viewers (ADR 0042 R-C — newest)

Kernel-safe trame/pyvista. Need the **`[viewer]` extra** (trame +
trame-vuetify; `ipywidgets` for the inline controls).

```python
def show_web(self, *, stage=None, show=True, controls=True,
             render_mode="client")            # Results.py:1559
def serve_web(self, *, stage=None, render_mode="client", port=None,
              open_browser=True, title="apeGmsh", **start_kwargs)   # Results.py:1608
```

```python
from apeGmsh import Results

# Zero-setup sample: real apeSees-emitted model + synthetic cantilever push.
results = Results.demo(n_steps=6, tip_drift=2.0)        # verified: tests/test_results_demo.py::test_results_demo_classmethod

# In a notebook: inline trame view + ipywidgets step slider / layer toggles.
wv = results.show_web(render_mode="client")             # returns a WebViewer   # verified: tests/viewers/test_trame_backend.py::test_show_web_returns_viewer_without_display
wv.set_step(3)                                          # programmatic scrub + re-render   # verified: tests/viewers/test_trame_backend.py::test_web_viewer_step_clamps

# Outside a notebook: standalone vuetify3 web app (blocks until Ctrl-C).
# results.serve_web(render_mode="client", port=8080)
```

`render_mode` (a friendly alias, **not** the raw pyvista backend name) —
valid values are exactly:

| value | behaviour |
|---|---|
| `"client"` (default) | WebGL in the browser; fast camera |
| `"server"` | image-streamed from the kernel; laggy, for very large models |
| `"hybrid"` | toolbar toggle between the two |

Any other value raises `ValueError("Unknown render_mode ...")`
(`tests/viewers/test_trame_backend.py::test_unknown_render_mode_raises`).

`Results.demo(**kwargs)` delegates to `make_demo_results(*, length=10.0,
n_elements=8, n_steps=6, tip_drift=2.0, path=None)`
(`from apeGmsh.results import make_demo_results`). It writes two files
into `path` (default: a fresh temp folder): `demo_model.h5`, a **real**
apeSees model archive, and `demo_results.ladruno`, stamped against it,
then returns `Results.from_ladruno(..., model_h5=...)` of the pair. The
displacements are a **synthetic analytic cantilever shape** — no
OpenSees solve runs, so the stamp names the backend `unknown`.
`n_steps<1` / `n_elements<1` raise `ValueError`.
(`tests/test_results_demo.py::test_make_demo_results_shape`,
`::test_demo_rejects_bad_args`)

### Web-viewer gotchas

- `WebViewer` / `show_web` / `serve_web` are **not** top-level exports —
  reach them via `Results` methods or
  `from apeGmsh.viewers.web_viewer import WebViewer, show_web, serve_web`.
- A `WebViewer` requires a `Results` with **bound FEMData**, else
  `RuntimeError("WebViewer requires a Results with bound FEMData ...")`.
  Construct with `fem=` or call `results.bind(fem)` first.
- It is **view-only** at construction: it renders the substrate plus
  whatever diagrams the director already holds, and `show_web` /
  `serve_web` add **no** deformed/contour layer for you. `wv.director`
  is implementation of this hatch, not an authoring surface — no
  compatibility promise, and a picture built through it cannot enter a
  session snapshot, be pinned, or be rendered by the MCP `render` verb
  (ADR 0098 Amendment 2). A picture you can keep is a `ResultsSession`.
- `show(controls=True)` degrades **gracefully** if `ipywidgets` is
  missing (returns the bare view); `WebViewer.controls()` instead
  **raises** `RuntimeError` pointing at the `[viewer]` extra.
- Static fallback: if pyvista can't launch the in-notebook trame server
  (usually missing `nest_asyncio2`), `show()` emits a `RuntimeWarning`
  to `pip install nest_asyncio2` and degrades to a static image with no
  controls.
- `APEGMSH_SKIP_VIEWER` short-circuits both `show()` (returns `None`)
  and `serve()` (returns the *unstarted* server, never blocks) — the
  CI/headless guard.
- Picking is **off** on the web (`TrameBackend.supports_picking() ->
  False`); deferred to Phase R-D.

## 7. Quick reference

```python
# Construct (model= / model_h5= REQUIRED — except from_ladruno)
Results.from_ladruno(path, *, fem=None, merge_partitions=True, model_h5=None)  # the route
convert_mpco("run.mpco", "run.ladruno", model_h5="model.h5")   # stock on-ramp
Results.from_recorders(spec, out_dir, *, fem=fem, model=model)
Results.demo(n_steps=6, tip_drift=2.0)              # demo_model.h5 + demo_results.ladruno
# retiring (ADR 0131): from_native(path, *, model=...), from_mpco(path, *, model_h5=...)

# User file <stem>.user.h5 (ADR 0130)
write_user_file("run.user.h5", "model.h5", results_file="run.ladruno",
                combinations=[...], results=[...])          # atomic; returns Path
combos = results.combinations() ; combos.names
combos.evaluate("ULS", "displacement_z") ; combos.evaluate("ULS", user_result="drift")

# Broker chain
results.model            # OpenSeesModel — never None
results.model.fem        # FEMData neutral zone (use this)
results.fem              # locally-bound snapshot (Optional)

# Provenance (never raises)
results.lineage.warnings ; results.lineage.assert_clean()
results.bind(fem)        # new Results; no hash validation

# Query (labels/PGs, not tags)
results.stages ; results.stage("dynamic")
results.nodes.get(pg=..., component="displacement_z", time=None, stage=None)
results.elements.get(...) ; results.elements.{gauss,fibers,layers,line_stations,springs}
results.modes ; results.eigen_modes

# Custom scalar expressions (ADR 0076 — node + gauss composites)
results.nodes.define("speed", "mag(velocity)")          # mag() = vector norm
results.elements.gauss.define("dcr", "von_mises_stress / 250.0", units="-")
results.nodes.get(component="speed") ; results.nodes.available_components()  # feeds picker
results.save_definitions()          # -> <results>.defs.json (auto-loaded on reopen)
# NO and/or/if-else (use where + & |); NO min/max (use minimum/maximum)

# After-solve check (agents: assess, not Qt — assess.md)
report = results.assess(figures=True)
print(report.text)                       # lineage is RES.LINEAGE in findings
# inspect/diagnose stay inventory when you need a table or one empty query

# Render (humans who asked to look; default blocking=None auto-detects)
results.plot.contour("displacement_z", step=-1)    # [plot] extra
results.show_web()                                  # notebook-safe; [viewer] extra
results.serve_web(port=8080)                        # standalone web app
results.viewer()                                    # auto: block in scripts, subprocess in notebooks
results.export_animation("run.mp4", fps=30)         # headless video/GIF (§5)
```

Locking tests (run green this session): `tests/test_results_bind.py`,
`tests/test_results_modes.py`, `tests/test_result_chain.py`,
`tests/test_results_demo.py`, `tests/viewers/test_trame_backend.py`.
