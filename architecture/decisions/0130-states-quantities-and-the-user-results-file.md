# ADR 0130 — States and quantities: modes, spectra, user combinations and user results in a `<stem>.user.h5`

**Status:** Proposed (2026-11-01). The `.user.h5` zones and the combination
descriptor are an irreversible schema decision and go through the program's
architect pair before slice 1 (ADR 0112 D2).

## Context

apeGmshViewer reads one model and one results file per set (ADR 0112, ADR
0122 D1). What a run produces is wider than that. The YOO Guayaquil project,
the richest real user, already produces all of the following, and keeps most
of it outside any file the viewer reads (inventory from the YOO session,
2026-11-01):

- **Many modes.** SLE runs 60 modes after a P-Δ preload; the slab footfall
  run records 20 (cracked) or 160 (uncracked) modes with `recorder.Ladruno -N
  modesOfVibration`. The modal table (periods, participation, effective and
  generalised mass) lives in an SLE `.npz` or is computed and discarded
  (footfall `Σmφ²`).
- **Per-mode response spectrum.** SLE reads base, storey, drift, pier-cut,
  column and coupling-beam quantities per mode and per direction, then forms
  CQC and the 100/30 envelope per quantity in Python (`checks.Combiner`), the
  rule of ADR 0124 D6. None of it reaches a results file; the viewer export
  writes CQC envelopes as unsigned "static" stages, which reads as a deformed
  shape and is not one.
- **Load combinations.** Both YOO pipelines solve unit cases and superpose
  after the run. The slab writes each of its 7 combinations as a full copy
  stage in native `results.h5`; SLE keeps only check rows.
- **Custom results.** Wood-Armer and rib components, ACI 318 service
  deflection with `Ie/Ig`, CCIP-016 footfall `R`, D/C ratios, drift checks,
  pier and coupling-beam resultants, storey quantities, drift at plan points.
  The slab writes the mesh-based ones into native `results.h5` through
  `NativeWriter`; the rest go to CSV and JSON.
- **Stiffness states.** Cracked and uncracked slabs, the Bischoff states
  `D+SD → D+SD+0.25L → D+SD+L`, and the SLE ablation variants are each a
  separate run with modified stiffness. The Bischoff loop overwrites one
  `.ladruno`, so only its last state survives on disk.

Three decisions already bound the answer:

- ADR 0122 D2 freezes native `results.h5` and deletes it at slice 10, so the
  file YOO's combinations and custom results live in has no future.
- ADR 0122 D3/D4: the fork owns the `.ladruno` format, and apeGmsh writes
  `INFO/PROVENANCE` attributes into it "and nothing else".
- ADR 0112 D3: the app does not write files; direct editing is "a later ADR".

Storing user data in a group inside the `.ladruno` was considered and
rejected (Alternatives): HDF5 cannot write into an existing file atomically,
so a crash while writing user data can make a multi-gigabyte recorder file
unreadable, and the recorder rewrites the file on every run, which would
delete combinations a user authored.

## Decision

### D1 — Two axes: which state, which quantity

The app presents results along two independent choices.

**State** (the phase bar), in four families:

| Family | What it is | Source |
|---|---|---|
| **Stages** | what the run recorded: static steps, transient time | `.ladruno` |
| **Modes** | eigen shapes plus the modal table | `.ladruno` shapes; table in `.user.h5` (D6) |
| **Spectra** | per-mode response-spectrum increments, per case and direction | `.user.h5` (D5) |
| **User Combinations** | factored or ruled combinations of the above | `model.h5` and `.user.h5` (D4) |

**Quantity** (the contour and line-diagram pickers), in two groups:

| Group | What it is | Source |
|---|---|---|
| **Results** | what the recorder wrote: displacement, stress, section force | `.ladruno` |
| **User Results** | what the project computed: design fields, checks, entity quantities | `.user.h5` (D7, D8) |

Any state can be combined with any quantity that is defined on it. A User
Combination is evaluated per quantity when it is read (D4), never by
combining displacements and deriving the rest.

Above both sits a **variant** choice, for runs of the same structure with
different stiffness (D9).

### D2 — The set gains one file, `<stem>.user.h5`

```text
<stem>.h5           model          neutral + /opensees/ (+ /opensees/combinations, D4)
<stem>.geometry.h5  geometry
<stem>.ladruno      fork recorder  stages, modes; apeGmsh writes the stamp only (ADR 0122 D3)
<stem>.user.h5      apeGmsh + app  combinations, spectra, modal table, user results, entities
```

`.user.h5` is owned by apeGmsh. Python writes it; the app writes it only in
the combination zone (D4). Every write is atomic: write a temporary file in
the same folder, then replace. The file carries `/meta` with `model_hash` and
`fem_hash` (byte-equal to `<stem>.h5`), and `results_hash` of the
`.ladruno` its derived zones were computed from. A file whose `results_hash`
no longer matches the `.ladruno` is **stale**: the app lists its derived
content as stale and does not draw it as current. Authored combinations do not
go stale; they reference states by name.

This amends ADR 0122 D1 (a set reads one recorded results file **and** one
user file) and ADR 0112 D3 (the app writes one file, and one zone in it).
Each zone has its own version key (ADR 0112 D2, ADR 0023).

### D3 — `.user.h5` zones

| Zone | Content |
|---|---|
| `/meta` | pairing hashes, writer, units |
| `/combinations` | combination descriptors authored in Python or the app (D4) |
| `/modes` | the modal table for the `.ladruno` modes (D6) |
| `/spectra/<case>` | spectrum definition and per-mode increments (D5) |
| `/results/<name>` | user results on nodes, elements or entities (D7) |
| `/entities` | section cuts, storeys, point sets (D8) |

The layout inside each zone is slice 0's architect-pair output. Element
fields use the `.ladruno` bucket conventions (`COLUMN_MAP`, `COMP_NAMES`,
`QUADRATURE/GP_PARAM`), so one reader path serves both files. NaN means
"not defined here" (the YOO slab uses it to grey out non-slab nodes; that
convention becomes a rule).

### D4 — A combination is a descriptor, evaluated on read, declared from either side

A combination is a named descriptor:

- **terms**: a list of `(source, factor)`, where a source is a stage (and
  step), a mode, a spectrum case, or another combination (no cycles);
- **rule**: `linear` (the factored sum), `srss`, `cqc` (with the damping
  ratio), `abs_sum`, `directional` (100/30 or another fraction, as the
  envelope over its permutations), `envelope` (max, min or absmax over the
  terms);
- **sign**: signed for `linear`; unsigned for the modal rules and envelopes.
  An unsigned result is shown as a contour only. The app does not draw it as
  a deformed shape.

Combinations are declared from both sides, as the owner asked:

- **Python**: `apeSees` declares combinations next to the patterns they
  combine. They are archived in `model.h5` under `/opensees/combinations`,
  derived at write time like the other `/opensees` records.
- **Python after the run, or the app**: written to `.user.h5`
  `/combinations`. The app is the only writer that may change this zone at
  run time; it keeps a backup of the previous version.

Names are unique across both. A `.user.h5` combination may reference a
`model.h5` one, never the reverse.

Evaluation happens on read, per quantity (ADR 0124 D6), in two evaluators:
the Python `Results` and the app's TypeScript. They share one conformance
fixture (a model, its files and golden combined values per rule), run in
both test lanes.

A combination is valid within one variant (D9). Combining stages from runs
with different stiffness is refused. Linear superposition of a nonlinear
stage is allowed and badged ("superposition assumes linearity").

### D5 — Spectra are their own family

A spectrum case is: a direction (or directions), the spectrum table
(`T`, `Sa`), the damping ratio, the modal set it uses, and, per mode, the
increment of every recorded quantity about the reference state (ADR 0124 D4:
the static modal-force solve on the preloaded domain).

The per-mode increments come from the ADR 0124 driver's extractors, in
Python, because that driver refuses a model with a declared recorder (ADR
0124 D4). apeGmsh writes them to `.user.h5` `/spectra/<case>`. CQC, SRSS and
100/30 are D4 rules over a case's modes, so the combined demand is a User
Combination. A fork option that imposes the modal state without committing
(the request in ADR 0124 D4) would let the recorder write the increments to
the `.ladruno`; this family's location then moves, and its descriptor does
not change.

Gravity is combined separately (ADR 0124 D7): `1.0 D + L_exp ± E` is a
`linear` combination of a stage and a spectrum combination, which is why
spectra and stages share one combination language.

### D6 — Modes carry their table

The `.ladruno` holds the shapes and eigenvalues (`modesOfVibration`). The
table that the engineer reads first (period, participation factor, effective
mass ratio and cumulative per direction, generalised mass) is written to
`.user.h5` `/modes` from `modalProperties` or from the driver that ran the
eigen solve. The app shows it beside the mode list. A mode beyond the table's
length is shown without its row, not refused.

### D7 — User Results are named fields with a declared target

A user result is a named group of components with units, on one target:

- **nodes** (a nodal field);
- **elements**, at Gauss points or line stations, with the `.ladruno`
  bucket conventions;
- **entities** (D8).

Each user result is either **per state** (it has a value for named stages,
modes, spectrum cases or combinations) or **state-free** (a design check
such as D/C or footfall `R`, shown regardless of the phase bar). The Python
API writes them through one call into `.user.h5`; the slab's
`yoo.slab.outputs` custom stages migrate to it before ADR 0122 slice 10.

### D8 — Entities that are not mesh items

`/entities` holds three kinds, each named:

- **section cuts**: a plane and the element set it cuts, from
  `src/apeGmsh/cuts`; quantities are resultants (forces and moments about a
  reference point);
- **storeys**: a level, a node set and a reference point (the centre of mass);
- **point sets**: coordinates, each with an optional host element.

The app draws an entity where it is (a cut plane, a storey outline, points)
and colours or labels it with its user results. SLE's pier cuts,
coupling-beam ends, storey forces and plan-point drifts fit here.

### D9 — Stiffness states are variants: separate sets, one manifest

A stiffness state is a different model, so it is a separate set with its own
hashes. A **variant manifest** in the folder lists the sets of one study,
each with a label and the parameters that differ (for example
`state: cracked`, `bischoff: D+SD+0.25L`). The app switches between variants
and compares two (side by side, or the difference of a quantity defined on
both). A multi-pass procedure such as Bischoff writes each pass as its own
stem.

### D10 — The model file must describe the analysed model

A combination, a spectrum or a user result on a model the `model.h5` does not
describe pairs by `fem_hash` alone, and its element results can land on the
wrong elements. SLE adds raw commands after `apeSees` (coupling-beam shells
replaced by frames and RBE2, diaphragms, storey masses, a P-Δ preload), so its
`model.h5` is the mesher's, not the analysed model. This ADR does not solve
that. It records it as a prerequisite for SLE adoption, for a follow-up.

## Consequences

- One combination language covers load combinations, response-spectrum
  rules and the gravity-plus-seismic sum, and no combination is stored as a
  copied stage.
- The recorder's `.ladruno` stays as the fork wrote it; a crash in user-data
  writing cannot damage it.
- The app writes a file for the first time. The write is confined to one
  zone, it is atomic, and it keeps a backup.
- Two evaluators (Python and TypeScript) must agree. The conformance fixture
  is the price of keeping the app free of Python (ADR 0112 D2).
- A set may now be three to four files. Pairing rules stay hash-based.
- `.user.h5` adds to the program's schema surface: `h5-schema.md` gains a
  section, and its zone versions are checked like the others.

## Relation to other ADRs

- **ADR 0112**: amends D3 (the app writes one zone of one file). D2's
  "own optional zone with its own version key" governs every zone here.
- **ADR 0122**: amends D1 (a recorded file plus a user file per set). D3/D4
  are unchanged: nothing but the stamp is written into the `.ladruno`. The
  native writers YOO uses retire on 0122's timeline; their custom stages
  move to D7.
- **ADR 0124**: D5 and D6 here are its D4 and D6 made storable; its D7 is a
  `linear` combination here.
- **ADR 0075 / 0127**: the modal drivers and mode-set primitives produce the
  modes and the table of D6.
- **ADR 0128**: a view spec may name a state family member and a quantity
  group; its spec version bumps when it does.

## Alternatives rejected

- **A consumer group inside the `.ladruno`** (`/apegmsh/...`, lowercase like
  the stamp). One file, no pairing to add. Rejected: a crash mid-write can
  make the whole recorder file unreadable; the recorder rewrites the file on
  each run, deleting authored combinations; freed space is not reclaimed
  without `h5repack`; a partitioned run has no single file to hold it; and it
  needs a fork-owned layout change (ADR 0122 D4).
- **Materialised combinations** (each combination written as a stage, as the
  slab does). Duplicates data per combination, cannot represent unsigned
  modal rules honestly, and goes stale silently when a source changes.
- **Combinations in `model.h5` only.** Simple, but every new combination
  needs a re-run of the generator, and the app could not author one.
- **Spectra under Modes.** Mixes eigen shapes with response increments that
  depend on a spectrum, a direction and a reference state; the owner ruled
  them a family of their own.

## Slice plan

| Slice | Chain | Work | Gate |
|---|---|---|---|
| 0 | V | Architect pair: `.user.h5` zone layouts and the combination descriptor schema; `h5-schema.md` section | maintainer accepts |
| 1 | X | Python writer and reader for `.user.h5` (`/meta`, `/combinations`, `/results`), atomic write, staleness | slice 0 |
| 2 | X | Combination evaluator in `Results`, per quantity, all D4 rules; the conformance fixture | slice 1 |
| 3 | V | App: phase-bar families (Stages, Modes, User Combinations) and the Results / User Results picker groups | slice 1 |
| 4 | V | App: TypeScript evaluator against the conformance fixture; unsigned results as contour only | slice 2, 3 |
| 5 | B | `apeSees` combination declarations archived in `model.h5` `/opensees/combinations` | slice 0 |
| 6 | X | `/modes` table from `modalProperties`; `/spectra` from the ADR 0124 driver | slice 1; ADR 0124 PR 2 |
| 7 | V | App: Spectra family, modal table panel | slice 6 |
| 8 | X+V | `/entities`: cuts, storeys, point sets, with their user results | slice 1 |
| 9 | V | App: authoring combinations into `.user.h5` (atomic write, backup) | slice 4; human gate |
| 10 | V | Variant manifest and the app's switch and compare | slice 3 |
| 11 | — | YOO migration: slab custom stages and combinations to `.user.h5`, before ADR 0122 slice 10 | slice 2, 8 |

## Open questions

1. The variant manifest's name and home (a JSON file beside the sets, or a
   zone in each `.user.h5` pointing at its siblings).
2. Whether `apeSees` combinations may reference spectrum cases (they exist
   only after the driver runs), or only stages and modes.
3. The undo depth of app-authored combinations beyond one backup.
4. Whether a user result may be both per state and an envelope over states,
   or whether that is always a combination.

## References

- YOO session inventory, 2026-11-01: `yoo/yoo/sle/{builder,modal_store,checks}.py`,
  `sle/visor/export_results.py`, `yoo/yoo/slab/{states,outputs,pipeline,footfall,model}.py`.
- apeGmsh issues from the reference-model sweep: #2667–#2673.
- Reference models: `C:\nmb\My Libraries\nmb\gitAPE Companion\apeGmsh Reference Models\`.
