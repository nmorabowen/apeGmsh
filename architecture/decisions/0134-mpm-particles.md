# ADR 0134 — MPM in apeGmsh: the particle set beside the deck, particle results in `.ladruno`, a three.js points layer

**Status:** Accepted, with Amendments 1–3 (2026-11-15, 2026-11-19). Accepted 2026-11-13 (drafted by the fork's MPM program (WP-232a), ratified by the
owner on #2976, "Yes, ratify"). Merging the PR that flips this Status (`human-gate`) is the
acceptance. The pointers in ADRs 0112, 0023 and 0113 land with it (ADR 0121 D5).

**Owner:** nmora

**Program:** the fork's WP-232 (program issue nmorabowen/OpenSees#1075; the fork's ADR
`Ladruno_implementation/232_ladruno_mpm_general_driver_adr.md` §7). T cards the apeGmsh slices on
#1203, as chain M. The fork's `Ladruno_implementation/232a_mpm_contracts.md` specifies both file
contracts field by field, and the fork's `tests/data/mpm/` holds their fixtures.

**Amends** [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) D1 (one more artifact,
D1 here). **Departs** from ADR 0112 D2's per-zone version keys and from ADRs 0023 and 0113 for that
artifact's integer version key (D1). **Builds on** ADRs 0114, 0116, 0121, 0122, 0127, 0130, 0131,
0132 and the charter's P3, P4 and P9. **Supersedes** nothing.

## Context

The fork is adding a general Material Point Method (MPM) driver. It is an `mpm …` command family
whose explicit engine runs particles over a background grid that is reset every step. Its ADR splits
the work the same way apeGmsh's charter does:

- apeGmsh owns geometry, seeding, transport, reading results and viewing;
- the fork owns the mechanics and the results format.

That split is also the licence-clean one. It matches the fork's ADR-60a, which says apeGmsh is a
seeder, not a runtime remesher (apeGmsh's own ADR 0060 is unrelated: remote HPC submission).

Two files cross the boundary:

- **Contract 1, the particle set** (apeGmsh → fork). At 10⁵–10⁶ particles, per-particle commands in
  a deck are not an option, so the set travels as HDF5 and the deck reads it with
  `mpm particles -file {path}`. Each particle carries:
  - its position, volume, and mass or density;
  - GIMP half-lengths;
  - initial velocity and stress;
  - a body id and a **material tag**.
- **Contract 2, particle results** (fork → apeGmsh). This is a new group inside `.ladruno`:
  `MODEL_STAGE[*]/MODEL/PARTICLES` and `RESULTS/ON_PARTICLES`.
  - The change is additive: `FORMAT_VERSION` stays 1 and `FORMAT_MINOR_VERSION` becomes 2.
  - Particles are removed or added during a run, so outputs are ragged. They are stored concatenated
    with an offset table, the CSR form `.ladruno` already uses for variable connectivity.

Seven questions need an apeGmsh answer, and each one touches an accepted ADR:

1. Where the particle set lives, and who writes it (ADR 0112 D1 and D2, `_artifact_policy`).
2. Where seeding runs (charter P3 and P9).
3. How results arrive, and what today's readers do with them (ADRs 0122, 0131).
4. What a particle result is to `Results`.
5. How particles are drawn (ADRs 0112 D8, 0116).
6. Who writes material tags (charter P4).
7. Whether MPM has a live path (ADR 0121).

An eighth follows from ADR 0121 D4: the `mpm` primitives and verbs grow two frozen registries.

## Decision

### D1 — The deck emitter writes the particle set beside the deck it emits

There is one rule for the stem, the folder, the path in the deck, overwrites, and the live path.

- **Who writes it.** Every deck emitter of `apeSees` — `tcl(path)`, `py(path)` and the others that
  write a deck — writes the particle set whenever the model declares MPM bodies. It writes it **next
  to the deck it emits**, as `<deck stem>.particles.h5`, **in the same call** that writes the deck.
  Emitting `ops.tcl("runs/case1.tcl")` writes `runs/case1.particles.h5`. The set follows its deck's
  explicit-path semantics.
- **What the deck says.** Each deck emitter generates the `mpm particles -file …` line itself, for
  its own target. The line **anchors the path to the deck's own folder**, so the deck and its
  particle set move together whatever the working directory:
  - Tcl: `mpm particles -file [file join [file dirname [info script]] {case1.particles.h5}]`
    (the pattern `emitter/tcl.py` already uses for per-rank decks);
  - Python: `os.path.join(os.path.dirname(os.path.abspath(__file__)), "case1.particles.h5")`
    (`emitter/py.py` has no `__file__` anchoring today; this adds it).

  The fork opens the path exactly as given; a relative path resolves from the working directory
  (contract §1).
- **Not an archive row.** The `-file` line is not a `command()` row. It depends on the deck's stem,
  which the archive does not know, so it stays out of `/opensees/commands` and out of `model_hash`.
  The archive carries the MPM declarations (D2). A replay re-emits the line for its own target and
  reproduces the particle **content**, under a new `SET_ID`. It does not reproduce a byte-identical
  file.
- **Its stem is the deck's, and it should be the model's** (the owner's ruling on #2976,
  2026-11-13: "Yes, same stem"). The viewer pairs by stem only. A particle set pairs with its
  model only when the deck stem equals the model stem, so `ops.tcl("slab.tcl")` beside `slab.h5`
  writes `slab.particles.h5`. A deck, or a particle set, at another stem opens without a model,
  under the viewer's existing "no model is open" banner. M2 builds no discovery by hash.
- **Overwrites: a rule of its own.** `src/apeGmsh/_artifact_policy.py` cannot govern this file.
  Its `is_apegmsh_artifact` needs `/meta` zone keys, which a Contract-1 file does not have, and its
  P4 never replaces a file. The emitter therefore overwrites an existing `<deck stem>.particles.h5`
  only when that file carries this contract's provenance: `FORMAT = "ladruno-mpm-particles"` and
  `GENERATOR = "apeGmsh"`, written by an apeGmsh emit. Any other file at that path refuses the emit,
  by name.
- **Together or not at all.** When the particle set is refused, **no deck is written**. Both are
  checked before either is written, so a re-emit never pairs a new deck with a stale set.
- **Serial only, one writer.** MPM decks are serial: the fork refuses `mpm` in OpenSeesSP and
  OpenSeesMP. The emitter therefore refuses, by name, a model with MPM bodies on a partitioned or a
  per-rank emit (for example `tcl(per_rank=True)`). The emit has a single writer, and no rank gate is
  needed.
- **No live path. MPM is deck-only by this ADR's decision.** It follows
  [ADR 0121](0121-charter-v2-b-with-amendments.md) D2, which makes emission the official path. The
  bridge refuses a model that declares MPM bodies on the live path, by name.

**The format.** The layout is Contract 1 (`FORMAT = "ladruno-mpm-particles"`,
`FORMAT_VERSION = 1`). The fork owns it, as it owns `.ladruno` (ADR 0122 D4): the fork's reader
admits or refuses the file, and a user may write one without apeGmsh. apeGmsh changes it only through
a fork PR.

**Pairing.** The file's `PROVENANCE` group carries:

- `session_id`;
- `apegmsh_version` and `apegmsh_commit`;
- `fem_hash` and `model_hash`, when the session has them.

The emitter takes these values from the **in-memory lineage** at emit, not from `model.h5`, because
`model.h5` may not be on disk yet, or may hold a previous run's lineage. They
record where the set came from. Pairing is by stem (above), not by these values.

`SET_ID`, a uuid4 at the file's root, is copied by the fork into the `.ladruno` it writes
(`MODEL/PARTICLES@SOURCE_SET_ID`). The results then name the exact particle set they came from.

**ADR 0112.** The particle set adds a row to D1's artifact table: "particle set, at deck emit,
`<deck stem>.particles.h5`, new". It is recorded as an amendment there when this ADR is accepted.
Like `.user.h5` ([ADR 0130](0130-states-quantities-and-the-user-results-file.md)) and
`.workspace.h5` ([ADR 0132](0132-workspace-file-and-variants-manifest.md), the subject of ADR 0112's
pointer of 2026-11-10), it is a file of the set, not a zone of `model.h5`.

**The version key is an exception** to ADR 0112 D2's per-zone version keys, and to the semver keys
with a floor per zone of ADRs 0023 and 0113. The file carries an **integer** `FORMAT_VERSION` that
the fork owns, as `.ladruno` does. apeGmsh's readers gate on the exact integer, which today means
`(1,)`. There is no floor, no shim ledger and no schema corpus for it. This ADR records the
exception by name.

**The viewer's pairing.** `.particles.h5` joins the set as a suffix of its own, modelled on ADR 0132
D1's `.workspace.h5`:

- In `apeGmshViewer/src/main/pairing.ts`, `SUFFIXES` matches `.particles.h5` case-insensitively and
  before `.h5`, so `case1.particles.h5` is never opened as a model whose stem is `case1.particles`.
- The open set (`OpenSet`, `candidates`) gains a `particles` slot.
- `classify`'s refusal message lists the new suffix.
- `<s>.particles.h5` pairs by stem with `<s>.ladruno` and `<s>.h5` when they are present. The fork
  writes `<s>.ladruno` beside the deck when the deck names it so. Pairing is by stem only: a file at
  another stem opens without a model, under the existing "no model is open" banner.

### D2 — Where seeding runs: in the bridge's emitter, from FEMData alone

Seeding runs in the bridge, at emit, from the `FEMData` the bridge was built on, with no session and
no gmsh. Charter P3 keeps `gmsh.*` out of the bridge (no gmsh imports; testable without booting
gmsh), and P9 lets `apeGmsh.opensees` depend only on `apeGmsh.mesh.FEMData`.

- **Lattice seeding** is the v1 method. The bridge places `n` points per grid cell per axis, at the
  cell's sub-cell centres. It keeps a point when the point falls **inside an element of the body's
  physical group**, tested point-in-element on the FEMData mesh, in numpy.
  - Supported families: `tri3`, `quad4`, `tet4` and `hex8`. Simplices use barycentric coordinates;
    `quad4` and `hex8` use the inverse isoparametric map, solved by Newton.
  - A point on a face shared by two elements is kept once.
  - Any other family is refused loudly, by name. This includes the higher-order and pyramid or wedge
    elements.
- **Each kept particle** gets volume `∏(h/n)` (times the plane-strain thickness in 2D), GIMP
  half-lengths `h/(2n)`, and the body's density.
- **A uniformity report** gives the particles per cell against the declared grid, so partial cells
  on a body's boundary are visible.
- **Gauss-point seeding is not in v1.** `fem/_quadrature.py` states its own scope ("minimal Gauss
  quadrature rules", sufficient where "the absolute scale of the quadrature weights cancels"). Its
  tetrahedron rule is Keast's 11-point rule, whose negative centroid weight would make a
  negative-volume particle in every tet. When Gauss-point seeding comes, it uses positive-weight rules
  only, a per-family "supported for seeding" table with loud refusal, and a signed-Jacobian check
  (`compute_jacobian_dets` returns `abs(det)`).
- **Reproducible from the archive.** The archive carries the FEMData mesh (the neutral zone) and the
  bridge's MPM declarations as `command()` rows stored in `model.h5`
  ([ADR 0114](0114-the-archive-is-the-program.md): `command()` is the archive's escape for a verb
  with no typed method, and its rows are archived under `/opensees/commands`). The declarations are
  the grid, the bodies, the particles per cell and the basis. A replay that emits from `model.h5`
  therefore reproduces the particle content, under a new `SET_ID` (D1). Contract 1 refuses zero or negative volumes, and apeGmsh
  never writes one.

### D3 — Particle results are the `.ladruno` particle group, read at `FORMAT_MINOR_VERSION ≥ 2`

ADRs 0122 and 0131 stand: apeGmsh reads one results format, with no new results file and no native
writer. Particle results are read from the fork's `.ladruno` group, defined by Contract 2.

- `LADRUNO_SUPPORTED_FORMAT_VERSIONS` stays `(1,)` in Python and TypeScript.
- **The fork pins a particle stage's `KIND`.** It is `"transient"` for explicit (mode F) and
  implicit-dynamic runs, and `"static"` for quasi-static runs. It is never empty and never any other
  value (contract §2.3, rules C2-KIND and C2-MODEKIND). Today's TypeScript reader (`readHost` in
  `apeGmshViewer/src/reader/ladruno.ts`, through its `KIND_MAP`) throws a whole-file layout error on
  any other `KIND`, so the pin keeps the file opening there.
- **A particle-only stage** is written as the recorder's empty stage: `EMPTY_PARTITION = 1`,
  zero-length `MODEL/NODES`, and empty FE result groups.
- **What today's readers do**, stated by scope:
  - **A file with at least one FE stage** opens in today's Python reader, and in Python whenever
    `model_h5=` is given. The reader ignores the particle groups.
  - **An all-particle file** (every stage particle-only) is **refused** by today's Python reader.
    `LadrunoReader.fem()` skips every `EMPTY_PARTITION` stage, and `Results.from_ladruno` then raises
    "carries no MODEL group … Pass model_h5=".
  - Today's TypeScript reader does not read `EMPTY_PARTITION`. It opens both kinds of file without a
    layout error, provided `KIND` is pinned as above, and shows no particles.

  That is acceptable until the reader slice lands.
- **The reader slice (chain M, link M2)** teaches `Results.from_ladruno` (`results/readers/_ladruno.py`)
  and the viewer (`apeGmshViewer/src/reader/ladruno*.ts`) to read the particle group when
  `INFO/FORMAT_MINOR_VERSION ≥ 2` and the stage has `MODEL/PARTICLES`. The readers:
  - accept the fork's attribute encodings: 1-element arrays, fixed-length `NULLTERM` strings, and an
    empty string written as an absent attribute;
  - accept int64 ids, converted to `Number` with the existing safe-integer check;
  - open an all-particle file without a model archive.

  **Its oracles** run on the fork's two fixtures, `tests/data/mpm/particles_mixed_v1m2.ladruno` (an
  FE stage, then a particle stage) and `tests/data/mpm/particles_v1m2.ladruno` (all particles). Each
  is copied into apeGmsh's fixtures with its source sha.
  - **Before M2** (pinning today's behaviour):
    - the Python reader opens the mixed fixture and ignores the particles;
    - the Python reader refuses the all-particle fixture with its no-MODEL error;
    - the TypeScript reader opens both without a layout error.
  - **After M2:**
    - both readers read the particles of both fixtures;
    - the fork's `ladruno_format.validate()` passes on both, and on a real run in the `live-fork`
      lane.
- **Part files.** A part file (`<stem>.part-<N>.ladruno`) with particles is refused by name. The
  fork's v1 driver is serial, and Contract 2 forbids it.
- **apeGmsh's own writers.** apeGmsh's `.ladruno` writers (`_ladruno_writer.py` and the MPCO
  converter) write no particles and keep minor 1.

### D4 — The particle-cloud result ontology

A particle result is not a node, element or Gauss result: its carriers move, appear and disappear.
It gets its own level and is never keyed by node or element ids.

- **Level.** `ResultLevel.PARTICLES = "particles"` in `results/readers/_protocol.py`.
- **Slab.** `ParticleSlab(component, values, particle_ids, offsets, time)` keeps the file's own
  ragged layout, with no copy:
  - `values` is `(S, nComp)`, concatenated over outputs;
  - `offsets` is `(T+1,)`, `particle_ids` is `(S,)` and `time` is `(T,)`.

  `ParticleSlab.output(k)` returns one output's `(ids, values)`. `ParticleSlab.dense()` returns
  `(T, M, nComp)`, aligned to the stage's registry ids, with `NaN` where a particle is absent.
  `.item()` follows ADR 0115 D3.
- **Composite.** `results.particles.get(component=…, body=…, time=…)`.
  - Selection is by **body name** (Contract 2 carries the names) or by explicit particle ids.
  - It does not mix in `_SelectionMixin`, whose node, element and label selectors mean nothing for
    particles.
  - `results.particles.bodies()`, `.registry()` (id, body, material tag, mass) and `.grid()` read the
    per-stage metadata.
- **Components** are spelled as the file spells them (ADR 0122 D7): `POSITION/x`, `STRESS/sigma12`,
  `STATE_<token>/<label>`. No token table is added. In plane strain, `STRESS` is `[11, 22, 12]` and
  σ₃₃ is the separate `STRESS33` field (contract §0.1).
- **Outputs** are keyed by index and `STEP`, never by `TIME`, which may restart inside a stage
  (contract §2.5). Stage-name binding (`_bind_stage_names`) works unchanged.

### D5 — Particles are drawn by a three.js points layer in apeGmshViewer

ADRs 0112 D8 and 0116 decide the technology: a new view is three.js in apeGmshViewer, and the Qt
viewers get nothing. ADR 0087's "dots" are Qt chrome glyphs, and their own row says they are never
used for another point cloud, so they are not reused.

- **One `THREE.Points` per stage**, built after the node layer (`renderer/nodes.ts`,
  `makeNodePoints`). Its position buffer is filled per output from the file by hyperslab
  (`OFFSETS[k]..OFFSETS[k+1]`), never with the whole history in memory.
- **Colour** comes from any particle field component, through the existing contour ramp and unit
  display. `NaN` (`STATE_*` on particles whose material does not answer) draws in the "no value"
  colour.
- **The grid** (from `MODEL/PARTICLES/GRID`) is drawn as an optional wire box.
- **The seeded set** (`<s>.particles.h5`) may be shown before a run, from the same layer.
- **Measurements.** The work gets a row in `apeGmshViewer/MEASUREMENTS.md` (ADR 0112 D5), at 10⁵ and
  10⁶ particles: time to first frame, frame rate while scrubbing outputs, and memory. ADR 0112's data
  server is considered only if those numbers demand it.
- **Visual check.** The viewer's visual check (`apegmsh-viewers-visual-check`) runs with evidence,
  and the owner approves the screen (ADR 0112 D7).

### D6 — Material tags are resolved at emit (charter P4)

Users never type tags. A body declares its material by label, and the bridge allocates the
`nDMaterial` tag at build (`TagAllocator`). The emitter then writes Contract 1's `MATERIAL_TAG`
column from that allocation, in the same pass that emits the material and the deck (D1, D2). No
public API asks for a tag.

### D7 — Bridge primitives only after a pinned fork release; MPM is deck-only

- apeGmsh consumes the `mpm` primitives only when both of these hold:
  - the fork PR that adds `mpm` has merged into `ladruno`;
  - `FORK_PIN` names a fork release that contains it, with its four keys bumped together.
- Each member carries:
  - the `ladruno_fork` marker;
  - a stock refusal that names the fork, raised at build time or listed as a `_FORK_ONLY_*` entry
    (ADR 0121 D4, ruling 1);
  - a warrant-row test;
  - a row in `docs/concepts/backend-capabilities.md`;
  - capability probes that return `None` when they are inconclusive.
- **The member stays Inventory** until its warrant-row test *passes*, not skips, in the nightly
  `live-fork` lane (ADR 0121 ruling 1; ruling 3 for skips).
- **MPM is deck-only (D1; ADR 0121 D2).** A staged MPM run, for example a K0 start followed by a plastic stage,
  uses `parameter … pattern <mpmTag>` and `updateParameter` in the emitted deck. The live route
  refuses any model with MPM bodies, by name.
- Paths in emitted Tcl are brace-quoted.

### D8 — Exactly two registry entries (ADR 0121 D4)

When accepted, this ADR admits **exactly** these two census entries, and no others:

```python
ADMITTED["primitive families"]["mpm"] = "0134"
ADMITTED["emitter verb families"]["mpm"] = "0134"
```

They follow the shape of [ADR 0127](0127-modal-basis-primitive-family.md):

- **Base class.** `apeGmsh.opensees._internal.types.MPM(Primitive)`.
- **Allocator kind.** `mpm`, as one row in `apesees._KIND_BY_FAMILY`, read by the registry lock
  `tests/opensees/contract/test_verbs_lock.py` (check (h)).
- **Family line.** `Family("mpm", MPM, …:ALL_MPM)` in `tests/families.py`.
- **Verb family.** The verb family `"mpm"` in `emitter/verbs.py::FAMILIES`.
- **Contract test.** A parity test checks every member's fork command line on all five emitters, as
  `tests/opensees/parity/test_chain_m_command_parity.py` does for `ModalBasis`.

The members are bridge declarations that emit `command()` rows ([ADR 0114](0114-the-archive-is-the-program.md)
D1 and D3), and those rows are stored in `model.h5` (D2). This differs from ADR 0127 D2, whose
members never reach the archive.

**The bridge slice (chain M, link M3) may add members only**, named after the fork guide's `mpm`
sub-commands. Any other registry entry needs a new ADR: a constraint kind, an MCP tool, a habitat
verb, or another family.

## Consequences

- A deck with MPM bodies leaves one more file, `<deck stem>.particles.h5`, which can be large (about
  100 bytes per particle before compression). Re-emitting from `model.h5` regenerates its content,
  under a new `SET_ID`.
- **Reading particle results with today's apeGmsh:**
  - a fork file with at least one FE stage opens with its particles ignored;
  - an all-particle file needs `model_h5=` in Python;
  - nothing breaks on `FORMAT_VERSION`, which stays 1.

  Reading the particles themselves waits for M2.
- `Results` gains a level whose slabs are ragged. Code that assumes `(T, N)` slabs does not apply to
  it, and the composite keeps it apart from node and element selection.
- Seeding supports four element families in v1 and refuses the rest.
- The fork owns both contracts. Restructuring the particle set is a fork PR away for apeGmsh, the same
  price ADR 0122 D4 pays for `.ladruno`.

## Alternatives rejected

- **A FEMData stream in `model.h5`**, with the `compose-streams` obligations and the neutral-zone
  bump:
  - the set holds bridge-allocated material tags, which a solver-neutral, pre-bridge FEMData cannot
    hold (P4);
  - it would put up to hundreds of megabytes into the archive the viewer opens for every model;
  - the fork's reader would then parse apeGmsh's archive and its zone versioning.
- **A new `/particles` zone in `model.h5`.** It has the same size and coupling problems. ADR 0112's
  integer policy (no int64) would also conflict with the int64 particle ids the fork's ADR fixes.
- **The particle set next to the script, sharing `<stem>.h5`'s stem.** A deck emitted to another
  folder could not name it by a path that survives a move. One rule (D1), with the file beside the
  deck, avoids that.
- **Seeding with Gmsh's inside test.** It needs a live gmsh/OCC session in `apeGmsh.opensees`
  (P3, P9), and it cannot be replayed from `model.h5`.
- **A live MPM path.** This ADR keeps MPM deck-only (D1; ADR 0121 D2 makes emission the official
  path).
- **Governing the particle file by `_artifact_policy`.** That policy needs `/meta` zone keys the file
  does not have, and its P4 never replaces a file. D1 gives the file a rule of its own.
- **A new results file for particles**, or particles in native `results.h5`. ADRs 0122 and 0131 allow
  one results format and no new native writer.
- **A `.ladruno` `FORMAT_VERSION` bump to 2.** Every new fork file would become unreadable by today's
  apeGmsh, which opens `(1,)` only. The group is additive, so the minor moves instead.
- **Per-particle Tcl commands.** They are unusable at 10⁵–10⁶ particles.
- **The Qt viewers or ADR 0087's dots.** Both are frozen (ADR 0112 D8).
- **Gauss-point seeding in v1** with today's quadrature rules, because of the negative weights (D2).

## References

- The fork:
  - `232_ladruno_mpm_general_driver_adr.md` §7, and its ADR-60a;
  - `232a_mpm_contracts.md` (both contracts, field by field);
  - `ladruno_schema_v1.md`;
  - `Ladruno_scripts/ladruno_recorder_tests/ladruno_format.py` (the executable spec);
  - the fixtures `tests/data/mpm/particle_set_v1.h5`, `tests/data/mpm/particles_v1m2.ladruno` and
    `tests/data/mpm/particles_mixed_v1m2.ladruno`.
- apeGmsh ADRs:
  - [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md);
  - [ADR 0114](0114-the-archive-is-the-program.md);
  - [ADR 0116](0116-render-technologies.md);
  - [ADR 0121](0121-charter-v2-b-with-amendments.md);
  - [ADR 0122](0122-ladruno-canonical-results-format.md);
  - [ADR 0127](0127-modal-basis-primitive-family.md);
  - [ADR 0130](0130-states-quantities-and-the-user-results-file.md);
  - [ADR 0131](0131-all-in-ladruno.md);
  - [ADR 0132](0132-workspace-file-and-variants-manifest.md);
  - ADRs 0023 and 0113 (the semver keys D1 excepts).
- apeGmsh code and docs:
  - `architecture/charter.md` P3, P4 and P9;
  - `src/apeGmsh/_artifact_policy.py`;
  - `apeGmshViewer/src/main/pairing.ts` (`SUFFIXES`, `candidates`);
  - `apeGmshViewer/src/reader/ladruno.ts` (`readHost`, `KIND_MAP`);
  - `src/apeGmsh/results/Results.py` (`from_ladruno`);
  - `architecture/h5-schema.md` ("Integer policy for the ADR 0112 zones");
  - `results/schema/_versions.py`;
  - `FORK_PIN`;
  - `docs/concepts/backend-capabilities.md`.

## Amendment 1 (2026-11-15): D7's staged-run parameter target is `loadPattern`

D7 says a staged MPM run uses `parameter … pattern <mpmTag>`. The fork spells this target
`loadPattern`: in classic Tcl, `parameter … pattern` matches only `pattern <tag> lambda`
(`TclParameterCommands.cpp:134-135`), and the generic branch that reaches the MPM host's
`setParameter` matches `loadPattern` (`:235`, `:248`). The fork's ADR
(`232_ladruno_mpm_general_driver_adr.md`, finding 3) and its guide use `loadPattern` throughout.

This amendment changes one word. Where D7 reads `parameter … pattern <mpmTag>`, read
`parameter … loadPattern <mpmTag>`. The rest of D7 stands: MPM is deck-only, and the live route
refuses any model with MPM bodies by name.

The bridge does not emit this route yet. Since #3001, a staged emit of a model with MPM
declarations is refused by name; emitting the `loadPattern` stage path is a follow-up.

## Amendment 2 (2026-11-15): a lattice point on the interface of two bodies goes to the body declared first

D2 seeds each body on its own. As M2 shipped it, a lattice point that two bodies both claim was
refused, whatever the reason. The owner ruled on this for #3005 ("Yes, first body"). This amendment
changes D2 only; Amendment 1 changes D7.

**Why.** Bodies that share a face are the ordinary case: a block resting on soil, a fill against a
wall. When that face lies on a lattice coordinate, every lattice point on it is inside both bodies'
closed regions. The refusal then failed every such two-body model, and the only way out was to move
the face or the grid by a fraction of a sub-cell.

**The rule.**

- A lattice point two bodies claim is an **interface point** when it lies on a boundary facet of
  each body, and two of those facets, one from each body, face each other: their outward normals have
  a negative dot product. A boundary facet is an element facet that no other element of the same body
  shares, matched by node ids. "Lies on" uses the seeder's existing tolerance in reference
  coordinates. There is no new parameter.
- An interface point is kept once, by the body declared first. Declaration order is already the
  particle set's order (Contract 1), so the result is deterministic, and swapping two declarations
  swaps the owner.
- Any other shared point is an overlap and is still refused. The message names both bodies, the
  number of shared points, how many of them are not on an interface, and one of those points. This
  covers a point on a facet inside one body's mesh, and a point where both bodies lie on the same
  side of a boundary they share.
- A body that the rule leaves with no lattice point of its own is refused by name, as an empty body
  already is.
- The uniformity report gains `interface_points`: per `(first, later)` pair of body names, the number
  of points the later body gave up to the earlier one. Its printed form lists them when there are any.

**What stays.** The overlap check is still made on the lattice, as before. An overlap that holds no
lattice point is not seen, and neither is one whose only lattice points lie on facing boundaries.
An overlap thinner than one lattice spacing in which one body's boundary crosses another's at a
lattice point, with the two boundary facets there facing each other (outward normals with a negative
dot product), now seeds instead of being refused, because the crossing point is read as an interface
point (for example, a body B that dips 0.075 into A = [0, 0.625]² and crosses A's top edge at
(0.375, 0.625)); a crossing whose facets do not face each other is still refused. A same-side sliver
thinner than one lattice spacing that lies inside a body, against a face where that body is meshed
non-conformingly, also seeds instead of being refused, because that face counts as a boundary on both
sides (for example, A made of two hex blocks that share no node at x = 0.375, with B a sliver inside
A on that face; a conforming A refuses it). A model in which no two bodies claim a point seeds
exactly as before.

## Amendment 3 (2026-11-19): `Gridvel` and `GridvelAffine` members; `track` and `Stage` deferred

**Status:** Accepted (ratified by the maintainer on 2026-11-19, on #3064: "as recommended"). This
amendment adds two members under D8 ("may add members only, named after the fork guide's `mpm`
sub-commands"). It changes no other decision.

**Why now.** Before `ladruno-v20261118b` (fork #1109, interface D18), the explicit engine (mode F)
accepted `mpm gridvel` and ignored it. From that release on, mode F applies every gridvel form in
both projections, so a member can be proven against the fork. `FORK_PIN` names that release
(cfe478254).

**The members.** Both emit the fork's `gridvel` sub-command (`OPS_LadrunoMPM.cpp::cmdGridVel` at the
pin), and both are deck-only like every other member (D1, D7):

| Member | Fork line |
|---|---|
| `mpm.Gridvel(name, box, v, *, start=None, translate=None, profile=None)`, declared with `ops.mpm.gridvel(...)` | `mpm gridvel <name> box <lo..> <hi..> v <num\|free ×ndm> [start <t0>] [translate <axis>] [profile <t0> <Tr> <t1>]` |
| `mpm.GridvelAffine(name, box, L, *, ref=None)`, declared with `ops.mpm.gridvelAffine(...)` | `mpm gridvel <name> box <lo..> <hi..> affine <L ndm×ndm, row-major> [ref <x..>]` |

`box` is `(lo, hi)`. Values are in the model's units, and times are domain time; there is no
timeSeries binding. `L` is given by rows and written row-major.

**Refused by apeGmsh, by name, before the fork sees the deck.** At construction:

- a box whose corners differ in length, or are not 2 or 3 long, or have `lo > hi` on an axis;
- a `v` of the wrong length, a token other than a number or `"free"`, or every component `"free"`;
- `profile` together with `start`; a profile with `Tr < 0` or `t1 < t0 + 2 Tr`;
- a `translate` axis outside the box's dimension, or on a `"free"` component (the box would never
  move: the fork moves it by that component, which a free component leaves at 0);
- `translate` without `start` or `profile`. The fork moves the box by `v (t − start)`, and its
  default start is `-1e300`, so the box leaves the grid at the first step and the gridvel silently
  prescribes nothing (`OPS_LadrunoMPM.cpp` default, `GridVel::displacementFactor`). A gridvel
  without `translate` needs neither;
- an `L` that is not ndm × ndm, or a `ref` of the wrong length;
- a string where a number or a sequence of numbers belongs (`"12"` is not the row `(1, 2)`);
- a name that is not one bare Tcl word: whitespace, or one of `$ [ ] ; " \ { }`. The name is the
  first user string a Tcl deck writes as a live word, so `g$x` would reach the fork as another name
  than the py deck passes.

At the call, and again when the block is emitted (`mpm_block`): a second gridvel of one name.
`Gridvel` and `GridvelAffine` share the fork's names. At emit, a box of another dimension than the
grid's is also refused.

**Left to the fork, documented.** On a B-spline grid, the fork refuses a box that reaches the ghost
layer behind a `fix` face (WP-232i B10). No apeGmsh deck reaches that case, because `Grid` admits
only the `gimp` and `linear` bases. A test fails when a B-spline basis is admitted, so the static
check is added together with the basis. At `mpm analyze`, the fork also refuses a gridvel and a
`fix` face, or two gridvels, that prescribe different values on one node component.

**Mode F only.** apeGmsh emits the explicit engine and has no `mode` member. Mode A's gridvel rules
do not apply to an apeGmsh deck: the `constraints` handler that a non-zero value needs under `Plain`,
and `condForce`.

**Deferred.**

- **`track`.** The fork's fix #1112, queued after WP-232l, may change it, so no member offers it.
- **`setAffine`.** It changes `L` between `mpm analyze` calls. An apeGmsh deck declares one `L`.
- **`Stage` (`mpm stage`).** WP-232l (fork #1116) may give it pore-pressure semantics. It is added
  once that interface freezes. Until then it stays a `NOT_OFFERED` row of the fork-surface gate
  (`tests/opensees/parity/mpm_fork_surface.py`), with the reason "after WP-232l".

**Oracles.** The parity test (D8) gains the members' fork lines on all five emitters. The fork-surface
gate moves `gridvel` from a `NOT_OFFERED` row to a member. Each member has a warrant row (D7). The
fork tests (`tests/opensees/integration_ladruno/test_mpm_gridvel_live.py`) run a mode-F deck on the
pinned fork and check closed forms. Under `Gridvel`, every particle inside the box moves at `v` and
is displaced by `v T`, and a body outside the box stays at rest, as both bodies do in a control deck
without the gridvel. With `translate` and `profile`, the displacement is `v Φ(T)`. Under
`GridvelAffine`, the particle velocity is `L (x − ref)`.

## Amendment 4 (2026-11-23): the `Stage` member; Amendment 3's deferral is lifted

**Status:** Proposed (#3120). It is accepted when the owner OKs the PR that adds it (`human-gate`).
It adds one member under D8 and changes no other decision.

**Why now.** Amendment 3 held `mpm stage` back in case WP-232l (fork #1116) gave it pore-pressure
semantics. It did not. The pin (`ladruno-v20261123b`, 3eb195bbc) contains #1116 (`ae3e57ad6`), and
there the `stage` branch of `OPS_LadrunoMPM.cpp::OPS_LadrunoMPM` and
`LadrunoMPMSoil.cpp::LadrunoMPMPattern::stageWalk` are still WP-232k's walk. WP-232l adds one line to
the walk, `pencilDt = -1`, which resets route S's cached pencil. The fork's
`tests/mpm_twophase_interface.md` has no stage row.

**The member.** `mpm.Stage(material, stage)`, declared with `ops.mpm.stage(material, stage)`,
emits `mpm stage <matTag> <s>` on all five emitters; the fork answers the number of particles it
reached. The walk reaches every particle whose material has the tag, through that material's own
`materialState` parameter, and passes `s` through unchanged. `material` is a handle or a registered
name, and its tag is resolved at emit, as `Body` resolves it (D6). The row is written in declaration
order among the `Analyze` rows, and that order is the staging. `stage` leaves the fork-surface
gate's `NOT_OFFERED` rows.

**Admitted (owner ruling 1 on #3120).** `DruckerPrager` stages 0, 1 and 2 (0 elastic, 1 elastic
with updated moduli, 2 elastoplastic, its default), and `LadrunoSANISAND` stages 0 and 1. The table
is `mpm.STAGES`, keyed by the class itself. The fork also stages `LadrunoHS`, which has no apeGmsh
class.

**Refused by apeGmsh, by name.** At construction:

- a material that is not an `NDMaterial`, or whose class (a subclass included) is not in `STAGES`;
- a `stage` that is a bool, a float or a string, or outside its class's values.

At emit (`mpm_block`):

- a `Stage` whose material no `Body` uses (the fork refuses it: "no particle material has tag");
- a `Stage` with no `Analyze` after it (owner ruling 2 on #3120: refused, not warned);
- two `Stage`s of one material with no `Analyze` between them, since the first stages no step;
- two `LadrunoSANISAND` materials staged to different values with no `Analyze` between them. The
  fork's SANISAND flag is class-wide: staging one SANISAND tag stages every SANISAND particle.
- a `LadrunoSANISAND` body. At `mpm analyze`, the fork refuses SANISAND particles in explicit mode F
  on a gimp or linear grid ("... at particles in explicit mode F needs a B-spline basis", rule K0).
  apeGmsh emits mode F only, and `Grid` admits gimp and linear only, so no apeGmsh deck can run one.
  A test fails when a B-spline basis is admitted (Amendment 3's B10 pattern); the refusal is then
  narrowed to gimp and linear in the same PR.

A model with `ops.stage` blocks is still refused, unchanged (Amendment 1's `loadPattern` route
stays a follow-up).

**Oracles.** The parity test (D8) gains `Stage`'s fork line on all five emitters. The fork-surface
gate moves `stage` to the members. A warrant row pairs the stock refusal with
`tests/opensees/integration_ladruno/test_mpm_stage_live.py`, which runs a 2-D mode-F column on gimp
on the pinned fork: a weak `DruckerPrager` body, its sides fixed, under gravity and local damping,
staged 0, analyzed, staged 2 and analyzed. Each `mpm stage` reaches all 160 particles. At the end
of stage 0 the deep particles hold `σxx/σyy = ν/(1−ν)` to 0.02 and lie outside the yield surface.
After stage 2 every particle is on or inside it. A control deck with no `Stage` is elastoplastic
from its first step, because `DruckerPrager`'s default stage is 2: at the end of each analyze it
is inside the surface, with a lateral ratio above `ν/(1−ν) + 0.05` at depth.

So the control ends in another state than the stage-0 run. It also ends in another state than
the staged run, by its path rather than in closed form: a level, laterally confined column has
one static state on the cone, and the two runs differ because the control yields under the
dynamic overshoot of its suddenly applied weight and then unloads. At the pin the final stresses
differ by up to 1.4 kPa, and the test asserts more than 1 % of the base stress.
