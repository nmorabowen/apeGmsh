# ADR 0134 — MPM in apeGmsh: the particle set beside the deck, particle results in `.ladruno`, a three.js points layer

**Status:** Accepted (2026-11-13; drafted by the fork's MPM program (WP-232a), ratified by the
owner on #2976, "Yes, ratify"). Merging the PR that flips this Status (`human-gate`) is the
acceptance. The pointers in ADRs 0112, 0023 and 0113 are owed to a follow-up, because those files
are outside this slice's card (ADR 0121 D5).

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
