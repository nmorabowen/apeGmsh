# ADR 0134 — MPM in apeGmsh: the particle set at emit, particle results in `.ladruno`, a three.js points layer

**Status:** Proposed (2026-11-13). Drafted by the fork's MPM program (WP-232a); the
maintainer accepts it by merging the PR that lands it (`human-gate`). Until then it
admits nothing (ADR 0121 D4, D5).

**Owner:** nmora

**Program:** the fork's WP-232 (program issue nmorabowen/OpenSees#1075; its ADR
`Ladruno_implementation/232_ladruno_mpm_general_driver_adr.md` §7). apeGmsh slices are carded
on #1203 by T. The two file contracts are specified field by field in the fork's
`Ladruno_implementation/232a_mpm_contracts.md`, with fixtures in `tests/data/mpm/` there.

**Amends** [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) D1 (one more artifact,
D1 here). **Excepts** this file's integer version key from ADRs 0023 and 0113 (D1). **Builds on**
ADRs 0116, 0121, 0122, 0127, 0130, 0131, 0132 and the charter's P4. **Supersedes** nothing.

## Context

The fork is adding a general Material Point Method (MPM) driver: an `mpm …` command family
whose explicit engine runs on particles over a background grid that is reset every step. Its
ADR splits the work as apeGmsh's charter already does: apeGmsh owns geometry, seeding,
transport, reading results and viewing; the fork owns the mechanics and the results format.
That is also the licence-clean split, and it matches ADR-60a's standing rule that apeGmsh is a
seeder, not a runtime remesher.

Two files cross the boundary:

- **Contract 1, the particle set** (apeGmsh → fork). At 10⁵–10⁶ particles, per-particle
  commands in a deck are not an option. The set travels as HDF5 and the deck reads it with
  `mpm particles -file {path}`. Each particle carries its position, volume, mass or density,
  GIMP half-lengths, initial velocity and stress, a body id and a **material tag**.
- **Contract 2, particle results** (fork → apeGmsh). A new group inside `.ladruno`
  (`MODEL_STAGE[*]/MODEL/PARTICLES` and `RESULTS/ON_PARTICLES`), additive:
  `FORMAT_VERSION` stays 1 and `FORMAT_MINOR_VERSION` becomes 2. Outputs are ragged (particles
  are removed or added during a run), so they are stored concatenated with an offset table, the
  CSR form `.ladruno` already uses for variable connectivity.

Six questions need an apeGmsh answer, and each touches an accepted ADR: where the particle set
lives (ADR 0112 D1, D2), how results arrive (ADRs 0122, 0131), what a particle result is to
`Results` (the result ontology), how it is drawn (ADRs 0112 D8, 0116), who writes material tags
(charter P4), and what a staged MPM run may do live (ADR 0121, charter fix "live runs flat
only"). A seventh follows from ADR 0121 D4: the `mpm` primitives and verbs grow two frozen
registries.

## Decision

### D1 — The particle set is an emit artifact, `<stem>.particles.h5`, beside the deck

The particle set is **not** a FEMData stream in `model.h5`. It is a file the emitter writes
**next to the script**, in ADR 0112 D1's words, beside `<stem>.h5`, named `<stem>.particles.h5`.
The deck names it brace-quoted: `mpm particles -file {<stem>.particles.h5}`. The fork resolves a
relative `-file` path from **the directory of the deck that names it**, not from the process's
working directory (contract §1). An emitted deck therefore names the file by its bare name and
stays valid when the folder moves. Live runs write the file before the `mpm particles` call. The
layout is Contract 1 (`FORMAT = "ladruno-mpm-particles"`, `FORMAT_VERSION = 1`). The fork owns it
as it owns `.ladruno` (ADR 0122 D4): the fork's reader admits or refuses the file, and a user may
write one without apeGmsh. apeGmsh changes it only through a fork PR.

This adds a row to ADR 0112 D1's artifact table: "particle set, after emit, new". It is recorded as
an amendment there when this ADR is accepted. ADR 0112 already carries a pointer of 2026-11-10 that
widens D1 to a file a run does not rewrite, `<stem>.workspace.h5` ([ADR 0132](0132-workspace-file-and-variants-manifest.md)).
The particle set is the opposite case: a run writes it. D1's rule holds for it:

- it is written unconditionally whenever the model has MPM bodies;
- it sits at a conventional path next to the script;
- the next run overwrites it.

D2's rule holds as well. The file is its own optional artifact with its own version key, and a
reader that ignores it is unaffected. Like `.user.h5`
([ADR 0130](0130-states-quantities-and-the-user-results-file.md)) and `.workspace.h5`, it is a file
of the set, not a zone of `model.h5`.

**The viewer's pairing.** `.particles.h5` joins the set's pairing rule as a suffix of its own,
modelled on ADR 0132 D1's `.workspace.h5`:

- `apeGmshViewer/src/main/pairing.ts` matches it case-insensitively and before `.h5` in `SUFFIXES`
  (:55-60), so `slab.particles.h5` pairs with `slab.h5` and is never opened as a model with the stem
  `slab.particles`;
- the open set gains a `particles` slot beside `model`, `geometry`, `results`, `user` and
  `workspace` (`OpenSet` and `candidates`, :41-50 and :84-89);
- `classify`'s refusal message lists it.

**Its version key is an exception to ADRs 0023 and 0113.** apeGmsh's files carry semver zone keys
with a floor per zone (ADR 0113). This file instead carries an **integer** `FORMAT_VERSION` that the
fork owns, as `.ladruno` does. apeGmsh's readers gate on the exact integer, which today means
`(1,)`. They take no floor and no shim ledger, and the schema-corpus machinery does not cover the
file. ADR 0113's floor rule does not apply to it. This ADR records that exception by name.

It pairs with its model like the geometry sibling: the file's `PROVENANCE` group carries
`session_id` (equal to `<stem>.h5` `/meta/session_id`), `apegmsh_version`,
`apegmsh_commit`, and `fem_hash` / `model_hash` copied byte-equal when the archive has them.
`SET_ID`, a uuid4 at the file's root, is copied by the fork into the `.ladruno` it writes
(`MODEL/PARTICLES@SOURCE_SET_ID`), so the results name the exact particle set they came from.

**What `model.h5` holds instead** is the *declaration* that produced the set: the grid, the
bodies, the particles per cell, the basis. That is bridge data and lands with the `mpm`
primitives (D7) in the `/opensees/` zone, as `decls` rows, so the set is reproducible from the
archive and the archive stays the model.

### D2 — Particle results are the `.ladruno` particle group, read at `FORMAT_MINOR_VERSION ≥ 2`

ADRs 0122 and 0131 stand: apeGmsh reads one results format, and there is no new results file
and no native writer. Particle results are read from the fork's `.ladruno` group defined by
Contract 2.

- `LADRUNO_SUPPORTED_FORMAT_VERSIONS` stays `(1,)` in Python and TypeScript. The change is a
  minor: a reader that predates it opens the file and skips the particle groups.
- `Results.from_ladruno` (`results/readers/_ladruno.py`) and the viewer
  (`apeGmshViewer/src/reader/ladruno*.ts`) read the particle group when
  `INFO/FORMAT_MINOR_VERSION ≥ 2` and the stage has `MODEL/PARTICLES`. They accept the fork's
  attribute encodings (1-element arrays, fixed-length strings, an empty string written as an
  absent attribute) and the int64 ids, converted to `Number` with the existing safe-integer check.
- A particle-only stage is written exactly as the fork's recorder writes an empty stage
  (`EMPTY_PARTITION = 1`, zero-length `MODEL/NODES`, empty FE result groups), so a reader that
  predates minor 2 opens it as an empty stage instead of failing.
- A part file (`<stem>.part-<N>.ladruno`) with particles is refused by name: the fork's v1
  driver is serial and Contract 2 forbids it.
- apeGmsh's own `.ladruno` writers (`_ladruno_writer.py`, the MPCO converter) write no
  particles and keep minor 1. An MPCO file has no particles, so the converter has nothing to add.
- The fork's `ladruno_format.validate()` checks the particle group; apeGmsh's tests run it on
  the fork's fixture `tests/data/mpm/particles_v1m2.ladruno` (copied into apeGmsh's fixtures with
  its source sha) and, on the `live-fork` lane, on a real run.

### D3 — The particle-cloud result ontology

A particle result is not a node, element or Gauss result: its carriers move, appear and
disappear. It gets its own level and is never keyed by node or element ids.

- **Level.** `ResultLevel.PARTICLES = "particles"` in `results/readers/_protocol.py`.
- **Slab.** `ParticleSlab(component, values, particle_ids, offsets, time)`: `values` is
  `(S, nComp)` concatenated over outputs, `offsets` `(T+1,)`, `particle_ids` `(S,)`, `time`
  `(T,)` — the file's own ragged layout, with no copy. `ParticleSlab.output(k)` returns one
  output's `(ids, values)`; `ParticleSlab.dense()` returns `(T, M, nComp)` aligned to the stage's
  registry ids with `NaN` where a particle is absent. `.item()` follows ADR 0115 D3.
- **Composite.** `results.particles.get(component=…, body=…, time=…)`. Selection is by **body
  name** (Contract 2 carries the names) or by explicit particle ids; it does not mix in
  `_SelectionMixin`, whose node / element / label selectors have no meaning for particles.
  `results.particles.bodies()`, `.registry()` (id, body, material tag, mass) and `.grid()` read
  the per-stage metadata.
- **Components** are spelled as the file spells them (ADR 0122 D7): `POSITION/x`,
  `STRESS/sigma22`, `STATE_<token>/<label>`. No token table is added.
- **Stages** map as for other results; stage-name binding (`_bind_stage_names`) works unchanged.

### D4 — Particles are drawn by a three.js points layer in apeGmshViewer

ADRs 0112 D8 and 0116 decide the technology: a new view is three.js in apeGmshViewer, and the Qt
viewers get nothing. ADR 0087's "dots" are Qt chrome glyphs whose own row says they are never
used for another point cloud, so they are not reused.

- One `THREE.Points` per stage, built after the node layer (`renderer/nodes.ts`,
  `makeNodePoints`), with a position buffer filled per output from the file by hyperslab
  (`OFFSETS[k]..OFFSETS[k+1]`), never the whole history in memory.
- Coloured by any particle field component through the existing contour ramp and unit display;
  `NaN` (`STATE_*` on particles whose material does not answer) draws in the "no value" colour.
- The grid (from `MODEL/PARTICLES/GRID`) is drawn as an optional wire box.
- The seeded set (`<stem>.particles.h5`) may be shown before a run, from the same layer.
- The work gets a row in `apeGmshViewer/MEASUREMENTS.md` (ADR 0112 D5) at 10⁵ and 10⁶
  particles: time to first frame, frame rate while scrubbing outputs, memory. ADR 0112's data
  server is considered only if those numbers demand it.
- The viewer's visual check (`apegmsh-viewers-visual-check`) runs with evidence, and the owner
  approves the screen (ADR 0112 D7).

### D5 — Seeding: lattice first; Gauss-point seeding only with positive weights

- **Lattice seeding** is the v1 method: `n` particles per grid cell per axis, kept where the point
  lies inside a body (Gmsh's inside test or the body's own mesh), with volume `∏(h/n)` (times the
  plane-strain thickness in 2D), GIMP half-lengths `h/(2n)`, and a particles-per-cell uniformity
  report against the declared grid.
- **Gauss-point seeding is not in v1.** `fem/_quadrature.py` states its own scope ("minimal Gauss
  quadrature rules", sufficient where "the absolute scale of the quadrature weights cancels"), and
  its tetrahedron rule is Keast's 11-point rule with a negative centroid weight, which would make a
  negative-volume particle in every tet. When it comes, it uses positive-weight rules only, a
  per-family "supported for seeding" table with loud refusal, and a signed-Jacobian check
  (`compute_jacobian_dets` returns `abs(det)`).
- Contract 1 refuses zero or negative volumes; apeGmsh never writes one.

### D6 — Material tags are resolved at emit (charter P4)

Users never type tags. A body declares its material by label; the bridge allocates the
`nDMaterial` tag at build (`TagAllocator`), and the **emitter** writes Contract 1's
`MATERIAL_TAG` column from that allocation, in the same pass that emits the material. A seeded
set held before emit carries body labels only. Until the `mpm` primitives land (D7), the seeding
writer takes the label → tag map as a private argument that only the emitter and its tests
pass; no public API asks for a tag.

### D7 — Bridge primitives only after a pinned fork release; staged MPM runs are deck-only

- apeGmsh consumes the `mpm` primitives only when two things hold: the fork PR that adds `mpm` has
  merged into `ladruno`, **and** `FORK_PIN` names a fork release that contains it, with its four
  keys bumped together.
- Each member carries:
  - the `ladruno_fork` marker;
  - a stock refusal that names the fork, raised at build time or as a `_FORK_ONLY_*` entry (ADR
    0121 D4, ruling 1);
  - **a warrant-row test**. That test stays Inventory until it *passes*, not skips, in the nightly
    `live-fork` lane (ruling 1; ruling 3 for skips);
  - a row in `docs/concepts/backend-capabilities.md`;
  - capability probes that return `None` when they are inconclusive.
- **Staged MPM runs are deck-only.** Live runs are flat (charter fix, ADR 0121 D7, ruling 2). A
  K0 start followed by a plastic stage uses `parameter … pattern <mpmTag>` + `updateParameter`
  in an emitted deck; the live route refuses a staged MPM program by name.
- Paths in emitted Tcl are brace-quoted.

### D8 — The registries this ADR admits (ADR 0121 D4)

When accepted, this ADR is the Accepted ADR that the census entries cite. It follows the shape of
[ADR 0127](0127-modal-basis-primitive-family.md), which admitted `ModalBasis`, as far as that shape
is known today:

- **Base class.** `apeGmsh.opensees._internal.types.MPM(Primitive)` is the abstract base for the
  `mpm` sub-commands: grid, body, particles, grid conditions, damping, output and analyze.
- **Allocator kind.** `mpm`, as one row in `apesees._KIND_BY_FAMILY`, so the registry lock
  (`test_verbs_lock.py` check (h)) finds it.
- **The Family line.** `tests/families.py` gains `Family("mpm", MPM, …:ALL_MPM)`, and
  `tests/freeze_census.py` gains `ADMITTED["primitive families"]["mpm"] = "0134"`.
- **The verb family.** `emitter/verbs.py::FAMILIES` gains `"mpm"`, with its census entry citing this
  ADR.
- **The contract test.** A parity test checks every member's fork command line on all five emitters,
  as `tests/opensees/parity/test_chain_m_command_parity.py` does for `ModalBasis`.

Unlike `ModalBasis` (ADR 0127 D2), `mpm` members are registered on the bridge and reach decks and
`model.h5`, because the particle-set declaration is archived (D1). Their exact names, arguments and
census entries depend on the fork guide's command names, which do not exist yet. **The bridge slice
(chain M, link M3) amends this D8 with the exact census entries** before it lands. This ADR admits
no new constraint kind and no MCP tool or habitat verb.

## Consequences

- A model with MPM bodies leaves one more file: `<stem>.particles.h5`, which can be large
  (about 100 bytes per particle before compression). It is regenerable from `model.h5`.
- Particle results open in today's apeGmsh as a model with no particles until the readers learn
  minor 2; no released apeGmsh breaks on a fork file.
- `Results` gains a level whose slabs are ragged. Code that assumes `(T, N)` slabs does not apply
  to it, and the composite keeps it apart from node and element selection.
- The fork owns both contracts. apeGmsh's freedom to restructure the particle set is a fork PR
  away, which is the same price ADR 0122 D4 pays for `.ladruno`.

## Alternatives rejected

- **A FEMData stream in `model.h5`** (with the `compose-streams` obligations and the neutral-zone
  bump). The set holds bridge-allocated material tags, which FEMData, being solver-neutral and
  pre-bridge, cannot hold (P4); it would put up to hundreds of megabytes in the archive the viewer
  opens for every model; and the fork's reader would then parse apeGmsh's archive and its zone
  versioning.
- **A new `/particles` zone in `model.h5`** with its own key. Same size and coupling problems, and
  the ADR-0112 integer policy (no int64) would conflict with the int64 particle ids the fork's ADR
  fixes.
- **A new results file for particles**, or particles in native `results.h5`. ADRs 0122 and 0131
  allow one results format and no new native writer.
- **A `.ladruno` `FORMAT_VERSION` bump to 2.** It would make every new fork file unreadable by
  today's apeGmsh, which opens `(1,)` only; the group is additive, so the minor moves.
- **Per-particle Tcl commands.** Unusable at 10⁵–10⁶ particles.
- **The Qt viewers or ADR 0087's dots.** Frozen (ADR 0112 D8).
- **Gauss-point seeding in v1** with today's quadrature rules. Negative weights (D5).

## References

- The fork: `232_ladruno_mpm_general_driver_adr.md` §7; `232a_mpm_contracts.md` (both contracts,
  field by field); `ladruno_schema_v1.md`; `Ladruno_scripts/ladruno_recorder_tests/ladruno_format.py`
  (the executable spec); fixtures `tests/data/mpm/particle_set_v1.h5` and
  `tests/data/mpm/particles_v1m2.ladruno`.
- apeGmsh: [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md),
  [ADR 0116](0116-render-technologies.md),
  [ADR 0121](0121-charter-v2-b-with-amendments.md),
  [ADR 0122](0122-ladruno-canonical-results-format.md),
  [ADR 0131](0131-all-in-ladruno.md),
  [ADR 0127](0127-modal-basis-primitive-family.md),
  [ADR 0130](0130-states-quantities-and-the-user-results-file.md),
  [ADR 0132](0132-workspace-file-and-variants-manifest.md), ADRs 0023 and 0113 (the semver keys D1
  excepts), `architecture/charter.md` P4, `apeGmshViewer/src/main/pairing.ts`,
  `architecture/h5-schema.md` ("Integer policy for the ADR 0112 zones", the geometry sibling),
  `results/schema/_versions.py`, `FORK_PIN`, `docs/concepts/backend-capabilities.md`.
