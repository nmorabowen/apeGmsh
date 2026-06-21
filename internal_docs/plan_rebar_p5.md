# Plan — `g.rebar` P5 (composed-Part libraries · beam dowel · twist)

Forward-looking implementation plan for the three externally-blocked P5
items left after the `g.rebar` geometry layer shipped (column/beam/circular/
wall generators, bundling, full ACI detailing, straight + mesh-native curved
geometry — all on `main`). Produced by a survey → synthesize → critique
multi-agent workflow and hand-folded with the critique fixes.

**Status: PLAN ONLY — no code yet.** Two independent tracks.

---

## The load-bearing finding (verified)

The keystone lives in the **neutral H5 zone**, *not* the opensees zone — this
**contradicts the naive read** (and two of the survey agents). Do not "fix" it
back.

- The composed-Part cage library saves/loads via `FEMData.to_h5` / `from_h5`
  → the **neutral** zone (`mesh/_femdata_h5_io.py`, `NEUTRAL_SCHEMA_VERSION =
  2.13.0`, line 165). The opensees zone (`apeSees(fem).h5` → `emitter/h5.py`,
  `SCHEMA_VERSION 2.19.0`) is the *deck-emit* path the library never touches.
- `reinforce_ties` is a **separate list** on the broker (`FEMData.py:765`),
  **not** inside `fem.elements.constraints`. So neither the neutral
  constraint-bucketing writer (`_femdata_h5_io.py:899-979`, iterates
  `.constraints`) nor the compose pipeline (`_compose.py:1248/1252`, walks
  `.constraints`) ever sees it. The neutral IO has **zero** reinforce handling
  today.
- Precedent to copy for dtype + ragged (vlen) arrays: **`surface_coupling`**
  (`mesh/_record_h5.py:205-283`; encode `_femdata_h5_io.py:1189`; decode 2279).
- `ReinforceTieRecord` already declares `tag_rewrite_spec` (`_kernel/records/
  _constraints.py:343`) — compose just never invokes it for ties.

⇒ Keystone bump is **neutral 2.13.0 → 2.14.0**. The opensees-zone 2.19→2.20
deck fix is a *separate, lower-priority* follow-on (A4), only for reinforced
apeSees decks, not for the cage library.

---

## Track A — H5 tie persistence + compose (the keystone, P5.1)

Ship **A1 → (A2+A3 together) → A4**. Highest value; unblocks the composed-Part
cage library.

### Pre-A1 decision (answer before writing A1's encoder/tests)
**Does `snapshot_id` / `fem_hash` (ADR 0021) include `reinforce_ties`
content?** If not, A1 reading ties back into the broker can shift `snapshot_id`
for *reinforced* models (previously they "round-tripped" by silently dropping
ties). Decide: (a) add ties to the content hash (lineage now tracks
reinforcement, document a one-time hash shift for existing reinforced `.h5`),
or (b) explicitly exclude them. This sets the A1 test matrix.

### A1 — persist + read back `ReinforceTieRecord` (neutral zone) · effort M · ✅ SHIPPED
**Shipped** (neutral schema **2.14.0 → 2.15.0** — `main` had moved past the
plan's 2.13.0 via ADR 0068). `/reinforce_ties` group + `reinforce_tie_payload_
dtype`; the `snapshot_id` check resolved to **Option B** (ties excluded from the
hash — `compute_snapshot_id` already excludes constraints too), so reinforced
round-trips are hash-stable. Tests in `tests/mesh/test_reinforce_tie_h5_
roundtrip.py`. The original step text (written against 2.13.0) follows.

- Add `reinforce_tie_payload_dtype()` to `_record_h5.py`, modeled on
  `surface_coupling_payload_dtype`. Scalars: `rebar_node` i64, `bond`
  utf8(''=None), `perfect`/`bond_scale`/`kt`/`kt_alpha`/`dtcr`/`excess` f64
  (NaN=None), `enforce` utf8, `bipenalty`/`in_bounds` u8, `name` utf8. Ragged
  vlen: `host_nodes` vlen(i64), `weights` vlen(f64), `direction` len-3.
  (1-to-N — simpler than `surface_coupling`'s CSR-of-CSR.)
- Add `_encode_reinforce_tie` / `_decode_reinforce_tie` mirroring the
  surface-coupling encoders; reuse the `_opt_scalar`/`_opt_vec3` helpers for
  symmetric None↔NaN / ''↔None.
- **Dedicated `/reinforce_ties` group + its own `_write_reinforce_ties` /
  `_read_reinforce_ties`.** Do **not** put it under `/constraints/` — the
  `_read_constraints` subset-match dispatcher (`_femdata_h5_io.py:2061`) would
  silently skip or mis-dispatch it. Call the writer from `write_neutral_zone`
  right after `_write_constraints`; wire the reader into `read_fem_h5` and pass
  the list via `reinforce_ties=` (constructor already accepts it,
  `FEMData.py:749`).
- Bump `NEUTRAL_SCHEMA_VERSION` 2.13.0 → 2.14.0 + version-history entry. Gate
  the new dataset on a **non-empty** list so tie-free files stay byte-stable.
- Drop the deferral warning in `FEMData.to_h5:1796-1807`.
- **Tests** (`tests/mesh/test_reinforce_tie_h5_roundtrip.py`): perfect-bond +
  bond-by-name round-trip with field-by-field equality (incl. None vs NaN);
  tie-free model → no `/reinforce_ties` group + unchanged `snapshot_id`;
  **reinforced model → `snapshot_id(fem) == snapshot_id(from_h5(to_h5(fem)))`**
  (the critique's must-add test); neutral-version assertion → 2.14.0.
- **Open question (re-added from survey, do not drop):** partitioned (MPI)
  reinforce-tie **dedup** — a tie on a cross-partition-shared host replicates
  per owning rank (cf. MP-constraint `_partition_dup` INV-1/INV-4). Either add
  an A1 partitioned-round-trip no-dup test, or explicitly scope A1 to
  non-partitioned and gate partition support as a later phase.

### A2 + A3 — compose teach-in **with** the cross-Part guard (one PR) · effort M · ✅ SHIPPED
**Shipped.** `g.compose` rewrites/merges `reinforce_ties` (offset `rebar_node`/
`host_nodes`, prefix `name`/`bond`); host ties preserved across the merge; new
`ComposeReinforceCrossPartError` + `_guard_reinforce_cross_part`. The guard is
**not** extended to tied-contact `SurfaceCouplingRecord` (those legitimately
bridge Parts — a deliberate scope call vs the critique). Tests in
`tests/mesh/test_compose_reinforce_ties.py`. Original step text follows.

Merged because A2 alone ships a **silent cross-Part corruption window** (the
critique's blocker): A2 offsets cross-Part tie tags uniformly → broken
conformal topology, no error.
- A2: add a `reinforce_ties` walk to compose (parallel to the `.constraints`
  walk at `_compose.py:1252`); run each through `_rewrite_record` (890-966)
  using `tag_rewrite_spec` (scalar `rebar_node` + array `host_nodes` offset;
  `name`/`bond` namespace-prefix to match material prefixing so re-emit
  `name_to_tag` still resolves). Add `reinforce_ties` to `_RewrittenBundle`
  and merge into `new_fem.elements.reinforce_ties` in `_merge_bundle_into_fem`.
- A3 (same PR): a compose guard that raises `ComposeUnsupportedError` when a
  tie's `rebar_node` + any `host_node` span **different** Parts (check against
  `_part_node_map`; degrade to no-op when absent). Apply the same check to
  `SurfaceCouplingRecord.slave_records` (the symmetric latent gap). v1 = reject
  (same-Part authoring); document recovery.
- **Tests**: two-Part compose carries both tie sets with offset tags +
  prefixed bond names; `weights`/`direction` survive unchanged; compose →
  to_h5 → from_h5 keeps merged ties (A1×A2); cross-Part tie raises with the
  offending node range; single-Part regressions pass.

### A4 — opensees-zone deck round-trip (separable follow-on) · effort M
Only needed for reinforced *apeSees decks*, not the cage library.

#### A4 minimal — retire the false deviation warning · ✅ SHIPPED
**Shipped.** A code re-survey at A4 time found the plan's premise was
**partly obviated by A1**: `apeSees(fem).h5(path)` writes the **neutral**
zone (with A1 ties, #706) into the *same* archive as the `/opensees` deck
zone, so a reinforced `model.h5` already carries its reinforcement — it
round-trips via `FEMData.from_h5` → `apeSees(fem).tcl()/py()/run()` (the
forward path re-runs `emit_reinforce_ties`). The
`H5ReinforceDeviationWarning` ("the H5 deck will be missing its embedded
reinforcement") was therefore **false**. A4-minimal:
- Retired `H5ReinforceDeviationWarning` (class + `__all__` + the
  `embedded_rebar` emission); `H5Emitter.embedded_rebar` is now a
  **silent** deck-zone no-op (the neutral zone owns persistence). Comments
  in `emitter/h5.py` document the deferred deck record.
- Rewrote `test_reinforce_emit.py::test_h5_defers_deck_zone_without_warning`
  (asserts no warning + no deck-zone reinforce record) and added
  `test_reinforce_composite.py::test_apesees_h5_deck_roundtrips_ties_via_
  neutral_zone` (a reinforced `apeSees.h5` → `read_fem_h5` recovers all
  ties, no warning).

#### A4 full — dedicated deck record + deck-replay · ⬜ DEFERRED (documented open item)
Not needed for any cage workflow; deferred. The original step text follows.
**Re-survey caveat (do before starting):** `OpenSeesModel.build()` →
`_replay_into` (`_internal/compose.py`) currently **does not replay MP
constraints at all** (equalDOF / rigidLink / rigidDiaphragm / embeddedNode)
— they are persisted + read into RO records but never re-emitted on the
deck-replay path. Making reinforce ties uniquely deck-replayable would
either be inconsistent with that, or imply also closing the MP-constraint
deck-replay gap. Scope this decision first.
- Replace the no-op `embedded_rebar` (`emitter/h5.py`). **Pass/keep
  the source `ReinforceTieRecord` and reuse the A1 encoder** — do **not**
  reconstruct it by parsing the positional Tcl-style args (the critique's
  cleaner path; eliminates the brittle inverse parser). `ReinforceTieRecord`
  has **no `ele_tag` field** (the tag is allocated at emit time), so deck
  replay must either persist the emitted `ele_tag` (parallel dataset) or
  re-allocate from a tag allocator seeded past the max replayed element tag
  (avoid colliding with the directly-replayed element tags).
- Stage-aware dual-append (ADR 0055): route ties to `_stage_current.
  reinforce_ties` when a stage block is open, write in `_write_stages`.
  Persist under `/opensees/constraints/reinforceTie` (named-lookup reader,
  like `embeddedNode` — safe; the opensees constraints reader does NOT
  subset-dispatch).
- Reader in `h5_reader.py` → deck-replay re-runs `emit_reinforce_ties`
  (bond NAME stored, name→tag deferred to re-emit, Option B, matches
  `build.py:3398`), AND surface the MP-constraint deck-replay gap above.
- Bump opensees `SCHEMA_VERSION` 2.19.0 → 2.20.0 + two-version-window tests.
  Gate dataset on non-empty list to keep tie-free deck `model_hash` stable.

---

## Track B — beam dowel (P5.2) + twist (P5.3) · independent track

Gated on a human decision (B0). Do **not** start code before B0.

### B0 — human-decision gate · effort S (no code) · ✅ RESOLVED (user, 2026-06-21)
**Decisions locked** (user, AskUserQuestion). Two of the three are *lighter*
than this section first implied because the infrastructure already exists
(see the re-survey notes per item):

1. **Orientation storage form → serialized `Orientation` + `roll_deg`.**
   Store an `Orientation` (default **`AlongBeam`**) + `roll_deg` on the
   `Bar`/`Path` L1 spec; the bridge derives each segment's `vecxz` at build
   via the **existing** `compute_vecxz_for_element` (`build.py:1422`).
   *Re-survey:* the orientation fan-out (`is_orientation_transform` +
   `compute_vecxz_for_element` + `_vecxz_key` dedup, `build.py:1407–1460`)
   already emits one `geomTransf` per distinct per-element `vecxz` for smooth
   beam-columns. The rebar gap is only that a bar emits as a **polyline of
   segments**, each needing its own tangent → a per-segment driver that
   *reuses* this machinery, not new orientation math.
2. **Mixed-ndf → `ndf=6` rebar nodes on the `ndf=3` host, via the existing
   per-node ndf overlay** (ADR 0048/0049 `ops.ndf` + `nodes_ndf`).
   `LadrunoEmbeddedRebar` couples the 3 translations; the 3 rotational DOFs
   are left for B0.3's stabilizer. *Re-survey:* per-node ndf already exists,
   so this is a wiring step (mark beam-rebar nodes `ndf=6` in the overlay),
   not new infrastructure.
3. **Twist policy → try existing `zeroLength` + SP FIRST; no new C++ class
   tag** (ADR 20 D6 option 1). Pin the rebar node's unconstrained rotational
   DOFs with an SP/`fix` on the rotational DOFs or a soft `zeroLength` to a
   ghost node with small rotational stiffness. **Escalate to a new fork C++
   ghost-node `zeroLength` (B2 option 2) ONLY if this proves insufficient.**
   Open sub-choice deferred to B2/B3: automatic stabilizer on every beam
   rebar vs explicit per-element opt-in.
- Recorded here + handoff (ADR-0010 / ADR-0067 carry the cross-references).

### B1 — cage auto-emits structural rebar elements · DESIGN RESOLVED (user, 2026-06-21) · effort L+ (multi-PR)

**Load-bearing finding (Explore re-survey, 2026-06-21).** `g.rebar` today emits
**geometry + coupling only**. `LadrunoEmbeddedRebar` (33005) is a **pure
coupling** element — it ties the rebar node's translational DOFs to host nodes
and carries **no axial stiffness** (its own header: *"the rebar's own axial
stiffness lives on a separate rebar element (corotTruss/beam) along the bar"*).
The bar's **structural element is the user's job today** (`ops.element.CorotTruss(
pg=…)`, exactly like the reference `Ladruno_scripts/ladruno_rc.py`).
`RebarMember.element` (`"truss"`/`"beam"`) is **stored metadata, never consumed**
(`core/RebarComposite.py:67`; grep-confirmed no downstream reader).

**Architecture decision (user, AskUserQuestion 2026-06-21): the cage AUTO-EMITS
the bar's structural element, behind an opt-in flag.**
- `place(..., emit_elements=False)` — **default off** (no behavior change for
  existing users who hand-emit `ops.element` on the rebar PG; opt-in avoids
  double-emit).
- `emit_elements=True` → consume `RebarMember.element`: emit **`CorotTruss`** for
  `"truss"`, **`dispBeamColumn`** for `"beam"`, one structural element chain along
  each bar PG, in ADDITION to the coupling (embedded ties / conformal shared
  nodes).

**Beam section model (user decision): a circular FIBER section built from `db` +
the bar's uniaxial steel material** (physically-derived axial + bending/dowel
stiffness, consistent with `db`, nonlinear-capable, matches the cage's
material-by-name model). NOT an elastic section (would need a separate `E` input).

**The wiring channel = mirror `g.reinforce`'s declare → resolve → emit** (the
proven broker path, `ReinforcementsComposite` → `resolve_reinforce` →
`FEMData.reinforce_ties` → `build.emit_reinforce_ties`):
1. **Declare** — `place(emit_elements=True)` records per-bar structural-element
   intent (PG, element kind, material, area; for beam: section-from-`db`,
   `beamIntegration`, `Orientation`+`roll_deg`).
2. **Resolve** — at `get_fem_data`, map each rebar PG to its line-element
   connectivity → a NEW broker record list on `FEMData` (e.g.
   `rebar_elements`), sibling to `reinforce_ties`.
3. **Emit** — a NEW bridge pass (`emit_rebar_elements`) emits
   `CorotTruss` / `dispBeamColumn`; H5-persist the record (neutral, mirror the
   A1 `/reinforce_ties` precedent).

**OPEN implementation question to resolve FIRST (B1a kickoff):** new dedicated
broker record + emit pass, **vs** routing through the EXISTING element-group /
`ops.element(pg=…)` machinery (how does a normal beam PG get emitted as
`forceBeamColumn` today — can the cage register that intent rather than invent a
record?). Pick the lower-surface path before coding.

#### B1a — opt-in flag + truss auto-emit · ✅ SHIPPED (emit path)
**OPEN question RESOLVED:** a dedicated `emit_rebar_elements` build pass
(mirrors `emit_reinforce_ties`), NOT reuse of the registered-`Element`
fan-out — because the bar's dim-1 line cells are **dropped from a dim-3
`FEMData`**, so connectivity must be resolved from the live mesh at
`get_fem_data` (like reinforce) and carried on the record; and material is
resolved by **name** at emit (Option B). Reusing the `Element` fan-out would
have needed mid-build primitive construction with deferred name resolution —
more coupling for no gain.
- `place(emit_elements=False)` flag (default off → byte-identical to today).
- `RebarElementRecord` (`_kernel/records/_rebar.py`, carries `connectivity`);
  `g.rebar` captures opted-in bars (`_emit_members`) and `resolve()` extracts
  their Line2/Line3 cells from gmsh at `get_fem_data` (`_fem_factory` calls it)
  → `fem.elements.rebar_elements`.
- `emit_rebar_elements` (`opensees/_internal/build.py`) emits one `CorotTruss`
  per line cell (material by name via `name_to_tag`); called after
  `emit_reinforce_ties` on both flat + per-rank paths; partitioned **fail-loud**
  (per-rank routing deferred, like reinforce). `element="beam"` raises
  `NotImplementedError` (→ B1b).
- Tests `tests/rebar/test_rebar_emit_elements.py` (4): off=no CorotTruss;
  on=one CorotTruss/cell w/ correct area+material; unregistered material fails
  loud; beam raises. 4361 mesh+opensees green; ruff+mypy clean.
- **B1a.2 (follow-on, deferred):** neutral-H5 persistence of
  `fem.elements.rebar_elements` (mirror the A1 `/reinforce_ties` pattern; the
  record already carries resolved `connectivity`). `FEMData.to_h5` **warns
  loud** in the interim (honest deferral — round-tripped file omits them; the
  in-memory place → `apeSees().tcl/py/run` path is unaffected).

#### B1b — beam auto-emit + ungate · effort L
- Circular **fiber section from `db` + steel material** + `beamIntegration` +
  `geomTransf`.
- **Per-segment `vecxz` driver** reusing `compute_vecxz_for_element`
  (`build.py:1422`); dedup distinct `vecxz` (`VECXZ_TOL`), one `geomTransf` per
  distinct vector — the smooth-beam fan-out (`build.py:1407-1460`) already exists,
  reuse it per polyline segment. Handle hook-segment tangent-reversal sign flips.
- **`ndf=6` rebar nodes** via the existing ADR 0048/0049 per-node overlay (B0.2);
  host stays `ndf=3`.
- **Persist `Orientation`+`roll_deg`** on the bar spec (`_kernel/defs/rebar.py`,
  serializable, round-trips) — add ONLY when B1b consumes it (avoid the
  metadata-only anti-pattern that bit `element`).
- **Remove the gates**: `RebarComposite.py:1129-1137` + the
  `NotImplementedError` in `transform.py:176/210/244` (or guarantee concrete
  `vecxz` substitution before `_emit`).
- Twist stabilization → **B2** (B0.3: try `zeroLength`+SP first).
- **Tests**: distinct-tag `geomTransf` per `vecxz` change (Tcl + py); reversed
  hook sign-flip; orientation survives to_h5/from_h5; **`make_conformal` on
  fragmented curved rebar** (critique missing-area); `ndf=6` rebar node emitted.
- **Adversarial gate warranted** (touches bridge build / ndf / transform —
  core-adjacent).

### B2 — twist stabilizer · effort **XL** (cross-repo, re-estimated)
- **First** (from B0 #3): if existing `zeroLength` + SP suffices, B2 collapses
  to an emit-ordering + SP change with **no new class tag** — strongly prefer.
- Only if not: new OpenSees C++ ghost-node soft-`zeroLength` variant — new
  `classTags.h` entry, `LADRUNO` header stamp, `LEDGER_implementations.md` row,
  `banner_features.txt` + `patch_banner.py`, Zone-A Ubuntu CI green. This is a
  **multi-PR fork effort** with the fork's stranded-commit / auto-merge hazards
  (per `~/.claude/CLAUDE.md`) — not a single "L".
- Zero ghost mass (keep `CentralDifferenceLadruno` diagonal); strict emit order
  (ghost → SP `u_G=u_R` → `zeroLength(R,G,k_twist)` → assemble); small-rotation
  vs co-rotational per ADR 20 D6 Mode-T.

### B3 — apeGmsh ghost-node tags + persistence · effort M · depends B1+B2
- Reserve a ghost-tag range above `max(node ids)`; allocate one ghost per
  beam-rebar tie before element emission so `fem_eid→ops_tag` (ADR 0026) covers
  ghosts; reserve in `tag_allocator.py` / `tag_resolution.py`.
- Track ghost decls on the broker; neutral `/nodes/ghost_info` dataset (another
  minor bump — batch with B1's bump if they land together); reader reconstructs
  ghosts and **errors if `ghost_info` is missing** on a `beam` rebar (never
  silently lose stabilization).

---

## Recommended sequence

`A1` ✅ → `A2+A3` ✅ → `A4-min` ✅ (`A4-full` ⬜ deferred) → **`B0` ✅** →
**`B1 design` ✅** → **`B1a` ✅ (opt-in flag + truss auto-emit)** →
`B1a.2` (neutral-H5 persistence of `rebar_elements`) → `B1b` (beam: fiber
section + per-segment orientation + ndf=6 + ungate) → `B2` (twist) → `B3`.

Track A is complete. **B0 + the B1 design are resolved** (auto-emit architecture
+ fiber-section-from-`db` beam model locked above). **B1a is the next coding
effort** — start by resolving the OPEN question (new broker record vs reuse the
existing element-group / `ops.element(pg=)` machinery), then ship the opt-in
`emit_elements` flag + truss auto-emit. B2's twist policy is **decided** (B0.3:
try `zeroLength`+SP first; new fork C++ class only if that fails).

## Cross-cutting / migration
- One-time `snapshot_id` shift for existing reinforced `.h5` once lineage
  tracks ties (owned by the A1 decision; document for users).
- Confirm `Results.from_h5` / `from_native` never needs `reinforce_ties`
  (read path doesn't re-emit) — state it as a scoping decision.

*Workflow provenance: `wf_b9e99b9e-30d` — 6-agent survey → synthesize →
critique; final revise folded by hand after two API drops on the large output.*
