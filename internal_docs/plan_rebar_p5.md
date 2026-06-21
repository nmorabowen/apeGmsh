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

### A2 + A3 — compose teach-in **with** the cross-Part guard (one PR) · effort M
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
- Replace the no-op `embedded_rebar` (`emitter/h5.py:1286-1308`). **Pass/keep
  the source `ReinforceTieRecord` and reuse the A1 encoder** — do **not**
  reconstruct it by parsing the positional Tcl-style args (the critique's
  cleaner path; eliminates the brittle inverse parser).
- Stage-aware dual-append (ADR 0055): route ties to `_stage_current.
  reinforce_ties` when a stage block is open, write in `_write_stages`.
- Reader in `h5_reader.py` → `fem.elements.reinforce_ties` so `from_h5` →
  `build` re-runs `emit_reinforce_ties` (bond NAME stored, name→tag deferred to
  re-emit, Option B, matches `build.py:3398`).
- Bump opensees `SCHEMA_VERSION` 2.19.0 → 2.20.0 + two-version-window tests;
  retire `H5ReinforceDeviationWarning`. Gate dataset on non-empty list to keep
  tie-free deck `model_hash` stable.

---

## Track B — beam dowel (P5.2) + twist (P5.3) · independent track

Gated on a human decision (B0). Do **not** start code before B0.

### B0 — human-decision gate · effort S (no code)
1. **ADR-0010 Phase-4 scope:** rebar polyline segments do **not** align with
   smooth-element integration points, so a new **per-segment** `vecxz` driver
   is needed regardless. Decide storage form: serialized `Orientation` +
   `roll_deg` (regenerable, compact) vs pre-computed per-segment `vecxz`
   (lossless, larger).
2. **`ndf=6` vs `ndf=3`** (critique prerequisite): beam-rebar nodes need 6 DOF
   (the rotational DOFs are exactly what create the twist zero-energy mode)
   while the solid host is `ndf=3` and `LadrunoEmbeddedRebar` couples
   translations only. Decide how mixed-ndf nodes are inferred + emitted.
3. **Twist policy / class-tag avoidance** (critique major): decide **first**
   whether an **existing `zeroLength` + SP on the torsional DOF** (ADR 20 D6
   option 1) solves the twist mode with **zero new C++ class tag**, before
   reserving one. Also: automatic `-torsion_stiffness_guard` on every beam
   rebar vs explicit per-element.
- Record decisions in ADR-0010 / ADR-0067.

### B1 — ungate curved/hooked beam-element rebar · effort L
- Per-segment driver: midpoint tangent → `orientation.triad_at` →
  `resolve_vecxz` (`_orientation.py:440-479`); dedup distinct `vecxz`
  (`VECXZ_TOL`), one `geomTransf` per distinct vector — extend the fan-out at
  `build.py:1407-1453`. Wire each segment's beam element to its transf tag.
- **Add the `ndf=6`-on-rebar-node handling** (B0 #2) — a step, not a footnote.
- Remove the gates: `RebarComposite.py:1129-1137` and the
  `NotImplementedError` in `transform.py:176/210/244` (or guarantee concrete
  `vecxz` substitution before `_emit`).
- Persist per-segment orientation per B0 (extends the A1 record or a sibling
  `reinforce_bars` record); handle hook-segment tangent-reversal sign flips.
- **Tests**: distinct-tag `geomTransf` per `vecxz` change (Tcl + py); reversed
  hook sign-flip; orientation metadata survives to_h5/from_h5; **a step/test
  owning `make_conformal` on fragmented curved rebar** (critique missing-area).

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

`A1` → `A2+A3` (one PR) → `A4` → **`B0` (human gate)** → `B1` → `B2` → `B3`.

Track A is execution-ready once the **`snapshot_id` question** is answered.
Track B must not start code until **B0** (especially the reuse-existing-
`zeroLength`-vs-new-class decision and the `ndf=6` call).

## Cross-cutting / migration
- One-time `snapshot_id` shift for existing reinforced `.h5` once lineage
  tracks ties (owned by the A1 decision; document for users).
- Confirm `Results.from_h5` / `from_native` never needs `reinforce_ties`
  (read path doesn't re-emit) — state it as a scoping decision.

*Workflow provenance: `wf_b9e99b9e-30d` — 6-agent survey → synthesize →
critique; final revise folded by hand after two API drops on the large output.*
