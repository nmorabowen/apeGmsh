# Handoff — `g.rebar` reinforcement-cage authoring (ADR 0067)

Status: **P0–P4 + full ACI detailing arc + bundled bars + 4th (`wall`)
generator + mesh-native curved geometry shipped** (PRs #687–#700, all on
`main`). All adversarial-review gates folded. **168 rebar tests green.** **P5
Track A is essentially complete — the composed-Part cage keystone (H5 tie
persistence + compose carry/guard) shipped (A1 #706, A2+A3 #707) and the
A4-minimal warning fix shipped; only the optional A4-full deck record +
all of Track B remain** (below).

`g.rebar` lets you author reinforcement cages — longitudinal bars, stirrups,
hooks, whole columns/beams/circular columns — as geometry, and **delegates**
the concrete↔steel coupling to machinery that already ships in apeGmsh. It does
not invent an embedding element; it routes to `g.reinforce`
(`LadrunoEmbeddedRebar`) or to gmsh `embed`.

See `src/apeGmsh/opensees/architecture/decisions/0067-reinforcement-cage-authoring.md`
for the design rationale and **`internal_docs/guide_rebar.md`** for the
user-facing guide.

**Detailing arc shipped (2026-06-20):** §25.7.2.3 cross-ties (#687), §18.7.5
column confinement auto-derive (#688), §18.6.4 beam hoop zone (#689), twin-tail
stirrup closure (#690), overlapping cell-hoop style (#691), circular columns —
hoops or spiral (#692), user guide (#693). A rectangular column/beam and a
circular column under `ACI318_seismic` are now fully self-detailing.

---

## The three layers

| Layer | What | Where |
|---|---|---|
| **L1** specs | frozen, serialisable data: `Hook` `Path` `Bar` `Stirrup` `Cage`, layout inputs `BarLayout`/`TieLayout`, fluent `BarBuilder`, detailing `Raw`/`ACI318`/`ACI318_seismic` + `BarCatalog` | `src/apeGmsh/_kernel/defs/rebar.py`, `src/apeGmsh/rebar/detailing.py` |
| **L2** composite | `g.rebar` — geometry generation + `place()` coupling router + `column()`/`beam()`/`circular_column()`/`wall()` generators | `src/apeGmsh/core/RebarComposite.py` |
| **L3** fluent | `g.rebar.bar(...).through(...).hook_end(...).as_(name)` | `BarBuilder` in `_kernel/defs/rebar.py` |
| hook math | pure (numpy) bend-plane + fillet primitives | `src/apeGmsh/rebar/_geometry.py` |

Public surface: `from apeGmsh.rebar import (Hook, Path, Bar, Stirrup, Cage,
BarLayout, TieLayout, BarBuilder, ACI318, ACI318_seismic, Raw, BarCatalog,
DetailingError)`. The composite is `g.rebar` on a live session.

---

## Quickstart

```python
from apeGmsh import apeGmsh
from apeGmsh.rebar import Cage, BarLayout, TieLayout, Hook, ACI318_seismic, BarCatalog

with apeGmsh(model_name="col") as g:
    g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3.0, label="Col")
    g.rebar.use_standard(ACI318_seismic(BarCatalog(unit_length=0.0254)))  # model in metres

    # standardized member → a Cage
    cage = g.rebar.column(
        section=("rect", 0.5, 0.5), height=3.0, cover=0.05,
        longitudinal=BarLayout(n_x=2, n_y=2, db="#8"),
        ties=TieLayout(db="#3", spacing=0.30, hinge_spacing=0.10, hinge_length=0.60))

    g.rebar.place(cage, into="Col", coupling="conformal")   # shared-node perfect bond
    g.mesh.sizing.set_global_size(0.3)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data()
```

Authoring primitives + fluent:

```python
bar = g.rebar.bar([(0,0,0),(0,0,3)], db="#8", material="rebar",
                  end_hook=Hook.standard_90())
tie = g.rebar.stirrup_rect(0.5, 0.5, 0.04, db="#3", material="rebar", z=1.0)
cage = Cage(bars=(bar,), stirrups=(tie,))

L1 = (g.rebar.bar(db="#8", material="rebar")          # fluent (no points → builder)
        .through([(0,0,0),(0,0,3)]).hook_end(Hook.standard_90()).as_("L1"))
```

---

## Coupling model (the central idea)

`g.rebar.place(cage, into, *, coupling=, per_member_coupling=, bond=, perfect=, ...)`

| coupling | mechanism | host requirement | notes |
|---|---|---|---|
| `"conformal"` | gmsh `embed` of the bar curves into the host **before** `generate()` → shared nodes, perfect bond | single un-meshed volume (label OR PG) in the **same session** | MPI-OK; generated cages are inset interior so they mesh without a boundary PLC |
| `"embedded"` | forwards each member to `g.reinforce` → `LadrunoEmbeddedRebar` | host must be a **physical group**; needs `bond=<LadrunoBondSlip name>` **xor** `perfect=<axial penalty>` | **single-process only** (partitioned `LadrunoEmbeddedRebar` raises) |
| mixed | `per_member_coupling={role: coupling}` | per the chosen modes | longitudinal conformal + ties embedded, etc. |

`coupling="conformal"` across a composed Part **raises** (`embed` can't cross
a Part boundary). `place()` runs a **Pass-0** that validates the whole cage +
host before mutating gmsh, so a bad cage never leaves the model half-built.

---

## Detailing standards

`Raw()` (explicit-only — every code rule raises `DetailingError`),
`ACI318()`, `ACI318_seismic()` over a `BarCatalog(unit_length=, base=)`.
`"<k>db"` length tokens and `Hook.standard_90()/standard_135()/standard_180()/
seismic_135()` resolve at bind time. ACI 318-19 Table 25.3.1/25.3.2 bend
diameters + hook tails are encoded and **independently re-verified** (Gate C).
`bar_diameter`/`bar_area` feed `ReinforceDef`; imperial `#N` uses the ASTM
nominal area, metric/raw uses π·d²/4.

---

## v1 limitations (warned + intentional)

These are documented behaviours, not bugs — a `warnings.warn` fires for each:

1. ~~**Cross-ties / supplementary legs not generated.**~~ **SHIPPED** (ADR
   0067 §8). `column()`/`beam()` generate ACI 318 §25.7.2.3 cross-ties for the
   intermediate (`n>2` per face) bars by default (`crossties=True`): a column
   leg per intermediate bar at every tie level (135° + 90° hooks, alternated
   end-for-end per §18.7.5.2); a beam vertical leg per index-aligned interior
   top/bottom pair at every stirrup station. Legs carry `role="crosstie"`,
   use the tie bar size, and resolve hooks from the cage standard (role-aware
   end-hook resolution). `crossties=False` restores the bare hoop. Embedded
   coupling is robust; conformal cross-ties form bar/tie T-junctions needing
   `make_conformal`. A count-mismatched beam now ties **every** interior bar to
   its nearest opposite-face bar (legs may be inclined; warned) — no interior
   bar is left unsupported. **Wide-section alternative (column AND beam):**
   `confinement_style="overlapping_hoops"` tiles the core with closed
   overlapping cell-hoops (every bar at a hoop corner) instead of straight legs.
2. ~~**Hinge densification is data-driven, not standard-derived.**~~ **SHIPPED
   for columns** (ADR 0067 §8). An `ACI318_seismic` column with no
   `TieLayout(hinge_spacing=, hinge_length=)` now auto-derives the §18.7.5
   confinement zone: `l_o` = max(depth, ln/6, 18 in) and `s_o` = min(¼·b_min,
   6·d_b,long, 4+(14−h_x)/3 in ∈ [4,6] in), via `ACI318_seismic.confinement_
   length`/`confinement_spacing`. `ties.spacing` governs outside the zone; an
   explicit hinge layout overrides; non-seismic stays uniform; a warning reports
   the derived values. **Beams too** (§18.6.4): an `ACI318_seismic` beam
   auto-derives the hoop zone `2h` + spacing min(d/4, 6·d_b,long, 6 in) via
   `ACI318_seismic.beam_confinement_length`/`beam_confinement_spacing`.
3. ~~**Stirrup closure is a single hook.**~~ **SHIPPED** (ADR 0067 §3).
   `place(twin_tail=True)` (default) emits the real twin-tail seam — both free
   ends of a closed stirrup carry the closure hook (two tails overlapping at
   the seam corner). `twin_tail=False` restores the single hook.
4. **Conformal embedding of boundary-touching bars** trips a tetgen PLC. The
   generators avoid this by insetting the cage interior; hand-authored bars
   whose endpoints sit on a host face should use `coupling="embedded"`.
   `on_conformal_infeasible="embedded"` only catches *embed-time* failures,
   not the mesh-time PLC.
5. **Circular columns shipped** (`circular_column(...)`, #692): `n_bars` on a
   circle + discrete circular hoops or a `spiral=True` helix, polygon-
   approximated with `n_segments` sides/turn, §18.7.5 confinement auto-derived
   (`h_x` = bar chord spacing).
6. ~~**Bundled (multi-bar) longitudinal positions are not generated.**~~
   **SHIPPED** (ACI 318-19 §25.6). `BarLayout(bundle=2|3|4, bundle_pattern=…)`
   on `column`/`beam`, `circular_column(bundle=…)`, and the hand-authoring
   `g.rebar.bundle(points, n=, db=, material=, toward=, …)` each expand a
   position into a contact bundle of individual offset bar lines (each a
   distinct member). The cluster sits on the nominal cover line and stacks
   inward (`u` toward the section centre, tangential `v = axis × u`); `"auto"`
   → line/triangle/square by count. Validation caps 1–4 bars (`#14`/`#18` → 2)
   and fails loud if the stack would cross the section centre. **Caveat:** at a
   corner the tangential pair leans toward a face by ≤ √2/2·d_b — inherent to
   bundling; inset for the equivalent diameter `√n·d_b` for strict corner cover.
   Beam overlapping-hoop style + full mismatched cross-tie support are now
   shipped (see limitation #1).
7. **Mesh-native curved geometry — SHIPPED.** `Path(curve="polyline"|"arc"|
   "spline")` (+ `arc_center` for arc); `circular_column(true_arc=True)` →
   true-arc hoops + spline spiral so the mesher seeds nodes on the true curve
   (vs the `n_segments` polygon, still the default); hand authoring via
   `g.rebar.bar(..., curve=, arc_center=)` / `g.rebar.stirrup(...)`. The
   realised line **elements stay straight 2-node chords** (no curved line
   element in OpenSees — only the Bézier solid/surface elements are curved);
   `true_arc` upgrades node placement, not the element. Emit fix: arc-center
   points are popped from the apeGmsh registry on `occ.remove` (else the
   geometry validator flags stale metadata at mesh time — also latent for
   true-arc hooks under conformal meshing).

---

## P5 — status (plan: `internal_docs/plan_rebar_p5.md`)

The plan (survey→synthesize→critique workflow) found the keystone is the
**neutral** H5 zone, not the opensees deck zone — see the plan doc. Two tracks:

**Track A — composed-Part cage library (the keystone). A1+A2+A3 SHIPPED:**

| Phase | Status | What |
|---|---|---|
| **A1** | ✅ #706 | `ReinforceTieRecord` round-trips through the **neutral** `model.h5` (`/reinforce_ties` group, schema **2.14.0→2.15.0**; `reinforce_tie_payload_dtype`). `snapshot_id` excludes ties (Option B, verified) so reinforced round-trips are hash-stable. |
| **A2+A3** | ✅ #707 | `g.compose` rewrites/merges ties (offset `rebar_node`/`host_nodes`, prefix `name`/`bond`) + preserves host ties; `ComposeReinforceCrossPartError` guard rejects cross-Part ties (NOT extended to tied-contact — those legitimately bridge Parts). |
| **A4-min** | ✅ | Retired the **false** `H5ReinforceDeviationWarning` (silent deck no-op now). Re-survey at A4 time found `apeSees(fem).h5()` already embeds the A1 neutral-zone ties in the deck archive, so a reinforced `model.h5` round-trips via `FEMData.from_h5` → `apeSees().tcl/py/run` — the warning ("deck will be missing reinforcement") was false. Tests: `test_h5_defers_deck_zone_without_warning` + `test_apesees_h5_deck_roundtrips_ties_via_neutral_zone`. |
| **A4-full** | ⬜ DEFERRED | Dedicated `/opensees/constraints/reinforceTie` deck record + `OpenSeesModel.build()` deck-replay, schema 2.19.0→2.20.0. **Not needed for any cage workflow.** Re-survey caveat: `_replay_into` does **not** replay MP constraints (equalDOF/embeddedNode/…) either — scope that gap first. See `plan_rebar_p5.md` §"A4 full". |

The composed-Part cage library now works through the neutral zone:
`g.compose("cage.h5", label=…)` carries the cage's ties (offset + prefixed)
into the host model. Open items (documented, not regressions — from the
adversarial review): partitioned (MPI) reinforce-tie **dedup** in the neutral
zone (A1 is non-partitioned-tested only); and a composed cage's **bond name is
namespace-prefixed** (`{label}.bond`), so the matching `LadrunoBondSlip`
material must be declared *after* `g.compose(...)` but *before*
`apeSees(fem).build()` for the re-emit `name→tag` resolution to find it.

**Track B — beam dowel (P5.2) + twist (P5.3). B0 RESOLVED (user, 2026-06-21); B1 next:**

| Phase | Status / decision |
|---|---|
| **B0** (human gate) | ✅ **RESOLVED.** (1) Orientation = serialized `Orientation`(default `AlongBeam`)+`roll_deg` on the bar spec, bridge derives per-segment `vecxz` via the EXISTING `compute_vecxz_for_element`; (2) beam-rebar nodes `ndf=6` via the existing ADR 0048/0049 per-node overlay (host stays `ndf=3`); (3) **try existing `zeroLength`+SP first — no new C++ class tag** (ADR 20 D6 option 1), escalate only if insufficient. See `plan_rebar_p5.md` §B0. |
| **B1** `element="beam"` rebar | ✅ **DESIGN RESOLVED** (user, 2026-06-21). KEY finding: the cage emits geometry+coupling only; `LadrunoEmbeddedRebar` is coupling-ONLY (no axial stiffness); `RebarMember.element` is metadata never consumed; the structural bar element is user-emitted today. **Decision: cage AUTO-EMITS the structural element behind an opt-in `place(emit_elements=…)` flag** (truss→`CorotTruss`, beam→`dispBeamColumn`), wired via a declare→resolve→emit broker channel mirroring `g.reinforce`. Beam section = **circular fiber from `db` + steel material**. Split into **B1a** (opt-in flag + truss auto-emit; resolve the new-record-vs-reuse-element-machinery question first) → **B1b** (fiber section + per-segment `vecxz` reusing `compute_vecxz_for_element` + `ndf=6` + ungate). See `plan_rebar_p5.md` §B1. |
| **B2/B3** twist stabilization | ⬜ `LadrunoEmbeddedRebar` ties translations only → rotational zero-energy mode. Per B0.3, **try existing `zeroLength`+SP** (apeGmsh-only ghost-tag alloc + H5 persistence). A new fork C++ ghost-node `zeroLength` (XL, cross-repo) is the fallback ONLY if the existing route can't stabilize the mode. |

---

## Working notes

- **Run tests:** `PYTHONPATH=src python -m pytest tests/rebar/` — apeGmsh is
  *not* pip-installed in the default Python here; `PYTHONPATH=src` imports this
  worktree directly (v2.0.0; gmsh 4.15.2 present). **168 tests**, ~2 s. The P5
  Track-A tests live under `tests/mesh/`:
  `test_reinforce_tie_h5_roundtrip.py` (A1) +
  `test_compose_reinforce_ties.py` (A2/A3). (2 `openseespy` Windows-DLL test
  failures under `tests/opensees/` are a pre-existing local-env gap, not a
  regression — CI is green.)
- **Scope `ruff --fix` to exact files**, never a directory — `ruff --fix
  src/apeGmsh/` will auto-fix dozens of pre-existing-debt lines across unrelated
  files. (Recover with `git checkout --` on everything outside your target set.)
- **Tests:** `tests/rebar/` — `test_rebar_specs.py` (L1), `test_detailing.py`
  (ACI), `test_rebar_geometry.py` (bend math), `test_rebar_composite.py`
  (coupling), `test_rebar_hooks.py` (hook emission), `test_rebar_generators.py`
  (column/beam/fluent), `test_rebar_bundles.py` (ACI §25.6 bundled bars),
  `test_rebar_wall.py` (wall curtains + cross-ties), `test_rebar_true_arc.py`
  (mesh-native curved geometry).
- **Gotchas baked in (don't re-trip):** `add_line`/`add_arc` take point refs
  not coords; `add_wire` rejects `label=`; hook arc+line weld via **point-tag
  reuse** (no `make_conformal` — which renumbers entities); arc-center
  construction points are `occ.remove`d (else they mesh as phantom nodes);
  `center_of_mass(tag, *, dim=3)`; `g.reinforce` is not chain-phase guarded so
  `place()` guards itself; emit geometry eagerly in `place()`.

Built phase-by-phase with adversarial-review gates after P0, P2, P4 (each a
multi-agent workflow that found real bugs — see the commit history).
