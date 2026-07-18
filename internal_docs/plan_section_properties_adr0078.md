# Plan — Section-properties analyzer (ADR 0078)

Implements **ADR 0078**
(`src/apeGmsh/opensees/architecture/decisions/0078-section-properties-analyzer.md`).

**Goal.** Native in-process cross-section analysis — geometric / warping /
plastic / stress on any meshed 2-D face — as the `SectionProperties(fem)`
broker in `apeGmsh/sections/`, with declarative bridge binding via
`ops.section.ComputedSection` and the S6 Qt inspector. No new runtime
dependency; the PyPI `sectionproperties` package is a dev-only CI oracle.

**Verified facts this plan rests on** (probed live 2026-07-16; re-verify
anything load-bearing at slice start — the verify-against-live-source law):

- FE kernel: `apeGmsh/fem/_shape_functions.py` — per-type `N`/`dN` for
  tri3/tri6/quad4/quad8/quad9 (+3-D), `get_shape_functions` `:841`,
  `compute_physical_coords` `:859`, `compute_jacobian_dets` `:884`;
  `_quadrature.py` — `gauss_quad_2d` `:49`, `gauss_tri` `:95`. Node
  ordering is gmsh ordering **by construction** (kernel already consumes
  gmsh connectivity for HRZ + Gauss extrapolation) — no re-mapping risk.
- Input broker: `g.mesh.queries.get_fem_data(dim=2)` → `FEMData` with
  `nodes` coords `(N, 3)`, per-type `ElementGroup` connectivity arrays,
  PG membership views (`mesh/FEMData.py`).
- Section base + emit-time tag resolution: `opensees/section/beam.py`
  (`ElasticSection` fields `E, A, Iz, Iy, G, J, alphaY, alphaZ`; 2-D/3-D
  form auto-selected), `section/_tag_resolver.py`. Consumers type against
  the `Section` base: `integration.py:111` (`Lobatto.section: Section`,
  `resolve_tag` at `:118`, `dependencies()` at `:121`),
  `aggregator.py:83`, `element/zero_length.py:269`, `element/shell.py`.
  Bridge ns registration: `opensees/_internal/ns/section.py` (`_SectionNS`).
- Builders: `sections/_builder.py` — `W_solid :250`, `rect_solid :362`,
  etc.; all `(shape params, length, *, anchor, align, label, lc,
  translate, rotate) -> Instance`. `g.parts.fragment_pair(label_a,
  label_b, *, dim)` — `core/_parts_fragmentation.py:227`.
- CAD import: `g.model.io.load_dxf :761`, `load_step :556`, `load_iges :484`.
- Qt: `ViewerWindow` has the `QT_QPA_PLATFORM=offscreen` raise-guard to
  reuse; gallery tests use the QTimer-driver + screenshot pattern.

**Execution hazards (standing memory — apply to every slice):**

- Python = `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe`, always.
- The editable install resolves `apeGmsh` to the **main-repo `src/`**, not
  the worktree — run worktree tests with the worktree `src/` first on
  `PYTHONPATH` (or `python -m pytest` from a shell with it exported) and
  deliver via PR off `origin/main`.
- Judge test health by **targeted testpaths**, never the whole tree on
  Windows (baseline cp1252 pollution cascade).
- Warn-as-contract: any slice that adds a warning class must
  `pytest -W error::<Category>` it.
- Every PR: `--base main`. Never stack. Merge in order; rebase the next
  slice onto fresh main after each merge. `CHANGELOG ## Unreleased`
  re-conflicts — regenerate that hunk at rebase time.
- `ComputedSection` lives under `src/apeGmsh/opensees/` → ruff hard gate +
  mypy ratchet apply; keep both at-or-below baseline. `apeGmsh/sections/`
  gets ruff repo-wide.

**Success criterion (whole task).** All six slices merged; the PyPI-oracle
CI job green (skip-if-not-installed elsewhere); `ComputedSection` deck
byte-identical to a hand-typed `ElasticSection`; inspector smoke test
green; ADR 0078 flipped to Accepted with per-slice PR numbers recorded.

---

## Model & review policy

**Model assignments** (per slice, for the implementing session/agents):

| Work | Model | Why |
|---|---|---|
| S2 warping solver, S5 lowering + `ComputedSection`, S1 API core (naming-law accessors) | **Opus 4.8** | math-heavy / pattern-setting / silent-failure-prone |
| All adversarial-review agents (G-A/G-B/G-C) | **Opus 4.8** | refutation quality is the point |
| S3 plastic bisection, S4 stress recovery | Opus 4.8 preferred; Sonnet feasible with G-B still gating | closed-form over solved fields |
| S6 Qt inspector, flat-face builders (mechanical mirror of `W_solid`), docs/CHANGELOG chores | Sonnet feasible | scaffolding with existing patterns |

**Adversarial-review gates** (multi-agent `Workflow` runs — precedent: the
ADR 0073 30-agent review, ADR 0045 13-agent design workflow; user has
opted in for these junctures). Each gate: N independent finders with
distinct lenses → per-finding refuter panel (≥2/3 to confirm) →
confirmed findings become blocking fixes on the slice PR before merge.

- **G-A — after S2 core is up as a draft PR** (the critical juncture: the
  singular-system + shear-integral math). Lenses: (1) formulation vs
  Pilkey (warping BVP, Lagrange-multiplier regularization, shear-centre
  elasticity + Trefftz integrals, `As` from unit-shear complementary
  energy, composite `G`-weighting); (2) numerics (semidefinite solve
  conditioning, multiplier row scaling, quadrature degree per element
  type); (3) test-refutation (try to construct a mesh/shape where the
  implementation and the oracles disagree — anisotropic aspect ratios,
  thin-walled open shapes, `G`-override strips); (4) API conformance vs
  ADR field list.
- **G-B — after S5 lowering lands in a draft PR** (the silent-failure
  juncture: axis/sign mapping). Lenses: (1) the `Ixx_c→Iz` / `Iyy_c→Iy` /
  `alphaY`/`alphaZ` mapping against OpenSees local-axis docs *and* an
  end-to-end numeric check (cantilever tip deflection both axes, apeGmsh
  frame vs closed form, strong/weak swapped-section refutation);
  (2) emit-time behavior (deck byte-equality, memoization = one solve,
  fail-loud paths, 2-D vs 3-D `ElasticSection` form selection);
  (3) composite reference-`E` semantics.
- **G-C — before flipping the ADR to Accepted** (completeness critic,
  one pass): every API-contract field/method/error in ADR 0078 exists
  and behaves as written; naming law enforced everywhere (grep for
  accessor bypasses); docs/skill updated; anything missing becomes a
  follow-up list in the ADR's status line.

Gates G-A and G-B are **blocking**: the slice PR does not merge until
confirmed findings are fixed and re-verified.

---

## S1 — `SectionMaterial` + broker skeleton + geometric analysis (PR-1)

Files: `sections/_materials.py`, `_analysis.py`, `_geometric.py`,
`_errors.py`, `_mesh_snapshot.py`; tests under `tests/sections/`.

1. `_errors.py`: `SectionMeshError`, `CompositeSectionError`,
   `SectionAnalysisError`, `SectionAccuracyWarning` (re-export from
   `apeGmsh.sections`).
2. `SectionMaterial` frozen dataclass per ADR — including the `G=`
   override (validation: `E > 0`, `-1 < nu < 0.5`, `G > 0` when given,
   `fy > 0` when given).
3. `_mesh_snapshot.py`: constructor-time extraction from `FEMData` →
   flat arrays (coords `(N,2)` after planarity reduction, per-type
   connectivity, `element → material` index from PG map) + all input
   gates (2-D-only, one-plane, PG exact-cover, `disconnected` policy
   stored). Connected-component labeling here (scipy.sparse.csgraph over
   node adjacency) — computed once, used by S2.
4. `_geometric.py`: one Gauss loop accumulating the E-weighted integrals;
   `GeometricProperties` with the rigidity-form fields + the unprefixed
   property law (`CompositeSectionError` guidance message names
   `transformed(e_ref=...)`) + `transformed()`.
5. `summary()` + `_repr_html_` on the analyzer and `GeometricProperties`.
6. Export `SectionProperties` from `apeGmsh` top level.

**Verify (gate):** analytic oracles — rectangle (A, I, Z, r exact),
circle, offset rectangle (global vs centroidal + Steiner), rotated
rectangle (φ, principal I); composite two-material strip (hand-computed
EA/EIxx_c); geometric-only mode == E=1; disconnected two-rectangle
geometric() (common-centroid Steiner terms, **no flag needed**);
PG-coverage fail-loud (uncovered + doubly-covered); accessor law tests
(homogeneous passes, composite raises, `transformed` round-trip).
`ruff` clean; targeted `pytest tests/sections -q` green.

## S2 — Warping solve + disconnected policy (PR-2) — **gate G-A**

Files: `sections/_warping.py` (+ `_assembly.py` if shared with S4).

1. Assemble `K = Σ ∫ Bᵀ G B |J| dw` per element type via the fem kernel;
   `scipy.sparse` COO → CSC, one `splu` factorization per connected
   component.
2. Pure-Neumann regularization: bordered system with the Lagrange row
   `∫ N dA` (enforces `∫ω dA = 0`) — **not** node pinning (ADR
   rationale: pinning distorts shear-centre integrals on coarse meshes).
3. Three RHS solves (ω, Ψ, Φ per Pilkey); derived quantities: `GJ`,
   shear centres (elasticity + Trefftz), `EΓ`, `GAs_x/GAs_y/GAs_xy`,
   monosymmetry constants.
4. `disconnected="sum"`: per-component solves + the combination rules
   from the ADR (ΣGJᵢ, ΣGAsᵢ, GJ-weighted shear centre, per-part
   results on `WarpingProperties.parts`); default `"raise"` names the
   component count.
5. `SectionAccuracyWarning` on tri3/quad4 (test with `-W error`).

**Verify (gate, then G-A review before merge):** circle `J = πr⁴/2` to
mesh-convergence tolerance; rectangle `J` vs the classical series; thin
channel shear centre vs closed form; circular tube `As ≈ 0.9A`-class
check vs the package oracle; **PyPI oracle** (dev-only, skip-if-absent):
I-section + holed box on matched meshes — `J`, shear centre, `As`, `Γ`
within convergence distance; tri3→tri6 convergence study documents the
warning; two-rectangle `"sum"` exactness; unfragmented-touching-faces
fail-loud; `G`-override bound test (strip `G→0` → sum, rigid `G` →
connected solve, monotone in between). **Then run G-A.**

## S3 — Plastic analysis (PR-3)

`sections/_plastic.py`: bisection on the neutral-axis intercept along
centroidal x/y and principal 11/22 (each trial: per-element side split
by Gauss-point classification, signed `∫fy dA` imbalance); derived
plastic centroids, `Mp_*`, shape factors; `fy`-on-every-material gate;
non-bracketing → `SectionAnalysisError`.

**Verify (gate):** rectangle `Z = bh²/4`, circle `Z = 4r³/3` (as
`Mp/fy`); asymmetric T-shape vs PyPI oracle; composite steel+`fy`-less
concrete raises naming the PG; shape-factor rectangle = 1.5.

## S4 — Stress recovery + plots (PR-4)

`sections/_stress.py`: six **unit-load fields** computed once from the
cached solves (σ per unit `N/Mxx/Myy`, τ per unit `Mzz/Vx/Vy`), stored
on the analyzer; `stress(...)` = linear blend; Gauss→node extrapolation
via `results/_gauss_extrapolation` machinery, averaging within material
regions only; `SectionStress` (`get(component, pg=)` with per-action
terms, `plot()` tricontour); `plot_mesh` / `plot_section` glyph overlay.
Disconnected `"sum"` load distribution per the ADR (Mzz ∝ GJᵢ, V ∝
flexural shares).

**Verify (gate):** pure `N` → uniform `σ = N/A`; pure `Mxx` →
`σ = M·y/I` at extreme fibres; circle under `Mzz` → `τ = M·r/J`;
equilibrium checks (∫σ dA = N, ∫σ·y dA = Mxx, ∫τ dA = V) to quadrature
tolerance; blend-vs-direct identity (unit-field blend == recomputing);
PyPI oracle field comparison on the I-section; matplotlib figures
smoke-tested headless (Agg).

**S4 as-shipped deviations** (2026-07-17): stress recovery on
`disconnected="sum"` sections raises `SectionAnalysisError` ("not yet
implemented — analyze the parts as separate sections") instead of the
ADR's per-part load-distribution rules; recovery is exact nodal
evaluation (`_fe.block_nodal`) rather than Gauss→node extrapolation
(strictly better — no extrapolation error); explicit sign conventions
documented and equilibrium-tested: `Mxx` tension at `+y`, `Myy` tension
at `+x` (`M = ∫σ·coord dA` both axes). G-C should treat the
disconnected-stress gap as a documented deferral. *(Deferral closed
post-acceptance: the per-part distribution rules above are implemented
in the ADR 0078 follow-up PR — see `tests/sections/
test_stress_disconnected.py` for the equilibrium + exactness gates.)*

## S5 — Bridge binding + flat-face builders (PR-5) — **gate G-B**

1. `sections/_lowering.py`: the **single** axis-mapping function
   (authoring→OpenSees; reference-`E` rules for
   geometric-only/homogeneous/composite) used by both paths.
2. `opensees/section/computed.py`: `ComputedSection(Section)` per the
   ADR (identity-hashed `analysis` field; `_emit` runs the lowering and
   emits the `Elastic` line; `dependencies() == ()`; fail-loud with the
   analyzer's `name`). Register in `_SectionNS`. `to_elastic_section()`
   on the analyzer calls the same lowering.
3. `sections/_builder.py`: `W_face`, `rect_face`, `rect_hollow_face`,
   `pipe_face`, `pipe_hollow_face`, `angle_face`, `channel_face`,
   `tee_face` — reuse each solid recipe's cross-section wire, skip the
   extrude, in-plane `translate/rotate`, auto-PG by label.

**Verify (gate, then G-B review before merge):** deck **byte-equality**
(`ComputedSection` vs hand-typed `ElasticSection` with the same
numbers, flat + inside `Lobatto`/`forceBeamColumn`); memoization (two
references, one solve — count `splu` calls); composite-without-`E`
raises at emit naming the section; W_face → analyzer → deck vs AISC
W-shape table (A, Ix, Iy, J, Z within catalog tolerance); builder faces
mesh + fragment cleanly (SRC example from the ADR end-to-end);
`tests/opensees` targeted suites green (section/tag-resolution paths);
mypy ratchet at-or-below. **Then run G-B.**

## S6 — Section inspector (PR-6)

`sections/_inspector.py`: Qt panel per the ADR (matplotlib canvas +
glyphs/contours, tabbed property tables with composite `e_ref` input,
six load spinboxes + component picker re-blending unit fields; no solve
on the UI thread — `analyze()` before window construction).
`sec.viewer(blocking=)` with the offscreen raise-guard (reuse
`ViewerWindow` pattern) and Qt-absent `ImportError` guidance.

**Verify (gate):** offscreen-guard + import-guard tests; blend-equals-
`stress()` identity through the panel's code path; QTimer-driven
screenshot smoke test (gallery pattern); notebooks documented
(`blocking=False`), no blocking call in any test.

## Close-out

- Run **G-C** (completeness critic) across the merged surface.
- Flip ADR 0078 → Accepted with PR numbers; update the README row.
- Docs: how-to recipe ("Compute section properties for a custom
  section"), apegmsh skill update (canonical skill only), CHANGELOG.
- Add the `sectionproperties` dev extra + the oracle CI job
  (skip-if-not-installed locally, installed in one CI lane).

## Risk register

| Risk | Slice | Mitigation |
|---|---|---|
| Singular-solve regularization subtly wrong (shear centre drift) | S2 | Lagrange row not pinning; G-A lens 1–2; channel shear-centre oracle |
| Shear-area integrals (ν-dependent Pilkey terms) mis-transcribed | S2 | PyPI oracle on `As` specifically; G-A refutation lens |
| Axis/sign mapping silently swapped | S5 | G-B numeric cantilever check both axes; swapped-section refutation |
| tri6/quad8 midside ordering mismatch | S1/S2 | reuse of the fem kernel (already gmsh-ordered); extrapolation tests |
| Composite accessor law bypassed internally | S1+ | G-C grep sweep; accessor tests on every dataclass |
| Interface averaging bleeding across materials | S4 | region-restricted `get(pg=)` tests on the SRC section |
| Qt surface flaking CI | S6 | offscreen guard; screenshot test local-only if CI lacks GL (existing skip pattern) |
