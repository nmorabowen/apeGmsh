# Fork divergence roster

This page lists every known place where stock OpenSees and the Ladruno fork
give a different answer to the same deck. Each row names the engine that is
correct and the oracle that decides it. A row without an oracle blames
whichever engine the author happened to distrust. The "71 % soft equation
ties" row did exactly that (errata E3 below).

The roster was set by the expert panel (`internal_docs/plan_expert_panel_2026-09.md`,
"A two-sided divergence roster, one oracle per row"; issue #1192) and is
locked by `tests/opensees/unit/test_fork_divergence_roster.py`.

## How to read and edit the table

- **Correct engine** is one of `stock`, `fork`, `policy` (the engines differ by
  design and neither answer is wrong) or `pending #<issue>` (a maintainer
  ruling is owed).
- **Oracle** opens with its kind, one of `closed form:`, `cross-engine:`,
  `reference run:`, `contract:`, `n/a (policy)` or `pending #<issue> (<why>)`,
  then gives the value and its source. The lock test refuses any other
  opening and a body too thin to name an answer.
- **Pinning tests** are apeGmsh pytest node ids, or fork tests written
  `fork:<path>::<name>@<ladruno sha>`. `none` means nothing pins the row yet.
  The lock test checks that every apeGmsh node id still exists, and it holds
  the count of `none` rows at or below its baseline.
- A row is never deleted. When upstream ships a fix, set its status to
  `closed (<evidence>)` and keep the row.
- Fork evidence was read from `origin/ladruno` at `2afc5502e` (2026-10-11).

## Roster

### Stock wrong, fork right

| id | symptom | stock behaviour | fork behaviour | correct engine | oracle | evidence | pinning tests | status |
|---|---|---|---|---|---|---|---|---|
| D1 | `TenNodeTetrahedron` stiffness, mass, body force and reactions read 1/6 of the true value | `shp3d` sets `xsj = Jdet`, where `Jdet` is already the element volume, so the 1/6 is applied twice | `xsj = 6.0*Jdet`; the block matches the closed form | fork | closed form: uniform-strain block, `nu = 0`, `K = E·A/H` exactly (`test_tet10_volume_live.py` docstring) | fork `ee942546c`, PR #520 (merge `36c3fa357`); upstream fix OpenSees/OpenSees#1833 (open, 2026-10-10: drops `/6.0` from `Jdet`, equivalent to the fork's fix); apeGmsh refusal `emitter/live.py::_TET10_STOCK_DEFECT` | `tests/opensees/live/test_tet10_volume_live.py::test_tet4_block_matches_closed_form`<br>`tests/opensees/live/test_tet10_volume_live.py::test_tet10_block_is_one_sixth_on_stock`<br>`tests/opensees/live/test_tet10_volume_live.py::test_tet10_block_matches_closed_form_on_a_fixed_fork`<br>`tests/opensees/live/test_tet10_volume_live.py::test_live_emitter_refuses_tet10_on_stock`<br>`tests/opensees/unit/test_live_stock_gates.py::test_tet10_refused_on_stock` | open: stock wrong through openseespy 3.8.0; upstream fix pending in OpenSees/OpenSees#1833, and stock stays wrong until a released openseespy ships it; the live emitter refuses tet10 on stock. The fork's fix (`xsj = 6·Jdet`, `ee942546c`) and #1833's (`Jdet` without `/6`, derivatives `1/Jdet`) differ in form, so a fork↔upstream sync may conflict in `TenNodeTetrahedron::shp3d` |
| D2 | UW `DruckerPrager` past the tension cutoff keeps an unreturned stress (`I1 = 8.08` against a cutoff of `0.449` on the ADR 0108 driver) | the cutoff residual row is never assembled, so the tensile point is not returned | the point returns to the cutoff or the corner (branch 2 or 3) | fork | closed form: on a hydrostatic path the return lands at `I1 = sqrt(2/3)·σy/ρ` | fork ADR-95, PR #803 (merge `61b3efa04`); apeGmsh ADR 0108 | `tests/opensees/integration_ladruno/test_ladruno_dp_branch_live.py::test_a_driven_tensile_point_is_returned_to_the_cutoff`<br>`tests/opensees/integration_ladruno/test_ladruno_dp_branch_live.py::test_pre_fix_engine_is_self_consistent` | open: no apeGmsh test asserts the stock defect; the pins run on the fork |
| D3 | `ASDPlasticMaterial3D` + `DruckerPrager_YF` misclassifies states over the apex | the apex region is wrong (a wall at `etabar = 0`, then under dilatant flow) | the exact apex criterion; associated and dilatant wedges return to the cone | fork | closed form: the apex criterion `p ≥ p_apex` at ψ = 0; cross-engine: the linear control leg against UW `DruckerPrager` | apeGmsh ADR 0105 Amendments 2–4; fork PRs #815 (`bbf657d49`), #832 (`67474aeb7`), #836 (`2db3f0889`) | fork:tests/test_adr94f_asd_apex_fallback.py::test_94f_zero_dilatancy_over_apex_wedge_is_accepted@2afc5502e<br>fork:tests/test_f8_asd_dp_associated_apex.py::test_f8_associated_wedge_returns_to_the_cone_not_the_apex@2afc5502e | open: pinned on the fork side only; no response token, so apeGmsh cannot read the branch back |
| D4 | a singular factorization with the zero pivot on equation 1 is reported as success | `BandGenLinLapackSolver` (also FullGen, BandSPD) returns `-info+1`, which is `0` when `info == 1`; Newton never learns and the push runs to the end | returns `-info`; the failure propagates and the curve truncates | fork | contract: LAPACK `info > 0` means `U(info,info)` is exactly zero, so the solve failed; closed form for the values that do come back: `M = min(EI0·κ, Mp)` | fork `804c4bda3`; apeGmsh `c3ace110`, #1021; upstream still `return -info+1;` | `tests/sections/test_mc_b7.py::test_partial_curve_when_the_section_goes_singular`<br>`tests/sections/test_mc_b7.py::test_the_singular_section_reports_the_closed_form_on_any_build` | open: on stock the values are exact (symmetry holds the residual at zero); only the report differs |
| D5 | `algorithm Newton -hall` (also ModifiedNewton `-hall`, NewtonHallM, ExpressNewton) under `BackwardEuler`, `Newmark1` or `Collocation` | `formEleTangent` has no `HALL_TANGENT` branch: BackwardEuler and Newmark1 assemble an all-zero element tangent, and Collocation assembles `c2·C + c3·M` with no stiffness | the branch is transposed from `Newmark::formEleTangent` | fork | reference run: the Hall factors sum to 1, so with `Kt ≈ Ki` the `-hall` tangent is `K` and the run must match a run without `-hall`. For `BackwardEuler` the reference is its own non-hall run (the fork measured `uz` equal to it exactly); for `Newmark1` and `Collocation` it is plain `Newmark` (3.5e-16 and 4.7e-13 relative), per `acee81d33` | fork ADR-77 C0-2, `acee81d33`; #1192 R2 table | none | open: no test in either repo. Not reachable from apeGmsh's typed API, which has no `BackwardEuler`, `Newmark1` or `Collocation` integrator |

### Fork differs by policy

| id | symptom | stock behaviour | fork behaviour | correct engine | oracle | evidence | pinning tests | status |
|---|---|---|---|---|---|---|---|---|
| D6 | a fork-only integrator type (`ExplicitBathe`, `LadrunoLoadControl`, …) run on a build that lacks it | `integrator <unknown type>` is accepted without error and the previous integrator keeps stepping | the integrator runs (a fork build that predates it behaves like stock) | policy | n/a (policy). Stock: silent accept of an unknown type. Fork: the type exists. apeGmsh refuses fork-only integrators on stock before the call | `emitter/live.py::_FORK_ONLY_INTEGRATORS` and `_LADRUNO_LOAD_CONTROL_REQUIRED`; the stock false positive is recorded in `test_explicit_integrators_live.py::_fork_has` | `tests/opensees/unit/test_ladruno_load_control_gate.py::test_stock_build_refused`<br>`tests/opensees/unit/test_ladruno_load_control_gate.py::test_fork_without_the_integrator_refused_before_the_call` | open: gated in apeGmsh; deck emission (Tcl/py) still runs silently on stock |
| D7 | an unknown option token on a fork class | stock does not know the classes | three policies. Fatal (`LadrunoOptSpec.h`, WP-167): the LadrunoRCConcrete family and the cohesive hinges. Warn and continue: the explicit integrators, `LadrunoArcLength`, `BezierTet10`/`BezierTri6`, `LadrunoBrick`/`LadrunoBrick20`. Silent: `LadrunoDispBeamColumn2d/3d` (the option ladder has no `else`), `EnergyBalanceRecorder`, `LadrunoMonitorRecorder` ("unknown options are ignored (forward-compatible)") | policy | n/a (policy): which policy each class applies, as listed | fork PR #923 (merge `8572e8118`), `SRC/material/LadrunoOptSpec.h` | fork:tests/test_ladrunoRCConcrete_failclosed.py::test_undeclared_token_fails_and_names_it@2afc5502e<br>fork:ci/test_quirk_unknown_token.py::test_legacy_list_is_a_shrink_only_ratchet@2afc5502e | open: the warn and silent groups remain |
| D8 | `-crackedNu` / `-betaC` on `LadrunoRCConcrete` / `LadrunoRCFiniteStrain` | stock does not know the classes | `ladruno` refuses both flags (WP-167, #923); before #923 it ignored them silently. They exist only on fork PR #877 (open); #873 closed unmerged | policy | n/a (policy). Partial physical oracle: the PV20 panel, where `cracked_nu = 0` reproduces the MCFT hand solution | fork #873 (closed), #877 (open), #923; apeGmsh `_rc_c2_flags.py`, errata E4 | `tests/opensees/unit/test_ladruno_rc_c2_gate.py::test_no_ladruno_build_carries_the_flags_yet`<br>`tests/opensees/unit/test_ladruno_rc_c2_gate.py::test_live_refuses_c2_flags_before_the_call`<br>`tests/opensees/integration_ladruno/test_ladruno_rc_cracked_nu_live.py::test_live_route_refuses_c2_flags_while_no_build_carries_them` | open until fork #877 merges; then set `LADRUNO_RC_C2_MIN_BUILD` |

### Fork against fork, across compilers

| id | symptom | stock behaviour | fork behaviour | correct engine | oracle | evidence | pinning tests | status |
|---|---|---|---|---|---|---|---|---|
| D9 | `LadrunoSANISAND` pinned to `ManzariDafalias` is not bit-identical | n/a (stock has no `LadrunoSANISAND`) | Linux pin `ladruno-v20261005` (`011b02586`, CI run 37516539499): `-0.0024933580194803247` (LadrunoSANISAND) against `-0.002493359038926103` (ManzariDafalias). Windows build `79e062367`, measured 2026-10-11: `-0.0024933580244094733` against `-0.002493358492888701`. The two builds are at different commits: `79e062367` is an ancestor of `011b02586`, and `ManzariDafalias.cpp` changed in between (WP-158, WP-160 and others). The two LadrunoSANISAND values agree to 4.9e-12 absolute (2e-9 relative); the gap sits in the ManzariDafalias reference value | pending #1523 | pending #1523 (divergence under diagnosis) | #1523 | `tests/opensees/integration_ladruno/test_ladruno_sanisand_live.py::test_i1_pinned_ladruno_sanisand_reproduces_manzari_bit_identically` | open: `xfail(strict=True)` on Linux stays until #1523 settles the oracle; the test also fails, unmarked, on the Windows build `79e062367` |

## Withdrawn claims

- **E3, "equation ties run 71 % too soft on stock".** The test stacked a
  `stdBrick` on stock `TenNodeTetrahedron`. A 6×-soft block in series gives
  2/7, which is 71.43 % soft. That is row D1, not a tie defect
  (`internal_docs/plan_architectural_strains_2026-09.md`, errata E3).

## Follow-ups

- D5 has no pin. A pin needs either a typed `BackwardEuler`/`Newmark1`/
  `Collocation` integrator in apeGmsh or a fork-side test.
- D2 and D3 have no test that asserts the stock defect itself.
- The rule for fork-option effect tests is in
  [testing.md](testing.md#fork-option-effect-tests).
