# ADR 0109 program — footfall vibration, orchestration handoff

Companion to `src/apeGmsh/opensees/architecture/decisions/0109-footfall-vibration-frf-method.md`
(read it first; this file adds only what an agent needs to run one slice
without the conversation that produced the ADR).

## Shape of the program

Two PRs, four implementation slices, one review per PR, one validation
slice after merge. S0 and S1 touch disjoint files and run in parallel;
everything else is sequential.

```
PR-A:  S0 kernel  ‖  S1 FRF matrix   →  review R-A  →  merge
PR-B:  S2 driver  →  S3 map + how-to + skill  →  review R-B  →  merge
after: S4 Robot validation (needs a Robot licence; human in the loop)
```

| Slice | Agent | Model | Effort | Why this model |
|---|---|---|---|---|
| S0 kernel | general-purpose | sonnet | medium | every equation and every oracle number is written down; the work is transcription plus tests |
| S1 FRF matrix + scale assertion | general-purpose | opus | high | the D2 subtlety (no exported generalised mass, refusal path), live-emitter internals, fork oracle to 1e-6 |
| R-A review | general-purpose, read-only | opus | high | adversarial, diff-only, hunts unit and normalisation errors |
| S2 driver + result | general-purpose | opus | medium | `apesees.py` conventions, selection resolution, API index gate |
| S3 map + docs + skill | general-purpose | sonnet | medium | mechanical; the docs gate and the skill sync are the risks |
| R-B review | general-purpose, read-only | opus | high | as R-A |
| S4 Robot validation | opus + owner | medium | a script through apeRobot and a written comparison; nothing to merge in `src/` |
| Orchestrator | this session (Fable) | — | low | dispatch, read reports, run the PR mechanics; never re-derive a slice |

The Agent tool takes `model=`; effort is not a parameter, so "effort" is
enforced by the prompt: a closed file list, a closed verification command,
and an explicit stop condition. Every prompt below is self-contained.

## Token rules (apply to every prompt)

- Read the ADR and the files named in the slice; nothing else without a
  stated reason. Never read `apesees.py` whole — `grep -n` then `sed -n`
  ranges.
- Run only the named test files with the venv interpreter
  `C:\Users\nmb\venv\opensees_env\Scripts\python.exe -m pytest <file> -q`.
  Not the suite. The gates that pytest does not cover: `ruff check` and
  `mypy` on the touched files (baseline is zero on both).
- Report back in under 300 words: files touched, test command and its
  last line, anything left undone. No transcript, no code in the report.
- Do not commit. The orchestrator commits with `git commit -F <file>` per
  PR (never a multi-line `-m` on Windows PowerShell).

## S0 — kernel (`src/apeGmsh/opensees/analysis/footfall.py`)

Pure numpy, no OpenSees import. Functions (D1): `dynamic_coefficient(f)`,
`resonant_buildup_factor(beta)`, `harmonic_for_dominant(f)`,
`effective_impulse(f_step, f_n, body_weight)`,
`walking_low_frequency(frf_max, f_dom, beta, body_weight)`,
`walking_high_frequency(f_n, phi_i, phi_j, f_dom, beta, body_weight, dt=0.005)`
returning `(a_espa, a_peak, t, a_t)`, `tolerance_limit(occupancy, f, kind="curve")`,
`dominant_frequency(freq, frf_mag, f_max)`. Accelerations are in the units
of `frf_max × force`; the kernel never divides by g — the driver does.

Oracle test `tests/opensees/unit/test_footfall_kernel.py`, Example 7.1,
Q = 168 lb, β = 0.025, g = 386 in/s², `%g = 100·a/g`:

- `dynamic_coefficient(3.49) ≈ 0.069`, `resonant_buildup_factor(0.025) = 0.9375`;
  `walking_low_frequency(0.0344 %g/lb, 3.49, 0.025, 168) = 0.374 %g` (rtol 1 %).
- `harmonic_for_dominant(12.6) = 6`, `f_step = 2.1`;
  `effective_impulse(2.1, 12.6, 168) = 1.01 lb·s` (rtol 1 %).
- Mode 22 alone: `a_p = 2π·12.6·(−3.15)²·1.01/1000 lb/kip / 386 = 0.206 %g`.
- All 38 modes, `phi_i = phi_j = φ` below, `dt = 0.005`, window `1/2.1 s`:
  peak `0.865 %g` (rtol 2 %), `a_espa = 0.314 %g` (rtol 2 %).
- `tolerance_limit("office", 8.85) ≈ 0.00553` (0.553 %g), `(… 9.35) ≈ 0.00584`,
  `(… 6.0) = 0.005`; returned as a FRACTION of g; `kind="table"` is flat. Occupancies:
  office/residence/church/school 0.5 %g; shopping/dining/indoor bridge 1.5 %g;
  outdoor bridge 5 %g.
- Edge pins: ρ at β = 0.01 and 0.03 boundaries; Table 7-1 boundaries
  (11, 13.2, 15.4, 17.6 Hz); `dominant_frequency` ignores points above `f_max`.

Table 7-2 (mode, f_n Hz, φ at backspan, (in./kip·s²)^½):

```
1 3.49 0.449   2 3.80 0      3 4.89 -0.513  4 6.88 0      5 7.05 1.67
6 7.31 -0.553  7 7.72 0      8 7.88 0       9 8.07 -1.47  10 8.85 1.83
11 8.98 0      12 9.35 -2.57 13 9.42 0.565  14 9.63 0     15 10.1 -1.80
16 10.4 -0.649 17 10.7 0     18 10.9 -1.35  19 11.2 0     20 12.0 1.93
21 12.6 0      22 12.6 -3.15 23 13.1 0      24 13.3 0.286 25 13.7 0
26 14.5 0.832  27 14.6 0     28 14.7 0.044  29 15.5 1.00  30 16.0 0
31 16.9 -1.47  32 17.3 0     33 17.5 1.18   34 17.6 1.08  35 17.7 0
36 18.6 1.93   37 19.2 0     38 19.8 0.810
```

Stop when the test file is green and ruff/mypy are clean on the two files.

## S1 — FRF matrix (`src/apeGmsh/opensees/analysis/footfall_frf.py`)

Read: ADR D2/D4/D6; `analysis/modal.py` (`ModalPropertiesResult`,
`_damping_channel_args`, `_node_tag`); `emitter/live.py` lines 936–1030;
`tests/opensees/live/test_modal_sweeps_live.py` (fixture + gate);
`tests/opensees/live/test_modal_properties_live.py` (how `-return` keys
are read).

Deliver `frf_matrix(props: ModalPropertiesResult, *, exc_nodes, resp_nodes,
dof, freq, damp=None, modal_damp=None, rayleigh=None) -> FRFMatrix` with
`FRFMatrix.accel(i, j)` (complex, `−Ω²H`), `.magnitude`, and
`grid_for(props, f_min, f_max, n_extra=30, cluster=0.05)` (D4). The
eigenvector scale assertion exactly as D2: for each mode, over components
`MX MY MZ RMX RMY RMZ` present in `props.properties`, `partiMass_C /
partiFactor_C²` where `|partiFactor_C| > 1e-8` must be `1 ± 1e-6`; zero
checkable modes → refuse; any failing → refuse; record
`normalization = "asserted(k of p)"`. Refuse `unorm=True` results (detect
from a flag the driver passes; do not guess).

Tests `tests/opensees/live/test_footfall_frf_live.py`, same gates as the
sweeps file:
1. tip-mass cantilever, `exc = resp = [2]`, `dof = 1`: `|accel|` equals the
   SDOF closed form `Ω²/|ω² − Ω² + 2iξωΩ|` with the alphaM-only ξ (rtol 2 %).
2. `has_fork` only: the same pair against `apeSees.frequency_response(load=<unit
   load at node 2 dof 1>, node=2, dof=1, resp="accel", num_modes=2, …)` on an
   identical model — rtol 1e-6. If the fork's pattern needs building on the
   bridge before the sweep, do it through the public `ops.pattern.Plain`
   surface; do not add emitter methods.
3. mutation: scale one φ column by 2 inside the assertion's input and assert
   the refusal message names the mode.

Stop when green (2 may skip on a stock build — say so in the report).

## S2 — driver (`apeSees.footfall_walking`, `FootfallResult`)

Read: ADR D3/D5/D6/D8; `apesees.py` ranges for `modal_properties`
(≈ 9034–9116), `_modal_prereqs_and_guards`, `_run_modal_sweep`, and how
`node=` is resolved there; `analysis/footfall.py` and `footfall_frf.py`
from S0/S1; one existing result dataclass in `analysis/modal.py` for style.

Signature: `footfall_walking(*, num_modes, body_weight, g, response_nodes,
excitation="self", excitation_nodes=None, dof=3, occupancy="office",
limit="curve", damp=None, modal_damp=None, rayleigh=None, f_max=20.0,
n_extra=30, dt=0.005, solver="-genBandArpack") -> FootfallResult`.
Node sets accept what `node=` accepts on the sweep drivers plus a
sequence of them; `excitation_nodes` defaults to `response_nodes`.
Per response node: `f_dom`, `frf_max`, `a_p_lf`, `a_espa_hf`, `a_p`,
`regime`, `exc_node`, `limit`, `ratio` (D5), plus `frf(j, i=None)` and the
per-mode rows. Warn (not refuse) when the highest extracted mode is below
`f_max`, naming `eigen_feast`.

Tests `tests/opensees/live/test_footfall_driver_live.py`: a two-bay
elastic shell strip (or two tip-mass cantilevers on one deck) where
`self` and `full` agree on the diagonal, `exc_node` is the loaded node,
`regime` flips when mass is scaled to push `f_dom` across 9 Hz, and
`damp=`/`modal_damp=` with equal ratios agree. Unit tests for argument
validation go next to `test_apesees_modal_validation.py`.

New public method ⇒ rebuild the committed API index (find the script with
`grep -rln "_api_index" scripts/`); two CI lanes fail otherwise.

## S3 — map, how-to, skill

Read: ADR D5; `results/writers/_native.py` (`NativeWriter`, `write_nodes`,
and how a stage/time vector is opened); `results/Results.py::from_fem`;
`docs/how-to/point-load.md` (voice and length); `skills/apegmsh/` (sync
via `scripts/sync_skill.py`, never `~/.claude`).

Deliver `FootfallResult.to_results(fem, path)` (one frame, components
`footfall_ap`, `footfall_ratio`, `footfall_fdom`, `NaN` off the response
set); `docs/how-to/footfall-vibration.md` (one worked floor, in the
course voice, no bulk reference); a CHANGELOG `ADDED` entry with **no**
`](src/…)` links and newest-first; a short skill note pointing at the
how-to. Test: write + `Results.from_fem(..., kind="native")` reads the
three components back with the right shape; the docs gate
(`mkdocs build --strict`) is green.

## R-A / R-B — review prompts

Read-only. Input: `git diff main...HEAD -- <slice paths>` and the ADR.
Hunt, in order: unit errors (g, %g, lb vs kip, Hz vs rad/s), the D2
refusal path (can a wrongly scaled basis get through?), the dominant
frequency slice (< 9 Hz vs full band), off-by-one in the ESPA window, and
any public surface not in the ADR. Return at most five findings, each
with a reproducer; "no finding" is a valid answer. The orchestrator fixes
or rebuts each finding before the PR.

## S4 — Robot validation (after PR-B merges)

Through `apeRobot` on a licensed machine: the same floor as the S3 how-to
built in Robot, a `DYNAMIC_FOOTFALL` case with the AISC option and self
excitation over the same response nodes, `IRobotFootfallResults.Frequency`
and `.A` read back; the comparison uses our FRF with the 1st-edition
coefficients (α = 0.5/0.2/0.1/0.05, Q = 157 lb, R = 0.5) to isolate what
Robot does with `R` and `FootstepsNumber`. Output: a report in
`internal_docs/`, and the ADR's "What was measured" paragraph. No gate.
