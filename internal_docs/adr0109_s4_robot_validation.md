# ADR 0109 S4 — `footfall_walking` against Autodesk Robot (RSA 2026, v26.0)

Same beam both sides: the how-to's 6 m simply supported bay (`docs/how-to/footfall-vibration.md`),
12 elements, `E = 200 GPa`, `A = 0.0083 m²`, `I = 3.5e-4 m⁴`, β = 3 %, Q = 747 N,
response = excitation = midspan. Robot takes the floor mass from density
(`material.RO = m·g/A`, `IgnoreDensity = False`); shear deformation switched off
(`Bars.SetShearForces(False)`) so both sides are Euler–Bernoulli. Two floors: the
how-to's 500 kg/m and a 4400 kg/m variant tuned to f₁ ≈ 5.5 Hz. Scripts, JSON dumps
and the three `.rtd` files are in `_adr0109_s4/`.

## Floor 1 — 500 kg/m, f₁ = 16.33 Hz (our regime `high`)

| | Robot | apeGmsh |
|---|---|---|
| f₁ / f₂ (Hz) | 16.326006087 / 65.300604683 | 16.326006087 / 65.300604683 (rel 4.6e-15 / 9.3e-12) |
| critical / dominant freq | 2.2 Hz — the *top of the walking band*; no harmonic reaches 16.3 Hz | f_dom = 16.326 Hz (FRF peak) |
| governing branch | transient (`RF_Overall = RF_Transient`) | `regime = "high"` (Eq 7-4…7-6) |
| acceleration | `A` = 0.016757 m/s² = **0.171 %g** (resonant, off-resonance) | `a_p = a_espa_hf` = **1.201 %g** |
| response factor | `RF_Resonant` 2.173, `RF_Transient` **11.987**, `RF_Overall` 11.987 | `limit` 1.020 %g, `ratio` **1.177** |
| velocity | `VRMS` 1.1987 mm/s, `VRMQ` 1.4849 mm/s | (no velocity metric) |
| 1st-ed flavour on our FRF | — | **not evaluable**: no harmonic i = 1…4 of 1.6–2.2 Hz reaches 16.33 Hz (max 8.8 Hz) |

## Floor 2 — 4400 kg/m, f₁ = 5.50 Hz (our regime `low`)

| | Robot | apeGmsh |
|---|---|---|
| f₁ / f₂ (Hz) | 5.503495529 / 22.012829348 | 5.503495529 / 22.012829348 (rel 4.7e-15 / 9.3e-12) |
| critical / dominant freq | 1.8320832 Hz **step** frequency → 3rd harmonic at 5.4962 Hz | f_dom = 5.5035 Hz; 1st-ed harmonic 3, f_step 1.8345 Hz |
| governing branch | resonant (`RF_Overall = RF_Resonant`) | `regime = "both"`, low branch governs |
| acceleration | `A` = 0.063286 m/s² = **0.645 %g** | `a_p = a_p_lf` = **0.573 %g** (`a_espa_hf` 0.231 %g) |
| response factor | `RF_Resonant` **8.909**, `RF_Transient` 5.399 | `limit` 0.500 %g, `ratio` **1.145** |
| velocity | `VRMS` 0.7816 mm/s, `VRMQ` 0.8796 mm/s | — |
| 1st-ed flavour on our FRF | — | h = 3, α₃ = 0.1, R = 0.5 → **a_p1 = 0.481 %g** |

`ExcitationNode` = 7 (midspan) under both `SELF` and `FULL` (excitation set `{4, 7}`),
matching our `full` run's `exc_node`. SCI P354 (`ExcitationForces = 2`) gives
`A` = 0.050257 m/s², `RF` = 10.051, and `A/RF` = 0.005000 exactly — the BS 6472 base.

## The AISC enum value: there is none

`IRobotFootfallAnalysisParams.ExcitationForces` accepts **only 1 (`I_FAEF_CONCRETE_CENTRE`)
and 2 (`I_FAEF_SCI_P354`)**. Setting 3 or 4 raises no `COMError`, `put` succeeds,
`SetAnalysisParams` returns `True` — and the value is **silently ignored**: the property
keeps whatever it held before. On a virgin case (default 1) `= 3` and `= 4` both read back
1; on a case already set to 2, `= 3`, `= 4` and `= 0` all read back 2, while an in-range
`= 1` takes immediately (`robot_enum_fresh.json`). Results confirm it — the EF = 3 and
EF = 4 cases returned byte-identical numbers to the EF = 1 case. So **Stairs SCI P354 and
AISC DG11 (2003) are UI-only on RSA 2026; the COM surface exposes two of the four options
and there is no integer that selects AISC**. Robot's AISC option could not be exercised;
everything below is measured against CCIP-016 (and SCI P354 where noted).

## Reading

**Where it agrees.** The modal basis agrees to machine precision (4.6e-15 on f₁ both
floors) — the two programs are solving the same beam. The *critical frequency* agrees:
Robot's 1.8320832 Hz is exactly its own f₁/3, our 1st-edition pass independently picked
harmonic 3 at 1.8345 Hz — both say "the third walking harmonic sits on the fundamental",
which is the only thing the 1.6–2.2 Hz band can do to a 5.5 Hz floor. The FRF-shaped part
agrees too: sweeping Robot's damping at 1.5 / 3 / 6 % fits
`A = √[(C·ρ/2ζ)² + K²]` to 0.05 %, i.e. the resonant term is exactly `1/(2ζ)`-shaped
with a ζ-independent SRSS remainder from the off-resonant harmonics. Feeding our own
`|A_jj(f₁)| = 1.2626e-3 m/s²/N` into that fit backs out Robot's 3rd-harmonic force as
**50.7 N (DLF 0.0679 of Q)** — a plain CCIP-016 walking coefficient, applied to the same
transfer function we compute.

**Where it differs, and by how much.** On the low floor Robot's 0.645 %g is **1.13×** our
0.573 %g and **1.34×** the 1st-edition flavour of our own FRF (0.481 %g). All of that gap
is the *force model*, not the structure: 50.7 N (CCIP) vs α₃·Q·R = 37.35 N (DG11 1st ed).
On the high floor the two disagree by construction. Robot has no equivalent of Eq 7-4…7-6:
its resonant branch reports 0.171 %g (a harmonic 7 Hz off resonance) and its impulsive
branch reports a *velocity*, `RF_Transient = VRMS/1e-4 = 11.99`. Converting our
`a_espa_hf = 1.201 %g` to the equivalent-sinusoid RMS velocity gives 0.812 mm/s against
Robot's 1.199 mm/s (**1.48×**) — the same physics, different impulse constant
(DG11 Eq 1-6 gives 3.43 N·s at f_step = 2.2 Hz, CCIP's `54 f^1.43/f_n^1.30` gives 4.42 N·s,
ratio 1.29) and a different RMS window. A 16 Hz floor is where our second-edition branch
earns its place: Robot's AISC option — had it been reachable — is resonant-only and would
have returned essentially nothing here.

**`R`.** Robot's CCIP-016 option carries no `R` at all — the backed-out 50.7 N is
1.36× `α₃·Q·R` and 0.68× `α₃·Q`, i.e. it is neither, it is the CCIP coefficient. `R`
(the 1st edition's 0.5 for a walker not permanently at the point of maximum response)
lives only in the AISC option, which COM cannot select. Nothing measurable about `R` can
be extracted from Robot through this interface.

**`FootstepsNumber`.** It is the resonant build-up factor and nothing else. Going 20 → 6
scaled `A` and `RF_Resonant` by **0.693264** on *both* floors, while `RF_Transient`,
`VRMS` and `VRMQ` were unchanged to the last digit. The predicted
`[1 − e^{−2πζ·6}] / [1 − e^{−2πζ·20}] = 0.693264` at ζ = 0.03 — six figures. So Robot uses
`ρ_N = 1 − exp(−2π ζ N)`, the CCIP analogue of the second edition's
`ρ = 50β + 0.25 / 12.5β + 0.625 / 1.0`. At the default N = 20 and β = 3 % that is
ρ = 0.977, against the second edition's ρ = 1.0 — a 2 % difference, negligible beside the
force-model gap. At N = 6 it drops to 0.677.

## Two incidental findings

1. **The Footfall case runs its own modal solve, and it is not the `DYNAMIC_MODAL` one.**
   Same model, same session: `Eigenvalues.Value(footfall_case, m)` returns 5.496250 and
   21.897648 Hz where the modal case returns 5.503496 and 22.012829 (−0.132 %, −0.523 %).
   The error growing as mode² is the signature of a lumped mass matrix on 12 elements.
   Harmless at this size, but it means Robot's footfall numbers are not read off the modal
   case's frequencies, and its `Frequency` output is a *step* frequency, not a mode.
2. **`VRMS` / `VRMQ` are not defined under SCI P354.** `IRobotFootfallResults.get_VRMS`
   and `get_VRMQ` fail with `COMError (-2147467259, 'Unspecified error')` — `E_FAIL`,
   `0x80004005` — on an `ExcitationForces = 2` case, while every other field reads fine.
   Any wrapper must read those two defensively.

## Reproducing

```
C:\Users\nmb\venv\opensees_env\Scripts\python.exe    _adr0109_s4\run_apegmsh.py   -> apegmsh.json
C:\Users\nmb\Documents\Github\apeRobot\.venv\Scripts\python.exe  _adr0109_s4\run_robot.py        -> robot.json,  footfall_{high,low}.rtd
                                                     ... probe_robot.py     -> robot_probe.json, footfall_probe.rtd
                                                     ... probe_enum_fresh.py -> robot_enum_fresh.json
C:\Users\nmb\venv\opensees_env\Scripts\python.exe    _adr0109_s4\analyze.py       -> every derived number above
```

Robot was launched hidden (`visible=False`) and closed by the context manager in every run;
no user session was attached to. Nothing under `src/` was touched, on either repo.

> The three `.rtd` / `.RT_` Robot files (2.2 MB) are NOT committed; `_adr0109_s4/run_robot.py` rebuilds them from the JSON inputs on any machine with a licensed Robot.
