# ADR 0135 — The material tester: apeGmshViewer runs the fork's `OpenSees.exe` on one declared material

**Status:** Proposed (2026-11-18). Merging the PR that flips this Status to
Accepted is the acceptance. The pointer in ADR 0112 lands with it (ADR 0121 D5).

**Owner:** nmora

**Rulings:** the owner's answers in the design session of 2026-11-18. The engine
is `OpenSees.exe` run by the app. The tester is an in-app window. 3D nD gets a
new fork command. The first materials are Steel02, Concrete02,
`LadrunoUniaxialJ2`, `LadrunoRebarBuckling`, `LadrunoBondSlip` and
`LadrunoCohesiveHinge`. The first version offers strain protocols, parameter
overrides and stress control. D7 and D8 are the session's proposed defaults and
are open to the owner's correction in review.

**Amends** [ADR 0112](0112-files-are-the-model-and-a-read-only-app.md) D4: the
app may start one more kind of process, the solver, under D1 to D3 here. The
wall against apeGmsh and Python stands unchanged.

**Leaves standing:** ADR 0112 D2, D3, D6 and D7, and ADR 0095 INV-2. The app
writes nothing to the script, to OCC or to `model.h5`.

## Context

The owner wants to pick a declared material in the viewer, set a few things and
see its response. A response has to be computed by something, and the app holds
no constitutive code.

ADR 0112 D4 says the app "reads the artifacts and the schema documents only".
Its lint (`scripts/check_app_wall.py`, W3) bans a subprocess into Python or
apeGmsh, and `src/main/source.ts` records the editor as the one process the app
starts besides Electron. W3 would not notice a spawn of OpenSees, but running a
solver is a change of policy, so it is decided here.

Probes on 2026-11-18, against the installed fork at
`C:\Program Files\Ladruno\OpenSees\bin` (`ladrunoBuild` reports `8b5e0593`,
which is not the `FORK_PIN` sha `3e738308`). The parameter values were chosen
for the probe, not taken from a model.

| Probe | Result |
|---|---|
| `model testUniaxial` with `uniaxialTest`, `strainUniaxialTest`, `stressUniaxialTest`, `tangUniaxialTest` | Runs Steel02, Concrete02 and all four Ladruno uniaxial materials, including `LadrunoRebarBuckling` wrapping Steel02 and `LadrunoUniaxialJ2` |
| Agreement | 10,000 cyclic Steel02 steps give the same peak stress and final tangent as openseespy |
| Speed | 124–177 ms per 2,000-step run, process start included; 248 ms for 20,000 steps printing every step |
| Output | Everything `puts` writes arrives on **stderr**; stdout is empty |
| Failure | A bad or unknown material stops the script, and the process still **exits 0** |
| Tcl reach | `exec`, `open`, `socket` and `source` are present; after `rename` they are gone and the tester still works |
| Commit | `model testUniaxial N` commits on every Nth `strainUniaxialTest`; there is no separate commit command |
| Stress control | Plain Newton diverges on the first reversal of `LadrunoUniaxialJ2`. A bracketed search with exactly N = 60 trials per step reaches every target on the J2 and Steel02 paths |
| Stress control, softening | The same prototype correctly reports `-35` as unreachable on Concrete02 (peak `-30`), but it also misses the reachable `-29`, because its doubling step jumps over the peak |
| 3D nD | The fork has `testUniaxial`, `testPlaneStress` and `sectionTest`; no 3D tester was found |

What the file offers: `/opensees/materials` holds each material's `type`, `tag`
and positional arguments (`params`, `params_str`), enough to rebuild its exact
command without Python. `/opensees/decl_params` holds fields by name and `$decl`
references to other declarations. `params_names` names the argument slots only
when the arguments equal the fields, and
`tests/opensees/contract/params_names_ledger.txt` lists `LadrunoUniaxialJ2` and
`LadrunoRebarBuckling` as unnamed and `LadrunoCohesiveHinge` as unchecked.

## Decision

### D1 — The app runs the fork's executable, and nothing else computes

The Electron main process runs `OpenSees.exe` once per test, on a generated Tcl
script. There is no constitutive code in TypeScript, no Python and no apeGmsh
import. The tester requires the fork: a run starts by calling `ladrunoBuild`,
and an executable without it is refused by name. The build id is shown in the
window and kept with each run.

All spawning lives in one module, `src/main/materialTest.ts`. A new rule in
`check_app_wall.py` fails a process spawn of any other kind outside that module
and `src/main/source.ts`, so this exception cannot widen unnoticed.

### D2 — The user chooses the executable; the model never does

The path comes from, in order: the app's own setting, `$APEGMSH_OPENSEES`, the
default install path. Nothing read from an artifact can influence it. The child
environment matches what `opensees_ladruno.bat` sets (`PATH` entries and
`TCL_LIBRARY`, relative to the executable). With no usable executable, the
window says so and offers the setting; the rest of the app is unaffected.

### D3 — A model file is untrusted input to an interpreter

A `model.h5` can come from anyone, and its strings would reach Tcl, which can
start programs. The same threat drove VU13 and VU14 for go-to-source.

1. **Only validated tokens reach the script.** The material type and each flag
   must match `-?[A-Za-z][A-Za-z0-9_]*`. Each number is written by the app from
   a finite float. Any other token refuses the run and names the slot. No
   quoting or escaping is attempted.
2. **Only `uniaxialMaterial` lines come from the file** (`nDMaterial` under D9).
   The driver around them is fixed text in the app.
3. **The script first removes** `exec`, `open`, `socket`, `source`, `cd`,
   `file`, `glob`, `load`, `package` and `interp`. This is defence in depth
   behind rule 1.
4. **The process is contained:** no shell, a fresh temporary working directory
   deleted afterwards, a 10 s timeout, caps on steps and output, one run per
   window, and a superseded run is killed.
5. **Success is proved, not assumed.** Exit code 0 means nothing here. Result
   lines carry a fixed prefix on stderr, and a run counts only if the closing
   sentinel arrives and the row count matches. Otherwise the window shows the
   solver's message and no curve.

### D4 — A test rebuilds the material from the file, with its dependencies

The script declares the material and, first, every material it references,
following `$decl` values in `decl_params`, with the file's tags unchanged so
reference slots stay valid. A reference that cannot be resolved refuses the run.
A file with no `decl_params` can test only a material with no references.

### D5 — An in-app window, inside the one-way data flow

A new `WindowKind`, `materialTest`, joins the registry, opened by a **Test**
button on uniaxial rows of the Materials & sections window. It follows ADR 0112
D6: the panel dispatches events, an effect calls main over IPC, and results
return as store events. Runs live in the store as session state, separate from
model state. The owner approves the screen from a screenshot (ADR 0112 D7).

### D6 — What can be set

- **Strain protocols:** monotonic, cyclic from an amplitude list with a number
  of steps per ramp, and pasted values.
- **Parameter overrides, by argument slot.** The window lists the numeric slots
  of the material's command. A slot is labelled by `params_names` when the file
  has it, otherwise by its preceding flag and position (`-iso voce · 2`). Flags
  and slots holding a dependency's tag are locked. This needs no schema change
  and covers the unnamed Ladruno materials, at the cost of plainer labels there.
  Python-side validation does not run, so the fork's parser is the judge and its
  message is shown.
- **Stress control:** target stresses reached by a bracketed search in the fixed
  driver, using `model testUniaxial N` and exactly N trials per step so that the
  Nth commits. A step that fails is marked and the run stops there, because a
  target beyond a peak has no solution. The search must not step over a peak:
  reaching `-29` on the Concrete02 probe is an acceptance test.
- **Axes** default to strain and stress. A small table in the app relabels
  `LadrunoBondSlip` (slip, bond stress) and `LadrunoCohesiveHinge` (rotation
  jump, moment). Values are in the model's units.

### D7 — What is shown and kept

The window plots the response curve, the input history, the tangent and the
dissipated energy (trapezoidal integral, computed in the app and labelled so).
Runs overlay for comparison, and the declared values are always available as the
baseline. A run can be exported to CSV through a save dialog. Runs are not saved
with the model; storing them in `<stem>.workspace.h5` is a later decision under
ADR 0132.

### D8 — Overrides never return to the model

Overrides exist only in the test. The window can copy the changed values as text
and offers go-to-source for the material (ADR 0112 D3). The script stays the
source of truth.

### D9 — nD goes through a new fork command

A fork PR adds a tester that holds a copy of one `NDMaterial` for a requested
material type (3D, plane stress and the others the material supports), sets a
full strain vector, returns stress and tangent, and commits **explicitly**. The
exact command is specified in that PR. apeGmsh takes it through a `FORK_PIN`
bump, and D3 extends to `nDMaterial` lines. Which stress components are held at
zero or at a confinement value is designed in that slice.

## Slices

| Slice | Delivers | Done when |
|---|---|---|
| **S1 (first)** | `materialTest.ts` runner, script generator with D3 and D4, the wall rule, the window with strain protocols and the response plot | All six first materials plot from a real `model.h5`; hostile-token tests pass; the owner approves the screen |
| S2 | Overrides by slot, overlays, the other plots, CSV | An override is compared with the declared curve |
| S3 | Stress control | The D6 acceptance cases pass, including softening materials |
| S4 | The fork's nD tester and the `FORK_PIN` bump | Fork tests pass and the pin is merged |
| S5 | nD in the window, plane stress and 3D | First nD materials chosen by the owner plot |

CI has no fork executable in the app's lane. The generator and the output parser
are pure and tested there, and the runner is tested against a stand-in
executable. A run against the real fork belongs in the nightly `live-fork` lane.

## Consequences

- The tester shows the code that analyses run, so it cannot drift from it. It
  shows the *installed* build, which may differ from `FORK_PIN`; the window
  states the build for that reason.
- The app gains an optional dependency on an installed fork, and a viewer-only
  machine simply lacks this one window.
- The app now hands model-derived values to an interpreter, so D3 is a permanent
  obligation for every new material family added to the tester.

## Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Python companion writing a result file the viewer watches | No live feedback, two tools to keep in step, and a new file format |
| The fork compiled to WebAssembly | Keeps the app process-free, but adds a heavy build with unproven coverage of the fork's materials. It can replace the runner later without changing the window |
| Constitutive laws in TypeScript | A second copy of the physics that would drift |
| Loading `opensees.pyd` | Needs Python, which ADR 0112 D2 and D4 exclude |
| Editing parameters by name only | Would leave out the unnamed Ladruno materials the owner chose first, or wait for a schema change |

## Open questions

- Better names for unnamed argument slots, written by the bridge into the file.
  This is a schema change with its own decision, and D6 does not wait for it.
- The nD stress-control and confinement design (S4, S5).
