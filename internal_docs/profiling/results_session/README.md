# Results-session window profiling

Profiles `results.session().show()` — the ADR 0098 window: pane docks, the
reconciler pump, and the gestures that run **while the window is open**.
Sibling of [`../box_wave/`](../box_wave/), which covers `model.viewer` /
`mesh.viewer` and is about scene build; this one is about interaction.

> **Import note:** the venv has a PEP-660 editable install of apeGmsh
> pinned to the *main* checkout. The script prepends this worktree's `src`
> to `sys.path` and hard-aborts if that shim fails. The first
> `[profile] apeGmsh loaded from:` line must point inside this worktree —
> if it doesn't, the numbers are meaningless.

## Run

```bash
python internal_docs/profiling/results_session/profile_session.py replay
```

Two stages:

- **`replay`** (default) — drives a deterministic gesture workload and
  prints per-gesture ms (mean / p50 / p95 / max). Comparable across
  commits, which "I wiggled the mouse for ten seconds" is not.
- **`interact`** — opens the window and profiles whatever *you* do until
  you close it (the box_wave idiom), for chasing something you can feel
  but cannot script.

Knobs: `--panes N` (default 3), `--repeat K` (default 20), `--gestures
scrub orbit slot scope pane`, `--results DIR` (any folder with `model.h5`
+ `run.h5`; defaults to the ssi_frame_wall bench), `--no-profile`
(timings only — cProfile inflates wall clock ~1.4x, so trust `--no-profile`
for absolute cost and the `.prof` for attribution).

The five gestures map to what a user actually does: `scrub` steps the
session instant (playback / scrubber drag), `orbit` rotates the active
pane's camera, `slot` fills and clears a contour, `scope` flips a physical
group, `pane` adds and closes a view.

## Output

- `replay.prof` / `interact.prof` — pstats dumps (gitignored).
- Per-gesture table + top-30 cumulative / top-20 tottime on stdout.
- `profile_scope.ini` — the throwaway QSettings scope. Profiling must not
  read or write your real saved arrangement: a restored layout changes
  what is being measured, and the run would then save its own result back
  over yours.

```bash
snakeviz internal_docs/profiling/results_session/replay.prof   # Icicle view
```

## Baseline — ssi_frame_wall / c001_baseline_small, 2026-08-19

`--repeat 12 --no-profile`, 3 panes (contour+deform+clip / line diagram /
history plot), on the Amendment 3 dock panes:

| gesture | mean ms | p50 | p95 |
|---|---:|---:|---:|
| scrub  | 405 | 401 | 425 |
| pane   | 336 | 332 | 359 |
| slot   | 166 | 118 | 285 |
| scope  |  57 |  27 | 173 |
| orbit  |  18 |  16 |  37 |

Boot, for reference: `import apeGmsh` 0.93 s, `SessionWindow(...)` ~1.6 s
(0.84 s first-touch `pyvistaqt`/`pyvista.plotting`, 0.78 s the
`QtInteractor` creations, ~0.5 s matplotlib font metrics). Cold imports —
it does not grow with the model.

## Confirmed by the profile — the scrub hotspot

`scrub` is the one that matters: the scrubber defaults to 30 fps and one
step costs **~285 ms** (`--no-profile`, 1 pane), i.e. **~3.5 fps**.

Pane count is *not* the driver — one pane is already 284 ms:

| panes | scrub mean ms |
|---|---:|
| 1 (contour+deform+clip) | 284 |
| 2 (+ line diagram)      | 399 |
| 3 (+ history plot)      | 408 |

The plot pane is nearly free (+9 ms), which is the cheap-cursor path in
`PlotReconciler` working as designed. The cost is one mesh pane's realize.
Attribution over 10 steps, one pane:

| | cum s | ncalls | per step |
|---|---:|---:|---|
| `_reconciler._flush` | 3.86 | 20 | the pump |
| ↳ `realize_pane` / `_realize_mesh` | 3.27 | 10 | once per step, correct |
| ↳↳ **`extrapolate_gauss_slab_to_nodes`** | **2.43** | **20** | **twice per step** |
| ↳↳↳ `extrapolate_gauss_slab_per_element` | 1.61 | 20 | 164 020 `np.broadcast_to` |
| `_contour.attach` | 1.60 | 10 | full re-attach every step |
| `_contour.update_to_step` | 1.25 | 10 | |
| `_reconciler._teardown` | 0.47 | 10 | |
| VTK `Render` | 0.74 | 80 | 7 ms/step — GL is *not* the problem |

### Trajectory (ADR 0098 Amendment 4)

`--repeat 10 --no-profile`, ms/step:

| after | 1 pane, Gauss | 1 pane, nodal | 3 panes, Gauss |
|---|---:|---:|---:|
| baseline | 294 | 127 | 405 |
| A4 step 1 — the unscoped-accumulator bug | 247 | — | — |
| A4 step 2 — one store per `Results` | 239 | 126 | 353 |
| **A4 step 4 — the cursor fast path** | **18** | **18** | **74** |

**16x on one pane, 5.5x on three, and under the 33 ms / 30 fps budget.**

Steps 1 and 2 bought less than predicted (294 → 239, not → 120) and the
profile said why: with the accumulator and the RAM slab working, what was
left was almost entirely **attach**, which the reconciler ran on every
step — `attach` 209 ms (of which `_attach_gauss_node` 138 and
`_build_gauss_state` 67), `_teardown` 45, `add_layer` 20. None of that
was data reading; it was rebuilding a diagram that did not need
rebuilding. Step 4 removes it, which is also why the Gauss and nodal
cases converge: with no re-attach per step, the extrapolation is paid
once at attach and never again.

The other gestures are unchanged by design — they are structural, so they
take the full path: `slot` 148, `scope` 60, `pane` 325, `orbit` 18.

Three separable findings, in the order I'd attack them:

1. **The Gauss→node extrapolation runs TWICE per step.** `attach` does one
   for the step it is constructed at, then `update_to_step` immediately
   does another for the step actually wanted. Halving this is pure
   bookkeeping — no algorithmic change.
2. **A pure time change tears down and re-attaches the whole diagram.**
   The reconciler signature includes the instant, so every scrub step is a
   full `_teardown` + `realize_pane` + `attach`. The plot pane already has
   a cheap cursor path that slides the playhead over cached arrays; the
   mesh pane has no equivalent. This is the structural fix and the biggest
   win.
3. **`extrapolate_gauss_slab_per_element` is a per-element Python loop** —
   164 020 `np.broadcast_to` calls for 10 steps (~16 k/step, i.e. one per
   element). Vectorizable, and it would also speed up the non-viewer
   consumers of `results/_gauss_extrapolation.py`.

**The contour's field matters more than the pane count** — measured with
`--contour`, one pane, 10 steps:

| contour field | source | scrub mean ms |
|---|---|---:|
| `stress_zz` | Gauss | 294 |
| `displacement_z` | nodal | **127** |

So the Gauss→node extrapolation is ~167 ms of the 294, and findings 1 and
3 are worth exactly that. But note the nodal case is still **127 ms — 8
fps**, with no extrapolation in it at all: that residual is finding 2, the
full teardown-and-re-attach every step, and it is the one that has to be
fixed for playback to feel like playback.

Unmeasured, deliberately: `pane` at 336 ms is dominated by creating a
fresh `QtInteractor` (GL context) per added view, which is the same cost
boot pays. It is a real number but it is not a per-frame cost.
