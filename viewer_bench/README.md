# viewer_bench

Models that exist to exercise the viewers, not to teach apeGmsh.

The curriculum under `examples/` answers "how do I build this?". This
answers a different question: *does the viewer hold up against a model
with everything in it at once?* These models are meant to be rebuilt
rarely, opened often, and pointed at whatever part of the Results window
is being worked on.

This folder is an **APE Studio habitat** (ADR 0095), stamped with
`python -m apeGmsh.studio init --name viewer_bench --model ssi_frame_wall
--no-git`, so bench runs are recorded the same way any other model's runs
are: one immutable case per run, with logs, git provenance and a verify
tally in `run.json`.

```text
viewer_bench/
├── ape.project.yaml            # habitat manifest; entry -> build.py
├── APE/                        # playbook (instructions, memory, catalogs)
├── models/ssi_frame_wall/
│   ├── src/                    # build.py (driver) + check_slots.py (verify)
│   └── cases/<case>/           # run.json + results/ + logs/ + deck/
├── reports/  postmortem/  references/  tools/  scripts/
└── README.md                   # this file
```

`--no-git`: the habitat lives inside the apeGmsh repo, so it is versioned
by that repo — a nested `git init` would be wrong. Its own `.gitignore`
keeps bulk case data out while tracking each `run.json`.

| Bench | What it stresses |
|---|---|
| [`ssi_frame_wall`](models/ssi_frame_wall/src/) | The canonical ADR 0098 fixture — all seven result slots, three element families, three stages, mixed DOF |

## ssi_frame_wall

A 3-D linear-elastic soil-structure model: a soil block with a raft cast
into its surface, carrying a three-storey frame with a shear wall and
floor slabs.

```
                    slabs (ShellMITC4)            wall (ShellMITC4)
                    ────────────────┐            ┌──────────
   z = 9.6 ─────────────────────────┼────────────┼──  beams (forceBeamColumn)
   z = 6.4 ─────────────────────────┼────────────┼──
   z = 3.2 ─────────────────────────┼────────────┼──  columns (forceBeamColumn)
   z = 0   ═══════════ grade beams ═╧════════════╧══
           ░░░░░░░░░░░ raft, 14×14×0.8 (tet4) ░░░░░░░░     ← embedded tie
   ░░░░░░░░░░░░░░░░░░░ soil, 30×30×15 (tet4) ░░░░░░░░░░░░░░░
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ fixed base ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

Units are kN, m, s, t. Everything is elastic — the bench is about
pictures, not material behaviour.

### Why it is shaped this way

Each choice below exists to feed something the viewer has to draw.

**Three element families in one model.** tet4 solids, quad4 shells and
line elements, so the `element_types` scope axis has three real answers
and every kind of cell renders in one scene.

**The frame is `forceBeamColumn`, not `elasticBeamColumn`.** The `line`
slot reads `results.elements.line_stations`, which only a *sectioned*
beam produces — an elastic beam-column has no sections to probe, so it
would leave the slot permanently empty. The section is elastic, so the
model stays linear.

**The structure meets the ground through an embedded tie.** Solid nodes
carry 3 DOF and frame/shell nodes carry 6. Landing a column straight on
a brick node leaves its rotational DOF with no stiffness — legal and
singular. Instead the grade-beam grillage at z = 0 (which contains all
nine column bases) is tied into the raft with
`g.constraints.embedded(..., rotational=True)`, one
`ASDEmbeddedNodeElement` per node. The soil and raft, both 3-DOF, are
simply conforming — fragmented into one mesh, no constraint needed.

**The soil carries no self-weight.** It keeps its density for mass, but
gravity is applied only to the raft and the structure. Soil self-weight
would settle the whole block and swamp the structural displacement
field, which is the thing you actually want to look at. What you get
instead is a clean bearing-pressure bulb under the raft. (This is the
usual geostatic convention: the in-situ state is the datum.)

**Gravity is one load case of consistent nodal loads**, not element body
forces — volume gravity on the raft, surface traction on slabs and wall,
line load on the frame. That is what puts glyphs in the `loads` slot; a
`body_force=` on the solid elements would show nothing.

**Three stages, one file.** `gravity` (static), six `mode_*` stages, and
`dynamic` (150 captured instants under a generated three-frequency pulse
at 0.30 g, Rayleigh damping fitted to the *computed* modes 1 and 3, not
to guessed frequencies). The dynamic stage is what makes the scrubber,
plot cursors and the §7 time link testable at all.

**The soil box's support faces are named groups even though nothing is
declared on them.** Meshing a volume also meshes its boundary surfaces,
and `get_fem_data()` collects the whole gmsh mesh rather than only
grouped entities — so a 3-D model carries a coincident tri3 skin over
every solid whether you name it or not. Naming it (`soil_base`,
`soil_sides`) beats leaving ~2000 elements unclaimed: they become
scopeable, and a `tri3` entry on the element-types axis has an
explanation. It also makes the bench cover a case worth covering —
a physical group with cells but no results behind them.

**Both analysis chains use `algorithm Linear`.** The model is linear, so
Newton would factorize twice per step to confirm a residual that is
already zero.

### What feeds what

| Slot / feature | Fed by |
|---|---|
| `contour` | nodal displacement, and Gauss stress extrapolated to nodes |
| `vector` | nodal displacement vectors; Gauss principal stresses |
| `gauss` | `stress` at the tet integration points (soil + raft) |
| `line` | `line_stations` — N / V / M / torsion on columns, beams, grade beams |
| `sand` | any colour-mapped field over the solids |
| `loads` | the `dead` pattern, imported with `p.from_model("dead")` |
| `reactions` | `reaction_force` on every supported soil-box node |
| deform poses | displacement, velocity, acceleration (dynamic) + 6 mode shapes |
| time link, scrubber, plots | 150 captured instants in the `dynamic` stage |
| scope: physical groups | `soil` `raft` `columns` `beams` `grade_beams` `slabs` `wall` `soil_base` `soil_sides` |
| scope: element types | `tet4` `quad4` `line2`, plus the `tri3` boundary skin |
| clip / section planes | a solid block worth cutting into, with a raft inside it |
| selection → plot | any node or Gauss point; the roof nodes have the best histories |

### Running it

As a case, which is the normal way — it records the run:

```bash
python scripts/run_case.py --model ssi_frame_wall --case c002_something --script models/ssi_frame_wall/src/build.py --verify models/ssi_frame_wall/src/check_slots.py
```

That writes `models/ssi_frame_wall/cases/<case>/` with `results/`
(the `model.h5` + `run.h5` pair and the slot stills), `logs/run.log` and
`logs/verify.log`, a `deck/` disclosure note, and `run.json` carrying the
git provenance and the PASS/FAIL tally. Run records are immutable: a case
id that already has `run.json` is refused rather than overwritten.

Or by hand, from wherever you want the artifacts:

```bash
python models/ssi_frame_wall/src/build.py --out /some/dir
```

`check_slots.py` is the acceptance gate: it fills every slot, renders a
still off each, checks the dynamic stage actually moves, and exits
non-zero if anything refuses. A failure there is a bench defect — the
model is only canonical while every slot has something real behind it.

Then open it:

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel

r = Results.from_native("run.h5", model=OpenSeesModel.from_h5("model.h5"))
r.viewer()
```

### Sizes

`--size small` (default) is the everyday bench. `--size large` refines
the same model for viewer performance work — pane counts, repaint cost,
scrubber smoothness — without changing anything about what it contains.

| | small | large |
|---|---|---|
| shell / frame division | 0.75 m | 0.40 m |
| soil size field | 1.2 → 4.0 m | 0.6 → 2.0 m |
| dynamic capture stride | every 4th step | every 10th step |

### Findings so far

What the bench has already turned up, in the order it turned up.

**Fixed — `capture_modes` could not read a mixed-DOF model.**
`DomainCapture.capture_modes` gated rotational capture on the model
*envelope* `ndf`, then asked every node for DOF 4–6 by index. In any
solid + frame assembly the 3-DOF solid nodes have no DOF 4–6 and the
call raises, so no such model could ever record a mode shape. It now
reads each node's whole eigenvector in one call, which is correct for
mixed ndf and 6× fewer solver calls besides.
(`src/apeGmsh/results/capture/_domain.py`)

> **This entry was wrong for a while, and how it went wrong is worth
> more than the fix.** The repair was made in the worktree where this
> bench was built and **never merged**; the paragraph claiming it was
> fixed *was* merged, with the bench itself. So `main` carried a
> document asserting a fix that its code did not contain, and every
> mixed-DOF model still died at the first mode. It surfaced only when
> the bench was re-run months later and failed at exactly the line the
> README said was repaired.
>
> Two things to take from it. A findings list copied forward is a
> claim about code, and claims about code have to be re-checked
> against the code — a green document is not a green test. And the
> reason nothing else caught it: the fix shipped with no regression
> test, so there was nothing to notice its absence. The test now
> exists (`tests/test_results_domain_capture.py`,
> `test_capture_modes_reads_a_mixed_dof_model`), and it needed the
> `_FakeOps` double to be made **strict** first — the double answered
> `0.0` for a DOF the node does not have, where real openseespy
> raises, so a lenient double would have passed the buggy code.

**Fixed by ADR 0098 Amendment 3 — the pane host was never the central
widget, and making it one crashed the window.** These were one finding,
not two, and that is what made the answer a design change rather than a
patch. The diagnosis below is kept because it is the evidence the
amendment rests on; what follows it is what the amendment did.

Measured: `SessionWindow.__init__` mounts the host via
`set_central_widget` (`viewers/session/_window.py:150`) and it is seated;
the instant the window is first shown, `centralWidget()` returns `None`.
Traced — `setCentralWidget` is called exactly once and
`takeCentralWidget` never — and the host object stays alive as a child,
so it paints at its untouched 640x480 default on top of the docks while
the two side docks split the whole width (1254 + 1253 px of 2560). That
640x480 is exactly the pane cluster you see floating in the window.

The obvious cause is construction order: the shell is built
`central_interactor=False`, so `_build_layout` arranges the docks,
applies `resizeDocks` and snapshots the default state while the window
has **no** centre. It is not the persisted layout (clearing
`QSettings('apeGmsh','ResultsViewer')` changes nothing, and the stored
schema is 7 against this window's 1 so the restore returns early) and not
maximization (plain `show()` does it too).

**But the obvious fix takes a native crash.** Occupying the centre from
construction with a placeholder — so the docks are arranged around a real
centre and `set_central_widget` merely replaces it — makes the host truly
central, and then the first show dies:

```
Windows fatal exception: access violation
  viewer_window.py:1322 in present   ->  showMaximized()
```

and on a plain `win.show()` as well. So the panes' GL contexts are never
realized inside the window's layout today, and **the broken layout is what
keeps the window alive**. Fixing the geometry means fixing multi-context
realization first — pane interactors created lazily after the window is
up, or one render window with N viewports (the alternative ADR 0098
Amendment 1 weighed and rejected). Amendment 1's prototype claimed N
`QtInteractor`s in nested splitters were verified on Windows and CI Mesa;
this path is not that, or the code has drifted from it.

Three repairs were tried and **all reverted**: a `QTimer.singleShot(0)`
re-seat after show, a deferred `resizeDocks`, and the placeholder centre.
The first two measure perfectly on a fixed-size `show()` (centre
1863x876, docks 260/380/84) and wreck the real *maximized* window; the
third crashes outright. This needs an Amendment 1 decision, not another
patch.

Side trap found on the way, worth its own line: `_build_layout` imports
`QtCore` **locally**, so code moved out of that method silently loses the
name — and the `resizeDocks` block sits inside `except Exception: pass`,
which turns the resulting `NameError` into a no-op. That pairing cost two
rounds of measurement.

**How it was settled.** Amendment 3 stopped trying to make one host
central. Each pane is now its own `QDockWidget` in the shell's window, so
`saveState` covers them and no single widget has to hold N GL contexts
inside a splitter tree. `centralWidget()` is legitimately `None` — that
is the design, not the symptom it used to be. Re-run on this bench after
the amendment merged: three panes open, float, re-dock, tabify and close
without a single Qt or VTK message, and the process exits 0.

**Open — the `contour` and `vector` slots cannot see derived scalars.**
`resolve_contour_topology` tests the quantity against
`results.inspect.components(stage=…)`, which lists only *recorded*
columns. Derived scalars (`von_mises_stress`, `principal_stress_*`) and
anything registered through `results.nodes.define(...)` are absent from
that set, so the slot refuses them — even though `Gauss("von_mises_stress")`
on the same run paints happily, and ADR 0098 §5 requires a contour and a
Gauss slot of the same quantity to *share one legend*, which is
unreachable while the contour cannot accept the quantity at all. A
recorded component (`Contour("stress_zz")`) works, so the gap is exactly
"derived and custom scalars are invisible to the gate". The fix is to
test `available_components()` instead.
(`src/apeGmsh/viewers/session/_specs.py`)

**Open — the Tcl emitter writes numpy reprs into a `Path` series.**
`ops.timeSeries.Path(values=tuple(numpy_array))` emits
`timeSeries Path 2 -values np.float64(0.0) np.float64(0.0) ...`, which is
not valid Tcl. `np.float64` *is* a `float` subclass, so the primitive's
type check passes and only the formatting is wrong — under numpy 2 its
`str()` gained the constructor repr. The live path is unaffected
(openseespy takes the objects directly), so this only bites a deck you
emit and run through the binary. A `float(v)` at format time fixes it.
(`src/apeGmsh/opensees/emitter/`)

**Not a bug, but the trap that cost the most — `loadConst` freezes
*every* pattern.** The bench's first working transient ran 600 steps,
captured 150 instants, reported no convergence failure, and did not move
at all: `opspy.loadConst("-time", 0.0)` after gravity had frozen the load
factor of the `UniformExcitation` too, at its t = 0 value of zero. The
bridge emits its whole deck at `ops.run()`, so a pattern it declares
cannot be born "after" `loadConst` — the excitation is therefore created
through raw openseespy inside `solve()`. `check_slots.py` now asserts the
dynamic stage actually moves, because every still it renders looked
perfectly fine while the model stood still.

**Open — the `sand` slot reads nodal components only.** `Sand("stress_zz")`
refuses with `NoDataError` although the contour slot extrapolates the
same Gauss field to nodes without complaint.

**Known and already documented — the `materials` scope axis refuses.**
No element→material index is published, so the axis fails loud rather
than drawing the wrong set. The bench confirms the refusal is legible.

`check_slots.py` probes the two open gaps on every run without failing
the gate, and says `NOW SUPPORTED` the day one of them starts working.
