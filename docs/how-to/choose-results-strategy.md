# Running & reading: choose your path

You have a meshed model and a typed `apeSees(fem)` bridge. This page settles
the two decisions between you and a `results` object, **how to run** OpenSees
and **how to read** what it wrote, and gives the working calls for each
combination. Whichever cell you pick, the `Results` query API you read with
afterwards is the same.

## The two axes

**Run** decides where the analysis executes. *In-process* builds the model
into the live openseespy domain of your own Python session and steps it
there. *Export* writes a standalone `.tcl` or `.py` deck (`ops.tcl(...)` /
`ops.py(...)`) that you run somewhere else: a cluster, STKO, or a separate
process.

**Read** decides how the numbers get back into apeGmsh's label and query
world: the **Ladruno fork's own recorder** (a `.ladruno` HDF5 file),
**classic recorders** (OpenSees `.out` / `.xml` files), or **MPCO**
(STKO's `.mpco` HDF5).

On this page `ops` is always the bridge and `opspy` is the OpenSees module
the bridge builds into (`opspy = get_ops()`, from
`apeGmsh.opensees.emitter.live`). Beside a fork build,
`import openseespy.opensees` can bind a second module with its own, empty
domain, so the recipes never import it.

## The grid

|  | **Read: `.ladruno`** → `from_ladruno` | **Read: classic recorders** → `from_recorders` | **Read: MPCO** → `from_mpco` |
|---|---|---|---|
| **Run: in-process** | `recorder_spec.emit_default("run", model_h5=...)`. The default: on the fork it records the fork's own file; on a stock build it falls back to the MPCO cell, and on pip openseespy to the classic cell, so one script runs on every build. | `recorder_spec.emit_recorders("out", model=model)`. Nodes, element forces, gauss points, line stations and fibres; layer and modal records raise by name. | `ops.recorder.MPCO(...)` before `ops.run()`, or `recorder_spec.emit_mpco(...)`. Fibres, layers and modes, on a build that has the MPCO recorder. |
| **Run: export** | `read_deck_mpco("run.mpco", model_h5=...)` converts the `.mpco` of the bridge's own deck to `.ladruno` beside it. | `recorder_spec.to_tcl_commands(...)` in a driver around `ops.tcl(...)`. Reproducible, check-in-able, cluster-friendly. | `ops.recorder.MPCO(...)` before `ops.tcl(...)`. One `recorder mpco` line in the deck; run it under an OpenSees that has the recorder. |

`recorder_spec` is a `ResolvedRecorderSpec`, which holds concrete node and
element IDs; a physical group resolves to them once, with
`fem.nodes.select(pg=...).ids`. The recipes below build one.

Modes are not a recorder record on any in-process route. `ops.eigen(n)`
returns an `EigenResult`, and its `.results()` writes the modes to a file
and opens them as a `Results` (`to_native(path)` keeps the file, and
`to_ladruno(path)` writes them as a `.ladruno`). The
[modal example](../examples/modal-analysis.md) walks through it.

Start with the default route unless something pushes you off it. It needs
nothing beyond the openseespy you have, picks the richest recorder your
build offers, and is what the tutorials use. The other cells answer
specific pressures: classic recorders when you want plain OpenSees output
files, MPCO when the results must open in STKO, and export when the run has
to happen somewhere your notebook is not.

## The recipes

Every recipe solves the same model: a 3 m steel cantilever carrying a 10 kN
tip load, applied in ten load steps. Its tip deflects *PL*³/3*EI* = 6.75 mm,
and each recipe reads that number back.

```python
import numpy as np
from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees, OpenSeesModel
from apeGmsh.opensees.emitter.live import get_ops
from apeGmsh.results.spec import ResolvedRecorderRecord, ResolvedRecorderSpec

opspy = get_ops()

L, E, b, h, P = 3.0, 200e9, 0.10, 0.20, 10_000.0
A, Iz = b * h, b * h**3 / 12.0

with apeGmsh(model_name="cantilever") as g:
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
    p1 = g.model.geometry.add_point(L, 0.0, 0.0)
    beam = g.model.geometry.add_line(p0, p1)
    g.model.sync()
    g.physical.add(1, [beam], name="Beam")
    g.physical.add(0, [p0], name="Fixed")
    g.physical.add(0, [p1], name="Tip")
    g.mesh.sizing.set_global_size(L / 10.0)
    g.mesh.generation.generate(1)
    fem = g.mesh.queries.get_fem_data(dim=1)

def build_bridge():
    """A fresh bridge: the cantilever and a ten-step static analysis chain."""
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear()          # a 2-D transform takes no vecxz
    ops.element.elasticBeamColumn(pg="Beam", transf=transf, A=A, E=E, Iz=Iz)
    ops.fix(pg="Fixed", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        pat.load(pg="Tip", forces=(0.0, -P, 0.0))
    ops.constraints.Plain(); ops.numberer.Plain(); ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10); ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=0.1); ops.analysis.Static()
    return ops

# What to record: the tip's displacement, by node id.
recorder_spec = ResolvedRecorderSpec(
    fem_snapshot_id=fem.snapshot_id,
    records=(ResolvedRecorderRecord(
        category="nodes", name="tip",
        components=("displacement_x", "displacement_y"),
        dt=None, n_steps=None,
        node_ids=np.asarray(fem.nodes.select(pg="Tip").ids),
    ),),
)
```

Each recipe starts from a fresh bridge, because a recorder declared on a
bridge rides along into everything that bridge emits afterwards.

### In-process, the default route

`ops.run()` builds the model and its analysis chain into OpenSees without
analysing, and `emit_default` opens the recorder your build supports around
your own loop:

```python
ops = build_bridge()
ops.h5("model.h5")                  # the model archive every route reads with
ops.run()
with recorder_spec.emit_default("run", model_h5="model.h5") as run:
    for _ in range(10):
        opspy.analyze(1)
opspy.wipe()

results = run.results()
```

`run.route` says which recorder ran:

| Build | Writes | `run.results()` |
|---|---|---|
| the Ladruno fork | `run.ladruno` | stamps the file with `model.h5`'s pairing, then `Results.from_ladruno` |
| stock, with MPCO | `run.mpco` | `Results.from_mpco`, after one `DefaultRouteWarning` |
| pip, no MPCO | classic `.out` files in `run_out/` | `Results.from_recorders`, after one `DefaultRouteWarning` |

Don't step with the bridge's own `ops.analyze(steps=1)` here: it rebuilds
the domain from scratch, and the recorder the block opened goes with it.
Read with `run.results()` after the block has closed.

The stock MPCO file is not converted to `.ladruno`, and it stays on
`from_mpco`. An unstaged run maps to one stage, and that stage's kind
(static or transient) has to be a fact. Your loop drives OpenSees
directly, so a `loadConst` or a second `opspy.analysis(...)` inside the
block never reaches `model.h5`. `read_deck_mpco` below does not apply to
an in-process run.
`emit_default` refuses a rerun onto an existing `run.ladruno` unless you
pass `overwrite=True`. It has no route for modal records; modes come from
`ops.eigen(n)`, as the grid says.

### In-process, classic recorders

Reach for this when you want plain OpenSees recorder files from a notebook
run whatever the build, and need nodes, element forces, gauss points, line
stations or fibres:

```python
ops = build_bridge()
ops.h5("model.h5")                  # the model archive the read needs
model = OpenSeesModel.from_h5("model.h5")
ops.run()
with recorder_spec.emit_recorders("out", model=model) as live:
    live.begin_stage("load", kind="static")
    for _ in range(10):
        opspy.analyze(1)
    live.end_stage()                # removing the recorders flushes out/
opspy.wipe()

results = Results.from_recorders(recorder_spec, "out", fem=fem, stage_id="load",
                                 model=model)
```

Each stage's files are prefixed `<stage>__`, which is why the read names
`stage_id=`. The spec names elements by FEM id, but OpenSees records them
by element tag, and the bridge numbers elements sequentially by default,
so the two differ. `model=` carries the pairing the model was built with,
and every element recorder is emitted under the right tags. A spec with an
element record refuses to emit without it; a node-only spec does not need
it, because nodes keep their FEM ids.

### In-process, MPCO

Reach for this when the results must open in STKO. Declare the recorder on
the bridge and it opens with the domain:

```python
ops = build_bridge()
ops.recorder.MPCO(file="run.mpco", nodal_responses=("displacement",))
ops.h5("model.h5")
ops.run()
for _ in range(10):
    opspy.analyze(1)
opspy.wipe()                        # closes the recorder, flushing run.mpco

results = Results.from_mpco("run.mpco", fem=fem, model_h5="model.h5")
```

If you already hold a `recorder_spec`, wrapping the loop in
`with recorder_spec.emit_mpco("run.mpco"):` does the same without touching
the bridge. The [MPCO how-to](results-mpco.md) covers partitioned runs and
where to write the file.

### Export, classic recorders

Reach for this for cluster jobs and for decks you keep under version
control. `ops.tcl` writes the model and its analysis chain, and a short
driver adds the recorders and the analysis:

```python
from pathlib import Path

ops = build_bridge()
ops.h5("model.h5")
model = OpenSeesModel.from_h5("model.h5")
ops.tcl("model.tcl")
Path("run.tcl").write_text("\n".join([
    "source model.tcl",
    "file mkdir out",
    *recorder_spec.to_tcl_commands(output_dir="out", model=model),
    "analyze 10",
    "wipe",
]) + "\n")
```

Run `run.tcl` wherever OpenSees lives, bring `out/` back beside `model.h5`,
and read it. A deck's files carry no stage prefix, so the read takes no
`stage_id=`:

```python
results = Results.from_recorders(recorder_spec, "out", fem=fem, model=model)
```

For an openseespy deck,
`recorder_spec.to_python_commands(output_dir="out", model=model)` returns
the same recorders as `ops.recorder(...)` lines.

### Export, MPCO

The STKO-shop production path, and the one that scales to parallel runs.
The bridge's MPCO recorder rides into the deck as one `recorder mpco` line:

```python
ops = build_bridge()
ops.recorder.MPCO(file="run.mpco", nodal_responses=("displacement",))
ops.h5("model.h5")
ops.tcl("model.tcl", analyze_steps=10)
```

Run `model.tcl` under an OpenSees that has the MPCO recorder, then:

```python
results = Results.from_mpco("run.mpco", fem=fem, model_h5="model.h5")
```

`read_deck_mpco("run.mpco", model_h5="model.h5")`, from
`apeGmsh.results.live`, reads the run as `.ladruno` instead. It is only for
a `.mpco` written by running the bridge's own exported deck (`ops.py` or
`ops.tcl`) out of process, as emitted. A staged deck (`ops.stage` blocks,
each recording its kind) converts beside itself, and the `.mpco` is kept.
An unstaged deck converts on its own run record: a deck emitted with
`analyze_steps=`, like the one above, writes `run.mpco.run.json` beside the
`.mpco` once its analyze loop has returned, and the conversion checks that
record against the `.mpco`, `model.h5` and the program (one `analysis`
declaration, and the steps it ran). When a check fails, the deck is read
with `from_mpco` after one warning that names the check. Other refusals
raise, such as an archive of another model or an incomplete partition set. An in-process run is not a deck, so read its
`.mpco` with `from_mpco`.

## The read side is identical

Whichever recipe ran, the query is the same:

```python
tip = results.nodes.get(pg="Tip", component="displacement_y")
print(f"tip deflection = {tip.values[-1, 0] * 1e3:.2f} mm")
results.close()
```

```text
tip deflection = -6.75 mm
```

Everything else about reading, from stages and time slicing to modes,
fibers and the viewers, works the same way whichever constructor opened the
file; the [Results concept page](../concepts/results.md) walks through it.

## Notes / gotchas

- **Every constructor needs the model.** `from_recorders` (and
  `from_native`, which opens the modes file `EigenResult.to_native` writes)
  take an in-memory `model=` (`OpenSeesModel.from_h5(...)`); `from_mpco`
  takes `model_h5=` as a path, because MPCO files carry no `/opensees/`
  zone. Omitting it raises `TypeError`. `from_recorders` also needs `fem=`.
  `from_ladruno` takes `model_h5=` too, and binds that archive's physical
  groups and labels. `emit_default` passes the model for you.
- **Bridge-declared recorders are not what `from_recorders` reads.** The
  read finds files by the names its spec generates (`out/tip_disp.out`
  above). Recorders declared on the bridge ride into `ops.run()` and every
  deck, but name their files their own way: `ops.recorder.declare(...)`
  after the declaration (`out/default__default__disp.out`),
  `ops.recorder.Node(...)` after its `file=`. Use them when you consume the
  `.out` files yourself, and generate the recorders from the spec when
  apeGmsh should read them back.
- **Layers and modes have no classic path.** `emit_recorders` records
  fibre records (`section fiberData2` at each integration point), and raises
  at `__enter__` on a layer or a modal record, naming it. Record layers with
  MPCO or on the fork; take modes from `ops.eigen(n)`.
- **MPCO needs a build that has it.** Not every openseespy build ships the
  MPCO recorder (STKO's bundled Python does), and `emit_mpco` raises at
  `__enter__` with a remediation pointer when it is missing. Without such a
  build, the default route falls back to classic recorders, which record
  fibres but not layers.
- **Loads are opt-in (ADR 0051).** MP constraints auto-emit, but
  `g.loads.*` cases do not: import one into a bridge pattern with
  `pat.from_model(case)`, or author loads with `pat.load(...)` as the
  recipes do. Masses and supports are re-declared on the bridge
  (`ops.mass`, `ops.fix`).

## See also

- Concept: [Results](../concepts/results.md), the read model that every
  constructor on this page opens onto.
- How-to: [Export to a Tcl or openseespy script](export-script.md) ·
  [Get results via MPCO](results-mpco.md).
- API: [`apeGmsh.results.Results`](../api/results.md): `from_recorders`,
  `from_mpco`, the fork's `from_ladruno`, and `from_native` for modes.

---

*Next: [See the three strategies agree on one model](../examples/results-strategies.md).*
