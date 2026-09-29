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
world: **native capture** (apeGmsh queries the live domain and writes its
own HDF5), **classic recorders** (OpenSees `.out` / `.xml` files), or
**MPCO** (STKO's `.mpco` HDF5).

On this page `ops` is always the bridge and `opspy` is the openseespy module
(`import openseespy.opensees as opspy`).

## The grid

|  | **Read: native capture** → `from_native` | **Read: classic recorders** → `from_recorders` | **Read: MPCO** → `from_mpco` |
|---|---|---|---|
| **Run: in-process** | `ops.domain_capture(capture_spec, path=...)`. The default: every category from nodes to fibers and layers, plus mode shapes via `cap.capture_modes(n)`. | `recorder_spec.emit_recorders("out")`. Nodes, element forces, gauss points and line stations; fiber and layer records are skipped and modal ones raise. | `ops.recorder.MPCO(...)` before `ops.run()`, or `recorder_spec.emit_mpco(...)`. Fibers, layers and modes, on a build that has the MPCO recorder. |
| **Run: export** | *(none: capture needs a live domain in your Python session)* | `recorder_spec.to_tcl_commands(...)` in a driver around `ops.tcl(...)`. Reproducible, check-in-able, cluster-friendly. | `ops.recorder.MPCO(...)` before `ops.tcl(...)`. One `recorder mpco` line in the deck; run it under an OpenSees that has the recorder. |

`capture_spec` is a `DomainCaptureSpec`, which names what to record by
physical group. `recorder_spec` is a `ResolvedRecorderSpec`, which holds
concrete node and element IDs. The recipes below build both.

Start with in-process capture unless something pushes you off it. It needs
nothing beyond a stock openseespy, covers the most, and writes one file that
carries its own model, which is why the tutorials use it. The other cells
answer specific pressures: classic recorders when you want plain OpenSees
output files, MPCO when the results must open in STKO, and export when the
run has to happen somewhere your notebook is not.

## The recipes

Every recipe solves the same model: a 3 m steel cantilever carrying a 10 kN
tip load, applied in ten load steps. Its tip deflects *PL*³/3*EI* = 6.75 mm,
and each recipe reads that number back.

```python
import numpy as np
import openseespy.opensees as opspy
from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees, OpenSeesModel

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
```

Each recipe starts from a fresh bridge, because a recorder declared on a
bridge rides along into everything that bridge emits afterwards.

### In-process, native capture

`ops.run()` builds the model and its analysis chain into openseespy without
analysing, so your loop can call `cap.step` after every increment:

```python
from apeGmsh.results.capture import DomainCaptureSpec

ops = build_bridge()
capture_spec = DomainCaptureSpec(opensees=ops)
capture_spec.nodes(pg="Tip", components=["displacement"])

ops.run()
with ops.domain_capture(capture_spec, path="run.h5") as cap:
    cap.begin_stage("load", kind="static")
    for _ in range(10):
        opspy.analyze(1)
        cap.step(t=opspy.getTime())
    cap.end_stage()
opspy.wipe()

results = Results.from_native("run.h5", fem=fem,
                              model=OpenSeesModel.from_h5("run.h5"))
```

Don't step with the bridge's own `ops.analyze(steps=1)` here: it rebuilds
the domain from scratch on every call, so each `cap.step` would record the
first increment again. The run file carries its own model, which is why the
same path feeds `OpenSeesModel.from_h5`. For mode shapes, call
`cap.capture_modes(n)` inside the block on a model that has mass.

### In-process, classic recorders

Reach for this when you want plain OpenSees recorder files from a notebook
run and need only nodes, element forces, gauss points or line stations. A
`ResolvedRecorderSpec` holds IDs rather than names, so resolve the physical
group once, as an array:

```python
from apeGmsh.results.spec import ResolvedRecorderRecord, ResolvedRecorderSpec

recorder_spec = ResolvedRecorderSpec(
    fem_snapshot_id=fem.snapshot_id,
    records=(ResolvedRecorderRecord(
        category="nodes", name="tip",
        components=("displacement_x", "displacement_y"),
        dt=None, n_steps=None,
        node_ids=np.asarray(fem.nodes.select(pg="Tip").ids),
    ),),
)

ops = build_bridge()
ops.h5("model.h5")                  # the model archive the read needs
ops.run()
with recorder_spec.emit_recorders("out") as live:
    live.begin_stage("load", kind="static")
    for _ in range(10):
        opspy.analyze(1)
    live.end_stage()                # removing the recorders flushes out/
opspy.wipe()

results = Results.from_recorders(recorder_spec, "out", fem=fem, stage_id="load",
                                 model=OpenSeesModel.from_h5("model.h5"))
```

Each stage's files are prefixed `<stage>__`, which is why the read names
`stage_id=`.

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
ops.tcl("model.tcl")
Path("run.tcl").write_text("\n".join([
    "source model.tcl",
    "file mkdir out",
    *recorder_spec.to_tcl_commands(output_dir="out"),
    "analyze 10",
    "wipe",
]) + "\n")
```

Run `run.tcl` wherever OpenSees lives, bring `out/` back beside `model.h5`,
and read it. A deck's files carry no stage prefix, so the read takes no
`stage_id=`:

```python
results = Results.from_recorders(recorder_spec, "out", fem=fem,
                                 model=OpenSeesModel.from_h5("model.h5"))
```

For an openseespy deck, `recorder_spec.to_python_commands(output_dir="out")`
returns the same recorders as `ops.recorder(...)` lines.

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

- **Every constructor needs the model.** `from_native` and `from_recorders`
  take an in-memory `model=` (`OpenSeesModel.from_h5(...)`); `from_mpco`
  takes `model_h5=` as a path, because MPCO files carry no `/opensees/`
  zone. Omitting it raises `TypeError`. `from_recorders` also needs `fem=`.
- **Bridge-declared recorders are not what `from_recorders` reads.** The
  read finds files by the names its spec generates (`out/tip_disp.out`
  above). Recorders declared on the bridge ride into `ops.run()` and every
  deck, but name their files their own way: `ops.recorder.declare(...)`
  after the declaration (`out/default__default__disp.out`),
  `ops.recorder.Node(...)` after its `file=`. Use them when you consume the
  `.out` files yourself, and generate the recorders from the spec when
  apeGmsh should read them back.
- **Fibers, layers and modes skip the classic path.** `emit_recorders`
  warns and skips fiber and layer records, and raises at `__enter__` on a
  modal one. Route those through native capture or MPCO.
- **MPCO needs a build that has it.** Not every openseespy build ships the
  MPCO recorder (STKO's bundled Python does), and `emit_mpco` raises at
  `__enter__` with a remediation pointer when it is missing. Without such a
  build, native capture gives the same fiber, layer and modal coverage.
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
- API: [`apeGmsh.results.Results`](../api/results.md): `from_native`,
  `from_recorders`, `from_mpco` and the fork's `from_ladruno`.

---

*Next: [See the three strategies agree on one model](../examples/results-strategies.md).*
