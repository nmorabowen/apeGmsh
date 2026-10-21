# Read a node's displacement and reactions

Pull a displacement or reaction-force time history out of a finished run,
targeting nodes by physical-group name. This is the bread-and-butter
post-processing read: open the results file, ask one composite for one
component, index `.values`.

The shape in one line:

> a `Results` (`run.results()`, or any `Results.from_*` constructor) →
> `results.nodes.get(pg=, component=)` → a frozen slab with `.values`
> `(T, N)`, `.node_ids`, and `.time`.

## Recipe

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel

# Every Results constructor REQUIRES the read-side broker (model= /
# model_h5=); omitting it raises TypeError (ADR 0020 INV-1). Pass fem=
# too so pg=/label= name-resolution works against your session mesh.
# Here: the classic .out files a pip run of emit_default("run", ...)
# wrote to run_out/ (run.results() opens the same thing for you).
model = OpenSeesModel.from_h5("model.h5")
results = Results.from_recorders(recorder_spec, "run_out/", model=model, fem=fem)

# --- Displacement: u_y history at every node in PG "Top" ---
disp = results.nodes.get(pg="Top", component="displacement_y")
disp.values        # ndarray (T, N) — one column per node in "Top"
disp.node_ids      # the matching node IDs, same column order
disp.time          # (T,) time/pseudo-time axis

u_y_final = disp.values[-1]          # last step, all nodes -> (N,)
u_y_node0 = disp.values[:, 0]        # full history at the first node -> (T,)

# --- Reaction: total base reaction in y, summed over PG "Base" ---
react = results.nodes.get(pg="Base", component="reaction_force_y")
Ry = react.values[-1].sum()          # last-step sum over the base nodes
```

`nodes.get(...)` is keyword-only: `pg=` / `label=` / `selection=` / `ids=`
pick *which* nodes (named selectors union), `component=` picks *what*, and
`time=` slices the step axis (`time=-1` for the last step,
`time=[0, 50, 99]` for specific steps). Component names are the canonical
apeGmsh vocabulary: `displacement_x/y/z`, `reaction_force_x/y/z`,
`velocity_*`, `acceleration_*`.

## Notes / gotchas

- **Reactions must be recorded.** `reaction_force_*` only exists in the
  file if you asked for it on the write side — a `nodes` record whose
  components include `reaction_force_y` (the [results strategy
  how-to](choose-results-strategy.md) declares one). If the read comes
  back empty, the component was never recorded. Call
  `results.nodes.available_components()` to see what's actually there.
- **`model=` is mandatory; `fem=` is for names.** Without `model=` the
  constructor raises `TypeError`. Without a bound `fem`, the embedded
  `/model/` snapshot resolves IDs but `pg=`/`label=` may miss session-side
  labels — pass `fem=` (or `results.bind(fem)`) and target by name.
- **`.values` is `(T, N)` — index, don't iterate.** Last step is
  `values[-1]`; one node's history is `values[:, j]`. For a single-node
  PG it's still 2-D: `values[:, 0]`.
- **Sum reactions, don't average.** A support reaction is a force per
  node; the support's total reaction is `.values.sum(axis=1)` over the
  base PG (often negated to compare against applied load).
- **Target names, never raw tags.** `pg="Base"` is stable across remesh
  and boolean ops; a hard-coded node tag is not.

## Read a result straight after the run

You do not always need a results file. After a live `ops.analyze(...)`, the bridge can answer
from the domain it just solved. `ops.nodes.one(label=...)` picks the one node a geometry label
names (it raises if the label picks none or several), and `.read(component)` returns a float in
model units, at the last committed state:

```python
tip_ux = ops.nodes.one(label="tip").read("displacement_x")

base = ops.nodes.get(pg="Base")                  # a NodeSet: several nodes
base_shear = base.read("reaction_force_x").sum()  # ndarray (N,), summed with numpy
```

Elements read the same way, with `ops.elements.get(pg=...)` or `label=` or `ids=`. An element
`.read` answers only components that hold one value per element, which today means
`"axial_force"` on `Truss` and `CorotTruss`; it returns an array in the set's order.
Anything richer, such as beam end forces, raises and points you to `Results`.

`label=` is accepted wherever `pg=` is, on `ops.fix`, `ops.mass`, `ops.recorder.Node` and the
selectors above, so a model script can name a support or a master once, in the geometry, and
use that name to constrain it, record it and read it back. `recorder.Element(label=...)` works
under either tag mode: the label resolves to FEM element ids, and the build translates them to
the emitted OpenSees tags as it does for `pg=`. The archive keeps each label beside the ids it
resolved to, as provenance only: replaying a `model.h5` uses the stored ids, never the label.

A `.read` answers only from a domain this bridge solved live. After `ops.py(path, run=True)` or
`ops.tcl(path, run=True)` the solve happened in another process, so `.read` raises and names the
way out: `ops.results(path)` returns the same `Results` object as above, built from the
recorders you declared. A typed `ops.recorder.Node` must use `time_format="dt"` to be read
this way; without it the text file has no time column and `ops.results` refuses. A single value is `.item()` on a slab narrowed to one node and one step:

```python
tip_uy = ops.results("model.tcl").nodes.get(
    label="tip", component="displacement_y", time=-1).item()
```

`.item()` raises unless the slab holds exactly one value, which is the point: it replaces
`values[-1, 0]`, which answers silently from whatever the slab holds. Running a Tcl deck with
`run=True` also writes a driver next to the deck, named `<deck>.run.tcl`, so `model.tcl` gives
`model.tcl.run.tcl`. It sources the deck and
prints a completion marker, and the next run overwrites it.

## See also

- **Concept:** [Reading & filtering results](../concepts/results.md)
  — the full selector menu (named, geometric, time, stage), additive
  composition, the `.select()` chain, and the five slab shapes.
- **Concept:** [Obtaining the database](../concepts/results.md)
  — the five execution strategies that produce the file you read here,
  and how `reaction_force` is declared on each.
- **Visual:** for a kernel-safe view of the same results in a notebook,
  use `results.show_web()` (trame web viewer). `results.viewer()`
  defaults to `blocking=None` (auto); an explicit `blocking=True`
  still crashes the Jupyter kernel.
- **API:** [`apeGmsh.results.Results`](../api/results.md) — composite
  methods, the slab dataclasses, and `from_native` / `from_mpco` /
  `from_recorders` signatures.

---

*Next: [Get results via MPCO (STKO)](results-mpco.md).*
