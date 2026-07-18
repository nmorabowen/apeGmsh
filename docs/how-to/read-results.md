# Read a node's displacement and reactions

Pull a displacement or reaction-force time history out of a finished run,
targeting nodes by physical-group name. This is the bread-and-butter
post-processing read: open the results file, ask one composite for one
component, index `.values`.

The shape in one line:

> `Results.from_native(path, model=...)` → `results.nodes.get(pg=,
> component=)` → a frozen slab with `.values` `(T, N)`, `.node_ids`,
> and `.time`.

## Recipe

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel

# Every Results constructor REQUIRES the read-side broker (model= /
# model_h5=); omitting it raises TypeError (ADR 0020 INV-1). Pass fem=
# too so pg=/label= name-resolution works against your session mesh.
model = OpenSeesModel.from_h5("model.h5")
results = Results.from_native("run.h5", model=model, fem=fem)

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
  file if you asked for it on the write side — e.g.
  `spec.nodes(components=["displacement", "reaction_force"], pg="Base")`
  (capture) or the equivalent recorder declaration. If the read comes
  back empty, the component was never captured. Call
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

## See also

- **Concept:** [Reading & filtering results](../internal_docs/guide_results_filtering.md)
  — the full selector menu (named, geometric, time, stage), additive
  composition, the `.select()` chain, and the five slab shapes.
- **Concept:** [Obtaining the database](../internal_docs/guide_obtaining_results.md)
  — the five execution strategies that produce the file you read here,
  and how `reaction_force` is declared on each.
- **Visual:** for a kernel-safe view of the same results in a notebook,
  use `results.show_web()` (trame web viewer); avoid the default blocking
  `results.viewer()`, which crashes the Jupyter kernel.
- **API:** [`apeGmsh.results.Results`](../api/results.md) — composite
  methods, the slab dataclasses, and `from_native` / `from_mpco` /
  `from_recorders` signatures.

---

*Next: [Get results via MPCO (STKO)](results-mpco.md).*
