# Compose modules into one model

Stitch independently-built, **saved** `model.h5` parts into one larger FEM by
tag-offsetting and namespacing each module — no re-meshing, no re-running
geometry. Reach for this when you want to build a big assembly out of small,
reusable parts that each live in their own session and file.

## The recipe

Build each part once and `g.save(...)` it. Then reload one part as the **host**
in *chain phase* with `apeGmsh.from_h5(...)` (this skips the gmsh build entirely),
graft the rest in with `g.compose(...)`, bridge the interfaces with chain-phase
constraints, and save or emit the result.

```python
from apeGmsh import apeGmsh

# Host stays in "chain phase": FEM loaded straight from the neutral zone, no gmsh.
g = apeGmsh.from_h5("column.h5")          # host PGs keep their BARE names: "top", ...

# Graft saved modules in. `label` namespaces every PG/label the module brings.
g.compose("beam.h5", label="beam", translate=(0.0, 3.0, 0.0))
g.compose("beam.h5", label="beam2", anchor="mount_pad")   # `anchor` is translate sugar

# Bridge the interface. Host port is bare; composed-module port is "{label}.{pg}".
g.constraints.tied_contact(master_label="top", slave_label="beam.end")

g.save("frame.h5")                        # durable assembly...
# from apeGmsh import apeSees
# apeSees(g._fem).tcl("frame.tcl")        # ...or emit a solver deck
```

`g.compose(source, *, label, translate=(0,0,0), rotate=None, anchor=None,
partition_rank=None, properties=None, max_compose_depth=None,
compose_size_per_module=None)` returns a `ComposedModule` handle.

### Inspect without merging

```python
info = g.compose_inspect("beam.h5")   # metadata-only read — does NOT merge or mutate
info["pg_inventory"]                  # sorted PG names the module would bring in
info["neutral_schema_version"]        # e.g. "2.10.0"

g.compose_list()   # tuple[ComposedModule, ...] — modules composed into this session
g.compose_tree()   # tuple[ComposeTreeNode, ...] — nested-compose hierarchy
```

## Namespacing — `'{label}.{pg}'`

The `from_h5` host keeps every PG/label **bare**. Every composed module's PGs and
labels are prefixed with its `label` and a dot: a part PG `top` composed under
`label="beam"` becomes `beam.top`. When you wire interface constraints or query the
assembled FEM, reference the **bare** name for the host and the **namespaced**
name for composed modules.

## Notes / gotchas

- **`label` is strict (fails loud with `ComposeLabelError`):** non-empty, no `.`,
  no `/`, no whitespace, and may not start or end with `_`.
- **`anchor` and a non-zero `translate` are mutually exclusive** —
  `ComposeAnchorError`. Use one or the other.
- **Chain phase has no gmsh.** `g.compose / g.constraints.* / g.save` work;
  `g.model.geometry.*` and `g.mesh.generation.*` raise. Build geometry in the
  part's own session *before* saving.
- **`tied_contact` needs dim=2 element groups.** If the saved module has none, the
  chain-phase router raises a hard `ValueError` telling you to re-extract the source
  with `dim=None` before saving. (Interface constraints otherwise route silently;
  if a declared tie resolves nothing it's a no-op — confirm it landed by checking
  `len(list(g._fem.elements.constraints))` grew.)
- **Loads and masses follow the same contract** after composing: loads
  declared via `g.loads.*` are opt-in — import each case into a pattern with
  `p.from_model("<case>")` — and masses and support fixities are re-declared on
  the bridge. Because loads do not auto-emit, there is no double-count trap.
- **Nested compose caps at depth 3** (`max_compose_depth=`); beyond it raises
  `ComposeDepthExceededError`.

## Declarative alternative — `Assembly` + `couple`

For spatially coupling several saved modules, a higher-level declarative builder
ships from a **sub-path** (`apeGmsh.Assembly` is deliberately not exposed —
the top-level session *is* the assembly):

```python
from apeGmsh.assembly import Assembly

g = (
    Assembly("frame")
    .add("col", "column.h5")                              # first add = HOST (bare PGs)
    .add("beam", "beam.h5", translate=(0.0, 3.0, 0.0))    # composed under label "beam"
    .couple("col", "beam", kind="equal_dof",
            ports=("top", "end"), dofs=[1, 2, 3])         # name BARE per-part PGs
    .materialize()                                        # -> composed apeGmsh session
)
g.save("frame.h5")
```

`couple(kind=...)` supports `equal_dof` and `tied_contact`. It's a thin wrapper over
`from_h5` + `compose` + `g.constraints.*`, and fails loud (`AssemblyError`) if a
couple names an unknown part or ties nothing.

## See also

- **Concept:** [Parts & assembly](../internal_docs/guide_parts_assembly.md) — the
  multi-part mental model behind compose and the `Assembly` builder.
- **Concept:** [FEM broker](../internal_docs/guide_fem_broker.md) — the
  `save` / `from_h5` neutral-zone round-trip that compose builds on.
- **Example:** [Part assembly notebook](../examples/notebooks/10b_part_assembly.ipynb)
  — the in-session `g.parts.add` precursor; the
  [GitHub examples gallery](https://github.com/nmorabowen/apeGmsh/tree/main/examples)
  has the cross-session compose runs.
- **API:** `apeGmsh.from_h5`, `g.compose` / `g.compose_inspect` / `g.compose_list`
  / `g.compose_tree`, and `apeGmsh.assembly.Assembly`.
