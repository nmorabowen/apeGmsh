# Session — `apeGmsh`

The top-level session object. Owns a single Gmsh kernel and wires
all composites (`model`, `mesh`, `parts`, `constraints`, `loads`,
`masses`, …). The OpenSees bridge is **not** a session composite —
import it explicitly via `from apeGmsh.opensees import apeSees`.

## Native persistence

The session can persist the **neutral zone** (the solver-agnostic
`FEMData` snapshot — nodes, elements, physical groups, labels,
loads, masses, constraints) to a native `model.h5`. Two write
paths are exposed on the session:

```python
# Autosave: write the neutral zone on context-manager exit
with apeGmsh(model_name="Tower", save_to="model.h5") as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.physical.add_volume("body", name="body")
    g.mesh.generation.generate(3)
# model.h5 now exists

# Manual: write at any point inside the session
with apeGmsh(model_name="Tower") as g:
    ...
    g.save("model.h5")        # explicit path
```

`apeGmsh(save_to=..., overwrite=True)` configures autosave at
construction; the file is written on `end()` / context exit.
`overwrite=False` makes a pre-existing target fail-loud on save.
`g.save(path=None)` writes immediately and returns the resolved
`Path`; with no argument it reuses `save_to`, and raises
`RuntimeError` if neither a path nor `save_to` was supplied.

Without `save_to`, `end()` still writes: `model.h5` goes to the
conventional path `<dir>/<model_name>.h5` (with `<dir>` the running
script's directory, or `$APEGMSH_ARTIFACT_DIR`), and the geometry
sibling `<model_name>.geometry.h5` beside it. The default `model_name`
is the running script's stem, so `python frame.py` leaves `frame.h5`
next to `frame.py`. In a notebook, under `python -c` or from stdin
there is no script: the session has no name, nothing is written
automatically, and one warning says so; pass `model_name=` or
`save_to=<file>`. A parameter sweep that wants to keep each run's
output sets `model_name` per run, because a later run of the same
script replaces the file, while a file another script wrote at that
path is kept (one warning). A notebook's cells are not scripts, so a
notebook replaces the file on every run: give each notebook its own
`model_name`. Under MPI only rank 0 writes automatically, and a mesh
partitioned with `g.mesh.partitioning` gets no automatic write; an
explicit `save_to=` is written regardless of either, and a composed model
is not partitioned for this purpose and writes.

Both paths write the **neutral zone only**. The OpenSees zone
(typed primitives, recorders, analysis chain) is written
separately by the bridge via `apeSees(fem).h5(path)` — see the
[OpenSees bridge](opensees.md) page.

### Chain-phase reassembly

`apeGmsh.from_h5(path, *, model_name=None, verbose=False)` rebuilds
a session **directly from a `model.h5`**, skipping the Gmsh build
entirely. The returned session is a *chain-phase* session: it has
no live kernel, so geometry/meshing verbs are unavailable, but it
can still `save` and feed the bridge.

```python
g = apeGmsh.from_h5("model.h5")        # no gmsh; loads the neutral zone
ops = apeSees(g.mesh.queries.get_fem_data(dim=3))
```

## Composition

Composing saved models is
[`Assembly`](../how-to/assemble-saved-models.md) (`from apeGmsh.assembly
import Assembly`, [ADR 0117](https://github.com/nmorabowen/apeGmsh/blob/main/architecture/decisions/0117-assembly-compose-v2.md)):
`instance` places a saved `model.h5` under a namespaced label, the tie
and coupling verbs join instances by dotted port, and `bridge` builds
one `apeSees`. The session-level `g.compose` and the v1 `Assembly`
verbs (`add`, `couple(part_a, part_b, ports=)`, `materialize`) were
removed in AS5-c.

The session keeps the compose **readers**, which open any composed
file, an assembly archive included:

```python
g = apeGmsh.from_h5("stack.h5")      # an archive written by asm.h5(...)
g.compose_list()                     # -> (ComposedModule, ...)
g.compose_tree()                     # nested-compose hierarchy
g.compose_inspect("pier.h5")         # header of a file, without composing it
```

`g.compose_inspect(path)` returns a dict (`fem_hash`,
`neutral_schema_version`, `tag_span_max`, `pg_inventory`,
`label_inventory`, `record_counts`, `compose_tree`, …).
`g.compose_list()` enumerates the modules composed into the file the
session holds; `g.compose_tree()` returns their nested-compose
hierarchy. In the viewer, composed parts are colourable by the
string-keyed Module modes (`'Module'`, `'Module: Root'`,
`'Module: Leaf'`).

## Package

::: apeGmsh

## Session class

::: apeGmsh._core.apeGmsh

## Base

::: apeGmsh._session._SessionBase
