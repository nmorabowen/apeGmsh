# Export to a Tcl or openseespy script

Write a standalone, runnable OpenSees deck from the typed bridge instead of
solving in-process. Reach for this when you want to version-control the model,
hand it to a collaborator, or run it under a different OpenSees binary.

## The recipe

Build the model through `apeSees(fem)` exactly as you would for an in-process
run, then call `ops.tcl(path)` and/or `ops.py(path)` instead of
`ops.analyze(...)`:

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

L, E = 3.0, 200e9
b, h = 0.10, 0.20
A, Iz = b * h, b * h**3 / 12.0
P = 10_000.0

# --- 1. Geometry + named physical groups ---
with apeGmsh(model_name="cantilever") as g:
    p0   = g.model.geometry.add_point(0.0, 0.0, 0.0)
    p1   = g.model.geometry.add_point(L,   0.0, 0.0)
    beam = g.model.geometry.add_line(p0, p1)
    g.model.sync()

    g.physical.add(1, [beam], name="Beam")
    g.physical.add(0, [p0],   name="Fixed")
    g.physical.add(0, [p1],   name="Tip")

    g.mesh.sizing.set_global_size(L / 10.0)
    g.mesh.generation.generate(1)
    fem = g.mesh.queries.get_fem_data(dim=1)

# --- 2. Declare the model on the typed bridge ---
ops = apeSees(fem)
ops.model(ndm=2, ndf=3)

transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
ops.element.elasticBeamColumn(pg="Beam", transf=transf, A=A, E=E, Iz=Iz)
ops.fix(pg="Fixed", dofs=(1, 1, 1))

with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
    pat.load(pg="Tip", forces=(0.0, -P, 0.0))

# --- 3. Emit instead of solving ---
ops.tcl("cantilever.tcl")   # OpenSees Tcl deck
ops.py("cantilever.py")     # equivalent openseespy script
```

Each call writes a complete, self-contained model definition: the model
builder, every node, the materials / sections / transforms, element
connectivity (with physical-group comments), `fix` commands, nodal masses,
load patterns, and any MP constraints (`equalDOF`, `rigidLink`,
`rigidDiaphragm`, `ASDEmbeddedNodeElement`). Run the result with
`opensees cantilever.tcl` or `python cantilever.py`.

## Notes / gotchas

- **The deck has no analysis chain.** `ops.tcl` / `ops.py` emit the *model* —
  not `constraints` / `numberer` / `system` / `integrator` / `analysis` /
  `analyze`. Append your solver recipe (or that of your collaborator) to the
  emitted file. To bake an `analyze` line into the deck, pass
  `analyze_steps=` (and optionally `analyze_dt=`).
- **This is the alternative to in-process capture.** In a notebook you'd run
  `ops.run()` / `ops.analyze(...)` and read results back through `Results`.
  Exporting decouples *declaring* from *running*: the model leaves your Python
  session as a plain text file.
- **Emit calls are separate statements, not a fluent chain.** Write each on
  its own line — `ops.tcl(...)` then `ops.py(...)`. Each builds the model
  internally (an implicit `ops.build()`), so order between them doesn't matter.
- **`run=True` subprocesses the deck for you.** `ops.tcl("m.tcl", run=True)`
  shells out to an `opensees` binary (override with `bin=`); `ops.py(..., run=True)`
  runs the script under Python. Without `run=`, the call only writes the file.
- **Loads are opt-in.** `g.loads.*` do **not** auto-emit (ADR 0051): bring a
  session load case into the deck with `p.from_model(case)` inside a pattern,
  or author one directly with `pat.load(...)`. A declared case that no pattern
  imported triggers `WarnUnconsumedModelLoads` at build.
- **For a runnable native HDF5** (deck zone *plus* the broker neutral zone the
  viewer / `Results` read), use `apeSees(fem).h5(path)` — the session-side
  `g.save()` / `fem.to_h5()` write the neutral zone only and are not runnable
  decks.

## See also

- Concept: [OpenSees bridge guide](../internal_docs/guide_opensees.md) — §6 covers
  `ops.tcl` / `ops.py` / `ops.h5` / `ops.run`, the build step, and the deck
  contents in depth.
- Tutorial: [Your first model](../tutorials/first-model.md) — builds the same
  cantilever and solves it in-process (the path this recipe replaces).
- Related: [Run a static analysis](index.md#solve-the-opensees-bridge) — the in-process counterpart.
