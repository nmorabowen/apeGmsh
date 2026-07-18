# Results

This page teaches you what a `Results` object is — the container that turns
a finished run's output files back into the named, queryable world you built
the model in — and why one read API covers every way the run was recorded.

The [bridge page](opensees-bridge.md) ended on a symmetry: the same `"Tip"`
name that placed the load fetches the deflection back out. `Results` is
where that symmetry lives. It is a post-processing container bound to a
model: it mirrors the `FEMData` composite shape — `results.nodes`,
`results.elements.gauss`, and friends — and speaks the same
`pg=` / `label=` / `ids=` selection vocabulary, so once you know how to
query a mesh you already know how to query results on that mesh. The
container itself owns no arrays; it reads lazily from a backing file
(native HDF5, MPCO, or transcoded recorder output), which is why opening a
multi-gigabyte run is instant and why `Results` has a close-when-done
lifecycle rather than a load-everything constructor.

## One container, three doors

A run can leave its numbers on disk in three formats, and each has a
constructor. Which one you use is decided entirely by *how the run was
recorded* — the object you get back is the same class with the same query
surface:

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel

# All three REQUIRE the model broker — omitting it raises TypeError.
model   = OpenSeesModel.from_h5("model.h5")
results = Results.from_native("run.h5",  fem=fem, model=model)          # domain capture
results = Results.from_recorders(spec, "out/", fem=fem, model=model)    # classic .out files
results = Results.from_mpco("run.mpco", model_h5="model.h5")            # STKO MPCO
```

`from_native` opens apeGmsh's own HDF5, written by domain capture.
`from_recorders` parses classic OpenSees `.out`/`.xml` files against the
recorder spec that declared them, transcodes them once into a cached native
file, and opens that — re-reads with unchanged inputs hit the cache.
`from_mpco` reads STKO's `.mpco` HDF5 directly, including multi-partition
parallel runs (pass one `run.part-0.mpco` and siblings are auto-discovered
— the [MPCO how-to](../how-to/results-mpco.md) is the end-to-end recipe).

Every constructor **requires the model broker** — `model=` as an in-memory
`OpenSeesModel` for `from_native`/`from_recorders`, `model_h5=` as a *path*
to the canonical `model.h5` for `from_mpco` (MPCO files carry no solver
zone of their own). Omit it and you get a `TypeError` at the call site, not
a broken object later. That's deliberate: a results file is arrays plus IDs,
and only through the model — `results.model`, chaining to
`results.model.fem` — do those IDs become the mesh, the labels, and the
physical groups you named. Pass `fem=` (or call `results.bind(fem)`) to
bind your session-side `FEMData` when you have it; the snapshot embedded in
a native file resolves IDs fine, but the session-side one carries the
richer labels, Parts, and mesh-selection sets that make name-based queries
work everywhere. No hash check ties the two together — pairing a `FEMData`
with a results file from the same run is your responsibility, though
`results.lineage` gives you a tamper-evident `fem → model → results` hash
chain that *warns* on drift (and `lineage.assert_clean()` when you want
drift to be fatal).

Which door you end up at is the tail end of a decision you made on the
write side — in-process capture, classic recorders, or MPCO, each crossed
with running in-process or exporting a deck. That choice has its own pages:
the [strategy grid](../how-to/choose-results-strategy.md) walks the
trade-offs, and the [three-strategies example](../examples/results-strategies.md)
proves the point of this section by recording one model several ways and
watching the identical read code return the identical number.

## Stages, time, and modes

A results file is not one flat time history. Real analyses come in
segments — gravity, then pushover; static, then transient — and the file
keeps them apart as **stages**, each with a name, a `kind` (`"static"`,
`"transient"`, `"mode"`), and its own time axis. A freshly opened `Results`
spans all stages; when the file holds exactly one, reads resolve to it
automatically, and when it holds several you pick:

```python
results.stages                         # list[StageInfo]

gravity = results.stage("gravity")     # stage-scoped Results
sigma = gravity.elements.gauss.get(component="stress_xx", pg="Body")
gravity.n_steps, gravity.time          # stage metadata as properties
```

A stage-scoped `Results` is the same object, narrowed — every composite and
query works on it unchanged. (Passing `stage="gravity"` per read works too;
scoping once just reads better.) Within a stage, `time=` slices the step
axis on any read: an int indexes steps (`time=-1` is the last step), a list
picks specific steps, a float asks for the nearest time value, and a
`slice(a, b)` windows over time values.

Eigenmodes are stages too — `kind="mode"`, one per mode, each carrying its
eigenvalue, frequency, and period as stage attributes. The `.modes`
accessor hands them back as mode-scoped `Results`:

```python
for mode in results.modes:
    print(mode.mode_index, mode.frequency_hz, mode.period_s)
    shape = mode.nodes.get(component="displacement_z")   # (1, N) — one "step"
```

A mode shape is just a one-step nodal field, so everything you know about
reading displacements applies verbatim — the
[modal-analysis example](../examples/modal-analysis.md) runs this end to end.

## Reading fields — the same vocabulary you wrote with

Reads go through a composite tree that mirrors the FEM broker: `results.nodes`
for nodal fields, `results.elements` for per-element-node forces, and
sub-composites `gauss`, `line_stations`, `fibers`, `layers`, and `springs`
for integration-point, beam-station, fiber, shell-layer, and spring-level
data. Every level has the same `.get(...)`:

```python
disp = results.nodes.get(pg="Top", component="displacement_z")
disp.values      # ndarray (T, N) — one column per node in "Top"
disp.node_ids    # matching IDs, same column order
disp.time        # (T,) time axis

sigma = results.elements.gauss.get(pg="Body", component="stress_xx")
```

What comes back is a frozen **slab**: a `values` array whose first axis is
always time, plus the location metadata that says what each column is —
`node_ids` on a `NodeSlab`, element indices and natural coordinates on a
`GaussSlab`, fiber positions on a `FiberSlab`, and so on. Slabs are plain
data; from here it's numpy (`.values.sum(axis=1)` for a total base
reaction, `.values[:, 0]` for one node's history).

The selectors are the ones you already know. `pg=`, `label=`, and
`selection=` resolve physical groups, geometry-time labels, and
mesh-selection sets against the bound `FEMData` — this is why binding
matters — and multiple named selectors union. `ids=` is the surgical
override when you've computed IDs yourself. On top of the named selectors
sit spatial helpers — `nearest_to(point)`, `in_box(lo, hi)`,
`in_sphere(center, r)`, `on_plane(point, normal, tol)` — for the queries no
name covers: the node nearest a target coordinate, a story-level cut, a
slice through mid-span. The two families compose additively: named
selectors define the candidate set, the spatial helper narrows it, and an
empty intersection is a zero-row slab, not an error. The same composition
is available as a fluent chain via `.select()`, the results-side twin of
the [selection chain](selection.md) you used on geometry:

```python
slab = (results.nodes.select(pg="Base")       # candidate set
          .in_box(lo, hi)                     # narrow spatially
          .get(component="reaction_force_z")) # terminal read
```

That is the entire read model: pick a composite for the topology level,
name the *where* with the selectors, name the *what* with `component=`,
slice the *when* with `time=` or a stage scope. The component vocabulary
(`displacement_z`, `reaction_force_x`, `stress_xx`, `von_mises_stress`,
section forces, fiber stresses, …) and the full slab shapes are reference
material, not concepts — they live in the
[Results API reference](../api/results.md), and the
[read-results how-to](../how-to/read-results.md) is the focused recipe for
the everyday displacement-and-reaction pull.

Because the container is lazy over an open file handle, close it when
you're done — `results.close()`, or open it in a `with` block. On Windows
especially, an open handle blocks a capture script from re-creating the
same file on the next run.

## Knowing what's in a file

A results file behaving unexpectedly is almost always a vocabulary or
coverage question — you're asking for a component the write side never
recorded, or in a stage that doesn't carry it. The container can tell you.
`print(results)` (equivalently `results.inspect.summary()`) lists stages,
kinds, step counts, and available components;
`results.nodes.available_components()` — and the same call on every
composite — enumerates exactly what that level holds; and when a specific
component comes back empty, `results.inspect.diagnose("stress_xx")` prints
a per-level routing report showing where the component lives or why it's
missing. Print first, query second.

## Seeing it

The same object drives the viewers. `results.viewer()` opens the
interactive Qt/VTK viewer — but note that it is **blocking and in-process
by default**, which is fine from a terminal script and fatal to a Jupyter
kernel. In a notebook use the kernel-safe web viewer instead, or spawn the
Qt viewer as a subprocess:

```python
results.show_web()                # kernel-safe web viewer, inline in the notebook
results.show_web(stage="gravity") # activate a specific stage
results.viewer(blocking=False)    # Qt viewer in a subprocess; kernel keeps running
results.serve_web()               # standalone web app at a local URL
```

`show_web()` renders through a `pyvista.trame` backend with a step slider
and per-layer controls; `serve_web()` serves the same view as a standalone
page outside Jupyter. Both need the viewer extra
(`pip install "apeGmsh[viewer]"`). And if you just want to try them with
nothing solved yet, `Results.demo().show_web()` renders a built-in
cantilever-pushover sample. The viewers get a fuller treatment in the
[save-reload-view tutorial](../tutorials/save-reload-view.md) and the
[viewers API reference](../api/viewers.md).

That closes the loop the mental model opened: names go in at geometry time,
survive meshing, drive the solver, and come back out of `Results` — one
vocabulary from the first `label=` to the last slab.

---

*Next: [back to the learning path](../tutorials/learning-path.md).*
