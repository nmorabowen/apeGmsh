# Running & reading: choose your path

You have a meshed model and a typed `apeSees(fem)` bridge. Two independent
decisions stand between you and a `results` object: **how to run** OpenSees,
and **how to read** what it wrote. This page is the fork. Pick a cell; the
`Results` query API is identical no matter which one you land on.

## The two axes

- **RUN** — *in-process* drives the live `openseespy` domain from your
  notebook (`ops.analyze(...)` in the same kernel); *export* writes a
  standalone `.tcl` / `.py` deck (`ops.tcl(...)` / `ops.py(...)`) you run
  elsewhere — a cluster, STKO, or a separate process.
- **READ** — how the run's output gets back into apeGmsh's label/query
  world: **native capture** (apeGmsh probes the domain and writes its own
  HDF5), **classic recorders** (OpenSees `.out`/`.xml` files), or **MPCO**
  (STKO's `.mpco` HDF5).

## The grid

|              | **Read: native capture** → `from_native` | **Read: classic recorders** → `from_recorders` | **Read: MPCO** → `from_mpco` |
|--------------|-------------------------------------------|-------------------------------------------------|------------------------------|
| **Run: in-process** (notebook) | **`spec.capture(...)`** — the default, broadest path. apeGmsh queries the live domain each `cap.step(t)` and writes native HDF5. Every topology level (nodes → gauss → fibers → layers → springs) **plus modal via `cap.capture_modes(n)`**. This is what the tutorials use. | **`spec.emit_recorders("out/")`** — classic recorders pushed into the live domain, no subprocess. Lightweight nodes / elements / gauss / line_stations. **No** fibers / layers; **modal raises** — use capture. Read one stage at a time with `stage_id=`. | **`spec.emit_mpco("run.mpco")`** — MPCO recorder in-process. Native fibers / layers / modal. **Requires an STKO-built openseespy**; vanilla builds raise at `__enter__`. |
| **Run: export** (cluster / external) | — *(capture needs a live domain; export decks can't carry it)* | **`ops.tcl(..., recorders=spec)`** / **`ops.py(...)`** — emit a deck, run OpenSees anywhere, parse the `.out`/`.xml`. Reproducible, check-in-able, cluster-friendly. | **`ops.tcl(..., recorders=spec, mpco=True)`** — one `recorder mpco` line; run under **STKO-loaded** OpenSees, parallel-aware. The STKO-shop production path. |

> Native capture is a *run-side* technique — it only exists in-process,
> so that cell of the export row is intentionally empty. Everything else
> has a home.

## When to reach for each

- **In-process + capture** (`spec.capture` → `from_native`): the default.
  Interactive, broadest coverage, modal handled natively, zero subprocess.
  Start here unless something pushes you off it. **The tutorials use this
  on purpose** — it is the least-surprise path.
- **In-process + recorders** (`spec.emit_recorders` → `from_recorders`):
  when you want plain-OpenSees recorder semantics in the notebook and only
  need nodes / elements / gauss / line_stations. Lighter than capture for
  very long runs.
- **In-process + MPCO** (`spec.emit_mpco` → `from_mpco`): when you have an
  STKO-built openseespy and want fibers / layers / modal written by the
  battle-tested STKO recorder — see [Get results via MPCO](results-mpco.md).
- **Export + recorders** (`ops.tcl`/`ops.py` → `from_recorders`): cluster
  jobs, reproducible decks, non-Python tooling — see
  [Export a standalone deck](export-script.md).
- **Export + MPCO** (`ops.tcl(mpco=True)` → `from_mpco`): big parallel runs
  in the STKO ecosystem — also covered in [results-mpco.md](results-mpco.md).

## The read side is identical

Whichever cell you picked, the run produced a file (or a directory of
recorder files) and a canonical `model.h5`. From there the query surface
does not change:

```python
from apeGmsh import Results
from apeGmsh.opensees import OpenSeesModel

# Constructors differ ONLY in which file each strategy wrote.
# All three REQUIRE the model broker — omitting it raises TypeError.
model   = OpenSeesModel.from_h5("model.h5")
results = Results.from_native("run.h5",  fem=fem, model=model)      # capture
# results = Results.from_recorders(spec, "out/", fem=fem, model=model)  # recorders
# results = Results.from_mpco("run.mpco", model_h5="model.h5")          # MPCO (path, not object)

# From here ON, the code is the same for every strategy — target by PG NAME.
disp  = results.nodes.get(pg="Top",  component="displacement_z")
sigma = results.elements.gauss.get(pg="Body", component="stress_xx")
for mode in results.modes:
    print(mode.mode_index, mode.frequency_hz)

# Notebook-safe viewer. results.viewer() defaults to blocking=True and
# CRASHES a Jupyter kernel — use show_web() instead.
results.show_web()
```

## Notes / gotchas

- **`model=` / `model_h5=` is required on every constructor.** `from_native`
  and `from_recorders` take the in-memory `model=` object; `from_mpco` takes
  `model_h5=` as a **path** (MPCO files carry no `/opensees/` zone). Omitting
  it raises `TypeError`. `from_recorders` additionally needs `fem=`.
- **Loads are opt-in (ADR 0051).** MP constraints auto-emit, but
  `g.loads.*` do **not**: import a load case into a bridge pattern with
  `p.from_model(case)` (or author one via `pat.load(...)`). Masses and support
  fixities/SPs are re-declared on the bridge (`ops.mass` / `ops.fix`).
- **`emit_recorders` can't do modal or fibers/layers.** Modal records raise
  at `__enter__`; fibers / layers warn-and-skip. Route those through
  `spec.capture` (native) or `spec.emit_mpco` (STKO build).
- **`emit_mpco` / `tcl(mpco=True)` need an STKO-built openseespy.** Vanilla
  distributions don't ship the MPCO recorder. If you don't have STKO's
  bundled Python, use native capture for the same fibers/layers/modal
  coverage.
- **`from_recorders` after `emit_recorders` needs `stage_id=`.** Per-stage
  files are prefixed `<stage>__`; pass `stage_id="gravity"` matching your
  `begin_stage` name, or the loader won't find them.

## See also

- Concept: [Obtaining results — the five strategies](../internal_docs/guide_obtaining_results.md)
  — the full A₁/A₂/A₃/B/C₁/C₂ breakdown with coverage tables and a decision
  flowchart this grid summarizes.
- How-to: [Export a standalone deck](export-script.md) (recorder export) ·
  [Get results via MPCO](results-mpco.md) (STKO).
- API: [`apeGmsh.results.Results`](../api/results.md) — `from_native`,
  `from_recorders`, `from_mpco`, the composite query surface, and slab shapes.
