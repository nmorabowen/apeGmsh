# apeGmsh examples

Each example lives in its own folder. Outputs (`outputs/model.h5`,
`outputs/run.ladruno` or `outputs/run_out/`, etc.) are written under that
folder and are git-ignored.

```
EOS Examples/
├── curriculum/        ← 20-slot structured learning path (start here)
│   ├── 01-fundamentals/
│   ├── 02-building-blocks/
│   ├── 03-assemblies/
│   ├── 04-coupling/
│   └── 05-analysis-types/
├── case_studies/      ← deeper real-world demos (chevron brace, footing, LTB, …)
├── geometry/          ← pure CAD/meshing demos, no analysis
└── archive/           ← historical / lower-priority notebooks kept for reference
```

## How an example writes outputs

```python
from apeGmsh import workdir
OUT = workdir()                        # creates ./outputs/
model_h5 = OUT / "model.h5"            # the bridge's archive (ops.h5)
run_path = OUT / "run"                 # emit_default sets the suffix
```

`workdir()` returns `./outputs/` and creates it if missing — that
directory is per-example because each notebook lives in its own
folder.

## Results methods used here

| Method | When |
|---|---|
| `ResolvedRecorderSpec.emit_default` (recommended) | Record an in-process OpenSees run on the route the build supports: `run.ladruno` on the Ladruno fork, `run.mpco` on a stock build with MPCO, classic `.out` files in `run_out/` on pip openseespy. Read it back with `run.results()`. |
| `EigenResult.results()` | Open the modes of a bridge eigen solve as a `Results`, one stage per mode (`to_native(path)` keeps the file). An `EigenResult` comes only from `apeSees.eigen(n)` on a model built through the bridge. |
| `capture_live_modes(model_h5, n, path)` | The modes of a plain openseespy deck: solves `ops.eigen(n)` on the live domain and writes one native stage per mode, read with `Results.from_native` (notebook 17). |
| `Results.from_recorders` | Run OpenSees with disk-backed `.out` recorders, then read them back. |
| `Results.from_native` | Open an existing apeGmsh native HDF5 file (e.g. the modes file `to_native` wrote). |

The manual `NativeWriter` path is intentionally not used in these
examples — it's a niche option for non-OpenSees solvers that want to
emit apeGmsh-format HDF5.

## Curriculum path order

See [`../CURRICULUM.md`](../CURRICULUM.md) for the full 20-slot
specification (titles, learning goals, verification criteria, prereqs).
The tier folders match that document.
