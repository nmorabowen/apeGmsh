# apeGmsh examples

Each example lives in its own folder. Outputs (`outputs/capture.h5`,
`outputs/recorders/`, etc.) are written under that folder and are
git-ignored.

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
results_capture = OUT / "capture.h5"   # DomainCapture target
out_dir         = OUT / "recorders"    # .out files (Results.from_recorders)
```

`workdir()` returns `./outputs/` and creates it if missing — that
directory is per-example because each notebook lives in its own
folder.

## Results methods used here

| Method | When |
|---|---|
| `DomainCapture` (recommended) | Capture results during an in-process OpenSees run. Writes a single `capture.h5`. |
| `Results.from_recorders` | Run OpenSees with disk-backed `.out` recorders, then read them back. |
| `Results.from_native` | Open an existing apeGmsh native HDF5 file (e.g. one produced earlier). |

The manual `NativeWriter` path is intentionally not used in these
examples — it's a niche option for non-OpenSees solvers that want to
emit apeGmsh-format HDF5.

## Curriculum path order

See [`../CURRICULUM.md`](../CURRICULUM.md) for the full 20-slot
specification (titles, learning goals, verification criteria, prereqs).
The tier folders match that document.
