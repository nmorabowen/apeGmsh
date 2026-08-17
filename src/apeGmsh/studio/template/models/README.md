# models/ — lineage contract

One model per `models/<id>/`:

```text
models/
└── <id>/
    ├── src/             # scripts that own geometry + analysis
    └── cases/            # one folder per run
        └── <case>/
            ├── run.json  # case manifest
            ├── deck/     # exported Tcl / openseespy deck
            ├── results/  # recorder outputs, profile.h5, curves
            └── logs/     # run logs
```

`src/` owns idealization → geometry → analysis; scripts, not notebooks, are
the model language. `cases/<case>/` is one run: it **MUST** write `run.json`
+ `deck/` + `results/` + `logs/`. `deck/` holds the emitted deck **or** a
one-line README naming why none exists (an in-process openseespy run emits
nothing — disclosure beats fabrication). Promote lasting curves/stills into
`reports/` — `.apegmsh/` (Studio-generated) is not the archive.

Cases **SHOULD** be run through the habitat's runner, which owns the
layout, the logs, the deck disclosure, and the `run.json` record:

```text
python scripts/run_case.py --model <id> --case <case> \
    --script models/<id>/src/.../driver.py [--verify .../verify.py]
```

The model script runs with cwd = `results/` (outputs land in place; the
optional verify reads the same cwd). A failed run is a recorded run; an
existing `run.json` is refused, never overwritten — new question, new
case id. Richer oracle blocks (named metrics with tolerances) are added
to `run.json` by hand after the run.

`run.json` **SHOULD** carry the source provenance fields from
`scripts/_habitat.py::git_provenance()` — `model_sha` (HEAD when the case
ran) and `git_dirty` (uncommitted edits at run time). The runner records
them at launch; a hand-rolled case adds them itself:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from _habitat import git_provenance

manifest.update(git_provenance() or {})
```

In an un-versioned habitat `git_provenance()` returns `None` — omit the
fields. A `git_dirty: true` run is disclosed in the stage report, not
rejected. See `APE/instructions/checkpoints.md`.
