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

`run.json` **SHOULD** carry the source provenance fields from
`scripts/_habitat.py::git_provenance()` — `model_sha` (HEAD when the case
ran) and `git_dirty` (uncommitted edits at run time):

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
