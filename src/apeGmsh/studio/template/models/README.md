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
+ `deck/` + `results/` + `logs/`. Promote lasting curves/stills into
`reports/` — `.apegmsh/` (Studio-generated) is not the archive.
