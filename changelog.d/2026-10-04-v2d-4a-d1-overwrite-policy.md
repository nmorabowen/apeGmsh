### CHANGED — the D1 overwrite policy: the session's default name is the script's stem, and one rule decides what the automatic write may replace (ADR 0112 D1, program slice V2d part 4a, #1307)

`apeGmsh(model_name=None)` is the new default: the session is named after
the `__main__` script, so `python frame.py` leaves `frame.h5` and
`frame.geometry.h5` beside it. With no real script (a notebook, `-c`,
stdin) the session has no name, `end()` writes nothing automatically and
warns once; `model_name=` always wins, `save_to=<file>` needs no name, and
an empty `model_name` is refused (the old default `"ModelName"` is gone).
`FEMData.model_name` carries the session's name: `from_gmsh` stamps it,
`FEMData.to_h5` writes it to `/meta/model_name` by default, `from_h5`
reads it back, derived copies inherit it, and no hash reads it. The
ownership check that `end()` applied since V2b (#1305) is now the module
rule `apeGmsh._artifact_policy.artifact_target_is_ours`, which the
session delegates to and the bridge's automatic write (part 4b) will share.
Its new rows: a file from this run (equal `session_id`) that holds a zone
the write would drop — the bridge's neutral + `/opensees` — is kept
**silently** when its `snapshot_id` equals the snapshot's (`fem_hash`,
lineage INV-1) and with one "stale" warning when the model changed after
it was written; a file from another run is replaced only when its
`/provenance` names a script this run's provenance also names, or names no
script, and is kept with one warning otherwise (a parameter sweep that
keeps each run's output sets `model_name` per run; `save_to=` is exempt).
A foreign file, `overwrite=False` and an older file holding a zone the
write would drop keep V2b's warnings. Under MPI only rank 0 writes
(`OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `PMIX_RANK`, `MV2_COMM_WORLD_RANK`,
`SLURM_PROCID`); a partitioned run — a composed model included, since
ADR 0038 gives each module a partition — gets no automatic write and one
warning, `save_to=` still writing. `fem_hash` reads nodes, elements,
groups, labels and `composed_from`, so a load, mass or constraint declared
after the bridge's emit does not make the bridge's file stale; the gap is
pinned by a test. A bench case (`tests/benchmarks/test_artifact_policy_cost.py`)
measures the hash against the write it gates.
