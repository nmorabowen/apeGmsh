# FEM Broker — `FEMData`

Solver-agnostic snapshot returned by `g.mesh.queries.get_fem_data(dim)`.

## Native persistence

`FEMData` round-trips to a native `model.h5` (the **neutral zone**)
without any solver in the loop:

```python
fem = g.mesh.queries.get_fem_data(dim=3)
fem.to_h5("model.h5", model_name="Tower")

restored = FEMData.from_h5("model.h5")     # integrity-checked load
```

`to_h5(path, *, model_name="", apegmsh_version="", ndf=0)` writes
the snapshot; `from_h5(path, *, root="/")` reads it back. `root=`
selects a non-root group when the neutral zone lives nested inside
a larger file (e.g. the two-zone canonical file written by the
bridge). The read is **integrity-checked**: a `snapshot_id`
mismatch between the written and re-derived content raises
`MalformedH5Error`.

The neutral zone is written at schema `NEUTRAL_SCHEMA_VERSION`
(defined in `mesh/_femdata_h5_io.py`); the OpenSees zone written by
the bridge carries its own `SCHEMA_VERSION`. Each zone has a
compatibility floor, `NEUTRAL_SCHEMA_FLOOR` and `SCHEMA_FLOOR`: a
reader opens every file from the floor up to its own version and
refuses anything older or newer
([ADR 0113](https://github.com/nmorabowen/apeGmsh/blob/main/architecture/decisions/0113-compatibility-is-a-floor-per-zone.md)).

This is the same neutral zone the session writes via
`apeGmsh(save_to=...)` / `g.save()` — see the
[Session](session.md#native-persistence) page. `FEMData.from_h5`
is also the entry point `apeGmsh.from_h5` and `g.compose` build on
for chain-phase reassembly.

::: apeGmsh.mesh.FEMData
