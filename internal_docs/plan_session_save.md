# Design Note: Session Autosave (`save_to`)

> **Status:** Implemented — Phase 1 (autosave) and Phase 2 (full
> `FEMData.from_h5` round-trip) shipped together. See
> [tests/test_session_save.py](../tests/test_session_save.py) and
> [tests/test_femdata_from_h5.py](../tests/test_femdata_from_h5.py).

## Problem

`g.mesh.queries.get_fem_data()` is a query and never touches disk. To
persist a model the user must explicitly call `fem.to_h5(path)` after
extracting the snapshot. There is no way to checkpoint at session
close, and no symmetric reload entry point on the session itself.

Use cases that motivate this:
- Build the mesh in one script, run analysis in another.
- Cache the broker so a long meshing job is not re-run on each tweak.
- Resume work in a later session.

## Non-goals

- **Versioned checkpoints** (`lhc.h5.0001`, ...). Out of scope.
- **OpenSees enrichment in autosave.** OpenSees is a downstream
  consumer of `FEMData`. It produces its own artifacts (`tcl`, `py`,
  enriched `h5`) via explicit user calls and must never feed back into
  session lifecycle. The session writes the neutral zone only.
- **Hooking autosave to `get_fem_data()`.** Queries do not mutate disk.

## Design

### Constructor flag

```python
apeGmsh(model_name="lhc",
        save_to="lhc.h5",      # path → autosave on end()
        overwrite=True)         # else raise on existing file
```

`save_to=None` (default) leaves behavior unchanged.

### Lifecycle hook

`_SessionBase.end()` runs the save **before** `gmsh.finalize()` (the
write needs the live model), wrapped so a save failure still finalizes
gmsh:

```python
def end(self):
    if self._active:
        if self._save_to:
            try:
                self._do_save(self._save_to)
            except Exception as exc:
                # log; do not re-raise — gmsh must still finalize
                ...
        gmsh.finalize()
        self._active = False
```

`__exit__` already calls `end()`, so `with apeGmsh(..., save_to=...)`
gets autosave for free.

### `_do_save` — one call, no branching

```python
def _do_save(self, path):
    fem = self.mesh.queries.get_fem_data()       # all dims
    fem.to_h5(path, model_name=self.name, ...)
```

The session knows nothing about who else might want to write into the
file. OpenSees, future LS-DYNA, etc. are separate user-driven actions
on their own composites — never invoked from session lifecycle.

### Manual checkpoint

```python
g.save()              # uses save_to
g.save("alt.h5")      # ad-hoc path; does not change save_to
```

Same `_do_save` underneath. `g.save()` without `save_to` set raises.

### Resume — symmetric, separate entry point

```python
g = apeGmsh.from_h5("lhc.h5")
```

Two flavors possible (pick one — see open questions):

| Flavor | Behavior | Cost |
|---|---|---|
| **Frozen broker** | Reconstruct `FEMData` only. No live gmsh model. Further `g.model.*` calls raise. | ~1-2 days. Reuses `FEMData.from_native_h5`. |
| **Live rehydration** | Reconstruct gmsh model from stored geometry/mesh, rebuild composites. Full `g.model.*` available. | Significantly larger lift; needs round-trip mesh import. |

Frozen broker covers the analysis-script use case. Live rehydration
covers "open the model and add a new boundary condition," which is a
much rarer ask.

## Resolved decisions (open questions from the original plan)

1. **`get_fem_data(dim=None)` for autosave** — adopted. The all-dim
   path round-trips fine in tests; the writer handles per-type element
   subgroups, and the reader reconstructs them by code.
2. **Resume flavor** — `FEMData.from_h5(path)` only. No frozen-session
   wrapper, no live rehydration. Symmetric with `FEMData.to_h5(path)`
   and `FEMData.from_msh(path)`. Live-session rehydration stays
   deferred (low demand; cost would be 5x+ the current scope).
3. **`overwrite=False`** — raises `FileExistsError`. Explicit > implicit.
4. **Save failures** — log a `UserWarning` and continue to
   `gmsh.finalize()`. The user still sees the traceback in the warning.

## Implementation (as shipped)

| File | Change |
|---|---|
| [src/apeGmsh/_session.py](../src/apeGmsh/_session.py) | `end()` calls `_do_save(self._save_to)` (via `getattr`, so `Part` is unaffected) before `gmsh.finalize()`. Failures emit `UserWarning` and proceed. |
| [src/apeGmsh/_core.py](../src/apeGmsh/_core.py) | `apeGmsh.__init__` accepts `save_to`, `overwrite`. `g.save(path=None)` and `_do_save(path)` methods. |
| [src/apeGmsh/mesh/FEMData.py](../src/apeGmsh/mesh/FEMData.py) | New classmethod `FEMData.from_h5(path)` delegating to `read_fem_h5`. |
| [src/apeGmsh/mesh/_femdata_h5_io.py](../src/apeGmsh/mesh/_femdata_h5_io.py) | New `read_fem_h5(path) -> FEMData` plus per-record decoders for all 5 constraint types, 3 load types, and masses. Mirror of `write_fem_h5`. |
| [tests/test_session_save.py](../tests/test_session_save.py) | 9 tests — autosave, manual save, overwrite, finalize-after-failure. |
| [tests/test_femdata_from_h5.py](../tests/test_femdata_from_h5.py) | 17 tests — round-trip every record type + error paths + session integration. |

### Round-trip coverage

Nodes / elements / PGs / labels / mesh selections / `NodePairRecord` /
`NodeGroupRecord` / `NodeToSurfaceRecord` / `InterpolationRecord` /
`SurfaceCouplingRecord` (mortar operator) / `NodalLoadRecord` /
`ElementLoadRecord` / `SPRecord` / `MassRecord` / `snapshot_id`.

### Known round-trip gaps (writer-side, by design)

- `SurfaceCouplingRecord.slave_records` — derivable from
  `master_nodes` / `slave_nodes` / `mortar_operator`; not stored.
- `NodeToSurfaceRecord.{rigid_link_records, equal_dof_records}` —
  derivable from `master_node` / `slave_nodes` / `phantom_nodes`;
  not stored.

Both come back as empty lists after `FEMData.from_h5(...)` and are
re-derived by the resolver when needed.

## What this does **not** preclude

- A later `g.opensees.export.h5(path)` call still works exactly as
  today — it overwrites the neutral file with an enriched version, on
  the user's explicit instruction.
- Versioned checkpoints, multi-stage saves, and partial saves can be
  added later as opt-in flags without breaking this baseline.
