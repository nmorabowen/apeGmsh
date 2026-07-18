# Save and reload a model

Persist a meshed model to a native `model.h5` and bring it back later
without re-running geometry or meshing. Reach for this to checkpoint a
long build, hand a model to a colleague, or resume into composition on
another day.

## The recipe

Give the session a `save_to=` path and apeGmsh autosaves the neutral zone
on context-exit. Reload it as a bare snapshot with `FEMData.from_h5`, or
as a resumable chain-phase session with `apeGmsh.from_h5`.

```python
from apeGmsh import apeGmsh, FEMData

# --- Save: autosave on exit -------------------------------------------
with apeGmsh(model_name="plate", save_to="plate.h5", overwrite=True) as g:
    g.model.geometry.add_box(0, 0, 0, 1.0, 1.0, 0.1, label="body")
    g.physical.add_volume("body", name="body")
    g.mesh.generation.generate(dim=3)
    g.save()                       # optional explicit checkpoint mid-session
# neutral zone is written here, in __exit__ (i.e. end()), before gmsh.finalize()

# --- Reload (A): a bare snapshot, integrity-checked -------------------
fem = FEMData.from_h5("plate.h5")  # raises on a tampered/corrupt file
print(fem.info.summary())          # query nodes/PGs/loads with no live gmsh

# --- Reload (B): a resumable chain-phase session (no gmsh) ------------
g2 = apeGmsh.from_h5("plate.h5")   # skips the gmsh build entirely
g2.compose("bolt.h5", label="bolt", translate=(0.5, 0.5, 0.0))
g2.save("assembly.h5")
```

Exact signatures (verified against `src/`):

```python
apeGmsh(..., *, save_to: str | Path | None = None, overwrite: bool = True)  # _core.py:84
apeGmsh.save(self, path: str | Path | None = None) -> Path                  # _core.py:251
apeGmsh.from_h5(cls, path, *, model_name=None, verbose=False) -> apeGmsh    # _core.py:142
FEMData.to_h5(self, path, *, model_name="", apegmsh_version="", ndf=0)      # FEMData.py:1603
FEMData.from_h5(cls, path, *, root="/") -> FEMData                          # FEMData.py:1548
```

## Notes / gotchas

- **Autosave fires on `end()`, not eagerly.** Nothing is written until
  `__exit__` / `g.end()` runs — if the process dies mid-build, the file
  never lands. Call `g.save()` explicitly when a checkpoint matters.
- **`save_to=None` (the default) disables autosave.** With a path set,
  `overwrite=False` against an existing file raises `FileExistsError`;
  `g.save()` with neither an argument nor a `save_to=` raises `RuntimeError`.
- **Autosave catches-and-warns on write failure** so gmsh still finalizes.
  A silently-warned failure can lose data — prefer an explicit `g.save()`
  when persistence is the point.
- **Neutral zone only.** `g.save()` / `FEMData.to_h5` write the
  solver-agnostic mesh (nodes, elements, PGs, labels, constraints, loads,
  masses, per-node ndf) — **no `/opensees/` zone**. To persist the
  OpenSees deck alongside it, use `apeSees(fem).h5("model.h5")`, which
  writes **both** the neutral and opensees zones.
- **`from_h5` is fail-loud.** `/meta/snapshot_id` is re-verified against
  the recomputed hash on read; a mutated neutral zone raises
  `MalformedH5Error`, a missing `/meta` raises `MalformedH5Error`, and a
  wrong schema major raises `SchemaVersionError`. No silent corruption.
- **Two different `from_h5` methods** — don't confuse them:
  `FEMData.from_h5` returns a snapshot you can query and emit;
  `apeGmsh.from_h5` returns a **chain-phase session** with *no gmsh
  state*, so `g.model.*` and `g.mesh.generation.*` will fail. Only
  `compose(...)`, `save(...)`, and the chain-phase interface
  constraints/loads/masses work there.
- **Composed/results files nest the layout.** A standalone `model.h5`
  reads at `root="/"`; a composed `results.h5` carries the same layout
  under `/model/`, so pass `FEMData.from_h5("results.h5", root="/model")`.
- **Bandwidth is not stored** — it is recomputed from connectivity on
  read, so never rely on a persisted value.

## See also

- **Concept:** [The FEM broker (`FEMData`)](../internal_docs/guide_fem_broker.md)
  — the immutable snapshot and the full native-persistence round-trip contract.
- **Tutorial:** [Save, reload, and view a model](../tutorials/save-reload-view.md)
  — the same workflow walked end to end.
- **How-to:** [Compose modules into one model](compose-modules.md)
  — graft saved `.h5` parts together via `apeGmsh.from_h5` + `g.compose`.
- **API:** [`FEMData` native persistence](../api/fem.md#native-persistence)
  and [Session persistence](../api/session.md#native-persistence).

---

*Next: [Compose modules into one model](compose-modules.md).*
