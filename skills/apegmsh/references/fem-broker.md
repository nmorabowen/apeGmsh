# FEMData — the solver broker + native persistence
<!-- skill-freshness: verified against apeGmsh main@8eeda7a3 (2026-07-06) · if weeks old, re-verify signatures in src/apeGmsh/ before trusting exact tags/signatures -->

`FEMData` is the **immutable, solver-agnostic snapshot** apeGmsh hands to
any FEM backend. Once you have `fem = g.mesh.queries.get_fem_data(...)`
you no longer need a live Gmsh session to query the mesh — the snapshot
is self-sufficient, and it round-trips to a native `model.h5` (Part B).

All signatures below are read from `src/apeGmsh/mesh/`.

---

# Part A — the snapshot

## Construction

```python
# From a live session (usual path). After generate(), before partition.
fem = g.mesh.queries.get_fem_data(dim=3)               # verified: tests/test_femdata_to_h5.py::test_to_h5_snapshot_id_in_meta
fem = g.mesh.queries.get_fem_data()                    # dim=None → every dim present
fem = g.mesh.queries.get_fem_data(dim=3, remove_orphans=True)

# Direct factory (equivalent), from src/apeGmsh/mesh/_mesh_queries.py:122
from apeGmsh import FEMData
fem = FEMData.from_gmsh(dim=3, session=g, ndf=6, remove_orphans=False)  # FEMData.py:1508

# From a standalone .msh file (no live session)
fem = FEMData.from_msh("bridge.msh", dim=2)            # FEMData.py:1534
```

Exact signatures:

```python
g.mesh.queries.get_fem_data(dim: int | None = None, *, remove_orphans: bool = False) -> FEMData
FEMData.from_gmsh(dim=None, *, session=None, ndf: int = 6, remove_orphans: bool = False)
FEMData.from_msh(path, dim: int | None = 2, *, remove_orphans: bool = False)
```

`from_gmsh`/`get_fem_data` auto-resolves any pre-mesh declarations
(`g.loads.*`, `g.masses.*`, `g.constraints.*`) into resolved record sets
attached to the snapshot. `ndf` (from_gmsh only) controls the padding of
load / mass vectors (6 = full 3-D frame/shell, 3 = 3-D solid or 2-D
frame, 2 = 2-D solid). Per-node ndf is **not** a session channel — it is
inferred by the `apeSees` bridge from the incident element classes
(element-less decoupled nodes use `ops.ndf`); see `opensees-bridge.md`.

`remove_orphans=True` drops mesh nodes that aren't connected to any
element at the requested `dim`. Nodes referenced by constraint, load,
or mass records are **always** kept — you won't lose a master node just
because the mesh didn't land on it.

### The snapshot is cached & identity-stable

Repeat `get_fem_data()` calls return the **same object identity** until a
broker mutation invalidates the cache (every `g.constraints.X` /
`g.loads.X` / `g.masses.X` call bumps an internal counter). The session
has one canonical "chain head" snapshot that
`FEMData.with_*` / `FEMData.compose(...)` transform (see compose.md).

## Top-level layout

```
fem.nodes            → NodeComposite
fem.elements         → ElementComposite
fem.info             → MeshInfo
fem.inspect          → InspectComposite
fem.partitions       → list[int]      (shortcut for fem.nodes.partitions)
fem.snapshot_id      → str            32-char hex content hash (FEMData.py:1491)
```

`repr(fem)` prints the full `inspect.summary()`, so just printing the
object is a fast way to sanity-check the snapshot.

### `fem.snapshot_id` — the content hash

A deterministic 32-char hex digest over nodes / elements / PGs / labels /
constraints / loads / masses / ndf. Two snapshots with identical content
produce the same `snapshot_id` — this is the linking primitive for
persistence (it is re-verified on `from_h5` read, Part B) and lineage
(`results.lineage`, see results.md). Distinct from the lineage `fem_hash`
(computed from the HDF5 neutral zone), though `from_h5` makes them agree.

## `fem.nodes` — NodeComposite

### Bulk access

```python
fem.nodes.ids       # ndarray(N,) dtype=object — iterates as plain int
fem.nodes.coords    # ndarray(N, 3) float64
len(fem.nodes)
```

IDs are `dtype=object` on purpose: iterating yields plain Python `int`
(OpenSees and other C-extension solvers reject numpy integer scalars on
some paths). Don't cast.

### Selection — `.select(...)`, label/PG/part, never raw tags

`fem.nodes.select(...)` is the **single accessor** (the old `.get` /
`.get_ids` / `.get_coords` were removed). It returns a `MeshSelection`
(point family — spatial verbs test node **coordinates**) that chains
refinement verbs and terminates at `.ids` / `.coords` / `.result()`:

```python
base = fem.nodes.select(pg="Base")              # label OR PG OR part → MeshSelection
fem.nodes.select(target="Base")                 # target resolves label → PG → part
fem.nodes.select(pg=["Top", "Bottom"])          # union via list
fem.nodes.select(pg="Body", partition=3)        # AND-intersected with a partition

base.ids        # list[int]
base.coords     # ndarray(N, 3)
for nid, xyz in base.result():                  # pair-iteration, clean for emission
    ops.node(nid, *xyz)

# spatial narrowing (half-open [lo, hi); on_plane tol= REQUIRED) + set algebra
corner = (fem.nodes.select(pg="Body")
              .in_box((0, 0, 0), (1, 1, 1))
              .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6))
all_bcs = fem.nodes.select(pg="Base") | fem.nodes.select(pg="Wall")
```

Signature:
`fem.nodes.select(target=None, *, pg=None, label=None, tag=None, partition=None, dim=None, ids=None)`
(`FEMData.py:365`). `target=` resolution order is **label → physical
group → part label** (matches `LoadsComposite` auto-resolve, so the same
name works everywhere); a name in none raises with the candidates
printed. Prefer labels/PGs; raw Gmsh tags work but tie you to
live-session state. `.select()` is the **only** accessor — see
`gotchas.md`.

### Sub-composites on `fem.nodes`

```
fem.nodes.physical      → PhysicalGroupSet   (per-PG node/element slices)
fem.nodes.labels        → LabelSet           (per-label slices)
fem.nodes.constraints   → NodeConstraintSet  (equal_dof, rigid_beam, …)
fem.nodes.loads         → NodalLoadSet
fem.nodes.masses        → MassSet
fem.nodes.sp            → SPSet              (single-point prescribed)
```

### Per-node ndf is INFERRED — not a snapshot authoring channel

Per-node DOF count is **inferred by the `apeSees` bridge** from each
node's incident element classes (ADR 0048); an element-less decoupled
node states it with `ops.ndf(handle, ndf=K)` (ADR 0049). There is **no**
`g.node_ndf` session composite — it was removed. You do not set ndf on
the broker; the full story is in `opensees-bridge.md`. (`fem.nodes.ndf_for(nid)`
exists and returns the stored value when ndf metadata is attached, but it
is a read accessor, not the authoring path.)

## `fem.elements` — ElementComposite

Iterating `fem.elements` yields one `ElementGroup` **per element type**:

```python
for group in fem.elements:
    print(group.type_name, len(group), group.dim, group.npe)
    for eid, conn in group:
        solver.element(group.type_name, eid, conn, mat_id)
```

`len(fem.elements)` is the total element count across all groups.

```python
fem.elements.ids            # ndarray(E,) int64 — all element IDs concatenated
fem.elements.is_homogeneous # True if one element type
fem.elements.connectivity   # ⚠ TypeError when >1 element type — see traps
fem.elements.type_table()   # DataFrame: code, name, gmsh_name, dim, order, npe, count
```

### Selection — `.select(...)` (centroid family)

`fem.elements.select(...)` is the single accessor (the old `.get` /
`.resolve` were removed). Spatial verbs test element **centroids**.

```python
body = fem.elements.select(pg="Body")               # → MeshSelection
body.ids                                            # list[int]
body.connectivity                                   # ndarray(N, npe) — homogeneous selection only

# filter to one element type, optionally narrowed spatially
tets = fem.elements.select(pg="Body", element_type="tet4").in_box((0,0,0),(5,5,3))
tets.connectivity                                   # homogeneous → safe

# mixed mesh — go through the GroupResult and resolve()
gr = fem.elements.select(label="col.web").result()
ids, conn = gr.resolve()                            # single type
ids, conn = gr.resolve(element_type="hex8")         # pick one from a mixed selection
```

Signature:
`fem.elements.select(target=None, *, pg=None, label=None, tag=None, dim=None, element_type=None, partition=None, ids=None)`
(`FEMData.py:1064`). Terminals: `.ids` / `.coords` (centroids) /
`.connectivity` (homogeneous only — `TypeError` on mixed) / `.result()`
(→ `GroupResult`; call `.resolve()` on it, `element_type=` to pick from a
mixed selection).

### Sub-composites on `fem.elements`

```
fem.elements.physical     → PhysicalGroupSet
fem.elements.labels       → LabelSet
fem.elements.constraints  → SurfaceConstraintSet  (tie/embedded interpolations)
fem.elements.loads        → ElementLoadSet         (beamUniform, surfacePressure)
```

## `fem.info` / `fem.inspect`

```python
fem.info.n_nodes
fem.info.n_elems
fem.info.bandwidth          # semi-bandwidth; recomputed from connectivity, never stored
fem.info.summary()          # "N nodes, M elements (tet4:4711), bandwidth=1234"

print(fem.inspect.summary())            # multi-line: types, PGs, labels, counts
fem.inspect.node_table()                # DataFrame of all nodes
fem.inspect.element_table()             # DataFrame with 'type' column
print(fem.inspect.constraint_summary())
```

`fem.inspect` is computed on demand and stores no state. Bandwidth wants
small node numbers — call `g.mesh.partitioning.renumber(method="rcm")`
**before** `get_fem_data` (renumbering is a Gmsh-side op; the snapshot
reads what Gmsh hands it). Note: bandwidth is for dense 1-based tags,
**not** solver bandwidth — OpenSees' numberer handles that.

## Common traps (Part A)

- `fem.elements.connectivity` raises `TypeError` on multi-type meshes.
  Use `fem.elements.select(element_type=...).connectivity`, or
  `.result().resolve(element_type=...)`, or iterate
  `for group in fem.elements:`.
- `fem.nodes.ids` is `dtype=object`; `fem.elements.ids` is `int64`.
  Intentional asymmetry — node IDs are consumed one at a time by Python
  solvers, element IDs in bulk.
- After `g.end()`, resolving a raw `(dim, tag)` DimTag raises
  `RuntimeError` (Gmsh is gone). Use labels / PGs baked into the
  snapshot instead.
- `get_fem_data(dim=None)` returns **every** element present — `dim=2`
  surface elements (loads / tied-contact / rigid diaphragms) *and*
  `dim=3` volume elements. Filter with `get(dim=3)` for the volume mesh.
- `remove_orphans` here is a **mesh-node** filter — unrelated to the
  geometry method `g.model.geometry.remove_orphans()` (which sweeps
  dangling CAD entities). Don't conflate the two.

---

# Part B — native `model.h5` round-trip

A `FEMData` snapshot persists to a self-describing HDF5 **neutral zone**
and rebuilds losslessly. This is the apeGmsh-native model file — no
OpenSees needed to write or read it.

## Write: `FEMData.to_h5` / `g.save()` / `save_to=`

```python
# Direct write (FEMData.py:1603)
fem.to_h5("plate.h5", model_name="plate", apegmsh_version="2.0.0", ndf=0)

# Autosave on context-exit / g.end()  (verified: tests/test_session_save.py::test_autosave_writes_on_exit)
with apeGmsh(model_name="plate", save_to="plate.h5", overwrite=True) as g:
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 0.1, label="body")
    g.physical.add_volume("body", name="body")
    g.mesh.generation.generate(dim=3)
    # neutral zone is written in end()/__exit__, before gmsh.finalize()

# Explicit checkpoint mid-session  (verified: tests/test_session_save.py::test_manual_save_uses_save_to_when_no_arg)
g.save()                # writes to the ctor save_to= target
g.save("ckpt.h5")       # or an explicit path
```

Exact signatures:

```python
FEMData.to_h5(self, path, *, model_name: str = "", apegmsh_version: str = "", ndf: int = 0) -> None
apeGmsh.__init__(..., *, save_to: str | Path | None = None, overwrite: bool = True)   # _core.py:84
apeGmsh.save(self, path: str | Path | None = None) -> Path                            # _core.py:251
```

Write semantics & traps:

- `g.save()`/`to_h5` write **only the neutral zone** (no `/opensees/`).
  To persist the OpenSees deck too, use `apeSees(fem).h5(path)` (writes
  **both** neutral + opensees zones — see opensees-bridge.md).
- `save_to=` does **not** autosave eagerly. The write happens in
  `end()` (i.e. `__exit__` or explicit `g.end()`). If the process dies
  before `end()`, nothing is written.
- `g.save()` with neither an explicit `path` nor a ctor `save_to=`
  raises `RuntimeError` (verified:
  `tests/test_session_save.py::test_manual_save_without_path_or_save_to_raises`).
- `overwrite=False` + an existing target raises `FileExistsError`
  (verified: `tests/test_session_save.py::test_overwrite_false_raises_on_existing`).
- Autosave on `end()` **catches and WARNS** (does not raise) on write
  failure so gmsh still finalizes — a silently-warned autosave failure
  can lose data. Prefer an explicit `g.save()` when persistence matters.

## Read: `FEMData.from_h5` — integrity-checked

```python
from apeGmsh import FEMData
fem = FEMData.from_h5("plate.h5")               # verified: tests/test_femdata_from_h5.py::test_round_trip_nodes_and_elements
fem = FEMData.from_h5("results.h5", root="/model")   # composed/results file: rich layout under /model/
```

Signature: `FEMData.from_h5(cls, path, *, root: str = "/") -> FEMData`
(FEMData.py:1547). Rebuilds nodes, elements (per type), PGs, labels,
mesh selections, constraints, loads, masses, and per-node `ndf` —
everything the writer round-trips.

**Fail-loud integrity** (no silent corruption):

- `/meta/snapshot_id` is re-verified against the recomputed hash of the
  rebuilt FEM. A tampered neutral zone (mutated coords) raises
  `MalformedH5Error('snapshot_id mismatch')` (verified:
  `tests/test_femdata_from_h5.py::test_snapshot_id_verified_on_read`).
- Missing `/meta` → `MalformedH5Error` (verified:
  `tests/test_femdata_from_h5.py::test_missing_meta_raises`).
- Wrong schema major → `SchemaVersionError` (verified:
  `tests/test_femdata_from_h5.py::test_wrong_schema_major_raises`).

```python
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
from apeGmsh.opensees._internal.schema_version import SchemaVersionError
```

Round-trip facts:

- **bandwidth is NOT persisted** — recomputed from connectivity on read
  via `_compute_bandwidth` (verified:
  `tests/test_femdata_from_h5.py::test_bandwidth_recomputed_on_read`).
  Don't rely on a stored bandwidth value.
- **Per-node ndf round-trips** — the resolved inferred ∪ `ops.ndf`
  values persist under the bridge `/opensees/nodes_ndf` zone (ADR 0049,
  written by `apeSees(fem).h5()`) and reload stably (verified:
  `tests/opensees/h5/test_nodes_ndf_persist.py::test_nodes_ndf_roundtrip_hash_stable`).
  Files without ndf metadata load with `ndf=None`.
- `from_h5`'s rebuilt `snapshot_id` equals the source's
  `/meta/snapshot_id`, and the lineage `fem_hash` matches it (verified:
  `tests/test_femdata_from_h5.py::test_from_h5_lineage_fem_hash_matches_snapshot_id`).
- `root=` reads/writes into a sub-group. Standalone `model.h5` uses
  `root="/"` (rich layout at file root); a composed `results.h5` carries
  the same layout under `/model/`, so pass `root="/model"` (verified:
  `tests/test_femdata_from_h5.py::test_from_h5_with_root_kwarg`).

## `apeGmsh.from_h5` — chain-phase session reload (no gmsh)

Two **different** `from_h5` classmethods exist — don't confuse them:

| Method | Returns | Source |
|---|---|---|
| `FEMData.from_h5(path, *, root="/")` | a `FEMData` snapshot | FEMData.py:1547 |
| `apeGmsh.from_h5(path, *, model_name=None, verbose=False)` | a **chain-phase session** | _core.py:141 |

```python
from apeGmsh import apeGmsh
g2 = apeGmsh.from_h5("plate.h5")     # verified: tests/test_femdata_from_h5.py::test_session_save_then_from_h5
g2.compose("bolt.h5", label="bolt", translate=(10, 0, 0))   # compose works
g2.save("assembly.h5")
```

A chain-phase session has **NO gmsh state**: `g.model.*` and
`g.mesh.generation.*` will fail. Only `compose()` / `compose_inspect()` /
`compose_list()` / `compose_tree()` / `save()` and the chain-phase
interface constraints/loads/masses work. See compose.md for the full
chain-phase contract.

## Schema constants — two independent zones (ADR 0023)

The file carries **two per-zone version constants on different cadences**
— do not collapse them into one number:

| Constant | Value | Source | Written by |
|---|---|---|---|
| `NEUTRAL_SCHEMA_VERSION` | **`"2.26.0"`** | `src/apeGmsh/mesh/_femdata_h5_io.py` | `to_h5` / `g.save()` |
| bridge `SCHEMA_VERSION` | **`"2.19.0"`** | `src/apeGmsh/opensees/emitter/h5.py:414` | `apeSees(fem).h5()` opensees zone |

The neutral zone now also persists embedded-rebar ties (`/reinforce_ties`,
`2.15.0`), auto-emitted rebar elements (`/rebar_elements`, `2.16.0`), fork
contact records (`/contacts`, `2.21.0`; `cell` knob `2.23.0`, edge-edge
fields `2.25.0`), `g.embed` ties (`/embed_ties`, `2.22.0`), rigid-plane
contacts (`/contact_planes`, `2.24.0`), and the reinforce `corot` flag
(`2.26.0`) — all round-trip through `FEMData.to_h5` / `from_h5`.

```python
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION   # "2.26.0"
from apeGmsh.opensees.emitter.h5 import SCHEMA_VERSION           # "2.19.0"
# Both constants move with the source — confirm the exact number there before
# relying on it; the two-version reader window below is the durable contract.
```

**Two-version reader window (ADR 0023):** a reader at `X.Y` accepts only
`X.Y.*` and `X.(Y-1).*`. Older minors, newer minors, or a different
major all raise `SchemaVersionError` — a newer-than-reader file is
*refused*, never silently tolerated. New code reads the per-zone keys
(`neutral_schema_version` / `opensees_schema_version` /
`results_schema_version`); the single envelope `/meta/schema_version`
key is back-compat only.
