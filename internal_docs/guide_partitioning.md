# apeGmsh mesh partitioning

A guide to driving OpenSeesMP from apeGmsh — decomposing a mesh into
ranks, watching that decomposition flow through `FEMData` into the
`apeSees` bridge, and reading the resulting partitioned `model.h5`
back into a viewer.

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
# ... geometry, parts, mesh ...
```

This guide focuses on **mesh partitioning for parallel solvers**
(OpenSeesMP). For node/element **renumbering** (bandwidth reduction
via RCM / Hilbert / METIS-ordering) see the same composite's
`renumber()` method, documented in `guide_meshing.md` §
"Renumbering".

## Tasks on this page

- [Partition into 4 with Gmsh-native METIS](#3-recipe-1-gmsh-native-partitioning-into-4) · [Weighted partitioning with pymetis](#4-recipe-2-weighted-partitioning-with-pymetis)
- [Round-trip a partitioned model through HDF5](#5-hdf5-round-trip-and-post-processing) · [Inspect partitions in the viewer](#6-viewer-integration)
- [Replicate cross-partition constraints](#7-cross-partition-constraints-adr-0027) · [Run a deck single-process vs OpenSeesMP](#8-single-process-vs-openseesmp)


## 1. Overview

Mesh partitioning splits a single mesh into `N` non-overlapping
sub-domains so that one OpenSeesMP rank can own each sub-domain.
A structural engineer reaches for it for one of three reasons:

- **Drive OpenSeesMP.** Large nonlinear models (millions of elements,
  fiber sections at every Gauss point, history-dependent materials)
  factor faster on a distributed sparse-direct solver like Mumps
  than on a single-process direct solver. Partitioning is the
  prerequisite for shipping the model to `mpiexec -np N OpenSeesMP
  model.tcl`.
- **Decompose a large model.** Even on one machine, splitting a
  model into K ranks via MPI on K cores often beats single-process
  OpenSees because OpenSeesMP's Mumps backend is genuinely parallel
  (whereas single-process OpenSees serialises the factor).
- **Get per-rank result subsets.** The partitioning information is
  persisted into the broker (`fem.partitions`) and into the emitted
  `model.h5` zone (`/opensees/partitions/`). Post-processing can
  iterate rank-by-rank for memory-constrained reads.

Partition data flows through three layers of apeGmsh:

| Layer | Symbol | What it holds |
|---|---|---|
| **Mesh** | `g.mesh.partitioning` | The Gmsh `gmsh.model.mesh.partition(N)` action. Returns a `PartitionInfo` with per-rank element counts and (when weighted) per-rank weight sums. |
| **Broker** | `fem.partitions` | A `PartitionSet` yielding `PartitionRecord(id, node_ids, element_ids, weight_sum)` per rank. Also feeds `fem.nodes.select(partition=N)` / `fem.elements.select(partition=N)`. |
| **Bridge** | `apeSees(fem).tcl / .py / .h5` | When `len(fem.partitions) > 1`, every emitter brackets per-rank content in `partition_open(K)` / `partition_close()` blocks (`if {[getPID] == K} { ... }` in Tcl, `if getPID() == K:` in Python, `/opensees/partitions/partition_NN/` groups in HDF5). |

Single-partition models (`len(fem.partitions) <= 1`) emit
byte-identical output to a non-partitioned model — partitioning is
opt-in and zero-cost when not used.


## 2. Two flavors of partitioning

`g.mesh.partitioning.partition(n_parts, *, weights=None, backend=None)`
dispatches to one of two backends based on whether you supply
weights:

### Flavor A — Gmsh-native (element-count balance)

```python
info = g.mesh.partitioning.partition(n_parts=4)
```

Calls Gmsh's built-in METIS binding (`gmsh.model.mesh.partition(N)`)
directly. Balances by **element count** — every rank gets roughly
the same number of elements, irrespective of how expensive each
element is to solve. No optional dependencies. The default.

### Flavor B — Weighted (sum-of-weights balance)

```python
weights = [...]            # one float per element, ordered as below
info = g.mesh.partitioning.partition(n_parts=4, weights=weights)
```

Builds the element dual graph from Gmsh's connectivity in apeGmsh,
calls an external METIS binding (`pymetis` by default, or
`networkx-metis`), and pushes the assignment back into Gmsh via
`partition_explicit(...)` so downstream consumers (the FEMData
broker, the h5 round-trip, the apeSees bridge) see the same model
state as the native path.

Weight ordering: the `weights` sequence must match
`gmsh.model.mesh.getElements(dim=-1, tag=-1)` traversal order — every
element across every dimension, in Gmsh's native iteration. Length
mismatch raises `ValueError`.

### When to use which

| Flavor | Backend | When |
|---|---|---|
| **A** (default) | Gmsh-native METIS | Elastic linear or modestly nonlinear models where per-element cost is roughly uniform. Tetrahedra-only or hex-only meshes. Any model that fits in memory on a single process and you just want MPI speedup. |
| **B** (weighted) | `pymetis` / `networkx-metis` | Fiber-section beam-columns mixed with elastic beams (the fiber elements cost ~10–100× more per Gauss point). Mixed solid + shell models where solid cells dominate. Any model where the per-element factor cost varies by orders of magnitude. |
| **B** with `weights=None`, explicit backend | `pymetis` | Force the external path for backend testing without supplying weights — unit weights are used, same balance as Flavor A. |

The `partition_explicit(n_parts, elem_tags, parts)` method is also
available for callers that already have a vector of rank assignments
(e.g. from a custom load-balancer or a domain-decomposition heuristic
you trust more than METIS for your problem class). It bypasses both
flavors and writes the assignment straight into Gmsh.


## 3. Recipe 1 — Gmsh-native partitioning into 4

A complete pipeline: build a simple frame, partition it into 4
ranks, inspect the broker view, and emit a `.tcl` deck.

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="recipe1", verbose=False) as g:
    gm = g.model.geometry

    # Two-storey, four-column space frame: corner points at z=0/H1/H2.
    H = 3.0  # storey height
    L = 5.0  # bay
    base = [
        gm.add_point(0, 0, 0), gm.add_point(L, 0, 0),
        gm.add_point(L, L, 0), gm.add_point(0, L, 0),
    ]
    floor1 = [
        gm.add_point(0, 0, H), gm.add_point(L, 0, H),
        gm.add_point(L, L, H), gm.add_point(0, L, H),
    ]
    floor2 = [
        gm.add_point(0, 0, 2 * H), gm.add_point(L, 0, 2 * H),
        gm.add_point(L, L, 2 * H), gm.add_point(0, L, 2 * H),
    ]

    cols, beams = [], []
    for floor in (floor1, floor2):
        prev = base if floor is floor1 else floor1
        for i in range(4):
            cols.append(gm.add_line(prev[i], floor[i]))
        for i in range(4):
            beams.append(gm.add_line(floor[i], floor[(i + 1) % 4]))

    g.physical.add_curve(cols, name="Columns")
    g.physical.add_curve(beams, name="Beams")
    g.physical.add_point(base, name="Base")

    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=1)
    g.mesh.partitioning.renumber(dim=1, method="simple", base=1)

    # >>> partition into 4 ranks (Flavor A) <<<
    info = g.mesh.partitioning.partition(n_parts=4)
    print(info)                       # PartitionInfo(4 parts: P1:N, P2:N, ...)

    fem = g.mesh.queries.get_fem_data(dim=1)

# Inspect the broker view.
print(f"len(fem.partitions) = {len(fem.partitions)}")
for rec in fem.partitions:
    print(f"  rank {rec.id}: {rec.n_elements} elements, "
          f"{rec.n_nodes} nodes")

# Drive apeSees and emit the partitioned Tcl deck.
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)
t = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
ops.element.elasticBeamColumn(
    pg="Columns", transf=t,
    A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
)
t2 = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
ops.element.elasticBeamColumn(
    pg="Beams", transf=t2,
    A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
)
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
ops.tcl("frame_partitioned.tcl")
```

Run it under OpenSeesMP:

```bash
mpiexec -np 4 OpenSeesMP frame_partitioned.tcl
```

The emitted `frame_partitioned.tcl` carries the partitioning shape:

```tcl
# (header — model, materials, transforms, etc. — at indent 0)

if {[info commands getPID] == ""} { proc getPID {} { return 0 } }

if {[getPID] == 1} {
    # rank 1 owns these nodes / elements
    node 12 0.0 0.0 1.5 -ndf 6
    node 13 0.0 0.0 2.25 -ndf 6
    ...
    element elasticBeamColumn 5 12 13 0.01 ... 1
    ...
}

if {[getPID] == 2} {
    ...
}

if {[getPID] == 3} {
    ...
}

if {[getPID] == 4} {
    ...
}

# Analysis chain — emitted GLOBALLY outside every rank block.
constraints Transformation
numberer ParallelPlain
system Mumps
test NormDispIncr 1.0e-6 25
algorithm KrylovNewton
integrator LoadControl 1.0
analysis Static
analyze 1
```

Three structural features worth noting:

1. **`proc getPID` shim emitted exactly once** at the top of the file.
   This makes the deck runnable under single-process OpenSees too
   (which has no `getPID`); the shim returns 0 so only rank 0's block
   executes. See § 8.
2. **One `if {[getPID] == K}` block per rank.** Per-rank content
   sits inside the block; foreign nodes (declared because a rank's
   constraint references a node it doesn't own — see § 7) are
   declared with `node <tag> ... -ndf 6` before the constraint line.
3. **Analysis chain lives globally** (outside every rank block). Per
   ADR 0027 INV-5, `numberer ParallelPlain` and `system Mumps` are
   auto-emitted when `len(fem.partitions) > 1`.


## 4. Recipe 2 — Weighted partitioning with pymetis

Install the optional backend:

```bash
pip install apeGmsh[partition-pymetis]
```

Windows note: `pymetis` has no PyPI Windows wheel. Use conda-forge
instead — `conda install -c conda-forge pymetis`. Or skip Flavor B
on Windows and use Flavor A (which goes through the bundled Gmsh
binding).

Then weight the partition. The most common case is mixing
fiber-section beams (expensive) with elastic beams (cheap):

```python
# Same frame fixture as Recipe 1 — partition step replaced.
# Suppose the columns use fiber sections and the beams are elastic;
# fiber elements cost ~10× more per solve step.
fem_preview = g.mesh.queries.get_fem_data(dim=1)
col_ids = set(int(t) for t in fem_preview.elements.get(pg="Columns").ids)

# Weight vector ordered as Gmsh's flat element traversal.
import gmsh
all_tags = []
for d in range(4):
    _, etags_list, _ = gmsh.model.mesh.getElements(dim=d, tag=-1)
    for etags in etags_list:
        all_tags.extend(int(t) for t in etags)

weights = [10.0 if t in col_ids else 1.0 for t in all_tags]

info = g.mesh.partitioning.partition(
    n_parts=4, weights=weights,                         # pymetis by default
)
print(info)
# PartitionInfo(4 parts: P1:N1, P2:N2, ...; weights[P1:W, P2:W, ...])

assert info.weights_per_partition is not None
spread = (max(info.weights_per_partition.values())
          - min(info.weights_per_partition.values()))
print(f"weight spread = {spread:.1f}  (target: small)")
```

`PartitionInfo.weights_per_partition` is populated whenever
`weights=` was passed; it stays `None` for Flavor A.

Backend override (rare):

```python
g.mesh.partitioning.partition(
    n_parts=4, weights=weights, backend="networkx-metis",
)
```

Backend dispatch rules (from `_mesh_partitioning.py`):

| `weights`  | `backend`            | Path                              |
|------------|----------------------|-----------------------------------|
| `None`     | `None`               | Gmsh-native METIS (Flavor A)      |
| `None`     | `"gmsh"`             | Gmsh-native METIS (explicit)      |
| `None`     | `"pymetis"`          | pymetis, `vwgt = ones(N)`         |
| `None`     | `"networkx-metis"`   | networkx-metis, unit weights      |
| sequence   | `None`               | pymetis (default for weighted)    |
| sequence   | `"pymetis"`          | pymetis with the given weights    |
| sequence   | `"networkx-metis"`   | networkx-metis with weights       |
| sequence   | `"gmsh"`             | `ValueError` — no vwgt API        |


## 5. HDF5 round-trip and post-processing

`ops.h5(path)` writes the bridge's canonical `model.h5`. When
`len(fem.partitions) > 1`, the bridge populates the partition zone
(group structure landed in opensees schema 2.10.0; current bridge
version is 2.12.0):

```
/meta                                 (carries opensees=2.12.0)
/opensees/...                         (nodes, elements, materials, ...)
/opensees/partitions/                 (n_partitions = N attr)
  partition_00/
    element_ids        int64 (n_elem_rank_0,)
    node_ids           int64 (n_node_rank_0,)   # native + foreign decls
    boundary_node_ids  int64 (n_boundary_0,)    # shared with another rank
    @rank              = 0
    @n_elements        = ...
    @n_nodes           = ...
  partition_01/
    ...
/opensees/element_meta/{type_token}/
  partition_ids        int64 (n_elem_of_type,)  # parallel to other columns
```

Group names and the `rank` attr / `partition_ids` values are
**0-based** (matching `OpenSeesMP::getPID()`).  The broker's
`PartitionRecord.id` remains Gmsh's 1-based label — that is
Gmsh-side traceability metadata, not a runtime rank.  The mapping
is `rank == i`, `PartitionRecord.id == fem.partitions[rank].id`.

The new parallel `partition_ids` column on every
`element_meta/{type}/` group lets a reader filter elements by rank
in O(N) without re-reading the partition groups. Sentinel `-1`
marks any element emitted outside a partition block (i.e. before
the first `partition_open` call — used for global-scope elements,
which today is empty in the partition path).

Read it back:

```python
from apeGmsh.opensees import OpenSeesModel

om = OpenSeesModel.from_h5("frame_partitioned.h5")
print(om._fem.partitions)
# PartitionSet(4 partitions, ids=[1, 2, 3, 4])

for rec in om._fem.partitions:
    print(rec)
# PartitionRecord(id=1, n_nodes=12, n_elements=10)
# ...
# (PartitionRecord.id retains Gmsh's 1-based label;
# the matching OpenSeesMP rank is the enumerate index — rank 0
# is fem.partitions[0] is PartitionRecord(id=1, ...).)
```

`OpenSeesModel.from_h5` rehydrates the broker via
`FEMData.from_h5(...)`, which restores `fem.partitions`. The
`PartitionSet` round-trips losslessly.

For lower-level access — the bridge-emitted per-rank emit log
(which includes foreign-declared nodes from cross-partition MP
constraints) — use the `H5Model` reader directly:

```python
from apeGmsh.opensees.emitter import h5_reader

with h5_reader.open("frame_partitioned.h5") as model:
    for rec in model.partitions():
        print(f"rank {rec.rank}: {len(rec.element_ids)} elements, "
              f"{len(rec.boundary_node_ids)} boundary nodes")
```

Schema version: **opensees 2.12.0** (current bridge
`SCHEMA_VERSION`). The rank attr / `partition_ids` row values became
0-based in the 2.11.0 bump; 2.12.0 added the ADR 0035 embeddedNode
columns. Per ADR 0023 two-version reader window, both 2.11.x and
2.12.x files are accepted.

### 5.1. MPCO recorder regions under partitioning (ADR 0027 INV-4)

For the un-partitioned MPCO recipe — declaring the recorder and reading
the `.mpco` back with `Results.from_mpco` — see
[Get results via MPCO (STKO)](../how-to/results-mpco.md). This section
covers what changes once `len(fem.partitions) > 1`.

When a partitioned model declares
`ops.recorder.MPCO(nodes_pg="...", elements_pg="...", file=...)`
(or any filter form: `nodes=`, `elements=`, `*_pg=`), the bridge
splits emission into two scopes per ADR 0027 INV-4:

* The `recorder mpco ... -R <tag>` **declaration** emits **once**
  in global scope (outside every `if {[getPID] == K} {...}` block).
  One MPCO declaration is sufficient under OpenSeesMP — the
  recorder writes to disk, not to the model topology.
* The `region <tag> -node ... -ele ...` line emits **per-rank**
  inside each rank's `partition_open(K)` block, carrying only the
  intersection of the resolved filter ids with that rank's owned
  nodes/elements. Ranks whose intersection is **empty** emit no
  `region` line at all (MPCO handles a missing per-rank region as
  a no-op — the recorder declaration's `-R <tag>` still resolves
  on the ranks that did emit).
* The `<tag>` is the **same scalar** across every emitting rank
  and on the global recorder line. MPCO post-processing stitches
  the per-rank `.mpco` outputs by `<tag>` identity at read time.

For un-partitioned models (`len(fem.partitions) <= 1`) the emit
is byte-identical to the pre-INV-4 behaviour — one `region` plus
one `recorder mpco` in global scope, with the same tag.


## 6. Viewer integration

The interactive Qt mesh viewer surfaces partition assignment in
three places:

```python
from apeGmsh.viewers.data._viewer_data import ViewerData

view = ViewerData.from_fem(fem)
g.mesh.viewer(view=view)
```

1. **ColorMode "Partition"** — the Display tab's colour-mode
   selector adds a `"Partition"` entry. Selecting it colours each
   element by its owning rank (one stable colour per partition id),
   matching the per-entity dispatch shape used by the outline tree
   below.
2. **Outline tree "Partitions" section** — when the
   `ViewerData` carries partition labelling, the outline tree
   shows a "Partitions" header alongside Element Types, Parts,
   Loads, Masses. Toggling a row hides/shows every element on that
   rank.
3. **Boundary-node overlay** — under the Partitions section is a
   single "Boundary nodes" row. Toggling it draws a marker on every
   node that is shared between at least two ranks (the same set
   that lands in the H5 `boundary_node_ids` dataset). Useful for
   diagnosing whether a cross-partition MP constraint is going to
   replicate (see § 7).

Single-partition meshes hide the "Partitions" section entirely —
the outline tree adapts to whatever labelling the snapshot carries.


## 7. Cross-partition constraints (ADR 0027)

When you declare a multi-point constraint at session time and the
partitioner later places the master and slave on different ranks,
the bridge follows the policy in ADR 0027:

- **Replicate the constraint on every owning rank.** The exact
  same `rigidDiaphragm` / `rigidLink` / `equalDOF` line is emitted
  inside every owning rank's `if {[getPID] == K}` block, byte-for-byte
  identical across ranks (INV-1).
- **Declare foreign node tags before the constraint line.** If
  rank K's block references a node tag K does not natively own, the
  bridge emits `node <tag> <x> <y> <z> -ndf 6` on K's block before
  the constraint line (INV-2). The coordinates come from the broker and
  the `ndf` from the bridge's deterministic per-node inference — identical
  to those used on the natively-owning rank — applied in
  `_internal/build.py::emit_mp_constraints_partitioned` (ADR 0048; a node
  whose inferred `ndf` equals the envelope elides `-ndf` and falls back to
  the `ops.model(ndm, ndf=)` envelope).
- **Phantom nodes get broker-deterministic tags** (INV-3). When a
  `node_to_surface` constraint synthesises phantom nodes at
  build time, those tags and coordinates come from one canonical
  allocator and are the same scalar on every rank that hosts a
  constraint referencing them.

`embeddedNode` (ASDEmbeddedNodeElement) is the one exception:
because it carries an element tag, **element ownership wins** — the
constraint is emitted only on the rank that owns the host element,
with the embedded node declared as foreign there.

Concrete example. The three-storey frame from Recipe 1 with a
rigid-floor diaphragm per storey:

```python
# Session-time declaration BEFORE the partition step.
g.physical.add_point([master_node_floor_1], name="MasterF1")
g.physical.add_point(floor_1_corner_points, name="FloorF1")
g.constraints.rigid_diaphragm(
    master_label="MasterF1", slave_label="FloorF1",
    plane_normal=(0., 0., 1.),
    constrained_dofs=[1, 2, 6],
    name="floor_1",
)
# ... repeat for floor_2, floor_3 ...
```

With 4 ranks, the floor corners typically scatter across multiple
ranks (the corners are far apart in the dual graph). The emitted
deck contains the same `rigidDiaphragm` line on every rank that
owns at least one slave corner:

```tcl
if {[getPID] == 0} {
    ...
    # Foreign-node declarations for nodes rank 0 does not own.
    node 7 5.0 5.0 3.0 -ndf 6
    node 11 0.0 0.0 3.0 -ndf 6

    # MP-comment + rigidDiaphragm (byte-identical across owning ranks).
    # constraint floor_1
    rigidDiaphragm 3 5 7 9 11
}

if {[getPID] == 1} {
    ...
    node 5 0.0 0.0 3.0 -ndf 6
    node 9 5.0 0.0 3.0 -ndf 6
    # constraint floor_1
    rigidDiaphragm 3 5 7 9 11
}
```

Both rank 0 and rank 1 emit the **same** `rigidDiaphragm 3 5 7 9 11`
line; whichever ranks natively own which slaves, the foreign-side
ranks declare the missing nodes before the line. The OpenSeesMP
constraint handler dedupes equivalent declarations at solve time —
the redundancy is correctness-preserving, not a foot-gun.

This means a model whose load path depends on cross-partition rigid
diaphragms produces the **same answer** under OpenSeesMP as under
single-process OpenSees — the diaphragm coupling is preserved.


## 8. Single-process vs OpenSeesMP

The `proc getPID` shim in § 3 makes a partitioned deck syntactically
parseable under single-process OpenSees:

```tcl
if {[info commands getPID] == ""} { proc getPID {} { return 0 } }
```

If the runtime has no `getPID` (no MPI loaded), the shim defines it
to return 0, and only the rank-0 block executes. Other ranks' blocks
are skipped by the `if {[getPID] == K}` gate.

The guard probes `info commands`, **not** `info procs`: OpenSeesMP
registers `getPID` via `Tcl_CreateCommand`, i.e. as a C *command*
that `info procs` cannot see. An `info procs` guard silently
overrides the native command with the rank-0 fallback, so every MPI
rank builds rank 0's submodel — the run looks green while solving
N copies of one subdomain.

### Runtime-conditional numberer + system

When `len(fem.partitions) > 1` and the user has not explicitly
declared a numberer / system, the bridge auto-emits a **runtime
conditional** (ADR 0027 INV-5, amended 2026-05-23) so the same deck
runs under both OpenSeesMP and single-process OpenSees:

```tcl
# ParallelPlain only exists in OpenSeesMP; fall back to RCM under single-process OpenSees.
if {[catch {numberer ParallelPlain} _err]} { numberer RCM }
# Mumps requires OpenSeesMP + MPI; fall back to UmfPack under single-process OpenSees.
if {[catch {system Mumps} _err]} { system UmfPack }
```

The Py emitter produces the equivalent `try / except Exception`
shape. The LiveOps emitter performs the same try/fallback
in-process, emitting one `UserWarning` per fallback.

Under regular OpenSees, the runtime prints two `WARNING` lines
(`No Numberer type exists (Plain, RCM only)`, `system Mumps is
unknown or not installed`); Tcl `catch` swallows them and the
fallback executes. The deck reaches end-of-file and `analyze`
converges. Under OpenSeesMP the primary `ParallelPlain` +
`Mumps` apply.

### Overriding the auto-emit

Declare the analysis chain explicitly before `ops.tcl(...)` /
`ops.py(...)` / `ops.h5(...)` to suppress the auto-emit:

```python
ops.numberer.RCM()         # or ParallelPlain — your choice
ops.system.UmfPack()       # or Mumps — your choice
# ... rest of analysis chain ...
ops.tcl("model.tcl")
```

If your explicit override is MP-incompatible (e.g. `RCM` with
`len(fem.partitions) > 1`) the bridge fires a `UserWarning`
("serial numberer 'RCM' explicitly declared … OpenSeesMP requires
a parallel numberer") but respects your choice — you've opted
into a single-process deck.

### H5 emission

The H5 emitter records the **primary** in `/opensees/analysis/`
attrs (`numberer`, `system`) and the **fallback** in companion
attrs (`numberer_runtime_fallback`, `system_runtime_fallback`).
Readers see both choices; round-trip through `OpenSeesModel`
preserves the pair.

The shim itself, by contrast, has been stable since P4 and is not
expected to change.


## 9. Gotchas and known limitations

- **Partition id conventions are producer-dependent; the runtime rank is not.**
  Three model-building paths can populate `fem.partitions`, and they do
  **not** share a `PartitionRecord.id` base:
  - **Gmsh-native** (`g.mesh.partitioning.partition`) — 1-based ids
    `1..N` (no partition 0), Gmsh's METIS labels.
  - **`g.compose(...)`** — 0-based ranks: the host owns `id=0` and each
    composed module gets `1, 2, ...` (ADR 0038 rank model). `host=0`
    is intentional — it aligns with `OpenSeesMP::getPID()`. See
    [Compose modules into one model](../how-to/compose-modules.md).
  - **Parts** (`g.parts.add` + `fragment_all`) — produce **no**
    partitions; Parts are assembly/labeling units, orthogonal to solver
    decomposition. Call `partition(...)` (or compose) separately to get
    ranks.

  The **runtime rank** the bridge emits is producer-independent: it is
  the 0-based enumerate index over sorted `fem.partitions`
  (`runtime_rank_from_partition_record` — the single source of truth,
  **not** `id - 1`). So Gmsh's `id=1` and compose's `id=0` both map to
  runtime rank 0. The Tcl/Py gates are `if {[getPID] == K}`,
  `K in 0..N-1`. Iterate `for rank, rec in enumerate(fem.partitions):`
  rather than hard-coding ids. (Schema 2.10.0 had 1-based gates — a bug
  where rank 0 had no work and partition N was dropped under `mpiexec
  -np N`; 2.11.0 fixed it at the build seam.)

  **On disk there are two partition representations, with different
  bases by design:**
  - the **neutral zone** `/partitions/{id}/` stores the **raw producer
    id** (a composed archive has `/partitions/0/` for the host; a Gmsh
    archive has `/partitions/1/`...). Round-trip-consistent —
    `select(partition=k)` survives H5 unchanged.
  - the **OpenSees zone** `/opensees/partitions/partition_NN/` stores
    the **0-based runtime rank** (`rank` attr / `partition_ids` column,
    0-based since the schema 2.11.0 bump), matching `getPID()`.

  A reader that needs the runtime rank must use the OpenSees zone (or
  re-enumerate `fem.partitions`), **not** the neutral-zone `id`.

- **`select(partition=N)` keys on the raw producer id — its meaning is
  producer-dependent.** On a Gmsh-partitioned model `select(partition=1)`
  is the first partition and `select(partition=0)` raises; on a composed
  model `select(partition=0)` is the host and `select(partition=1)` is
  the first module. This is a known ergonomic wart, **not** a
  correctness bug in emitted analysis (the bridge normalizes by
  position). A producer-agnostic selector (resolve by runtime rank) is a
  possible future additive API; today, know which producer built your
  model before keying on a raw partition id.

- **`pymetis` has no PyPI Windows wheel.** Install via
  `conda install -c conda-forge pymetis`, or use Flavor A which
  routes through the bundled Gmsh binding (no extra deps).

- **`networkx-metis` is not on PyPI under any name.** The upstream
  is source-install only:
  `pip install git+https://github.com/networkx/networkx-metis.git`.
  Captured as `partition-networkx` extra but
  `pip install apeGmsh[partition-networkx]` installs only the
  `networkx` graph library — `nxmetis` itself must be installed
  separately.

- **MPCO recorder regions split global declaration from per-rank
  region (ADR 0027 INV-4).** Per ADR 0027 §"Regions interaction" a
  recorder region's `element_ids` / `node_ids` are **intersected**
  with per-rank ownership before emission. This is **implemented**:
  the `recorder mpco ... -R <tag>` declaration emits once globally
  while the `region <tag> -node ... -ele ...` line emits per-rank
  inside each `if {[getPID] == K}` block, carrying only that rank's
  intersection (ranks with an empty intersection emit no `region`
  line). The `<tag>` is the same scalar across every emitting rank,
  so MPCO stitches the per-rank `.mpco` outputs by tag identity at
  read time. See § 5.1 for the full emit shape.

- **Per-node `ndf` is inferred on the bridge, not declared.** apeGmsh
  infers each node's DOF count from the element classes declared on it
  (a `ShellMITC4` node → 6, a `stdBrick` node → 3); there is no
  `g.node_ndf` composite. `ops.model(ndm, ndf=K)` keeps `ndf` only as the
  **envelope** — the builder default and the fallback for element-less
  nodes. The sole explicit per-node channel is `ops.ndf(target, ndf=K)`,
  restricted to element-less **decoupled** nodes. See
  [ADR 0048](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0048-infer-per-node-ndf-from-elements.md)
  (inference) and
  [ADR 0049](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0049-decoupled-nodes.md)
  (`ops.ndf`).

  Foreign node declarations under cross-partition replication (§ 7)
  consume the resolved per-node `ndf` directly: the partitioned fan-out
  in `_internal/build.py::emit_mp_constraints_partitioned` passes
  `-ndf K` on the `node(...)` call when a node's inferred value differs
  from the envelope, and elides it where they match. This unblocks
  mixed-ndf partitioned models such as `ndf=3` solid nodes coexisting
  with `ndf=6` shell nodes across rank boundaries.

  Cross-rank consistency is **determinism-guaranteed** (ADR 0048): every
  OpenSeesMP rank runs the *same* inference over the same broker and the
  same element declarations, so a foreign / ghost node resolves to the
  identical `ndf` its owning rank assigns — without cross-rank
  communication and without folding a per-node `_ndf` array into
  `fem_hash` (the mechanism the superseded
  [ADR 0033](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0033-s2-emit-wiring-per-node-ndf.md)
  previously relied on).

- **`partition()` after `renumber()` is the canonical order.**
  Call `g.mesh.partitioning.renumber(dim=, method="simple",
  base=1)` first, so the broker captures contiguous 1-based tags;
  then `partition(N)`. Partitioning is a labelling pass over an
  already-numbered model — it does not renumber.

- **`unpartition()` restores monolithic state.** After a partition
  call, `g.mesh.partitioning.unpartition()` removes the partition
  structure and the next `get_fem_data()` produces a broker with
  `len(fem.partitions) == 0`. The emitter then writes byte-identical
  output to a never-partitioned model.


## 10. Reference

- ADR 0027 — Cross-partition MP-constraint emission policy
  (`src/apeGmsh/opensees/architecture/decisions/0027-cross-partition-mp-constraints.md`).
  Defines INV-1 / INV-2 / INV-3 / INV-4 / INV-5 for partitioned
  emission.
- ADR 0023 — Per-zone schema versioning
  (`src/apeGmsh/opensees/architecture/decisions/0023-per-zone-schema-versioning.md`).
  The opensees `2.9.0 → 2.10.0` bump driven by the partition zone
  follows the additive-minor rule + two-version reader window.
- ADR 0024 — Emitter Protocol widening for regions
  (`src/apeGmsh/opensees/architecture/decisions/0024-emitter-protocol-widen-region.md`).
  The per-rank intersection rule in § 9 sits on top of ADR 0024's
  filtering machinery.
- ADR 0026 — H5ModelReader Protocol contract
  (`src/apeGmsh/opensees/architecture/decisions/0026-h5modelreader-protocol-contract.md`).
  Defines the read-side surface that `H5Model.partitions()` slots
  into.
- `guide_meshing.md` — mesh generation, renumbering (RCM /
  Hilbert / METIS-ordering).
- `guide_fem_broker.md` — `FEMData` snapshot lifecycle; how
  `fem.partitions` is built from the same private dicts that power
  `fem.nodes.select(partition=N)`.
- `guide_opensees.md` — the `apeSees` bridge: typed namespaces,
  emit verbs, recorder declarations. Partition-aware emission is
  transparent — every code path in that guide works unchanged
  under partitioning.


??? note "For maintainers — source map"

    Grounded in the current source:

    - `src/apeGmsh/mesh/_mesh_partitioning.py` — `_Partitioning` sub-composite,
      `PartitionInfo` return value, weighted-backend dispatch
    - `src/apeGmsh/_kernel/records/_partitions.py` — `PartitionRecord`
    - `src/apeGmsh/_kernel/record_sets.py` — `PartitionSet`
    - `src/apeGmsh/opensees/apesees.py::_maybe_auto_emit_parallel_numberer`
      — INV-5 auto-emit policy
    - `src/apeGmsh/opensees/emitter/tcl.py::partition_open` /
      `partition_close` — per-rank Tcl brackets + `getPID` shim
    - `src/apeGmsh/opensees/emitter/h5.py` — schema 2.10.0
      `/opensees/partitions/partition_NN/` groups
    - `src/apeGmsh/opensees/architecture/decisions/0027-cross-partition-mp-constraints.md`
      — replication policy
