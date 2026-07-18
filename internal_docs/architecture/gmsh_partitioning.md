# Gmsh Mesh Partitioning

Domain decomposition: how Gmsh partitions meshes, the METIS graph partitioner, partition entities, ghost elements, and the pipeline to parallel OpenSees.

---

## What Partitioning Is

Partitioning divides a mesh into $N$ non-overlapping subdomains (partitions). Each element belongs to exactly one partition. The goal is twofold:

1. **Load balancing** — each partition should have roughly the same computational cost (element count, or weighted element count)
2. **Minimal interface** — the boundary between partitions (shared nodes/faces) should be as small as possible, since inter-partition communication is expensive in parallel solvers

```
Original mesh                    Partitioned (N=3)
┌─────────────────────┐          ┌───────┬──────┬──────┐
│                     │          │       │      │      │
│                     │   ──►    │  P1   │  P2  │  P3  │
│                     │          │       │      │      │
└─────────────────────┘          └───────┴──────┴──────┘
                                    ↑         ↑
                                 interfaces (shared nodes)
```

Partitioning is a **post-meshing** operation. You generate the full mesh first, then partition it.

```python
gmsh.model.mesh.generate(3)
gmsh.model.mesh.partition(N)
```

---

## METIS Graph Partitioner

Gmsh uses **METIS** as its primary partitioning algorithm. METIS converts the mesh into a graph (elements → nodes, adjacency → edges) and partitions the graph to minimize the edge cut while balancing node weights.

### The mesh-to-graph conversion

Each mesh element becomes a graph node. Two graph nodes are connected by an edge if the corresponding elements share a face (3D) or an edge (2D). The weight of a graph node can be the computational cost of the element.

```
Mesh elements:                Graph:
┌─────┬─────┐                 (e1)───(e2)
│ e1  │ e2  │                  │       │
├─────┼─────┤        →        (e3)───(e4)
│ e3  │ e4  │
└─────┴─────┘
```

METIS then partitions this graph into $N$ subsets, minimizing the number of edges that cross partition boundaries (the "edge cut").

### Two algorithms

| Option                 | Code | Method               | Best for                        |
| ---------------------- | ---- | -------------------- | ------------------------------- |
| `Mesh.MetisAlgorithm`  | `1`  | Recursive bisection  | Small $N$ (< 8), quality focus  |
| `Mesh.MetisAlgorithm`  | `2`  | K-way                | Large $N$, speed focus          |

**Recursive bisection** splits the graph in half, then splits each half, recursively. Produces high-quality partitions but cost grows with $N$.

**K-way** partitions directly into $N$ parts in a single pass. Faster for large $N$, slightly lower quality.

```python
gmsh.option.setNumber("Mesh.MetisAlgorithm", 1)   # default: recursive
```

### Objective function

METIS can optimize for two objectives:

```python
gmsh.option.setNumber("Mesh.MetisObjective", 1)
# 1 = minimize edge-cut (default) — fewer shared faces between partitions
# 2 = minimize communication volume — total data exchanged between partitions
```

**Edge-cut** counts the number of edges crossing partition boundaries. **Communication volume** considers that a node shared by $k$ partitions requires $k-1$ communications, and minimizes the total. Communication volume is usually the better metric for parallel FEM, but edge-cut is the default and works well for most structural problems.

### Load balancing weights

By default, all elements have equal weight. You can assign per-element-type weights to account for different computational costs:

```python
# Hex elements cost more than tets (more integration points, larger stiffness matrix)
gmsh.option.setNumber("Mesh.PartitionHexWeight", 8)
gmsh.option.setNumber("Mesh.PartitionTetWeight", 4)
gmsh.option.setNumber("Mesh.PartitionTriWeight", 2)
gmsh.option.setNumber("Mesh.PartitionQuadWeight", 3)
gmsh.option.setNumber("Mesh.PartitionLineWeight", 1)
gmsh.option.setNumber("Mesh.PartitionPrismWeight", 6)
gmsh.option.setNumber("Mesh.PartitionPyramidWeight", 5)
# -1 = automatic (default)
```

### Load imbalance tolerance

```python
gmsh.option.setNumber("Mesh.MetisMaxLoadImbalance", -1)
# -1 = default (30 for K-way, 1 for Recursive)
# Higher values allow more imbalance in exchange for lower edge-cut
```

### Additional METIS controls

```python
gmsh.option.setNumber("Mesh.MetisEdgeMatching", 2)
# 1 = Random, 2 = Sorted Heavy-Edge (default) — coarsening strategy

gmsh.option.setNumber("Mesh.MetisMinConn", -1)
# -1 = default — minimize maximum connectivity of partitions

gmsh.option.setNumber("Mesh.MetisRefinementAlgorithm", 2)
# 1 = FM-based cut refinement, 2 = Greedy (default), 3 = Two-sided node FM
```

---

## Partition Entities

After partitioning, Gmsh creates **new discrete entities** called partition entities. This is the key design decision: partitioning doesn't just label elements — it restructures the model topology.

Each partition entity:
- Has its own `(dim, tag)` — a regular entity in the model
- Belongs to one or more partitions (interface entities belong to multiple)
- Tracks its **parent entity** — the original entity it was carved from
- Has a full BRep boundary representation (if `PartitionCreateTopology = 1`)

```python
gmsh.model.mesh.partition(3)

# The model now has new entities
entities = gmsh.model.getEntities()
for e in entities:
    partitions = gmsh.model.getPartitions(e[0], e[1])
    if len(partitions) > 0:
        parentDim, parentTag = gmsh.model.getParent(e[0], e[1])
        print(f"Entity {e} → partitions {partitions}, parent ({parentDim}, {parentTag})")
```

### Partition topology

With `PartitionCreateTopology = 1` (default), Gmsh creates a full BRep for the partitioned model:

- **Partition volumes** — each partition's portion of the original volume
- **Partition surfaces** — interior surfaces of a partition + interface surfaces shared between partitions
- **Partition curves** — edges of partition surfaces
- **Partition points** — vertices

Interface entities (surfaces between two partitions) belong to **both** partitions. This is how shared nodes are identified.

```
Original model:                Partitioned (N=2):
┌───────────────┐              ┌───────┬───────┐
│               │              │       │       │
│  Volume (3,1) │    ──►       │ (3,5) │ (3,6) │   ← partition volumes
│               │              │  P1   │  P2   │
└───────────────┘              └───────┴───────┘
                                       │
                                    (2,10)          ← interface surface
                                  partitions=[1,2]     belongs to both
```

### Configuration options

```python
gmsh.option.setNumber("Mesh.PartitionCreateTopology", 1)
# 1 (default) — create full BRep for partition entities
# 0 — skip topology creation (lighter, but no boundary queries)

gmsh.option.setNumber("Mesh.PartitionCreatePhysicals", 1)
# 1 (default) — create physical groups for partitions based on existing PGs
# 0 — don't create partition physical groups

gmsh.option.setNumber("Mesh.PartitionSplitMeshFiles", 0)
# 0 (default) — write everything to one file
# 1 — write one .msh file per partition
```

### Querying partition entities

```python
# Get all entities and filter for partitioned ones
for dim, tag in gmsh.model.getEntities():
    parts = gmsh.model.getPartitions(dim, tag)
    if len(parts) == 0:
        continue    # not a partition entity
    
    parentDim, parentTag = gmsh.model.getParent(dim, tag)
    entity_type = gmsh.model.getType(dim, tag)
    boundary = gmsh.model.getBoundary([(dim, tag)])
    
    if len(parts) == 1:
        # Interior entity — belongs to one partition
        print(f"Interior: ({dim},{tag}) in P{parts[0]}, parent ({parentDim},{parentTag})")
    else:
        # Interface entity — shared between partitions
        print(f"Interface: ({dim},{tag}) shared by P{parts}, parent ({parentDim},{parentTag})")
```

---

## Ghost Elements

Ghost elements are copies of elements near partition boundaries. They belong to a neighboring partition but are stored as "ghosts" in the local partition so that each processor has the information it needs for assembly without communication.

```
Partition 1              Partition 2
┌──────────┐            ┌──────────┐
│          │            │          │
│  e1  e2  │ e3  e4     e3  e4 │ e5  e6  │
│          │            │          │
└──────────┘            └──────────┘
         ^^^^              ^^^^
         ghost in P2       ghost in P1
```

Elements `e3` and `e4` are in the boundary region. Partition 1 owns them, but Partition 2 gets ghost copies (and vice versa for `e3`/`e4` in P1's perspective).

### Enabling ghost cells

Ghost cells are **off by default**. Enable them before partitioning:

```python
gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 1)
gmsh.model.mesh.partition(N)
```

### Querying ghost elements

After partitioning with ghost cells, ghost entities appear in the model. Query them:

```python
for dim, tag in gmsh.model.getEntities():
    ghostTags, ghostPartitions = gmsh.model.mesh.getGhostElements(dim, tag)
    if len(ghostTags) > 0:
        print(f"Entity ({dim},{tag}) has {len(ghostTags)} ghost elements")
        for etag, part in zip(ghostTags, ghostPartitions):
            print(f"  Element {etag} is ghost from partition {part}")
```

### Why ghost elements matter

In parallel FEM, each processor assembles its local stiffness matrix. For elements adjacent to the partition boundary, the processor needs information about neighboring elements to:

- Correctly assemble shared DOFs
- Compute inter-element quantities (stress recovery, error estimation)
- Apply flux-based boundary conditions at partition interfaces

Without ghost elements, this information requires explicit communication. With ghost elements, each processor has a local copy of the boundary layer and can assemble without waiting.

---

## Interface Handling

The partition interface is where the subdomains meet. Shared nodes at the interface need special treatment in parallel solvers.

### Identifying interface nodes

Interface nodes are classified on partition entities that belong to multiple partitions:

```python
interface_nodes = set()

for dim, tag in gmsh.model.getEntities():
    parts = gmsh.model.getPartitions(dim, tag)
    if len(parts) > 1:  # interface entity
        nodeTags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
        interface_nodes.update(int(t) for t in nodeTags)

print(f"Interface nodes: {len(interface_nodes)}")
```

### Identifying interface elements

Elements on the interface are those with at least one node on an interface entity:

```python
# Alternatively, use getBoundary on partition volumes to get interface surfaces
for dim, tag in gmsh.model.getEntities():
    parts = gmsh.model.getPartitions(dim, tag)
    if len(parts) == 1 and dim == 3:  # interior partition volume
        boundary = gmsh.model.getBoundary([(dim, tag)])
        for bdim, btag in boundary:
            bparts = gmsh.model.getPartitions(bdim, btag)
            if len(bparts) > 1:
                print(f"Partition {parts[0]}: interface surface ({bdim},{btag}) "
                      f"shared with partitions {bparts}")
```

### Partition boundary data for solver setup

For parallel OpenSees, you need to know at each partition boundary:
- Which nodes are shared
- Which partitions share them
- Which DOFs need to be communicated

This can be extracted by iterating partition entities with `len(partitions) > 1`.

---

## Manual Partitioning

Instead of METIS, you can assign elements to partitions explicitly:

```python
gmsh.model.mesh.partition(
    N,
    elementTags=[e1, e2, e3, e4, e5, e6],
    partitions=[1, 1, 1, 2, 2, 2]
)
```

The `elementTags` and `partitions` arrays must be the same length. Each element is assigned to the specified partition.

### Use cases

- **Physics-based partitioning** — separate soil and structure into different partitions for different solver strategies
- **Manual load balancing** — when element cost varies significantly and METIS weights aren't sufficient
- **Predefined substructuring** — when the domain decomposition follows a specific engineering logic (e.g., one partition per floor of a building)

### Combining with physical groups

A practical pattern: use physical groups to define regions, then assign partitions based on region membership:

```python
# Assume physical groups define structural regions
soil_entities = gmsh.model.getEntitiesForPhysicalGroup(3, soil_pg_tag)
struct_entities = gmsh.model.getEntitiesForPhysicalGroup(3, struct_pg_tag)

# Collect element tags per region
soil_elems = []
for tag in soil_entities:
    _, etags, _ = gmsh.model.mesh.getElements(3, tag)
    for arr in etags:
        soil_elems.extend(arr)

struct_elems = []
for tag in struct_entities:
    _, etags, _ = gmsh.model.mesh.getElements(3, tag)
    for arr in etags:
        struct_elems.extend(arr)

# Assign partitions
all_tags = soil_elems + struct_elems
all_parts = [1]*len(soil_elems) + [2]*len(struct_elems)
gmsh.model.mesh.partition(2, elementTags=all_tags, partitions=all_parts)
```

---

## MSH4 Partitioned Format

The MSH4 file format natively supports partitioned meshes. When you write a partitioned mesh, the file contains additional sections.

### `$Entities` section — partition entities

Partition entities appear alongside regular entities, with additional fields:

```
$Entities
numPoints numCurves numSurfaces numVolumes
...
$PartitionedEntities
numPartitions numGhostEntities
ghostEntityDim ghostEntityTag ghostPartition ...
numPoints numCurves numSurfaces numVolumes
pointTag parentDim parentTag numPartitions partitions... ...
...
$EndPartitionedEntities
```

Each partition entity records:
- Its own `(dim, tag)`
- The parent entity it came from `(parentDim, parentTag)`
- The partition(s) it belongs to

### `$Elements` section — partition assignment

Elements are grouped by entity (including partition entities). Since each partition entity belongs to a specific partition, the element-to-partition mapping is implicit through the entity classification.

### Split files

With `Mesh.PartitionSplitMeshFiles = 1`, Gmsh writes one `.msh` file per partition:

```
model_1.msh    ← partition 1 elements + shared interface nodes
model_2.msh    ← partition 2 elements + shared interface nodes
model_3.msh    ← partition 3 elements + shared interface nodes
```

Each file is a self-contained mesh for its partition, including the interface nodes (duplicated across files). This is the format most parallel solvers expect.

### Unpartitioning

To restore the original (unpartitioned) model:

```python
gmsh.model.mesh.unpartition()
# Removes all partition entities, restores original model structure
```

---

## OpenSees Parallel Pipeline

OpenSees provides two parallel frameworks: **OpenSeesSP** (single-interpreter, parallel equation solving) and **OpenSeesMP** (multiple interpreters, domain decomposition). Gmsh partitioning maps to both, but the workflow differs.

### OpenSeesSP

In OpenSeesSP, the model is built on all processors, but the equation system is solved in parallel. The partitioning is handled internally by OpenSees — you don't typically use Gmsh partitioning here.

However, Gmsh partitioning can still be useful for **pre-computing** the domain decomposition and passing it to OpenSees via the numberer, to ensure the partitioning matches your engineering intent.

### OpenSeesMP — the primary use case

In OpenSeesMP, each processor builds and owns a portion of the model. This is where Gmsh partitioning directly maps:

```
Gmsh partition 1  →  Processor 0: builds elements + nodes for P1
Gmsh partition 2  →  Processor 1: builds elements + nodes for P2
Gmsh partition N  →  Processor N-1: builds elements + nodes for PN
```

The workflow:

```python
# Step 1: Partition in Gmsh (pre-processing, single process)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.partition(num_processors)

# Step 2: Extract per-partition data
for proc in range(num_processors):
    partition_id = proc + 1
    
    # Find partition entities for this processor
    nodes_for_proc = {}
    elements_for_proc = []
    
    for dim, tag in gmsh.model.getEntities():
        parts = gmsh.model.getPartitions(dim, tag)
        if partition_id in parts:
            # Get nodes
            ntags, ncoords, _ = gmsh.model.mesh.getNodes(dim, tag)
            coords = ncoords.reshape(-1, 3)
            for i, nt in enumerate(ntags):
                nodes_for_proc[int(nt)] = coords[i]
            
            # Get elements (only from entities owned by this partition)
            if len(parts) == 1:  # interior entity
                etypes, etags, enodes = gmsh.model.mesh.getElements(dim, tag)
                # ... collect elements
    
    # Step 3: Write per-partition OpenSees script or data file
    write_opensees_partition(proc, nodes_for_proc, elements_for_proc)
```

### Interface nodes in OpenSeesMP

Shared nodes at partition boundaries need special handling. In OpenSeesMP, interface nodes exist on **multiple processors**. The framework uses `send`/`recv` or the `Parallel` domain to synchronize DOFs at shared nodes.

The critical data to extract from Gmsh:

```python
# For each partition, identify which of its nodes are shared
shared_node_map = {}  # node_tag → set of partition IDs

for dim, tag in gmsh.model.getEntities():
    parts = gmsh.model.getPartitions(dim, tag)
    if len(parts) > 1:  # interface entity
        ntags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
        for nt in ntags:
            nt = int(nt)
            if nt not in shared_node_map:
                shared_node_map[nt] = set()
            shared_node_map[nt].update(parts)
```

This map tells each processor which nodes it shares with which neighbors — essential for setting up the parallel communication pattern.

### sendSelf / recvSelf

In OpenSees, every element and material implements `sendSelf()` and `recvSelf()` for parallel communication (see the OpenSees skill for details on `classTags.h` and `FEM_ObjectBroker`). The partitioning determines what gets sent where:

- Elements are created only on their owning processor
- Nodes at interfaces are created on all sharing processors
- Materials are local to each element (no sharing)
- Loads applied to interface nodes must be applied on all sharing processors (or on one and communicated)

### Boundary conditions at partition interfaces

Boundary conditions on shared nodes need care:

- **Displacement BCs** (fix, fixX, etc.) — apply on all processors that own the node
- **Loads** — apply on **one** processor only (to avoid double-counting), or split proportionally
- **Constraints** (equalDOF, rigidLink) — both constrained and retained nodes must be on the same processor, or the constraint must span the interface using MP_Constraint

### Practical recommendations for apeGmsh

1. **Partition after physical groups** — PGs define material regions and BCs; partition within these regions, not across them when possible
2. **Use `PartitionCreatePhysicals = 1`** — this automatically creates partition-level physical groups from the original PGs, preserving the material/BC labels per partition
3. **Extract interface data explicitly** — don't rely on ghost cells for OpenSees; extract shared nodes and their partition membership directly
4. **Write per-partition input files** — each processor reads its own file, which is simpler than having all processors parse a global file
5. **Validate partition quality** — check that each partition has roughly equal element count and that the interface area is reasonable:

```python
for p in range(1, N+1):
    elem_count = 0
    for dim, tag in gmsh.model.getEntities():
        parts = gmsh.model.getPartitions(dim, tag)
        if parts == [p] and dim == 3:
            _, etags, _ = gmsh.model.mesh.getElements(dim, tag)
            elem_count += sum(len(e) for e in etags)
    print(f"Partition {p}: {elem_count} elements")
```

---

## SimplePartition Plugin

Besides METIS, Gmsh offers a `SimplePartition` plugin for geometric (non-graph-based) partitioning:

```python
gmsh.plugin.setNumber("SimplePartition", "NumSlicesX", 4)
gmsh.plugin.setNumber("SimplePartition", "NumSlicesY", 2)
gmsh.plugin.setNumber("SimplePartition", "NumSlicesZ", 1)
gmsh.plugin.run("SimplePartition")
```

This creates axis-aligned slices — a chessboard-like partition. No graph analysis, no optimization. Useful for:

- Simple geometries where spatial partitioning is obvious
- Debugging partitioning workflows (predictable, visual results)
- Cases where METIS isn't available

Not recommended for production structural analysis — it ignores element connectivity and can create poorly balanced partitions for irregular meshes.

---

## Summary — Partitioning Data Flow

```
Mesh (unpartitioned)
    │
    │  gmsh.model.mesh.partition(N)
    │  (METIS graph partitioning or manual assignment)
    ▼
Partition entities created
    │
    ├── Interior entities:  (dim, tag) → partitions=[p]
    │       └── Elements owned exclusively by partition p
    │
    ├── Interface entities: (dim, tag) → partitions=[p1, p2, ...]
    │       └── Shared nodes at partition boundary
    │
    └── Ghost entities (optional): (dim, tag) → ghost elements from neighbor
    │
    │  Extract per-partition data
    ▼
Per-processor:
    ├── Nodes:        own + shared interface nodes
    ├── Elements:     own only (no ghosts for solver)
    ├── Shared nodes: node_tag → [partitions that share it]
    ├── BCs/Loads:    from physical groups (partition-level PGs)
    └── Materials:    from physical groups (partition-level PGs)
    │
    │  Write per-partition solver input
    ▼
OpenSeesMP: each processor reads its partition file
```
