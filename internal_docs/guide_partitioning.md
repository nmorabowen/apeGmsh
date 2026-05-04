# apeGmsh mesh partitioning and renumbering

A guide to mesh partitioning for parallel solvers and node/element
renumbering for bandwidth reduction.

Grounded in the current source:

- `src/apeGmsh/mesh/_mesh_partitioning.py` — `_Partitioning` sub-composite
- `src/apeGmsh/solvers/Numberer.py` — renumbering algorithms

All snippets assume a meshed model:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
g.model.geometry.add_box(0, 0, 0, 10, 10, 3)
g.mesh.sizing.set_global_size(0.5)
g.mesh.generation.generate(3)
```


## 1. Why renumber

Gmsh assigns node and element tags as it meshes. These tags are
typically non-contiguous (gaps from deleted entities, merged nodes)
and unordered (no spatial locality). This causes two problems:

1. **Sparse arrays** — if max_tag = 50000 but you only have 10000
   nodes, solver arrays waste 5× memory.
2. **High bandwidth** — the stiffness matrix has nonzero entries
   wherever two nodes share an element. If adjacent nodes have
   tags far apart, the bandwidth is large and direct solvers are
   slow.

Renumbering solves both: contiguous IDs (no gaps) and optionally
bandwidth-reducing orderings.


## 2. Simple renumbering

```python
g.mesh.partitioning.renumber(dim=3, method="simple", base=1)
fem = g.mesh.queries.get_fem_data(dim=3)
```

`method="simple"` assigns contiguous IDs starting from `base`:
- Nodes: 1, 2, 3, ..., N
- Elements: 1, 2, ..., E

The ordering is whatever Gmsh's internal traversal produces — no
bandwidth optimization. Use this when you just need contiguous IDs
(e.g., for OpenSees which requires base=1).


## 3. RCM renumbering

```python
g.mesh.partitioning.renumber(dim=3, method="rcm", base=1)
fem = g.mesh.queries.get_fem_data(dim=3)
print(f"Bandwidth: {fem.info.bandwidth}")
```

`method="rcm"` uses the Reverse Cuthill-McKee algorithm to
minimize the semi-bandwidth of the stiffness matrix. This is the
gold standard for direct solvers (Cholesky, LDL^T) — lower
bandwidth means less fill-in, less memory, faster factorization.

**How much does it help?** For a typical 3D tet mesh, RCM reduces
the bandwidth by 3-10× compared to the natural ordering. For
structured hex meshes, the improvement is smaller (the natural
ordering is already decent).

**When to use RCM:**
- Direct solvers (sparse Cholesky, LDL^T) — always
- Iterative solvers (CG, GMRES) — usually helps with ILU
  preconditioner
- Explicit dynamics — no benefit (diagonal mass matrix)


## 4. Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `dim` | 2 | Element dimension for adjacency graph |
| `method` | `"rcm"` | `"simple"` (contiguous), `"rcm"`, `"hilbert"`, or `"metis"` |
| `base` | 1 | Starting ID. 1 for OpenSees/Abaqus, 0 for C-style |


## 5. Mesh partitioning (for MPI)

```python
g.mesh.partitioning.partition(n_parts=4)
```

Splits the mesh into `n_parts` sub-domains using METIS graph
partitioning. This creates Gmsh partition entities that can be
exported to separate mesh files for parallel solvers.

```python
# Remove partitioning (restore monolithic mesh)
g.mesh.partitioning.unpartition()
```

**When to use:**
- MPI-parallel OpenSees runs
- Domain decomposition methods
- Parallel mesh export (each partition → separate file)


## 6. Public API

`renumber()` is the only public entry point on `_Partitioning`. It handles
both nodes and elements in a single call; the per-step helpers
(`_renumber_nodes_simple`, `_renumber_elements_simple`) are private and
should not be invoked directly.


## 7. Standalone Numberer

For renumbering a FEMData object without modifying the Gmsh model:

```python
from apeGmsh.solvers import Numberer

fem = g.mesh.queries.get_fem_data(dim=3)
numb = Numberer(fem)
nm = numb.renumber(method="rcm", base=1)   # → NumberedMesh
print(nm)
```


## 8. Best practices

1. **Always renumber before `get_fem_data()`** — the FEM broker
   captures whatever IDs exist at extraction time. Renumber first
   so the broker has clean, contiguous, optimized IDs.

2. **Use `base=1` for OpenSees** — OpenSees expects 1-based node/
   element tags. `base=0` is for C-style solvers.

3. **Use `method="rcm"` for direct solvers** — the bandwidth
   reduction is free (runs in milliseconds) and can speed up
   factorization by orders of magnitude.

4. **Call renumber only once** — renumbering mutates the Gmsh model.
   Calling it twice produces correct but redundant work.


## See also

- `guide_meshing.md` — mesh generation and sizing
- `guide_fem_broker.md` — extracting FEM data after renumbering
- `guide_opensees.md` — OpenSees model building
