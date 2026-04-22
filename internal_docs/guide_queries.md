# apeGmsh model queries

A guide to the query operations available on `g.model.queries` — removing
entities, topology queries, bounding boxes, center of mass, boundary
extraction, adjacency traversal, and the model registry.

Grounded in the current source:

- `src/apeGmsh/core/_model_queries.py` — the `_Queries` sub-composite

All snippets assume an open session:

```python
from apeGmsh import apeGmsh
g = apeGmsh(model_name="demo")
g.begin()
```


## 1. Entity removal

```python
# Remove a single volume (and its bounding surfaces, curves, points)
g.model.queries.remove(vol_tag, dim=3, recursive=True)

# Remove surfaces only (leaves curves and points intact)
g.model.queries.remove([surf1, surf2], dim=2, recursive=False)
```

`recursive=True` removes the entity AND all its lower-dimensional
boundaries. `recursive=False` removes only the specified entity —
but this can leave dangling boundaries that confuse later operations.

**Rule of thumb:** use `recursive=True` unless you specifically need
to keep the boundary for a different entity.


## 2. Duplicate removal and healing

```python
# Remove duplicate entities (same geometry, different tags)
g.model.queries.remove_duplicates(tolerance=1e-6)
```

After importing multiple STEP files or running boolean operations,
the model can accumulate duplicate entities — surfaces that occupy
the same space but have different tags. `remove_duplicates()` merges
them into single entities.

```python
# Make conformal (fragment everything against everything)
g.model.queries.make_conformal(tolerance=1e-6, dim=3)
```

`make_conformal()` is a convenience wrapper around
`gmsh.model.occ.fragment()` that fragments ALL entities of a given
dimension against each other. The result is a conformal mesh — shared
faces at every interface. This is what `g.parts.fragment_all()` does
under the hood, but without part tracking.


## 3. Bounding box

```python
# Bounding box of a single entity
xmin, ymin, zmin, xmax, ymax, zmax = g.model.queries.bounding_box(
    vol_tag, dim=3
)

# Bounding box of multiple entities
bbox = g.model.queries.bounding_box([tag1, tag2], dim=3)
```

Returns the axis-aligned bounding box (AABB) as six floats.
For multiple entities, returns the enclosing box that contains all of them.


## 4. Center of mass

```python
cx, cy, cz = g.model.queries.center_of_mass(vol_tag, dim=3)
```

Returns the geometric centroid of the entity, computed by the OCC
kernel. For volumes, this is the volume centroid. For surfaces, the
area centroid. For curves, the length centroid.

Useful for:
- Placing labels at entity centers
- Checking that boolean operations produced expected geometry
- Constraint resolution (COM matching in the Parts system)


## 5. Mass (volume/area/length)

```python
measure = g.model.queries.mass(vol_tag, dim=3)  # volume in model units
```

Returns the geometric measure: volume for dim=3, area for dim=2,
length for dim=1. The name "mass" is inherited from the Gmsh API —
it computes the geometric measure, not physical mass (no density).

Useful for sanity checks:

```python
vol = g.model.queries.mass(slab_tag, dim=3)
print(f"Slab volume: {vol:.2f} m³")
expected_mass = vol * 2400  # density × volume
```


## 6. Boundary extraction

```python
# Get the boundary surfaces of a volume
boundary_surfs = g.model.queries.boundary(vol_tag, dim=3)
# Returns list of (dim-1, tag) pairs: [(2, s1), (2, s2), ...]

# Get the boundary curves of a surface
boundary_curves = g.model.queries.boundary(surf_tag, dim=2)
# Returns list of (1, tag) pairs
```

Boundary extraction returns the lower-dimensional entities that
form the boundary of the input entity. This is the OCC topological
boundary, not a mesh boundary.

Use cases:
- Finding surfaces to apply boundary conditions
- Traversing the model topology
- Identifying shared faces between volumes


## 7. Adjacency queries

```python
# Get all volumes that share a face with this volume
adj_vols = g.model.queries.adjacencies(vol_tag, dim=3)
```

Returns entities that share boundary entities with the input.
Two volumes are adjacent if they share at least one face.

Useful for:
- Identifying which parts touch
- Validating that fragmentation created expected interfaces


## 8. Spatial search

```python
# Find all entities inside a bounding box
entities = g.model.queries.entities_in_bounding_box(
    x0=0, y0=0, z0=0,
    x1=10, y1=10, z1=10,
    dim=3,
)
# Returns list of (dim, tag) pairs
```

Finds all entities of a given dimension whose bounding box
overlaps the specified region. This is an approximate search — it
finds entities whose AABB intersects, not whose actual geometry
intersects.


## 9. Model registry

```python
df = g.model.queries.registry()
```

Returns a pandas DataFrame listing every entity in the model with:
- `dim` — dimension (0, 1, 2, 3)
- `tag` — Gmsh entity tag
- `type` — entity type name
- `bbox` — bounding box
- `com` — center of mass
- `mass` — geometric measure

This is the most complete snapshot of the model state. Useful for
debugging and introspection:

```python
df = g.model.queries.registry()
# Filter to volumes only
vols = df[df['dim'] == 3]
print(f"{len(vols)} volumes, total volume: {vols['mass'].sum():.2f}")
```


## See also

- `guide_basics.md` — geometry creation and boolean operations
- `guide_selection.md` — spatial selection queries
- `guide_cad_import.md` — STEP/IGES I/O operations
