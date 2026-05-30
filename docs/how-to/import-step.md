# Import & heal a STEP file

Pull external CAD (`.step` / `.stp`) into a session, clean up the dirty
geometry that legacy exporters ship, and turn it into a meshable,
solver-ready model. This is the reason many people reach for apeGmsh —
but remember: STEP gives you **geometry only**. Physical groups, mesh
sizing, and meshing are still yours to define.

## The recipe

Look before you heal: `diagnose()` reports the geometry's health without
touching it, so you only reach for `heal=` when there are real slivers.

```python
from apeGmsh import apeGmsh

g = apeGmsh(model_name="bracket")
g.begin()

# 1. Import raw, then inspect. load_step does NOT create physical groups.
g.model.io.load_step("bracket.step")          # default: highest_dim_only=True → volumes only

report = g.model.io.diagnose()                # -> ImportHealth, non-mutating
print(report)                                 # solids, dim_counts, short_edges, tiny_faces, suggested_tolerance

# 2. Heal only if slivers are present. heal="auto" is scale-aware
#    (≈ 1e-6 · bbox diagonal) — the right default across unit systems.
if report.is_suspect:
    imported = g.model.io.load_step(
        "bracket.step", heal="auto", dedupe=True   # re-import clean
    )
else:
    imported = g.model.io.load_step("bracket.step")

bodies = imported[3]                          # all imported volume tags

# 3. Name what the solver cares about — by SELECTION/QUERY, never raw tags.
g.physical.add(3, bodies, name="Steel")

base = g.model.queries.entities_in_bounding_box(
    -1e3, -1e3, -1e-3, 1e3, 1e3, 1e-3, dim=2)     # -> [(dim, tag), ...]
g.physical.add(2, [t for _, t in base], name="Fixed_Support")

# 4. (multi-body assemblies) make the interface conformal before meshing.
g.model.queries.make_conformal(dims=[3])

# 5. Size, mesh, hand the snapshot to the bridge.
g.mesh.sizing.set_size_global(max_size=5.0)
g.mesh.generation.generate(dim=3)
fem = g.mesh.queries.get_fem_data(dim=3)

g.end()
```

From here build OpenSees through the typed `apeSees(fem)` bridge and
target everything by the physical-group names above (`"Steel"`,
`"Fixed_Support"`) — never raw entity tags.

## Notes / gotchas

- **`heal=` is one-shot import+clean.** `heal="auto"` (== `heal=True`)
  derives a scale-aware tolerance; a `float` pins an absolute tolerance.
  A fixed absolute tolerance is meaningless across unit systems — a
  `1e-3` mm gap and a `1e-3` m gap differ by 1000×. Prefer `"auto"`.
- **`diagnose()` never mutates.** It scans the live OCC geometry and
  returns a frozen `ImportHealth`; `is_suspect` is `True` only when
  slivers (`short_edges` / `tiny_faces`) exist. A surface-only import is
  *not* flagged — shell models import that way on purpose. On a *raw*
  import apeGmsh already emits a `WarnGeomImportHealth` advisory for you.
- **`dedupe=True`** merges coincident entities that STEP assemblies
  often repeat across bodies. Runs after healing when both are set.
- **STEP carries no physical groups.** After `load_step` you always
  define your own. Need named regions to survive the CAD round-trip?
  Export to `.msh` from Gmsh, or use DXF layers (`load_dxf`).
- **Multi-body ≠ conformal.** Imported bodies come in as independent
  solids. For a shared-node mesh across an interface, run
  `make_conformal` (or `g.model.boolean.fragment`) *before* meshing.
- **One kernel per session.** `load_step` uses the OCC kernel; don't mix
  in geo-kernel entities or booleans between them will fail.

## See also

- Concept guide: [Importing CAD and meshes](../internal_docs/guide_cad_import.md) — STEP/IGES/`.msh`, the full healing knob set, and the three import entry points.
- Example: [examples index](../examples/index.md) — the STEP-import rung (CAD → physical groups → mesh → solve).
- Related recipe: [Tag a face as a physical group](../internal_docs/guide_queries.md) — naming imported faces by bounding-box / boundary query.
- API: [`g.model.io`](../api/model.md) — `load_step`, `heal_shapes`, `diagnose` / `ImportHealth`, and `make_conformal`.
