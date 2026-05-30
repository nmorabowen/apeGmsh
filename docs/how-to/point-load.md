# Add a point load

Apply a concentrated force and/or moment at every node of a named target
(a physical group or label). Reach for this when the load lives on a node
set — a column top, an anchor point, a loaded pin.

## Recipe

Declare the load on the session with `g.loads.point(...)`, scoped inside a
named `pattern`. In v2.0 a session load **auto-emits** to the solver — the
`apeSees(fem)` bridge synthesizes a `Plain` pattern and fans the force across
the target's nodes for you. You do **not** re-declare it on the bridge.

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="cantilever") as g:
    box = g.model.geometry.add_box(0, 0, 0, 10, 1, 1, label="body")

    eps = 1e-6
    base = g.model.queries.entities_in_bounding_box(
        -eps, -eps, -eps, eps, 1 + eps, 1 + eps, dim=2)
    tip = g.model.queries.entities_in_bounding_box(
        10 - eps, -eps, -eps, 10 + eps, 1 + eps, 1 + eps, dim=2)

    g.physical.add(3, ["body"], name="Body")
    g.physical.add_surface([t for _, t in base], name="Base")
    g.physical.add_surface([t for _, t in tip], name="Tip")

    # Concentrated force on the tip node set — auto-emits to the solver.
    with g.loads.pattern("Lateral"):
        g.loads.point("Tip", force_xyz=(0.0, 0.0, -5e4))

    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)   # <- loads resolve here

# Bridge: do NOT re-declare the point load — it is already emitted.
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(pg="Body", material=conc)
ops.fix(pg="Base", dofs=(1, 1, 1))
ops.py("out/cantilever.py")
```

`g.loads.point("Tip", ...)` resolves to one nodal load per node of the `Tip`
target. **Every targeted node receives the same `force_xyz`** — so for a
single concentrated force apply it to a single-node target (e.g. a `point`
label or `point_closest`), not a multi-node face. Add `moment_xyz=(Mx, My, Mz)`
for rotational DOFs; either vector may be omitted, but at least one must be set.

## Notes / gotchas

- **Don't double-declare.** A `g.loads.point` is *already* emitted by the
  bridge. If you *also* add `p.load(...)` on a bridge `pattern.Plain` for the
  same nodes, the force lands **twice** (reactions come out at exactly 2×).
  Pick one channel — session `g.loads.*` **or** bridge `p.load`, never both.
- **One force per node, not per target.** `point` gives every node of the
  target the full vector. To split a single force across a face, use
  `g.loads.face_load(...)`; to hit one node, target a single-node entity.
- **No raw tags.** Target by PG name or label (`"Tip"`), never by entity tag.
- **2-D moment:** pass a length-1 tuple `moment_xyz=(Mz,)`.
- **Coordinate-driven variant:** when the point isn't on a named entity, use
  `g.loads.point_closest(xyz, ...)` to snap to the nearest mesh node.

## Bridge alternative (`p.load`)

If you'd rather drive the force from the bridge — e.g. to tie it to a custom
`timeSeries` — skip the `g.loads.point` declaration entirely and apply it on a
bridge pattern instead:

```python
ts = ops.timeSeries.Linear()
with ops.pattern.Plain(series=ts) as p:
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))
```

`p.load` fans `pg=` across the group's nodes at build time. Use **exactly one**
of the two channels for any given load.

## See also

- Concept: [Loads guide](../internal_docs/guide_loads.md) — point, line, surface, body loads and the tributary/consistent reduction
- Concept: [The apeSees bridge — auto-emit vs re-declare](../internal_docs/guide_opensees.md)
- Tutorial: [10-minute first model](../tutorials/first-model.md)
- API: `apeGmsh.core.LoadsComposite.point`
