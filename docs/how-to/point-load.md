# Add a point load

Apply a concentrated force and/or moment at every node of a named target
(a physical group or label). Reach for this when the load lives on a node
set — a column top, an anchor point, a loaded pin.

## Recipe

Declare the load on the session with `g.loads.point.force(...)`, scoped inside a
named **case**. Loads are **opt-in** (ADR 0051): the session declaration
resolves onto `fem.nodes.loads` and persists to `model.h5`, but it reaches the
runnable deck only when a bridge pattern **imports** the case with
`p.from_model("Lateral")`. (Alternatively, skip `g.loads` entirely and author
the force directly on a bridge pattern with `p.load(...)` — see the bridge
section below.)

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

    # Concentrated force on the tip node set — grouped under case "Lateral".
    with g.loads.case("Lateral"):
        g.loads.point.force("Tip", (0.0, 0.0, -5e4))

    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)   # <- loads resolve here

# Bridge: import the "Lateral" case into a pattern so it reaches the deck.
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(pg="Body", material=conc)
ops.fix(pg="Base", dofs=(1, 1, 1))
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("Lateral")
ops.py("out/cantilever.py")
```

`g.loads.point.force("Tip", ...)` resolves to one nodal load per node of the `Tip`
target. **Every targeted node receives the same `force`** — so for a
single concentrated force apply it to a single-node target (e.g. a `point`
label or `point.force_closest`), not a multi-node face. Use
`g.loads.point.moment("Tip", (Mx, My, Mz))`
for rotational DOFs.

## Notes / gotchas

- **Import the case, or it won't apply.** A `g.loads.point.force` is *not*
  auto-emitted — without `p.from_model("Lateral")` (or an ad-hoc `p.load`) the
  force never reaches the deck. The deck is authoritative: the bridge applies
  exactly the cases you import and doesn't audit the geometry's case-list, so a
  case you don't import is simply not applied. Because nothing auto-emits, there
  is no 2× double-count trap.
- **One force per node, not per target.** `point.force` gives every node of the
  target the full vector. To split a single force across a face, use
  `g.loads.surface.force_resultant_center_mass(...)`; to hit one node, target a single-node entity.
- **No raw tags.** Target by PG name or label (`"Tip"`), never by entity tag.
- **2-D moment:** pass a length-1 tuple `g.loads.point.moment("Tip", (Mz,))`.
- **Coordinate-driven variant:** when the point isn't on a named entity, use
  `g.loads.point.force_closest(xyz, ...)` to snap to the nearest mesh node.

## Bridge alternative (`p.load`)

If you'd rather drive the force entirely from the bridge — e.g. you have no
session-side `g.loads` declaration to import — author it directly on a pattern:

```python
ts = ops.timeSeries.Linear()
with ops.pattern.Plain(series=ts) as p:
    p.load(pg="Tip", forces=(0.0, 0.0, -5e4))
```

`p.load` fans `pg=` across the group's nodes at build time. `p.from_model(case)`
and `p.load(...)` mix freely in the same pattern — import the session cases you
want and add ad-hoc bridge loads alongside.

## See also

- Concept: [Loads guide](../internal_docs/guide_loads.md) — point, line, surface, body loads and the tributary/consistent reduction
- Concept: [The apeSees bridge — opt-in load import (`from_model`)](../internal_docs/guide_opensees.md)
- Tutorial: [10-minute first model](../tutorials/first-model.md)
- API: `apeGmsh.core.LoadsComposite.point`

---

*Next: [Fix supports & boundary conditions](supports-bcs.md).*
