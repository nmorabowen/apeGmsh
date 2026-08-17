# Tie two non-matching meshes

Join two separately-meshed parts across a shared interface without
remeshing them conformally. Reach for this when two members meet at a
face or a point but their nodes don't line up.

Two recipes, picked by what the interface looks like:

- **`g.constraints.tie`** — non-conformal surfaces. Each slave node is
  projected onto the closest master face and its DOFs are interpolated
  from that face's shape functions (`u_slave = Σ Nᵢ · u_masterᵢ`). The
  meshes do **not** need matching nodes.
- **`g.constraints.equal_dof`** — co-located nodes. The resolver finds
  master/slave node pairs whose coordinates match within `tolerance` and
  ties the selected DOFs (`u_slave[i] = u_master[i]`). Use this when the
  two parts genuinely share node positions at the boundary.

Both are **declared pre-mesh against physical-group / part names**,
**resolved at `get_fem_data`**, and **auto-emitted by the typed bridge**
(ADR 0022) — you never hand-write `ops.equalDOF` or the
`ASDEmbeddedNodeElement` penalty elements.

## Recipe

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

g = apeGmsh(model_name="tie_demo")
g.begin()
# ... build two parts that meet at an interface, mesh them ...

# --- Non-matching surfaces: interpolate slave DOFs from the master face
g.constraints.tie(
    "flange_surface",   # master part/PG label
    "web_surface",      # slave part/PG label
    tolerance=1.0,      # max projection distance slave -> master face
)

# --- OR: co-located nodes -> tie the matching pairs directly
g.constraints.equal_dof(
    "beam_a_end",
    "beam_b_end",
    dofs=[1, 2, 3],     # 1-based; couple translations only
    tolerance=1e-6,
)

# Resolve: definitions become concrete node-level records on the broker
fem = g.mesh.queries.get_fem_data(dim=3)

# Build OpenSees through the typed bridge. The tie auto-emits here --
# nothing else to declare for it.
ops = apeSees(fem)
# ... ops.section / ops.element / ops.fix / ops.mass / loads ...
ops.run(...)
```

## Order-mismatched interfaces: `method="mortar"`

The default tie is a *collocation*: each slave node is pinned to the
master's interpolated field at one projected point. That is exact for
matched interpolation orders, but across an **order mismatch** — hex20
faces tied onto hex8 faces — it over-constrains the quadratic side: the
slave's midside nodes are forced onto a bilinear field, so the surface
cannot deform quadratically. Measured on a solid steel fuse, that
stiffened the rib-root fixity by enough to move the device stiffness
+7.7 % against a conformal reference.

The fix is the integral-mortar tie:

```python
g.constraints.tie("plate_face", "rib_face",
                  method="mortar", enforce="equation", dofs=[1, 2, 3])
```

`method="mortar"` integrates the bond over the facet overlaps with a
dual (biorthogonal) slave basis instead of collocating nodes, so
neither side's interpolation order is imposed on the other — and it is
master/slave symmetric in a way collocation is not. It works on
composed assemblies (`from_h5` + `compose`, or `Assembly` with
`kind="tie"` and `method="mortar"` in the couple), which is exactly
where order-mismatched models live, since `set_order` is global per
gmsh session. Requirements and limits (ADR 0086): `enforce="equation"`
(so it needs the Lagrange handler and an unsymmetric solver, like any
equation tie — and, for the *live* run, a Ladruno fork build; see
[Backend capabilities](../concepts/backend-capabilities.md)); a
**flat, coincident** interface (`tolerance` is the
out-of-plane gap allowance); convex facets with **straight edges**
(every midside node at its edge midpoint — the kernel integrates on the
corner polygon, which only equals the curved facet for straight edges);
**no master facet overlapping another** (the coverage check counts
multiplicity, so two layers of master surface can hide an untied slave
strip); tri6 *slave* facets are refused (swap the sides — tri6 is fine
as master). Every degenerate case is a hard `MortarTieError`; a mortar
tie never silently resolves to nothing.

## Notes / gotchas

- **Target names, never raw tags.** Both factories take part / PG labels
  so the constraint survives a remesh — don't pass node or entity tags.
- **`tie` vs `equal_dof`.** If the meshes share nodes, `equal_dof` is
  cheaper and exact. If they don't, `equal_dof` finds no pairs — use
  `tie`. For large or doubly non-matching interfaces, escalate to
  `g.constraints.tied_contact` (bidirectional projection).
- **DOFs are 1-based** (`1=ux, 2=uy, 3=uz, 4=rx, 5=ry, 6=rz`). On a
  3-DOF solid model only `1–3` exist; omit `dofs` to tie all available.
- **Don't re-emit it.** The bridge auto-emits MP constraints from the
  snapshot (ADR 0022). Adding a matching raw `ops.equalDOF` /
  `ops.rigidLink` on top double-constrains the interface.
- **Tune `tolerance`.** Too tight and no pairs/projections are found;
  too loose and you couple nodes that shouldn't be. `equal_dof` defaults
  to `1e-6` (geometric match); `tie` defaults to `1.0` (projection gap).

## See also

- **Concept:** [Constraints guide](../concepts/constraints.md)
  — full taxonomy, the resolve pipeline, and how records land on the
  broker.
- **Bridge:** [OpenSees bridge guide §4.4](../concepts/opensees-bridge.md)
  — MP-constraint auto-emit and stage-binding ties by name (SSI).
- **Example:** [Tie non-matching meshes](../examples/tie-non-matching-meshes.md)
  — two solid blocks meshed at different sizes joined by
  `g.constraints.tie`, load transmitted exactly across the interface.
- **API:** [`g.constraints`](../api/constraints.md) — `tie`, `equal_dof`,
  and the rest of the constraint factory signatures.

---

*Next: [Compute section properties for a custom section](section-properties.md).*
