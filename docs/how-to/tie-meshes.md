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
