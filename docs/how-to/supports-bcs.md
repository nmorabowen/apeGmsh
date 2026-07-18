# Fix supports & boundary conditions

Hold a model down by fixing the nodes of a named face — pin, roller, or
fully fixed — and, where you need it, push a face to a **non-zero
prescribed displacement**. Reach for this whenever a part would
otherwise float free, or when you want to drive a support by an imposed
settlement / motion instead of a force.

Supports are **single-point constraints (SPs)**. They are *not*
auto-ingested from the session — you **re-declare** them on the bridge:
homogeneous fixities with `ops.fix(...)`, non-zero prescribed values with
`p.sp(...)` inside a pattern. (Contrast: `g.loads.*` and `g.constraints.*`
*do* auto-emit.) Target a physical-group **name**, never a raw tag.

## Recipe

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

with apeGmsh(model_name="support_demo") as g:
    g.model.geometry.add_box(0, 0, 0, 2.0, 1.0, 1.0, label="Block")
    g.model.sync()

    g.physical.from_label("Block", name="Block")
    g.model.select(dim=2).on_plane((0, 0, 0.0), (0, 0, 1), tol=1e-6).to_physical("Base")
    g.model.select(dim=2).on_plane((0, 0, 1.0), (0, 0, 1), tol=1e-6).to_physical("Top")

    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

# ── Bridge — supports are re-declared here, by PG name ──────────────
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
conc = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
ops.element.FourNodeTetrahedron(pg="Block", material=conc)

# Homogeneous SP — every node in "Base" pinned in x, y, z.
ops.fix(pg="Base", dofs=(1, 1, 1))           # 1 = fixed, 0 = free

# Non-zero prescribed displacement — drive "Top" down 0.01 in z.
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.sp(pg="Top", dof=3, value=0.01)        # dof is 1-based

ops.tcl("model.tcl")
```

`ops.fix(pg="Base", dofs=(1, 1, 1))` fans the mask across every node of
the `Base` group at build time and emits one `fix $node 1 1 1` line each;
`p.sp(pg="Top", dof=3, value=0.01)` emits `sp $node 3 0.01` lines inside
the `pattern Plain {…}` block.

### Pin vs. roller vs. fully fixed

The `dofs` tuple is a length-`ndf` mask (`1` = fixed, `0` = free). Pick
the pattern; the table is for the common `ndm`/`ndf` pairs:

| Support | `ndm=2, ndf=2` | `ndm=3, ndf=3` (solid) | `ndm=3, ndf=6` (frame/shell) |
|---|---|---|---|
| Roller (free in x, vertical held) | `(0, 1)` | `(0, 0, 1)` | `(0, 0, 1, 0, 0, 0)` |
| Pin (translations held, rotations free) | `(1, 1)` | `(1, 1, 1)` | `(1, 1, 1, 0, 0, 0)` |
| Fully fixed / encastre | `(1, 1)` | `(1, 1, 1)` | `(1, 1, 1, 1, 1, 1)` |

For a 3-DOF solid there are no rotational DOFs, so "pin" and "fully
fixed" coincide — `(1, 1, 1)`. The distinction only appears once nodes
carry rotations (`ndf=6`).

## Notes / gotchas

- **`len(dofs)` must equal `ndf`.** A mismatch raises `BridgeError` at
  `build()`. Pin a 6-DOF node with `(1, 1, 1, 0, 0, 0)`, not `(1, 1, 1)`.
- **`dofs` is a 0/1 mask; `dof=` in `p.sp` is a 1-based index.** Same
  word, two conventions: `ops.fix(dofs=(1, 1, 1))` is a per-DOF flag
  vector, while `p.sp(dof=3, value=…)` names *which* DOF (the 3rd) to
  prescribe.
- **Homogeneous vs. prescribed is the deciding split.** Zero-value
  fixities are model-level → `ops.fix`. Any non-zero imposed value goes
  in a pattern → `p.sp`. Don't try to fake a prescribed displacement with
  `ops.fix`.
- **Supports are NOT auto-ingested.** If you declared a face SP on the
  session via `g.displacements.surface(...)`, it resolves into `fem.nodes.sp` and
  feeds the viewer / `Results`, but it does **not** reach the runnable
  deck — re-declare it: homogeneous → `ops.fix`, prescribed → `p.sp`.
- **`g.constraints.bc` vs. `g.displacements` ownership.**
  `g.constraints.bc` owns permanent homogeneous fixes; `g.displacements`
  owns prescribed motion (any non-zero value, or a pattern-bound /
  time-varying value). A zero authored via `g.displacements` is an
  allowed pattern-bound hold, not a silent alias for `bc`.
- **Target names, not tags.** `pg=` resolves a physical-group **or** an
  apeGmsh label (FEM-direct, no promotion needed). Keep PG names
  dimension-unique so `pg=` is never ambiguous.
- For a **stage-bound** support that only comes online in a later
  analysis stage, use `s.fix(pg=…, dofs=…)` inside a `with ops.stage(...)`
  block instead of the flat `ops.fix`.

## See also

- Concept: [`guide_opensees.md` §3.4](../concepts/opensees-bridge.md)
  — the `ops.fix` deck verb, selective ingest, and §4.2 pattern-scoped
  `p.sp`.
- Concept: [`guide_loads.md` §11](../concepts/loads-and-masses.md) —
  `g.displacements.surface(...)`, the session-side face SP (rigid-body motion at a
  face centroid) that resolves into `fem.nodes.sp`.
- Related recipe:
  [Prescribe a support displacement (SP)](index.md#physics).
- API: `apeSees.fix(*, pg=|nodes=, dofs=)` and the pattern verb
  `p.sp(*, pg=|node=, dof=, value=)`.

---

*Next: [Tie two non-matching meshes](tie-meshes.md).*
