# Apply gravity / self-weight

Add the body weight (`ρ · g`) of a meshed solid as a load. Reach for this
whenever a model carries its own dead load — a footing, a dam, a soil
column, an RC volume — instead of (or alongside) a hand-applied nodal load.

Gravity is a body load with a convenience wrapper: you give it a **volume
target by name**, a gravity vector `g`, and a `density`. Like every
`g.loads.*` factory it is declared pre-mesh, resolved at `get_fem_data`,
and **auto-emitted by the typed bridge** — you never open a pattern or
write `eleLoad` for it.

## Recipe

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

g = apeGmsh(model_name="gravity_demo")
g.begin()
# ... build a part, mesh a volume into the PG "rc_volume" ...

# Self-weight of the concrete volume, grouped in a dead-load pattern.
with g.loads.case("self_weight"):
    g.loads.gravity("rc_volume", g=(0, 0, -9.81), density=2400)

# density=(0,0,-9.81) is the default g; the one-liner is just:
#     g.loads.gravity("rc_volume", density=2400)
# density=None tells the bridge to read ρ from the assigned material
# (only valid with target_form="element").

# Resolve: the gravity def becomes per-node body-force records on the broker.
fem = g.mesh.queries.get_fem_data(dim=3)

# Build OpenSees through the typed bridge. Gravity auto-emits here --
# nothing else to declare for it.
ops = apeSees(fem)
# ... ops.section / ops.element / ops.fix / ops.mass ...
ops.run(...)
```

## Notes / gotchas

- **Don't double-apply.** The gravity you declared via `g.loads.gravity`
  already auto-emits. Do **not** *also* hand the same elements a body force
  through the bridge (a raw `eleLoad -bodyForce` / element `body_force=`
  for the identical volume). That stacks two copies of self-weight and
  doubles the dead load — a silent, order-of-magnitude error in the
  reactions. Declare it in exactly one place.
- **Target volumes only.** Gravity must resolve to 3-D entities. Targeting
  a surface or curve is a **no-op** — the volume iterator silently skips
  non-3D dim-tags. (Self-weight of a shell belongs on the shell section's
  thickness × density, not on a load.) If a gravity def produces zero
  records, you almost certainly targeted the wrong-dimension label.
- **`g` is unit-sensitive.** `(0, 0, -9.81)` is for SI-metre models. For
  a kg-mm-s model use `(0, 0, -9810)`. A unit mismatch in `g` or `density`
  (kg/m³ vs g/cm³) shows up as an order-of-magnitude error in `ΣF`.
- **`density=None` needs element form.** Reading ρ from the material is
  only available with `target_form="element"`; the default
  `target_form="nodal"` **requires** an explicit `density` and raises
  `ValueError` without one.
- **Sanity-check ΣF.** After `get_fem_data`, sum `force_xyz` over the
  pattern and compare against `ρ · g · V` by hand — this is the cheapest
  way to catch a flipped sign or a doubled load.

## See also

- **Concept:** [Loads guide §8](../internal_docs/guide_loads.md) — the
  define→resolve pipeline, `reduction` (tributary vs consistent),
  `target_form` (nodal vs element), and how body forces land on the broker.
- **Bridge:** [OpenSees bridge guide](../internal_docs/guide_opensees.md)
  — why loads are pattern-scoped at the OpenSees level and are imported
  opt-in from the snapshot with `p.from_model(case)` (ADR 0051).
- **API:** [`g.loads`](../api/loads.md) — `gravity`, `body`, and the rest
  of the load-factory signatures.

---

*Next: [Apply a face pressure or traction](face-pressure.md).*
