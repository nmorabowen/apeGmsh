# Apply a face pressure or traction

Put a distributed surface load — wind, snow, water, a uniform "pull" — on
a face (or, in 2-D, an edge) physical group. Declare it pre-mesh against
the group **name** with `g.loads.surface`; it resolves to equivalent
nodal forces at `get_fem_data` and **auto-emits** through the typed
bridge.

## Recipe

```python
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

g = apeGmsh(model_name="pressure_demo")
g.begin()
# ... geometry, parts, the "Roof" / "FacadeW" surface PGs, mesh ...

with g.loads.case("snow"):
    # Pressure: scalar, perpendicular to each face. Positive magnitude
    # pushes INTO the face, so it follows a sloped/curved surface without
    # you resolving components by hand.
    g.loads.surface.pressure("Roof", -3.0e3)

with g.loads.case("wind_X"):
    # Traction: a vector applied the same way on every face regardless of
    # orientation (a uniform "pull").
    g.loads.surface.traction("FacadeW", (1.2e3, 0, 0))

# Resolve: surface loads become equivalent nodal force records on the broker.
fem = g.mesh.queries.get_fem_data(dim=3)

# Build OpenSees. The g.loads.* surface loads auto-emit as synthesized
# Plain patterns -- you do NOT re-declare them on the bridge.
ops = apeSees(fem)
ops.model(ndm=3, ndf=3)
# ... ops.nDMaterial / ops.element / ops.fix / ops.mass ...
ops.run()
```

## Notes / gotchas

- **`surface.pressure` is pressure; `surface.traction` is traction.** Pressure is
  perpendicular to each face (right for wind/snow/water on a sloped or
  curved surface). Traction takes a vector and ignores face
  orientation. Sign: positive `magnitude` pushes *into* the face.
- **Don't double-declare.** A surface load on `g.loads.*` auto-emits at
  bridge build. Declaring the *same* load again with a bridge
  `p.load`/`eleLoad` channel **doubles** it — reactions come out at 2×.
  One load, one channel.
- **Element `pressure=` is a different thing — and it can bite a
  benchmark.** The 2-D quad/tri elements (`FourNodeQuad`, `Tri31`,
  `SixNodeTri`) take a `pressure=` constructor arg. That is OpenSees'
  *element-native* uniform normal pressure applied around each element's
  **perimeter edges** — it does not know which boundary you meant. On a
  uniaxial tension/pressure benchmark it loads the interior element edges
  too and gives the wrong resultant. For a clean traction on a *named*
  boundary face, use `g.loads.surface.traction(pg="...")` (resolved nodal
  traction) and leave element `pressure=` unset. Reach for element
  `pressure=` only when you genuinely want OpenSees' per-element edge
  pressure behaviour.
- **Higher-order faces want `reduction="consistent"`.** Tributary
  (default) splits `p·A` equally to corner nodes — exact for tri3/quad4.
  For tri6/quad8/quad9, pass `reduction="consistent"` so the curved
  Gauss-point normal and mid-side weighting are integrated correctly.
- **Sanity-check ΣF.** Sum `force_xyz` over `fem.nodes.loads.by_pattern(pat)`
  and compare to `p·A` by hand — a flipped sign or a load on the wrong
  face shows up immediately.

## See also

- **Concept:** [Loads guide §7](../internal_docs/guide_loads.md) — surface
  pressure vs traction, the resolve pipeline, and `reduction` /
  `target_form`.
- **Bridge:** [OpenSees bridge guide](../internal_docs/guide_opensees.md)
  — what auto-emits from `g.loads.*` vs what you re-declare.
- **Tutorial:** [Plate in tension](../tutorials/plate-in-tension.md) — a
  surface traction driving a 2-D continuum model end to end.
- **API:** [`g.loads`](../api/loads.md) — `surface`, `line`, `gravity`,
  and the rest of the load factories.
```
