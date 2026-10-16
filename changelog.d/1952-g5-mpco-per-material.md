### FIXED — the spec MPCO emit records Tri31 strain as `material.strain`; damage and Tri31 strain have live value oracles on the fork lane (#1952)

A gauss strain record on `Tri31` (`element_class_name="Tri31"`) emitted
`-E strains`. Tri31 has no element-level `strains` response, so MPCO wrote
no Tri31 bucket and `Results.from_mpco(...)` returned the strain of the other
elements in the file with no Tri31 rows and no warning. The emit now writes
`-E material.strain` for any `strains` token on a class in
`PER_MATERIAL_STRAIN_CLASSES`, and the reader already walks that group.
Tri31 stress keeps `stresses`, and every other class is unchanged.

MPCO has no per-token element filter: `-R` scopes the whole recorder. In a
mixed mesh, `material.strain` therefore also records the other classes that
answer it. On a `FourNodeQuad` it reads back equal to
`eleResponse(e, "strains")`, whether or not the quad has its own `strains`
record.

`damage_*` already emitted `material.damage`. A new fork-lane file,
`tests/opensees/integration_ladruno/test_mpco_per_material_live.py`, checks
both responses against `ops.eleResponse(e, "material", gp, ...)` at rtol 1e-10.
The damage case is a `Brick` with `ASDConcrete3D` pulled past cracking. Its
bucket holds 2 components per Gauss point (`d+`, `d-`). On a pip
`openseespy`, which has no MPCO recorder, neither response has a spec route.
