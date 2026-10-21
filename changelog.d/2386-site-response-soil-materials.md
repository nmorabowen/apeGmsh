### ADDED — typed site-response soil materials: `PressureIndependMultiYield`, `PressureDependMultiYield(02)`, `PM4Sand`, `PM4Silt`; their stage flip; and a refusal of the multi-yield soils under `algorithm Linear` (#2386, #2387, #2390)

`ops.nDMaterial` now carries the soil family that 1-D and 2-D site
response needs. A script no longer declares `PressureIndependMultiYield`
on the private `NDMaterial` base and registers it by hand. Each class
mirrors its OpenSees parser argument for argument, with the parser's
defaults, and is validated client-side. A parser that exits or silently
resets a value (a zero strength, `n_surf` out of range,
`pt_angle > friction_angle`, an unordered backbone) now raises at the
call. Each class emits its full positional list, so the declaration
archives with its field names. The multi-yield classes accept a user
modulus-reduction curve, `backbone=((shear strain, G/Gmax), ...)`.

- `s.update_material_stage` flips the new classes, as it already flipped
  the SANISAND family. It also accepts `stage=2` (elastic with
  pressure-dependent moduli) for the two `PressureDependMultiYield`
  classes, and refuses stage 2 for every other class. PM4Sand
  `FirstCall` / `PostShake` and the multi-yield `bulkModulus` /
  `shearModulus` updates go through `s.update_parameter(...,
  material=mat)`. Both verbs were already archived in `model.h5`; the
  schema does not change.
- A soil material that does not fit its element raises at the element
  call. Without this, OpenSees calls `exit(-1)` and the interpreter dies.
  The checked cases are a bare multi-yield `nd=3` on a quad or triangle,
  `nd=2` on a solid, a plane-stress view of one, and PM4Sand / PM4Silt on
  a solid. An `nd=3` material behind `ops.nDMaterial.PlaneStrain` stays
  valid. A `LadrunoUP` region, which takes its dimension from its cells,
  is checked at build.
- A multi-yield soil solved with `algorithm Linear` is refused at build,
  in a flat chain, a stage or a strategy rung. These materials compute
  their stress only in `getStress()`, and a Linear step commits before it
  forms the residual again, so each step commits the previous state's
  stress. A shaken 1-D column then answered about 2 % of the Newton
  surface acceleration, whatever the constraint handler (an upstream
  OpenSees defect, `PressureIndependMultiYield.cpp` `commitState`).
  `PM4Sand` / `PM4Silt` integrate in `setTrialStrain` and are not
  affected. `integrator CentralDifference` has the same defect; the
  build does not check it yet.
