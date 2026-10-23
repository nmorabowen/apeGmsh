### FIXED — stage-flag updates that reach nothing, or the wrong material, are refused at stage close (#2458, #2460)

Three `s.update_parameter` stage-flag routes gave a silent wrong answer.
Each one is now refused at stage close, by name:

- **`s.update_parameter("updateMaterialStage", v, ...)`.**
  `updateParameter` sets only `Parameter::theDouble`. Every material that
  takes this name (ManzariDafalias, LadrunoSANISAND, SAniSandMS, PM4Sand,
  PM4Silt, the stress-density models) reads `Information::theInt`, which
  nothing sets. The run kept a stage that depended on uninitialised memory,
  while the bridge counted the flip. Use `s.update_material_stage`, or
  `s.update_parameter("materialState", ...)`, which reads `theDouble`.
- **`materialState` on a class that has no such branch.** The
  multi-yield soils (PressureIndependMultiYield, PressureDependMultiYield,
  PressureDependMultiYield02) have no `materialState` in `setParameter`, so
  OpenSees warns that no object took it and the run stays elastic. The call
  is now refused at once, as is a `materialState` with no `material=`. The
  classes that take it are listed in `MATERIAL_STATE_CLASSES`. A refusal
  never suggests a route that does nothing for the class.
- **`materialState` through a `FourNodeQuad` host.**
  `FourNodeQuad::setParameter` reads any `material...` name as its own
  per-point address, so the flip never reached the material. The update is
  now accepted only when every targeted element that carries the named
  material is of a `MATERIAL_STATE_ELEMENTS` class (#2459's allowlist).
- **`materialState` over a tag-blind neighbour.** DruckerPrager,
  BoundingCamClay, StressDensityModel and stressDensity take `materialState`
  without checking the tag it names, so a flip of a sand over elements that
  also carry one of them switched it too. They are listed in the new
  `TAG_BLIND_STAGE_PARAMETER` table in `stage/_builder.py`. An update whose
  targets carry such a class under another tag is refused. So is an
  `s.update_material_stage` beside a class that takes `updateMaterialStage`
  that way.

The bridge's own `-ele` sweep (#2459) lists only the carriers of the swept
material, so it never reaches a tag-blind class. A test pins this.
