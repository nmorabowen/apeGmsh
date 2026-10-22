### ADDED — frame property modifiers: `FrameModifiers` and `modifiers=` on the elastic frame elements

`FrameModifiers(A, Avy, Avz, J, Iy, Iz, mass, weight)` (in
`apeGmsh.opensees`) multiplies a frame's section properties the way ETABS
frame property modifiers do, in OpenSees local axes.
`FrameModifiers.etabs(A=, As2=, As3=, J=, I22=, I33=, mass=, weight=)` takes
the ETABS names: with the `geomTransf` `vecxz` along ETABS local 3, OpenSees
local z is ETABS 3 and local y is ETABS 2, so `As2 -> Avy`, `As3 -> Avz`,
`I22 -> Iy` and `I33 -> Iz`. A cracked-section column is
`modifiers=FrameModifiers.etabs(I22=0.7, I33=0.7)`.

`modifiers=` is accepted by `elasticBeamColumn`, `ElasticTimoshenkoBeam`,
`forceBeamColumn` and `dispBeamColumn`, through `ops.element.*` and on the
standalone specs (`ops.register`). The declaration keeps the gross values;
the emitted command carries `base x modifier`:

- `elasticBeamColumn` / `ElasticTimoshenkoBeam` emit the scaled `A`, `Iz`,
  `Iy`, `J`, `Avy`, `Avz` and `-mass`. `E` and `G` are never scaled.
- `forceBeamColumn` / `dispBeamColumn` reference a modified copy of their
  `beamIntegration` and of each `ElasticSection` in it, registered by the
  bridge as synthesised declarations. The section you declared is never
  changed. Groups that share a section and a modifier set share one copy.
  On a shear `ElasticSection`, `alphaY` / `alphaZ` are rescaled by
  `m_Av / m_A`, so the `A` modifier leaves the shear areas alone.

Modifiers that cannot be applied are refused at construction, never
ignored:

- a shear modifier on the Euler-Bernoulli `elasticBeamColumn` or on an
  `ElasticSection` without `alphaY`;
- `Iy`, `J` or `Avz` on a 2-D element or section;
- `mass` on an element declared without `mass=`;
- any section in the integration rule other than an `ElasticSection`;
- `weight` other than 1. apeGmsh builds no frame self-weight, because
  OpenSees beam-columns take no `eleLoad -selfWeight`. Scale the
  `beam_uniform` you declare instead.

The archive keeps the values that were emitted, so `OpenSeesModel.from_h5`
replays the modified model as it ran. There is no schema change.
`LadrunoForceBeamColumn` / `LadrunoDispBeamColumn` take no `modifiers=`.

Validated live on stock openseespy and on the Ladruno fork, at `rtol=1e-10`.
The tests check cantilever tip displacement under axial load, bending about
both ETABS axes, Timoshenko shear and torsion against closed form, and
modal frequencies under the `mass` modifier.
