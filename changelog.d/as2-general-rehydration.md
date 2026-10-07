### ADDED — Assembly rehydrates transforms, beam integrations and dampings; compose carries `rebar_elements` (program AS2a, #1539)

`Assembly.bridge()` now rehydrates more of each instance's archived model
content under `{instance}.` names (ADR 0117 D4):

- `geomTransf` `Linear`, `PDelta` and `Corotational`;
- the uniform-section `beamIntegration` rules (`Lobatto`, `Legendre`,
  `NewtonCotes`, `Radau`, `Trapezoidal`);
- `damping` `Uniform`, `SecStif`, `URD` and `URDbeta` attached by an
  element's `damp=`;
- `section Elastic`, `uniaxialMaterial Elastic`, and the elements
  `elasticBeamColumn`, `forceBeamColumn`, `dispBeamColumn` and
  `FourNodeTetrahedron`, with their `-mass`, `-cMass`, `-iter` and `-damp`
  flags.

Every archived declaration of an instance is parsed before the first one is
registered, so a refusal leaves the bridge untouched. These still raise
`AssemblyError`: a damping attached by region (`on=`), a `-factor` time
series, and element args that vary inside one physical group, such as an
orientation-derived transform. The last needs the per-row selector of AS2b
(#1542).

`g.compose` and the assembly now carry a source module's
`elements.rebar_elements`. The bar cells move with the module, the bar
group is prefixed like every group, and the bar material becomes
`{label}.{material}`. The `ComposeDroppedStreamWarning` for that stream is
gone. **Behaviour change:** a module's bars composed with `g.compose` now
emit on the host bridge, which must declare the material under its
prefixed name (for example `A.rebar`).
