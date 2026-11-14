### ADDED — `Results.from_ladruno` reads MPM particles: `results.particles` and `ParticleSlab` (ADR 0134 D3, D4; #2981)

A `.ladruno` at `INFO/FORMAT_MINOR_VERSION >= 2` may hold a particle
group in a stage (`MODEL/PARTICLES` and `RESULTS/ON_PARTICLES`, the
fork's contract 232a §2). `Results.from_ladruno` now reads it through
`results.particles`:

- `results.particles.get(component=..., body=..., ids=..., time=..., step=...)`
  returns a `ParticleSlab` in the file's own ragged layout: `values`
  `(S, nComp)`, `particle_ids` `(S,)`, `offsets` `(T+1,)`, `time` and
  `step` `(T,)`. `.output(k)` gives one output, `.dense()` gives
  `(T, M, nComp)` aligned to the stage registry with `NaN` where a
  particle is absent, and `.item()` follows ADR 0115 D3.
- Components are spelled as the file spells them: `POSITION`, or one
  column such as `POSITION/x`, `STRESS/sigma12`,
  `STATE_<token>/<label>`. Particles are selected by body name or by
  id. Outputs are selected by index or by `step=`, and a float `time=`
  is refused, because the Domain time may restart inside a stage.
- `.bodies()`, `.registry()` (id, body, material tag, mass), `.grid()`,
  `.info()` and `.components()` read the stage metadata.
  `ResultLevel.PARTICLES` lists the components in
  `results.inspect.components()`.

An all-particle file now opens without `model_h5=`. Its `fem` is `None`,
and a node, element or Gauss read of a particle-only stage raises
`ValueError` that names the stage. A file a run left open reads every
committed output and ignores a torn tail (contract §2.5), with
`ParticleOutputsTruncatedWarning`. A torn tail includes a last `OFFSETS`
entry that is smaller than the one before it. The reader refuses each of
these with the contract's rule code:

- a part file with particles;
- a particle group in a minor-1 file;
- half a particle group;
- an inexact dtype;
- a `PARTITION_REDUCTION` other than `NONE`;
- a `STEP` that does not strictly increase;
- an `OFFSETS` that decreases before its last entry.

A refused stage raises on every access.
`LADRUNO_SUPPORTED_FORMAT_VERSIONS` stays `(1,)`.
