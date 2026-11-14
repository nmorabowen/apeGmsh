### ADDED — apeGmshViewer reads MPM particles: the `.ladruno` particle group, the seeded `<stem>.particles.h5`, and its pairing by stem

ADR 0134 D1 and D3, the data side; the points layer that draws them comes
next (M2-S4). The layouts are the fork's MPM contracts (Contract 1, the
particle set; Contract 2, the `.ladruno` particle group).

- **Particle results.** At `INFO/FORMAT_MINOR_VERSION >= 2`,
  `LadrunoFile.particles` reads each stage's `MODEL/PARTICLES` (grid,
  bodies, registry) and `RESULTS/ON_PARTICLES` (`STEP`, `TIME`, `OFFSETS`,
  fields). Output `k` is read on request, one hyperslab of `ID` and one of
  `DATA` over `OFFSETS[k]..OFFSETS[k+1]`, never the whole history. A
  killed run's file reads every committed output: the outputs are the
  prefix that `STEP`, `TIME`, `OFFSETS` and every dataset's length all
  hold, rows past it are not read, and the index warns what was dropped.
  The reads do not depend on the chunk shape or the string encoding. A
  stage that breaks the contract is refused by name, with its rule code,
  and the file's other stages still read. A part file that holds
  particles is refused (serial only).
- **The particle set.** `openParticleSet` reads a `<stem>.particles.h5`
  (positions, bodies, grid, ids, mass or density) and gates its integer
  `FORMAT_VERSION` exactly at 1.
- **Pairing.** `<stem>.particles.h5` is a kind of its own, matched
  case-insensitively before `.h5`, so `case1.particles.h5` is never a model
  of stem `case1.particles`. The open set gains a `particles` slot. It pairs
  by stem only with `<stem>.h5` and `<stem>.ladruno`; a set at another stem
  opens without a model.
- **Banner.** An all-particle `.ladruno` opened with no model now says its
  particles read without one, instead of "Results not drawn". The same
  sentence is ready for a particle set, which the points layer shows.
