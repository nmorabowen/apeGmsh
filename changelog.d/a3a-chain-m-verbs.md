### ADDED — `ops.residual_modes`, `ops.ritz_vectors` and `response_spectrum_analysis(require_mass=)` (chain A link A3a)

Two live one-shots wrap the Ladruno fork's chain M modal-basis commands. Each
builds a fresh live domain and returns a `ModalPropertiesResult` for the new
mode set:

- `ops.residual_modes(dirs, num_modes=, period="rayleigh", replace=False,
  tol=None)` runs `eigen`, then `residualModes -dir …`, then
  `modalProperties`. The missing-mass pseudo-modes are appended last, and
  the mass ratio of each direction reaches 100 %. `period=T` keeps the
  Rayleigh λ; RSA then reads Sa at `T` and sums the pseudo-modes into one
  rigid mode (fork ADR 213 D3b).
- `ops.ritz_vectors(n, dirs=(), patterns=(), cycles=None, tol=None)` runs
  `ritzVectors` in place of `eigen`. Its starting loads are the inertia of
  each direction, then the reference loads of each `ops.pattern.Plain`. It
  needs a declared analysis.

Both refuse a partitioned model or an MPI `system` before anything is
emitted, because the fork refuses both commands on OpenSeesSP.

`response_spectrum_analysis` takes `require_mass=` (a ratio in (0, 1]) and
emits `-requireMass`. The fork then refuses an analysis whose modes carry
less of the free mass. Its docstring now says that `-combine` combines
nodal displacements only, and that `spectrum_after_gravity` is the way to
combine forces, drifts and storey shears.

All three refuse with `ForkTooOldError` on a build without chain M, before
the model is built. The commands are the first `command()` tokens (ADR 0114
D2): the `analysis.modal.ResidualModes` / `RitzVectors` primitives, under a
new `ModalBasis` family (ADR 0127, admitted under ADR 0121 D4). Neither is
registered on a bridge, so decks and `model.h5` do not carry them.
