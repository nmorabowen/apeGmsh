### ADDED — a capability probe for the fork's chain M and P1 RSA commands, and live-fork tests that run them (chain A link A4)

`apeGmsh.opensees._target.rsa_features_of(ops)` reports which of the Ladruno
fork's modal-basis and response-spectrum commands the in-process build has:
`residualModes`, `ritzVectors`, `responseSpectrumAnalysis -requireMass` and
`responseSpectrumPerMode`. It checks that each command is present on the
module and never runs one, so it needs no model. The result is cached per
module. `ops.capabilities().rsa` carries it.

`RsaFeatures.require(feature)` and `require_rsa_feature(feature)` raise
`ForkTooOldError` ("fork too old: needs chain M (ladruno-v20261027+) for
residualModes; …") when the build lacks the feature. The later chain A
wrappers call them so an older build fails up front, not with an opaque
OpenSees error. No command can show `-requireMass`, so the probe infers it
from `residualModes`, which merged after it on the fork.

`tests/opensees/integration_ladruno/test_rsa_chain_m_live.py` runs the four
features on the fork's 10-DOF podium model:

- `residualModes` brings the X mass ratio to 100 %.
- The Ritz values bound the eigenvalues from above.
- Each mode's `responseSpectrumPerMode` base reaction is minus its modal
  inertia.
- `-requireMass` refuses a three-mode shortfall.

On a build without the commands (today's pin, `ladruno-v20261023`) each
test skips with the probe's refusal.
