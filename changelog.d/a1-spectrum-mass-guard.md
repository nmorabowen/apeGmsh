### FIXED — `spectrum_after_gravity` refuses a spectrum direction whose modes miss its mass (chain A, link A1)

Stock OpenSees and Ladruno forks older than `ladruno-v20261027` (WP-211) count the mass on a rigid
diaphragm's constrained nodes as fixed under `constraints Transformation`. `modalProperties` reports the
full `totalMass` but a short `totalFreeMass`, so Γ came out zero (all floors' mass lost) or too low (some
floors' mass lost), with no error. `spectrum_after_gravity` now refuses a spectrum direction whose free mass
falls short of the nodal mass on DOFs no `fix` holds, naming the likely cause. It also refuses a direction
whose modes' effective masses sum to at most `1e-6` of its free mass. The element-mass refusal is
unchanged. Both pins return `totalFreeMass`: `ladruno-v20261023` (the current `FORK_PIN`) loses the mass and
is refused, `ladruno-v20261027` counts it and runs. Direction 3 of an `ndm 3 ndf 3` model no longer raises
`KeyError`: `modalProperties` names its third column `RMZ`.
