### FIXED — `spectrum_after_gravity` refuses a spectrum direction whose modes carry no mass (chain A, link A1)

Stock OpenSees and Ladruno forks older than `ladruno-v20261027` (WP-211) count the mass on a rigid
diaphragm's constrained nodes as fixed under `constraints Transformation`: `modalProperties` reports the
full `totalMass` but a zero `totalFreeMass` and zero Γ, so the per-mode response came out as zero with no
error. `spectrum_after_gravity` now refuses a spectrum direction that has nodal mass but no free mass,
naming the likely cause, and a direction in which every mode's Γ is zero. The element-mass refusal is
unchanged. Both pins checked return `totalFreeMass`: `ladruno-v20261023` (the current `FORK_PIN`) has the
loss and is refused, `ladruno-v20261027` counts the mass and runs.
