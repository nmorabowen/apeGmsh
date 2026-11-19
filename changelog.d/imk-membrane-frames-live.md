### TESTS — `LadrunoIMKBeam` as `LadrunoMembrane` frames, and its `plasticRotation` read from `.ladruno`, pinned on the fork

A new fork-lane test, `tests/opensees/integration_ladruno/test_ladruno_imk_membrane_live.py`,
pins two things YOO's nonlinear model relies on. First, `LadrunoMembrane(frames=...)`
given a `LadrunoIMKBeam` spec hands its tributary slab load to the IMK beams, which carry
it exactly as `elasticBeamColumn` beams do (to 1e-6), and the four beams' end shears sum to
the panel load. Builds before fork WP-200 (34df942c7) refused element loads on the IMK beam
and dropped this load without a word. Second, `recorder ladruno` with
`elem_responses=("plasticRotation",)` records the IMK hinge rotations, and
`Results.from_ladruno(...).elements.get(component="plasticRotation")` returns them equal to
the live `eleResponse` at every step. The fork's `plasticRotation` is the total hinge-spring
rotation θh (the elastic spring part included), not the plastic part alone. No apeGmsh
source changed: the spec, the `frames=` resolution, the recorder and the raw-token read
already supported this. Verified on fork build f0b9a1df6.
