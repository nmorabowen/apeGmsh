### ADDED — Particle readers know `PORE_PRESSURE` and refuse a reserved field with the wrong `COMPONENTS` (#3121)

The Ladruno fork's release `ladruno-v20261123b` (WP-232l) adds one particle
output field, `PORE_PRESSURE`: the excess pore pressure of a two-phase
particle, compression positive, and 0 on a single-phase particle. Its one
component is `p`. A two-phase particle's `STRESS` is the effective stress
σ′. `results.particles.get(component="PORE_PRESSURE/p")` reads it, and
apeGmshViewer reads it too.

Both readers now carry the fork's whole catalogue of reserved particle
fields (`PARTICLE_FIELDS` in the fork's checker, `ACCELERATION` included).
A reserved field whose `COMPONENTS` differ from the catalogue's for the
file's `SPATIAL_DIM` is refused by name, with code C2-CAT. The Python reader
refuses the stage and the viewer refuses the field. Before this, the Python
reader did not check the catalogue at all, and the viewer read a
`PORE_PRESSURE` field with any `COMPONENTS`. A field name the catalogue does
not list still reads as the file spells it.
