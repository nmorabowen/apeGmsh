### FIXED — a carrier of a second sand activated after a plastic flip is refused (#2534)

A stage that activates an element of a ManzariDafalias or
LadrunoSANISAND material, after an earlier stage left the class-wide
stage flag plastic, is refused unless it flips the group again (#2445).
That check counted per material, so an element carrying a second
ManzariDafalias, one the flip never named, passed. The flag is one C++
static for the class, so that element entered the plastic stage with no
`Elastic2Plastic()` of its own: on stock openseespy 3.7.1 it gives, bit
for bit, the answer of the refused same-material case, off the
per-element reference.

The check now counts per class-wide group, at the stage close, at the
emit and on the `model.h5` replay. A model that did this, which earlier
replayed byte for byte, is now refused by name. Flip the whole group in
the stage that activates the element (`s.update_material_stage(materials=
[sand, sand_c], stage=1)`): measured, that gives the per-element answer.
SAniSandMS, PM4Sand and PM4Silt are not counted, because their flip only
sets the flag.
