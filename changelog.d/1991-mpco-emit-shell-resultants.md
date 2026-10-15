### FIXED — `spec.emit_mpco` records shell resultants under `stresses` and refuses an unmapped gauss component (#1991)

A gauss record of shell resultants (`expand_shorthand("shell_resultant", ...)`)
emitted `recorder mpco ... -E section.force`: `bending_moment_*` matched the
beam prefix and `membrane_force_*` / `transverse_shear_*` matched nothing and
were dropped, so `Results.from_mpco(...)` read an empty Gauss slab with no
warning. Gauss records on `emit_mpco`, `to_mpco_tcl_command` and `to_mpco_python_command` now
take the token the MPCO reader looks under (`gauss_keyword_for_canonical`):
`stresses` for continuum stress and shell resultants, `strains` for
continuum strain and shell generalized strains. Every catalogued shell class
reads back at `rtol=1e-10` on the fork lane. A gauss component with no MPCO
token now raises `ValueError` naming the record and the component instead of
vanishing from the recorder line.
