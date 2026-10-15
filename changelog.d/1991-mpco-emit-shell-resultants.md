### FIXED — `spec.emit_mpco` records every gauss component under a group the MPCO reader reads, or refuses it (#1991)

A gauss record of shell resultants (`expand_shorthand("shell_resultant", ...)`)
emitted `recorder mpco ... -E section.force`: `bending_moment_*` matched the
beam prefix and `membrane_force_*` / `transverse_shear_*` matched nothing and
were dropped, so `Results.from_mpco(...)` read an empty Gauss slab with no
warning. Gauss records on `emit_mpco`, `to_mpco_tcl_command` and
`to_mpco_python_command` now take the group the MPCO reader looks under:

- `stresses` / `strains` for continuum stress and strain, shell resultants
  and shell generalized strains (`gauss_keyword_for_canonical`);
- `material.damage` for `damage*` (these raised before);
- `material.equivalentPlasticStrain` for `equivalent_plastic_strain*` (it
  emitted `strains`, the total-strain bucket, and read back empty);
- `material.plasticStrain` for `plastic_strain_*` and the scalars derived
  from it, `equivalent_plastic_strain_current` included (these raised before).

Every catalogued shell class, ASDConcrete3D damage and PEEQ, and the
LadrunoJ2 plastic-strain tensor read back at `rtol=1e-10` on the fork lane. A
gauss `axial_force` (Truss) emitted `section.force` and read back empty; it now
raises, because MPCO writes the truss `axialForce` bucket with no `GAUSS_IDS`
and the reader refuses it. Any other gauss component with no MPCO token raises
`ValueError` naming the record and the component instead of vanishing from
the recorder line.
