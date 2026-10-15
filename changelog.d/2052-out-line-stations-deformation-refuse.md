### FIXED — a `.out` line-stations record refuses section deformations instead of reading forces under their names (#2052)

On the `.out` recorder route, a line-stations record of section deformations
(`axial_strain`, `curvature_y/z`, `torsional_strain`, `shear_strain_y/z`)
emitted the same `section force` recorder as a force record (or `localForce`
for an elastic beam), and `Results.from_recorders(...)` read the section
forces back under the deformation names, with no warning. The emit now raises
`ValueError` naming the record and every deformation component, as the MPCO
emit has since #2025. A record that mixes forces and deformations refuses as a
whole. The refusal reaches every `.out` route: `to_tcl_commands`,
`to_python_commands`, `emit_recorders` (before any recorder is issued) and the
recorder file listing. Force-only records emit unchanged. Section deformations
read back from the fork's native recorder.
