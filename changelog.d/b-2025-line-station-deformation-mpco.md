### FIXED — `spec.emit_mpco` refuses line-station section deformations instead of dropping them (#2025)

A `line_stations` record of section deformations (`axial_strain`,
`torsional_strain`, `curvature_y`, `curvature_z`) wrote no `recorder mpco`
token, and `shear_strain_y` / `shear_strain_z` matched the `shear` prefix and
wrote `section.force`. In both cases `Results.from_mpco(...)` read an empty
line-station slab with no warning. The MPCO reader cannot decode a
`section.deformation` bucket either: a fork run wrote its components as
`eps,kappaZ,kappaY,theta,gammaY,gammaZ`, and the read raised on `eps`. So
`emit_mpco`, `to_mpco_tcl_command` and `to_mpco_python_command` now raise
`ValueError` naming the record and the component for every line-station
section deformation, and for any line-station component with no routing.
Section forces still record `section.force`. Section deformations read back
from the fork's native recorder.
