### FIXED — an MPCO gauss strain record that names no element class refuses by name instead of reading Tri31 back empty (#2083)

The MPCO emit picks a gauss strain record's token from its
`element_class_name`: `material.strain` for Tri31, which has no
element-level strain response, and `strains` for every other class. A
record that named no class got `strains`, so `Results.from_mpco` returned
no Tri31 rows and no warning.

A `ResolvedRecorderRecord` holds element ids, not a physical group or a
bridge, so the emit cannot look the class up. A gauss record with a
continuum strain component (`strain_*`, or a derived strain scalar such as
`von_mises_strain`) and no `element_class_name` now raises `ValueError`
naming the record, the component and `element_class_name=`. It fails on
every MPCO route (`spec.emit_mpco`, `spec.emit_default`, the Tcl and Python
MPCO lines, `mpco_ops_args`) before any recorder is made. Shell
generalized strains (`membrane_strain_*`, `curvature_*`,
`transverse_shear_strain_*`) and stress records do not depend on the class
and are unchanged.

To migrate, pass the elements' C++ class on the record, for example
`element_class_name="Tri31"`, and split a record whose elements span
several classes into one record per class.
