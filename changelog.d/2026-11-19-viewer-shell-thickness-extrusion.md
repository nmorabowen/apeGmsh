### ADDED — apeGmshViewer draws shells and membranes extruded by their thickness

The legend's **extrude** toggle, which drew frame elements with their
cross-section, now also draws shell and membrane faces as slabs of their
section thickness, centred on the mid-surface. The thickness is read from
what the model file already archives, so the schema is unchanged:

- a shell (`ShellMITC4`, `ShellMITC9`, `ShellDKGQ`, `ShellDKGT`,
  `ShellNLDKGQ`, `ShellNLDKGT`, `ASDShellQ4`, `ASDShellT3`) takes its
  section's `h` (`ElasticMembranePlateSection`, `ElasticPlateSection`,
  `PlateFiber`), the sum of a `LayeredShell`'s layers, or a
  `LadrunoShellModifier`'s inner section;
- a `LadrunoMembrane` takes `-thick t` or the `t` of `-stiff E nu t`; one
  declared with `pg=` gives its mesh cells the thickness of its declaration.

No shell the bridge writes carries a mid-surface offset, so none is drawn. A
thickness that cannot be read stays a zero-thickness face. `LadrunoSolidShell`
is not affected: it is a solid with real thickness, and the bridge does not
emit it.

The slab is built from the faces as drawn: it follows the deformed shape,
takes the faces' contour or group colours, and is closed by side walls only
where the surface ends, meets a panel at an angle (over 30 degrees), or changes
thickness. Neighbouring panels in one plane meet without a gap. The mid-surface
stays the picking target. The toggle is enabled for a model with shells or
membranes and no frames.

Cost: the load keeps one float per drawn triangle, and only when the model has
a shell or a membrane. The slab geometry is built only while the toggle is on:
about 6 vertices per extruded triangle plus 6 per wall edge, each with its
position and colour or ramp coordinate.
