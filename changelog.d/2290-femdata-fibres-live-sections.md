### ADDED — a raw deck on stock records fibre stress and strain over a FEMData archive (#2290)

`spec.emit_default` runs a raw openseespy deck with a FEMData archive
(`fem.to_h5`). On stock openseespy a `fibers` record over such an archive
is now recorded and read; it used to be refused by name, because its
stations were counted from each element's `beamIntegration`, which only the
bridge's archive holds.

- **The section facts come from the live domain.** At `__enter__`, before
  the run, each recorded element's section locations are read with
  `sectionLocation`, and each section's fibre count with `fiberData2`. The
  read slices the `.out` fibre file by those facts, under the rule the
  bridge's `beamIntegration` counts follow.
- **Missing or inconsistent facts are refused by name.** An element with
  no sections, or a section that answers no whole fibre (an elastic
  section, say), is refused by FEM id before the run. After the run, these
  are refused: elements with different station counts, sections with
  different fibre counts, a file whose stations are not the live sections,
  and a file whose fibre count is not the live one.
- **An `elements` record with no `nodal_resisting_*` component** is now
  refused at `__enter__` over a FEMData archive. It used to be refused only
  when the files were read, after the run.

The bridge's own fibre read is unchanged, and so is what it writes.
