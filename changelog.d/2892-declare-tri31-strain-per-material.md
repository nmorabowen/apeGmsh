### FIXED — `ops.recorder.declare(gauss=<strain>)` on Tri31 records strain per Gauss point; a record mixing Tri31 with another class refuses by name (#2892)

Tri31 has no element-level `strains` response, so the `recorder Element
... strains` line a bridge strain declare emitted on a Tri31 group wrote
the time column alone, with no warning. Passing
`element_class_name="Tri31"` emitted the same line.

- **The fix.** The bridge reads each record's element classes from its
  element plan. A gauss strain record on Tri31 now emits one
  `material <gp> strain` line per Gauss point (Tri31 has one), written to
  `<file_root>/<name>__<record>__gauss_material<gp>.out`. A live run's file
  holds three strain columns per element.
- **The refusals.** A strain record whose groups or `ids=` span Tri31 and
  a class that answers `strains` (for example FourNodeQuad) raises a
  `BridgeError` naming both classes. So does an `element_class_name=`
  that contradicts the record's elements when either side is Tri31. The
  raise comes at emit, before any deck file is written.
- **`ModelData`.** With no element plan, an explicit
  `element_class_name="Tri31"` is honoured. A strain record with no class
  keeps `strains`, as before.
