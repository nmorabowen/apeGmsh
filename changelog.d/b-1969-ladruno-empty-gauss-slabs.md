### FIXED — `.ladruno` Gauss reads no longer come back empty for LST `stresses` and truss `axialForce` (#1969)

Three records were in the file, yet `results.elements.gauss` answered with an
empty slab and no warning:

- **LadrunoLST recorded under `stresses`.** The element tags no component
  names, so the reader names the nine `C1..C9` columns from the response
  catalog. That lookup matched only the `stress` spelling, so the same record
  under `stresses` listed no components. The recorder token is now mapped to
  its catalog token first (`stresses` → `stress`, `strains` → `strain`), and a
  column count that fits no catalog layout still raises
  `GaussLayoutMismatch`.
- **LadrunoLST Gauss points had no position.** The element answers no
  `integrationPoints` response, so the file carries no `GP_PARAM` and
  `natural_coords` came back with shape `(3, 0)`. A catalog-named bucket now
  takes the catalog rule's points, (2/3, 1/6), (1/6, 2/3), (1/6, 1/6), in the
  element's order. When the file does carry `GP_PARAM`, the file still wins.
- **Truss / CorotTruss recorded under `axialForce`.** The force read only as a
  line station. It now also reads as the one-Gauss-point `axial_force` that
  DomainCapture and the `.out` reader give. This applies only when the
  column's own label (`N`) is the catalog's `axial_force`.
