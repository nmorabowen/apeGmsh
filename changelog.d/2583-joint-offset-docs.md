### CHANGED — docs: frame members, joint offsets and the source-entity stream (#2551 S3)

The bridge concepts page and the `apegmsh` skill now document `jnt_offset=` on
`geomTransf`, the per-element source entity (`fem.elements.entity`,
`source_entity`, neutral 2.36), joint offsets split over each gmsh curve's
ends, and the refusals. This supersedes the per-element refusal prose of the
opensees 2.37.0 joint-offset entry: end-sensitive offsets refuse by curve
where the snapshot carries source curves, and by the 30-150 degree corner
rule only where it carries none.
