### FIXED — an MPCO strain read no longer drops Tri31 elements in silence (#2893, follow-up of #2083)

Tri31 answers no element-level strain: `ops.eleResponse(eid, "strains")`
is empty on it, so an MPCO record of `strains` leaves its elements out of
every strain bucket. A gauss strain read then returned the other classes'
elements and no Tri31 rows, with no warning. On
`tests/fixtures/simplex_gp/tri.mpco`, a whole-model `strain_xx` read
returned elements 2 to 5, and a read of the Tri31 element 1 returned an
empty slab.

The read now raises `MPCOStrainNotRecorded`, a `ValueError`, when a Tri31
element in scope has no row in any strain bucket. The message names the
elements and says how to get the rows: declare the recorder with
`element_class_name="Tri31"`, which records strain per Gauss point under
`material.strain`. Elements in scope are the file's own element listing,
narrowed to the requested ids or physical group, and only for a strain
component Tri31 could answer. The read is unchanged when:

- the file holds the Tri31 rows (for example under `material.strain`);
- the read leaves the Tri31 elements out, as `element_ids=[]` does;
- the component is a stress, or a strain Tri31 cannot answer
  (`curvature_xx`);
- the stage lists no elements.

On a partitioned run only the partition that holds Tri31 refuses.
