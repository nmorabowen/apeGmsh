### FIXED — a `.ladruno` strain read no longer drops Tri31 elements in silence (#2895, follow-up of #2893)

Tri31 answers no element-level strain: `ops.eleResponse(eid, "strains")`
is empty on it, so a record of `strains` leaves its elements out of every
strain bucket. A `.ladruno` gauss strain read then returned the other
classes' elements and no Tri31 rows, with no warning. On
`tests/fixtures/simplex_gp/tri.ladruno`, a whole-model `strain_xx` read
returned elements 2 to 5, and a read of the Tri31 element 1 returned an
empty slab.

The read now raises `MPCOStrainNotRecorded`, the `ValueError` the MPCO
reader raises for the same case (#2893), when a Tri31 element in scope
gets no row back. The message names the elements (fem ids under a tag
map) and says how to get the rows: declare the recorder with
`element_class_name="Tri31"`, which records strain per Gauss point under
`material.strain`. Elements in scope are the stage's own `MODEL/ELEMENTS`
listing, narrowed to the requested ids or physical group, and only for a
strain component Tri31 could answer. The read is unchanged when:

- the file holds the Tri31 rows (for example under `material.strain`);
- the read leaves the Tri31 elements out, as `element_ids=[]` does;
- the component is a stress, or a strain Tri31 cannot answer
  (`curvature_xx`);
- the stage lists no elements.

On a partitioned set only the partition that holds Tri31 refuses, and a
gauss read of a mode stage still refuses as a mode stage first.
