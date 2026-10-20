### FIXED — native reader warns about unplaceable simplex Gauss rows only for groups the read returns (#2275)

`NativeReader.read_gauss` checked a group's stored simplex Gauss coordinates
before applying the `element_ids` filter, so a read that excluded an
unplaceable TenNodeTetrahedron group still emitted
`LegacySimplexGaussCoordsWarning` for it (and used up the once-per-class
warning). The check now runs after the filter. A new test also pins the
`coord_system` gate that keeps a gmsh-frame class from being remapped.
