### FIXED — `g.labels.rename` refuses a new name that already labels another group at the same dim (#2577)

`g.labels.rename("B", "C")` when a label `C` already existed at one of
`B`'s dimensions used to recreate `B`'s group under the name `C`. Gmsh keeps
one physical group per name and dimension, so the recreated group ended up
with no name and `B`'s entities silently dropped out of every label. It now
raises `ValueError` naming the existing label and the dimension, before any
group is touched at any dimension. A label `C` at a different dimension
still does not block the rename, and renaming a label to its own name stays
a no-op. To merge two labels, remove one and re-add its entities under the
other name.
