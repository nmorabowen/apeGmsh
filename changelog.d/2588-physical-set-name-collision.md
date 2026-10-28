### FIXED — `g.physical.set_name` refuses a name another group already holds at that dim (#2588)

`g.physical.set_name(dim, tag, name)` called gmsh's `setPhysicalName`,
which keeps one group per name and dim and drops a clashing call without a
word: both groups kept their old names and nothing was raised. It now raises
`ValueError` naming the group that holds the name, before any gmsh write,
so neither group changes. This matches `g.labels.rename` after #2585.

It also refuses names that start with the reserved `_label:` prefix, as
`g.physical.add` does: the name index skips label groups, so a clash with a
label could not be checked, and a user group renamed into that namespace
would vanish from `get_tag`. Setting a group's own current name, or `""`,
is still a no-op.
