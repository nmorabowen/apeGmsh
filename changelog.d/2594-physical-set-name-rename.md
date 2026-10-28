### FIXED — `g.physical.set_name` renames a named group, and refuses a name held at another dim (#2594)

`g.physical.set_name(dim, tag, name)` promised "Assign (or rename)", but on a
group that already had a name it did nothing: gmsh's `setPhysicalName` keeps a
name the group already has, and nothing was raised. It now renames in place:
the group keeps its tag and entities, takes the new name, and no longer
answers to the old one, through `get_fem_data` and the `model.h5` round trip.
A group at another dim that shares the old name (only raw gmsh can make one)
keeps it.

It also refuses, before any gmsh write, what it used to accept or drop
without a word:

- a name that a group at another dim already holds, as `g.physical.add`
  refuses it (a physical-group name maps to one dimension);
- a label's group (rename it with `g.labels.rename`);
- `""` for a named group (use `g.physical.remove_name`).

Setting a group's own current name, including `""` on an unnamed group, is
still a no-op.
