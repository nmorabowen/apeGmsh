### FIXED — `g.model.boolean.fragment` returns only target-dimension tags, as documented

`fragment` documented "tags of all surviving entities at the target
dimension" but returned the top pieces of *every* input dimension, mixed in
one flat list. `fragment(slabs, columns + points, dim=2)` gave the slab
surfaces, then the column curves, then the points, with no dimension to tell
them apart and with repeated tags. It now returns only the entities at the
objects' highest dimension. That is `dim` for bare integer tags; for labels
or dimtags it is the objects' own dimension, so `fragment("slabs", "cols")`
on surfaces still returns the surfaces under the default `dim=3`.
Same-dimension calls (all volumes, the common case) are unchanged. Code that
read the tools' lower-dimension pieces from the return value should query
them with `gmsh.model.getEntities(d)` or through their labels.
