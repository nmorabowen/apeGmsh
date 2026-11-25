### FIXED — a distributed load on a mesh selection refuses instead of loading nothing

A mesh selection stores nodes and no element connectivity, so
`g.loads.line`, `g.loads.surface.*`, `g.loads.gravity` and `g.loads.volume`
on one resolved to **zero** records without a word (the element form of the
line and surface loads failed with an unpacking error). They now raise a
`ValueError` that names the target. A selection also shadows a physical
group of the same name, which is the default name `from_physical` gives it;
the message says to use `pg=` or `label=` there. Point loads and face
resultants take selections as before.
