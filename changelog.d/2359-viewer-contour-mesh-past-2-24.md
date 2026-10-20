### FIXED — apeGmshViewer: the contour's and the mesh build's node and element lookups no longer cap at 2^24 (#2359)

The readers no longer stop at 2^24 nodes (#2354, #2358), but the renderer path
still did. The contour, the deformed shape and the section caps found a node's
column through a `Map` over the field's `_ids` (`columnOf`), and the element
contour joined its rows through a `Map` over the element ids (`joinElements`).
The mesh build resolved node ids through a `Map` over every model node, and
picked colour groups through a `Map` over every grouped element
(`colourGroups`). It deduplicated solid faces and outline edges through a `Map`
and a `Set`. V8 caps a `Map` and a `Set` at 2^24 entries, so more than
16,777,216 nodes, elements, faces or edges stopped the load or the contour with
an unnamed `RangeError`. These now go through a sorted typed-array lookup with a
`Map`'s reading API (`IdLookup` in `mesh/build.ts`), and the deduplication sorts
typed arrays. Each answers what its `Map` answered: a repeated node id keeps its
first column, a cell in two groups keeps the first group, and a node id that the
model lists twice resolves to its last row. The contour values, the deformed
shape, and the `buildMesh` output are byte-identical on every committed
`model.h5`, `.ladruno` and native `results.h5`. Collections of this kind elsewhere
in `state/` (the load, the search index, the activation filter, the adjacency, the
line diagrams, the extrapolation plan) are left for follow-up slices. `buildMesh`
itself still stops at about 2^27 entries in its plain-array position buffers,
about 14.9 million drawn triangles (#2362).
