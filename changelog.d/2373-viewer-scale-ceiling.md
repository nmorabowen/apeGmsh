### CHANGED — apeGmshViewer refuses a model past its supported scale by name, at open (#2373)

The maintainer set the viewer's supported scale on #2362: 2^24 (16,777,216)
nodes and 2^24 elements per stage, and as many triangles as the mesh buffers
hold. Past those sizes the app used to stop with an unnamed V8 `RangeError`,
from a `Map` or a `Set` in the state code or from a plain array in the mesh
build. Each open route now checks its counts as soon as it knows them, mostly
from the dataset shapes before any row is read. A larger file is refused with a
message that names the file, the count, the limit and the cause, for example
`stage "MODEL_STAGE[1]" has 16,777,217 nodes; the viewer supports at most
16,777,216 (2^24) nodes per stage (JavaScript Map/Set limit, ruling on #2362)`.

The routes are a `.ladruno`, where elements are its Gauss and line-station
buckets, counted per result token; a partitioned `.ladruno` set, whose nodes
are counted across the parts with an interface node once; a native
`results.h5`; and a `model.h5`. The mesh build counts its triangles, and its
outline edges, before it fills a buffer. It refuses more than 14,913,080
triangles or 22,369,621 edges: a JavaScript array holds 2^27 entries, at 9
floats per triangle and 6 per edge. A count exactly at a limit opens. A refused
results file leaves the loaded model on screen, with the message where the
results would be. Every committed fixture opens as before.
