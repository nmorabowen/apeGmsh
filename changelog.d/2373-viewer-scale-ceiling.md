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

The routes are a `.ladruno`, a partitioned `.ladruno` set, a native
`results.h5` and a `model.h5`. A `.ladruno` stage's nodes are its readable node
set, and a refused result does not count. Its elements are the Gauss buckets,
plus the line-station buckets in a verified file, counted per result token. A
set counts its nodes across the parts, with an interface node once. A
`model.h5` counts its OpenSees-only elements beside its cells. The mesh build
counts its drawn cells, line segments, triangles and outline edges before it
fills a buffer. It refuses more than 22,369,621 segments or edges, or
14,913,080 triangles: a JavaScript array holds 2^27 entries, at 6 floats per
segment or edge and 9 per triangle. A count exactly at a limit opens. A refused
results file leaves the loaded model on screen, with the message where the
results would be. Every committed fixture opens as before.
