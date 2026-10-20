### FIXED — apeGmshViewer reads native `results.h5` node histories past 2^24 nodes (#2358)

The viewer's native reader found a node's column for its history
(`readNodeHistory`, behind the node probe and the history plot) through a `Map`
from node id to column, built from every id in the stage's `_ids`. A `Map` holds
at most 2^24 entries in V8, so a stage with more than 16,777,216 nodes failed its
first node-history read with an unnamed `RangeError`. It now uses the binary
search the `.ladruno` reader uses (#2354). The reads are unchanged, bit for bit,
on every committed native `results.h5`, and an unknown node id is still refused
by name. A node id that a stage's `_ids` lists twice is now refused when the
file opens, naming the file, the dataset, and the id with its rows. Before, the file
opened silently: a step read returned both columns, the contour kept the first, and
the node history read the second. The viewer reads one partition per stage, so a
repeat there is always within one chunk; boundary nodes shared across partitions
never reach it. No other collection in the native reader grows with the node or
element count.
