### FIXED — apeGmshViewer reads `.ladruno` node histories past 2^24 nodes (#2354)

The viewer found a node's column for its history (`readNodeHistory`, behind the
node probe and the history plot) through a `Map` from node id to column, built
from every node id of the stage. A `Map` holds at most 2^24 entries in V8, so a
stage with more than 16,777,216 nodes failed its first node-history read with an
unnamed `RangeError`. A part set had the same limit when it opened: it built the
union of its parts' node ids with a `Set` and a `Map`. Both now use a sorted
typed array and a binary search, as the element-id checks do. An ascending ID
list, which the fork writes, is searched in place. The reads are unchanged, bit
for bit, on every committed `.ladruno`, and an unknown node id is still refused
by name.
