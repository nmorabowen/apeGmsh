### FIXED — `ops.recorder.Ladruno(nodes_pg=..., elements_pg=...)` no longer drops the nodes

One OpenSees `MeshRegion` cannot hold a node list and an element list:
`setNodes` and `setElements` each wipe both, and `-ele` rebuilds the nodes
from the elements. The single `region $tag -node ... -ele ...` line the
bridge wrote for a Ladruno recorder with both selectors therefore kept only
the element nodes, and `nodes_pg` was lost behind an OpenSees warning. In an
embedded-rebar model (`g.rebar.place(coupling="embedded", emit_elements=True)`
with `elements_pg="solids"`), no bar node was recorded and the viewer drew the
cage undeformed.

A Ladruno recorder with a node selector and an element selector now emits two
regions, `region $t -ele ...` and `region $tn -node ...`, and passes
`-R $t -RN $tn` (fork WP-234): `-N` rows cover the element nodes plus the
`-RN` nodes, `-E` rows the `-R` elements only. This holds on the flat, staged
and partitioned routes. Only a fork build whose `ladrunoSchema` lists
`recorderLadrunoNodeRegion` reads `-RN`. On another build the live route raises
`BridgeError` naming the capability, and a Tcl / openseespy deck checks the
running build at its start and stops with the same message. A recorder with one
selector side, and every MPCO recorder, still emits one region.
