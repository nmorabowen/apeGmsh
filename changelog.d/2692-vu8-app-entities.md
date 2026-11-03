### apeGmshViewer: draws the `.user.h5` `/entities` zone: cuts, storeys and point sets (ADR 0130 slice 8)

The app now reads `/entities` of the set's `.user.h5`. It draws each standing row
where it is: a section cut as its bounding polygon, or as its plane where the plane
crosses the model; a storey as the plan outline of its nodes at its level; a point
set as a marker at each point. A cut shows a tick to its kept side, and a cut or a
storey shows a marker at its reference point. Only a current zone is drawn: a stale
one (the model hashes differ) is not. A storey naming a node the model does not hold
is not drawn, and the app logs why.

The reader follows Python's. A missing table, a malformed column, bad offsets or an
element index space other than OpenSees tags refuse the zone. A bad row refuses
alone and keeps its position. An `ON_ENTITIES` user result naming a row that the
zone does not hold, or that it refuses, is refused alone as `unresolved`; before
this slice the app refused every such result. The app's zone target and floor gain
`entities` at 1.0. The app reads a cut's stored resultant to the bit. On the STKO
elastic frame it equals the cut kernel's.
