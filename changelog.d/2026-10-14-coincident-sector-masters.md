### FIXED — unfragmented sector masters no longer double-claim seam slave nodes (#1964, #1638 item 2)

Several NTS contacts on one slave label de-duplicated their seam slave nodes
only when their masters shared a node id. Two pile sectors meshed separately
(coincident at the seam but not fragmented) share none, so 12 seam slave
nodes stayed in both contacts, the fork added their tractions, and nothing
warned. Masters now also touch when a node of one lies within `tol` of a node
of the other (an exact distance test, `tol` = 1e-6 x the two masters' joint
bounding-box diagonal), so the unfragmented case is de-duplicated and warns
`ContactSlaveOverlapWarning` exactly like the fragmented one.

This also applies to separately meshed bodies that abut, for example a slab
and a wall standing on it, each the master of an NTS contact on one soil label:
their shared slave nodes stay with the first contact, with the warning. A later
contact left with no slave node is refused, and the refusal names its own
`slave_entities=` as the remedy, as for fragmented masters.
