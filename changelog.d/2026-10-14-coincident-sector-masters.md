### FIXED — unfragmented sector masters no longer double-claim seam slave nodes (#1964, #1638 item 2)

Several NTS contacts on one slave label de-duplicated their seam slave nodes
only when their masters shared a node id. Two pile sectors meshed separately
(coincident at the seam but not fragmented) share none, so 12 seam slave
nodes stayed in both contacts, the fork added their tractions, and nothing
warned. Masters now also touch when a node of one coincides in space with a
node of the other (within 1e-6 of their joint bounding-box diagonal), so the
unfragmented case is de-duplicated and warns `ContactSlaveOverlapWarning`
exactly like the fragmented one.
