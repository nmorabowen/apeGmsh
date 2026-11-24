### FIXED — apeGmshViewer draws piles on soil springs as broken lines in the deformed shape (#3138)

With the deformed shape on, each `zeroLength` spring between a pile node and its
fixed anchor was stretched into a stub, because each end moved by its own node's
displacement; thousands of stubs made the piles look dashed when no contour was
on. A segment of zero reference length now stays zero-length in the deformed
shape, as it draws undeformed. The Springs card still shows the springs' forces.
