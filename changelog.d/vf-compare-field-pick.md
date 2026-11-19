### FIXED — apeGmshViewer: picking a legend field while a compare variant's user result is drawn draws the picked field (#2943)

With a variant study open and a compare variant's user result drawn
(the variants panel's Draw), a field picked in the legend changed the
contour pick but left the compare result on screen, so only the panel's
Hide brought the picked field back. Picking the field the contour
already held did nothing at all.

Now any contour field pick, nodal, element or the window's own user
result, replaces the compare result as Hide does, and the picked field
is drawn. A pick the viewer refuses (a field the stage does not record)
leaves the compare result as it was.
