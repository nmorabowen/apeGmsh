### ADDED — apeGmshViewer: section planes, part 1 (V4e1, #1834; ADR 0083 slice 1)

The app's **Section planes** panel cuts the model with up to six planes, as
the Qt results viewer does (ADR 0083 Parts 1, 4 and 5). Add a plane normal to
X, Y, Z or the current view, through the model's centre; flip it, turn it on
or off, delete it, set its normal from the chips or three Custom fields
(normalised on entry; a zero normal is refused), and move it with a slider
over the undeformed model's extent along the normal or an unclamped numeric
field. "Apply cuts" turns every cut off and keeps the planes. Activating a
seventh plane is refused with a notice. Planes are world-space three.js
clipping planes on every drawn material, put on before every render, so the
faces, edges, lines, contour, deformed shape, ghost, missing-displacement
markers and selection are all cut, including objects made after the plane
(a stage change, a re-read, the ghost turned on), and the deformed shape moves
through the planes while scrubbing. A click whose hit is on the cut-away side
is a miss; clicking through the hole onto geometry behind it is not attempted
(ADR 0083 open question 3). A solid is cut hollow: caps (V4e3) and plane
gizmos (V4e2) come later. Planes last for the window session and survive
re-reads. With six planes cutting, the deformed contour scrub measured
136 steps/s at worst (`apeGmshViewer/MEASUREMENTS.md`).
