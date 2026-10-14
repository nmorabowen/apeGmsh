### ADDED — apeGmshViewer: the section-plane gizmo (V4e2, #1903; ADR 0083 slice 2)

Each section plane in the app now has a gizmo, as the Qt results viewer's
does (ADR 0083 S2): a translucent quad over the undeformed model's
cross-section and an arrow pointing at the kept side. Dragging the quad (or
the arrow's shaft) with the left button moves the plane along its normal, one
offset event per move, with the gain bounded when looking nearly along the
normal. Dragging the arrow's tip turns the plane like a trackball about the
quad's centre, through one new `setClipPose` event per move that sets the
normal and the offset together, so the plane pivots about its quad and not
the world origin. A zero or non-finite pose is refused and the plane stays as
it was. The gizmo is never cut, by its own plane or another. It is never a
raycast target, so a left click on a gizmo still picks the model under it.
A press that misses every gizmo reaches camera navigation untouched, a legend
or panel over a gizmo takes the press, and the pointer leaving the canvas
ends a drag. The panel's **Show plane gizmos** footer button is live, and
each plane row has its own gizmo button. The deformed contour scrub with six
planes cutting and their six gizmos drawn is measured in
`apeGmshViewer/MEASUREMENTS.md`.
