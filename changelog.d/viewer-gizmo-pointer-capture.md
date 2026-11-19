### FIXED — apeGmshViewer keeps a section-plane gizmo drag when the pointer leaves the canvas (#1919)

A gizmo drag now holds pointer capture, so it continues over a panel or outside
the window and ends on release. A cancelled pointer or a lost capture still ends
it, leaving the plane at the pose of the last move.
