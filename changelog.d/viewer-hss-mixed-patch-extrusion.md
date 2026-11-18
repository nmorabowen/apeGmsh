### FIXED — apeGmshViewer extrudes a fibre section of rect walls and circ corners (a cold-formed HSS)

With Frames → extrude on, a `Fiber` section whose patches mix `rect` and
`circ` patches was drawn as a bare line, because the reader drew a patch
outline only when every patch was one kind. The reader now unions such a
mix exactly: edges the patches share cancel, and what is left is the outer
loop plus any hole. An HSS of four wall rects and four quarter-ring corners
extrudes as a rounded square tube. A mix that overlaps, covers a point
twice, or holds a full-circle `circ` still draws nothing rather than a
wrong outline.
