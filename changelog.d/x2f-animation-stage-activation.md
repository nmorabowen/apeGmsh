### Fixed — `Results.export_animation` draws only each stage's active elements again (X2-f, #1722)

Since X2-b (#1700) moved `export_animation` onto the session path, a staged
model animated every element at every stage. Before that, the Qt viewer hid
the elements not yet activated, and those already removed, in the animated
stage. The session path now does the same, from the same staged-program data,
for the animation only. A model with no staged program, or a capture stage
that pairs with no program stage, still draws every element. A scoped view
whose elements are all inactive in that stage raises instead of drawing the
full mesh. `render`, `render_pack`, `assess`, the CLI `render` and the
`viewer()` window draw what they drew before: they never filtered by stage.
