### Fixed — staged models draw only each stage's active elements, in every view (X2-f, #1722)

Since X2-b (#1700) moved `Results.export_animation` onto the session path,
a staged model animated every element at every stage. Before that, the Qt
viewer hid the elements not yet activated, and those already removed, in
the animated stage. Every realized mesh pane now applies that rule, from the
same staged-program data. **Behaviour change:** stills on staged models now
hide inactive elements too. That covers `Results.render`, `render_pack`,
`assess(figures=True)`, the CLI `render` and the session window's panes,
which used to draw every element. A model with no staged program, or a
capture stage that pairs with no program stage, still draws every element.
A scoped view whose elements are all inactive in the stage raises instead
of drawing the full mesh.
