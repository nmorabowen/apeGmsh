### FIXED — apeGmshViewer: each `npm run capture` step starts from the view the model opened on, and a timed-out wait stops the run

The capture harness's steps used to start from whatever the step before them
left, so a still depended on the order of the sequence (#3079 was one case):
the clip stills took an element field a file refuses, the combination stills
ran against the fake `<model>-other.h5` the workspace step opened, and the
deformed-contour still took the component the contour step picked last. Its
waits returned `false` on timeout, and most call sites captured anyway, so a
slow machine produced a wrong still without a word.

- `resetView` (`src/renderer/harness/reset.ts`) runs before every capture
  step. It brings back the model main opened (reading it again after the
  workspace step's set switch), the capture window's size, the mesh phase,
  colouring by group at the opening opacity, the undeformed shape, the
  contour defaults with the first results stage's default pick, every
  overlay and section plane off, and the camera pose the model opened on.
- `until` and `untilSteady` throw a `WaitTimeout` naming the still. The
  workspace and combination steps' polling loops do the same, and
  `combine-listed` waits for the combination's colours on screen instead of
  a fixed 800 ms.
- The workspace step opens the real model again after its set-switch still.

The contour, deformed-shape and cut-face steps that colour by contour now
pick the shown stage's default field themselves. Stills change once where a
step had inherited state; the before/after pairs went to D7 review.
