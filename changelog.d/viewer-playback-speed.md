### ADDED — apeGmshViewer: playback speed, a slider that follows the mouse, and no flicker per step

The step bar has a speed selector (×1, ×2, ×4, ×8, ×16). Playback now follows a
clock and asks for a new step only when the last one has arrived, so a model that
cannot keep up skips steps and still moves; before, each request replaced the one
before it and `tims_c3b` drew 2 of its 51 steps.

Dragging the step slider updates the view as it moves, one step per frame at most,
and the release lands on the step under the pointer. The bar updates in place
instead of replacing the slider being dragged.

A step being read no longer adds a line to the Deformed and Contour cards or text
to the step readout, which moved the panels on every step. It is said in the card's
heading and by a mark beside the readout, and only when the read takes over 400 ms.
