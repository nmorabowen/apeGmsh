### CHANGED — apeGmshViewer: a fixed-width selection halo, `F` frames the selection, a pulse on select (ADR 0112, V2e round 2)

A single picked element is unmistakable at full-model zoom: a screen-space
halo of fixed width (3 device pixels, `src/renderer/selection.ts`) is drawn
on top of the thick outline and fill, independent of element size and zoom,
for beams, shells and solids alike; the halo swells to three times its width
and back over 650 ms when the selection changes; the mask it needs exists
only while something is selected. `F` dispatches the new `frameSelection`
store event (any panel can), and the frame effect moves the camera to the
selection's bounding sphere, or to the whole model when nothing is selected.
`npm run capture` writes a third still, `<out>.framed.png`, after `F`, and
`npm run measure` records a second orbit with the selection active
(`orbitSelected` in its JSON).
