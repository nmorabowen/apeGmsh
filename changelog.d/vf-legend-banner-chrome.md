### FIXED — apeGmshViewer: the notices banner no longer covers the colour scale; a user result offers no history or step range (#2924, #2902)

- An app-chrome still draws the notices banner open, and a long list
  covered the colour scale, which sat at a fixed 168 px. The scale stack
  now sits 8 px below the banner's measured bottom, as the left rail
  already did, in stills and in the window; under a one-line banner it
  stays at 168 px.
- A user result has one state, so its scale's "history" and "this step"
  buttons both showed the result's own range under two names. Its scale
  now offers "own range" (on unless the range is fixed, and the way back
  from a fixed range) and "fixed". Every other contour keeps "history",
  "this step" and "fixed".
