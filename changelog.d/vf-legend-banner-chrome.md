### FIXED — apeGmshViewer: the notices banner no longer covers the colour scale; a user result offers no history or step range (#2924, #2902)

- An app-chrome still draws the notices banner open, and a long list
  covered the colour scale, which sat at a fixed 168 px. The scale stack
  now sits 8 px below the banner's measured bottom, as the left rail
  already did, in stills and in the window; under a one-line banner it
  stays at 168 px.
  A banner at its height cap no longer pushes the scales off the bottom:
  they start by half the window's height at the latest and are bounded
  16 px above the bottom, as the left rail is.
- A user result and a User Combination have one state, and a difference
  is read at the step shown only, so their scale's "history" and "this
  step" buttons both showed one range under two names. A user result and
  a combination now offer "own range" and "fixed"; a difference offers
  "this step" and "fixed". The first button is on unless the range is
  fixed, and is the way back from a fixed range. A Results field keeps
  "history", "this step" and "fixed".
