### FIXED — apeGmshViewer: the command palette's contour range list offers what the legend offers (#2949)

- The palette's range list (Contour range) still offered "history" and
  "this step" for a user result, a User Combination and a difference,
  each of which has one range. It now offers the legend's buttons for
  every contour kind: "own range" and "fixed" for a user result or a
  combination, "this step" and "fixed" for a difference, and "history",
  "this step" and "fixed" for a Results field. Picking the own range
  leaves a fixed range and otherwise changes nothing, as the legend's
  button does.
