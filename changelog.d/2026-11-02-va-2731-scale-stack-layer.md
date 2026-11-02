### Fixed — apeGmshViewer: an app still with `left_rail` off omits the viewport scales too (ADR 0128 D9, #2731)

Since the contour's and the line diagram's colour bars moved off their left-rail cards onto viewport scales (#2701), an app still with every layer off still laid out the scale stack, because no capture layer named it. The `left_rail` layer now omits the scales with the cards they came from, so a still with `left_rail: false` lays out neither.
