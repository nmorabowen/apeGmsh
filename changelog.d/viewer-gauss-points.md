### ADDED — apeGmshViewer: Gauss point and beam station markers

**show** on the Display card's Gauss row draws the Gauss points of the results stage shown, and the beam stations of a
`.ladruno`, as accent-coloured filled circles in one draw call. Each point is its element's corners weighted by the
corner shape functions at the point's natural coordinates, and moves under the deformed shape by the same weights of
its corners' displacements, so interior points of solids move too. Hidden groups and inactive elements are left out.
They turn on at a size chosen for the model, smaller than the nodes'; a slider sets your own.

Groups the file gives no usable coordinates or verified element ids for are counted in a note, never placed by
guess. `.ladruno` element groups whose `@PARAM_DOMAIN` is `"bary"` (the reference tet and Bézier tri6 sets) are among
them: the reader keeps no coordinates for that domain yet.
