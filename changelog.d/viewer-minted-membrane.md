### FIXED — apeGmshViewer draws bridge-minted `LadrunoMembrane` panels (#2667)

A `LadrunoMembrane` minted with `fem_eid = -1` is now drawn as the polygon its
archived `-nodes` and `-hole` loops name, and it counts in the header's
"OpenSees-only" total. A `pg=` panel is not drawn twice, because its mesh cells
already show it. Any other minted row with no connectivity to draw is named in
a warning instead of vanishing.

The opening view now counts lines as well as faces, so a one-storey frame with a
slab opens in the isometric view with its columns visible.
