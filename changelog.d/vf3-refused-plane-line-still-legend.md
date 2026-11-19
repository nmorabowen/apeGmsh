### FIXED — apeGmshViewer: a refused shot's banner names the plane line, and an app still's colour scales clear the mesh (#2959)

A render shot that was refused after its view was drawn (for example a
deformed shape asked of a stage with no displacement) wrote a manifest
`banner` list without the derived contour's out-of-plane line, which an
ok shot of the same view carries. The refused entry now states the same
lines as the ok entry.

In an app-chrome still the banner is drawn open, with every line, and
the colour scales sat under it. With the plane line the banner grew
taller, and the scales covered the top of a 2-D mesh by about 60 px. In
an app-chrome still the scales now sit at the bottom of their column,
64 px off the image's bottom edge and above the Windows menu. The
interactive window is unchanged: the scales stay under the banner.
