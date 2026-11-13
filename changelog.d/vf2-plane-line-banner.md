### FIXED — apeGmshViewer: a von Mises contour that leaves out-of-plane zz at 0 says so on the banner (#2951)

A derived contour (`von_mises_stress`, `von_mises_strain`) on 2-D
elements that the model does not classify as plane stress or plane
strain, or whose material gives no Poisson's ratio, leaves their zz at
0. Until now only the render manifest's `contour.derived.warning` said
so. The live app, an app-chrome still and a model-chrome still with its
banner layer said nothing.

The banner now shows the same line as the manifest, for example
"Out-of-plane stress_zz recovery is incomplete: 15 of 15 element(s)
could not be classified as plane stress / plane strain. stress_zz is
left at 0 there." The manifest's `banner` list includes it too. Picking
a stored component, turning contours off, or opening a set whose model
classifies every element removes the line. A model that classifies
every element never shows it.
