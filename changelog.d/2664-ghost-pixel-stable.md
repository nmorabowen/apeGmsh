### apeGmshViewer: the reference ghost draws the same pixels every time

A view spec may now ask for `deform.ghost: true` on a model drawn opaque. Spec v1 refused it because two
shots of one spec differed by up to 1,200 pixels and 8 levels: a blended triangle in
the 4x MSAA canvas gives other pixels from frame to frame with nothing changed. The
ghost's faces and outline edges now draw into their own target with no MSAA, tested
against the depth of the opaque scene, and one full-screen triangle lays the result
over the canvas. The target is 2x the canvas, so a ghost shown on screen draws the
opaque scene twice. Lines that were anti-aliased over 4 samples are now anti-aliased
over 4 pixels of the target, and the ghost now lies over the deformed mesh's outline
edges, where it used to lie under them, so its pixels change slightly in the app as
well as in a still.

Over translucent surfaces (surface opacity below 1, the default for a model with both
solids and lines, and the 3-D frames the line diagram draws at 0.25) the ghost stays
under the mesh, with its earlier look, and is not
pixel-identical: the mesh's own blended triangles are in the MSAA canvas. So a view
spec refuses a ghost shot there, with that reason, and the other shots still run. The
app-chrome capture race is tracked apart in #2696.
