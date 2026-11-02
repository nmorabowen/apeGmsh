### apeGmshViewer: the reference ghost draws the same pixels every time

A view spec may now ask for `deform.ghost: true`. Spec v1 refused it because two
shots of one spec differed by up to 1,200 pixels and 8 levels. Blended triangles
tested against depth already in the 4x MSAA buffer give other pixels each frame,
with nothing changed. The ghost's outline edges and faces now blend before the depth
buffer holds anything, ahead of the opaque mesh.

The look changes in one way, in the app as well as in a still: the ghost's surfaces
sit behind an opaque deformed mesh wherever the two overlap, and no longer tint or
line the parts of it they used to pass in front of. Elsewhere the fill is within 2
levels of what it was. The ghost's 1-D lines are drawn as before.
