### FIXED — apeGmshViewer: a coarse contour shows only colours of its colormap, never a blend between corner colours (#2939)

The app coloured a contour per vertex and let the GPU blend those colours
across each triangle. On a coarse mesh that blend left the colormap: a
one-element quad under jet showed a purple diagonal between a blue corner and
a red one, a colour jet never has. The numbers in the legend and the manifest
were right; the picture was not.

Now each vertex carries where its value falls on the colormap, and every pixel
looks its colour up in the colormap itself (a 1-D texture). This applies to
the mesh faces, the extruded frame sections and the cut-face caps. A
primitive with no data is still the no-data grey everywhere. Beams drawn as
lines still blend their two end colours, as before.
