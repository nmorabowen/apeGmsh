### FIXED — apeGmshViewer: the BC toggle reads a pinned flat slab as 3-D

With no `/meta/ndm` in the file, the BC toggle decided a model flat in z was 2-D unless a rotation was restrained,
so a slab pinned in x, y and z (fix mask `[1,1,1,0,0,0]`) drew its z restraint as a rotation. A fix mask longer than
3 now also marks the model as 3-D.
