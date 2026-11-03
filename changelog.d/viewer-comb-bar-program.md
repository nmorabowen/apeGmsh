### FIXED — apeGmshViewer: the bar along a distributed load's arrow tails draws again

Since the load arrows became GPU instances, the bar that joins the tails of a beam's distributed-load arrows was
drawn by the arrows' own shader, because three.js cached both under one key, so the bar did not appear. Each glyph
shader now compiles to its own program.
