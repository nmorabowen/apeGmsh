### CHANGED — apeGmshViewer: load arrows sized on screen, thinned when dense, one colour by default

The load and prescribed-SP arrows keep their place in 3-D but are now sized on screen. Each head is one flat
triangle, twice as long as it is wide, turned toward the camera, and as many pixels long as the new **head** field
on the Loads card (12 by default), at any zoom. Each shaft keeps its length in model units but is never drawn
shorter than 15 or longer than 80 pixels (times the scale field), so zooming in no longer fills the screen with one
arrow and zooming out no longer shrinks them to dots. Where arrows would overlap on screen, the new **thin out**
switch (on by default) draws the largest load of the cluster; zooming in brings the rest back. The arrows are drawn
in one colour by default; the colour menu also offers the load type's colour or jet over each kind's magnitude, and
the card's chips follow it. A prescribed SP's cross is one flat bar.

The arrows are GPU instances: on a 100 000-node load case they take 6.5 MB instead of 46 MB and build in 185 ms
instead of 314 ms, and zooming or orbiting rebuilds nothing.
