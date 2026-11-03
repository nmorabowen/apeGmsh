### CHANGED — apeGmshViewer: load arrows have flat 2:1 heads, one colour by default

The load and prescribed-SP arrows keep their place in 3-D but their heads are no longer six-sided cones: each head
is one flat triangle, twice as long as it is wide, turned toward the camera, and the shaft stops at its base. A
prescribed SP's cross is one flat bar. Every head is the same size, set by the new **head** field on the Loads
card (a fraction of the reference arrow length, 0.3 by default), so a small load keeps a readable head. The arrows
are drawn in one colour by default; the card's colour menu also offers the load type's colour or jet over each
kind's magnitude, and the card's chips follow it.
