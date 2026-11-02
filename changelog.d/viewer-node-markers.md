### ADDED — apeGmshViewer: node markers with a chosen size

**show** on the Display card's Nodes row draws a filled circle at every drawn node, in one draw call, and follows
the deformed shape, hidden groups and section cuts. They turn on at a size chosen for the model; a slider sets your
own and **auto** chooses again. Off by default, and the markers cost nothing until first turned on.

The Loads panel no longer recomputes its summary on a change that cannot affect it, such as a hover or a marker
size. On a 129k-cell model that took a store change from about 159 ms to under 1 ms.
