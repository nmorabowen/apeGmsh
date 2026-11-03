### ADDED — apeGmshViewer: a Reactions card with reaction-force arrows on a results step

When the results step on screen records reaction forces (`reaction_force_x/y/z`), a **Reactions** card appears
in the left rail. Its **3D reactions** switch (off by default) draws the reactions as arrows: the tip on the node,
pointing along the force so it points at the support from outside, with a short tick across the tail at 45° to the
shaft's perpendicular to tell it from a load. A menu draws the **resultant** at each node, its **X, Y, Z
components** along the global axes as three arrows, or one of them. The arrows follow the step as it changes. The
card draws reactions at the model's supports only (nodes a `fix`, an SP or a support hold restrains); an **all
nodes** switch adds what the results record elsewhere, a free node's residual or a constrained node's constraint force.
The card has its own scale, head size (pixels), colour (one colour, the reaction colour, or jet over the magnitude) and
thinning, which keeps or drops a node's arrows together.
