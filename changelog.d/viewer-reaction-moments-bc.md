### ADDED — apeGmshViewer: reaction moments as half circles; the BC toggle draws the supports

The Reactions card now draws reaction moments (`reaction_moment_x/y/z`) as half circles about the moment axis, in
the plane perpendicular to it, with a flat head by the right-hand rule and the reaction tick; its new F and M
switches show forces, moments or both, and the card appears for a step that records only moments. Viewed in the
arc's own plane a half circle goes edge-on, the price of drawing the true axis. The Display card's BC button,
until now a placeholder, draws the model's supports at the stage on screen: an arrow into the node, with the SP bar
at its tip, for each restrained translation, and a half circle with no head for each restrained rotation. It needs
no results. DOFs follow the structural numbering (3 per node in 2-D, 6 in 3-D).
