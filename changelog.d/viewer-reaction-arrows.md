### ADDED — apeGmshViewer: reaction-force arrows on a results step

When the results step on screen records reaction forces (`reaction_force_x/y/z`), the Loads card offers a
**reactions** switch (off by default). It draws one arrow per node with a reaction: the tip on the node, pointing
along the reaction so it points at the support from outside, with a short tick across the tail at 45° to the shaft's
perpendicular to tell it from a load. Reactions follow the step as it changes, and share the load arrows' head size,
scale, colour menu and thinning; by type they take the theme's dark teal. The card now also shows on a model with
no loads when its results record reactions.
