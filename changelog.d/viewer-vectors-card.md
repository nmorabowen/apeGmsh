### apeGmshViewer: the Vectors card draws a nodal vector field as arrows

A new Vectors card in apeGmshViewer draws one results step's nodal vector field. It is
the viewer twin of the Qt vector-glyph diagram. Its field picker offers each nodal field
the stage records with two or more axes, such as `DISPLACEMENT`, `VELOCITY` and
`ACCELERATION` in a `.ladruno`. Each arrow starts at its node and points along the vector,
and it follows the deformed shape. A rotational field (`ROTATION`, `ANGULAR_VELOCITY`)
draws a half circle about each node's axis instead. A 2-D model's lone `Rz` is not
offered, because it may hold a pore pressure. The card draws the resultant, the X, Y and Z
components, or one axis. It colours the glyphs in one colour, by type (translation or
rotation), or in jet by magnitude. Head size, scale and thin-out work as on the Reactions
card. Vectors below 1e-6 of the step's largest are left out as noise, and thinning is
decided per node. The arrow record gains an `anchor` field: a vector's tail sits on its
node, while a load's or a reaction's tip still does.
