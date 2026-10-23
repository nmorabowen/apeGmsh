### FIXED — a stage flip counts only the elements in the domain when it runs (#2528)

The #2445 check holds a plastic flip of a ManzariDafalias or
LadrunoSANISAND material to every element that carries it. It counted
every carrier, whichever stage activates it. So a stage that flipped a
sand over pg A, every element alive at that stage, was refused for
leaving out pg C, which carries the same sand but is activated by a later
stage and is not in the domain yet. Since #2464 the emit and the
`model.h5` replay refused it too.

A flip now counts only the carriers alive when it runs: emitted globally,
or activated by that stage or an earlier one. The same rule runs at the
stage close, at the emit and on the replay. A carrier activated after a
plastic flip is still refused unless the stage that activates it flips
the group again (#2445), and that flip must reach every carrier alive at
it, the new one included.

The emit now gives each element spec its owning stage, read off the
stages' activations, as the replay reads it off the archive. Before, the
emit treated every spec as global. The stage close knows only the stages
closed so far, so a carrier registered before the flipping stage closes,
with no stage yet activating it, still counts as global there. Register
it after that stage.
