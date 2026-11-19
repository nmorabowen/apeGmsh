### FIXED — the live route refuses a fork element option the build ignored (#2524)

Several Ladruno elements print a warning for an option they do not know and
still create the element without it: `LadrunoQuad`, `LadrunoCST` and
`LadrunoLST` print `ignoring unknown option <x>`, and `BezierTri6`,
`BezierTet10`, `LadrunoBrick` and `LadrunoBrick20` print
`unknown option '<x>' for <Type> <tag>` (measured at FORK_PIN
ladruno-v20261103). The live route verified such a call because it
succeeded, so the model ran without the option and nothing said so. The
live fork-element gate now reads the build's output on success too, and
refuses by name: the element type, the option, and the build. A known option
the build says it ignored (`LadrunoBrick ... -damp is only supported with
-formulation std|bbar; ignoring ...`, `-bulkViscosity is not applied with
-geom finite; ignoring it`, `-bipenalty ignored with -enforce al`) is refused
the same way. The element the build made without the option is removed from
the domain again, so a corrected retry can reuse its tag.

The gate now verifies a call by its shape (every string in place, every
number as `#`), not by its `-flag` set alone, so a stray token such as a
stock-style thickness `0.2` after an earlier clean call is still read. A
failing call on a taken tag, or a failing call of a class the build has
already built, no longer says the element "requires the Ladruno fork build".

A rejected option value (`unknown -formulation 'xyz' for LadrunoBrick 5`,
`LadrunoKinematicCoupling: unknown -enforce 'xyz'`,
`LadrunoForceBeamColumn3d: unknown -Integration type 'xyz'`, and the other
forms the pin prints) now says "value 'xyz' of option -formulation is not
supported by this build" instead of "requires the Ladruno fork build".

`LadrunoDispBeamColumn` printed nothing for an unknown flag on builds before
fork commit 34df942c7; at the pin it prints and fails the call, so it needs
no apeGmsh-side options table. A build older than that commit still drops
the flag in silence.
