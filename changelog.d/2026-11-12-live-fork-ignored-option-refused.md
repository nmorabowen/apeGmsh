### FIXED — the live route refuses a fork element option the build ignored (#2524)

Several Ladruno elements print a warning for an option they do not know and
still create the element without it: `LadrunoQuad`, `LadrunoCST` and
`LadrunoLST` print `ignoring unknown option <x>`, and `BezierTri6`,
`BezierTet10`, `LadrunoBrick` and `LadrunoBrick20` print
`unknown option '<x>' for <Type> <tag>` (measured at FORK_PIN
ladruno-v20261103). The live route verified such a call because it
succeeded, so the model ran without the option and nothing said so. The
live fork-element gate now reads the build's output on success too, and
refuses by name: the element type, the option, and the build.

A rejected option value (`unknown -formulation 'xyz' for LadrunoBrick 5`,
`LadrunoKinematicCoupling: unknown -enforce 'xyz'`,
`LadrunoForceBeamColumn3d: unknown -Integration type 'xyz'`, and the other
forms the pin prints) now says "value 'xyz' of option -formulation is not
supported by this build" instead of "requires the Ladruno fork build".

`LadrunoDispBeamColumn` printed nothing for an unknown flag on builds before
fork commit 34df942c7; at the pin it prints and fails the call, so it needs
no apeGmsh-side options table. A build older than that commit still drops
the flag in silence.
