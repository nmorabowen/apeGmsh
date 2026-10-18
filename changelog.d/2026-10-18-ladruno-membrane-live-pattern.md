### FIXED — the LadrunoMembrane fork warrant test applies its pressure through a pattern (#2143)

The fork-side live test for `LadrunoMembrane` read every corner reaction as
exactly 0 at FORK_PIN `ladruno-v20261017`. The fork was right: the element's
`-pressure` is a reference value that `LadrunoMembrane::addLoad` applies only
under an active pattern's `eleLoad -type -surfaceLoad`, and the test's deck had
no pattern. The test now opens a `Linear` series and a `Plain` pattern with
`surface_load(ele=...)` on the membrane. Its 160 kN equilibrium and 40 kN per
corner assertions are unchanged. Test-only; the bridge already emitted the
activation.
