### apeGmshViewer: the nD material tester names an old fork build, and its messages are the same run to run

- A fork build older than `model testNDMaterial` (WP-248) now gets "this fork build has no nD material tester … install ladruno-v20261123 or later" instead of a Tcl trace.
- A solver message no longer carries the run's random temporary folder: `(file "…\test.tcl" line N)` reads `(test script line N)`, so two runs of one failure give the same text and the same still.
