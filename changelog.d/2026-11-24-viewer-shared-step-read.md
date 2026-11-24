### CHANGED — one step read serves the viewport and the tool windows (ADR 0137 H3, #3059)

The app keeps its last four step reads, so the viewport and a tool window asking for the same step
share one read instead of two. A kept read is used only while the results file on disk is the one
read, and only for the same request; a tool that wants a step's nodal values can use any kept read
of that step. The Step readout tool window now also shows the chosen component's range over the
nodes at the step on screen, read this way. `npm run profile -- <model> --tool=step` profiles with
that tool window open.
