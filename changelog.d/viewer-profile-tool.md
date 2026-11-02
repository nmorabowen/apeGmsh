### ADDED — `npm run profile`: apeGmshViewer names its own bottlenecks, per scenario and per store subscriber

`npm run profile -- <model.h5>` in `apeGmshViewer/` opens the model in a real
view-mode window and drives it over the DevTools protocol through named
scenarios: load to the first frame, orbit, one store event, entering the
results, contour on, scrub, switching the contour field (nodal and element),
the deformed shape, hover, and the store event again with results on. For
each scenario it records the time per operation, the frame rate, a CPU
profile of the renderer and of the main process (top functions by self and
inclusive time, mapped to `src/` through the bundles' source maps), the
share of garbage collection, and the machine's CPU load. It also times every
store subscriber on each event, so a panel that recomputes on every change
is named by file and function. It writes `result.json` and a ranked
`summary.md`.

A scenario that changed nothing fails the run, checked against the store and
the viewport's pixels. A job that finishes late is reported as slow, with
its true time, never as a timeout. The viewer's source is unchanged: the tool
reaches the store with a one-shot debugger breakpoint, on a scratch profile
it deletes afterwards. Method and the first findings are in
`apeGmshViewer/MEASUREMENTS.md`, "Profiler".
