### CHANGED — `Results.viewer()` in a notebook no longer falls back to the web viewer for an in-memory Results (X2-c, #1664)

**Changed (behaviour):** inside a Jupyter / IPython ZMQ kernel,
`results.viewer()` (auto mode) on a hand-built, in-memory Results (one with
no backing file) now raises `RuntimeError` instead of opening `show_web()`.
The message names the way out: open the results from a file with
`Results.from_native`, `from_mpco`, `from_ladruno`, `from_recorders` or
`from_fem`, then call `viewer()` again. Every public constructor already sets
a file, so those Results keep spawning `python -m apeGmsh.viewers <path>` in
a notebook, unchanged. Nothing is converted to a temporary `.h5`: the reader
protocol does not carry everything a native file needs (spring results,
line-station local axes, the nodal-force frame), so a conversion would lose
data. `show_web()` and `serve_web()` still exist and can be called directly;
`Results.viewer()` no longer reaches them.
