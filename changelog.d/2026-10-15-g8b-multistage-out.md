### ADDED — `Results.from_recorders(..., stages=live.stages)` reads every stage of a live recorder run (#2035, G8b)

`Results.from_recorders` used to read one stage per call (`stage_id=`). It
now takes `stages=`, the stages a `spec.emit_recorders(...)` run completed
(`live.stages`, each with a `name` and a `kind`). It transcodes them into one
native file, with one stage each, carrying each stage's name and kind. Each
stage is read from its own `<name>__` files. The cache key covers every
stage. `stages=` cannot be combined with `stage_id=`, `stage_name=` or
`stage_kind=`. An empty `stages=`, or one that repeats a name, raises.
`RecorderTranscoder` takes the same `stages=`.

`tests/test_results_out_multistage.py` runs a gravity stage (`LoadControl`)
and then a `Newmark` stage at `dt=0.01` on stock openseespy. Both stages read
back through one call with kinds `static` and `transient`. Each stage's
last-step displacement equals `ops.nodeDisp` at `rtol=1e-10`.

The work fits under the X4-b LOC ratchet without raising it. The transcoder
parses every record through one dispatch table instead of four branches.
