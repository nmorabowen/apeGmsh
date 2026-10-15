### ADDED — `Results.from_recorders(..., stages=live.stages)` reads every stage of a live recorder run (#2035, G8b)

`Results.from_recorders` used to read one stage per call (`stage_id=`). It
now takes `stages=`, the stages a `spec.emit_recorders(...)` run completed
(`live.stages`, each with a `name` and a `kind`). It transcodes them into one
native file, with one stage each, carrying each stage's name and kind. Each
stage is read from its own `<name>__` files. The cache key covers every
stage. `stages=` cannot be combined with `stage_id=`, `stage_name=` or
`stage_kind=`, even when they are given their default values: both now
default to `None`, which still means `"analysis"` and `"transient"`. An
empty `stages=`, a repeated name, or a kind other than `static` or
`transient` raises before anything is written. A generator is read once.
`RecorderTranscoder` takes the same `stages=`.

`RecorderTranscoder` now writes a sibling `.partial` file and renames it onto
the target when the write succeeds. Before, a write that failed partway, for
example on a stage kind the writer refused, left a partial file. That file
sat at the cache path, so the next `from_recorders` call returned it.

`tests/test_results_out_multistage.py` runs a gravity stage (`LoadControl`)
and then a `Newmark` stage at `dt=0.01` on stock openseespy. Both stages read
back through one call with kinds `static` and `transient`. Each stage's
last-step displacement equals `ops.nodeDisp` at `rtol=1e-10`.

The work fits under the X4-b LOC ratchet without raising it. The transcoder
parses every record through one dispatch table instead of four branches.
