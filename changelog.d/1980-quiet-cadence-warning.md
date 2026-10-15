### FIXED — the `n_steps=` cadence warning fires only where a recorder is emitted (#1980)

A record with `n_steps=` and no `dt=` warns that a classic recorder will
record every step. Since #1947 it warned at every translation of the
record, so a live session warned at `with spec.emit_recorders(...)` and
again at each `begin_stage`, and `Results.from_recorders(...)` warned
once per record and stage while reading the files back. Now
`to_tcl_commands`, `to_python_commands` and `begin_stage` warn once per
record, and the read path and the `with` entry are silent.

The silent callers use `quiet_emit_logical` in `results/writers/_cache.py`,
which filters only that one `UserWarning` by its message. A test fires the
real warning and checks the filter still matches it. The filter changes
the process-wide warning state while it runs, so a warning raised on
another thread at that moment can be hidden; apeGmsh itself runs none of
these callers off the main thread.
