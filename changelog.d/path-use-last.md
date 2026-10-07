### FIXED — `Path` refuses the flags OpenSees drops on `time=`, gains `use_last`, and warns on a mid-stage unload (program slice R0-b, #1363)

`OPS_PathSeries` builds a `time=` Path as a `PathTimeSeries` and never
passes it `-startTime` or `-prependZero`, on stock or on the fork, so
`Path(time=..., start_time=...)` and `Path(time=..., prepend_zero=True)`
ran unshifted without a word. Both now raise `BridgeError` at construction
and say how to re-base the time axis. `Path` gains `use_last: bool = False`,
which emits `-useLast` so the series holds its last value past its support
instead of dropping to 0. On the `dt=` route both builds honour it. Stock
drops it on the `time=` route (only the Ladruno fork forwards it), and a
deck cannot know which build will run it, so `time=` with `use_last=True`
is refused as well. A stage whose pseudo-time window runs past a `Path`
series' last sample while that sample is non-zero and `use_last` is off
now warns with `SeriesEndsMidStageWarning`. A `dt=` series reads 0 at its
last sample's time too, so a ramp that ends exactly on the stage's last
increment unloads on that increment. `model.h5` stores the emitted tokens,
so `-useLast` round-trips with no schema change.
