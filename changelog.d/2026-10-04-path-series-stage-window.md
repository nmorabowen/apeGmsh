### FIXED — a stage `Path` series that is zero at every increment now warns instead of applying no load in silence (#1334)

Every stage closes with `loadConst -time 0.0`, so the next stage's
pseudo-time restarts at 0 unless `s.set_time(t)` moves it. A `Path` series
written on a global clock across stages (`time=(1, 2)` in a stage that runs
over `[0, 1]`) read 0 at every increment, and the deck exited 0 having
applied none of that pattern's loads. When a stage closes, the bridge now
evaluates each stage pattern's `Path` series at the pseudo-time of every
increment, the way OpenSees reaches and evaluates it (sequential `t += h`
from the stage's start; `PathTimeSeries` for `time=`, `PathSeries` for `dt=`
with `start_time` and `prepend_zero`). It raises
`SeriesOutsideStageWindowWarning` (exported from `apeGmsh.opensees`) when
every factor is zero. The warning names the stage, the pattern, the series'
support, the stage's window, and the fix (`s.set_time(t)` at the series'
onset, or a re-based time axis). It is silent for a series that is non-zero
at any increment, and for `Linear`, `Constant` and the other series. A
`file=` series is not checked, and neither is a stage whose analysis solves
for its advance (`DisplacementControl` and the other non-load-control
static integrators, an adaptive `LoadControl` with `min_lam`/`max_lam`,
`VariableTransient`). An analysis class the check does not know raises
`TypeError`.
