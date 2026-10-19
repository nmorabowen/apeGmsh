### TESTS — `convert_mpco` is proven across stage boundaries (ADR 0122 D8, #2177)

The `live-fork` lane runs a second MPCO -> `.ladruno` conformance oracle, on a
program with three `ops.stage` blocks: `gravity` (static, `LoadControl`, then
`loadConst`), `push` (static, `DisplacementControl`) and `shake` (transient,
`Newmark`), with an `eigen` solution recorded between gravity and push. The
`.mpco` and the `.ladruno` are recorded in one fork run. The converted file
passes the fork's `validate()` and equals the fork's file under the slice-6
comparator, now shared by both oracles. Separate checks compare each stage's
kind, name, step count, STEP ids and time axis with the program, and the last
time of each stage with `ops.getTime()`. Four mutations of the converted file
each fail the comparator: two stages swapped, a stage dropped, a wrong `KIND`
on stage 2, and a shifted time origin on stage 3. The fork's own stage `KIND`
is a strict xfail on #2103: a recorder with a modal channel labels every
stage `eigen`.
