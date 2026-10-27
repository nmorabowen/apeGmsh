### FIXED — `OpenSeesModel.from_h5` joins MP and equation-constraint rows to their emit order in one pass (#2544)

Reading an archive back (`OpenSeesModel.from_h5`, which `Assembly.bridge`
runs on every instance) joined each `equalDOF`, `rigidLink`,
`rigidDiaphragm`, `embeddedNode` and `equationConstraint` row to the
`/opensees/program` call that wrote it by scanning every program run for
every row. That is quadratic when each constraint is its own call. An
archive with 50,001 `equalDOF` rows took about 200 s to bridge on a desktop
and 140 s on a slow CI runner. The join now builds one row-to-index map per
verb, so the same bridge takes about 20 s. The result is unchanged: the
first run that covers a row still wins, and a row no run covers still
raises the same `LookupError`.

CI: the `suite` job's timeout returns from the temporary 60 minutes to 40,
and the Deps canary's `suite` job goes from 25 minutes, which it had hit
every night, to 40. The `suite` job also prints the runner's CPU and the
tests that take longer than 5 s.
