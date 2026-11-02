### User Combinations evaluated per quantity on read (ADR 0130 slice 2)

`Results.combinations()` returns the set's combination table, from `.user.h5`
`/combinations` joined with `model.h5` `/opensees/combinations`, and
`evaluate(state, component, level=...)` combines a recorded result on nodes,
element nodes, Gauss points or line stations, from the per-term values of that
quantity. Every rule is evaluated: `linear`, `srss`, `abs_sum`, `cqc` (over the
modes of one set, a case's increments, or a whole case with its `sa_period`
groups), `directional` and `envelope`. NaN propagates through each, and sign is
derived from the rule and its terms. `evaluate(state, user_result=...)` follows
Amendment 1 A5: a stored row wins, `envelope` reads the terms' rows, and any other
rule needs `@linear=1`. A mode term names its set by `STEP` id, a stale user result
is refused unless `allow_stale=True`, and a state handle of another `Results` is
refused. Spectrum terms wait for a `/spectra` reader (slice 6) and refuse until
then. `tests/fixtures/user_results/conformance.json` holds the inputs and goldens
the app's TypeScript evaluator will share.

The two combination validators are now one, `opensees/_internal/_combination_rules.py`:
its cycle search is linear (a 1,000-row chain took 47 s on each `from_h5`), and a
`model.h5` combination over a spectrum case reads `bad-term` rather than
`bad-param`. `ModelData.from_h5` names the opensees zone when `/meta` stamps no
version for it.
