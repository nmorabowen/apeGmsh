### ADDED — IMPL-EX time driver `ops.implex_time()` and the dTime-trap refusal (ADR 0113)

`ops.implex_time()` declares STKO's IMPL-EX `dTime` driver on a staged
model. The bridge finds the target elements itself: every element whose
section and material chain reaches an ASDConcrete3D or ASDConcrete1D with
`implex=True` or `eta > 0`, or an ASDSteel1D with `implex=True`.

It creates three persistent parameters (`dTime`, `dTimeCommit`,
`dTimeInitial`) over those elements; on a partitioned deck each rank
attaches only its own. It then writes each stage's increment immediately
before the stage's analyze loop, which is what STKO's
`STKO_DT_UTIL_OnBeforeAnalyze` does before every increment.

The Tcl and Python decks carry the same driver. The H5 archive and the live
emitter refuse a model that uses it. Partitioned decks are for OpenSeesMP;
OpenSeesSP is not supported, and the bridge cannot detect it at emit.
`ops.implex_time(mode="off")` declares that nothing writes `dTime*`.

The bridge now refuses the IMPL-EX trap. Once an element's materials get a
`dTime*` write through `s.update_parameter`, they stop following OpenSees'
own increment. From that stage on, every stage must write `dTime`, equal to
its own increment, on every element switched so far. A deck that does not
is refused at emit, and the refusal names the stage and the uncovered
elements.

The driver also refuses:
- an unstaged model;
- a stage without one known increment, such as `VariableTransient` or a
  `LoadControl` whose `min_lam` differs from its `max_lam`;
- a stage that activates or removes a target element;
- a target group with no elements;
- `s.update_parameter` writes of `dTime*` alongside it.

Not in this version: the adaptive transient loop, a `UniformExcitation`
inside a stage, and typed `implexAlpha` / IMPL-EX error control. ADR 0113
decides them as the next PRs.
