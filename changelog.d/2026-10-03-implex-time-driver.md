### ADDED — IMPL-EX time driver `ops.implex_time()` and the dTime-trap refusal (ADR 0113)

`ops.implex_time()` declares STKO's IMPL-EX `dTime` driver on a staged
model. The bridge finds the target elements itself: every element whose
section and material chain reaches an ASDConcrete3D or ASDConcrete1D with
`implex=True` or `eta > 0`. It creates three persistent parameters
(`dTime`, `dTimeCommit`, `dTimeInitial`) over those elements, each rank
attaching only its own on a partitioned deck, and writes each stage's
increment right after the stage's `analysis` line. That is what STKO's
`STKO_DT_UTIL_OnBeforeAnalyze` does before every increment. The Tcl and
Python decks carry the same driver; the H5 archive refuses a model that
uses it. `ops.implex_time(mode="off")` declares that nothing writes
`dTime*`.

The bridge now refuses the IMPL-EX trap. Once a stage writes any `dTime*`
through `s.update_parameter`, ASDConcrete stops following OpenSees' own
increment, so that stage and every later one must write `dTime` equal to its
own increment. A deck that does not is refused at emit, naming the stage.
The driver also refuses: an unstaged model, a stage without one known
increment (`VariableTransient`, an adaptive `LoadControl`), a stage that
activates or removes a target element, and `s.update_parameter` writes of
`dTime*` alongside it.

Not in this version: the adaptive transient loop, a `UniformExcitation`
inside a stage, and typed `implexAlpha` / IMPL-EX error control. ADR 0113
decides them as the next slices.
