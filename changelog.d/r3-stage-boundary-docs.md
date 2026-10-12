### Document what a stage boundary does

The concepts page and the skill reference now say what a stage boundary does: `loadConst -time 0.0` holds loads and resets time (unless `s.set_time`), `wipeAnalysis` clears the chain and the next stage re-emits all of it, the domain and recorders are kept, and `s.run(dt=)` has no effect under `Static`. A new test checks each claim against an emitted two-stage deck. Docs only; no code change.
