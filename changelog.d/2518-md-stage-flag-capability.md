### CHANGED — the ManzariDafalias constructor-reset check asks the build for `mdStageFlagPreserved`, on the deck, replay and live routes (#2518)

A Ladruno fork build without fork commit `43b235bbb` resets the class-wide
`ManzariDafalias` / `LadrunoSANISAND` stage flag whenever a new element or
material is built, so a model that builds one after a stage flipped the group,
and solves before flipping it again, ran the whole group elastic with no
warning (#2443, #2444). Since #2516 a deck of that shape checked the running
build against a list of measured `ladrunoBuild` shas, so a newer fork build
that carries the fix was refused until someone measured it.

Fork PR #1017 lists the fix in `ladrunoSchema` v1 as the capability
`mdStageFlagPreserved`, and `FORK_PIN` (ladruno-v20261103) reports it. The
check now asks for that name instead of a sha:

- **Deck and `model.h5` replay.** The guard at the deck's start runs on when the
  build has no `ladrunoBuild` (stock keeps the flag). A fork build passes only
  when `ladrunoSchema` answers schema version 1 with `mdStageFlagPreserved` in
  its `capabilities`: `regexp` in Tcl, `json.loads` in openseespy. A v0 schema,
  no `ladrunoSchema`, a failing call or another version stops the deck with the
  same message as before, which now names the capability.
- **Live.** `LiveOpsEmitter` keeps the same watch as the deck emitters and asks
  the engine it drives before `analyze`, `eigen` or `eigen -feast`. A fork whose
  schema is unknown (`schema is None`) is refused, never passed.

`model_hash` does not change. The sha table moves into the live reset test,
which still measures the reset on each build.
