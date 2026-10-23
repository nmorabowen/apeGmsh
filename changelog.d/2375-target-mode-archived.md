### FIXED — a pinned `OpenSeesTarget` survives `model.h5`: the replay deck stamps its backend and a fork pin stays a fork requirement (opensees 2.36.0, #2278)

A model built with `OpenSeesTarget(mode="fork")` or `mode="stock"` wrote
`# apeGmsh <v>; backend <mode>` in its deck, but `model.h5` dropped the
mode. A replay (`OpenSeesModel.from_h5(path).build(...)`) wrote an
unstamped deck. A fork-pinned archive with no fork-only verb also replayed
on any backend: `build('live')` did not assert the fork, and the
stage-close route checks did not apply the #2444 fork-reset refusal that
the bridge applies under `mode="fork"`.

- The writer stores the pinned mode as `/opensees@target_mode` (`"fork"`
  or `"stock"`). An unpinned model writes nothing, and an absent attribute
  reads as `"auto"`, so every older archive keeps its meaning. `binary`
  and `python` are machine paths and are never archived.
- A pinned archive's replay deck equals the bridge's deck byte for byte,
  stamp line included. A `"fork"` archive adds `"fork"` to
  `OpenSeesModel.replay_requires()`, so `build('live')` refuses a stock
  openseespy and names the pin, and the replay's stage-route checks apply
  the #2444 refusal. `to_h5` echoes the mode.
- The reader (`H5Model.target_mode()`, `OpenSeesModel.target_mode`)
  refuses a value that is not `"auto"`, `"fork"` or `"stock"`, the empty
  string included, with `MalformedH5Error` naming the attribute.
- The attribute is not in `model_hash`: the same model pinned or on auto
  hashes equal (maintainer ruling A on #2278).
- Opensees schema 2.36.0 (additive). A 2.35 reader would replay a fork pin
  as auto, so it now refuses a 2.36 file. An unpinned model's archive
  keeps its bytes apart from the version stamp.
