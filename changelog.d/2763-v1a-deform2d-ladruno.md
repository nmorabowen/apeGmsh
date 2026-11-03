### CHANGED — apeGmshViewer's `deform_2d` fixture is a `.ladruno` (#2763, ADR 0131 D10 stage 0)

The viewer's 2-D deformed-shape fixture `apeGmshViewer/fixtures/deform_2d.ladruno`
replaces the native `deform_2d.results.h5`. This is the first batch of the
viewer fixture migration (V1a). The new file is written by `write_nodal_history`:
one static stage of 3 steps, with `ON_NODES/DISPLACEMENT` `Ux,Uy` holding the
same closed form as before. It is stamped against the committed `deform_2d.h5`,
so the app pairs the two by `model_hash`. The stamp gives `opensees_backend` as
`"unknown"`, and the file passes the fork's own `ladruno_format.validate()` at
`FORK_PIN`.

`scripts/make_viewer_results_fixtures.py` gains `--deform-ladruno`, which
rewrites the `.ladruno` alone against the model already there. `--deform`
still composes the model and now writes the `.ladruno` beside it. The
deformed-shape, overlay, offer and spec tests read the new file, and the
contour golden gains its row. The stage now reads as `MODEL_STAGE[1]`, not
`load`, because the model declares no `ops.stage`.
