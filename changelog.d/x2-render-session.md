### CHANGED (behaviour) — `results.render`, `render_pack`, `assess(figures=True)` and the CLI `render` draw through `ResultsSession` (X2-d, #1665)

The four stills doors no longer build a `ResultsDirector`. Each `view=` token
becomes one session mesh view, drawn by `ResultsSession.render`. Signatures
are unchanged. The numeric oracle (`tests/viewers/test_x2_numeric_oracle.py`)
passes unchanged, so the data drawn is the same. The look follows the session,
so the image goldens in `tests/viewers/golden_stills/` were re-baselined from
the session path (maintainer ruling 2026-10-11).

Two requests are now refused with a `ValueError` that names the alternative
(maintainer ruling 2026-10-11):

- `view="reactions"` with `component="reaction_x"`, `"reaction_y"` or
  `"reaction_z"`: pass `component="reactions"` to get every axis.
- `deform=(field, scale)` with a field other than `displacement`, `velocity`
  or `acceleration`: pass `deform=SCALE` or one of those three fields.
