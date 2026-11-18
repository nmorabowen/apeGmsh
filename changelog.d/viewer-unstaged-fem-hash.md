### FIXED — apeGmshViewer shows no activation banner for an unstaged model paired by `fem_hash` (#1754)

A model with no stages has no activation to bind, so results paired with it by
`fem_hash` now read as unstaged, as they do under a `model_hash` pairing.
