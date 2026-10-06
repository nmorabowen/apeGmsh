### TESTS — K19: `ModelData` stays; its two existing replacements are now proven (#1506)

The maintainer's K19 decision cuts a `ModelData` capability only where a
replacement exists and a test exercises it. Nothing met that bar in a way that
allows a cut, so `ModelData` and its `apeGmsh.opensees` export are unchanged.
`tests/opensees/h5/test_model_data_replacements.py` proves the two replacements
that do exist: `OpenSeesModel.from_h5(p).fem` / `.ndm` / `.ndf` report what
`ModelData.from_h5(p)` reports, and `OpenSeesModel.from_h5(p).to_h5(q)` writes
the same file as `ModelData.from_h5(p).write(q)` on a `ModelData` archive, while
keeping the bridge deck and `/opensees/stages` that `ModelData` drops. The
capabilities with no replacement (orientation-only writing, bridge-less
recorders, enrichment of a loaded archive) are listed under "Gaps (kept)" in
`internal_docs/program/x1_modeldata_inventory.md`.
