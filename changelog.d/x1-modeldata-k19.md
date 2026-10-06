### CHANGED — K19: `ModelData` stays; `attach_recorders` now warns on a tag mismatch, and its replacements are proven (#1506)

The maintainer's K19 decision cuts a `ModelData` capability only where a
replacement exists and a test exercises it. Nothing met that bar in a way that
allows a cut, so `ModelData` and its `apeGmsh.opensees` export stay.

`ModelData.attach_recorders(ops)` now checks every recorder target against the
live OpenSees domain before issuing any `recorder` call. A targeted node that is
absent or sits at other coordinates than the fem node of the same id, or a
targeted element that is absent or joins other nodes, raises one `UserWarning`
naming the ids, because those recorders would observe the wrong entities
(`tag == fem_eid` does not hold). The recorders are still attached. Coordinates
match within 1e-3 of the shortest element edge, and in a parallel run
(`ops.getNP() > 1`) an id absent from this rank is not reported.

`tests/opensees/h5/test_model_data_replacements.py` proves the two replacements
that do exist: `OpenSeesModel.from_h5(p).fem` / `.ndm` / `.ndf` report what
`ModelData.from_h5(p)` reports, and `OpenSeesModel.from_h5(p).to_h5(q)` writes
the same file as `ModelData.from_h5(p).write(q)` on a `ModelData` archive, while
keeping the bridge deck and `/opensees/stages` that `ModelData` drops. The
node-pair byte-stability and `ndm` read-back oracles now also run on
`OpenSeesModel`. The capabilities with no replacement (orientation-only
writing, bridge-less recorders, enrichment of a loaded archive) are listed
under "Gaps (kept)" in `internal_docs/program/x1_modeldata_inventory.md`.
