### CHANGED — the opensees compatibility floor is 2.12.0 (ADR 0113 D3, #1303)

`SCHEMA_FLOOR` in `opensees/emitter/h5.py` rises from 2.11.0 to 2.12.0, and
with it `tests/fixtures/schema.py` `OPENSEES_FLOOR` and the app's
`ZONE_FLOOR.opensees` in `apeGmshViewer/src/reader/read.ts`. This is the
evidence gate of ADR 0113 D3, not a major bump: the schema corpus (#1329)
showed that every opensees-2.11 writer stamped neutral 2.7.0, below the
neutral floor 2.10.0, so no file of that era opened through today's readers
anyway. A 2.11.x `/opensees` zone now refuses as "too old: this reader
supports 2.12.x–2.22.x"; 2.12.0 opens. The maintainer approved the raise on
2026-10-04 (#1303).

The corpus keeps the 2.11 era's real file as a `below_floor` entry: INV-6
accepts entries below a floor only in that form, and a test opens the file
through each zone's own reader and requires the refusal to name the floor.
Two shim-ledger variants join the corpus, each from the last frozen writer
before its semantic change: `neutral_2.25_sp_cases` (prescribed
displacements under two `g.displacements.case` names, written before 2.26.1
split `/loads/sp/default` per case, read as one `default` case) and
`opensees_2.20_frame2d` (a frame declared `ops.model(ndm=2, ndf=3)` before
neutral 2.34.0, whose `/meta/ndm` stamp `read_spatial_ndm` salvages; the
declared 2 is not recoverable from such a file). The semantic dump
(`_semantic_dump.py`, format 2) now records `/mesh_selections`, the per-node
`ndf` stream and the writer's raw `/meta` ndm/ndf stamps, and the whole
corpus was rebuilt by `scripts/build_schema_corpus.py` with the same era
commits as #1329. The builder gained `--variant` and keeps below-floor
evidence eras in its plan, so raising a floor never drops the files that
justified it.
