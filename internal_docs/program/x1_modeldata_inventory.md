# X1-d: `ModelData` inventory (K19)

Slice #1485, chain X (#1199). Documentation only: nothing is deleted.
Base: `origin/main` at 96244748. Deletion is a maintainer gate.

P0.3 decided K19 as "cut `ModelData`, **if no functionality is lost**". This
inventory tests that condition. It does not hold: five of the nine
capabilities have **NO REPLACEMENT** (see the table below).

**K19 outcome (X1-e, #1506): nothing is cut.** The maintainer's decision
(#1199, 2026-10-06) was to cut a row only where a replacement exists and a
test exercises it. Two rows have a tested replacement (2, and the round-trip
half of 8); neither can be removed without removing a class that the gap rows
keep. `ModelData` and its `apeGmsh.opensees` export stay. The gaps are listed
under [Gaps (kept)](#gaps-kept). X1-e also re-points the H5 rewrite oracles
onto the replacement where one exists, and turns silent-wrong mode 1 (tag
correspondence) into a runtime warning on the live path.

## What `ModelData` is

`src/apeGmsh/opensees/model_data.py::ModelData` (693 lines, ADR 0018) is a
façade for users who write their OpenSees model **by hand** in vanilla
openseespy and never build an `apeSees` bridge. It binds a `FEMData`,
collects per-element `vecxz` orientation and recorder declarations, and
writes a `model.h5` whose `/opensees/transforms` + `/opensees/element_meta`
zone the results viewer joins to orient beams. It is exported from
`apeGmsh.opensees` (`__init__.py`, `__all__`), so it is public API.

## Capability to replacement table

"Replacement" means a way to get the same result **without** `ModelData` and
**without** adopting the full `apeSees` bridge, which is the audience's
defining constraint (ADR 0018 Context).

The last two columns are the K19 decision applied in X1-e (#1506). `T` is
`tests/opensees/h5/test_model_data_replacements.py`.

| # | Capability (`model_data.py`) | Replacement | Verdict (X1-d) | K19 verdict (X1-e) | Replacement test |
|---|---|---|---|---|---|
| 1 | `ModelData(fem, ndm=, ndf=, model_name=)` constructor, with `TypeError`/`ValueError` guards | `mesh/FEMData.py::FEMData.from_h5` supplies `fem`. Nothing replaces the `ndm`/`ndf` binding for a bridge-less user | partial | **kept** (G1) | none: the binding has no replacement |
| 2 | properties `fem`, `ndm`, `ndf` | `fem`: the object the caller already holds. `ndm`: `opensees/emitter/h5_reader.py::read_spatial_ndm`. `ndf`: the `/meta` attr. On an archive, `OpenSeesModel.from_h5(p).fem` / `.ndm` / `.ndf` give all three | covered (trivial) | **kept** (G2: replacement proven, not cut) | `T::test_opensees_model_reports_the_accessors_model_data_reports` |
| 3 | `oriented_elements(pg=, ele_type=, vecxz=)`: per-PG orientation inject, `fem_eid` resolved from the broker | `opensees/apesees.py` `geomTransf` with `vecxz=` writes the same zone, but only inside a full bridge model (materials, sections, elements). No orientation-only writer exists | **NO REPLACEMENT** | **kept** (G3) | none |
| 4 | `recorders(...)`: declare recorders for a hand-written deck, same grammar as `ops.recorder.declare` | `opensees/_internal/ns/recorder.py::_RecorderNS.declare`, which needs an `apeSees(fem)` instance. The shared grammar is `opensees/recorder.py::build_recorder_declaration` | **NO REPLACEMENT** (bridge-less) | **kept** (G4) | none |
| 5 | `attach_recorders(ops)`: issue `ops.recorder` calls into a live, already-populated session | `opensees/apesees.py::apeSees.run` wipes and rebuilds the domain (`_LiveRecorderSink` exists to avoid exactly that) | **NO REPLACEMENT** | **kept** (G4) | none |
| 6 | `recorder_commands(target="py"\|"tcl")`: recorder-only script lines to paste into a hand deck | `apeSees.py` / `apeSees.tcl` emit the whole deck, not recorder lines | **NO REPLACEMENT** | **kept** (G4) | none |
| 7 | `write(path)`: compose `model.h5` (neutral zone plus orientation zone) | `opensees/apesees.py::apeSees.h5` writes the same bytes for the orientation zone, from a full bridge model only | **NO REPLACEMENT** (bridge-less) | **kept** (G3) | none |
| 8 | `from_h5(path)`: load an archive and enrich it (`snapshot_id` carried opaque, ADR 0018 INV-8) | Round trip: `opensees/opensees_model.py::OpenSeesModel.from_h5` then `to_h5`, which preserves staged archives. Enrichment (adding orientation to a loaded file): none | partial (round trip covered, enrich not) | **kept** (G5) | round trip only: `T::test_round_trip_matches_model_data_on_a_model_data_file`, `T::test_round_trip_keeps_the_bridge_deck_model_data_drops`, `T::test_round_trip_keeps_a_staged_archive_without_warning` |
| 9 | `_LiveRecorderSink` (private helper for #5) | none needed once #5 is decided | n/a | **kept** with row 5 (G4) | n/a |

NO REPLACEMENT count: **5** (rows 3, 4, 5, 6, 7), plus 2 partial (rows 1, 8).

The rows split into two real jobs. (a) Orientation-only `model.h5` for a
hand-written deck (rows 3, 7, 8). (b) Recorder-only emission for a
hand-written deck (rows 4, 5, 6). Neither job has a bridge-free
replacement today.

## Gaps (kept)

Every row is kept. A row is listed here with the reason it stays; porting
any of them is a separate maintainer decision (no replacement API was built
in X1-e).

| Gap | Rows | Why it is kept | What a cut would need |
|---|---|---|---|
| G1 | 1 | The `ndm`/`ndf`/`model_name` binding for a bridge-less writer has no replacement; `FEMData.from_h5` supplies only `fem` | a bridge-less writer that takes the binding (G3) |
| G2 | 2 | A replacement exists and is proven (`OpenSeesModel.from_h5(p).fem` / `.ndm` / `.ndf`, same values as `ModelData.from_h5(p)`). It is not cut because the accessors belong to a class that stays: removing them breaks `md.ndm` callers and gains nothing, and the cut would have to edit `tests/opensees/h5/test_h5_meta_ndm_raw_readers.py:71` (it pins `ModelData.from_h5(p).ndm` through the pre-2.34.0 shim), which X1-e does not own. They go when the class goes | nothing beyond the class cut |
| G3 | 3, 7 | No orientation-only `model.h5` writer exists outside a full `apeSees` model (materials, sections, elements) | an orientation writer on `OpenSeesModel` or the H5 layer |
| G4 | 4, 5, 6, 9 | No bridge-less recorder emission: `_RecorderNS.declare` needs an `apeSees(fem)`, and `apeSees.run` rebuilds the domain instead of attaching to a live one | recorder-only emission on `apeSees`/`recorder`, or the capability recorded as dropped |
| G5 | 8 | The round trip is replaced and proven, staged archives included (the replacement keeps `/opensees/stages`; `ModelData` warns and would drop it). Enrichment, `from_h5` then `oriented_elements` then `write`, has no replacement, and both halves live in one method | G3, so enrichment can move with it |

The five test files that use `ModelData` as a tool, and where each oracle
now runs:

| File | Oracle | Runs on |
|---|---|---|
| `tests/opensees/unit/test_node_pair_zerolength.py` | H5-to-H5 rewrite is byte-stable and keeps node-pair `inline_connectivity` (ADR 0049) | **re-pointed**: parametrized over `OpenSeesModel.from_h5 -> to_h5` (which also keeps the source `model_hash`, `a == b`) and `ModelData.from_h5 -> write` (kept) |
| `tests/opensees/h5/test_h5_meta_ndm.py` | a `ModelData(ndm=2)` file reads back `ndm == 2` | **re-pointed** read-back: `OpenSeesModel.from_h5(p).ndm` and `ModelData.from_h5(p).ndm`. The write is `ModelData`'s (G3) |
| `tests/opensees/h5/test_bridge_provenance.py` | replay carries `/provenance` byte-equal | already parametrized over both rewriters; unchanged |
| `tests/opensees/h5/test_h5_stages_reader.py` | `ModelData.from_h5` warns on a staged archive | unchanged: the warning is `ModelData`'s own (G5); the replacement's staged round trip is pinned in `T` |
| `tests/viewers/test_viewer_orientation_from_model_h5.py` | viewer orients beams from an orientation-only file | unchanged: the fixture writes an orientation-only file (G3), which has no replacement |

Both rewriters stay on the re-pointed oracles because `ModelData` stays: dropping
its branch would remove coverage of a kept write path.

## The silent-wrong modes

All are documented in the module docstring (`model_data.py:36-89`). X1-d found
none detected at runtime; X1-e adds the check in mode 1 for the live path.

1. **Tag correspondence** (the mode the panel cites). Orientation and recorder
   selectors resolve to **FEM element/node ids**, while the user's
   `ops.element`/`ops.node` tags are typed by hand. If they differ, results
   land on the wrong elements. The safe pattern is manual: drive
   `ops.element` from `fem.elements`. **X1-e (#1506):**
   `attach_recorders(ops)` now checks every recorder target against the live
   domain before issuing any `recorder` call (read-only `getNodeTags`,
   `nodeCoord`, `getEleTags`, `eleNodes`): a node absent or at other
   coordinates, or an element absent or joining other nodes (whose nodes are
   then checked too), raises one `UserWarning` naming the ids. The recorders
   are still attached. `recorder_commands` and `write` have no domain to
   check, so they keep the banner and the docstring caveat. Pinned in
   `tests/opensees/h5/test_model_data_recorders.py` (five cases fail with the
   check removed).
2. **Staged laundering.** `from_h5` reads only the orientation zone, so
   `write()` on a staged archive silently drops `/opensees/stages`. It emits a
   `UserWarning` (ADR 0055 Phase 2); the file is still written.
   X1-e found the same laundering on an **unstaged** `apeSees` archive, with
   no warning: `ModelData.from_h5(p).write(q)` drops `/opensees/materials`,
   `sections` and `beam_integration` (and `/meta/lineage` changes). The
   warning fires only for `/opensees/stages`. Pinned as the INV-5 contrast in
   `T::test_round_trip_keeps_the_bridge_deck_model_data_drops`; widening the
   warning is a follow-up, not in X1-e's scope.
3. **Orientation-only scope** (ADR 0018 INV-5). No materials, sections,
   patterns or constraints are written, so a `ModelData` file is not a
   solvable model. The viewer needs no `ModelData` awareness; the zone is
   byte-equivalent to the bridge's.

## Callers

`python scripts/nav.py refs ModelData` finds **no class use outside
`model_data.py` and the package re-export**. Every other hit is a docstring
or comment. Full output is in the appendix.

| Class | Where | Notes |
|---|---|---|
| user-facing (public API) | `opensees/__init__.py:34,42` | import plus `__all__`: the only public surface. Also ADR 0018 and its README row |
| user-facing docs | none | `docs/` and `skills/apegmsh/` contain no `ModelData` page (the panel's "0 docs"). The `examples/` hit is `gmsh.view.addModelData`, a false positive; so is `mesh/View.py` |
| internal plumbing | `opensees/_internal/compose.py` (`_compose_model_h5`, shared with `apeSees.h5`) | the composer stays either way; only a docstring names `ModelData` |
| internal plumbing | `emitter/h5.py::add_oriented_elements`, `_internal/build.py::expand_pg_to_elements` and `_emit_recorder_declaration`, `recorder.py::build_recorder_declaration` | shared with the bridge, so they survive a cut. `add_oriented_elements` is only called by `ModelData` and would become dead |
| docstring or comment only | `opensees_model.py`, `emitter/h5_reader.py`, `_internal/typed_records.py`, `viewers/data/_h5_probe.py`, `viewers/results_viewer.py:912`, `results/capture/_domain.py:683`, `recorder.py:94,1486`, `apesees.py:10399`, `scripts/check_quirks.py` | text edits on a cut. `_domain.py:683` is a user-facing error message that names `ModelData.write` |
| test-only, dedicated | `tests/opensees/h5/test_model_data.py` (19 hits), `test_model_data_from_h5.py` (14), `test_model_data_recorders.py` (14), `test_model_data_ast_guard.py` (8) | they die with the class |
| test-only, **used as a tool** | `tests/opensees/unit/test_node_pair_zerolength.py:27,321-322`, `tests/opensees/h5/test_bridge_provenance.py:44,222`, `tests/opensees/h5/test_h5_meta_ndm.py:92-99`, `tests/opensees/h5/test_h5_stages_reader.py:351` | other contracts use `ModelData.from_h5(...).write(...)` as the H5-to-H5 rewrite oracle (byte stability, provenance, ndm, laundering warning). A cut must re-point these, not just delete them |
| test-only, viewer contract | `tests/viewers/test_viewer_orientation_from_model_h5.py:99,108` | builds the oriented fixture through `ModelData`; it would need an `apeSees`-built fixture |
| test-only, name only | `tests/test_check_quirks.py`, `tests/assess/test_import_guard.py`, `tests/opensees/integration/test_stress_zz_promotion.py` | mention the name in text |

Totals: 1 user-facing symbol (the export), 0 user-facing docs, 1 internal
plumbing seam that stays (the shared composer), 8 test files that exercise or
use the class, about 10 docstring-only mentions.

## Recommendation

**Do not cut as is.** The P0.3 condition ("if no functionality is lost") is
not met: two jobs (bridge-less orientation `model.h5`; bridge-less recorder
emission) have no replacement, and ADR 0018 exists to serve exactly that
audience. The panel's evidence is real but partial: 0 docs and a
silent-wrong mode, yet the class is public and four other test files use it
as the H5-to-H5 oracle.

Two defensible paths for the maintainer:

- **Keep** it as the supported bridge-less path, and close the gaps that
  make it risky: a docs page (the "0 docs" finding) and a stronger warning
  for the tag-correspondence mode.
- **Cut after porting 3 items**, then delete:
  1. recorder-only emission (rows 4-6): expose it on `apeSees`/`recorder`
     without building a model, or record the capability as dropped;
  2. orientation-only `model.h5` (rows 3, 7): a writer on `OpenSeesModel`
     (or the H5 layer) that adds orientation to a loaded archive;
  3. re-point the four "used as a tool" files and the viewer fixture onto
     `OpenSeesModel.from_h5`/`to_h5` and an `apeSees`-built archive.
  Then delete `model_data.py`, its 4 dedicated test files and the export;
  mark ADR 0018 superseded (ADRs are append-only); delete
  `add_oriented_elements` if no other caller remains.

Order: decide first whether the bridge-less audience is still wanted. If yes,
keep. If not, port items 1-3 and then cut; do not cut first.

## Appendix: `nav.py` output

`python scripts/nav.py refs ModelData`:

```
ModelData: 4 refs in 2 files, by enclosing scope (comments/docstrings excluded)
  opensees/model_data.py
      <module>  L91
      ModelData  L438
  opensees/__init__.py
      <module>  L34,42
```

`python scripts/nav.py refs model_data`:

```
no code references to 'model_data'
```

Prose and test callers come from `git ls-files` piped to
`grep -lE "ModelData|model_data"`, because nav ignores docstrings, comments
and non-Python files.
