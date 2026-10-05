# X1-d: `ModelData` inventory (K19)

Slice #1485, chain X (#1199). Documentation only: nothing is deleted.
Base: `origin/main` at 96244748. Deletion is a maintainer gate.

P0.3 decided K19 as "cut `ModelData`, **if no functionality is lost**". This
inventory tests that condition. It does not hold: five of the nine
capabilities have **NO REPLACEMENT** (see the table below).

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

| # | Capability (`model_data.py`) | Replacement | Verdict |
|---|---|---|---|
| 1 | `ModelData(fem, ndm=, ndf=, model_name=)` constructor, with `TypeError`/`ValueError` guards | `mesh/FEMData.py::FEMData.from_h5` supplies `fem`. Nothing replaces the `ndm`/`ndf` binding for a bridge-less user | partial |
| 2 | properties `fem`, `ndm`, `ndf` | `fem`: the object the caller already holds. `ndm`: `opensees/emitter/h5_reader.py::read_spatial_ndm`. `ndf`: the `/meta` attr | covered (trivial) |
| 3 | `oriented_elements(pg=, ele_type=, vecxz=)`: per-PG orientation inject, `fem_eid` resolved from the broker | `opensees/apesees.py` `geomTransf` with `vecxz=` writes the same zone, but only inside a full bridge model (materials, sections, elements). No orientation-only writer exists | **NO REPLACEMENT** |
| 4 | `recorders(...)`: declare recorders for a hand-written deck, same grammar as `ops.recorder.declare` | `opensees/_internal/ns/recorder.py::_RecorderNS.declare`, which needs an `apeSees(fem)` instance. The shared grammar is `opensees/recorder.py::build_recorder_declaration` | **NO REPLACEMENT** (bridge-less) |
| 5 | `attach_recorders(ops)`: issue `ops.recorder` calls into a live, already-populated session | `opensees/apesees.py::apeSees.run` wipes and rebuilds the domain (`_LiveRecorderSink` exists to avoid exactly that) | **NO REPLACEMENT** |
| 6 | `recorder_commands(target="py"\|"tcl")`: recorder-only script lines to paste into a hand deck | `apeSees.py` / `apeSees.tcl` emit the whole deck, not recorder lines | **NO REPLACEMENT** |
| 7 | `write(path)`: compose `model.h5` (neutral zone plus orientation zone) | `opensees/apesees.py::apeSees.h5` writes the same bytes for the orientation zone, from a full bridge model only | **NO REPLACEMENT** (bridge-less) |
| 8 | `from_h5(path)`: load an archive and enrich it (`snapshot_id` carried opaque, ADR 0018 INV-8) | Round trip: `opensees/opensees_model.py::OpenSeesModel.from_h5` then `to_h5`, which preserves staged archives. Enrichment (adding orientation to a loaded file): none | partial (round trip covered, enrich not) |
| 9 | `_LiveRecorderSink` (private helper for #5) | none needed once #5 is decided | n/a |

NO REPLACEMENT count: **5** (rows 3, 4, 5, 6, 7), plus 2 partial (rows 1, 8).

The rows split into two real jobs. (a) Orientation-only `model.h5` for a
hand-written deck (rows 3, 7, 8). (b) Recorder-only emission for a
hand-written deck (rows 4, 5, 6). Neither job has a bridge-free
replacement today.

## The silent-wrong modes

All are documented in the module docstring (`model_data.py:36-89`) and none is
detected at runtime.

1. **Tag correspondence** (the mode the panel cites). Orientation and recorder
   selectors resolve to **FEM element/node ids**, while the user's
   `ops.element`/`ops.node` tags are typed by hand. If they differ, results
   land on the wrong elements with no diagnostic. The safe pattern is
   manual: drive `ops.element` from `fem.elements`.
2. **Staged laundering.** `from_h5` reads only the orientation zone, so
   `write()` on a staged archive silently drops `/opensees/stages`. It emits a
   `UserWarning` (ADR 0055 Phase 2); the file is still written.
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
