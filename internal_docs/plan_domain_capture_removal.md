# Plan: remove DomainCapture, step 1 (the parity table)

This page maps every use of `DomainCapture` on `origin/main` to the route that
replaces it, says which result kinds each route must carry and whether a passing
test proves it, and lists every capability no route covers yet. It decides
nothing: the gaps in §5 go to the maintainer before step 2 starts.

- **Decision:** remove DomainCapture (maintainer, #1767, YES 2026-10-14).
- **Chain:** X (#1202). This is step 1, slice #1942.
- **Base:** `origin/main` at `dfc9bdd3`. The grep was re-run on `2dfe0390`
  and gives the same 204 / 180 files.
- **Steps:** 1 is this table. 2 migrates the docs, notebooks, examples and the
  skill. 3 is the human-gated deletion PR. Each later step gets its own card.
- **Status:** Proposed. Nothing here is approved until the maintainer answers §5.

## 1. What DomainCapture does

`src/apeGmsh/results/capture/_domain.py::DomainCapture` (2,673 lines) is a
context manager. The user drives the analysis loop and calls `step(t)` after each
`analyze`. The capture queries the live domain (`nodeDisp`, `nodeReaction`,
`eleResponse`), buffers in RAM, and writes a native results HDF5 at each
`end_stage()`. It is opened two ways: `ops.domain_capture(spec, path=...)` on the
bridge (`apesees.py`), or `DomainCapture.from_h5(model.h5, spec=, fem=, output=)`
without one. The spec is `DomainCaptureSpec` (`capture/spec.py`), with seven
categories. These are the capabilities a replacement has to match, numbered for
the tables below:

| # | Capability | Where it lives |
|---|---|---|
| C1 | Nodal kinematics, reactions and pore pressure | `_NodesCapturer` |
| C2 | Per-element-node forces, `globalForce` / `localForce` | `_NodalForcesCapturer` |
| C3 | Continuum Gauss stress and strain, from the response catalog | `_GaussCapturer` |
| C4 | Material-only and per-material responses read per Gauss point: `damage`, and Tri31 `strain` | `needs_per_material_query` |
| C5 | Beam-column line stations from `section force`, plus a two-station slab synthesised from `localForce` for elastic beams | `_LineStationCapturer` |
| C6 | Fibre sections: geometry once, then stress and strain each step | `_FiberCapturer` |
| C7 | Layered shells, with per-layer thickness and material read from the bridge and per-element quaternions from `_shell_geometry` | `_LayerCapturer` |
| C8 | Modes: `capture_modes()` runs `eigen` and writes one `kind="mode"` stage per mode, with eigenvalue, frequency and period | `capture_modes` |
| C9 | Many named stages with a kind (`static`, `transient`, `mode`) in one file, read as one `Results` | `begin_stage` / `end_stage` |
| C10 | Snapshots at exactly the instants the loop calls `step(t)`, with the caller's `t` | `step` |
| C11 | One native file, no intermediate files and no transcode; the tight-loop use in `guide_sensitivity.md` | the writer |
| C12 | Provenance root attributes `opensees_backend` / `opensees_build`, plus the `element_index_space = "fem"` stamp (#1764) | `__enter__` |
| C13 | Queries the module the bridge drives (fork or stock), not a second `openseespy` | `_lazy_ops` |
| C14 | Fem-to-ops element tag translation, refusing an element record the pairing does not cover (#1764) | `_require_element_pairing` |

## 2. The replacements

The card names three readers. Each sits behind a write route.

- **`.out` recorders → `Results.from_recorders(spec, dir, fem=, model=, stage_id=)`.**
  It is written live by `ResolvedRecorderSpec.emit_recorders(...)` (`LiveRecorders`),
  or by a deck from `to_tcl_commands` / the bridge's `ops.recorder.declare`. It
  transcodes nodes, element forces, Gauss points and line stations into a cached
  native file. It skips fibres and layers, and it raises on modal records.
- **MPCO → `Results.from_mpco(path, model_h5=)`.** It is written live by
  `ResolvedRecorderSpec.emit_mpco(...)` (`LiveMPCO`), or by a deck. The reader
  covers nodes, Gauss points, line stations (and `localForce` two-station slabs),
  fibres, layers, springs and material state. It reads every stage as
  `kind="transient"` and has no mode-shape reader. The emitter writes
  `modesOfVibration`, but nothing reads it back.
- **`Results.from_native(path, model=)`.** It reads any native file. Besides
  DomainCapture, its writers today are the recorder transcoder,
  `ParallelModalResult.to_native` (ADR 0077, modes from a parallel modal deck),
  `footfall_result` and `results/demo.py`.

`Results.from_ladruno` is a fourth reader, fork-only. The card does not name it,
but it already covers C4 (`damage`) and modes (`eigen` → `kind="mode"`) on the
fork. It is listed where it is the only route.

## 3. Parity by result kind

"Proven" means a test reads the kind back through the replacement and checks the
values against an oracle: the live domain (`nodeDisp` / `eleResponse`, the same
calls DomainCapture makes) or the closed-form numbers its DomainCapture twin
asserts. Each test below was run on this branch with
`C:\Users\nmb\venv\opensees_venv\Scripts\python.exe -m pytest <test> -q`. Tests
marked **(Tcl)** skip on CI and in a plain local run because no Tcl launcher is
found. They passed here only with `APEGMSH_OPENSEES_TCL` set to
`C:\Program Files\Ladruno\OpenSees\bin\OpenSees.exe`.

| Kind | `.out` → `from_recorders` | MPCO → `from_mpco` | `from_native` (other writer) | Status |
|---|---|---|---|---|
| Nodal (C1) | `tests/test_results_recorder_tag_parity.py::test_node_displacement_rows_are_the_labelled_nodes` (live domain) | `tests/test_results_mpco_emit_real.py::test_openseespy_full_cycle`; `::test_tcl_subprocess_full_cycle` (Tcl) | n/a | **Proven** on both |
| Element nodal forces (C2) | `tests/test_results_recorder_tag_parity.py::test_elements_global_force_rows_are_the_labelled_elements` (live domain); `tests/test_results_recorder_transcoder_nodal_forces_real.py::test_elastic_beam_3d_global_and_local_three_way_agreement` (Tcl) | `tests/test_results_mpco_nodal_real.py::test_elastic_beam_3d_global_and_local_force` (Tcl) **fails** at line 158: the element-id filter returns `[]` | n/a | **Proven on `.out`**; MPCO unproven (G9) |
| Gauss, continuum (C3) | `tests/test_results_recorder_tag_parity.py::test_gauss_stress_rows_are_the_labelled_elements` (live domain); `tests/test_results_recorder_transcoder_gauss_real.py::test_tet_stress_full_cycle_through_transcoder` (Tcl) | `tests/test_results_mpco_element_real.py::test_tet_stress_full_cycle` (Tcl) | n/a | **Proven for a tet**; other families unproven (G7) |
| Plane-strain `stress_zz` (C3) | `tests/opensees/integration_ladruno/test_stress_zz_plane_strain_live.py::test_out_recorder_round_trips_the_promoted_response` (fork-only) | n/a | n/a | **Proven on the fork only** |
| Per-material, `damage` / Tri31 strain (C4) | none: spec emit skips Tri31 strain and has no `damage` token | reader reads `material.damage` when the user hand-writes the token; spec `emit_mpco` maps no `damage` | `from_ladruno`: `tests/opensees/integration_ladruno/test_concrete_damage_capture_live.py::test_ladruno_brick_tension_damage_capture_and_ladruno_read` (fork-only) | **Gap** on stock (G5) |
| Line stations, `section force` (C5) | the recorded file only: `tests/test_results_recorder_tag_parity.py::test_line_station_section_force_file_holds_the_labelled_elements`; transcode on a minimal FEM: `tests/test_results_recorder_transcoder_line_stations_real.py::test_force_beam_3d_lobatto5_aggregated_three_way_agreement` (Tcl); on a real FEMData the transcode crashes (#1741, open) | `tests/test_results_mpco_line_real.py::test_force_beam_3d_aggregated_section`, `::test_force_beam_3d_bare_section` (Tcl) | n/a | **Proven on MPCO**; `.out` blocked by #1741 (G3) |
| Line stations, elastic beam (C5) | none: the transcoder has no `localForce` synthesis | `tests/test_results_mpco_local_force.py::test_results_read_line_stations_returns_two_station_slab` | n/a | **Proven on MPCO**; gap on `.out` (G3) |
| Fibres (C6) | none, by design | emit only: `tests/test_results_live_mpco.py::test_emit_mpco_handles_fibers_and_layers`; the value tests (`test_results_mpco_fiber_real*.py`) skip without the STKO fixtures | n/a | **Unproven** (G4) |
| Layers (C7) | none, by design | emit only (the same test); `test_results_mpco_layer_real.py` skips without the STKO fixtures | n/a | **Unproven** (G4) |
| Modes (C8) | raises (`LiveRecorders`), refused (`RecorderDeclaration` emit) | emitted, never read | `ParallelModalResult.to_native`: `tests/test_parallel_modal_to_native.py::test_to_native_round_trips_through_results_modes` (a synthetic harvest; needs an ADR 0077 parallel deck) | **Gap** for a stock single-process run (G1) |

The DomainCapture side of each comparison passes too:
- `tests/test_results_domain_capture_real.py::test_single_tet_static_capture` and `::test_single_tet_gauss_capture`;
- `tests/test_results_domain_capture_nodal_forces_real.py::test_elastic_beam_3d_global_and_local`;
- `tests/test_results_domain_capture_line_stations_real.py::test_force_beam_3d_lobatto5_aggregated`;
- `tests/test_results_domain_capture_fibers_real.py` (3 tests);
- `tests/test_results_domain_capture_layers_real.py::test_layered_shell_capture_round_trip`.

C13 (the bridge's module) is covered on both live routes:
`tests/test_results_live_recorders.py::test_defaults_to_the_module_the_bridge_drives`
and `tests/test_results_live_mpco.py::test_defaults_to_the_module_the_bridge_drives`
pass. C14 (tag translation) is covered on `.out` by the four
`test_results_recorder_tag_parity.py` tests above. MPCO is covered by
`ElementTagTranslator` in `from_mpco`, but its one end-to-end check is the
failing nodal-forces test (G9).

## 4. The table: every use, grouped by kind

`git grep -il "DomainCapture\|domain_capture" origin/main` lists **204** files.
Of those, **24** are binary HDF5 viewer fixtures. **180** are text, and
`git grep -il -I ...` lists exactly those 180. The 180 are the table: Appendix A
has one row per file, and the counts below sum to 180. The 24 binaries are left
out on purpose (Appendix B). They hold the string only as the `source_type`
attribute value, written by `scripts/make_viewer_results_fixtures.py` through
`NativeWriter`, not by DomainCapture. They follow G12.

The step column uses sub-steps of step 2 so each can be its own card: **2a** for
docs, **2b** for examples, **2c** for the skill, **2d** for scripts and
`viewer_bench`. Step 3 is the deletion PR, and "3 (note)" means a superseded
note added in step 3. "leave" means history that stays as written. "-" means no
change unless G12 changes the `source_type` value.

| Group (kind) | Files | How they use it | Replacement and kinds it must carry | Parity | Step |
|---|---|---|---|---|---|
| Public API and source, `src/apeGmsh` | 18 | The class, the spec and the package (3); `ops.domain_capture` on `apesees.py`. Five modules whose refusals or docstrings send users to DomainCapture. Three catalog helpers and `_shell_geometry.py` that only the capture calls. The `source_type` constant and the writer default. Comments in four modules, and the generated `_api_index.json` | Delete the class, the spec and the method. Retarget each refusal to `.out` or MPCO per kind; modal has no target (G1, G2). Remove the orphaned helpers | §3 per kind | 3 |
| Docs site, `docs/` | 19 | 12 pages (tutorials, examples, a concept page and a guide) record nodes with `ops.domain_capture` and `begin_stage`. `modal-analysis.md` is built on `capture_modes`. `choose-results-strategy.md` names capture as the default. Two asides, `design/results.md`, `results-mpco.md`, and the mkdocstrings entry in `api/results.md` | Nodal (C1), multi-stage (C9) → `.out` or MPCO. Modes (C8) → none (G1). The `api/results.md` entry goes with the class | Nodal proven; modes gap | 2a (18), 3 (1) |
| Examples and notebooks, `examples/` | 33 | 19 curriculum notebooks use `ops.domain_capture`: 18 record nodes, and `17_modal_analysis` records modes only. 12 case-study and archive notebooks call `DomainCapture(spec, path, fem, ndm=, ndf=)` after `from apeGmsh.results.spec import Recorders`; both are gone on main (G13). Two READMEs | Nodal (C1), line stations (C5) in `chevron_brace_retrofit_sine`, modes (C8) | Nodal proven; line stations MPCO only; modes gap | 2b |
| Skill | 4 | `opensees-bridge.md` names `DomainCaptureSpec` as the in-process route. `ladruno.md` names the unadopted live seam (ADR 0108 D5). Two derived mirrors | Edit `skills/apegmsh/` only, then run `sync_skill.py` | follows docs | 2c |
| Tests, `tests/opensees` | 12 | Four drive the class (staged h5, layers live, truss axial live, half of `stress_zz` promotion). Four use it as the vehicle for another property (`/meta/ndm` ×2, the shell-on-solid round-trip, provenance). Three fork-only. One asserts the refusal text | Delete what tests the class. Port what uses it as a vehicle (G7, G10). The refusal-text test follows G2 | per file, Appendix A | 3 |
| Tests, `tests/viewers` | 4 | Three build a staged composed results file through `ops.domain_capture` with `_FakeOps` imported from `tests/test_results_domain_capture.py`. One writes the `source_type` value only | Write the file with `NativeWriter` plus `model_h5_src=`; move `_FakeOps` | n/a (fixture) | 3 (3), - (1) |
| Tests, other | 46 | 22 test the class: the 14 `test_results_domain_capture*.py`, the four `test_results_catalog_*_real.py`, `test_results_shell_resultant_vocabulary.py`, `test_domain_capture_ops_binding.py` and two in `tests/results/`. One is their `tag_map` helper, `tests/fixtures/capture_tags.py`. 12 only write the `source_type` value through `NativeWriter.open` (`conftest.py` only names it in a docstring). 11 name the class in prose or asserted symbols, borrow its test helpers, or use it as one writer among several | Delete with the class, after G6 and G7 say what must move to a replacement-route test. `test_parallel_modal_to_native.py` imports `_MockFem` from the class's test module | per file | 3 (34), - (12) |
| Web viewer tests, `apeGmshViewer/test` | 3 | Assert `sourceType == "domain_capture"` on fixtures | None | n/a | - |
| Scripts | 7 | Four showcase renderers use `ops.domain_capture`: three for nodes or velocity, and `modal_sweep.py` for modes. `wire_eos_notebook.py` generates the stale notebook cell (G13). The fixture generator writes the `source_type` value. `check_quirks.py` cites the capture lesson | Nodal → `.out` or MPCO; modes → none (G1) | Nodal proven; modes gap | 2d (5), 3 (2) |
| `viewer_bench` | 2 | `ssi_frame_wall/src/build.py` captures nodes, Gauss, line stations and modes; the README tells the `capture_modes` story | Nodal, Gauss → `.out`; line stations → MPCO (#1741 on `.out`); modes → none (G1) | partial | 2d |
| Living architecture docs | 3 | `api-design.md`, `h5-schema.md` (the `source_type` values, the `capture_modes` layout), `parallel-execution.md` | Update with the code they describe | n/a | 3 |
| `internal_docs` | 11 | `Results_architecture.md` and two guides are living (`guide_sensitivity.md` recommends capture for tight loops, G15). Eight are plans, readability samples and study reports | Reword the 3 living files; leave the 8 history files | n/a | 3 (3), leave (8) |
| ADRs | 7 | 0046, 0055, 0077, 0093, 0098, 0108, 0121 | History: leave, add a superseded note in step 3. ADR 0108 D5 names DomainCapture as the future live seam (G14) | n/a | 3 (note) |
| CHANGELOG and fragments | 9 | `CHANGELOG.md` and 8 `changelog.d/` fragments | History: leave. Fragments are assembled, never edited | n/a | leave |
| Web viewer docs, `apeGmshViewer/` | 2 | `MEASUREMENTS.md`, `fixtures/README.md` describe the fixtures' `source_type` | None | n/a | - |
| **Total** | **180** | | | | |

The card's group estimates match: examples 33, `src` 18, `tests/opensees` 12,
ADRs 7. The docs group is 19, not "about 17". The other two hits are asides in
`fiber-moment-curvature.md` and `step-plate-with-hole.md`.

## 5. Gaps for the maintainer

Each gap is a capability DomainCapture has that no replacement covers, or covers
without a passing test. None is decided here. Each names the question and what
it blocks.

**G1: Modes on a stock, single-process run.** `capture_modes()` (C8) has no
replacement. `.out` refuses modal records: `LiveRecorders` raises, and the
bridge's `RecorderDeclaration` emit raises `NotImplementedError` "use
DomainCapture instead".
MPCO emits `modesOfVibration`, but `MPCOReader` labels every stage `transient`
and reads no mode shapes. The only native writer of mode stages besides the
capture is `ParallelModalResult.to_native`, which needs an ADR 0077 parallel
modal deck. On the fork, `from_ladruno` maps `eigen` stages to `mode`. *Question:*
which route carries modes after the deletion? *Blocks:*
- `docs/examples/modal-analysis.md`;
- the modal parts of `choose-results-strategy.md` and `design/results.md`;
- `17_modal_analysis.ipynb`;
- `scripts/render_showcase/modal_sweep.py` (a showcase animation);
- `viewer_bench/models/ssi_frame_wall/src/build.py`;
- the refusal text in G2.

**G2: Refusals that send users to DomainCapture.** Five `src` sites tell users
to use DomainCapture for what their own route cannot do:
- `_internal/build/recorders.py:137` (fibres, layers, modal);
- `live/_recorders.py:177` and `:248`;
- `spec/_emit.py:186` (Tri31 strain), plus the deck comment its Tcl and Python
  emitters write for a deferred category ("Use domain capture (Phase 7)");
- the `spec/_resolved.py:290` docstring.

`test_recorder_declaration_emit.py` asserts the text. Fibres and layers can point
at MPCO. Modal (G1) has no other target, and Tri31 strain (G5) only a
hand-written MPCO token. *Question:* what does each refusal name once the class
is gone?

**G3: Line stations on `.out`.** `from_recorders` crashes transcoding a
line-stations record on a real FEMData (#1741, open:
`_resolve_element_lengths` reads the element id as a PG tag). Parity is proven
only on a minimal FEM. The transcoder also has no `localForce` synthesis, so an
elastic beam gets no line stations on `.out` (C5); MPCO covers both cases.
*Question:* does #1741 land before step 2, or do line-station users (the chevron
notebook, `viewer_bench`) move to MPCO?

**G4: Fibres and layers have no passing value test on a replacement.** `.out`
cannot carry them, by design. On MPCO, emission passes
(`test_emit_mpco_handles_fibers_and_layers`). Every value-level reader test
(`test_results_mpco_fiber_real.py`, `..._force_beam.py`, `..._disp_beam.py`,
`test_results_mpco_layer_real.py`) skips without the STKO example fixtures:
`APEGMSH_STKO_EXAMPLES`, absent on CI and on this machine. Only mocks pass.
Unverified: whether the MPCO layer path carries what C7 writes (layer thickness
and material from the bridge, element quaternions). *Question:* is mock-level
MPCO evidence enough, or must an apeGmsh-built fibre and layered-shell model be
proven end-to-end on MPCO first?

**G5: Per-material responses on stock.** DomainCapture reads `damage` (any
class) and Tri31 `strain` per Gauss point (C4). The spec-driven `.out` emit
skips Tri31 strain, and its message points at MPCO or DomainCapture. Spec
`emit_mpco` maps no `damage` token, though `from_mpco` reads a hand-written
`material.damage`. The fork's `from_ladruno` covers `damage`, fork-only.
*Question:* is a hand-written MPCO token an acceptable replacement on stock, or
does the spec route need the tokens?

**G6: The real-solver catalog oracle.** `test_results_catalog_{solids,planes,shells,trusses}_real.py`
and `test_results_shell_resultant_vocabulary.py` check `RESPONSE_CATALOG`
layouts against real `openseespy` output. They do it only through
DomainCapture. Both replacements read through the same catalog, so deleting
these tests deletes the only real-solver check of plane, shell and truss
layouts. Solids keep one tet test per route, and those run locally only
(Tcl). *Question:* port these to a recorder or MPCO harness before step 3, or
accept the loss?

**G7: Element families proven only through DomainCapture.** Truss `axial_force`
via Gauss (`test_capture_truss_axial_live.py`), plane and shell Gauss (G6), the
fork's second-order solids LadrunoBrick20 and LadrunoLST
(`test_second_order_solids_live.py`), and the mixed-ndf shell-on-solid nodal
round-trip (`test_shell_on_solid.py`). No replacement-route test covers them.
*Question:* which must gain a replacement-route test before deletion?

**G8: Stages and cadence (C9, C10).** DomainCapture writes many named stages
with their kinds into one file and samples at the caller's instants with the
caller's `t`. `from_recorders` reads one stage per call (`stage_id=`), so a
three-stage run is three `Results`. `from_mpco` keeps all stages in one file but
labels each `transient`, so `static` is lost. Recorders sample every step, or by
`dt` / `n_steps`; arbitrary instants are not expressible. *Question:* are these
differences acceptable, or does a replacement need multi-stage reads and stage
kinds?

**G9: Evidence that runs only on this machine.** The Tcl-subprocess tests in §3
skip on CI, which has no Tcl launcher. They passed here against the installed
Ladruno build, which is older than `FORK_PIN`.
`test_results_mpco_nodal_real.py::test_elastic_beam_3d_global_and_local_force`
fails at the element-id filter (line 158, `[] == [1]`), so MPCO element forces
and MPCO tag translation have no passing end-to-end proof. The `stress_zz` and
`damage` proofs are `ladruno_fork`-marked, and CI never runs them. *Question:*
does local-only evidence count as "parity proven", and is the MPCO filter
failure a bug to fix before step 2?

**G10: Provenance on the default route.** DomainCapture writes
`opensees_backend` / `opensees_build` (C12). The recorder transcode leaves them
absent; `modal.py` says so ("as for a recorder transcode"). MPCO files are not
native and carry neither. Removing the capture removes provenance from the
default route. *Question:* accept, or add provenance to the transcode?

**G11: In-process with no files (C11).** Every replacement writes `.out` or
`.mpco` files, and `.out` is transcoded into a cached native file before
reading. Both also need the model archive (`model=` / `model_h5=`), which
`ops.domain_capture` takes from the live bridge. *Question:* is the file round
trip acceptable for tight loops (G15) and for the tutorials' first page?

**G12: The `source_type = "domain_capture"` value.** `NativeWriter.open`
defaults `source_type` to `SOURCE_DOMAIN_CAPTURE`, so `results/demo.py` and any
writer that omits it label their files `domain_capture`. These all write or
assert the value:
- the 24 committed viewer fixtures;
- `scripts/make_viewer_results_fixtures.py`;
- 13 Python test files (`conftest.py` only in a docstring);
- the three web viewer tests.

The value no longer carries meaning for verification (#1764: stamp only). Files
already on disk carry it. *Question:* keep the constant as a legacy value, change
the writer default (and regenerate the fixtures), or leave both?

**G13: Notebooks already broken on main.** Twelve notebooks are broken today:
10 in `case_studies/` and 2 in `archive/`. They import `Recorders` from
`apeGmsh.results.spec`; a probe of this branch raises `ImportError`. They also
call `DomainCapture(spec, path, fem, ndm=, ndf=)`, which the current signature
rejects. `scripts/wire_eos_notebook.py` generates that cell. This is not a
parity question. *Question:* migrate them in step 2b, retire them, or archive
them, and the generator with them?

**G14: ADR 0108 D5's named seam.** ADR 0108 D5,
`guide_ladruno_adr95_druckerprager_fix.md` and the skill's `ladruno.md` name
DomainCapture's per-material routing as the place a live `ladrunoBranch` capture
would go. Deleting the class removes that future seam. *Question:* drop D5's
plan, or re-home it, for example on the Ladruno recorder?

**G15: Tight finite-difference loops.** `internal_docs/guide_sensitivity.md`
recommends `ops.domain_capture` for a tight FD loop, to skip the file round trip
(C11). *Question:* is a recorder round trip per evaluation acceptable there?

## 6. What step 3 must also carry

These are not gaps, because a replacement exists. Step 3 breaks without them:

- `docs/api/results.md` holds `::: apeGmsh.results.capture._domain.DomainCapture`.
  `mkdocs build --strict` fails if the class goes and the entry stays.
- `src/apeGmsh/studio/_api_index.json` must be rebuilt in the same PR. The
  API-index gate goes red on any public signature change.
- `tests/test_results_domain_capture.py` exports `_MockFem` and `_FakeOps` to
  `test_parallel_modal_to_native.py` and the three viewer tests. Move them before
  deleting the module.
- The orphans are `_shell_geometry.py` (and its test,
  `tests/test_results_shell_geometry.py`, which the grep does not list), plus the
  three catalog helpers named in the `_response_catalog.py` row of Appendix A.

## 7. Verification

- The file list: `git grep -il "DomainCapture\|domain_capture" origin/main`
  gives 204, and `git grep -il -I ...` gives 180. Appendix A has 180 rows; the
  24-file difference is Appendix B.
- Each proof in §3 ran green with the venv Python. The (Tcl) rows needed
  `APEGMSH_OPENSEES_TCL` set; without it they skip.
- `test_results_mpco_nodal_real.py::test_elastic_beam_3d_global_and_local_force`
  fails as reported in G9. It is not claimed as proof.
- The `Recorders` probe in G13 ran with this worktree's `src` first on `sys.path`.

## Appendix A: every text hit (180)

"Kinds" is a heuristic scan for spec categories and `capture_modes` in the file.
Read it as a pointer, not an inventory.

| # | File | Group | Kinds | Action | Step |
|---|---|---|---|---|---|
| 1 | `src/apeGmsh/opensees/_internal/build/recorders.py` | src | - | fibres/layers/modal `NotImplementedError` says "use DomainCapture"; retarget (G1, G2) | 3 |
| 2 | `src/apeGmsh/opensees/_internal/ns/recorder.py` | src | - | comment; reword | 3 |
| 3 | `src/apeGmsh/opensees/_response_catalog.py` | src | - | comments, plus three helpers only DomainCapture calls (`has_dead_gauss_probe`, `needs_per_material_query`, `synthesize_line_station_layout_for_elastic_beam`, per `nav.py refs`); reword, then remove what is orphaned | 3 |
| 4 | `src/apeGmsh/opensees/analysis/modal.py` | src | modal | `to_native` docstring describes its layout as DomainCapture's; reword | 3 |
| 5 | `src/apeGmsh/opensees/apesees.py` | src | nodal, element(nodal forces) | public `ops.domain_capture(...)`; delete the method and its import | 3 |
| 6 | `src/apeGmsh/results/_shell_geometry.py` | src | - | only caller is the layer capturer; orphaned by the deletion | 3 |
| 7 | `src/apeGmsh/results/capture/__init__.py` | src | - | package exports; delete the package | 3 |
| 8 | `src/apeGmsh/results/capture/_domain.py` | src | nodal, element(nodal forces), modal | the class; delete | 3 |
| 9 | `src/apeGmsh/results/capture/spec.py` | src | nodal, gauss, layers, modal | `DomainCaptureSpec` and resolved records; delete | 3 |
| 10 | `src/apeGmsh/results/live/_recorders.py` | src | nodal, modal | modal and fibre/layer refusals send users to DomainCapture; retarget (G1, G2) | 3 |
| 11 | `src/apeGmsh/results/readers/_native.py` | src | - | docstring; reword | 3 |
| 12 | `src/apeGmsh/results/schema/_native.py` | src | - | `SOURCE_DOMAIN_CAPTURE` constant (G12) | 3 |
| 13 | `src/apeGmsh/results/spec/__init__.py` | src | - | module docstring; reword | 3 |
| 14 | `src/apeGmsh/results/spec/_emit.py` | src | element(nodal forces) | Tri31-strain skip message and the deferred-category comments name DomainCapture; retarget (G2, G5) | 3 |
| 15 | `src/apeGmsh/results/spec/_resolved.py` | src | gauss, layers | `emit_recorders` docstring points fibres/layers at DomainCapture; reword (G2) | 3 |
| 16 | `src/apeGmsh/results/transcoders/_recorder.py` | src | - | comments; reword | 3 |
| 17 | `src/apeGmsh/results/writers/_native.py` | src | - | `open()` defaults `source_type` to `domain_capture` (G12) | 3 |
| 18 | `src/apeGmsh/studio/_api_index.json` | src | - | generated index; rebuild after the method goes | 3 |
| 19 | `docs/api/results.md` | docs | - | mkdocstrings entry `::: ...DomainCapture`; must leave with the class or the strict build fails | 3 |
| 20 | `docs/concepts/opensees-bridge.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 21 | `docs/design/results.md` | docs | modal | design page describes the class and `capture_modes`; rewrite (G1) | 2a |
| 22 | `docs/examples/compose-modules.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 23 | `docs/examples/fiber-moment-curvature.md` | docs | - | one aside naming `DomainCaptureSpec`; reword | 2a |
| 24 | `docs/examples/modal-analysis.md` | docs | nodal, modal | whole example is `capture_modes`; blocked on G1 | 2a |
| 25 | `docs/examples/multipart-assembly.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 26 | `docs/examples/portal-frame.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 27 | `docs/examples/pushover-steel-frame.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 28 | `docs/examples/results-strategies.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 29 | `docs/examples/step-plate-with-hole.md` | docs | nodal | one aside naming `DomainCaptureSpec(...).nodes`; reword | 2a |
| 30 | `docs/examples/tie-non-matching-meshes.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 31 | `docs/guides/nonlinear_concrete_solver.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 32 | `docs/how-to/choose-results-strategy.md` | docs | nodal, modal | names DomainCapture the default and the modal route; rewrite the strategy table (G1, G4) | 2a |
| 33 | `docs/how-to/results-mpco.md` | docs | nodal, gauss | contrasts MPCO with domain capture; reword | 2a |
| 34 | `docs/tutorials/beam-and-composites.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 35 | `docs/tutorials/first-model.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 36 | `docs/tutorials/plate-in-tension.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 37 | `docs/tutorials/save-reload-view.md` | docs | nodal | nodal capture in a tutorial or example; move to a replacement | 2a |
| 38 | `examples/EOS Examples/README.md` | examples | - | index prose; reword | 2b |
| 39 | `examples/EOS Examples/archive/README.md` | examples | - | index prose; reword | 2b |
| 40 | `examples/EOS Examples/archive/plate_pyGmsh/plate_pyGmsh.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 41 | `examples/EOS Examples/archive/plate_viewer_v2/plate_viewer_v2.ipynb` | examples | nodal, element(nodal forces) | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 42 | `examples/EOS Examples/case_studies/cantilever_solid_to_frame/cantilever_solid_to_frame.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 43 | `examples/EOS Examples/case_studies/chevron_brace_kink_imperfection/chevron_brace_kink_imperfection.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 44 | `examples/EOS Examples/case_studies/chevron_brace_retrofit_sine/chevron_brace_retrofit_sine.ipynb` | examples | nodal, line_stations | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 45 | `examples/EOS Examples/case_studies/column_to_surface_with_reactions/column_to_surface_with_reactions.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 46 | `examples/EOS Examples/case_studies/footing_flexible_slab_on_springs/footing_flexible_slab_on_springs.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 47 | `examples/EOS Examples/case_studies/footing_rigid_block_on_springs/footing_rigid_block_on_springs.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 48 | `examples/EOS Examples/case_studies/ltb_midspan_load_brick/ltb_midspan_load_brick.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 49 | `examples/EOS Examples/case_studies/ltb_shell/ltb_shell.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 50 | `examples/EOS Examples/case_studies/ltb_uniform_moment_brick/ltb_uniform_moment_brick.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 51 | `examples/EOS Examples/case_studies/thick_walled_cylinder_lame/thick_walled_cylinder_lame.ipynb` | examples | nodal | already broken on main: imports the removed `Recorders` and passes `ndm=`/`ndf=` (G13) | 2b |
| 52 | `examples/EOS Examples/curriculum/01-fundamentals/01_hello_plate/01_hello_plate.ipynb` | examples | nodal, modal | nodes capture, plus prose pointing at `capture_modes` | 2b |
| 53 | `examples/EOS Examples/curriculum/01-fundamentals/02_cantilever_beam_2D/02_cantilever_beam_2D.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 54 | `examples/EOS Examples/curriculum/01-fundamentals/03_simply_supported_beam/03_simply_supported_beam.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 55 | `examples/EOS Examples/curriculum/01-fundamentals/04_portal_frame_2D/04_portal_frame_2D.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 56 | `examples/EOS Examples/curriculum/02-building-blocks/05_labels_and_pgs/05_labels_and_pgs.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 57 | `examples/EOS Examples/curriculum/02-building-blocks/06_sections_catalog/06_sections_catalog.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 58 | `examples/EOS Examples/curriculum/02-building-blocks/07_load_patterns/07_load_patterns.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 59 | `examples/EOS Examples/curriculum/02-building-blocks/08_boundary_conditions/08_boundary_conditions.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 60 | `examples/EOS Examples/curriculum/03-assemblies/10_parts_basics/10_parts_basics.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 61 | `examples/EOS Examples/curriculum/03-assemblies/10b_part_assembly/10b_part_assembly.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 62 | `examples/EOS Examples/curriculum/03-assemblies/11_boolean_assembly/11_boolean_assembly.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 63 | `examples/EOS Examples/curriculum/03-assemblies/12_interface_tie/12_interface_tie.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 64 | `examples/EOS Examples/curriculum/04-coupling/13_beam_to_solid/13_beam_to_solid.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 65 | `examples/EOS Examples/curriculum/04-coupling/14_contact_springs/14_contact_springs.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 66 | `examples/EOS Examples/curriculum/04-coupling/15_embedded_rebar/15_embedded_rebar.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 67 | `examples/EOS Examples/curriculum/04-coupling/16_tied_contact_nonmatching/16_tied_contact_nonmatching.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 68 | `examples/EOS Examples/curriculum/05-analysis-types/17_modal_analysis/17_modal_analysis.ipynb` | examples | modal | modal-only; blocked on G1 | 2b |
| 69 | `examples/EOS Examples/curriculum/05-analysis-types/18_buckling_column/18_buckling_column.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 70 | `examples/EOS Examples/curriculum/05-analysis-types/19_pushover_elastoplastic/19_pushover_elastoplastic.ipynb` | examples | nodal | nodal capture; move to a replacement | 2b |
| 71 | `.claude/skills/apegmsh-helper/references/ladruno.md` | skill | - | mirror; regenerated by `sync_skill.py` | 2c |
| 72 | `.claude/skills/apegmsh-helper/references/opensees-bridge.md` | skill | modal | mirror; regenerated by `sync_skill.py`, never edited | 2c |
| 73 | `skills/apegmsh/references/ladruno.md` | skill | - | "the live DomainCapture path is still not adopted" (ADR 0108 D5, G14) | 2c |
| 74 | `skills/apegmsh/references/opensees-bridge.md` | skill | modal | names `DomainCaptureSpec` as the in-process route; rewrite, then `sync_skill.py` | 2c |
| 75 | `tests/opensees/h5/test_h5_meta_ndm.py` | tests/opensees | nodal | pins `/meta/ndm` through `DomainCapture.from_h5` among other readers; drop that reader case | 3 |
| 76 | `tests/opensees/h5/test_h5_meta_ndm_raw_readers.py` | tests/opensees | nodal | same, for the raw-stamp readers; drop the capture case | 3 |
| 77 | `tests/opensees/h5/test_h5_staged_domain_capture.py` | tests/opensees | nodal | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 78 | `tests/opensees/integration/test_capture_layers_bridge_live.py` | tests/opensees | layers | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 79 | `tests/opensees/integration/test_capture_truss_axial_live.py` | tests/opensees | gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 80 | `tests/opensees/integration/test_recorder_declaration_emit.py` | tests/opensees | fibers, layers, modal | asserts the "use DomainCapture" refusal text; follows G2 | 3 |
| 81 | `tests/opensees/integration/test_shell_on_solid.py` | tests/opensees | nodal | mixed-ndf nodal round-trip proven only through DomainCapture (G7); port the round-trip | 3 |
| 82 | `tests/opensees/integration/test_stress_zz_promotion.py` | tests/opensees | gauss | capture-spec half of the promotion tests; delete that half, keep the emit half | 3 |
| 83 | `tests/opensees/integration_ladruno/test_concrete_damage_capture_live.py` | tests/opensees | gauss | capture half; the `.ladruno` read half covers damage on the fork (G5) | 3 |
| 84 | `tests/opensees/integration_ladruno/test_second_order_solids_live.py` | tests/opensees | gauss | gauss read-back of LadrunoBrick20/LST proven only through DomainCapture (G7) | 3 |
| 85 | `tests/opensees/integration_ladruno/test_stress_zz_plane_strain_live.py` | tests/opensees | gauss | one of three tests is the capture route; the `.out` and `.ladruno` tests stay | 3 |
| 86 | `tests/opensees/unit/test_backend_info.py` | tests/opensees | - | provenance written by DomainCapture (G10); port to the surviving writer or drop | 3 |
| 87 | `tests/viewers/test_render.py` | tests/viewers | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 88 | `tests/viewers/test_stage_activation.py` | tests/viewers | nodal | uses DomainCapture only to write a results file; rewrite on NativeWriter or a replacement route; imports `_FakeOps` from the class's test module | 3 |
| 89 | `tests/viewers/test_x2f_animation_stage_activation.py` | tests/viewers | nodal | uses DomainCapture only to write a results file; rewrite on NativeWriter or a replacement route; imports `_FakeOps` | 3 |
| 90 | `tests/viewers/test_x2g_stage_scoped_render.py` | tests/viewers | nodal | uses DomainCapture only to write a results file; rewrite on NativeWriter or a replacement route; imports `_FakeOps` | 3 |
| 91 | `tests/assess/test_assess.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 92 | `tests/assess/test_assess_opensees.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 93 | `tests/conftest.py` | tests (other) | - | docstring mentions the `source_type` value; reword if G12 changes it | - |
| 94 | `tests/fixtures/capture_tags.py` | tests (other) | - | `tag_map=` helper for DomainCapture tests; delete with them | 3 |
| 95 | `tests/results/test_capture_class_hint.py` | tests (other) | gauss, layers | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 96 | `tests/results/test_gauss_skip_warning.py` | tests (other) | - | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 97 | `tests/test_check_quirks.py` | tests (other) | - | self-test of the `openseespy-import` rule, which names the capture fallback; follows the rule text | 3 |
| 98 | `tests/test_custom_scalar_expressions.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 99 | `tests/test_derived_scalars_composite.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 100 | `tests/test_docs_results_api_drift.py` | tests (other) | modal | asserts the documented DomainCapture symbols exist; update with the docs | 3 |
| 101 | `tests/test_domain_capture_ops_binding.py` | tests (other) | - | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 102 | `tests/test_parallel_modal_to_native.py` | tests (other) | modal | imports `_MockFem` from `test_results_domain_capture.py`; move the helper | 3 |
| 103 | `tests/test_result_chain.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 104 | `tests/test_result_chain_subcomposites.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 105 | `tests/test_results_catalog_planes_real.py` | tests (other) | gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7); the real-solver catalog oracle (G6) | 3 |
| 106 | `tests/test_results_catalog_shells_real.py` | tests (other) | gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7); the real-solver catalog oracle (G6) | 3 |
| 107 | `tests/test_results_catalog_solids_real.py` | tests (other) | gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7); the real-solver catalog oracle (G6) | 3 |
| 108 | `tests/test_results_catalog_trusses_real.py` | tests (other) | gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7); the real-solver catalog oracle (G6) | 3 |
| 109 | `tests/test_results_composite.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 110 | `tests/test_results_domain_capture.py` | tests (other) | nodal, layers, modal | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 111 | `tests/test_results_domain_capture_composed.py` | tests (other) | gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 112 | `tests/test_results_domain_capture_fibers.py` | tests (other) | fibers | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 113 | `tests/test_results_domain_capture_fibers_real.py` | tests (other) | fibers | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 114 | `tests/test_results_domain_capture_gauss.py` | tests (other) | gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 115 | `tests/test_results_domain_capture_index_stamp.py` | tests (other) | nodal, element(nodal forces) | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 116 | `tests/test_results_domain_capture_layers.py` | tests (other) | layers | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 117 | `tests/test_results_domain_capture_layers_real.py` | tests (other) | layers | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 118 | `tests/test_results_domain_capture_line_stations.py` | tests (other) | gauss, line_stations | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 119 | `tests/test_results_domain_capture_line_stations_real.py` | tests (other) | line_stations | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 120 | `tests/test_results_domain_capture_nodal_forces.py` | tests (other) | element(nodal forces), gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 121 | `tests/test_results_domain_capture_nodal_forces_real.py` | tests (other) | element(nodal forces) | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 122 | `tests/test_results_domain_capture_real.py` | tests (other) | nodal, gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 123 | `tests/test_results_domain_capture_spec.py` | tests (other) | nodal, element(nodal forces), gauss, line_stations, fibers, layers, modal | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7) | 3 |
| 124 | `tests/test_results_element_index_stamp_dtype.py` | tests (other) | nodal | uses DomainCapture as one of the stamping writers; drop that writer case | 3 |
| 125 | `tests/test_results_element_response.py` | tests (other) | - | names DomainCapture in a comment, docstring or message; reword | 3 |
| 126 | `tests/test_results_femdata_native_roundtrip.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 127 | `tests/test_results_mpco_line_mock.py` | tests (other) | - | names DomainCapture in a comment, docstring or message; reword | 3 |
| 128 | `tests/test_results_native_roundtrip.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 129 | `tests/test_results_recorder_tag_parity.py` | tests (other) | - | names DomainCapture in a comment, docstring or message; reword | 3 |
| 130 | `tests/test_results_recorder_transcoder_gauss_real.py` | tests (other) | gauss | names DomainCapture in a comment, docstring or message; reword | 3 |
| 131 | `tests/test_results_recorder_transcoder_line_stations.py` | tests (other) | line_stations | names DomainCapture in a comment, docstring or message; reword | 3 |
| 132 | `tests/test_results_recorder_transcoder_line_stations_real.py` | tests (other) | line_stations | names DomainCapture in a comment, docstring or message; reword | 3 |
| 133 | `tests/test_results_recorder_transcoder_nodal_forces_real.py` | tests (other) | element(nodal forces) | names DomainCapture in a comment, docstring or message; reword | 3 |
| 134 | `tests/test_results_shell_resultant_vocabulary.py` | tests (other) | gauss | tests DomainCapture itself; delete with the class, after salvaging what its oracle covers (G6, G7); shell resultant names checked on a real solve (G6) | 3 |
| 135 | `tests/test_results_spatial_filters.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 136 | `tests/test_selection_idiom.py` | tests (other) | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 137 | `apeGmshViewer/test/results/pairing.test.ts` | web-viewer tests | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 138 | `apeGmshViewer/test/results/results.test.ts` | web-viewer tests | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 139 | `apeGmshViewer/test/store.test.ts` | web-viewer tests | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 140 | `scripts/check_quirks.py` | scripts | - | `openseespy-import` rule text cites the DomainCapture lesson; keep or reword | 3 |
| 141 | `scripts/make_viewer_results_fixtures.py` | scripts | - | writes `source_type="domain_capture"` on NativeWriter (G12); no class use | 3 |
| 142 | `scripts/render_showcase/modal_sweep.py` | scripts | modal | showcase animation from `capture_modes`; blocked on G1 | 2d |
| 143 | `scripts/render_showcase/moment_tensor.py` | scripts | nodal | velocity every step via `ops.domain_capture`; move to a replacement | 2d |
| 144 | `scripts/render_showcase/pushover_frame.py` | scripts | nodal | nodal capture; move to a replacement | 2d |
| 145 | `scripts/render_showcase/stills.py` | scripts | nodal | nodal capture; move to a replacement | 2d |
| 146 | `scripts/wire_eos_notebook.py` | scripts | nodal, modal | writes the stale `DomainCapture(..., ndm=, ndf=)` + `Recorders()` cell into notebooks (G13) | 2d |
| 147 | `viewer_bench/README.md` | viewer_bench | modal | history of the `capture_modes` rotation fix; reword with the bench | 2d |
| 148 | `viewer_bench/models/ssi_frame_wall/src/build.py` | viewer_bench | nodal, gauss, line_stations, modal | nodes + gauss + line stations + `capture_modes`; blocked on G1 and G3 | 2d |
| 149 | `architecture/api-design.md` | architecture | - | lists `ops.domain_capture`; update with the deletion | 3 |
| 150 | `architecture/h5-schema.md` | architecture | nodal, modal | `source_type` values, `capture_modes` layout, `DomainCapture.from_h5`; update with the deletion (G12) | 3 |
| 151 | `architecture/parallel-execution.md` | architecture | - | names the capture entry points; update with the deletion | 3 |
| 152 | `internal_docs/Results_architecture.md` | internal_docs | nodal, element(nodal forces), gauss, line_stations, fibers, layers, modal | living architecture note; reword with the deletion | 3 |
| 153 | `internal_docs/guide_ladruno_adr95_druckerprager_fix.md` | internal_docs | - | "the only live seam is DomainCapture" (G14); reword | 3 |
| 154 | `internal_docs/guide_sensitivity.md` | internal_docs | modal | recommends `ops.domain_capture` for tight FD loops (G15); reword | 3 |
| 155 | `internal_docs/plan_element_transcoding.md` | internal_docs | - | history (plan, handoff or readability study); leave | leave |
| 156 | `internal_docs/plan_phase_11b_line_stations.md` | internal_docs | - | history (plan, handoff or readability study); leave | leave |
| 157 | `internal_docs/plan_staged_h5_archival_phase2.md` | internal_docs | - | history (plan, handoff or readability study); leave | leave |
| 158 | `internal_docs/plan_staged_h5_phase5_partitioned.md` | internal_docs | - | history (plan, handoff or readability study); leave | leave |
| 159 | `internal_docs/readability/REPORT.md` | internal_docs | - | history (plan, handoff or readability study); leave | leave |
| 160 | `internal_docs/readability/api_gaps.md` | internal_docs | - | history (plan, handoff or readability study); leave | leave |
| 161 | `internal_docs/readability/loop1/c.py` | internal_docs | nodal, gauss | history (plan, handoff or readability study); leave | leave |
| 162 | `internal_docs/readability/samples/02_solid_cantilever.py` | internal_docs | nodal | history (plan, handoff or readability study); leave | leave |
| 163 | `architecture/decisions/0046-shell-on-solid-node-sharing-guard.md` | ADR | - | history; leave, add a superseded note | 3 (note) |
| 164 | `architecture/decisions/0055-staged-h5-archival.md` | ADR | - | history; leave, add a superseded note | 3 (note) |
| 165 | `architecture/decisions/0077-parallel-modal-analysis.md` | ADR | modal | history; leave, add a superseded note | 3 (note) |
| 166 | `architecture/decisions/0093-zerolength-interface-constraint.md` | ADR | - | history; leave, add a superseded note | 3 (note) |
| 167 | `architecture/decisions/0098-results-session-presentation.md` | ADR | - | history; leave, add a superseded note | 3 (note) |
| 168 | `architecture/decisions/0108-ladruno-branch-read-back.md` | ADR | - | history; leave, add a superseded note | 3 (note) |
| 169 | `architecture/decisions/0121-charter-v2-b-with-amendments.md` | ADR | - | history; leave, add a superseded note | 3 (note) |
| 170 | `CHANGELOG.md` | changelog | nodal, line_stations, fibers, modal | history; leave (fragments are assembled, never edited) | leave |
| 171 | `changelog.d/2026-10-03-doc-drift-1322-1323-1326-1328.md` | changelog | - | history; leave (fragments are assembled, never edited) | leave |
| 172 | `changelog.d/2026-10-03-meta-ndm-spatial.md` | changelog | - | history; leave (fragments are assembled, never edited) | leave |
| 173 | `changelog.d/2026-10-04-raw-ndm-readers.md` | changelog | - | history; leave (fragments are assembled, never edited) | leave |
| 174 | `changelog.d/2026-10-12-domain-capture-index-stamp.md` | changelog | - | history; leave (fragments are assembled, never edited) | leave |
| 175 | `changelog.d/2026-10-12-index-stamp-vlen.md` | changelog | - | history; leave (fragments are assembled, never edited) | leave |
| 176 | `changelog.d/f2b-backend-info.md` | changelog | - | history; leave (fragments are assembled, never edited) | leave |
| 177 | `changelog.d/v4a-results-measurements.md` | changelog | - | history; leave (fragments are assembled, never edited) | leave |
| 178 | `changelog.d/v4g-viewer-element-gate-stamp-only.md` | changelog | - | history; leave (fragments are assembled, never edited) | leave |
| 179 | `apeGmshViewer/MEASUREMENTS.md` | web-viewer docs | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |
| 180 | `apeGmshViewer/fixtures/README.md` | web-viewer docs | - | writes or asserts the `source_type` value only; no change unless G12 changes the value | - |

## Appendix B: the 24 binary hits, left out on purpose

Each holds `domain_capture` only as its `source_type` attribute value, written
through `NativeWriter` by `scripts/make_viewer_results_fixtures.py`. None was
produced by DomainCapture. Each follows G12, and none changes in step 2.

- `apeGmshViewer/fixtures/contour_nodata.results.h5`
- `apeGmshViewer/fixtures/contour_staged.results.h5`
- `apeGmshViewer/fixtures/deform_2d.results.h5`
- `apeGmshViewer/fixtures/elem_badid.results.h5`
- `apeGmshViewer/fixtures/elem_mixed.results.h5`
- `apeGmshViewer/fixtures/elem_mixed_unstamped.results.h5`
- `apeGmshViewer/fixtures/elem_solid.results.h5`
- `apeGmshViewer/fixtures/elem_staged.results.h5`
- `apeGmshViewer/fixtures/extrap_plane.results.h5`
- `apeGmshViewer/fixtures/extrap_solid.results.h5`
- `apeGmshViewer/fixtures/extrap_tet.results.h5`
- `apeGmshViewer/fixtures/results_small.h5`
- `apeGmshViewer/fixtures/results_stamped.h5`
- `apeGmshViewer/fixtures/staged_activation.femonly.results.h5`
- `apeGmshViewer/fixtures/staged_activation.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/fixed_S3.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/fixed_S8_nullpad.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/fixed_utf8_3.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/fixed_utf8_8_nullpad.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/np_bytes_S3.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/vlen_array.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/vlen_ascii_str.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/vlen_bytes.results.h5`
- `apeGmshViewer/fixtures/stamp_variants/vlen_str.results.h5`
