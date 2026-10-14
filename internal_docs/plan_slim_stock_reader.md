# Plan: the slim stock-only reader

The maintainer ratified this plan on 2026-10-14 (program slice [#1961](https://github.com/nmorabowen/apeGmsh/issues/1961), link X). It says how modes are read on pip `openseespy`, where no MPCO exists, and what is left open for fibres, layers and materials. The design brief follows verbatim after the rulings below.

## Ratified answers

The maintainer answered the brief's four open questions, each with the recommended option.

1. Fork export: `EigenResult.to_native` is allowed on the fork as an explicit export, which covers `eigen_feast`. The resolver never offers it there; "never on fork" is held in the resolver, not in the writer.
2. Source type: the writer stamps `source_type="eigen"`.
3. Opener: `EigenResult` gets `results()`, mirroring #1950's `results(model_h5=None)`.
4. Fibres and materials on pip: probe first. A small probe checks whether the pip 3.8.0 `.out` element recorder carries `fiberData`, `fiberData2` and material responses at `-precision 16`, against an analytic oracle. A by-ops-tag reader comes up only if the probe fails. Layers stay off pip and are refused by name.

## Build cards

- #1972: `EigenResult.to_native` (PR 1 of the migration table).
- #1973: the fibre and material probe (PR 3).

---
# Brief #1961: the slim stock-only reader (design)

Revision 2, after the Opus review (`review_1961.md`): B1 and N1–N6 addressed.

**Verdict up front: the card's stop condition holds.** Modes on pip need no new reader class: `EigenResult.to_native`, sharing one stage-writing loop with `ParallelModalResult.to_native`, covers them. Fibres, layers and materials are a separate maintainer decision, starting with a `.out` probe.

## 1. Problem and constraints

- pip `openseespy` 3.8.0 has no MPCO, so #1950's stock fallback there is `.out` (`results/live/_recorders.py::LiveRecorders`). It refuses modal records (L85, L172) and skips fibres and layers (L81).
- Today only `results/capture/_domain.py::DomainCapture.capture_modes` (L1143–1239) writes `kind="mode"` stages in a single process. It is deleted; #1946 covers `.ladruno` and MPCO only.
- `ParallelModalResult.to_native` (`opensees/analysis/modal.py` L777–869) writes the same layout (one `kind="mode"` stage per mode at `time=[0.0]`) and the same warn-then-0 rule for λ ≤ 0.
- `EigenResult` (`opensees/analysis/eigen.py`, 112 lines) is lazy. It holds `eigenvalues` and `_live` only: no fem, ndm or staleness check. Both `_ModalMixin.eigen` (stock or fork) and `_ModalMixin.eigen_feast` (fork only) return it (`opensees/procedures/_modal.py` L48–133, L593–664).
- Modes are nodal, so no element tags are involved (`capture_modes` queries `fem.nodes.ids` as ops tags). It writes `rotation_*` when the envelope `ndf >= 6` (`_domain.py` L538, L1198) and zero-fills shorter node vectors. `ParallelModalResult` writes `min(3, ndm, ndf)` translations (modal.py L856).
- Staleness: many procedures rebuild the one process-global domain with `LiveOpsEmitter(wipe=True)`, including `modal_properties`, `complex_eigen`, `modal_response_history`, `response_spectrum_analysis`, `_explicit`, `_frf`, `strut_tie` and a second bridge. Every such rebuild bumps `emitter/live.py::domain_epoch()` (L640, bumped in `LiveOpsEmitter.__init__`), the ADR 0115 D2 counter that `_readback.py::open_session` already keys on. Most of them never touch `apeSees._read_state`.
- Provenance (G10): `NativeWriter.open` refuses a build stamp on stock (`_native.py` L150–155). The stamp must name the module that actually solved, because `get_backend_info()` can name another module in a mixed setup (`_domain.py::DomainCapture._lazy_ops` docstring). `emitter/live.py::_backend_of(ops)` (L317) gives that verdict: it unwraps `_CoercingOps` and reuses the resolver's cached result.
- G11: a file round trip with a temp-dir default. `Results.from_native` requires `model=` (L355, ADR 0020 INV-1). The file is only usable if it embeds `/opensees` through `model_h5_src=`.
- G12: the ruling makes `"native"` the writer default, but that is not implemented yet: today's default is `SOURCE_DOMAIN_CAPTURE` (`_native.py` L114). Writers that know their source name it: `"parallel_modal"` (modal.py L820) and `"footfall_walking"` (`footfall_result.py` L231).
- Fork gate: #1950's `emit_default` never offers this route on the fork.
- The translator is mandatory for anything element-scoped. Precedent: `_domain.py::_require_element_pairing` (L274–323) and `spec/_emit.py::element_tag_map` / `to_ops_tags` (L295–363).

## 2. Options

**A. `EigenResult.to_native` (the stop condition).** About 60 lines plus a shared helper. One `nodeEigenvector` per fem node per mode, as in `capture_modes`.
- *Cost:* `EigenResult` gains four private fields (`_bridge`, `_ndm`, `_ndf`, `_epoch`), which `eigen` and `eigen_feast` must fill. The public API index changes. The lazy domain can be stale; see invariant 5.

**B. A new spec-driven stock reader class** for modes, fibres and materials.
- *Cost:* it rebuilds what is being deleted (`_FiberCapturer` L2113–2318, `_LayerCapturer` L2372–2673 plus `_shell_geometry.py`) and re-opens the translator failure class.

**C. Modes as a spec record on `emit_default` (#1950):** `begin_stage(kind="mode")` runs `eigen` itself.
- *Cost:* solver driving inside a recorder context, and wider #1950 scope for a stock-only branch.

**D. Fibres and materials over classic `.out` element recorders** (`section k fiberData`, `material k <resp>`), with the ids translated by `to_ops_tags`.
- *Cost:* unproven on pip 3.8.0; needs a spec-emit change and an `.out` fibre layout; cannot carry layers (bridge thickness and quaternions).

## 3. Decision

1. **Modes take Option A.** No reader class is added.
   - Public surface: `EigenResult.to_native(path=None) -> Path`, with no `fem=` override. `/model` (from `_bridge._fem`) and `/opensees` (from `_bridge.h5(...)`, embedded with `model_h5_src=`) then always describe one bridge. `Results.from_native(p, model=OpenSeesModel.from_h5(p))` opens the file alone.
   - `path=None` writes `<mkdtemp(prefix="apegmsh_modes_")>/modes.h5`.
   - A bridge mutated between `eigen` and `to_native` without a re-emit is documented as unsupported. Any re-emit is caught by the epoch.
   - Zero eigenvalues (an empty FEAST band) raise `ValueError` naming the band. No empty file is written.
   - `source_type="eigen"`, following the named-writer precedent; G12's `"native"` default stays the default for writers that name nothing.
   - Provenance comes from `_backend_of(self._live.ops)`.
2. **The shared loop.** Extract the stage loop of `ParallelModalResult.to_native` into a module function, `_write_mode_stages(writer, eigenvalues, node_ids, fields, n_trans, rotations)`. It holds the warning text and the 0-frequency rule, and both methods call it. Each caller passes its own rule:
   - `ParallelModalResult` passes `n_trans = min(3, ndm, ndf)` and `rotations = ndf >= 6` (unchanged);
   - `EigenResult` passes `n_trans = min(3, ndm)` and `rotations = bridge _ndf >= 6`, zero-filling short node vectors as `capture_modes` does.
3. **No backend branch in `to_native`.** It is an explicit export, never offered; "never chosen on the fork" is held where choosing happens (#1950's resolver, #1958's text). See Q1.
4. **Fibre, layer and per-material responses become a separate maintainer decision.** Recommendation:
   - first, a probe slice for Option D on pip 3.8.0;
   - if `.out` carries `fiberData` and `material k damage`, extend the spec `.out` emit with no in-process reader;
   - layers stay off pip and are refused by name;
   - only if D fails, a by-ops-tag reader with the translator mandatory, as in invariant 8.

## 4. Invariants

1. After `r = ops.eigen(n)`, `r.to_native(p)` writes exactly `len(r.eigenvalues)` stages, all with `kind="mode"` and `mode_index` values 1..n in order.
2. Each stage's `eigenvalue` equals `r.eigenvalues[k-1]` bit for bit. `frequency_hz = ω/2π` and `period_s = 2π/ω`, with `ω = √λ` (the code's formulas, asserted at rtol 1e-15). For λ ≤ 0, both are 0 and exactly one `UserWarning` names the mode.
3. For every fem node and mode, `displacement_*` equals the first `min(3, ndm)` entries of `ops.nodeEigenvector(tag, k)`, zero-filled past the vector's length. `rotation_*` is written iff the bridge's `_ndf >= 6`, and holds entries 4–6, or 0.0 for a node whose vector is shorter. A mixed solid-and-frame model never raises.
4. The written `/stages` match what `capture_modes` writes for the same domain: the same datasets, attrs and values, apart from the `source_type` attr.
5. `to_native` raises `RuntimeError` when `domain_epoch()` differs from `_epoch`. `_epoch` is captured right after the solve. A difference means any `LiveOpsEmitter` rebuilt the domain since then, from any procedure or any bridge. `to_native` never writes another solve's vectors, and a deck-only `ops.tcl(path)` does not invalidate a result.
6. Root attrs: stock writes `opensees_backend="stock"` and no build attr. On the fork it writes `"fork"` plus the `ladrunoBuild()` sha of the module that solved.
6a. When `eigen_feast` finds an empty band, `to_native` raises `ValueError` and creates no file.
7. `path=None` returns a path under `tempfile.gettempdir()`. That file opens with `Results.from_native(p, model=OpenSeesModel.from_h5(p))`, and `.modes` is non-empty.
8. *(Applies only if a fibre or material reader is ever built.)* Without a translator covering every requested fem id, it raises before any `ops` query. Rows are labelled `fem_eid`.
9. #1950's `emit_default` never yields a mode or eigen route. With `BackendInfo.kind == "fork"`, no message or route names `EigenResult.to_native`. This is not tested in PR 1: it goes into the done-when of #1950 (the resolver) and #1958 (the refusal text).
10. `ParallelModalResult.to_native` output is unchanged by the refactor: its existing tests pass unmodified.

## 5. Migration

| PR | Content | Files |
|---|---|---|
| **PR 1:** modes on stock. It does not depend on #1950, so it can start now, and it must land before the deletion PR | `opensees/analysis/eigen.py` (`to_native`, four private fields); `opensees/analysis/modal.py` (extract `_write_mode_stages`); `opensees/procedures/_modal.py` (fill the fields in `eigen` and `eigen_feast`); `tests/test_eigen_to_native.py` (new, fake ops with its own fem stub, not `_MockFem` from a file the deletion removes; covers invariants 1–7 and 10, including the fork half of invariant 6 through a fake `ladrunoBuild`); `tests/opensees/live/test_eigen_to_native_live.py` (new, live-stock: invariants 1–3, 6, 7); the regenerated API index; `changelog.d/eigen-to-native.md` | about 7 |
| **PR 2:** retarget | No new card: #1958 and the step-2 docs (`modal-analysis.md`, `17_modal_analysis.ipynb`, `modal_sweep.py`, `ssi_frame_wall`) name `EigenResult.to_native` on stock, `.ladruno` modes on the fork | — |
| **PR 3:** the fibre and material probe, as a test-only slice | On the live-stock lane, record `section 1 fiberData`, `section 1 fiberData2` and `material 1 damage`, using a material that exposes `damage`, with raw `ops.recorder` and `-precision 16`. Check against an analytic oracle (fibre strain = ε₀ − κy on an elastic section), not only against `eleResponse`, which shares `setResponse`. Report to the maintainer, who then picks D or a reader | 2 |

The parity oracle (invariant 4) must run in PR 1, while `DomainCapture` still exists.

## 6. Oracles

- **Live stock:** a 2-D cantilever with lumped masses and `ops.eigen(3)`. Eigenvalues are exact; every shape equals `nodeEigenvector` (atol 0); `frequency_hz` and `period_s` follow from the eigenvalues.
- **Mixed ndf:** a 3-D brick with a frame column (ndf 3 and 6) through a fake ops. No raise; the solid nodes' rotations are exactly 0.
- **Parity:** the same fake ops through `capture_modes` and `to_native`, then a dataset-by-dataset diff with an h5 walk.
- **Warn-as-contract:** run `-W error::UserWarning` on a positive-definite case, and catch exactly one warning on a fake case with λ < 0.
- **Staleness:** `eigen(3)`, then `modal_properties(5)`, then the first result's `to_native` must raise. Repeat with a second bridge's `eigen`. Both cases are ones a `_read_state` token misses. Also check that `eigen`, then `ops.tcl(path)` (deck only), then `to_native` succeeds.
- **Mutation checks:** dropping the per-node length guard must fail the mixed-ndf test. Comparing `_read_state` in place of the epoch must fail the `modal_properties` case.

## 7. What not to build

- No reader class for modes, no modal record on `emit_default` or `LiveRecorders`.
- No backend switch or fork refusal in `to_native`.
- No layer support on pip.
- No live in-memory results API: G15 tight loops read `ops.*` directly.
- No `results()` opener unless Q3 says so.

## 8. Open questions for the maintainer

1. On the fork, `EigenResult.to_native` is allowed but never offered (an explicit export only). Is that acceptable? The alternative, a refusal, would leave `eigen_feast` results with no native writer while `from_ladruno` mode fields are still None (#1946).
2. Should it write `source_type="eigen"`, following the named-writer precedent, or `"native"`, as the card says? No reader branches on the value.
3. Should `EigenResult` also get `results()`, mirroring #1950's `results(model_h5=None)`?
4. Fibres and materials: run the PR 3 probe first, or authorize a by-ops-tag reader now?
