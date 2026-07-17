# Changelog

## Unreleased — shell-on-solid conformity (S1a + S1b + S2 + S5) · Phase SSI-2.D stage-bound BCs and recorders · embedded-element pipeline hardening (#329 / #331) · ASDEmbeddedNodeElement option exposure (ADR 0035) · stage-bound constraints + `s.initial_stress` PUSH (Phase SSI-2.D extension) · **Phase SSI-2.E between-stage Domain mutators** · topology safety nets (P1/P3) + arc-line wire docs · embedded-host decomposition (ADR 0036) · **higher-order line broker split (ADR 0037)** · RecorderDeclaration element fan-out fix · **orphan-geometry sweep unification + `g.model.geometry` validation API** · **split-sweep auto-validation (closed-world / open-world)** · **raw-PG channel for `_user_intentional`** · **`g.model.geometry.add_arch` (apex-as-vertex two-arc arch)** · **damping definition `ops.damping` / `s.damping` (ADR 0053, D1–D5)** · **Ladruno J2 plasticity materials (`LadrunoJ2` / `LadrunoUniaxialJ2` / `LadrunoJ2Finite`)** · **Ladruno material wrappers (`LogStrain` / `InitDefGrad` / `StagedStrain` / `LadrunoRebarBuckling`)** · **Ladruno live Monitor recorder (`ops.recorder.Monitor` + `read_monitor` / `tail_monitor`)** · **`LadrunoBrick` fail-loud on a finite-strain material under `geom != "finite"`** · **`add_rectangle(plane=…)` canonical-plane rectangles** · **`ops.ndf` for element-less decoupled nodes + per-node ndf gates G1–G3 (ADR 0049 DOF half)** · **node-pair `ops.element.ZeroLength/CoupledZeroLength/TwoNodeLink(nodes=…)` springs to a decoupled ground (ADR 0049)** · **`g.parts.add_plane_wave_box` — soil box + ASDAbsorbingBoundary skin (ADR 0054, AB-1a)** · **`ASDAbsorbingBoundary3D` bridge element + `ops.element.absorbing_boundary` (ADR 0054, AB-2)** · **`s.activate_absorbing()` staged absorbing-boundary flip (ADR 0054, AB-3)** · **plane-wave SSI worked example (ADR 0054, AB-4)** · **`g.parts.add_absorbing_shell` — bring-your-own-box absorbing skin (ADR 0054, AB-1b)** · **loads / masses fit the per-node `ndf` not the model envelope (mixed-`ndf` `from_model` silent-drop fix)** · **layered (stratified) absorbing boxes + per-layer material (ADR 0054, AB-1c layered slice)** · **absorbing-skin aspect-ratio warning + centred-box mesh fix; rotation documented as unsupported (ADR 0054, AB-1c close-out)** · **staged-model H5 archival — write + read (ADR 0055 Phase 2, P2.1 + P2.2, schema 2.18.0)** · **results-viewer event/state Phase 1 — composition gate revived for backend-routed diagrams + outline eye-toggle dispatcher routing + deformed-ghost runtime state** · **REMOVED — deprecated standalone `apeGmshViewer/` app** · **viewer state-contract V1 — dispatcher-always + owner-fired events + `gesture_batch` (ADR 0056)** · **ActiveObjects initial-state seed + `qt`-marked window tests runnable per-file** · **viewer state-contract V2 — AST guard `test_viewer_state_contract.py` (ADR 0056 INV-5)** · **viewer state-contract V3 — mesh viewer joins the dispatcher (owner-fired VisibilityManager/OverlayVisibilityModel + owned overlay scales + widened guard)** · **viewer state-contract V4 — model viewer joins (double-render retired; ActiveObjects kept as focus-state owner, OQ3 resolved)** · **viewer state-contract V5 — projection audit (Session tab rebuilds from owners; never-worked "Load arrows" scale slider fixed); ADR 0056 Accepted (runway V0–V5 complete)** · **`LadrunoQuad` fork plane element (`ops.element.LadrunoQuad`, tag 33007)** · **`LadrunoCST` fork plane triangle (`ops.element.LadrunoCST`, tag 33008)** · **solution-strategy ladder + established profiles (ADR 0057 Phase A)** · **partitioned-H5 baseline fixes — capture dedupe + partitions restore + INV-5 fallback round-trip (ADR 0055 Phase 5 / P5.0)** · **fiber diagrams sit at the beam's TRUE integration stations (`FiberSlab.station_natural_coord` from MPCO GP_X / .ladruno GP_PARAM / live integrationPoints)** · **`g.constraints.kinematic_coupling` now emits the fork `LadrunoKinematicCoupling` (RBE2, tag 33012) — BREAKING, replaces the `equalDOF` expansion** · **`g.constraints.distributing_coupling` (RBE3) ships — emits the fork `LadrunoDistributingCoupling` (tag 33011), replacing the `NotImplementedError` stub** · **degraded GP world-coordinate reconstructions are loud (`WarnGaussCoordsApproximate`)** · **diagram scalar-state consolidation — `ScalarColorSupport` mixin + base `_scoped_results` (a `set_fmt` now survives colormap changes on every diagram)** · **viewers consume the remaining recorder channels — diagrams orient from `.ladruno` LOCAL_AXES + `plot.energy` / `plot.node_envelope` + dim-based plot facets** · **static gauss contours (`plot.contour(topology="gauss", averaging="averaged"|"discrete")`) + `plot.fibers` dot cloud** · **local-axes overlay triads resolve recorder-first (parity with the diagram frames)** · **partitioned-deck `getPID` shim guards with `info commands` (every MPI rank built rank 0's submodel)** · **partitioned emit: shared-node `mass` / pattern `load` lines dedup to the node's primary rank (OpenSeesMP sums them — interface nodes carried 2–3× mass)** · **Ladruno recorder whole-model energy channel (`ops.recorder.Ladruno(energy=True)` → `-G energy`, emitted last)** · **deform-follow regression fixed — contour / fiber-section / layer-stack / spring-force diagrams ride the deformed substrate again (dead `_sync_layer_grids` walk removed)** · **declarative diagram-kind registry (ADR 0058 S0) — four drifting per-kind tables collapse into `@register_diagram_kind`; loads/reactions survive session restore + presets; reactions catalog options un-shadowed** · **geometry→scene resolution seam (ADR 0058 S1) — `director.scene_for(geometry)` + registry `scene_resolver`, per-geometry DEFORM pump + scoped fan-out, `reference_points` moves onto `FEMSceneData`; copy cost measured (~7 MB / 2 ms at 124k cells → plain copies for S2, no COW)** · **absorbing-boundary guide (`internal_docs/guide_absorbing_boundary.md`)** · **remote HPC job submission (`apeGmsh.hpc` — `Cluster.submit`/`Job` over SSH + SLURM, ADR 0060)** · **`ops.run_remote` one-call remote analysis + `Job.wait` (ADR 0060 sugar)** · **coupling control knobs — `g.constraints.kinematic_coupling` / `distributing_coupling` accept `k` / `kr` / `enforce` / `bipenalty_dtcr` / `absolute` (`CouplingControl`, neutral schema 2.12.0)** · **coupling-knob H5 schema completion — `sr_cpl_*` mirror lane on `surface_coupling` + dtype/parity test reconciliation (post-#630 main fix)** · **partitioned staged H5 archival — last staged guard lifted, rank-agnostic stage capture (ADR 0055 Phase 5 / P5.1, schema 2.19.0)** · **staged `domainChange` is unconditional — pure-loading stages no longer merge into the previous `MODEL_STAGE` in the MPCO/Ladruno recorders (+ numeric stage-stamp ordering in the readers + viewer positional stage pairing)** · **RBE3 tributary-area weighting — `distributing_coupling(weighting="area")` computes per-independent areas and emits `-w`** · **RBE2 partitioned (OpenSeesMP) emit — single-canonical-rank routing for `kinematic_coupling` (was fail-loud)** · **docs: `guide_constraints.md` coupling sections reconciled (fork RBE2/RBE3 emit targets, knobs, area weighting, mortar refusal)** · **per-rank Tcl deck emission — driver + `ranks/rank<K>_<seq>.tcl` sourced fragments (`apeSees.tcl(per_rank=True)`, ADR 0061)** · **ADR 0055 ACCEPTED — compose filtered-audit (`compose_inspect`['filtered']) + real-staged-archive FILTER verification (Phase 3); staged-H5 runway complete** · **partitioned staged flat replay + domain-capture gate retired (ADR 0055 Phase 5 / P5.2 + P5.3)** · **coupling host auto-scalers (`k="auto"` / `k_alpha` / `host` / `bipenalty_wcap`)** · **concurrent geometry rendering — per-geometry `visible` flag (ADR 0058 S2b)**

<!-- ⚓ NEW ENTRIES GO DIRECTLY BELOW THIS COMMENT (newest first).
     Insert ONE contiguous "### ADDED/FIXED/CHANGED — ..." section per PR.
     Do NOT edit any existing line — in particular the single-line
     "## Unreleased — ..." ledger above is FROZEN (your section title is
     the highlight now). CHANGELOG.md merges with the union driver
     (.gitattributes), which silently keeps BOTH sides of any edit to an
     existing line instead of conflicting — duplicated-header mangling is
     guarded by tests/test_changelog_structure.py.
     Workflow + rationale: internal_docs/changelog_workflow.md -->

### ADDED — parallel modal Tier-1 P3: mode-shape harvest for the distributed-FEAST modal deck (ADR 0077)

`apeSees.modal_deck` now harvests **mode shapes**, merge-free (the replicated deck puts ALL nodes on every rank, so an ordinary rank-0 recorder carries the full field). The emit (`TclEmitter.eigen_feast_parallel(shape_nodes=, shape_ndf=)`) appends a rank-0-guarded block **after** the captured solve: a `mode_shapes.json` sidecar pinning the node→column map (sorted mesh node tags × the envelope-`ndf` dof list — the headerless `.out` rows become self-describing without the deck), then one `recorder Node -file mode_shape_<k>.out … "eigen k"` per **found** mode, a single `record` trigger (recorders never fire on their own — no analyze step runs in this deck), and `remove recorders` to close the files. Recorder creation is post-solve **by necessity and by design**: the band's mode count is dynamic (`llength $_lam`), recording an unfound mode corrupts the row (`NodeRecorder::record` skips a node whose eigenvector matrix lacks the column *without advancing its write cursor*), and the source shows post-solve creation is sound (the eigen dataFlag reads `Node::getEigenvectors()` only at record time; `Domain::addRecorder` does not auto-fire); DOFs a node does not carry are cursor-safe `0.0` padding. `ParallelModalResult.from_job` reads the sidecar + per-mode rows when present (a pre-P3 run dir still harvests eigenvalues; the shape accessors then fail loud), surfacing `mode_shape(node, mode)` (length-`ndf`, the `EigenResult` convention), `mode_shape_field(mode)` (`(n_nodes, ndf)`), and `shape_nodes`. **Live-verified** on the two-column frame (fork classic-Tcl `-feast` build; serial `OpenSees` + hydra `mpiexec -n 2 OpenSeesMP`, `LADRUNO_FEAST_MPI` rank-0/1 proof): a degeneracy-broken variant (distinct tip masses 100/120/140/160 → 4 distinct modes 97.46–123.28 Hz, λ rel err 2.3e-8 vs analytic `6e7/m`) gives per-mode **MAC = 1.0** (9 decimals) distributed-vs-serial **and** distributed-vs-live-openseespy plain-`eigen` oracle; the stock frame's exactly-degenerate 4-fold subspace matches with principal-angle cosines all 1.0 (1-to-1 MAC is basis-dependent under exact degeneracy). Deck-text + reader tests extended to 16 (`test_modal_deck_parallel_feast.py`, `test_parallel_modal_result.py`).

### CHANGED — parallel modal Tier-1 P2 live-verified; `modal_deck` corrected to a REPLICATED deck (ADR 0077)

The first live distributed run (fork classic-Tcl `-feast` build, PR nmorabowen/OpenSees#578, `mpiexec -n 2 OpenSeesMP.exe`) **refuted the P1 deck shape**: the fork's L3 FEAST requires every rank to assemble the **full** model (the RCI kernel slices the full 2n block system's triplets across ranks for the distributed dmumps — distribution lives inside the kernel, not in domain decomposition), so the partitioned `if {[getPID]==K}` deck failed `FeastEigenSOE::setSize — vertex not in graph`. `apeSees.modal_deck` now emits the model **flat/replicated** (partitions on the fem are ignored via the `supports_partitions = False` seam; RAM trade-off = full model per rank, the documented L3 regime) with a deterministic `constraints Transformation` / `numberer RCM` / `system UmfPack` preamble + the `getPID` shim; the `per_rank=` kwarg is gone (no partition spans exist). **Also corrected: the deck's `system` line is NOT part of the FEAST solve path** — the earlier "`system Mumps` is load-bearing" claim was a carry-over from the refuted plain-eigen design; a serial-`UmfPack` deck produced kernel-verified distributed solves on both ranks. End-to-end validation: apeGmsh-emitted `modal_deck(band=(0,200), certify=True)` → `mpiexec -n 2` → `ParallelModalResult.from_job` → 4 modes @ 123.280887 Hz, max rel err vs the analytic tip-mass oracle **9.7e-16**, with `LADRUNO_FEAST_MPI` distribution proof on both ranks. Mode-shape harvest (P3) is now **merge-free** (every rank holds all nodes → rank-0 recorder). Deck-text tests rewritten to pin the flat shape (no partition blocks even for a partition-authored fem). ADR 0077 updated in place (Tier-1 section, INV-1/3/4, P1/P2/P3).

### ADDED — parallel modal analysis Tier-1 P4 (partial): `ParallelModalResult` eigenvalue harvest (ADR 0077)

`ParallelModalResult` (`apeGmsh.opensees.analysis.ParallelModalResult`) is the eager result surface for a distributed-FEAST modal run: `eigenvalues` (a band output — the count is dynamic) + derived `omega` / `freq` / `periods` + `n_modes` + a `certified` flag. `ParallelModalResult.from_job(job_dir, out="eigenvalues.out")` harvests the eigenvalues from a completed `modal_deck` run dir — the rank-0 write-out format is pinned by `TclEmitter.eigen_feast_parallel` (a single whitespace-separated line of `λ = ω²`), so the reader is deterministic and verifiable without a live distributed run; a missing file raises `FileNotFoundError`. `participation_factors(...)` / `mass_ratios` raise `NotImplementedError` redirecting to the single-process `modal_properties` (upstream `modalProperties` is MPI-blind under partitioning; ADR 0077 INV-2), and `mode_shape(...)` raises pending ADR 0077 P3 (distributed mode-shape harvest — deferred until the fork classic-Tcl `-feast` build lets the eigenvector-recorder format be verified live). Locked by 6 unit cases (`tests/opensees/unit/test_parallel_modal_result.py`): whitespace-tolerant parse + derived quantities, `certified` passthrough, empty-band zero modes, missing-file raise, and the two fail-loud accessors.

### ADDED — parallel modal analysis Tier-1 P1: `apeSees.modal_deck` distributed-FEAST emit (ADR 0077)

`apeSees.modal_deck(path, *, band=(f_min, f_max), certify=False, target="tcl", per_rank=False, out="eigenvalues.out")` emits a partitioned Tcl deck that runs band-targeted distributed FEAST under `OpenSeesMP`. The partitioned emit already lays down the eigen preamble (`numberer ParallelPlain` / `system Mumps`, each with a single-process fallback; `system Mumps` is load-bearing — a serial system silently degrades FEAST to a per-rank local solve), so the driver appends only a **single captured** `set _lam [eigen -feast $f_min $f_max -rci [-certify]]` plus a rank-0 eigenvalue write-out (via new `TclEmitter.eigen_feast_parallel`) — never a second `[eigen …]` call (which would re-run the distributed solve and deadlock on a rank-0-only collective; ADR 0077 INV-5). `modalProperties` is **not** emitted (MPI-blind upstream → wrong effective mass; INV-2). Fails loud on `target="pymp"` (unlock 2a, not implemented), non-partitioned models (use single-process `eigen`/`modal_properties`, Tier 0), and staged models; `per_rank=True` splits per ADR 0061. The band (Hz) defines the mode count — there is no `num_modes`. **This is emit-only:** the live distributed run needs the fork classic-Tcl `-feast` parity build (ADR 0077 unlock 2b); mode-shape harvest + the `ParallelModalResult` surface are P3/P4 (deferred until the recorder format can be verified live). Locked by 6 deck-text cases (`tests/opensees/integration/test_modal_deck_parallel_feast.py`): captured solve emitted exactly once, preamble present, write-out block, `modalProperties` absent, and the three fail-loud guards.

### ADDED — parallel modal analysis ADR 0077 (Proposed) + Tier-0 serial-gather stopgap (P0)

ADR 0077 designs modal analysis on the partitioned / HPC path, and ships its first tier. **Tier 0 (this PR):** on a partition-authored model, `apeSees.eigen` / `modal_properties` already build the **full, gathered** model in one process (the live emitter's `supports_partitions = False`) and run stock serial ARPACK — so the modes (and, for `modal_properties`, participation factors / effective modal mass) are **exact**. It does not scale the eigensolve (whole model on one rank), documented as such in both docstrings. Pinned by a live regression that partitioned-serial eigen is **bit-identical** to the unpartitioned solve and `modal_properties` resolves participation on a partitioned model (`tests/opensees/live/test_eigen_partitioned_serial_gather.py`). **Tier 1 (distributed FEAST `eigen -feast … -rci`, deferred):** the ADR records that an originally-proposed "plain `eigen` over `MumpsParallelSOE` under `OpenSeesMP.exe`" path was **REFUTED by adversarial review** — under `_PARALLEL_INTERPRETERS` the `eigen` `ArpackSOE` gets no `setProcessID`/`setChannels` (parallel-eigen wiring is `_PARALLEL_PROCESSING`-only) so the `M*v` reduction stays local → per-rank-local garbage modes (the exact SP/MP non-composition the fork's FEAST line was built to resolve, and which `apesees.py:4285` already fails loud on for `ops.damping.modal`). The correct distributed path is fork FEAST (fork ADR 43, L3-only), reachable only once the classic-Tcl `-feast` parse gap is unlocked (fork `commands.cpp` parity or a PyMP `.py` deck); `modalProperties` stays MPI-blind and is deferred + raised-on in any distributed run. Design + phased plan (P0 done; P1–P5 gated on the unlock) in `architecture/decisions/0077-parallel-modal-analysis.md` with a full adversarial-review appendix.

### ADDED — custom-scalar `mag()` vector helper + definition persistence (ADR 0076 Slice 4, Accepted)

The `mag(<vector>)` special form and definition persistence land on the user-defined scalar feature (ADR 0076 now **Accepted**). `results.nodes.define("speed", "mag(velocity)")` expands a nodal **vector family** (`velocity` / `displacement` / `acceleration` / `force` / `rotation` / `moment` / … via new `_vocabulary.vector_family`) to the Euclidean norm of its *recorded* components — `sqrt(vx**2 + vy**2 + vz**2)`, clipping to present axes so a 2-D model yields the in-plane speed with no zero-fill. It refuses a component name (`mag(velocity_x)`), the combined `reaction` shorthand (forces+moments is not one vector), and a non-bare argument (`mag(velocity + 1)`). **Persistence:** `results.save_definitions()` writes `<results>.defs.json`; `Results.from_native` / `from_mpco` / `from_ladruno` auto-load it at open (best-effort — a stale operand from a richer run warns and is skipped, never blocks the open; loading is idempotent). This is the **same** sidecar as the Slice-3 viewer transport (unified — the subprocess launch now routes through `save_definitions`, so it persists too) and deliberately **not** an embedded H5 zone, so it works for read-only `.mpco` as well as native files. Definitions also **follow stage / mode derivation** — a scalar defined on `results` is readable on `results.stage("grav")` (`_derive` shallow-copies the frozen `ExprDef` registry). Locked by 12 new cases in `tests/test_custom_scalar_expressions.py` (mag engine 3-D/2-D + 5 rejections + composite; save→reopen auto-load, idempotency, stale-operand warn-not-fatal, stage propagation, missing-file no-op).

### ADDED — user-defined scalar expressions on Results (`results.nodes.define` / `results.elements.gauss.define`, ADR 0076)

Users can register a custom scalar as a restricted arithmetic expression over a composite's own `available_components()` and read it anywhere a `component=` is accepted — including the viewer picker. `results.nodes.define("kinetic_ish", "velocity_x**2 + velocity_y**2 + displacement_x")` and `results.elements.gauss.define("dcr", "von_mises_stress / 250.0", units="-")` become first-class components: `get(component=…)` computes them on read (sibling of the derived-scalar path, `_compute_derived`), and they surface in `available_components()` / `definitions` / the Add-Diagram picker with zero viewer-side special-casing. Registration is **per-composite** so the domain is implied — a node expression can't reference a Gauss field. New `results/_expr.py` parses the expression to a **restricted AST over numpy** (never `eval`): operators `+ - * / // % ** & |`, comparisons, numeric literals, the operand names, and a fixed elementwise function table (`sqrt abs exp log sin cos tan sign hypot minimum maximum clip where`). Adversarial-review hardening folded into ADR 0076: `and`/`or`/`if-else` are **rejected, not lowered** (they dispatch on array truthiness → ambiguous-truth `ValueError`; use `where(cond,a,b)` and `& / |`); `min`/`max` are **not** exposed (numpy's are field-collapsing reductions, Python's are variadic) — only the 2-arg elementwise `minimum`/`maximum`. Fail-loud at `define()` for parse errors, unknown operands (validated against the **union of all stages**), name shadowing (stored / derived / already-registered), and zero-operand expressions; the two inherently read-time failures — an operand absent from the *specific* requested stage, and operand coverage mismatch across independently-recorded fields — raise a legible `ExprError` at read rather than a raw numpy broadcast error. `undefine()` refuses to strand a dependent. Definitions reach the subprocess viewer through a **dedicated** `<results>.defs.json` sidecar + `--defs` arg (never the diagram `viewer-session.json`, which is snapshot-gated and overwritten on close); the in-process `viewer()` / `show_web()` share the live registries directly. Scalar-only operands, session-lifetime registration, and no unit/NaN policing are explicit v1 boundaries (`mag()` vector helper + H5 persistence deferred). Locked by `tests/test_custom_scalar_expressions.py` (34 cases: engine grammar/rejection/shape-guard, node + Gauss compose-on-derived + custom-on-custom + shadow/undefine guards, payload round-trip + argv `--defs` + `__main__` apply/malformed-survival).

### FIXED — ADR 0075 adversarial-review hardening (36-agent gate over the modal-family stack)

A 6-lens find + 3-skeptic-per-finding verify workflow over the full slice 1–5 diff confirmed 9 findings (1 refuted); all are fixed here. **Majors**: (1) negative damping ratios now refused in `_damping_channel_args` (`damp < 0`, any `modal_damp` entry `< 0`; `ξ = 0` stays legal) — the fork refuses `ξ < 0` on four of five family parsers but NOT on `responseSpectrumAnalysis`, where a single typo'd sign in a CQC `modal_damp` list makes `ρ_ij = √(ξ_i·ξ_j)` NaN and the combination kernel's `sqrt(s > 0 ? s : 0)` silently commits an **all-zero design displacement field**; (2) wrong-kind object handles are now `TypeError`-refused at all four excitation resolution sites (`base_accel=`/`series=`/`load=`/`input_psd=`) — `_resolve` kind-checks only name-string refs, and per-kind 1-based tag counters meant e.g. a Plain pattern passed as `base_accel=` emitted a numerically-colliding **timeSeries** tag and ran a plausible-looking transient with the wrong ground motion; (3) the two modal-history live equality oracles moved from dt=2e-4 to dt=5e-5 (n=1600) — two independent verifier rebuilds of the fork's exact PWL recurrence vs Newmark showed the **reference's own O(dt²) error is ~1.15 % at the old dt, above the 1 % assert**, so both tests would have failed deterministically against correct code the moment the fork rebuild unskips them. **Minors**: RSA now accepts a leading `T = 0` PGA-anchored spectrum (the fork refuses only negative `Tn` and clamps `T ≤ Tn[0]` to `Sa[0]`; the old message misattributed the stricter check to "the OpenSees -Tn contract" — this supersedes the slice-2 section's "positive strictly-increasing" wording); `ModalPropertiesResult` documents the `unorm=True` basis mismatch (participation factors live in the displacement-normalized basis, `mode_shape` always returns the RAW domain eigenvector — `Γ·φ` products are scale-consistent only under the default `unorm=False`); ADR 0075 gains the classic-Tcl `-feast` parse-gap caveat, scopes the `-out` passthrough to the three sweep drivers (`complex_eigen` has none), and corrects the damping contract to at-most-one-of with the RSA exception (optional channel, no `rayleigh=`). Locked by 7 new unit cases (negative/mixed-sign/zero damping, 3 wrong-kind-handle refusals, negative-period) + a live T=0-anchor acceptance test.

### ADDED — complex/state-space modal analysis: `apeSees.complex_eigen` + `ComplexEigenResult` (ADR 0075, slice 5 of 5 — family complete)

The fork ADR-46 `complexEigen` lands as a live-only tier-2 driver — the answer to "what is the damping ratio of the isolation mode?" for **non-classically damped** models (localized dashpots, bearings, radiation damping). `apeSees.complex_eigen(num_modes, *, solver=, tol=None, closed_form=False)` builds a fresh live domain, runs the real `eigen` projection basis, then `complexEigen`, and parses the fork's flat 7-per-mode return into a frozen `ComplexEigenResult` (`omega0`/`omega_d`/`zeta`/complex `lam`/`kind` 0-under/1-over/2-rigid/`resid` quality metric + `freq_d`). Default = the assembled Route B projection (element-by-element `getDamp()`/`getMass()` — exactly the C a transient feels: scoped `region -rayleigh`, `betaKinit/betaKcomm`, material dampers, `-doRayleigh` switches all honored); `closed_form=True` = the fast global-Rayleigh diagonal Route A. Contract traps documented on the driver (fork guide): `getDamp()`-invisible damping (`modalDamping`, HHT-α numerical, `-doRayleigh`-default-OFF `Truss`/`zeroLength` families) is invisible; complex mode shapes are recorded via the existing Node-recorder `raw=("complexEigenRe<k>",)`/`Im<k>` escape hatch, not carried on the result; a declared `constraints Plain` triggers a `UserWarning` (MP-constrained models need a distributing handler for the shape push — guide trap #4). Live gated on the `complexEigen` attribute with the friendly fork-required error. Locked by from_flat parsing pins + validation cases + the Plain-handler warning pin, and live oracles (`tests/opensees/live/test_complex_eigen_live.py`): classical-Rayleigh `ζ_k = a0/(2ω_k)+a1·ω_k/2` and `ω_d = ω₀√(1−ζ²)` on **both** routes at 1e-6, `ω₀` echoes the real eigen basis, friendly error pinned on the deployed pre-ADR-46 build. **This closes the ADR 0075 five-slice runway.**

### ADDED — FEAST band-targeted eigen: `apeSees.eigen_feast` (ADR 0075, slice 4 of 5)

The fork ADR-43 band-targeted FEAST eigensolver joins the Emitter Protocol as a **separate method** `eigen_feast(f_min, f_max, *, certify=False)` — `eigen -feast $fmin $fmax [-certify]` returns **all** modes with natural frequency in `[f_min, f_max]` Hz, so the mode count is an *output* (`len(result.eigenvalues)`), which breaks `eigen`'s `num_modes` contract and rules out a solver-flag overload. `certify=True` adds the fork's Sturm/inertia completeness certificate (LDLᵀ inertia at the band edges; the solve refuses on a count mismatch with FEAST). The bridge driver reuses `EigenResult` unchanged (lazy `mode_shape`, possibly zero modes on an empty band) and — because the stock `ops.eigen` symbol exists on every build so a missing-attribute gate can't fire — pre-checks `capabilities().has_fork` for the friendly fork-required message (a pre-ADR-43 fork build still fails loud with the OpenSees `-feast` parse error). Deck note: the fork wires `-feast` into the interpreter/openseespy parser only — classic `OpenSees.exe`/`OpenSeesMP.exe` decks do not parse it yet, so the deck target is openseespy decks (documented in the Protocol + ADR 0075). Live oracle (`tests/opensees/live/test_eigen_feast_live.py`): band solve == `-fullGenLapack` spectrum filtered to the band, with `-certify` on; skips on pre-modal-family builds via the `modalResponseHistory` proxy (ADR-43 ships before ADR-44 on the fork timeline).

### ADDED — frequency-domain sweep drivers: `apeSees.frequency_response` / `steady_state_dynamics` / `random_response` (ADR 0075, slice 3 of 5)

The fork ADR-44 sweep commands land as **live-only bridge drivers** (ADR 0075 tier 2 — the value IS the interpreter return, so no Emitter-Protocol widening; the live emitter gates each on its own fork-only openseespy attribute with the friendly fork-required error). All three share a validated marshaller: `f_min/f_max/n_freq` band, `grid="lin"|"log"|"biased"` (biased clusters points around in-band modal peaks), excitation XOR (`base_accel_dir=` — harmonic base acceleration, no timeSeries, relative response — vs `load=` nodal-force pattern with the fork-refusal pre-checks), the exactly-one-of damping channel, `node/dof/resp("disp"|"vel"|"accel")/modes`, and `out=` ASCII passthrough; each auto-issues `eigen` + `modalProperties` on a fresh live domain. `frequency_response` returns an **eager** `FrequencyResponseResult` (Hz grid + complex FRF + `magnitude`/`phase`; `e^{+iΩt}` sign convention); `steady_state_dynamics` a `SteadyStateResult`; `random_response` (PSD→RMS, ADR-44 P3) takes the **one-sided G(f)-in-Hz** `input_psd=` timeSeries (defaults `grid="biased"` per the guide — a linear grid mis-integrates sharp resonances), optional `stats=`/`duration=` (Davenport expected peak), and returns a `RandomResponseResult` normalizing the fork's scalar-vs-list return shapes. Locked by 9 new validation cases + live SDOF closed-form oracles (`tests/opensees/live/test_modal_sweeps_live.py`): FRF magnitude AND phase-sign vs `H(Ω) = −1/(ω²−Ω²+2iξωΩ)`, load-channel static limit `P/k`, SSD == |FRF|, white-noise anchor `σ_x²=G0/(8ξω³)` at 2 % (a √2 miss flags a one-/two-sided mixup), stats/duration shape + `ν₀≈f_n`, zero-damped in-band refusal, and the friendly fork error pinned on pre-ADR-44 builds (the currently-exercised path; oracles unskip after a fork rebuild).

### ADDED — modal-response committing commands: `apeSees.modal_response_history` + `response_spectrum_analysis -combine` (ADR 0075, slice 2 of 5)

The two fork ADR-44 commands that **commit domain state** join the Emitter Protocol (deck-emitting — Tcl/py write the single command line; H5 no-ops; recording captures) and get bridge drivers. `apeSees.modal_response_history(dt=, n_steps=, num_modes=, …)` runs the fork's exact piecewise-linear modal-superposition transient (`modalResponseHistory`, Ladruno ADR-44 P1a): auto-issues `eigen` + `modalProperties` on a fresh live domain, commits one step per station so **ordinary recorders capture the history**, and returns a `ModalHistoryResult` (mode basis + final-station `node_disp/vel/accel`). `apeSees.response_spectrum_analysis(direction, periods=, accels=, combine=, num_modes=, …)` runs the list-form RSA with the fork's native combination stage (`-combine SRSS|CQC|ABS|TenPercent`, ADR-44 P1b) and returns a `ResponseSpectrumResult` whose `node_disp` reads the committed **combined design displacement field** (per-quantity combination caveat documented — never derive element forces from combined displacements). Excitation channels resolve dual-mode handles (`base_accel=`+`direction=` ground motion, or `load=`+`series=` nodal-force pattern — fork-refused pattern contents (sp constraints, moment tensors) pre-checked at the bridge with a friendly `BridgeError`); damping is the explicit exactly-one-of channel (`damp=`/`rayleigh=`/`modal_damp=`); CQC requires damping; `periods` must be positive strictly-increasing; staged models refused. **Live gating rides the `modalResponseHistory` attribute probe for BOTH commands** — the upstream `responseSpectrumAnalysis` parser silently ignores unknown flags, so an ungated `-combine` on a pre-ADR-44 build would commit per-mode displacements with no combination. Locked by a 15-case bridge validation matrix + emit-text pins + live oracles (`tests/opensees/live/test_modal_response_live.py`): modal transient == direct Newmark under **alphaM-only** Rayleigh on both excitation channels, SRSS == numpy `Γ·Sa/ω²·φ` hand-oracle, and the friendly fork-required error pinned on pre-ADR-44 builds (which is what the currently-deployed 2026-06-25 build exercises; the equality oracles unskip after a fork rebuild).

### ADDED — `modalProperties` surface: `apeSees.modal_properties` + `ModalPropertiesResult` (ADR 0075, slice 1 of 5)

First slice of the Ladruno modal-family consumption (ADR 0075; fork ADRs 43/44/45/46). The Emitter Protocol gains `modal_properties(*, unorm=False, out=None)` — upstream OpenSees `modalProperties` (Petracca's `DomainModalProperties`), the prerequisite state for every fork modal-response command: the live emitter passes `-return` and hands back the properties dict; Tcl/py emit `modalProperties [-unorm] [-file $out]`; H5 no-ops (runtime retrieval, no schema bump); recording captures. A new bridge driver `apeSees.modal_properties(num_modes, *, solver=..., unorm=False)` runs `eigen` → `modalProperties -return` on a fresh live domain (no analysis chain needed, staged models refused) and returns a frozen `ModalPropertiesResult`: eigenvalues + derived `omega`/`freq`/`periods`, the raw properties dict, component-keyed accessors (`participation_factors("MX")`, `mass_ratios`, `cumulative_mass_ratios` — percent, components `MX/MY/MZ/RMX/RMY/RMZ`, 2-D: `MX/MY/RMZ`), `total_mass` / `center_of_mass`, and the lazy `mode_shape(node, mode)` reader (EigenResult staleness contract). Also lands the shared `_damping_channel_args` exactly-one-of validator (`damp=` | `rayleigh=(a0,a1)` | `modal_damp=[ξ…]` → verbatim fork flags) for the upcoming ADR-44 drivers, and a pin that a registered **standalone** timeSeries emits without any pattern referencing it (the `-baseAccel`/`-inputPSD` excitation channels rely on this). Locked by `tests/opensees/unit/test_apesees_modal_validation.py` + emitter emit-text pins + `tests/opensees/live/test_modal_properties_live.py` (tip-mass cantilever: ~100 % MX mass in mode 1, `Γ₁·φ_tip,x = 1` hand identity — runs on stock openseespy, no fork marker).
### FIXED — `LadrunoUP` (ADR 0074) adversarial-review hardening: DOF-aliasing guard, etype legality, solver-gate rescope, replay bracket, silent-capture warning

Ten findings from an adversarial review of the LadrunoUP emission runway, most closing **silent-wrong-results** paths the count-based gates missed. **Rotation-vs-pressure DOF aliasing (new guard)** — a 2-D frame element (`elasticBeamColumn` &c., floor `ndm+1`) sharing a saturated equal-order LadrunoUP carrier node put its *rotation* DOF in the *pore-pressure* slot (`ndm+1`); both require `ndf=ndm+1`, so `infer_node_ndf` and the disjoint-set guard both passed and the fork setDomain checks only the DOF count — the run assembled bending stiffness into the pressure row with `rc=0`. `validate_ladruno_up_pressure_dof` now fails loud (pure-translation neighbours like trusses, and TH mid-edge shares, are correctly untouched) with the ADR-0069 separate-node fix. **Shape legality is now by Gmsh ETYPE, not node count** — an 8-node serendipity `quad8` surface in a 3-D model (or a 3-node `line3` curve in 2-D) aliased a legal count (`8`≈H8 / `3`≈T3) and emitted a degenerate element that singularized at run; the legality pass reads each element-group's true etype (also vectorized — the per-element straight-side loop became column-slice array ops, ~24× faster at 100k tet10, and equal-order pgs skip per-row boxing). **Solver gate (D4) rescoped** to what the deck emits-and-solves: a *declared* symmetric/diagonal system is still refused unconditionally (a model-only Tcl export catches `ProfileSPD`), but a *missing* system only raises for solve-bearing emits — so H5 archival, eigen-only, and model-only-skeleton exports are no longer falsely refused; staged decks validate each stage's own system (a `system=None` stage that analyzes now raises; a stray never-emitted global system no longer false-rejects); a partitioned deck with no system rides the ADR-0027 auto-emitted general Mumps/UmfPack. **`body=` bypassed the double-count guard** — LadrunoUP names its always-on self-weight `body` (accelerations), so `validate_body_force_double_count` (which grepped `body_force`) never warned on a `from_model` gravity overlap; it now reads either attribute (direction-only collinearity, so the unit difference is moot). **H5 replay lost the builder bracket** — the compose / `from_h5` replay re-emitted element lines after one global `model(ndm,ndf)`, so a mixed-envelope equal-order LadrunoUP archive replayed into a deck the fork parser refuses (`ndf != ndm+1`); replay now brackets gated element runs (coalesced, envelope-restored) in both the flat and staged paths. **Gauss capture no longer silently drops** — `LadrunoUP` is `has_gauss=True` with no `RESPONSE_CATALOG` layout, so a gauss recorder on a u-p pg captured nothing with zero signal; `DomainCapture` now emits a consolidated warning (mirroring the line-station path) steering the user to the node / `.ladruno` pressure channels. **Per-TYPE live fork verification** — the fork-only-element gate keyed on a single boolean, so a build knowing `LadrunoQuad` would wave an unknown `LadrunoUP` through unverified; it now tracks verified types in a set. **Single-source shape tables** — the shape / mid-edge / builder-ndf tables (previously duplicated across `element/solid.py`, `_internal/build.py`, and the registry) collapse into `_element_capabilities` (mid-edge order cross-referenced to `apeGmsh._basis`); `_BUILDER_NDF_GATED["LadrunoUP"]` references the registry's `ndf_required` instead of a second copy. New coverage: `tests/opensees/unit/test_ladruno_up_replay_and_forkgate.py` (replay bracket + per-type gate) and `tests/results/test_gauss_skip_warning.py`, plus DOF-aliasing / etype-legality / rescoped-solver cases in `test_ladruno_up_build_gates.py` and a model-only-export-allowed case + `body`-double-count case in the integration suites.

### ADDED — `LadrunoUP` Biot u-p porous element emission (ADR 0074): typed class, per-slot ndf inference, straight-side + solver build gates

The fork's unified saturated-porous continuum (`element LadrunoUP`, ELE 33017 — honest pore-pressure DOF at slot ndm+1) enters the normal mesh→snapshot→deck pipeline. **D1 — one typed class, shape from the mesh**: `ops.element.LadrunoUP(pg=, material=, Kf=, poro=, rhoF=, perm=|permH=+gammaW=, thick=, alpha=, Ks=, body=, fluidBody=, formulation=, lumped=, stab=, dynSeepage=, geom=)` fans tri3→T3 / quad4→Q4 / hexa8→H8 (equal-order) and tri6→Bézier T6 / tet10→Bézier Tet10 (Taylor–Hood; `-pOrder linear` appended automatically — the user never types it). Kwarg policy is pass-through: `None` is never emitted, the fork parser's defaults stay the single source of truth; construction mirrors the parser's fatality matrix (perm XOR permH/gammaW pairing, 0<n≤α≤1 storage police, thick-is-2D-only, the `-stab` grammar incl. `('auto', α0)`, dynSeepage on/off, geom='linear' only). **D2 — heterogeneous intra-element ndf**: `_ElemSpec` gains `ndf_floor_per_slot` (keyed by fan-out node count) and `infer_node_ndf` resolves per node-slot — TH vertices carry ndf=ndm+1, mid-edge nodes ndf=ndm, validated STRICTLY ({floor} ok-sets; a beam grabbing a mid-edge node fails loud) — so the hand-written two-step `model(ndf=…)` dance disappears into inference and the `-ndf` tokens emit/elide against either envelope choice. Fully additive: every existing element keeps the scalar path. The equal-order builder gate (parser demands builder ndf=ndm+1) rides `_BUILDER_NDF_GATED` with a new ndm-dependent form (`element_builder_ndf(cls, ndm)`). **D3 — straight-side pre-validation**: tri6/tet10 pgs are checked mid-edge-vs-midpoint at the fork's own 1e-6·edge tolerance at build, converting the fork's setDomain deactivate-and-singularize (a cryptic `analyze()` failure from openseespy) into a `BridgeError` naming element/node/offset + the Gmsh high-order-optimization hint. **D4 — solver build gate**: a LadrunoUP deck whose `system` is missing or symmetric-storage/diagonal ({Band,Profile,SProfile,ParallelProfile}SPD, SparseSYM, (MPI)Diagonal) refuses to build — the honest-p tangent is unsymmetric and the no-`system` default ProfileSPD silently drops one coupling block (measured p≈1e88 with rc=0, fork guide §2); allow-list UmfPack/SparseGeneral/FullGeneral/BandGeneral/Mumps, checked per stage on staged decks (wipeAnalysis re-defaults each stage), deliberately no escape hatch. Also: `stab=` on a TH pg and non-provider cells (tet4/prism) fail loud with pg context. **D5**: `LadrunoUP` joins `_FORK_ONLY_ELEMENTS` (deck emission anywhere; live run fails loud on stock builds) and `ops.capabilities()` gains `has_ladruno_up`; persistence rides the standard bridge-zone payload (no schema change). **D6**: fix/sp/mass/load records already validate against the RESOLVED per-node ndf — locked for TH meshes (a 3-slot mask on a 2-dof mid-edge node fails loud; the leading-u mask passes) — and the BC/recorder idioms (drained slot, staged-head Penalty-not-Transformation, NormDispIncr, disp-channel p read-back) are documented on the class. Locked by `tests/opensees/unit/primitives/test_elements_ladruno_up.py` (construction matrix + kwarg→token mapping + capability surface), `tests/opensees/unit/test_ladruno_up_build_gates.py` (per-slot inference, legality/straight-side pass, solver gate), and `tests/opensees/integration/test_ladruno_up_emission.py` (real Gmsh Q4/tri6/tet10 decks — auto `-pOrder`, `-ndf` emission/elision both envelopes, builder bracket, curved-disk refusal, G3 mask ergonomics, plus a fork-gated live mini-Terzaghi with monotone base-p decay).

### ADDED — scalar-bar orientation + size control (results-viewer color bar): style knobs, live setters, settings-tab controls

The diagram scalar bar (color legend) is now user-controllable end to end. **Style**: every scalar-bar-bearing diagram style (`ContourStyle`, `VectorGlyphStyle`, `GaussMarkerStyle`, `SandStyle`, `FiberSectionStyle`, `LayerStackStyle`) gains `scalar_bar_vertical: Optional[bool] = None` (bar orientation; `None` = the viewer theme's default, horizontal) and `scalar_bar_scale: float = 1.0` (on-screen size multiplier). **Runtime**: `ScalarBarSupport` gains `set_scalar_bar_vertical()` / `set_scalar_bar_scale()` live setters following the `set_fmt` runtime-override pattern, plus a shared `_make_scalar_bar_spec()` builder that replaces the seven copy-pasted inline `ScalarBarSpec` constructions — so orientation/size/fmt overrides now all survive show toggles AND colormap changes on every diagram. **IR + backend**: `ScalarBarSpec` gains `vertical`/`size`; `PyVistaQtBackend` derives width/height from the theme's per-orientation base dimensions × `size` (clamped to the viewport). **Widget fix folded in**: pyvista's `interactive=True` `vtkScalarBarWidget` re-derives its own geometry on enable — vertical bars snapped to the representation's stock 0.17 × 0.8 box regardless of the requested size — so the backend now re-asserts the requested layout on both the actor and the widget representation after creation; the bar stays freely draggable/resizable in the scene afterwards. **UI**: the settings-tab scalar-bar block (shared by contour / fiber / layer / gauss / vector-glyph / sand cards) gains an Orientation combo (Horizontal/Vertical) and a Size × spinner (0.2–4.0), staged via Apply like Show/Format. Locked by 4 new tests in `tests/viewers/test_contour_scalar_bar.py` (live orientation flip, size resize + viewport clamp, overrides survive show toggle, style knobs apply at attach).

### ADDED — streamed run logging + live progress for `ops.tcl(run=True)` / `ops.py(run=True)`

The two subprocess run paths previously called `subprocess.run(capture_output=True)`, which **swallowed all OpenSees terminal output on success** and, on a non-zero exit, dumped the *entire* stdout+stderr into one `RuntimeError` string — no log file, no "it started", no progress. Both now stream through a new `apeGmsh.opensees._run.stream_run()` helper (`Popen`, line-by-line tee). **The full raw output is always written to a log file** — `log=` overrides the path, otherwise `<deck>.log` next to the deck (`out/model.tcl` → `out/model.log`); verbatim, flushed live (tail-able), and complete even on a crash. **Console output is opt-in via a `verbose` boolean**: `False` (default) prints three lines only — a begin banner (`>> OpenSees | analyze N x dt=… -> deck`), the log path, and an `[OK] … (exit 0)` / `[FAIL] exited K … -> see <log>` end line; `True` additionally renders a live `step k/N pct% t=… elapsed …s` counter plus streamed `[warn]` lines and a warning tally. On a non-zero exit the raised `RuntimeError` now carries only the **last 15 log lines + the log path**, never the whole buffer. The live counter is fed by a throttled `APEGMSH_PROGRESS i=.. n=.. t=..` marker (~20 samples over the run, plus the final increment) that the Tcl/Py emitters inject into **both** the plain and strategy-ladder analyze loops — gated by a new `progress=True` kwarg (default on; a bare emitter stays marker-free, so decks not driven through `run=` are unchanged), with `flush stdout` / `flush=True` so it streams live through the pipe. The `py(run=True)` child runs under `PYTHONUNBUFFERED=1` so its stdout isn't block-buffered until exit. Console markers are ASCII (`>>` / `[OK]` / `[FAIL]` / `[warn]`) — Windows cp1252 terminals choke on unicode. Scope: the subprocess paths only; in-process `run()` / `analyze()` (which need fd-level redirection) and the already-fetched `run_remote()` SLURM logs are unchanged. `subprocess` import dropped from `apesees.py` (now owned by `_run.py`). Locked by `tests/opensees/unit/test_analyze_progress_markers.py` (marker default-off, cadence, shape+flush, strategy branch, tcl+py) and `tests/opensees/unit/test_run_streaming.py` (log-always on success+failure, minimal vs full console, tail-only exception, helpers) — 15 tests, driven with a throwaway python subprocess so no OpenSees binary is needed in CI; also run-verified end-to-end against real openseespy (live counter on success, tail+log on failure).

### ADDED — plastic-strain tensor + derived invariants (`plastic_strain_*`)

Extends the derived layer to the **plastic-strain tensor**: a new 6-component Voigt family `plastic_strain_xx … plastic_strain_xz` (engineering shear, like the total strain) that plasticity materials emit, plus computed invariants on `results.elements.gauss` — `equivalent_plastic_strain_current` (√(2/3·eᵖ:eᵖ), the *current* equivalent), `volumetric_plastic_strain`, `j2_plastic_strain`, `max_shear_plastic_strain`, `principal_plastic_strain_1/2/3`. The `plastic_strain` shorthand expands DOF-aware (3 comps in 2-D, 6 in 3-D). `equivalent_plastic_strain_current` is kept **distinct** from the material-emitted accumulated PEEQ `equivalent_plastic_strain` (monotonic history) — they are different quantities. Consumed the same compute-on-read way across all three readers: (1) the self-describing native reader surfaces `plastic_strain_*` for free; (2) **MPCO** records the whole-mesh per-GP tensor + PEEQ **today** via the material-response family (`-E material.plasticStrain` / `-E material.plasticStrainEq`, one recorder line each) — the material-state read path now recognises the `plasticStrain` parent token and names its per-GP components **positionally** by Voigt order (6 → 3-D, 3 → plane), and `plasticStrainEq` is added as an alias of `equivalentPlasticStrain`; (3) the `.ladruno` reader's `continuum_canonical` maps `epsp11`/`epsp_xx` **and** the ASDPlasticMaterial3D `pstrain11`/`pstrain_xx` spellings → `plastic_strain_*`, plus the scalar PEEQ labels `eqpstrain` / `ebarP` / `equivalentPlasticStrain` / `plasticStrainEq` → `equivalent_plastic_strain` (the fork still needs an element-side gauss forward for `.ladruno` — the materials expose the response, no element tags it yet). **Also FIXED in this change**: `continuum_canonical` only mapped `sigma|eta` digit stems, but the Ladruno solids tag total strains `eps11..eps13` — gauss **strain** reads from a LadrunoBrick `.ladruno` silently returned empty; `eps`/`epsilon` digit forms now map to `strain_*`. The **elastic ν out-of-plane recovery is *not* applied** to the plastic-strain tensor (plastic out-of-plane follows a material-dependent rule, e.g. incompressibility for J2) — εᵖ_zz stays as stored (0 in 2-D unless recorded). `PLASTIC_STRAIN` added to the gauss recorder/capture allow-lists. Locked by analytic tests (uniaxial, engineering-shear halving, volume-preserving equivalent) + a native-file composite test.

### ADDED — derived-quantity Phase 5: advanced invariants, ν-aware plane-strain, shell von Mises, principal-direction glyphs

Extends the derived-scalar layer (below) with four features. **Advanced invariants** (`results.elements.gauss.get`): `j3_stress` (third deviatoric invariant det s), `lode_angle` (degrees, [-30,30], NaN under hydrostatic) and `stress_triaxiality` (σ_mean/σ_vM, NaN where σ_vM≈0) — for pressure-sensitive (Mohr-Coulomb / Drucker-Prager) constitutive interpretation. **ν-aware plane-strain, per-element auto-detect (default `plane="auto"`)**: on 2-D data the out-of-plane component is now recovered *per element* from the model — `results/_plane_recovery.py` reads each element's plane type (a positional/flag arg on the `ElementRecord`: `quad`/`tri31`/`tri6n`/`BezierTri6` → `args[1]`, `LadrunoQuad`/`CST` → `-type` flag) and its material ν (`ElasticIsotropic` `params[1]`, K/G materials via ν=(3K−2G)/(2(3K+G)), ASDPlastic via the `PoissonsRatio` token) and synthesizes a `stress_zz`/`strain_zz` column: plane-strain → σ_zz = ν(σ_xx+σ_yy) (ε_zz = 0), plane-stress → σ_zz = 0 (ε_zz = -ν/(1-ν)(ε_xx+ε_yy)). A model mixing both is handled correctly. Falls back to zero (plane stress) where the model can't be parsed (e.g. a synthetic file with no `/opensees` zone), so existing behaviour is preserved when there's nothing to read. Overrides: `plane="strain"`/`"stress"` force the idealization globally (`nu=` required), `plane=None` disables recovery entirely. Because the synthesized column is *injected into the tensor* the derived math already assembles, a file that stores a real `stress_zz` (a fork recording the material's σ33) is used verbatim and this reconstruction is skipped — exact even for nonlinear plane-strain. **Consuming a recorded σ_zz**: the two self-describing readers surface it with **no catalog change** — the native reader keys gauss components by dataset name, and the `.ladruno` reader's `continuum_canonical` maps `sigma33` (digit) / `sigma_zz` (axis), now **case-insensitively**, → `stress_zz`; since `base_components_for` reads all six `stress_*`, a recorded 4th component drops straight into the tensor. The **MPCO/STKO** reader (catalog-locked) is now width-adaptive too: `_maybe_extend_plane_layout` extends a 3-component plane stress/strain bucket to a 4-component `[σxx, σyy, σxy, σzz]` layout when the stored `NUM_COLUMNS` is exactly one component/GP wider — 3-component `.mpco` files read unchanged, 4-component ones surface `stress_zz` in `available_components` and the invariants. The NaN-aware recovery fills only the Gauss points where a fork emits a NaN sentinel (material couldn't supply σ_zz), keeping recorded finite values. Exactness: plane-stress (σ_zz=0, any material) and elastic plane-strain are **exact**; nonlinear plane-strain is an elastic estimate unless σ33 is recorded, in which case it's exact. **Shell von Mises** (`von_mises_shell`): recovers the extreme-fibre in-plane surface stress from the stored membrane-force / bending-moment resultants (σ = N/t ± 6M/t²) and returns the through-thickness envelope (max of top / bottom surface von Mises) — the worst-case design demand; needs the shell thickness via `get(component="von_mises_shell", thickness=t)` (explicit, not read from the section). Advertised in `available_components()` only when the six in-plane resultants are stored. **Principal-direction glyphs** — a new gauss-topology viewer diagram (`kind="principal_glyph"`, "Principal directions (arrows)"): three arrows per Gauss point along the principal directions of the stress (or strain) tensor, each scaled by its principal magnitude and coloured by the *signed* principal value on a diverging map (compression ↔ tension), so stress flow (struts / ties, principal orientation around openings) reads at a glance. Reuses `GaussSlab.global_coords` for world placement and the arrow `GlyphLayer` for rendering; deform-follows the substrate; per-principal toggles (`show_p1/2/3`); auto-registers in the Add-Diagram dialog / catalog / session / presets (ADR 0058 S0). `PrincipalGlyphStyle` carries `family` (stress/strain), `scale`, `cmap`, and the `plane`/`nu` out-of-plane knobs. New pure helper `results/_derived.principal_frame` (values + eigenvectors). Locked by additions to `tests/test_derived_scalars.py` / `tests/test_derived_scalars_composite.py` (Lode/triaxiality/J3 analytic states, plane-strain σ_zz recovery + ν-required guard, shell membrane/bending/envelope recovery, principal-frame eigenvector directions) and the new `tests/viewers/test_principal_glyph_diagram.py` (registration, 3-arrows-per-GP layer, per-principal toggle, signed colouring, per-step update + deform-follow, plane-strain style, no-tensor `NoDataError`).

### ADDED — derived stress/strain scalars (von Mises, principal, Tresca, J2, invariants) computed on read

`results.elements.gauss.get(component=...)` now serves **derived scalars** computed from the stored 6-component Voigt tensor instead of only what the recorder wrote — no new storage, and it works retroactively on any existing native / `.mpco` / `.ladruno` file (all three funnel through the one gauss composite). **Stress**: `von_mises_stress`, `tresca_stress`, `j2_stress`, `max_shear_stress`, `mean_stress` (I₁/3, tension-positive) and `pressure_hydrostatic` (−I₁/3, compression-positive) as two explicit sign conventions, plus `principal_stress_1/2/3` (σ₁≥σ₂≥σ₃ via batched `eigvalsh`). **Strain**: `von_mises_strain` (equivalent), `volumetric_strain`, `j2_strain`, `max_shear_strain`, `principal_strain_1/2/3`. The strain tensor is assembled with the **engineering→tensor shear halving** (OpenSees reports γ_xy=2ε_xy) — without it every principal strain is wrong. The compute layer is a new pure-numpy module `results/_derived.py` (vectorized over `(T, N)`); the composite reads the base columns for the same selection and returns a synthesized `GaussSlab`. Derived names **auto-surface** in the Add-Diagram picker and `plot.contour(topology="gauss")` because `gauss.available_components()` advertises them — but only when a **complete tensor** is stored (the in-plane trio `{xx,yy,xy}` at minimum, covering genuine 2-D plane files and full 3-D; a partial tensor is not advertised, since an invariant off it would silently zero-fill). Requesting a derived scalar when the raw tensor was never recorded fails loud. **2-D** uses a plane-stress default (out-of-plane σ_zz=0); the ν-aware plane-strain path and principal *directions* are deferred. Vocabulary split into `DERIVED_STRESS_SCALARS` / `DERIVED_STRAIN_SCALARS` (`DERIVED_SCALARS` kept as a back-compat union); recorder emit routes each derived name to the raw `stresses`/`strains` token. Locked by `tests/test_derived_scalars.py` (analytic uniaxial / pure-shear / hydrostatic states + the engineering-shear correction + the completeness rule) and `tests/test_derived_scalars_composite.py` (compute-on-read parity vs base columns, principal ordering, missing-tensor guard, 2-D plane default).

### ADDED — point-cloud dot-size control (fiber section + sand): style knob, live setter, settings-tab spinner

Dot size on the two point-cloud diagrams is now user-controllable end to end. **Style**: `FiberSectionStyle` gains a real `point_size: float = 10.0` (screen-space pixels) — its `point_size_fraction` was a **dead knob** (documented as "dot radius as a fraction of the model diagonal" but never read anywhere; leftover from a world-sized-sphere design); it stays on the dataclass, documented DEPRECATED, solely so sessions saved before `point_size` existed still deserialize (`style_cls(**data)` would otherwise drop the spec). `SandStyle.point_size` already existed. **Runtime**: both diagrams gain `set_point_size()` / `current_point_size()` following the vector-glyph `set_scale` runtime-override pattern (frozen style untouched; layer re-emitted live). **Backend**: the in-place `update_layer` fast path now pushes `layer.point_size` onto the actor property — point size lives on the actor, not the dataset, so live size changes on point-cloud layers were silently dropped by the cheap-animation path. **UI**: a shared point-cloud settings panel (dot-size spinner + the standard color panel) — `fiber_section` moves from the bare color panel onto it, and `sand` gets its first settings card (it previously fell through to "No settings UI for kind 'sand' yet"). Locked by 5 new tests (style flow-through, live setter at the layer and at the actor property for both kinds, legacy-session dict without `point_size` restores).

### FIXED — fiber-section dot cloud invisible on GL stacks where sphere billboards draw nothing

`FiberSectionDiagram`'s 3-D dot cloud rendered its vertex-cell `MeshLayer` with `render_points_as_spheres=True`. On some GL stacks (verified 2026-07-07 on Windows, both `pv.Plotter(off_screen=True)` and an on-screen window) that flag draws **zero pixels** — a 2000-point cloud renders 36k+ pixels as flat points and literally nothing as billboards — so the fiber overlay was completely invisible in the results viewer while the scalar bar still appeared (render-reproduced with the `test_fiber_diagram.py` fixture: beam line + bar, no dots). Switched to flat GL points (same call already carries `point_size=10.0`), matching the sand diagram's convention; dots render everywhere now (48/48 fixture fibers visible in the after-shot). The layer-contract test pins `render_points_as_spheres is False` with the rationale.

### ADDED — sand volume plot: field-colored grain cloud inside solid elements (new `"sand"` diagram kind)

Surface contours only show a solid's field on its skin; the interior is invisible without clipping. The new **`SandDiagram`** (`kind="sand"`, "Sand volume plot" in the Add Diagram dialog) fills every 3-D element with small "sand" grains — random interior points colored by the nodal component interpolated at each grain — so a stress bulb, plastic zone, or propagating wave front reads at a glance *through* the volume. Grains are allocated **proportionally to element volume** (uniform spatial density regardless of mesh grading; |J|-at-centroid × parent volume via the shared shape-function catalog) and placed by uniform parent-domain sampling (tet4/10 Dirichlet simplex, hex8/20/27 cube, wedge6 tri×line; unsupported 3-D types skip LOUDLY via `WarnSandUnsupportedElements`). One shape-function weight row per grain drives position, per-step value, and deform-follow consistently — grains ride the deformed substrate and animate with the time scrubber / animation export for free. Rendered as a vertex-cell point-cloud `MeshLayer` (flat GL points, non-pickable — sphere billboards draw NOTHING on some Windows GL stacks, render-verified) so per-step updates hit the backend's in-place fast path; `occludes_substrate=True` hides the substrate fill (grains are strictly interior — behind an opaque fill the diagram would be invisible) leaving the wireframe as the volume outline. `SandStyle` knobs: `target_points` (grain budget), `point_size`, `cmap`/`clim` (standard `ScalarColorSupport` LUT + scalar bar), `opacity`, `seed` (reproducible clouds), and an optional **value-weighted density mode** (`weight_by_value=True`): each grain draws a fixed random threshold at attach and only shows at steps where the normalized |value| at its location exceeds it — dense sand where the field is strong, sparse where weak, flicker-free under animation (`density_floor` keeps a faint outline). Registry-driven wiring (ADR 0058 S0): the kind appears in the dialog/catalog/session/presets automatically. Locked by `tests/viewers/test_sand_diagram.py` (13 tests: registration, grain budget + containment + seed determinism, convexity bound, partition-of-unity step shift, deform-follow translate + reset, density masking, shell-only + missing-component `NoDataError`).

**Also FIXED in this change — `apply_visibility_mask` wrote the wrong ghost byte**: the backend's `_GHOST_HIDDEN_CELL` was `0x01`, which is VTK's **DUPLICATECELL** bit, not `HIDDENCELL` (`0x20`). Surface extraction happens to drop duplicate-ghost 1/2/3-D cells, so every existing `VisibilityMask` consumer rendered correctly — but the mapper's 0-D vertex path only honours the pure `HIDDENCELL` byte (even `0x21` fails), so any point-cloud layer mask was silently a no-op (render-verified: the sand density mask hid 17k grains in the IR and the dataset while the frame stayed byte-identical). Now `0x20`, which render-verifies as hiding both solid and vertex cells. `viewers/core/element_visibility.py` (the substrate's own writer) is untouched.

### FIXED — emit-memory runway review hardening (adversarial panel over #772–#778)

A five-lens adversarial review of the merged runway produced three real findings, all fixed here; everything else was refuted with evidence (aliasing/memo invariants, MassSet base-class coherence, duck-type consumer contracts, order determinism — no changes needed). (1) **Duplicate-eid tie-break restored to last-wins**: overlapping element PGs legally fan one FEM cell from two specs, producing duplicate `fem_eid` keys; the old `{eid: tag}` dict comprehension resolved them LAST-wins, but `FemToOpsTagMap._find`/`translate` used first-match `searchsorted` — silently retargeting recorder/damping/`remove_element` selections on such models. Both now use `side="right" - 1` (stable argsort preserves plan order among equals), reproducing the dict exactly; `items()` still yields both physical rows. (2) **`stream_finish()` moved inside the abort-guarded region** in `apeSees.tcl`: a failing `os.replace` mid-promotion (Windows file lock) previously escaped the handler, leaving mixed `.tmp`/promoted state with no cleanup; it now routes to `stream_abort` (remaining temps removed, driver-last ordering means no deck entry point can exist half-built), with the partial-promotion + re-run-heals contract documented on `stream_finish`, and `stream_abort` now also removes the eagerly-created `ranks/` dir when empty. (3) **Dead compose-path mass boxing removed**: `_rewrite_sourced_arrays` eagerly materialised `tuple(new_mass_set)` — one boxed `MassRecord` per node on every `compose()` (~GBs at multi-M nodes) — into a `_RewrittenBundle.mass_records` field that no code consumed; the field is gone, `mass_set` (columnar) is the only mass channel. New locks: `tests/opensees/unit/test_columnar_ownership_maps.py` (dict-parity battery for `FemToOpsTagMap`/`SortedIntToInt`/`NodePartitionOwners` incl. the duplicate-eid last-wins contract and scalar-vs-`translate` agreement) and three streaming tests (fault-injected `os.replace` mid-promotion → abort + re-run-heals; empty-`ranks/` cleanup; byte-identity assertions switched from newline-normalizing `read_text` to raw `read_bytes`).

### CHANGED — emit-memory runway CLOSED: plan + ADR 0065 status flipped to complete (docs only)

`internal_docs/plan_emit_memory_columnar.md` Status → COMPLETE with the measured milestone table (emit phase-peak 2,291 → 1,039 B/hex at the 103k-hex / 64-rank / staged reference across #772–#777; extrapolated traced peak @ 11M hexes 23.6–25.3 → 11.4 GB; `MassSet` 400–700 → 56 B/node), the remaining-ledger note (the pre-existing emit-loop transient, ~715 B/hex partitioned — attribute before spending), and the Route E triggers. ADR 0065 Status → ACCEPTED — fully shipped (Tier 1 2026-06-18; Tier 2 as `ops.tcl(stream=True)` in #777, with the demand-gate profile having run as plan M0). No code changes.

### ADDED — `ops.tcl(stream=True)` write-through streaming sink + live per-rank routing + atomic writes (ADR 0065 Tier 2 / plan_emit_memory_columnar.md A1–A3)

Retires the emit-side **line buffer** — the last Route-A ledger term (27.9 MiB / 239k resident line strings at the 103k-hex staged/64-rank reference). `apeSees.tcl(path, stream=True)` now writes the deck **through a live file sink** instead of accumulating `_LineBuf`: the buffer class is dual-mode (list accumulation stays the default, byte-identical to before; with an attached sink, `append` writes `indent + line + "\n"` straight through and stores nothing — the partition-indent logic already lived in `append`, so every emitter call site is captured for free). The banner + any `preamble()` lines buffered at attach time flush to the sink first; `preamble()`/`insert(0)` after streaming has begun fails loud. **Per-rank live routing** (`per_rank=True` + `stream=True`, the production path): `partition_open(K)` switches the active sink to `ranks/rank<K>_<seq>.tcl` (writing the ADR 0061 fragment banner, body lines streamed with the partition-level indent stripped live via `(indent + line).removeprefix("    ")` — the exact post-hoc transform, so intra-block nesting indents and the indent-0 override paths reproduce faithfully); `partition_close()` closes the fragment and writes the driver's source-guard + blank line. Fragment naming (`<seq>` = per-rank 0-based block counter) and every byte of driver + fragments are **identical to `_write_per_rank_tcl`'s output**; `PartitionSpan` recording retires in stream mode. **Atomic writes**: everything streams to `.tmp` siblings promoted via `os.replace` on clean completion (fragments before the driver); a mid-emit exception aborts and removes every temp — never a half-written final deck. Guards (v1): `stream=True` + `split=True` → `ValueError`; `ops.py(..., stream=True)` → `ValueError` (the HPC path is Tcl; the py emitter stays list-only); `lines()` / `line_buffer()` / `write_to()` / `partition_spans()` in stream mode → `RuntimeError`; `run=True` composes (the file exists after the replace). Measured (`--stream` added to `tests/benchmarks/emit_throughput_profile.py`; box 103k hexes / 64 ranks / staged, `--mem`): post-emit **resident** 37.3 → 8.0 MB (the 27.9 MiB line-buffer top-site is gone outright), emit **phase-peak** 1,321 → 1,039 B/hex (traced peak 137 → 108 MB, RSS 490 → 421 MB, extrapolated @ 11M hexes 14.5 → 11.4 GB). The remaining peak is a **pre-existing transient inside the emit loops themselves** (identical in list mode; 715 B/hex partitioned non-staged, 613 B/hex flat) — the next ledger term, out of Route-A scope. Locked by `tests/opensees/unit/test_emit_streaming_write.py` (stream-vs-list byte-identity on flat / partitioned / staged+partitioned; per-rank driver + every fragment vs the post-hoc writer incl. staged `<seq>` numbering; tracemalloc O(1) emit-peak ceiling; all guards; mid-emit-exception atomicity — 14 new tests). Full `tests/opensees` green (5049 passed); ruff + package-wide mypy (baseline 0) clean.

### FIXED — H5 deck-replay builds a `FemToOpsTagMap` (mypy ratchet red on main after B2+B3)

The B2+B3 slice (below) missed one caller: `_internal/compose.py`'s deck-replay reconstructs `{fem_eid: ops_tag}` from replayed element records and passed plain dicts to `emit_initial_stress_addtoparameter` / `emit_activate_absorbing`, which now type against `FemToOpsTagMap` — 3 mypy-ratchet errors on main (runtime-safe: those helpers only call the duck-typed `.get()`, and the full suite was green). Both replay sites now build the map via a new `FemToOpsTagMap.from_pairs((fem_eid, ops_tag), ...)` classmethod (pair order = `items()` order, mirroring the old dict's insertion order). mypy ratchet back to baseline 0.

### CHANGED — build-side ownership dicts/sets + tag map go columnar (ADR 0065 v2 / plan_emit_memory_columnar.md B2+B3)

Retires the remaining build-side per-entity Python containers behind the emit path — the terms left after B1+B4 (below). Three compact array-backed classes in `opensees/_internal/build.py`, each duck-typed to the mapping it replaces (`get` incl. default, `[]`, `in`, iteration-over-keys, `items`/`keys`/`values`, `==` against a plain dict) so consumers are annotation-only changes: **`FemToOpsTagMap`** replaces the five `{fem_eid: ops_tag}` dict comprehensions over the element plan (~160 B/element resident + the boxed triple walk; ~1 GB at LOH.1 scale) — resident form is two int64 arrays straight off the columnar plan, point lookups via `searchsorted` on a sorted view, plan-order `items()`, node-pair sentinel rows filtered at construction, and a vectorised `translate(eids) -> tags` for whole-selection resolution; **`SortedIntToInt`** replaces the `{fem_eid: rank}` / `{node_id: rank}` ownership dicts (`build_element_partition_owner`, `primary_owner_map`) with two sorted int64 arrays + a vectorised `translate_ranks` used by the per-rank bucketing; **`NodePartitionOwners`** replaces the `{node_id: set[rank]}` map from `build_node_partition_owners` — the single largest build-side term (one boxed Python `set` per node, ~315 B/hex at box-64-rank scale) — with a CSR layout (`node_ids` / `offsets` / `ranks`, ~2 int64 per node in the common single-owner case); `get()` yields a transient `frozenset` (the MP-constraint replication paths intersect against it), and `primary_owner()` reduces to the lowest rank vectorised (`_ranks[_offsets[:-1]]`, owner runs are ascending by construction). Measured (`--mem`, box 103k hexes / 64 ranks / staged): emit phase-peak **1,688 → 1,321 B/hex** on top of B1+B4's 2,144 → 1,688, extrapolated traced peak @ 11M hexes **18.6 → 14.5 GB**; the dominant remaining term is now the emitter line buffer (Route A, next slice). Decks byte-identical (byte-identity fixtures + full `tests/opensees`); ruff + mypy clean.

### CHANGED — columnar `MassSet` storage (ADR 0065 v2 / plan_emit_memory_columnar.md C1–C3)

`fem.nodes.masses` (`MassSet`) no longer keeps one resident `MassRecord` dataclass per node — at LOH.1 scale (~7M nodes) that boxed graph cost ~3–5 GB. It now stores masses in three parallel columns (`_node_ids: int64[N]`, `_mass: float64[N,6]`, `_names: dict[int,str]` sparse) and constructs a **transient** `MassRecord` on the fly when iterated / indexed. The resident store drops from ~400–700 B/node to **56 B/node** (measured 28.7 MB at 512k nodes). `MassRecord` (the view/API type) gains `slots=True`; nothing set ad-hoc attributes on it. The public surface is unchanged (`__iter__` / `__len__` / `__bool__` / `__getitem__` / `by_kind` / `by_node` / `total_mass` / `summary` / `_with_record`), and membership (`rec in fem.nodes.masses`) still works by dataclass value-equality. **Consumer-inventory finding:** every consumer (the OpenSees bridge `_emit_masses` / `_emit_masses_partitioned` / bucketed partitioned `mass_from_model` paths, the mass viewer tab, compose, h5 io) only reads `m.node_id` / `m.mass` / `m.name` — none rely on record identity (`is`) or in-place mutation, so transient records are safe. Producers: (a) the in-session resolver (`MassesComposite.resolve`) builds the columnar set directly from the sorted per-node accumulator — never boxing a records list (`mass_records` becomes a lazy property); (b) the numpy-native `_read_masses` at `FEMData.from_h5` **adopts** the already-columnar `/masses` compound dataset's `node_id` / `mass` columns with a single copy instead of boxing 7M records (measured `from_h5` peak 628 → 468 MB at 512k nodes); `_write_masses` fills the compound payload straight from the columns. The compose tag-rewrite (`tag_rewrite_spec` node_id offset) gains a **vectorized columnar fast path** (`_rewrite_mass_set` = one `node_ids + offset` array add; sparse-name namespace-prefix) and the compose merge concatenates host + bundle mass columns in one shot, retiring the O(N²) per-record `with_mass` append loop. No `model.h5` schema change — `/masses` was already columnar; SchemaVersion untouched, decks byte-identical (float `repr` preserved bit-for-bit, verified with awkward floats: 0.1, 1e-300, 17-significant-digit values). Verified: full `tests/opensees` (5035 passed), `tests/mesh` + compose + mass suites, a new float-identity/round-trip gate (`tests/test_mass_columnar_float_identity.py`), and a non-collected memory benchmark (`tests/benchmarks/mass_columnar_memory_profile.py`). ruff + mypy clean on touched files (no new diagnostics). Out of scope (optional C4): `NodalLoadSet` / SP records share the pattern but are untouched.
### CHANGED — columnar element plan + columnar PG fan-out (ADR 0065 v2 / plan_emit_memory_columnar.md B1+B4)

Cuts the dominant emit-time RAM term for large models — the per-element Python object graph (per-element plan tuples + ~54M boxed connectivity ints, ~4–6 GB at the LOH.1 ~6.7M-hex reference). The element fan-out and the pre-allocated element plan now keep their **resident** form columnar (int64 arrays straight off the FEMData group arrays, which are already numpy) and box a row only **transiently** at iteration. Two new internal containers in `opensees/_internal/build.py`: **`PGElementFanout`** (`.eids: int64[N]`, `.conn: int64[N,k]` or object-padded per-row for mixed-npe groups) replaces the memoised `list[tuple[int, tuple[int, ...]]]` returned by `expand_pg_to_elements` / `expand_spec_to_elements`; **`ElementPlanRows`** (`.eids`, `.conn`, `.tag_start`) replaces each spec's `list[tuple[int, tuple[int, ...], int]]` in `allocate_element_tags`. Both are duck-typed to the old list-of-tuples — iterating yields the exact same `(eid, conn)` / `(eid, conn, tag)` tuples in the same order, and `__len__` / `__getitem__` / `bool` behave like the old list — so every existing consumer (the flat / split / staged / partitioned emit loops, the `fem_eid_to_ops_tag` dict comprehensions, the recorder / rayleigh / damping region fan-outs, `sweep_asdconcrete_element_size`, `ModelData.oriented_elements`) is unchanged. `TagAllocator` gains **`allocate_block(kind, n)`** — reserves a spec's `N` element tags in one call with identical per-kind sequential counter semantics (element tags are allocated only in this one pass, so a spec's tags stay the contiguous block `[tag_start, tag_start+N)`; row `i`'s tag is `tag_start + i` positionally). `bucket_pre_allocated_by_rank` now yields per-rank **`ElementPlanRows` row-subset views** (arrays indexed by the owned-row positions, carrying explicit per-row tags) instead of re-materialising a tuple graph per rank — so partitioned / staged-partitioned emits no longer rebuild the boxed plan per rank. Node-pair (`pg=None`) specs are a 1-row fan-out carrying the `MISSING_FEM_ELEMENT_ID` (-1) sentinel, which fits int64 and is filtered out of the tag maps by value exactly as before. **Byte-identical** emitted Tcl + Py decks for flat, partitioned (multi-rank), staged+partitioned, per-rank fragment files, and split-module emit (the existing byte-identity fixtures + full `tests/opensees` suite pass). No new public API (both containers are internal). B2 (retiring the `fem_eid_to_ops_tag` dicts) and B3 remain deferred.

### CHANGED — emit-memory runway opened (ADR 0065 v2): M0 `--mem` attribution + A0 line-buffer copy fixes

First slice of the large-model emit-memory plan (`internal_docs/plan_emit_memory_columnar.md` — columnar element plan / columnar masses / Tier-2 streaming sink, targeting the ~28 GB OOM on 6.7M-hex partitioned+staged emits):

- **M0** — `tests/benchmarks/emit_throughput_profile.py --mem` adds tracemalloc + RSS-peak attribution over the build / emit / write phases (per-phase resident + peak, B/hex, extrapolation to 6.7M/11M hexes, top allocation sites at the post-emit resident point). This is the re-baseline gate ADR 0065 demanded before Tier 2, and the regression harness every subsequent slice must move. Wall-clock from a `--mem` run is documented as non-throughput.
- **A0** — `TclEmitter`/`PyEmitter` gain `line_count()` and a read-only `line_buffer()`; the split-emit module-span recording no longer calls `len(emitter.lines())` (which cloned the entire multi-M line list to read a length — four sites), and the per-rank / split deck writers consume `line_buffer()` instead of the `lines()` deck-sized copy. Behavior and emitted decks unchanged.

### ADDED — `OpenSeesModel.build()` deck-replay re-emits `g.reinforce` ties (ADR 0067 P5.1 "A4 full", reinforce leg)

Closes the `/opensees` deck-replay gap for embedded-reinforcement ties. Previously a reinforced `model.h5` loaded via `OpenSeesModel.from_h5().build("tcl"/"py"/"live")` **silently dropped** its `LadrunoEmbeddedRebar` ties — the deck-replay path (`_replay_into`) re-emitted nothing for them (the H5 emitter no-ops the tie deck record; persistence is the neutral zone's job). Now `_replay_into` gained a step **8b** that re-emits the ties **from the neutral-zone `fem`** (`fem.elements.reinforce_ties`, from the `/reinforce_ties` group) — which `OpenSeesModel.build` already passes in and already leans on for element-connectivity rehydration. Tie element tags are freshly allocated **past the max replayed element tag** (they share the element namespace, so a fresh 1-based counter would collide; ADR 0019 INV-5 already allows tag divergence across round-trip), and a tie's `-bond <name>` resolves via a name→tag map threaded from the `/opensees/names` sidecar (`OpenSeesModel._names`). This is the **cleaner re-emit-from-neutral design**, superseding the plan's original dedicated-`/opensees`-deck-record approach: **no new deck record, no `embedded_rebar` write, no opensees-zone schema bump** — the deck zone is unchanged. The `h5` re-emit caller passes no `fem`, so it is correctly skipped (the `h5` target persists ties via the neutral zone; `build('h5')` stays byte-stable). The canonical recovery for a reinforced model — `FEMData.from_h5` → forward re-emit — is unchanged. **Scoped / documented follow-on:** `_replay_into` still replays **no other** MP-constraint family (equalDOF / rigidLink / rigidDiaphragm / embeddedNode / contact / embed / equation ties); reinforce ties are the first and only family deck-replayed, and each of the rest could later re-emit from the neutral fem the same way. Also records that the deferred `s.mortar` / `s.tied_contact` stage-claim item is **already resolved** (`s.tied_contact` shipped in ADR 0034; `s.mortar` intentionally out of scope since the ADR 0073 contact refactor left no claimable record). Locked by `tests/opensees/h5/test_reinforce_deck_replay.py` (perfect + bond-by-name tcl re-emit, py target, tie-tag non-collision, and `build('h5')` persisting via the neutral zone). Static gates green (ruff opensees hard gate + mypy 0); full `tests/opensees` suite 4981 passed.

### ADDED — `g.reinforce(..., corot=True)` co-rotated bar axis for large host rotation (ADR 20 §10.5, R3c)

Exposes the fork `LadrunoEmbeddedRebar` **`-corot`** option on `g.reinforce(..., corot=True)`: the embedded-rebar coupling co-rotates the bar axis each step from the current host geometry, keeping the axial/transverse split frame-objective under large host rotation. `False` (default) ⇒ the frozen reference `-dir` (byte-identical to before). The fork needs a second point **B** along the bar to form `d̂_cur = normalize(Σ NshapeB·x − Σ Nshape·x)` from current host node positions; apeGmsh emits the **host-element-tag-free `-shapeB`** point-B weights so no host-element query is needed (the `-shape` path's corot sibling). The resolver computes each node's `shape_b` by stepping a short distance (`0.05·host_radius`) along the bar axis from the embed point and inverse-mapping that point into the **same** host element (`inverse_map_single`) — the magnitude cancels under the fork's normalisation, so a small in-host step robustly captures the host's local deformation gradient along the bar (the `±d̂` step is retried if the first direction leaves the host). The grammar builder (`embedded_rebar_args`) already carried `-corot -shapeB` from R3a; this wires the upstream lane — `g.reinforce` → `ReinforceDef` → `resolve_reinforce` (computes `shape_b`) → `ReinforceTieRecord` → `emit_reinforce_ties` — and **round-trips through `model.h5`** via additive `corot` + `shape_b` columns on `reinforce_tie_payload_dtype` (neutral schema **2.25.0 → 2.26.0**; presence-probed so an in-window 2.25.x file decodes `corot=False`, `shape_b=None`; omitted-knob round-trip stays byte-identical). The encode/decode validate `shape_b` parallel to `host_nodes` (mirroring `weights`). Closes the last R3 reinforcement leg (R3a explicit/AL/bipenalty already shipped). Locked by `tests/_kernel/resolvers/test_reinforce.py` (point-B weights sum to 1, secant along the bar axis, non-corot leaves `shape_b=None`), `tests/opensees/unit/test_reinforce_emit.py` (emit `-corot -shapeB`), and `tests/mesh/test_reinforce_tie_h5_roundtrip.py` (full corot round-trip + decode presence-probe). Static gates green (ruff opensees hard gate + mypy 0).

### ADDED — `g.constraints.contact(..., edge_edge=, edge_*=)` mortar edge-edge contact fallback (ADR 0073, fork ADR-57 E2–E7)

Exposes the fork `LadrunoContact` **edge-edge** (perpendicular segment-to-segment) contact fallback — the `cos_t→0` pairs the face-mortar clip degenerates on get a dedicated segment-to-segment penalty. Eleven new knobs on `g.constraints.contact(...)`, all **mortar-only**: `edge_edge` (enable the fallback, `-edgeedge`), `edge_kn` (`-edgeKn auto|<v>`, the edge penalty; `None` ⇒ the mortar penalty), `edge_band` (`-edgeBand`, the gap activation band), `edge_mu` / `edge_kt` / `edge_cohesion` / `edge_tau_max` (`-edgeMu`/`-edgeKt`/`-edgeCohesion`/`-edgeTauMax`, the edge Coulomb/Tresca friction cone), `edge_consistent_tan` (`-edgeConsistentTan`, the non-symmetric Csl tangent), `edge_soft` (`-edgeSoft [SOFSCL]`, the explicit Courant-stable SOFT penalty — `True` ⇒ the fork default 0.10, a float ⇒ an explicit SOFSCL), `edge_alm` (`-edgeAlm`, the one-scalar commit-cycle ALM), and `edge_aug_tol` (`-edgeAugTol`, the ALM tolerance). `ContactDef` mirrors the fork's two fail-loud gates: `edge_edge` requires `formulation="mortar"` (the fork routes the fallback off the mortar lane), and the `edge_*` params require `edge_edge=True` (the fork warns-and-IGNORES `-edge*` without `-edgeedge` — apeGmsh fails loud rather than silently drop them). Edge penalties/friction are range-validated (`edge_kn` takes the `"auto"` sentinel; the friction analogues allow the fork's zero sentinels; `edge_band`/`edge_aug_tol` are strictly positive) and an `edge_soft` SOFSCL > 1 warns (ω·dt = 2√SOFSCL > 2), mirroring the fork. Threaded through the full lane — `g.constraints.contact` → `ContactDef` → `ContactRecord` → `contact_args` (emits `-edgeedge` then the requested edge knobs, after `-cell`, before `-outward`) → `emit_contacts` — and **round-trips through `model.h5`** via additive edge columns on `contact_payload_dtype` (neutral schema **2.24.0 → 2.25.0**; `edge_kn` uses an auto/None/numeric tri-state, `edge_soft` a None/bare/numeric tri-state, everything else NaN-sentinelled; presence-probed so an in-window 2.24.x file decodes the fallback off; omitted-knob round-trip stays byte-identical). Closes the headline remaining ADR 0073 contact leftover — face-to-face contact + friction + `contactPlane` + the `-cell` knob were already shipped, leaving edge-edge as the one substantial contact piece. Locked by `tests/opensees/unit/test_contact_emit.py` (edge emit ordering + both fail-loud gates + range/SOFSCL validation + record→emit) and `tests/mesh/test_contact_h5_roundtrip.py` (full edge round-trip, bare-`edge_soft` tri-state, fallback-off defaults, decode presence-probe). Static gates green (ruff opensees hard gate + mypy 0).

### ADDED — `g.constraints.contact_plane(...)` rigid analytical-plane contact (ADR 0073)

Exposes the fork `contactPlane` command via `g.constraints.contact_plane(slave, *, normal=, point=, kn=, visc=None, soft=None, name=)` — a meshed slave surface contacting a fixed infinite **rigid plane** (frictionless, no master mesh), for a rigid floor / wall / foundation where the counter-body needn't be meshed. Emits one `contactSurface -slave <nodes>` + one `contactPlane tag slaveSurfTag nx ny nz px py pz kn [-visc μ] [-soft S]`, resolved by the same auto-emitted `LadrunoContact` handler (the `_fem_has_contacts` gate now triggers on `contacts` **or** `contact_planes`). `kn` is **required** numeric (the fork has no `"auto"` on contactPlane); `ContactPlaneDef` validates a non-zero normal + strictly-positive `kn`/`visc`/`soft`. Implemented as a full sibling stack of the face-to-face contact generator: `ContactPlaneDef` / `ContactPlaneRecord` → `ConstraintsComposite.contact_plane` / `resolve_contact_planes` → `fem.elements.contact_planes` (`_fem_factory` wiring) → `emit_contact_planes` (build pass) → the `contact_plane` Protocol method on all five emitters (tcl/py/live[fork-gated]/h5[deck no-op]/recording). **Round-trips through `model.h5`** via a NEW `/contact_planes` group (`contact_plane_payload_dtype`, neutral schema **2.23.0 → 2.24.0**; the group is omitted when there are none so a plane-free model stays byte-identical — `snapshot_id` unchanged). **Serial-only** like face-to-face contact: a `contact_plane` under partitioned (MPI) emit **fails loud** (the partitioned guard now fires on `contacts` OR `contact_planes`, so a plane is never silently dropped nor a spurious `LadrunoContact` handler auto-emitted). Closes the second of the three ADR 0073 deferred contact leftovers; the `-epsTie` alias is resolved as **intentionally not exposed** (redundant — it is a pure alias for the `-epsN` penalty slot that `eps_n` already emits). Only the edge-edge lane remains deferred. Locked by `tests/opensees/unit/test_contact_plane_emit.py` (grammar + def validation + emit pass) and `tests/mesh/test_contact_plane_h5_roundtrip.py` (round-trip incl. soft modes, group-omitted byte-stability, prior-minor window, encode fail-loud). Static gates green (ruff opensees hard gate + mypy 0).

### FIXED — `viewers/ui/_open_results.py` imports `OpenSeesModel` via the allowed package surface

Follow-up to the File → Open Results… feature (#757): `build_results`'s native path imported `OpenSeesModel` from the `apeGmsh.opensees.opensees_model` **submodule**, which `tests/test_viewers_pure_h5_consumer.py::test_viewers_have_no_mesh_or_opensees_imports` forbids — `viewers/` may only reach it through the package top-level (`apeGmsh.opensees`). #757's CI `suite` job flagged this, but the PR auto-merged before the fix landed (the repo has no required status checks, so `--auto` merged immediately while `suite` was still running), shipping the violation to `main`. This one-line change routes the import through `from apeGmsh.opensees import OpenSeesModel`, restoring the curated `suite` to green. No behaviour change.
### ADDED — `g.constraints.contact(..., cell=)` broad-phase cell-size knob (ADR 0073)

Exposes the fork `LadrunoContact` `-cell <frac>` option — the broad-phase spatial-hash bucket size as a fraction of the median segment diagonal (a performance-tuning knob; a huge value ⇒ one bucket ⇒ brute force). Applies to **both** formulations (NTS + mortar); omitted ⇒ the fork default. `ContactDef` requires it strictly positive (mirroring the fork parser's "need a positive frac"). Threads `cell` through the full lane — `g.constraints.contact` → `ContactDef` → `ContactRecord` → `contact_args` (emits `-cell` after the extension modifiers, before `-outward`) → deck emit — and **round-trips through `model.h5`** via an additive `cell` column on `contact_payload_dtype` (neutral schema **2.22.0 → 2.23.0**, presence-probed so an in-window 2.22.x file decodes `cell=None`; omitted-knob round-trip stays byte-identical). Closes one of the three ADR 0073 "still deferred" contact leftovers (the `-epsTie` alias and the `contactPlane` rigid-plane command remain). Locked by `tests/opensees/unit/test_contact_emit.py` (emit both lanes + numeric-kn triple padding + strictly-positive validation) and `tests/mesh/test_contact_h5_roundtrip.py` (`cell` folded into `_eq` + the NTS extensions round-trip). Static gates green (ruff opensees hard gate + mypy 0); 137 contact/parity/emission tests pass.

### ADDED — results viewer **File → Open Results…** with a format-aware model follow-up

The post-solve `ResultsViewer` window now carries a leftmost **File** menu with an **Open Results…** action: it pops a `QFileDialog` for the results file, sniffs the format from the file's contents, and only prompts for a second `model.h5` when that format actually needs it. The "is a model needed?" rule follows directly from how each loader sources its broker — `.mpco` → **required** (broker never embedded), native `.h5` without an embedded `/opensees` zone → **required**, `.ladruno` → **optional** (self-sufficient; a model only enriches orientation + lineage), native `.h5` with embedded model → **nothing asked**. The opened file appears in a **new** viewer window beside the current one. New module `viewers/ui/_open_results.py` holds the Qt-free helpers `sniff_results_format` / `model_requirement` / `build_results` (content sniffing: Ladruno `INFO/GENERATOR`, STKO `MODEL_STAGE[` groups, native `/model`+`/opensees`, with an extension fallback) plus the `run_open_dialog` Qt driver; `ViewerWindow.exec()` is split into `present()` (show/raise/render) + the `app.exec_()` tail so the second window joins the running event loop instead of nesting it (`ResultsViewer.show(enter_loop=False)`). Partitioned `.ladruno` siblings still auto-merge with no prompt; partitioned `.mpco` discovery is a deferred follow-up. Locked by `tests/viewers/test_open_results.py` (14: real-fixture ladruno round-trip, native/mpco classification, per-format requirement, build-results error paths). Full viewers suite green (1576).

### FIXED — `enforce="equation"` ties round-trip through `model.h5` without a deviation warning (ADR 0068, Open item 4 resolved)

The H5 *deck* emitter no longer raises `H5EquationConstraintDeviationWarning` for an `enforce="equation"` tie (EQ_Constraint). The warning was over-conservative: the equation tie is a resolved `InterpolationRecord`, and the **neutral** zone already persists its `enforce` route **and** the projection `weights` (schema 2.14.0) — everything `_emit_equation_tie` needs. So an equation-tied `model.h5` already round-trips via `FEMData.from_h5` → `apeSees(fem).tcl()/py()/run()` (the forward emit re-runs `_emit_one_interpolation` → `_emit_equation_tie`), exactly like the g.embed / g.constraints.contact / g.reinforce ties, which all no-op silently in the deck zone. The deck emitter now matches them (silent no-op + `_skipped_equation_constraints` counter); `H5EquationConstraintDeviationWarning` is retained in `__all__` as a **dormant** back-compat class (no longer raised). No schema bump — the ADR 0068 premise that this needed an `equationConstraint` group was stale. With this, **no fork-feature carries an H5 deviation warning** (reinforce / contact / embed / equation all recover via the neutral zone); the standalone `/opensees` deck-zone replay stays the shared low-priority follow-on. Locked by `tests/test_equation_tie_emission.py` (`test_h5_deck_emitter_equation_tie_no_deviation_warning` + `test_equation_tie_reemits_identically_after_h5_record_roundtrip`). ADR 0068 Open item 4 + both handoff docs updated.
### ADDED — results-viewer animation export (video / GIF) — interactive button + headless `Results.export_animation`

The results viewer can now export the time history as an **MP4 video or animated GIF**. The encoding engine (`apeGmsh.viewers.animation.export_animation` — drive the director step-by-step, capture `plotter.screenshot` frames, encode via `imageio`) already existed but was reachable only through a method that required the blocking `viewer.show()` to have run first (which tears the plotter down on return), so in practice nothing could call it. Two reachable entry points now wire it up: **(1) a 🎬 Export button on the Time Scrubber** in `results.viewer()` — opens a save dialog (`*.mp4` / `*.gif`, suffix selects the format), captures every step at the scrubber's current FPS, and runs behind a cancelable `QProgressDialog` with a wait cursor + status-bar result (cancel via raising out of the new `progress` callback deletes the partial file and restores the user's step). The frames are exactly what's on screen — deformation, contours, camera, theme. **(2) a headless `Results.export_animation(path, *, fps=30, step_stride=1, stage=None, deform=None, camera=None, window_size=(1280,720), setup=None)`** that reuses the full Qt viewer off-screen: it builds the real viewer via the new `ResultsViewer.show(run_loop=False)` (constructs the window + scene + deform pump, realizes the GL surface for screenshots, but never enters the blocking event loop), applies `stage` / `deform` (a scale, or `(field, scale)`) / `camera`, runs the optional `setup(plotter, director)` hook for custom diagrams, exports, and tears down — **without** closing the caller's `Results` HDF5 handle (`_own_results_close` guard on `_on_close`). MP4 needs the `apegmsh[animation]` extra (`imageio-ffmpeg`); GIF is Pillow-only. `export_animation` gained an optional `progress(done, total)` callback (1-based, fires per frame; raising cancels). Locked by `tests/viewers/test_animation.py` (+4: progress-callback invoked 1..total, cancel-via-exception restores the step, and a headless `Results.export_animation` GIF round-trip that asserts the borrowed Results stays queryable afterward). `APEGMSH_SKIP_VIEWER` short-circuits the headless path for CI / `nbconvert`.
### FIXED — global `ops.damping.rayleigh` no longer silently dropped under partitioned (MPI) emit (ADR 0053 × ADR 0027)

A **global** `ops.damping.rayleigh(...)` declared outside any stage was emitted in the flat (single-process) deck but **silently absent** from the partitioned (OpenSeesMP) deck — zero `rayleigh` lines — so an `np>1` run came out **undamped** (a plane-wave absorbing-boundary model showed a uniform ~14 % / max 45 % seq↔np4 surface discrepancy; the plane-wave handoff's load-bearing finding #1). Root cause: `apeSees._emit_partitioned` never called the global damping emitters that `_emit_flat` runs driver-post — stage-bound damping survived (re-emitted per stage by `_emit_stages_partitioned`), but the bridge's *global* `rayleigh_records` / `damping_attach_records` were dropped. The fix adds `_emit_global_damping_partitioned`, called once after the per-rank fan-out for both staged and non-staged decks. It **mirrors the stage-bound partitioned damping pass**: `rayleigh` (bare global *and* region-scoped `on=`) and the Damping-object `region -ele … -damp` attaches are emitted **once outside any `partition_open` block** — correct under OpenSeesMP because `MeshRegion::setElements` keeps "only those elements in the domain" (foreign `-ele` tags from other ranks are silently skipped), so each rank binds its locally-owned subset, and a bare `rayleigh` applies to each rank's local domain. The global pool is therefore **not** treated more restrictively than the stage pool (an earlier draft fail-louded on the region-scoped/attach forms, which was an arbitrary asymmetry — the stage path already emits the identical global `region -ele` line). **Modal** damping (`ops.damping.modal` → `eigen` + `modalDamping`) is the one form that fails loud (`BridgeError`): a bare `eigen` solves each rank's *local* subdomain under OpenSeesMP, so the modes — and the modalDamping built from them — would be **wrong**, not merely unwired (the stage path likewise refuses per-stage modal). Locked by `tests/opensees/integration/test_emit_partitioned_global_damping.py` (5: bare global rayleigh emits once outside the rank blocks on Tcl + Py; the staged finding-#1 scenario; region-scoped `-rayleigh` + Damping-object `-damp` emit as global region lines outside the blocks; modal fails loud). Full `tests/opensees` suite green (4966 passed); ruff (opensees hard gate) + mypy clean. Emission is unit-verified without MPI; a live OpenSeesMP run is the final confirmation.

### CHANGED — `g.rebar` B1b (beam-element rebar) design resolved (ADR 0067 P5.2)

Records the B1b implementation design (docs only). An investigation pinned every injection point and made the engineering sub-decisions (no further human gate): **(1) embedded-only** — beam rebar nodes need `ndf=6`, and conformal bars share the host's `ndf=3` solid nodes, so `element="beam"` + `coupling="conformal"` will raise; beam auto-emit requires `coupling="embedded"` (the bar's own nodes go cleanly `ndf=6`, tied by `LadrunoEmbeddedRebar`). **(2) circular fiber section** emitted directly in `emit_rebar_elements` via `section_open("Fiber")` → `patch("circ", …)` → `section_close()` (no new Fiber primitive — the dedicated pass already bypasses the registered-`Element` machinery), giving real bending/dowel stiffness. **(3)** a `Lobatto` `beamIntegration` + a per-segment `geomTransf Linear` (the round section is symmetric, so `vecxz` orientation is immaterial — a valid per-segment perpendicular suffices). **(4) `ndf=6` injection** by extending `infer_node_ndf` (`build.py:336`) to bump every node of a `beam` `RebarElementRecord` to 6 (the dispBeamColumn elements aren't `Element` specs, so they're invisible to the default inference). **(5) `dispBeamColumn`** per line cell. **(6)** ungate `RebarComposite.py:1129`. **(7) twist folds in** — `LadrunoEmbeddedRebar` ties translations only, so a beam rebar node's rotational DOFs are a zero-energy mode (singular tangent); B1b ships WITH the B0.3 stabilization (existing `zeroLength` + SP to a ghost node, no new C++ class tag). The implementation is a large, core-adjacent change (touches `infer_node_ndf` / transforms / sections / ghost nodes) and will get an adversarial review. See `internal_docs/plan_rebar_p5.md` §B1b.

### ADDED — auto-emitted rebar elements round-trip through the neutral `model.h5` (ADR 0067 P5.2 / B1a.2, neutral schema 2.16.0)

`fem.elements.rebar_elements` — the cage's `place(emit_elements=True)` structural elements (one `RebarElementRecord` per bar) — now **persist** through `FEMData.to_h5` / `from_h5`, into a new dedicated `/rebar_elements` group (a symmetric-compound dataset via `rebar_element_payload_dtype`, modeled on the A1 `/reinforce_ties` group). Previously B1a left them in-memory only, with `to_h5` warning loud that a round-tripped file would be missing them; that warning is now **removed**. The record carries the bar's resolved line-cell `connectivity` (flat `2·n_cells` int64, extracted from the live mesh at `get_fem_data`), so the rebar elements re-emit byte-faithfully after a round-trip. The bump is the **neutral** zone `NEUTRAL_SCHEMA_VERSION 2.15.0 → 2.16.0`; the two-version reader window tolerates 2.15.x (which simply has no `/rebar_elements` group). The group is **omitted** when no bar opted into `emit_elements`, so a plain cage stays byte-identical and its `snapshot_id` is unchanged (the hash excludes rebar elements, consistent with constraints / ties). `RebarElementRecord` is re-exported from `apeGmsh._kernel.records` and mapped in `tests/test_record_schema_parity.py` (`RECORD_TO_DTYPE`); `g.compose` now **preserves the host's** rebar elements across a merge (carrying a *source* Part's rebar elements with PG/material-name prefixing — parallel to the reinforce-tie carry — is a deferred compose teach-in). `_encode_rebar_element` fails loud on empty/odd connectivity. Locked by `tests/mesh/test_rebar_element_h5_roundtrip.py` (8: round-trip with connectivity equality, group-omitted + snapshot-stable, no warning, version stamp, encode-rejects-empty, prior-minor window read). The reinforce-tie window tests + the schema fixture are bumped for the 2.16.0 reader. Full mesh + `_kernel` + rebar + reinforce suites green.

### ADDED — `g.rebar.place(emit_elements=True)` auto-emits the bar's structural element (ADR 0067 P5.2 / B1a)

`g.rebar.place(..., emit_elements=True)` (default `False` — no behavior change) now makes the cage **auto-emit each bar's own structural element** at `apeSees` build time, so `place` → `get_fem_data` → `apeSees(fem).tcl()/py()/run()` produces a runnable structural model instead of geometry + coupling that you wire by hand. **B1a ships the truss path:** one `CorotTruss` per bar line cell, using the bar's uniaxial-material **name** (resolved at emit, Option B) and area (`π·d_b²/4`). This is the bar's own axial element — distinct from the `LadrunoEmbeddedRebar` coupling, which carries no axial stiffness. Implementation: a new `RebarElementRecord` (`_kernel/records/_rebar.py`) carries the bar's line-cell `connectivity`, which `g.rebar.resolve()` extracts from the **live mesh** at `get_fem_data` (the dim-1 cells are dropped from a dim-3 `FEMData`, so they can't be read back from `fem.elements`); `mesh._fem_factory` calls it onto `fem.elements.rebar_elements`, and a dedicated `emit_rebar_elements` build pass (mirroring `emit_reinforce_ties`) emits the elements on both the flat and per-rank paths. Partitioned (MPI) emit and `element="beam"` bars **fail loud** (per-rank routing and the beam stack — fiber section + `beamIntegration` + per-segment `geomTransf` + `ndf=6` + twist — are B1b). **Known limit (B1a.2 follow-on):** `fem.elements.rebar_elements` is not yet persisted to the neutral `model.h5`; `FEMData.to_h5` **warns loud** so a round-tripped file isn't silently missing them (the in-memory emit path is unaffected). Locked by `tests/rebar/test_rebar_emit_elements.py` (4: off=no CorotTruss; on=one per cell with correct area/material; unregistered material fails loud; beam raises). 4361 mesh+opensees tests green; ruff (opensees hard gate) + mypy clean.

### CHANGED — `g.rebar` Track-B B1 design resolved: cage auto-emits structural rebar elements (ADR 0067 P5.2)

Records the B1 design for `element="beam"` dowel rebar (docs only). A code re-survey established the load-bearing finding: `g.rebar` today emits **geometry + coupling only** — `LadrunoEmbeddedRebar` (33005) is a **pure coupling** element (no axial stiffness; the bar's own stiffness lives on a separate structural element), and `RebarMember.element` (`"truss"`/`"beam"`) is **stored metadata, never consumed** (the structural bar element is user-emitted today, like the reference `ladruno_rc.py`). **Decision (user):** the cage will **auto-emit the bar's structural element behind an opt-in `place(emit_elements=…)` flag** (default off → no behavior change; on → `CorotTruss` for truss, `dispBeamColumn` for beam), wired through a declare → resolve → emit broker channel mirroring `g.reinforce` (`ReinforcementsComposite` → `resolve_reinforce` → `FEMData.reinforce_ties` → `emit_reinforce_ties`). The beam section is a **circular fiber section from `db` + the bar's uniaxial steel material**. Split into **B1a** (opt-in flag + truss auto-emit) → **B1b** (fiber section + per-segment `vecxz` reusing the existing `compute_vecxz_for_element` fan-out + `ndf=6` rebar nodes via the ADR 0048/0049 overlay + gate removal). Twist stabilization (B2) is decided per B0.3: try existing `zeroLength`+SP first, no new fork C++ class tag unless that fails. Full design + the open B1a kickoff question (new broker record vs reuse the existing element-emission machinery) in `internal_docs/plan_rebar_p5.md` §B1; handoff Track-B table updated.

### CHANGED — `g.rebar` Track-B B0 decision gate recorded (ADR 0067 P5.2/P5.3)

The B0 human-decision gate for Track B (`element="beam"` dowel rebar + twist) is resolved and recorded (docs only). All three decisions landed on the lighter-than-feared path after a code re-survey showed the infrastructure largely exists: **(1) orientation** = serialized `Orientation` (default `AlongBeam`) + `roll_deg` on the bar spec, with the bridge deriving each segment's `vecxz` at build via the **existing** `compute_vecxz_for_element` (the smooth beam-column orientation fan-out already exists; the rebar gap is only per-segment polyline tangents); **(2) mixed-ndf** = `ndf=6` beam-rebar nodes via the **existing** ADR 0048/0049 per-node ndf overlay, host stays `ndf=3`, `LadrunoEmbeddedRebar` couples the 3 translations; **(3) twist** = try the existing `zeroLength`+SP first (ADR 20 D6 option 1), **no new fork C++ class tag** unless that proves insufficient. Recorded in `internal_docs/plan_rebar_p5.md` §B0 + the recommended sequence and the `handoff_rebar_cage.md` Track-B table. B1 (`element="beam"` rebar) is now execution-ready as a separate effort.

### FIXED — retired the false `H5ReinforceDeviationWarning` on reinforced `apeSees.h5` decks (ADR 0067 P5.1, A4 minimal)

`apeSees(fem).h5(path)` no longer warns that "the H5 deck will be missing its embedded reinforcement" — a claim that became **false** once A1 (#706) made `fem.elements.reinforce_ties` round-trip through the **neutral** zone. Because `apeSees.h5` writes that neutral zone into the *same* archive as the `/opensees` deck zone, a reinforced `model.h5` already carries its ties: it round-trips via `FEMData.from_h5` → `apeSees(fem).tcl()/py()/run()` (the forward path re-runs `emit_reinforce_ties`). `H5Emitter.embedded_rebar` is now a **silent** deck-zone no-op and the `H5ReinforceDeviationWarning` class (+ `__all__` entry + emission) is removed. A dedicated `/opensees/constraints/reinforceTie` deck record + `OpenSeesModel.build()` deck-replay (the "A4 full" item) stays **deferred** — not needed for any cage workflow, and gated behind the broader fact that `_replay_into` does not replay MP constraints either (documented in `internal_docs/plan_rebar_p5.md` §"A4 full"). Locked by `tests/opensees/unit/test_reinforce_emit.py::test_h5_defers_deck_zone_without_warning` (no warning + no deck-zone reinforce record) and `tests/test_reinforce_composite.py::test_apesees_h5_deck_roundtrips_ties_via_neutral_zone` (a reinforced `apeSees.h5` → `read_fem_h5` recovers all ties, no warning). Reinforce + opensees-h5 + rebar suites green (the two failing `tests/opensees/h5` cases are the pre-existing openseespy-Windows-DLL `ImportError`, not this change).
### ADDED — cross-partition `equationConstraint` replication under OpenSeesMP (ADR 0068 P5, Open item 2)

An `enforce="equation"` tie (`g.constraints.tie(..., enforce="equation")`) that straddles a partition boundary now emits under partitioned / OpenSeesMP output instead of fail-louding (`NotImplementedError`). `_plan_rank_constraints` (`opensees/_internal/build.py`) replicates the tie on **every rank that owns the slave OR any master** — the `rigidDiaphragm` replicate-on-owning-ranks rule, *not* the single-canonical-host-rank element rule the penalty (`ASDEmbeddedNodeElement`) tie uses (which, applied to a domain-level `EQ_Constraint`, would drop the constraint on slave-owning ranks and falsely error on a partition-cut master face — the adversarial finding that motivated the original fail-loud). A new `_RankConstraintPlan.equation_records` lane collects the per-rank ties; each owning rank ghost-declares the foreign slave/master nodes first (reusing `_add_foreign_or_phantom`) and emits byte-identical `equationConstraint` rows via `_emit_one_interpolation` → `_emit_equation_tie`. Because the equation route allocates no element tag, replicating it across ranks is tag-stream-neutral (penalty-tie tag determinism is unchanged). The cross-rank EQ-capable handler (`LadrunoProjection`/`Lagrange`, auto-emitted per Open item 1) resolves the constraint graph across subdomains. Locked by `tests/opensees/integration/test_emit_partitioned_replicate_on_both.py` (`test_cross_rank_equationConstraint_replicates_on_owning_ranks`: 3 per-DOF rows on both owning ranks, byte-identical, foreign-node decls precede them; `test_equationConstraint_single_owning_rank_no_spurious_replication`: a rank-local tie emits only on its owning rank). ruff clean + mypy baseline 0; the partitioned integration sweep stays green. The emission logic is unit-verified without MPI; a live OpenSeesMP run is the final confirmation when a multi-rank fork build is available.

### CHANGED — equation-tie handler auto-emit auto-detects implicit vs explicit (ADR 0068 P5, Open item 1)

When an `enforce="equation"` tie is present and the user declared no constraint handler, `BuiltModel._maybe_auto_emit_constraint_handler` now picks the EQ-capable handler by the **registered integrator** instead of always emitting `Lagrange`: an **explicit** integrator (`CentralDifference`/`CentralDifferenceLadruno`/`ExplicitBathe`/`ExplicitBatheLNVD`/`ExplicitDifference`) auto-emits the fork **`LadrunoProjection`** (Δt-neutral, momentum-conserving — a Lagrange multiplier's massless DOF would break the explicit mass solve); **implicit / no integrator** keeps **`Lagrange`** (exact). The classifier is the new shared `_is_explicit_integrator`, refactored out of `apeSees._check_explicit_solver_compat` so the two call sites can't drift. Declaring a handler is still the override (respected as before), so no `tie_handler=` kwarg was needed; INV-4 still fail-louds on `Transformation`/`Auto` + an equation tie; and a *soft* `OpenSeesAutoEmitWarning` now fires when a user explicitly pairs `Lagrange` with an explicit integrator + an equation tie (the massless-multiplier hazard). Locked by `tests/test_constraint_emission_phase7b.py` (`TestEquationTieHandlerAutoDetect`: explicit→LadrunoProjection, implicit→Lagrange, no-integrator→Lagrange, user-Lagrange+explicit→warns-but-respected). ruff clean + mypy baseline 0 held.
### FIXED — harden embedded-reinforcement tie H5 (de)serialization (ADR 0067 P5.1, adversarial review)

A multi-agent adversarial review of the P5.1 work (A1 H5 persistence + A2/A3 compose) confirmed a small cluster of serialization-boundary gaps; the rest of the review verified the design is sound (snapshot_id correctly excludes ties, compose tag-offset/accumulation/cross-Part guard all correct). Fixes: `_encode_reinforce_tie` now **fails loud** on a malformed `ReinforceTieRecord` instead of writing a record that would decode to garbage or emit an invalid `LadrunoEmbeddedRebar` — it rejects empty `host_nodes` (a tie must couple ≥ 1 host node), an empty-but-non-`None` `weights` array (keeps the `None` vs `[]` distinction unambiguous), and a `weights` length that doesn't match `host_nodes` (the documented "parallel" invariant). `_decode_reinforce_tie` mirrors the length check defensively so a corrupted file is refused loudly rather than silently emitting a wrong element. The cross-Part guard (`_guard_reinforce_cross_part`) gains a comment documenting its "node in no named Part is unconstrained" semantics (fires only on ≥ 2 distinct named Parts — avoids false positives on partial Part maps while still catching the real host-nodes-split-across-Parts case). Locked by `tests/mesh/test_reinforce_tie_h5_roundtrip.py` (encode rejects empty host / empty-array weights / mismatched weights; a pre-2.15.0 (2.14.0, no-`/reinforce_ties`-group) file still reads within the two-version window → empty ties). Known open items (documented, not regressions): partitioned-tie dedup, and bond-name re-emit requires the (namespace-prefixed) `LadrunoBondSlip` material to be declared after compose but before `apeSees.build()`.

### ADDED — tie-force recovery for equation-tied interfaces (ADR 0068 P5)

The non-matching equation-tie route (`g.constraints.tie(..., enforce="equation")`) can now report the interface force it carries — the apeGmsh analogue of LS-DYNA `*DATABASE_NCFORC` — via the fork OpenSees tie force `f = M(a_raw - a_proj)` (the `LadrunoProjection` handler's projection constraint force, ADR-30 P3/P4, already shipped on `ladruno`). Two routes, both minimal because the plumbing already existed: **(1) live query** — `apeSees.ladruno_projection_tie_force(node, dof)` (delegates to a new fork-gated `LiveOpsEmitter.ladruno_projection_tie_force`, mirroring `critical_time_step`) returns the tie force at a node/DOF from the last projection step of a prior live `analyze(...)`; fails loud (`BridgeError`) before any live run, and `RuntimeError` on a stock build with no `ladrunoProjectionTieForce`. Works implicit *and* explicit. **(2) recorder readback** — `ops.recorder.Ladruno(nodal_responses=("constraintTieForce",))` already emitted the `-N constraintTieForce` channel verbatim; the only gap was reading it back, closed by a single `_NODAL_RESULT_NAME_MAP` entry (`CONSTRAINT_TIE_FORCE` → `constraint_tie_force`) so a `.ladruno` file is consumable via `results.nodes.get(component="constraint_tie_force_x")` (vec3/node, explicit-only — the channel is scattered each commit by `CentralDifferenceLadruno`). Locked by `tests/test_tie_force_helper.py`: the readback mapping + the `BridgeError` gate run anywhere; the live legs (query == exact `F·m₂/(m₁+m₂)` tie force, wrong-handler guard, recorder round-trip) need an openseespy built from `ladruno` ≥ ADR-30 P3/P4 and skip cleanly on older builds. No declarative-vocabulary change (deferred); emission stays the documented `Ladruno(nodal_responses=…)` passthrough.

### CHANGED — `g.rebar` handoff refreshed for P5 Track-A progress (`internal_docs/handoff_rebar_cage.md`)

Brought the handoff current with the P5 Track-A work: the status header now reflects bundled bars / the `wall` generator / mesh-native curved geometry / and the composed-Part keystone (A1 #706 + A2+A3 #707) shipped; the P5 section is restructured into Track A (A1 ✅ neutral-H5 tie persistence, A2+A3 ✅ compose carry + cross-Part guard, A4 ⬜ opensees-deck follow-on) and the B0-gated Track B (beam dowel + twist), pointing at `internal_docs/plan_rebar_p5.md`; the partitioned-tie dedup open item and the P5 test locations (`tests/mesh/test_reinforce_tie_h5_roundtrip.py`, `test_compose_reinforce_ties.py`) are noted. Docs only.

### ADDED — `g.compose` carries embedded-reinforcement ties + cross-Part guard (ADR 0067 P5.1, A2 + A3)

Building on the A1 neutral-H5 tie persistence, `g.compose(...)` now **rewrites and merges** `fem.elements.reinforce_ties` so a composed-Part cage keeps its reinforcement. Each source's ties are offset-rewritten through the existing `tag_rewrite_spec` (`rebar_node` scalar + `host_nodes` array shifted by the module's tag offset; `name` + `bond` namespace-prefixed with the module label) and appended to the merged model; the host's own ties are preserved across the merge (previously the rebuilt `ElementComposite` dropped them). The geometric arrays (`weights`, `direction`) are left untouched by the rewrite. Both tracks ship in one PR to close the silent-corruption window the critique flagged: a new **cross-Part guard** (`_guard_reinforce_cross_part`) raises `ComposeReinforceCrossPartError` when a tie's `rebar_node` + `host_nodes` span two different source Parts — an embedded tie's bond assumes a co-meshed host, so the offset rewrite would otherwise produce broken conformal topology. (The guard is deliberately **not** extended to tied-contact `SurfaceCouplingRecord`s, which are designed to bridge surfaces and may legitimately span Parts.) The guard is a no-op for a source with fewer than two Parts or no Part map. Locked by `tests/mesh/test_compose_reinforce_ties.py` (carry + round-trip, constant tag offset with geometry preserved, host-tie preservation when composing a plain module, and the cross-Part guard: raises across Parts / passes same-Part / no-ops without Parts). Compose + mesh suites green.

### ADDED — embedded-reinforcement ties round-trip through the neutral `model.h5` (ADR 0067 P5.1, neutral schema 2.15.0)

`fem.elements.reinforce_ties` — the g.reinforce `LadrunoEmbeddedRebar` couplings (one `ReinforceTieRecord` per rebar node) — now **persist** through `FEMData.to_h5` / `from_h5`, into a new dedicated `/reinforce_ties` group (a symmetric-compound dataset via `reinforce_tie_payload_dtype`, modeled on `surface_coupling`). Previously `to_h5` **dropped** them with a deferral warning, so a reinforced model silently lost its reinforcement on round-trip — this was the keystone blocking composed-Part cage libraries (P5.1 / "do_first" of the P5 plan). The bump is the **neutral** zone `NEUTRAL_SCHEMA_VERSION 2.14.0 → 2.15.0` (the composed-Part library round-trips through the neutral zone; the opensees deck zone 2.19→2.20 is a separable follow-on, A4); the two-version reader window auto-tolerates 2.14.x (which simply has no `/reinforce_ties` group). The group is **omitted entirely** when there are no ties, so a tie-free model stays byte-identical and its `snapshot_id` is unchanged — and `snapshot_id` deliberately does **not** hash the tie overlay (consistent with how constraints are already excluded), so even a reinforced model round-trips with an identical `snapshot_id`. Optional scalars use NaN/`""` sentinels with `has_*` flags for the geometric vlen/fixed fields, so `None` survives distinct from empty; the `bond` material is stored by **name** (Option B; resolved to a tag at re-emit). Locked by `tests/mesh/test_reinforce_tie_h5_roundtrip.py` (perfect-bond + bond-by-name round-trip with field-by-field equality, tie-free group omission + snapshot stability, reinforced-model snapshot stability, no deferral warning, version stamp); the obsolete `test_to_h5_warns_ties_not_persisted` is flipped to assert persistence. Build on a real non-matching mesh (no fork build needed).

### ADDED — `g.rebar` P5 implementation plan (`internal_docs/plan_rebar_p5.md`)

A forward-looking, critique-hardened implementation plan for the three externally-blocked `g.rebar` P5 items (composed-Part cage libraries via H5 tie persistence; `element="beam"` dowel-action rebar; bar-axis twist stabilization). Produced by a survey → synthesize → critique multi-agent workflow. Records the verified load-bearing finding — the keystone is the **neutral** H5 zone (`NEUTRAL_SCHEMA_VERSION 2.13.0 → 2.14.0`, `reinforce_ties` persistence modeled on `surface_coupling`), **not** the opensees deck zone — and lays out Track A (A1 persist/read → A2+A3 compose teach-in with the cross-Part guard in one PR → A4 deck follow-on) and Track B (B0 human-decision gate → B1 per-segment `vecxz` + `ndf=6` → B2 twist stabilizer → B3 ghost-node persistence), with the open `snapshot_id`/lineage decision called out as the A1 pre-gate. Docs only — no code.

### ADDED — `g.rebar` mesh-native curved geometry (`Path(curve=…)` + `circular_column(true_arc=True)`)

The `Path` L1 spec gains a `curve` kind — `"polyline"` (default, straight segments), `"arc"` (true circular arcs about a new `arc_center`), or `"spline"` (one C2 interpolating spline through the points) — so a bar/stirrup can be authored as **mesh-native curved geometry** instead of a fixed authored polygon. The L2 emitter (`_emit_curve`, replacing `_emit_polyline`) welds each kind into gmsh `add_arc` / `add_spline` / `add_line` accordingly. `g.rebar.circular_column(true_arc=True)` uses it: the discrete hoops become true circular arcs and the spiral a spline, so the mesher seeds nodes on the true curve at the active element size rather than the `n_segments` polygon (deterministic polygon stays the default). Hand authoring exposes it via `g.rebar.bar(..., curve="arc", arc_center=…)` / `curve="spline"` and `g.rebar.stirrup(...)`. **Caveat (documented):** the realised FE elements are still straight 2-node chords — OpenSees has no curved line element — so `true_arc` upgrades *node placement / curve fidelity*, not the element type. Also fixes a latent stale-metadata leak: arc-center construction points are now popped from the apeGmsh registry when `occ.remove`d (previously only the OCC entity was dropped, which tripped the geometry validator at mesh time — this would have hit true-arc *hooks* under conformal meshing too). Locked by `tests/rebar/test_rebar_true_arc.py` (8 tests: L1 curve validation + round-trip, hoop-arc / spiral-spline generation, polygon default, embedded place, conformal mesh end-to-end, hand-authored arc/spline bars). 175 rebar + guard tests green.

### ADDED — `g.rebar.wall` — RC wall cages (the 4th standardized member)

`g.rebar.wall(length=, thickness=, height=, cover=, vertical_db=, vertical_spacing=, horizontal_db=, horizontal_spacing=, curtains=2, …)` builds a reinforced-concrete **wall** cage (vertical panel, plane x-z, thickness along y) — the fourth standardized member alongside `column()` / `beam()` / `circular_column()`. Vertical bars run along the height (spaced along the length) and horizontal bars run along the length (spaced up the height), in **one or two curtains**: `curtains=2` (default) places a layer `cover + db/2` in from each face, `curtains=1` a single layer at mid-thickness. Walls are spaced, not counted — `vertical_spacing` / `horizontal_spacing` are the max centre-to-centre pitches, rounded to an even division between the `end_cover` insets (new `_positions_by_spacing` helper). For a double curtain, `crossties=True` (default) ties the two curtains together through the thickness on a grid (`crosstie_spacing`, default twice the coarser bar spacing) with a 135° seismic + 90° hook resolved from the standard (ACI 318 §11.7.4 / §18.10.2.7); a single-curtain wall warns and emits none. Bars carry `role="vertical"` / `"horizontal"` / `"crosstie"`; each is an independent truss/embedded member, so `place()` (conformal or embedded) and `per_member_coupling` work unchanged. Vertical and horizontal bars of a curtain are idealised co-planar at the vertical-bar depth (a truss model). Boundary-element confinement is out of scope — model a confined wall end with `column()` over the boundary zone. Locked by `tests/rebar/test_rebar_wall.py` (7 tests: double-curtain grid + curtain planes, through-thickness cross-ties, single-curtain mid-thickness, single-curtain crosstie warning, embedded placement as independent members, validation, conformal mesh end-to-end). 167 rebar + guard tests green.

### ADDED — `g.rebar.beam` detailing parity: overlapping-hoop style + full mismatched cross-tie support

Two beam-detailing improvements bringing `g.rebar.beam()` to column-level parity. (1) **`confinement_style="overlapping_hoops"`** — the wide-beam sibling of the column knob: instead of straight vertical cross-tie legs, the cross-section is tiled with closed, overlapping rectangular cell-stirrups (one per adjacent bar column, the bottom edge on the bottom bars and the top edge on the top bars, so neighbours share a vertical edge and every bar sits at a hoop corner), alongside the outer perimeter stirrup. Cell-stirrups are `role="tie"` (twin-tail closure + seismic spacing apply). It needs equal top/bottom bar counts (a regular grid) and raises otherwise, directing to `"crossties"`. Default stays `"crossties"`. (2) **Cross-ties now support every interior bar on a count mismatch** — `_beam_crossties` ties each interior bar (top and bottom) to the *nearest* bar on the opposite face rather than only the index-aligned `min(n)−2` interior pairs. When the counts/positions align the legs stay vertical (unchanged); when they differ each interior bar still gets a leg to its nearest opposite bar (slightly inclined), so none is left unsupported, with duplicate bottom↔top pairs coalesced. The count-mismatch warning is retained (now noting the legs may be inclined). Locked by `tests/rebar/test_rebar_generators.py` (cell tiling count + closed 4-corner y-z rings, equal-count guard, embedded place, style validation, all-interior-supported-on-mismatch); two existing beam tests updated for the more-complete behaviour. 153 rebar tests green.

### ADDED — `g.rebar` bundled longitudinal bars (ACI 318-19 §25.6)

`BarLayout(bundle=2|3|4, bundle_pattern="auto"|"line"|"triangle"|"square")` replaces each longitudinal position in `g.rebar.column()` / `circular_column()` / `beam()` with a contact **bundle** of that many parallel bars — the geometry layer realises a bundle as that many individual offset bar lines, each a distinct truss/embedded member (so coupling and detailing are unchanged). The cluster is offset rigidly in the bar's cross-section frame: the outer bars sit on the nominal cover line and the cluster stacks inward toward the section interior (no bar is shallower than the single-bar position along the inward normal; at a true corner a tangentially-spread pair leans toward a face by at most √2/2·d_b — inherent to bundling, so inset for the equivalent diameter √n·d_b if strict corner cover matters). `"auto"` maps the count → line (2, side-by-side) / triangle (3) / square (4, 2×2); an explicit pattern must match the count. Cross-ties and hoops still engage the outer bar at the nominal position. A new hand-authoring helper `g.rebar.bundle(points, *, n, db, material, toward, pattern="auto", spacing=None, ...)` returns a tuple of `Bar` for free-form bundled bars (the cluster leans toward the `toward` interior point; `spacing` is the centre-to-centre offset, default the bar diameter — a contact bundle). Validation (ACI §25.6.1.1): 1–4 bars, the pattern matches the count, and a `#14`/`#18` bar is capped at 2 per bundle; the generators also fail-loud when the inward stack would cross the section centre. Locked by `tests/rebar/test_rebar_bundles.py` (17 tests: L1 validation, column/circular/beam expansion + cover-line geometry + inward stacking + fit guards, hand-authoring, embedded placement as independent members). 148 rebar tests green.

### CHANGED — `g.rebar` handoff refreshed for the shipped detailing arc (`internal_docs/handoff_rebar_cage.md`)

Brought the handoff doc current with PRs #687–#693: status header now reflects the full ACI detailing arc on `main` (135 tests), the L2 row + quickstart list `circular_column()`, limitation #1 notes the `confinement_style="overlapping_hoops"` alternative, a new item #5 records circular columns (hoops/spiral) shipped + the genuine remaining minor gaps (polygon-approx circles, bundled bars, no beam overlapping-hoops), the test count is 135, and a working note warns to scope `ruff --fix` to exact files (not a directory). Docs only.

### ADDED — `g.rebar` reinforcement-cage user guide (`internal_docs/guide_rebar.md`)

A user-facing guide for the now-complete `g.rebar` cage-authoring API: picking a `DetailingStandard` (Raw / ACI318 / ACI318_seismic) and the `BarCatalog` unit knob; the rectangular `column()` (cross-ties vs `confinement_style="overlapping_hoops"`, ACI §18.7.5 seismic confinement auto-derive), `beam()` (supplementary legs + §18.6.4 hoop zone), and `circular_column()` (hoops vs spiral) generators; `place()` conformal vs embedded coupling + `per_member_coupling` + `twin_tail`; hand-authoring bars/stirrups via the fluent builder; and the documented limits. Consolidates the §8/§3 detailing runway (PRs #687–#692) into one reference. Docs only — no code change.

### ADDED — `g.rebar.circular_column` — circular RC column cages (hoops or spiral) (ADR 0067 §8)

`g.rebar.circular_column(diameter=, height=, cover=, n_bars=, bar_db=, ties=...)` builds a circular column cage — the round sibling of the rectangular `column()`. `n_bars` longitudinal bars are evenly spaced on a circle (radius `D/2 − cover − tie − db/2`), and the transverse reinforcement is either discrete **circular hoops** (`spiral=False`, default — one closed ring per tie level on radius `D/2 − cover − tie/2`, densified in the hinge zones like the rectangular column) or a continuous **spiral** (`spiral=True` — one helix end-to-end at pitch `ties.spacing`). Rings/helix are polygon-approximated with `n_segments` sides per turn (default 24). Circular confinement laterally supports every bar, so there are no cross-ties. When the standard is `ACI318_seismic` and `ties` omits the hinge fields, the §18.7.5 confinement zone is auto-derived (`h_x` = the bar chord spacing on the circle). The spiral is emitted as a single `role="spiral"` truss member (a chain of segments along the helix); hoops are `role="tie"` stirrups (twin-tail closure applies). Bars/hoops are interior to the host, so both couplings mesh; placement reuses the same `place()` path. Locked by `tests/rebar/test_rebar_generators.py` (bars on circle + even spacing, closed-ring hoops on the tie radius, single-helix spiral spanning the height, seismic densification, embedded place into a cylinder, validation). 131 rebar tests green.

### ADDED — `g.rebar.column` overlapping cell-hoop confinement style (ADR 0067 §8)

`g.rebar.column(..., confinement_style="overlapping_hoops")` is an alternative to the default straight cross-ties for laterally supporting the intermediate (`n>2` per face) longitudinal bars: instead of straight cross-tie legs, the core is tiled with a grid of closed, overlapping rectangular **cell-hoops** — one per adjacent 2×2 bar block, corners on the longitudinal bar centerlines — so neighbouring cells share an edge (overlap) and every bar sits at a hoop corner. The outer perimeter hoop is emitted in both styles. This is the wide-section detail where a single perimeter hoop plus cross-ties would otherwise carry many long unsupported legs. The cell-hoops are ordinary `role="tie"` stirrups (twin-tail closure + the §18.7.5 seismic spacing apply); embedded coupling is robust, conformal forms shared-edge T-junctions needing `make_conformal`. Default `confinement_style="crossties"` is unchanged. Locked by `tests/rebar/test_rebar_generators.py` (cell tiling count + closed 4-corner rings on bar centerlines, embedded place, style validation). 129 rebar tests green.

### ADDED — `g.rebar` twin-tail stirrup/hoop closure (ADR 0067 §3)

A closed stirrup/tie is bent from a single straight bar, so its two free ends both terminate in a hook that overlap at the seam corner. `g.rebar.place(...)` now emits that **twin-tail** seam by default — both ends of a closed stirrup carry the closure hook — instead of the previous simplified single closure hook. The second tail mirrors the resolved closure detail (e.g. a seismic 135° hook) onto the start end, anchored at the same seam node with its own outward tangent so the two tails fan into the core. New `twin_tail=True` knob on `place()` (set `twin_tail=False` for the single-hook simplification). A stirrup with an explicit start hook, or one whose closure was dropped because no `DetailingStandard` is set, is unaffected; cross-ties (already two-ended) and longitudinal bars are untouched. Locked by `tests/rebar/test_rebar_hooks.py` (6 emitted curves = 4 ring legs + 2 tails with twin-tail; 5 = 4 + 1 with `twin_tail=False`). 122 rebar tests green.

### ADDED — `g.rebar.beam` auto-derives the ACI 318 §18.6.4 seismic hoop confinement zone (ADR 0067 §8)

The beam sibling of the column confinement auto-derive. When the active standard is `ACI318_seismic` and the `stirrups` `TieLayout` leaves `hinge_spacing`/`hinge_length` unset, `g.rebar.beam(...)` now auto-derives the special-moment-frame hoop zone instead of uniform spacing: the confined length `2h` (twice the member depth, §18.6.4.1) and the dense hoop spacing min(d/4, 6·d_b of the smallest longitudinal bar, 6 in) (§18.6.4.4), with `d` taken to the tension-bar centroid. The user's `stirrups.spacing` governs outside the hinge zones; an explicit `TieLayout(hinge_length=, hinge_spacing=)` overrides verbatim; a non-seismic standard (`ACI318`/`Raw`) stays uniform (no-op). The ACI numbers live on new `ACI318_seismic.beam_confinement_length(...)` / `beam_confinement_spacing(...)` methods (unit-safe via the catalogue), keeping the beam generator free of code constants. A warning reports the derived length/spacing. Locked by `tests/rebar/test_detailing.py` (2h, governing-term + 6 in cap + unit-safety + guards) and `tests/rebar/test_rebar_generators.py` (auto-derive densifies, non-seismic stays uniform). 125 rebar tests green.

### ADDED — `g.rebar.column` auto-derives the ACI 318 §18.7.5 seismic confinement zone (ADR 0067 §8)

When the active standard is `ACI318_seismic` and the `TieLayout` leaves `hinge_spacing`/`hinge_length` unset, `g.rebar.column(...)` now **auto-derives** the special-moment-frame confinement zone instead of falling back to uniform spacing (the previous behaviour, which only fired a warning). The confined-end length `l_o` = max(member depth, clear span / 6, 18 in) (§18.7.5.2) and the dense tie spacing `s_o` = min(¼·min member dimension, 6·d_b of the smallest longitudinal bar, 4 + (14 − h_x)/3 in clamped to [4, 6] in) (§18.7.5.3) are computed from the section geometry — `h_x` is the max centre-to-centre spacing of the laterally supported bars (the cross-tie/bar spacing, capped at 14 in; the full clear distance when `crossties=False`). The user's `ties.spacing` is used outside the hinge zones; an explicit `TieLayout(hinge_length=, hinge_spacing=)` overrides the code rule verbatim, and a non-seismic standard (`ACI318`/`Raw`) leaves the spacing uniform (no-op). The two ACI numbers live on the standard as new `ACI318_seismic.confinement_length(...)` / `confinement_spacing(...)` methods (unit-safe via the catalogue — the `s_o` equation is evaluated in inches), keeping the column generator's geometry logic free of code constants. A warning reports the derived `l_o`/`s_o`/`h_x` so the values are visible. Locked by `tests/rebar/test_detailing.py` (governing-term + s_o caps + unit-safety) and `tests/rebar/test_rebar_generators.py` (auto-derive densifies, explicit override, non-seismic stays uniform). 115 rebar tests green.

### ADDED — `g.rebar` cross-ties / supplementary legs for intermediate bars (ADR 0067 §25.7.2.3)

`g.rebar.column(...)` and `g.rebar.beam(...)` now generate ACI 318 §25.7.2.3 **cross-ties** that laterally support the intermediate (`n>2` per face) longitudinal bars — closing the largest v1 detailing gap (previously a single perimeter hoop was emitted with a warning). A **column** gets one cross-tie per intermediate bar at every tie level: a straight transverse leg spanning the section between the two opposite-face bars it engages, with a 135° seismic hook on one end and a 90° hook on the other, **alternated end-for-end** on consecutive levels (ACI 318 §18.7.5.2). A **beam** gets a vertical supplementary leg at every stirrup station for each index-aligned interior top/bottom bar pair (unpaired bars on a count-mismatched face are skipped with a warning). Cross-ties use the tie bar size/material, carry `role="crosstie"` (so `per_member_coupling={"crosstie": …}` can route them), and resolve their hook tails from the cage's `DetailingStandard` at `place` time (seismic-hoop kind; dropped with a warning when no standard is set, like a stirrup closure). New `crossties=True` knob on both generators (set `crossties=False` for the bare perimeter hoop). The end-hook resolution is now **role-aware** — transverse members (`tie`/`crosstie`/`hoop`/`stirrup`) detail their end hooks as seismic hoops and tolerate a missing standard; longitudinal-bar development hooks stay primary + required. `coupling="embedded"` is robust; conformal cross-ties form bar/tie T-junctions that need `make_conformal`. Locked by `tests/rebar/test_rebar_generators.py` (interior-bar support, end-for-end alternation, opt-out, embedded place with `ACI318_seismic`, beam aligned legs + count-mismatch warning). 107 rebar tests green.

### ADDED — DRM ASD absorbing boundary via `add_DRM_box_from_h5drm(absorbing=True)` (ADR 0066, D-4)

`g.parts.add_DRM_box_from_h5drm(..., buffer=N, absorbing=True)` (requires `buffer >= 1`) now wraps the buffered DRM box in a one-element **ASD absorbing skin** — a btype-tagged ghost ring on the four sides + bottom (never the free surface) — the production-SSI boundary (ADR 0054). The skin sits on the buffer's **outer, NON-dataset** faces (≥1 buffer layer between it and the DRM `b` shell), so H5DRM never sweeps the boundary into the effective-force set (`H5DRMLoadPattern.cpp:580`). It is built in the validated **z-down** frame (the same sliced-block machinery as the buffer, with the outer ring classified into btypes via the shared `_btype_for` — `plane_wave_box` can't be reused directly because it builds z-up). The result's `skin` is a complete `AbsorbingSkinResult` that drops straight into the existing tested facade: `ops.element.absorbing_boundary(skin=result.skin, material=…)` + the staged `s.activate_absorbing()` gravity→absorbing flip; assign soil/`stdBrick` over `result.domain_pg` (inner + buffer). With `absorbing=False` the boundary stays the bare buffer faces (apply `ops.fix` for the validated fixed far field). Locked by `tests/parts/test_h5drm_box.py` (17 btypes incl. the bottom subset, conformal node count + station coincidence with the skin present, and bridge composability — `absorbing_boundary(skin=…)` fans 17 `ASDAbsorbingBoundary3D` specs). **ADR 0066 runway D-1→D-4 complete.** Plan: `internal_docs/plan_drm_h5drm_adr0066.md`.

### ADDED — DRM exterior buffer via `add_DRM_box_from_h5drm(buffer=N)` (ADR 0066, D-3)

`g.parts.add_DRM_box_from_h5drm(..., buffer=N)` now wraps the inner DRM box in `N` exterior soil layers on the four sides + the bottom (never the free surface), at the same grid spacing — the stable path for a real run (a free DRM box diverges: the residual excites rigid-body modes). It is built as **one block sliced at the inner breakpoints** (the proven `plane_wave_box` machinery), so the inner/buffer interface is **conformal by construction** and the inner sub-volume still lands nodes exactly on the dataset stations, while the buffer hexes carry only **non-dataset** nodes — so H5DRM excludes the buffer/boundary elements from the effective-force set (`H5DRMLoadPattern.cpp:580`, the all-nodes-DRM rule). The `DRMBoxFromH5Result` gains `layers`, `buffer_pg`, `domain_pg` (inner+buffer roll-up — the target for material / `stdBrick` assignment), and `exterior_pgs` now reports the **outer** model-boundary faces (sides+bottom). **API note:** this folds the buffer into the existing builder (`buffer=` param, backward-compatible — `buffer=0` is the D-2 behavior) rather than the ADR's sketched separate `g.drm_buffer(drm)` call, because extending an already-transfinite box via `fragment` is fragile and a rebuild would discard the D-2 geometry. **Applying the boundary stays bridge-side** (not a builder param): `ops.fix(pg=…)` for a fixed far boundary, Lysmer/ASD elements for absorbing — the `asd` staged-flip path is D-4. Locked by `tests/parts/test_h5drm_box.py` (conformal node count == the extended structured grid, station coincidence with buffer, soil/buffer/domain + outer-face element counts). Plan: `internal_docs/plan_drm_h5drm_adr0066.md`.

### ADDED — `g.parts.add_DRM_box_from_h5drm` dataset-keyed DRM box (ADR 0066, D-2)

`g.parts.add_DRM_box_from_h5drm(h5drm=, crd_scale=1000.0, name=, names=, apply_transfinite=True)` — reads a ShakerMaker-style `.h5drm` DRM dataset and builds, in the live session, a single transfinite hex soil box whose nodes land EXACTLY on the dataset stations (so OpenSees' H5DRM node-matching is trivial — the fork study's "98/98 nodes" property). It validates that the stations form a complete, uniform, isotropic regular grid, centres the box on the file's `drmbox_x0` (z-down, metres) so the matching `ops.pattern.H5DRM(...)` defaults reproduce the coordinates exactly, and tags the soil **volume** PG plus the six outer **boundary-face** surface PGs (`xmin`/`xmax`/`ymin`/`ymax`/`top`/`bottom` — the dataset "b" shell, cross-checked against the `internal` flag with a `WarnDRMGridIrregular`). Returns a `DRMBoxFromH5Result` carrying the PGs (incl. `free_surface_pg`, the sides+bottom `exterior_pgs`, and a `boundary_all_pg` roll-up), the **frame contract** (`crd_scale` / identity `transform` / zero `x0` / `center`), and the grid descriptor (`origin` / `spacing` / `counts`). Geometry + PGs only — assign the soil material + `stdBrick` elements via the bridge (`ops.element.stdBrick(pg=result.soil_pg)`), the dataset-keyed sibling of the parametric `g.parts.add_DRM_box`. The exterior buffer (`g.drm_buffer`) is D-3. Plan: `internal_docs/plan_drm_h5drm_adr0066.md`.

### CHANGED — partitioned coupling/embedded split-across-ranks fail-loud now names the recovery path

When a `kinematic_coupling` (RBE2), `distributing_coupling` (RBE3), or `embeddedNode` (`ASDEmbeddedNodeElement`) has its required node set fragmented so that no single OpenSeesMP rank can assemble the one element, the build still fails loud (unchanged, deliberate — ADR 0027 treats this as a partitioner-input condition, not a recoverable one), but the two messages (`_canonical_coupling_rank` / `_canonical_host_rank` in `opensees/_internal/build.py`) now point at the remedy: re-partition **from the mesh phase** with `g.mesh.partitioning.partition_explicit(...)`, placing every required node's incident elements on one rank. New `guide_partitioning.md` §7.1 documents the case end-to-end — why element-backed couplings need a single canonical rank (vs idempotent `equalDOF`/`rigidDiaphragm` replication), why a boundary node is fine but a node-only contact between two bodies can split (METIS cuts by shared faces/edges, not nodes — `_build_dual_graph`), the `partition_explicit` recovery recipe, the chain-phase-freeze caveat (you must rebuild from the mesh phase, ADR 0038), and a note that the super-vertex-contraction partitioner that would make clusters indivisible to METIS is feasible-but-unbuilt. Docs + message strings only; no behavior change (the `split across partitions` match substring is preserved — 13/13 affected tests pass, mypy clean).
### ADDED — `ops.pattern.H5DRM` typed DRM load pattern (ADR 0066, D-1)

`ops.pattern.H5DRM(h5drm=, factor=1.0, crd_scale=1000.0, distance_tolerance=1.0, transform=None, x0=(0,0,0))` — a typed, field-carrying load pattern that drives a soil box with a regional incident wavefield read from an `.h5drm` dataset (e.g. a ShakerMaker synthetic) via OpenSees' H5DRM pattern. It owns the error-prone **frame handshake** validated in the OpenSees fork DRM study (ADR 0066 / PR #296): the defaults encode a model built centred at the lateral origin, z-down, in metres, so the default `crd_scale=1000` (km→m) with an identity `transform` and zero `x0` reproduces the dataset's station coordinates exactly. Emits the canonical 18-arg `pattern H5DRM tag file factor crd_scale dist_tol 1 T00..T22 x00..x02` line (Tcl + openseespy; `do_transform` always `1`) and round-trips through `model.h5` generically (no series, no body). Unlike every other pattern it has **no** `series=` — the motion history lives in the file. The 3-DOF≤8-node / no-base-input-mixing guards (which need the FEM snapshot) and the dataset-keyed box builder (`g.parts.add_DRM_box_from_h5drm`) + exterior buffer (`g.drm_buffer`) land in D-2/D-3. Plan: `internal_docs/plan_drm_h5drm_adr0066.md`.

### ADDED — finite-difference sensitivity driver (`apeGmsh.sensitivity`)

New `apeGmsh.sensitivity` sub-package — compute how a response changes with **any** model parameter (every damping channel included) by black-box finite differences on top of the `apeSees` bridge, with no solver edits and no analytic derivative. `Sensitivity.from_apesees(fem, build=, params=[Param(...)], response=Response(...))` builds the deck via your `build(ops, params)`, runs a transient, reads the response through `Results`, and differentiates it: `.gradient()` returns the full vector for one or many parameters (cost `2N` central / `N+1` forward solves), `.step_study()` exposes the step-size trust plateau, `.solve(target)` calibrates (1-D, damped Newton). The engine-free `FDSensitivity` core, `reduce_response` reducer, and `Param`/`Response` specs are fully unit-tested (`tests/sensitivity/unit/`); the one engine-coupled step (run-and-read) is a single replaceable `runner=` seam (`default_apesees_runner`). Ports the capability validated in the Ladruno OpenSees fork (PR #241 — all six damping channels demonstrated, SDOF + viscous matching closed forms to 0.00%, adversarially reviewed non-circular). See `internal_docs/guide_sensitivity.md`.

### REMOVED — `DeformedShapeDiagram` retired; deform-follow is now unconditional (ADR 0058 S4, ACCEPTED)

The final ADR 0058 slice closes the arc: the legacy `DeformedShapeDiagram` layer — which warped its own copy of the substrate and rendered an undeformed wireframe ghost — is **deleted**, because both of its jobs are now first-class concurrent-geometry features. A deformed shape is a geometry with deform enabled (the per-geometry DEFORM pump warps `scene.grid.points`, shipped S2b); the undeformed reference is the `add_reference_ghost` preset (shipped S3c). With the last self-warping layer gone, the **deform-follow contract is now unconditional**: `tests/viewers/test_deform_follow_contract.py` drops its sole `_EXEMPT` entry and the exemption mechanism itself (`test_exempt_list_is_live` retired) — every diagram class that paints on the substrate must follow the deformed configuration, with no opt-out to grow back. Session migration is fail-soft: a legacy session carrying a `deformed_shape` diagram drops just that spec on load with a clear `viewer.session/retired_diagram_dropped` log line (recognized-but-retired branch in `deserialize_spec`; no schema bump — the rest of the session's hierarchy loads intact). Also removed: the settings-tab deformed-shape panel, the demo block, and the deformed-shape test cases; the `"deformed_shape"` kind string survives only as the migration-recognition literal. **ADR 0058 is now Accepted — slices S0–S4 all shipped (#623–#651).**

### ADDED — reference-ghost preset + duplicate-with-layers (ADR 0058 S3c)

The two most-common concurrent-geometry asks become one gesture each. `GeometryManager.add_reference_ghost(geom_id)` is a preset verb (pure state, headless-testable) that composes the manager's own mutators into an EMPTY substrate-only geometry named `"<src> (reference)"`: deform off, nodes off, dimmed to the `GHOST_OPACITY` (0.3) module constant, with `offset` and `stage_id` copied from the source so the ghost overlays it exactly, and the source left active (the ghost is decoration — the underlying `duplicate()` flips the active pointer to the clone, and the preset restores it). It carries **no compositions** — a dimmed reference must not double the source's contours; layers on a ghost are a different gesture (duplicate-with-layers + manual deform-off). It coexists with `DeformedShapeDiagram`'s `_runtime_show_undeformed` ghost (that retirement is S4). `ResultsDirector.duplicate_geometry(geom_id)` is the director-level richer verb: it composes the manager's state-only `duplicate` with diagram reconstruction from each layer's `DiagramSpec`, replaying the exact `_apply_session` restore recipe (`kind_def(spec.kind).diagram_class(spec, results)`, `tag_map=` for `section_cut`), wrapped in `session_batch`. Composition membership is recorded **before** `registry.add` so attach resolves the clone's scene through the existing `scene_resolver` (`geometry_for_layer` hits the clone, not the active-geometry fallback); the active-composition pointer is restored by position; layers that fail to rebuild (`NoDataError`, unknown kind) are skipped fail-soft. The copy rule is "what's in the spec round-trips" — runtime overrides (`_runtime_show_undeformed`, live scale/colormap tweaks), probes/highlights, and manual per-scene hides are explicitly NOT copied. The outline geometry-row **Duplicate** action upgrades to the director verb; a new **Add reference ghost** action calls the manager verb inside `gesture_batch`. No session changes — a ghost is an ordinary geometry after creation (no linkage field, by design: rename-safe, independently deletable). Locked by `tests/viewers/test_scene_instances_s3c.py` (ghost: empty/dimmed/deform-off/nodes-off, offset+pin copied, source-active, unique names; duplicate: composition order, distinct instances with equal specs, registry growth, clone-scene resolution, runtime overrides not copied, fail-soft skip, active-comp restore by position, `section_cut` tag_map path; outline gesture wiring; qt end-to-end ghost pair + duplicate-with-layers).

### ADDED — per-geometry stage pin (ADR 0058 S3b)

A geometry can now be pinned to one stage while the viewport scrubs another — stage-A final configuration next to stage-B evolving, or step-by-step comparison of equal-length stages. `Geometry.stage_id` (`None` = follow the active stage) is owned by `GeometryManager.set_stage_pin(geom_id, stage_id)` firing the new granular `GEOMETRY_STAGE_PIN_CHANGED` (dispatcher matrix row STEP + DEFORM); `duplicate()` copies the pin. The time cursor stays **director-global** (per-geometry step cursors were ADR-rejected): a pinned geometry shows its stage's state at the global cursor clamped into the pinned range via the new `director.local_step_for_stage(stage_id)` (single-stage: `clamp(step, 0, n_steps(S)−1)`; combined mode: the cursor minus the stage's boundary start, so the pinned geometry plays its segment of the concatenated timeline and freezes outside it). The pin scopes three read paths under one *pinned-or-active* rule: the substrate deform read (`_read_deform_field(stage_id=)` / `_compute_deformed_pts`), the geometry's diagrams (the registry stamps a `stage_pin_resolver` — S2b `bar_prefix_resolver` mirror — consumed by the new `Diagram._effective_stage_id()`; an explicit per-diagram `spec.stage_id` still wins, and the `ReactionsDiagram` defensive override shares the same helper so the two can't drift; the STEP pump pushes per-diagram effective steps), and the per-scene `LAYER_STAGE` activation masks (`StageActivationController.mask_for_stage_id(sid)`, applied per geometry by `_materialize_scene` / `_sync_stage_layers` with the resync riding a RENDER-lane subscriber). A pin change re-attaches only that geometry's attached diagrams via a director typed-observer (GEOMETRY_REMOVED precedent — runs before the dispatcher pumps, so STEP/DEFORM land on fresh attachments). The shift-click time-history snap is now pin-aware (closing the third S2c deferral): `install_navigation` hands the picked prop through, the snap reads the hit geometry's grid, and the history tab is scoped to its stage pin (`director.read_history(stage_id=)` / `TimeHistoryPanel(stage_id=)`; pinned and active histories of the same node coexist as separate tabs). The geometry settings panel grows a Stage combo ("Follow active stage" + real stages; combined excluded; disabled on single-stage files). Session schema bumps to **v7** with an additive `GeometrySnapshot.stage_id` (legacy sessions read `None`; the restore applies the pin after the composition/layer loop so the reattach observer fires once against recorded membership). The Inspector's `read_at_pick` keeps reading the active stage (documented limitation); ghost preset + duplicate-with-layers are S3c. Locked by `tests/viewers/test_scene_instances_s3b.py` (mutator + event payload, matrix row + omnibus suppression, clamp helper incl. combined-mode windows, effective-stage composition incl. reactions, resolver stamping, filtered reattach walk, per-pin activation masks, v7 round-trip + legacy default, qt end-to-end pinned-hold/clamped-step/unpin-refollow over a real two-stage file).

### ADDED — per-geometry spatial offset (ADR 0058 S3a)

Geometries can now sit side by side: `Geometry.offset` (a rigid X/Y/Z translation in model units) is applied **at pump time** inside the DEFORM primitive — `points = reference + offset + scale·field` — never as an actor transform and never baked into `FEMSceneData.reference_points`, so the S2c invariant (world coordinates == grid coordinates) holds and every pick / overlay / box-projection path is offset-correct with zero change. The owner mutator is `GeometryManager.set_offset(geom_id, offset)` (length-3 validate, float-coerce, no-op on equal) firing the new granular `GEOMETRY_OFFSET_CHANGED` (dispatcher matrix row DEFORM-only); a RENDER-lane subscriber drops the hit scene's cached pick KD-tree and rebuilds visible node/element label overlays so snaps and labels never use a stale frame. `duplicate()` copies the offset. The `None` fast-path stays byte-identical at zero offset — a deform-off offset geometry pumps `reference + offset`, and zeroing the offset restores the legacy reset-to-reference path. The geometry settings panel grows an Offset X/Y/Z spinbox row; session schema bumps to **v6** with an additive `GeometrySnapshot.offset` (legacy sessions read `(0, 0, 0)`; malformed values degrade to zero). Stage pin is S3b; ghost preset + duplicate-with-layers are S3c. Locked by `tests/viewers/test_scene_instances_s3a.py` (mutator + event payload, matrix row + omnibus suppression, pump composition incl. fast-path, node-tree invalidation + label rebuild, offset-aware box pick, v6 round-trip + legacy default, qt end-to-end offset render/pick/compose).

### ADDED — geometry-aware picking (ADR 0058 S2c)

Under S2b every visible geometry's substrate actors are pickable, but pick resolution still assumed ONE scene — a pick on a non-active geometry's actor read coordinates off the wrong grid. S2c makes picking geometry-aware: the viewer keeps an `id(substrate actor) → (geometry_id, scene)` map in lockstep with `_scene_actors` (boot pair at show, clone pairs at materialization, dropped on geometry removal), and `install_results_pick` takes a `scene_resolver` so cell→element, the dim-pick gate, node-snap, the element highlight, and box-pick candidates all read the HIT geometry's scene (its deformed grid) at resolve time — box gestures (no single hit actor) resolve against the ACTIVE geometry; multi-geometry box picking is S3. The pick IR widens additively (ADR 0047 precedent): `PickResult` / `BoxPickResult` / `PointProbeResult` gain `geometry_id` (old constructors keep working), threaded through the selection log, the element-pick status line, and the PickReadoutHUD header — the geometry name shows only while MORE than one geometry is visible (mirrors the S2b scalar-bar prefix rule). Also closes the S2a known gap: `ProbeOverlay` and `LocalAxesOverlay` were constructed with the BOOT scene and held it forever; both now resolve the active geometry's scene at use time (construction scene stays as the headless/stub fallback), and per-pick probe reads take an explicit `scene=`. GP-marker hits keep `geometry_id=None` (overlay actors carry no geometry). Locked by `tests/viewers/test_scene_instances_s2c.py` (additive IR widening, hit-scene resolution incl. fallback/raising resolvers, actor-map lifecycle, overlay use-time resolution, HUD label gating, qt end-to-end pick on a deformed second geometry).

### ADDED — concurrent geometry rendering (ADR 0058 S2b)

Every geometry with `visible=True` now renders concurrently — its substrate fill + wireframe pair AND its diagrams, each at its own deform state; "active" is demoted to the editing target (deform editing, node cloud, label overlays). The new `Geometry.visible` flag is owned by `GeometryManager.set_visible()` (owner-fired `GEOMETRY_VISIBILITY_CHANGED`, dispatcher matrix row DEFORM + GATE), and the outline geometry-row eye now drives it directly — the Plan 03 v2 geometry-level `saved_visibility` composition cascade is retired (composition-row snapshots stay). The composition gate composes a third term: a layer shows iff `layer.is_visible AND composition gate AND owning_geometry.visible` (per geometry: the active composition's layers when one is active there, else all compositions' layers). Substrate actor-pair visibility is `geometry.visible AND geometry.show_mesh`, with each visible geometry's own `display_opacity` applied to its own pair. Scalar bars stay per-diagram, but while MORE than one geometry is visible each bar title is prefixed with the owning geometry's name (`"Geometry 2 — Sxx"`; single-geometry sessions keep unprefixed titles; bars refresh the prefix on re-create). Session schema bumps to v5 with an additive `GeometrySnapshot.visible`; old sessions (no flag) restore as **visible = is-active**, reproducing their previous active-only rendering. Per-geometry stage pin / spatial offsets / ghost preset are S3; picking disambiguation is S2c. Locked by `tests/viewers/test_scene_instances_s2b.py` (gate truth table, owner-fire + matrix row, scalar-bar prefix, session mapping, outline eye, qt concurrent-deform).

*(Merge note: the `## Unreleased` header items for P5.2+P5.3 (#634) and the coupling host auto-scalers (#635) keep being dropped by CHANGELOG conflict resolutions on other tracks — restored above again; their sections were never lost.)*

### CHANGED — flat deck emission sheds its per-incidence Python overhead (~1.5× on top of the partitioned fix)

Companion to the partitioned pre-bucketing below — this slice attacks the FLAT path's constant factor (the per-element cost every emit pays regardless of partitioning). Three changes, decks byte-identical (verified by diffing ~100 MB Tcl + Py decks at 512k hexes — flat, 64-rank, and staged-partitioned fixtures — before/after):

- **ADR 0048 ndf inference is evaluated per element CLASS, not per (element × node) incidence** — `infer_node_ndf` makes one capability-registry probe per class and resolves per-node floors + the strict ndf_ok gate with numpy over the connectivity; semantics are exactly the unit-tested `_infer_ndf_from_incidence` core (kept as the reference). Was ~30% of flat emit (8 registry probes + a dict-append per hex).
- **PG fan-outs are memoised per snapshot** — `expand_pg_to_elements` / `expand_pg_to_nodes` cache on a `WeakKeyDictionary` keyed by the (immutable) `FEMData`; one emit re-ran the same PG → Python-tuples materialisation for ndf inference, tag allocation, validators, and fix/mass/load fan-outs, and a same-snapshot re-emit (`ops.tcl` then `ops.py`) re-ran all of it again. Returned containers are documented read-only.
- **Emitter token dispatch is inlined per line** — `_join` (Tcl) / `_ops_call` (Py) dispatch int/float/str inline instead of calling `_fmt_value` per token, and `node` lines (the dominant deck band) render via a single f-string fast path behind exact-class guards (`bool` / numpy scalars fall through to the old path unchanged).

Measured (`time.perf_counter`, 512k hexes / 531k nodes, cold caches): flat `ops.tcl` 9.2 s → **6.2 s**; 64-rank `ops.tcl` 11.1 s → **7.9 s**; a same-snapshot follow-up `ops.py` lands at 4.9–6.2 s on the warm memo. (The "46.8 s flat" reading in earlier profiling notes was cProfile-inflated ~5× on this call-heavy path.)

### CHANGED — CHANGELOG merge-conflict treadmill ended (union driver + frozen header)

CHANGELOG.md now merges with git's built-in `merge=union` driver (`.gitattributes`), and the single-line `## Unreleased — item · item · …` ledger is **frozen**: PRs no longer append to it (that one line re-conflicted every open PR on every merge to main — five reconciliation rounds on #636 alone, and manual resolutions silently dropped the #634/#635 items three times). New entries are now exactly one contiguous `###` section inserted at the anchor comment above; the section title doubles as the highlight. Union keeps concurrent insertions from different PRs without conflicting (verified by simulation: concurrent PRs, stale-base PRs, and the duplicated-header failure mode that froze the ledger line). `tests/test_changelog_structure.py` guards the structure; workflow + in-flight-PR migration notes in `internal_docs/changelog_workflow.md`.

### CHANGED — partitioned deck emission drops its O(model × ranks) rescans (~2.5× at 512k hexes / 64 ranks)

Authoring-side emission wall-time — named by ADR 0061 as the next ceiling after the per-rank layout shipped — had three hidden O(model × ranks) terms in `_emit_partitioned`: the node-id → index lookup dict was rebuilt inside the per-rank loop, `emit_element_spec_partitioned` skip-scanned the FULL pre-allocated element plan once per rank (twice — owned-list build + main loop), and the global fix/mass passes re-ran the PG → nodes broker expansion per rank. Measured on a 512k-hex / 531k-node box at 64 ranks: `ops.tcl` 29.2 s → **11.7 s** (`ops.py` 28.2 s → 11.3 s); 64k hexes / 16 ranks: 2.5 s → 1.7 s.

- **Rank-independent work now happens once, before the per-rank loop**: the node-index lookup is hoisted; each element spec's plan is grouped by owner rank (`bucket_pre_allocated_by_rank`) and every rank walks only its own bucket (base loop + staged per-rank blocks); global fix/mass record targets are resolved once and bucketed by rank (`_bucket_fix_targets_by_rank` / `_bucket_mass_targets_by_rank` — fixes replicate on every owning rank, masses land on the primary rank only, semantics unchanged).
- The staged per-rank pass also stops rebuilding each rank's owned-node set per stage (stage-invariant — computed once by the caller).
- **Decks are byte-identical** — verified by diffing 100 MB Tcl + Py decks (flat-partitioned and staged-partitioned fixtures) emitted before/after; bucket construction preserves plan order and record/node order per rank.

### ADDED — per-rank Tcl deck emission (`apeSees.tcl(per_rank=True)`, ADR 0061)

Partitioned decks were monolithic: every MPI rank parsed the entire file and executed only its own `if {[getPID] == K} { ... }` blocks — per-rank parse cost and resident deck text were O(model) regardless of ownership (the measured Amdahl serial term of the emit → push → solve pipeline: ~5 s at 66k hexes, projected minutes/rank + node-RAM pressure at multi-M hexes).

- **`apeSees.tcl(path, per_rank=True)`** writes a driver deck at `path` plus one `ranks/rank<K>_<seq>.tcl` fragment per rank block (base topology + one per stage); the driver replaces each block with a one-line guard that `source`s the fragment, so a rank parses only the driver plus its own fragments — **O(global + model/np)**. Layout-only: deck semantics, the single-process rank-0 fallback (ADR 0027 INV-5), and the default monolithic output are unchanged (`partition_open`/`partition_close` span recording is observation-only).
- The driver keeps everything global and sequential — materials, sections, timeSeries, damping, analysis chains, recorders, and the staged skeleton (`domainChange` / `loadConst` / `wipeAnalysis` / analyze loops) — so stage ordering is preserved by construction. Locked by reassembly parity: re-inlining the fragments reproduces the monolithic deck line-for-line (base + staged-partitioned fixtures).
- Requires a partitioned model; mutually exclusive with `split=` (rank axis ≠ ADR 0043 module axis); `py()` per-rank is out of scope (Tcl is the HPC path). `Cluster.submit` / `ops.run_remote` need no changes — the driver is the deck entry point and the job dir is pushed wholesale.

### ADDED — ADR 0055 close-out: compose filtered-audit + Phase 3 verification (staged-H5 runway COMPLETE)

ADR 0055 flips **Proposed → Accepted** — every phase has shipped (P1 #569 · P2 #580/#586/#589 · P4 #590/#591/#634 · P5 #598/#612/#634 · P3 here). This slice delivers the last open item, Phase 3:

- **`g.compose_inspect(src)` gains a `filtered` key** (the ADR 0038 §195-196 filtered-audit that was never built): `{"stages": n, "time_series": n, "patterns": n}` — the droppable warn-kind analysis content a module carries, counted by the SAME probe the compose-time `ComposeFilterWarning` uses, so a module can be audited *before* composing it. Empty dict for vanilla modules.
- **FILTER+warn verified against a REAL staged archive** — the existing test hand-injected an `/opensees/stages` group because staged archives couldn't be written when it was authored; now an actual `ops.h5` staged archive composes with exactly one warning per droppable kind (stages + time-series) and the composed file inherits ZERO staged bytes.
- **P2.4 closed as absorbed** (guard-test inversion landed across P2.2/P2.3/P5.1; the replay oracle is a standing gate; the partitioned-raise fixture became a write-success case when Phase 5 lifted the guard).
- Also repairs main: `test_compose_facade.py` hardcoded neutral `"2.12.0"` and went red when #633 bumped to 2.13.0 — now imports `NEUTRAL_CURRENT` from the fixtures constants.

### FIXED — staged stages were lost in recorder output and the viewer (unconditional `domainChange` + numeric stage sort + positional stage pairing)

With more than ~3 stages, later stages silently vanished from `.ladruno`/`.mpco` results and the viewer's stage selector. Root cause: the MPCO/Ladruno recorders open a new `MODEL_STAGE[<stamp>]` group only when the Domain's change stamp moves, and the bridge emitted the per-stage `domainChange` barrier **only when the stage mutated the domain** — a pure-loading stage (nodal-load pattern + analyze) never moves the stamp (`Domain::addNodalLoad` deliberately doesn't flag it), so its steps were appended into the PREVIOUS stage's `MODEL_STAGE` group. Run-verified on the fork build: 5 pure-loading stages produced 1 group pre-fix, 5 post-fix.

- **`domainChange` is now an unconditional stage barrier** in both the flat and partitioned staged emits (the partitioned per-rank topology brackets stay content-gated; only the global barrier moved out). Captured staged H5 (`/opensees/stages` `domain_change` attr) and replay inherit the new shape.
- **Readers sort `MODEL_STAGE[<k>]` numerically** (`_ladruno.py` / `_mpco.py`): lexicographic order put `MODEL_STAGE[10]` before `MODEL_STAGE[2]`, scrambling stage ids from the tenth stage on. Multi-file readers delegate to reader 0 and inherit the fix.
- **Viewer stage-activation pairing gains a positional fallback** (`pair_capture_to_program`): MPCO/Ladruno capture stages are named `MODEL_STAGE[<stamp>]` — never equal to program stage names — so the name-only pairing silently never matched (filter rendered everything). When no capture name matches a program stage and the counts line up, capture stage *i* now pairs with program stage *i*; any real name match or a count mismatch keeps the name mapping (fail-soft unchanged).

### ADDED — partitioned staged: flat replay accepted + capture gate retired (ADR 0055 Phase 5 / P5.2 + P5.3)

Closes the Phase 5 runway (P5.4 true partitioned re-emit stays demand-gated):

- **P5.2 — flat replay accepted, zero source changes.** Post-P5.1 the stage buckets of a partitioned staged archive are rank-agnostic, so `OpenSeesModel.from_h5(...).build('tcl'/'py')` re-emits the FLAT single-process staged deck through the existing `_replay_staged_into` — the same degrade the non-staged partitioned path always had. Locked by `test_h5_partitioned_staged_replay.py`: the replayed deck is line-multiset-equal to the unpartitioned archive's replay; cross-rank shared-node HOLD/load lines emit exactly once; a `live`-marked smoke RUNS the replayed py deck under single-process OpenSees (the INV-5 runtime conditional + `getPID` shim make it portable by construction).
- **P5.3 — `ops.domain_capture` forwards the bridge for every build.** The last `bridge=None` gate (partitioned staged) is retired: the Composed run file now carries `/opensees/stages` + `/opensees/partitions` + the envelope ndf for partitioned staged captures — the feedstock the stage-aware viewer (ADR 0055 V1) reads, unlocking per-stage rendering of partitioned SSI runs. The stage-claimed phantom-node degrade (warn + sidecar-less) is unchanged.

### FIXED — coupling-knob H5 schema completion (post-#630 main red)

PR #630 added the six `cpl_*` `CouplingControl` columns to the `node_group` + `interpolation` payload dtypes but merged during a GitHub Actions ingestion stall, so its CI never ran — main went red on 6 schema tests. This completes the schema work the tests demanded:

- **`sr_cpl_*` mirror lane (schema 2.12.0, completing)** — the six `CouplingControl` columns are now mirrored into `surface_coupling_payload_dtype` as per-slave vlen arrays (`sr_cpl_has` / `sr_cpl_k` / `sr_cpl_kr` / `sr_cpl_enforce` / `sr_cpl_dtcr` / `sr_cpl_absolute`), wired through `_encode_surface_coupling` / `_decode_surface_coupling`, so a tied_contact / mortar slave record carrying explicit coupling knobs round-trips — the same lane-parity contract that `test_surface_coupling_sr_lane_mirrors_interpolation_dtype` locks (PR #337 precedent: the mirror lands under the same schema minor as the top-level extension). Pre-2.12.0 files decode to `control=None` (structural presence probe, as before).
- **Parity bookkeeping** — `control` is whitelisted on `NodeGroupRecord` / `InterpolationRecord` as "persisted decomposed into the cpl_* columns"; the `SR_TO_INTERP_COLUMN` map gains the six `sr_cpl_*` rows; the dtype-shape tests cover the new columns.

### ADDED — RBE2 partitioned (OpenSeesMP) emit: single-canonical-rank routing

`g.constraints.kinematic_coupling` now works under partitioned / OpenSeesMP emit (the handoff's deferred item C — previously the per-rank constraint planner raised `NotImplementedError` because the fork element, unlike the old idempotent `equalDOF` expansion, would allocate one element tag per owning rank ⇒ an N-fold over-constraint):

- **Single canonical rank** — mirrors the `ASDEmbeddedNodeElement` ownership rule: the `LadrunoKinematicCoupling` element emits on `min(intersection(node_owners[s] for s in slaves))` — the one deterministic rank where every tied slave is present — so every rank's planner agrees on the single emitter and exactly one element tag is allocated.
- **Ghost reference node** — the reference node is NOT required to co-locate: when it lives on another rank it is ghost-declared (`node` line before the element, INV-2) on the canonical rank, exactly like the embedded path's constrained node.
- **Fail loud on split slaves** — a slave set with no common rank raises with the per-slave owner map (same partitioner-input-bug stance as the embedded rule). The serial path is untouched.
- Stage-bound partitioned couplings ride the same planner, so `s.kinematic_coupling` under MP inherits the routing for free.
- Tests: planner-level canonical/ghost/min-rank/fail-loud units + integration coverage in `test_emit_partitioned_mp_constraint_replication.py` (element emitted exactly once with the ghost ref declared first; split set raises).

### ADDED — RBE3 tributary-area weighting (`weighting="area"`)

`g.constraints.distributing_coupling(..., weighting="area")` now computes each independent node's **tributary area** over the slave surface and emits `-w w1..wN`, so a force at the reference node distributes like a uniform traction on the surface (the handoff's deferred item B; previously a `NotImplementedError` stub — only `"uniform"` was wired):

```python
g.constraints.distributing_coupling("anchor", "footing_face",
    master_point=(0, 0, 2.0), weighting="area")
```

- **Lumping model matches `g.loads`:** each slave face's area (fan-triangulated polygon) is split equally among its nodes and accumulated per node — the same tributary model as `g.loads` surface resolution, so the RBE3 distribution is exactly proportional to a uniform tributary surface load. The fork normalizes by `W = Σw`, so only proportionality matters.
- **Order-safe:** weights are computed in (and aligned to) the record's **sorted independent order**, so `-w[i]` pairs with independent `i_i` — positional user-supplied weights were never viable because the resolver sorts the set internally.
- **Fail loud:** `weighting="area"` requires the slave label/entities to resolve to meshed surface faces (the `face_map` gate now covers area-weighted distributing couplings), and an independent node lying on no slave face raises instead of silently zeroing its load share.
- **No emit/H5 change needed:** `InterpolationRecord.weights` already emitted as `-w` and already round-trips through `model.h5`.

### ADDED — coupling host auto-scalers (`k="auto"` / `k_alpha` / `host` / `bipenalty_wcap`)

The fork coupling elements' host-element penalty auto-scalers are now reachable from `g.constraints.kinematic_coupling(...)` / `distributing_coupling(...)` (previously only the manual numeric knobs were wired — the handoff's deferred item A):

```python
g.constraints.kinematic_coupling("platen", "face",
    k="auto", k_alpha=1e3, host=4021)          # K_t = k_alpha * max|K_host(i,i)|
g.constraints.distributing_coupling("anchor", "ring",
    k=1e8, host=4021, bipenalty_wcap=0.1)      # m_p = K_t/(0.1*omega_host)^2
```

| kwarg | flag | meaning |
|---|---|---|
| `k="auto"` | `-k auto` | scale `K_t` off the host element's stiffness diagonal (requires `host`) |
| `k_alpha` | `-kAlpha $a` | multiplier for `k="auto"` (fork default `1e3`) |
| `host` | `-host $eleTag` | representative host element — given as a **FEM element id**; the bridge translates it to the emitted OpenSees tag |
| `bipenalty_wcap` | `-bipenalty -wcap $beta` | penalty mass from the host frequency instead of a hard `-dtcr` budget (requires `host`; mutually exclusive with `bipenalty_dtcr`) |

- **FEM-eid host translation:** `CouplingControl.host` stores the FEM element id (stable across emits); the MP-constraint emit pass now receives the bridge's `fem_eid_to_ops_tag` map (serial + partitioned + stage variants) and translates at emit time. A hosted control fails loud if the map is absent or the eid never emitted — no silent wrong-element scaling.
- **Validation** mirrors the fork's parse guards: `k="auto"` and `bipenalty_wcap` each require `host`; `k_alpha` only with `k="auto"`; a dangling `host` (nothing consuming it) is refused; `bipenalty_dtcr` + `bipenalty_wcap` are mutually exclusive; `enforce="al"` refuses both bipenalty modes.
- **H5 round-trip (neutral schema 2.12.0 → 2.13.0):** four columns added to the `cpl_*` lane — `cpl_k_auto` (uint8), `cpl_k_alpha` (f64), `cpl_host` (int64 FEM eid, `-1` = none), `cpl_wcap` (f64) — presence-probed, so 2.12.0 files decode with the v1 knobs only.
- **Schema-pin reconciliation with #632:** the sibling fix that un-redded main mirrored the v1 `cpl_*` columns into the `sr_cpl_*` lane — this change extends that mirror with the four auto-scaler columns (`sr_cpl_k_auto` / `sr_cpl_k_alpha` / `sr_cpl_host` / `sr_cpl_wcap`), and the neutral-version pin now reads from `tests/fixtures/schema.py` instead of a literal so the next bump is a one-file edit.

### ADDED — full control knobs on the fork coupling elements (RBE2 / RBE3)

`g.constraints.kinematic_coupling(...)` and `g.constraints.distributing_coupling(...)` now accept the fork elements' penalty / enforcement knobs, so the user has full manual control over the coupling mechanics (previously only the defaults were reachable):

```python
g.constraints.kinematic_coupling("platen", "face",
    k=1e10, kr=2e10, enforce="al", bipenalty_dtcr=2e-6, absolute=True)
g.constraints.distributing_coupling("anchor", "ring",
    k=5e8, bipenalty_dtcr=2e-6)
```

| kwarg | flag | meaning |
|---|---|---|
| `k` | `-k $Kt` | translational penalty (default `1e12`) |
| `kr` | `-kr $Kr` | rotational penalty (default fork-derived `K_t·ℓ²`) |
| `enforce` | `-enforce penalty\|al` | `al` = augmented Lagrangian (implicit only) |
| `bipenalty_dtcr` | `-bipenalty -dtcr $dt` | explicit critical-step target (RBE3's ref node is massless → needed for explicit runs) |
| `absolute` | `-absolute` | keep the absolute tie (skip the default `g0` stress-free birth) |

- A new frozen `CouplingControl` (`apeGmsh._kernel._coupling_control`) carries the knobs from the `*Def` onto the resolved record; the bridge appends `control.emit_flags()` to the emitted `LadrunoKinematicCoupling` / `LadrunoDistributingCoupling` line. Defaults are elided, so a knob-free coupling emits byte-identically to before (the resolver stores `control=None`).
- **Validation** mirrors the fork's parse guards: `enforce ∈ {penalty, al}`; `k`/`kr`/`bipenalty_dtcr > 0` if set; `enforce="al"` + `bipenalty_dtcr` is refused (implicit AL can't combine with explicit bipenalty).
- **H5 round-trip (neutral schema 2.11.0 → 2.12.0):** six `cpl_*` columns added to the `node_group` + `interpolation` payload dtypes (presence-probed, so pre-2.12.0 files decode as `control=None`); the knobs survive `model.h5` save/reload.
- ~~**Deferred (host-element auto-scalers)**~~ — shipped below (`k="auto"` / `k_alpha` / `host` / `bipenalty_wcap`). RBE3 tributary-area `-w` weighting is still a follow-up.

### ADDED — `ops.run_remote` one-call remote analysis + `Job.wait` (ADR 0060 sugar)

The deferred bridge sugar lands: `ops.run_remote("./job", cluster="esmeralda")` emits the Tcl deck into the job directory (with `analyze_steps=`/`analyze_dt=` passthrough), pushes it, `sbatch`es, blocks via the new `Job.wait(poll=, timeout=)` until terminal, and fetches everything back — raising `HPCError` with the remote stderr tail on any non-`COMPLETED` end state (artifacts are fetched **before** the raise; on failure the logs are the evidence). `np` defaults to the model's partition count (`len(fem.partitions)`, 1 for flat); `wait=False` returns the submitted `Job` immediately, and the JSON sidecar keeps it reloadable across sessions (`Job.load(job_dir)`). Run-verified end-to-end on Esmeralda: a real session-built 3-story frame, METIS 4-partition, Ladruno-fork OpenSeesMP under `srun --mpi=pmix_v3` — one call, COMPLETED, fetched (job 143706). Locked by `tests/hpc/test_run_remote.py` (real bridge emit over the fem stub, faked ssh seam) + `TestWait`.

### ADDED — remote HPC job submission (`apeGmsh.hpc`, ADR 0060)

*(Re-applied: this PR #624 entry was dropped from the Unreleased header + sections by a later CHANGELOG conflict resolution.)* `from apeGmsh.hpc import Cluster` — submit an emitted deck to a SLURM cluster from the local machine: `Cluster.load("esmeralda").submit(job_dir, np=8)` pushes the directory (single tarball over native `ssh`/`scp`, `BatchMode=yes`, zero new dependencies), renders an inspectable `job.sbatch` (LF + UTF-8 — CRLF breaks remote bash) next to the deck, and `sbatch`es it by absolute path (SLURM is often only on the *login-shell* PATH). The returned `Job` survives the session via a JSON sidecar (`Job.load(dir)`) and offers `status()` / `tail()` / `cancel()` / `fetch()`. Status is squeue-while-alive + a script-written `.exit_code` sentinel after — the only durable completion record on a cluster whose `sacct`/slurmdbd is down (Esmeralda's is). Connection facts stay in `~/.ssh/config`; cluster facts (paths, partition, `srun --mpi=pmix_v3` launcher template, env lines) live in `~/.apegmsh/clusters.toml` with fail-loud unknown-key/missing-key validation. Run-verified end-to-end on Esmeralda (2-rank smoke job: submit → RUNNING → COMPLETED → fetch).

### ADDED — absorbing-boundary guide (`internal_docs/guide_absorbing_boundary.md`)

A user-facing guide for the ADR 0054 surface, written from the run-verified plane-wave campaign: building the box (`g.parts.add_plane_wave_box` + siblings, the `AbsorbingSkinResult` name bag), declaring the skin (`ops.element.absorbing_boundary` — homogeneous / stratified / raw, and why the impedance material is read-never-emitted: the upstream command takes raw `G v rho`, there is no material-tag slot), base-input injection (`base_series` on `B`-cells only), the staged hold→absorbing lifecycle (`s.activate_absorbing` and its emitted one-shot `parameter` block, per-rank semantics), the modeling recipe (nodal soil-only mass — never mass the skin; fix the bottom outer plane for explicit; **global** mass-proportional Rayleigh — region `-ele` Rayleigh is inert on `rho=0` soil, and the element's `addCff` mirrors the domain α into its free-field columns by design), validation methodology (coherence / windowed decay / energy closure `−IE = KE + DW + RES` / seq↔par identity; quietness is NOT a valid check for stratified profiles, and a few-% base re-reflection at 4H/Vs is the expected Lysmer residual), and a pitfalls list.

### CHANGED — per-geometry scene instances + active-geometry switching (ADR 0058 S2a)

S1's `director.scene_for(geometry)` returned the same scene for every geometry; S2a makes it true. The director's cache is seeded `{boot geometry: bound scene}` at `bind_plotter(..., scene_factory=)` and a miss materializes lazily through the **viewer-injected factory** — the director never builds scenes itself, so the diagrams package stays pyvista-free (ADR 0042 INV-2; the plan's `on_scene_materialized` hook is folded into the factory, which both clones and wires). `clone_scene` (in `scene/fem_scene.py`) is the materialization primitive: deep-copied grid reset to `reference_points` (clones are born undeformed and unhidden), index arrays **shared** (`node_ids` / `node_id_to_idx` / `cell_to_element_id` / `element_id_to_cell` / `cell_dim` / `model_diagonal`), render-side fields `None`. The viewer's factory wires per-scene `ElementVisibility` (+ dispatcher), shares the plotter-scoped pick inventory + opacity controller, applies the CURRENT dim-filter and stage-activation state (both now re-applied to every materialized scene on change), and builds a hidden substrate fill + wireframe pair per scene (extracted `_add_substrate_actors`). **Switching is actor visibility, never re-attach**: a RENDER-lane `GEOMETRY_ACTIVE_CHANGED` subscriber flips the pairs (materializing on demand), re-points the active-actor handles, re-applies display/theme state, and rebuilds visible label overlays; `GEOMETRY_REMOVED` removes the pair and drops the director's cache entry; the DEFORM pump needed zero changes (S1 built the loop). `viewer._scene` becomes a property over `scene_for(active)` so every display-level consumer (status line, labels, probe radius, pick extract, node cloud — which stays active-only) is automatically right. Headless binds without a factory keep the S1 single-scene fallback. Behavior-preserving: the viewport still renders only the active geometry's substrate (S2b flips concurrency; S2c does pick disambiguation). Locked by `tests/viewers/test_scene_instances_s2a.py` (clone contract, cache/removal, qt A→B→A switch).

### CHANGED — geometry→scene resolution seam (ADR 0058 S1)

Every substrate-scene access in the results viewer now resolves through the geometry that owns it, while the viewport still renders exactly one substrate — the behavior-preserving plumbing slice ahead of concurrent geometries (S2). `ResultsDirector.scene_for(geometry)` is the seam (S1: every geometry maps to the single scene bound at `bind_plotter`; S2 swaps the internals for real per-geometry `FEMSceneData` instances), and `DiagramRegistry.bind(..., scene_resolver=)` resolves each diagram's attach scene through its owning geometry. The DEFORM pump restructures into a loop over `_render_geometries()` (S1: `[active]`): each geometry's deformed points come from **its own** deform state + **its scene's** `reference_points` (new `FEMSceneData.reference_points` field, captured at build — previously a viewer-level attribute), and the fan-out is scoped to that geometry's layers. Scoped pumps (newly-attached diagram) sync against the diagram's *owning* geometry's state rather than unconditionally the active one. Other geometries' diagrams are gate-hidden and re-pumped on activation, so nothing visible changes. Seam contract locked by `tests/viewers/test_scene_seam_s1.py`. The ADR's S1 memory measurement: a 23k-node / 124k-tet scene costs ~7 MB and deep-copies in ~2 ms — per-geometry plain copies are affordable; the copy-on-write escape hatch stays unused.

### CHANGED — declarative diagram-kind registry (ADR 0058 S0)

Diagram kinds self-register via `@register_diagram_kind(label=, style_class=, ...)` in their own module (`viewers/diagrams/_kinds.py`). Four hand-maintained per-kind tables — the Add-dialog `_KINDS` (+ derived `_KIND_TO_TOPOLOGY`), the catalog `_KIND_DEFINITIONS`, the session `_KIND_TO_STYLE`, and the preset `KIND_TO_STYLE_CLASS` — collapse into it; the dialog, settings tab, kind catalog, session codec, preset store, and session restore all consume the registry. Adding a diagram kind drops from ~6 touch points to the class file + tests; the registry guard (`tests/viewers/test_diagram_kind_registry.py`) fails any `Diagram` subclass that forgets to register.

The drift the tables had already accumulated is fixed en route: **(1)** the session and preset maps silently lacked `loads` / `reactions`, so those layers never survived a session save/restore and their presets saved but could never load — both now round-trip (locked by the guard's per-kind codec test); **(2)** the catalog's dedicated reactions branch sat *below* the generic `requires_data` branch and was unreachable — the settings-tab creation panel listed every nodal component for Reactions instead of the curated `reactions` / `reaction_x/y/z` options; the branch now precedes the generic one (parity with the modal dialog, which always had its own correct logic); **(3)** the `layer_stack` label unifies to "Layer stack (shells)" (the dialog said "(shell)", the catalog "(shells)"). One cosmetic reorder: the settings-tab creation combo lists Vector glyph after Layer stack (registry order = dialog order; the two tables disagreed before).

### ADDED — Ladruno recorder whole-model energy channel (`energy=` → `-G energy`)

`ops.recorder.Ladruno(..., energy=True)` (and the `Ladruno` primitive's `energy` field) emit the fork's whole-model energy-balance channel — `RESULTS/ON_DOMAIN/energyBalance`, components `KE/IE/DW/ULW/RES/ERR` — closing the emit half of the channel whose reader (`Results.energy()`) already shipped. `energy=True` alone is a valid recorder (the at-least-one-channel validation now counts it). The flag is emitted **last** on the recorder line, and that ordering is load-bearing: the fork's `-G` parser eagerly consumes trailing region-tag integers and cannot rewind past a following flag, so `-G energy -T nsteps 10` is a parse error while `-T nsteps 10 -G energy` runs (run-verified on the fork build; the ordering is locked by `test_ladruno_recorder.py::TestLadrunoEnergy`). Per-region energy (`-G energy <regionTag…>`) stays deferred on the bridge-region → OpenSees-tag seam.

### FIXED — partitioned emit: additive nodal quantities (`mass` / pattern `load`) emit on ONE rank

Under OpenSeesMP, shared (interface) nodes exist on every rank that defines them and the parallel assembly **SUMS** each domain's nodal contributions at the merged equations. The per-rank fan-out replicated `mass` lines (global `ops.mass`, stage-bound `s.mass`) and pattern `load` lines (`p.load`, `p.from_model` imports; global and stage-scoped) on **every** owning rank — correct for idempotent lines (`node`/`fix`/`sp`), wrong for additive ones: interface nodes carried their mass once per owning rank. Run-verified on an 8-partition stratified plane-wave model: 81 massed nodes emitted 177 `mass` lines and the partitioned transient diverged from the byte-identical sequential run by ~100 % of peak velocity.

Each massed/loaded node now emits on its **primary rank** only — the lowest owning runtime rank (`primary_owner_map`, deterministic). `fix` / `sp` / `node` lines keep the every-owner fan-out, and the stage-pattern empty-bracket pre-check mirrors the new filters so a rank whose only content is a non-primary shared loaded node doesn't open an empty block (Py-emitter `SyntaxError`). Re-verified live after the fix: the same 8-rank model emits exactly 81 `mass` lines and the **unpatched** partitioned run agrees with the sequential run to ~4e-15 of peak. Locked by `test_partitioned_additive_dedup.py` (mass/load once, fix/sp/node replicated) + `primary_owner_map` unit locks.

### FIXED — partitioned-deck `getPID` shim guards with `info commands`, not `info procs`

The Tcl partition shim emitted `if {[info procs getPID] == ""} { proc getPID {} { return 0 } }`. OpenSeesMP registers `getPID` via `Tcl_CreateCommand` — a C **command**, invisible to `info procs` — so the guard never detected the native command, the shim **overrode it with the rank-0 fallback, and every MPI rank silently built rank 0's submodel** (run-verified under `mpiexec -n 8 OpenSeesMP`: 8 byte-identical `results.part-N.ladruno` files, each carrying the same rank-0 node set). The guard now probes `info commands`, which sees both the native command and a prior proc, restoring distinct per-rank subdomains (node union = full model). Single-process behaviour is unchanged — on a build with no `getPID` at all the shim still installs the rank-0 fallback. The never-run-locally Scenario-C runtime smoke (`test_partition_pipeline_e2e.py`, skipped without `OpenSeesMP` + `mpiexec` on PATH) is why this never surfaced in CI; the unit lock (`test_emitter_partition_open_close.py::test_tcl_emitter_shim_uses_info_commands_guard`) now asserts the correct guard verb.

### ADDED — viewers consume the remaining recorder channels (LOCAL_AXES roll · `plot.energy` · `plot.node_envelope`)

A recorder→viewer consumption audit found three recorder outputs with reader surfaces but no rendering path; all three now land:

- **Interactive diagrams orient from the recorder's true beam frame.** `LineForceDiagram` now overlays each element's vecxz with the z-axis of the slab's `local_axes_quaternion` (`.ladruno` `MODEL/LOCAL_AXES`), and `FiberSectionDiagram` resolves frames recorder-first (`results.elements.local_axes()`, recorded frames only) → model `vecxz_for` (previously not consulted at all) → geometric default. The matplotlib `plot.line_force` already preferred the recorder frame; the Qt/web diagrams silently fell back to geometry — wrong cross-section roll on rolled sections, and on every model-less `Results.from_ladruno(...)` open (no `vecxz` source). The overlay parks the recorder z as the element's vecxz, so the deformed-substrate resync re-derives the same roll. Locked by `tests/viewers/test_diagram_recorder_frame.py` (rolled-frame fixture copies; GL-free).
- **`results.plot.energy(region=, stage=)`** — the Ladruno `-G energy` balance (`KE/IE/DW/ULW/RES` + `ERR` on a twin axis) as a 2-D time-history figure. The reader (`results.energy()`) existed; nothing consumed it.
- **`results.plot.node_envelope(component, measure="absmax"|"min"|"max")`** — paints the Ladruno `-envelope` per-node time-reduced extremes on the mesh (the file holds no time series for `contour` to read). Shares the contour paint path (extracted `_paint_node_scalar`).
- **FIXED en route:** `results.plot` facet extraction selected 1-D elements by literal type name (`line2`/`line3`) and skipped the solver-flavoured 1-D groups a `.ladruno`-synthesized `FEMData` carries (`truss`, beams) — every `plot.*` mesh render on such an open drew nothing. Now selects by `element_type.dim == 1` (endpoints = first two nodes, midnodes still dropped).

Audit residue, deliberately not in scope: director `TimeMode.ENVELOPE`/`RANGE` stay Phase-6-deferred (raise loudly); mode-stage viewing already works (stage selector + outline handle `kind="mode"`).

### FIXED — deform-follow regression: contour / fiber / layer-stack / spring diagrams ride the deformed substrate again

A deform-pipeline audit found that the viewer-side `_sync_layer_grids` walk — the mechanism that moved substrate-extracted submeshes under deformation by scattering `vtkOriginalPointIds` — iterates `d._actors`, which **migrated diagrams never populate** (post-ADR-0042 they emit backend-owned dataset copies and hold no VTK actors). Since the render-seam migration completed, four diagram kinds silently stayed at the reference configuration while the substrate warped: **Contour** (all five dispatch paths), **FiberSection** (dot cloud), **LayerStack** (shell submesh), **SpringForce** (glyph anchors).

- `Diagram.sync_substrate_points` is now the **only** deformation fan-out, and every rendering diagram implements it: contour + layer-stack re-sample via cached `vtkOriginalPointIds` rows (the map survives `separate_cells` as carried point data, so the shattered discrete-gauss path follows too); the fiber cloud re-derives each beam's frame from the deformed chord + the cached vecxz (recorder z / model vecxz / default) and rebuilds the dots; spring arrows re-sample their anchor-node substrate rows.
- The dead `_sync_layer_grids` walk is deleted; the stale base-class docstring ("extract_* layers follow automatically") is corrected to the new contract.
- En route: `SpringForceDiagram` anchor collection returned a positions array positionally zipped against the requested element ids — silently misaligned when any spring's node was missing from the view. Now keyed by eid.
- Endpoint+substrate-row collection is shared (`collect_endpoints_with_substrate_rows` in `_beam_geometry`, used by line-force and fiber-section).

Locked by `tests/viewers/test_diagram_deform_follow.py` — the shift/reset contract per diagram kind (contour ×4 paths, fiber, layer-stack, spring); the already-correct diagrams keep their own coverage. **The contract is enforced going forward** by `tests/viewers/test_deform_follow_contract.py` (the ADR 0056 guard pattern): it walks every `Diagram` subclass under `apeGmsh.viewers` and fails on any class inheriting the base no-op `sync_substrate_points` — a new diagram that forgets the hook fails at test time, not silently in a user's deformed view. Exemptions live in an explicit, reason-carrying `_EXEMPT` list (currently only `DeformedShapeDiagram`, which renders its own warp) and are stale-checked against the live class set.

### FIXED — local-axes overlay triads resolve recorder-first (frame parity with the diagrams)

Since the recorder-frame rewire, `LineForceDiagram` / `FiberSectionDiagram` orient from the `.ladruno` `MODEL/LOCAL_AXES` true beam frame — but the "Local axes" toolbar overlay still drew its triads from the model `vecxz` / geometric default, so on a model-less `Results.from_ladruno` open the arrows could show a different cross-section roll than the diagrams render with. The overlay now uses the same precedence: the shared `recorder_z_axes` helper (moved to `_beam_geometry`, recorded frames only — an explicit `ids=` would identity-pad unrecorded elements) feeds a new `vecxz_override=` channel on `iter_local_frames`, scoped to the director's active stage; any unscopable state (combined stage, no results) falls back to the model-frame behaviour unchanged. Locked by rolled-fixture overlay tests + an `iter_local_frames` override unit test.

### ADDED — static gauss contours + fiber dot cloud (`plot.contour(topology="gauss")` · `plot.fibers`)

The matplotlib renderer reaches viewer parity on the two element-level sources it couldn't draw:

- **`plot.contour(component, topology="nodes"|"gauss", averaging="averaged"|"discrete")`** — same vocabulary as the interactive `ContourStyle`. *Averaged*: GP→corner extrapolation (`extrapolate_gauss_slab_to_nodes`, pinv of the shape-function matrix) + cross-element nodal averaging → smooth contours. *Discrete*: every facet painted from its **own** element's corner values — element-boundary jumps stay visible, which matplotlib's per-face flat shading renders exactly. Facet extraction now also returns per-facet element ownership (`extract_facets_owned`; a boundary face belongs to exactly one volume element). `deformed=`/`wireframe=` compose with both.
- **`plot.fibers(component, ...)`** — the static counterpart of the viewer's `FiberSectionDiagram`: a 3-D dot cloud at the true world positions (station ξ from `FiberSlab.station_natural_coord` with the viewer's uniform-spread fallback; `(y, z)` section offsets in the beam frame, recorder-LOCAL_AXES-first) colored by `fiber_stress`/`fiber_strain`, ghost mesh underlay, `gp_indices=` station filter. Closes the docstring-deferred "fiber-section scatter" item; the 2-D per-section σ-ε panel stays deferred.

### CHANGED — diagram scalar-state consolidation (`ScalarColorSupport` + base `_scoped_results`)

The copy-paste across the diagram family is gone (backlog item #4 from the result-type review):

- **`ScalarColorSupport`** (extends `ScalarBarSupport`) now owns the runtime colour state (`_runtime_clim` / `_runtime_cmap` / `_initial_clim`), the live `set_clim` / `set_cmap` / `current_clim` setters, `autofit_clim_at_current_step` (the per-diagram `_scalar_values_for_autofit` hook is the one genuine variation), and the Qt LUT mirror (`_init_lut` / `_on_lut_changed` / `_teardown_lut`) — previously duplicated near-verbatim across Contour, VectorGlyph, GaussPoint, FiberSection and LayerStack.
- **`Diagram._scoped_results()`**: the stage-scoping helper was verbatim in 8 diagram classes — now on the base. `ReactionsDiagram` keeps its deliberately defensive override (`None` on a bad stage id) unchanged.
- **One deliberate unification**: the scalar-bar refresh on a LUT change now passes the runtime `fmt` for every diagram. Previously only the contour did — a `set_fmt` on the other four was silently lost on the next colormap change, while `set_show_scalar_bar` (which always passed `fmt`) preserved it. Locked by `test_fiber_fmt_survives_lut_change`.

Behavior locked by the existing suites (1388 viewers tests pass unchanged).

### FIXED — degraded GP world-coordinate reconstructions are loud (`WarnGaussCoordsApproximate`)

`GaussSlab.global_coords` has two locked, reasonable degradations that were **silent**: element types with no shape-function coverage (pyramids, P3+) take a `centroid + ½·bbox·ξ` approximation, and elements that can't be resolved against the `FEMData` (missing element, unknown nodes) leave their GPs parked **at the origin**. A mis-placed Gauss-marker cloud is indistinguishable from a correct one — the same ADR 0056 INV-6 bug class as the fiber-station fix. `compute_global_coords_from_arrays` now emits one aggregated `WarnGaussCoordsApproximate` per call naming the GP count + offending type codes (bbox path) and the unresolved element ids (origin path). The supported paths stay silent — locked by running the full results+viewers suites with the warning escalated to error. Also corrects the module's stale coverage note: the catalog covers lines/tris/quads/tets/hexes/wedges *including P2 forms*; only pyramids and P3+ fall back.

### ADDED — `distributing_coupling` (RBE3) ships as the fork `LadrunoDistributingCoupling` element (tag 33011)

`g.constraints.distributing_coupling(master_label, slave_label, *, master_point=, weighting="uniform", name=)` now emits the Ladruno-fork **`element LadrunoDistributingCoupling`** (class tag 33011), replacing the long-standing `NotImplementedError` stub (whose predecessor emitted a mechanically-wrong kinematic mean).

```
element LadrunoDistributingCoupling $tag $refNode $N $i1 ... $iN [-w $w1 ... $wN]
```

RBE3 is the **flexible** counterpart of RBE2 (`kinematic_coupling`): the reference node R is the weighted-average rigid-body fit of the independent set, and a force/moment at R is distributed to the set as a statically-equivalent pattern (`Σ Fᵢ = F`, `Σ rᵢ × Fᵢ = M`) **adding no stiffness** to the independents — so a load introduces at a point while the region stays free to deform. This completes the four-PR fork plane/coupling series (LadrunoQuad 33007, LadrunoCST 33008, LadrunoKinematicCoupling 33012, LadrunoDistributingCoupling 33011).

- **Reuses `InterpolationRecord`** (no neutral-zone schema change): the reference (dependent) node R is `slave_node`, the independents are `master_nodes` — the field names read backwards for RBE3 (R is the dependent), but the geometry maps 1:1 onto the emit. Routed by record type into `fem.elements.constraints` (the existing dispatch already mapped `DistributingCouplingDef → resolve_distributing`).
- **Emit branch:** `_emit_surface_couplings` / `_emit_surface_couplings_for_rank` now share `_emit_one_interpolation`, which branches on kind — `distributing` → the fork element (arbitrary independent count N≥1; the 3/4-Rnode embedded guard does **not** apply), `tie`/`embedded` → `ASDEmbeddedNodeElement` as before.
- **Fork-only:** added to `_FORK_ONLY_ELEMENTS` → stock OpenSees fails loud at the element line (it doesn't know tag 33011); emission to `.tcl`/`.py` works on any build.
- **Reference node must carry rotational DOFs** (ndf 6 in 3D / 3 in 2D) to transmit a moment; independents can be translation-only (3-DOF). The fork refuses a too-small reference at `setDomain`.
- **v1 weighting is `"uniform"`** (equal weights ⇒ `-w` omitted ⇒ the element's equal-weight default). `weighting="area"` raises `NotImplementedError` — tributary-area `-w` (apeGmsh computing per-independent areas) is a follow-up.
- **Partitioned (MPI):** distributing rides the same single-canonical-rank rule as `ASDEmbeddedNodeElement` (the rank owning all independents emits; fail-loud if they split across ranks) — no silent replication.
- `DistributingCouplingDef` lost its ASD-embedded fields (`stiffness`/`stiffness_p`/`rotational`/`pressure`/`dofs`) — it is no longer an `ASDEmbeddedNodeElement` carrier; the fork element owns its own `-k` penalty default (not yet surfaced — a follow-up, with `-enforce`/`-bipenalty`/`-host`/`-kr`).

### CHANGED — `kinematic_coupling` emits the fork `LadrunoKinematicCoupling` element (RBE2, tag 33012) — **BREAKING**

`g.constraints.kinematic_coupling(...)` (and the stage claim
`s.kinematic_coupling(name=)`) now emit the Ladruno-fork
**`element LadrunoKinematicCoupling`** (class tag 33012) instead of expanding to
one `equalDOF` per slave. The fork element is a penalty rigid-body driver that
carries the correct moment-arm transport `u_i = u_R + θ_R × d_i`, so an **offset
reference node is coupled rigidly** — the old `equalDOF` expansion ignored the
lever arm and was only correct when slaves were coincident with the master.

```
element LadrunoKinematicCoupling $tag $refNode $N $s1 ... $sN [-dof $c1 ...]
```

- **BREAKING / fork-only.** The deck emits on any build, but **running** it needs
  the Ladruno fork — stock OpenSees fails loud at the element line (it does not
  know class tag 33012; `LadrunoKinematicCoupling` is in `_FORK_ONLY_ELEMENTS`).
  This is intentional: the proper RBE2 mechanics are the reason the element
  exists. For a plain rigid same-DOF tie on coincident nodes, use
  `g.constraints.equal_dof`; `rigid_link` / `rigid_diaphragm` are unchanged.
- **`dofs` default is now "all the slave has"** (`-dof` omitted), emitted only
  when you restrict the component list — this resolves a mixed 3-/6-DOF slave set
  correctly (the element handles the ragged layout), where the old `[1..6]`
  default would force rotation DOFs onto translation-only solid faces.
- **The reference node must carry the rotational DOFs** (ndf 6 in 3D / 3 in 2D);
  the fork refuses a too-small reference at `setDomain`.
- Reuses the existing `NodeGroupRecord` (no neutral-zone schema change); the
  record round-trips through `model.h5` and staged-H5 archival as before, now
  replaying the element line.
- **Partitioned (MPI) emit fails loud** for `kinematic_coupling`: a fork element
  can't be safely replicated across ranks the way an `equalDOF` command was
  (each rank would allocate its own tag → an N-fold over-constraint). Use serial
  emit, or keep the coupling's reference + slaves on one partition.
  Single-canonical-rank handling (like `ASDEmbeddedNodeElement`) is a follow-up.
- Not yet exposed on the apeGmsh surface: `-k` / `-enforce al` / `-bipenalty` /
  `-host` / `-kr` / `-absolute` (the element's defaults — `k=1e12`, penalty,
  bipenalty off — give the rigid tie); a follow-up will surface them. The RBE3
  distributing coupling (`LadrunoDistributingCoupling`, tag 33011) remains the
  last of the four fork PRs.

### FIXED — fiber sections render at the beam's TRUE integration stations

`FiberSlab` gains `station_natural_coord` — the per-fiber station ξ ∈ [-1, +1] along the parent beam, read from the recorder's own integration-rule metadata instead of re-invented by the viewer. `FiberSectionDiagram` previously inferred station positions from a uniform spread (`ξ = -1 + 2·gp/(n−1)`), which mis-places fibers along the member for any non-uniform `beamIntegration` (a 3-point Gauss-Legendre force-based beam's end stations sit at ξ ≈ ±0.775, not ±1). Every reader had the truth on hand and dropped it:

- **MPCO**: the fiber bucket layout already resolved the connectivity's `GP_X` — and used only its *length*. The coordinates now ride the slab, tiled per fiber row (and sliced by the `gp_indices` filter).
- **.ladruno**: `QUADRATURE/GP_PARAM` keyed by `GAUSS_ID` — the same source the line-stations path reads. The layered-shell `material.fiber.*` spelling carries NaN: a shell's gauss id is a *surface* GP, not a beam station.
- **Live DomainCapture**: the fiber capturer already called `eleResponse(integrationPoints)` and kept only the count; it now normalises the physical positions to natural ξ (end-node length probe, NaN on failure) and the native writer/reader carry an optional `_station_natural_coord` fibers dataset — files written before the field read back as `None`.
- **Multi-partition / multi-file concat** NaN-fills mixed sources; all-absent stays `None`.

The diagram prefers the slab's true ξ and falls back to the uniform inference only for rows without one (pre-station files, failed probes) — loudly (`diagram.fiber_station_xi_inferred`, ADR 0056 INV-6).

### ADDED — partitioned staged H5 archival (ADR 0055 Phase 5 / P5.1, schema 2.19.0)

**The last `apeSees.h5()` fail-loud guard is lifted: PARTITIONED staged builds now archive to `model.h5`.** The capture is rank-agnostic by construction — the `/opensees/stages` zone carries the flat logical program once, while the per-rank emit shape stays derivable from the neutral `/partitions` zone:

- **Stage-bucket dedupe/merge under rank brackets.** The partitioned staged emit replicates stage-bound emission across owning ranks (ADR 0027 INV-1/INV-4 inside `_emit_stages_partitioned`): stage fix/mass/`remove_sp`/MP-constraint lines on shared nodes now capture once (with a single `emit_index` stamp); the stage's patterns — including the ADR 0052 HOLD pattern — re-open the same tag once per owning rank and now RESUME the captured record (per-rank `load`/`sp`/`sp_hold` subsets merge; shared-node lines capture once); per-rank stage-region fragments (`-node` member intersections sharing one tag) merge into the one logical region (member union, first-occurrence order).
- **Foreign ghost-node declarations never enter `owned_node_ids`.** The stage MP pass declares foreign nodes before replicated constraints (INV-2); the bridge now surfaces each stage's owned-node set via the `set_stage_owned_node_tags` side-channel (the `set_phantom_node_tags` idiom) so the capture can discriminate — an "already declared" heuristic mis-classifies a foreign decl that precedes its owning rank's bracket. The phantom-claim degrade (stage-claimed `node_to_surface`) is unchanged and stays fail-loud.
- **One `partition_NN` group per rank, ever.** Stage re-brackets RESUME the rank's accumulator instead of growing duplicate groups (which would also corrupt the write-time boundary-node intersection — two blocks of the same rank would see each other as "another rank"). `/opensees/partitions` and the `element_meta/*/partition_ids` columns now cover stage-owned topology.
- **Contract:** the stage zone of a partitioned build is CONTENT-equal (rank-major capture order) to the same model captured unpartitioned, masking the INV-5 `*_runtime_fallback` chain keys; `from_h5 → to_h5` of a partitioned staged archive is `model_hash`-stable; two fresh builds hash identically. Locked by `tests/opensees/h5/test_h5_partitioned_staged_capture.py`.
- **Schema 2.19.0** (no layout change — the bump marks that partitioned staged archives now exist; hard-floor window applies). The `ops.domain_capture` `bridge=None` degrade for partitioned staged builds is RETAINED pending its own verification slice (P5.3) — the run-file payoff (stage-aware viewer on partitioned SSI runs) lands there.

### FIXED — partitioned H5 archives: capture dedupe + partitions restore + INV-5 fallback round-trip (ADR 0055 Phase 5 / P5.0)

Three latent defects in the partitioned (non-staged) `model.h5` round trip, found while scoping ADR 0055 Phase 5 (`internal_docs/plan_staged_h5_phase5_partitioned.md`):

- **Rank-replicated captures duplicated rows in the archive.** The partitioned global pass replicates emission across owning ranks by design (ADR 0027 INV-1/INV-4 — a shared-boundary-node `fix` emits inside every owning rank's bracket; a cross-rank MP constraint replicates byte-identically), and the H5 capture stored the replicas verbatim: duplicated `/opensees/bcs/{fix,mass}` rows, duplicated `/opensees/constraints/*` rows, and a flat `from_h5 → build()` re-emit that double-applied them (doubled penalty stiffness, duplicate equalDOFs). Worse, a `Plain` pattern whose targets span ranks re-opens the same tag once per rank — two capture records with one tag **crashed `ops.h5` outright** on the `patterns/<type>_<tag>` group-name collision. Partition-bracketed captures now dedupe on full record identity, and a per-rank pattern re-open resumes the captured record (per-rank line subsets merge into the one logical pattern; shared-node load/sp lines capture once). Flat-build capture is untouched.
- **`from_h5 → to_h5` dropped the partition zone.** The re-write path was partition-blind: `OpenSeesModel` never loaded `/opensees/partitions` and the repopulated emitter re-drove the element pool with no rank brackets — the re-written archive lost the `partition_NN` groups (+ `boundary_node_ids`) and degraded every `element_meta/*/partition_ids` column to `-1`, drifting `model_hash` (both zones fold in). `OpenSeesModel` now carries the partition records (`.partitions()` accessor) and `to_h5` echoes them back through the new `H5Emitter.restore_partition_blocks` (the `restore_stage_blocks` store-and-echo pattern; `boundary_node_ids` recomputes byte-identically from the restored node sets, and `_element_ranks` re-stamps by tag).
- **The ADR 0027 INV-5 runtime-conditional chain degraded on replay.** A partitioned build auto-emits `numberer ParallelPlain`-with-`RCM`-fallback / `system Mumps`-with-`UmfPack`-fallback; the fallback attrs persisted but `_replay_analysis_chain` re-drove only the bare primaries — the H5 re-write silently dropped `*_runtime_fallback` (hash drift) and a tcl/py re-emit lost single-process portability. Replay now re-drives `parallel_runtime_fallback_numberer/system` when the attrs carry fallbacks.

A `from_h5 → to_h5` of a partitioned archive is now `model_hash`-stable end-to-end (locked by `tests/opensees/h5/test_h5_partitions_roundtrip.py`). Note: archives **written before this fix** carry the duplicated rows, so re-writing one produces a (correctly) different hash — the lineage chain warns, never raises.

### FIXED — viewer state-contract V5: the projection audit (ADR 0056)

The last adoption slice: every panel across the three viewers swept against INV-1 ("widgets and write-only attributes are never the sole holder; every panel can rebuild from owners alone"). The panels conform — except the shared Session tab (`PreferencesTab`), which failed three ways and is fixed:

- **The "Load arrows" slider never worked.** It fired a `"load_arrow"` scale key no owner ever had — a silent no-op from the day it shipped (writes went into the pre-V3 raw dict that no reader consumed) and a `KeyError` crash in the Qt handler after V3 made `set_scale` fail loud. Slider rows now use the owner's key vocabulary (`force_arrow` / `moment_arrow` / …), locked both ways against `OverlayVisibilityModel._SCALE_KEY_TO_OVERLAY` by the new `tests/viewers/test_preferences_projection.py`.
- **Sliders and the pick-color swatch initialize from their owners** (`overlay_scales=` / `pick_color=` constructor params; new public `ColorManager.pick_rgb` read path) instead of hardcoding 1.0× / `#E74C3C` — the can't-rebuild-from-owners gap INV-1 exists to kill. The model viewer constructs its `ColorManager` before the Session tab so the swatch has an owner to read.
- **Unbound controls are not built**: the mesh viewer's Session tab showed a pick-color row wired to nothing; the model viewer's showed five overlay-scale sliders wired to nothing — silent no-op surfaces (INV-6). A control whose callback is not provided is now omitted.

Verified on the live mesh viewer: all six sliders initialize from the owner and dragging "Force arrows" round-trips through `set_scale` — the same gesture crashed pre-V5.

**ADR 0056 is now Accepted** — V5 is the last adoption slice, so the runway (V0 #593 → V1 #597 → V2 #600 → V3 #602 → V4 #603 → V5) is complete and the ADR status flips Proposed → Accepted. Open questions 1 (shared dispatch module) and 3 (ActiveObjects kept) were resolved at V3/V4; question 2's guard widening to `diagrams/` stays gated on the `ui/` allowlist burn-down, as the ADR already decided.

### ADDED — solution-strategy ladder + established profiles (ADR 0057 Phase A)

`ops.strategy.Ladder(rungs=[...])` + `ops.strategy.profile(name)` attach an opt-in escalation ladder to an analyze loop via `s.run(..., strategy=)` (staged) or `apeSees.analyze(..., strategy=)` (flat live). The deck emitters (py + tcl) turn the #587 fail-loud per-increment loop into a rung-walking loop: rung 0 — the chain's own algorithm — gets first shot at every increment; a failed `analyze 1` re-issues the next rung's `algorithm` command and retries the *same* increment with a loud provenance print; a rescued increment restores rung 0; exhausting the ladder aborts with the fail-loud banner naming the ladder and rung count. The live emitter runs the same walk in-process and logs escalations to `LiveOpsEmitter.strategy_events`. `strategy=None` emission stays byte-identical to the pre-0057 loop.

**Established profiles** (stable names, evidence-revisable orderings): `"standard"`, `"non-smooth"` (aliases `"geotech"` / `"mohr-coulomb"` — deliberately **no line-search rung**: the 2026-06-10 zoned-tunnel campaign showed `NewtonLineSearch` stalling in five mesh/element configurations that plain `Newton` carried at identical tolerance), `"smooth-hardening"` (alias `"metal"`), `"penalty-stiff"`, `"exhaustive"`. Rungs are solution algorithms ONLY — tolerance relaxation, test swaps and integrator changes are excluded by design (ADR 0057 §6). H5 persistence of the declaration is Phase C (an H5 replay runs the plain loop); `Substep` rungs with exact-λ landing are Phase B.

### ADDED — `LadrunoCST` 3-node constant-strain triangle (Ladruno fork, tag 33008)

`ops.element.LadrunoCST(pg=, material=, thickness=, plane_type=, pressure=,
rho=, body_force=)` emits the fork's 3-node constant-strain triangle — the thin
2D sibling of `LadrunoQuad` (and the second of the four plane / coupling fork
features). A 1-point triangle is rank-sufficient, so there is **no
`-formulation` axis**; it reduces to upstream `Tri31`:

```
element LadrunoCST $tag $n1 $n2 $n3 $matTag [-type PlaneStrain|PlaneStress]
    [-thick $t] [-rho $r] [-body $bx $by] [-pressure $p]
```

- Same fork-gating (`_FORK_ONLY_ELEMENTS`), builder-ndf bracket
  (`_BUILDER_NDF_GATED`, the parser hard-gates `ndm/ndf == 2/2`), and fail-loud
  validation (`thickness > 0`, `plane_type ∈ {PlaneStrain, PlaneStress}`) as
  `LadrunoQuad`.
- Registry: gmsh tri3 (etype 2), `ndm/ndf={2}`, identity reorder.
- Result reads: `RESPONSE_CATALOG` `Triangle_GL_1` (1 GP, `stress_*`/`strain_*`),
  the same layout as `Tri31`.
- `PlaneStrain` is elided; the required `-thick` is always emitted.

The CST honestly volumetrically locks / mesh-biases localization — prefer
`LadrunoQuad` or `BezierTri6` for real 2D work (guide §CST). The RBE2 / RBE3
coupling elements remain follow-up PRs.

### ADDED — `LadrunoQuad` 2D plane continuum element (Ladruno fork, tag 33007)

`ops.element.LadrunoQuad(pg=, material=, thickness=, formulation=, plane_type=,
pressure=, rho=, body_force=)` emits the fork's unified 4-node plane
(plane-stress / plane-strain) continuum element — the 2D sibling of
`LadrunoBrick`, with the anti-locking treatment folded into one `-formulation`
selector (`std` / `bbar` / `ssp`):

```
element LadrunoQuad $tag $n1..$n4 $matTag [-formulation std|bbar|ssp]
    [-type PlaneStrain|PlaneStress] [-thick $t] [-rho $r] [-body $bx $by] [-pressure $p]
```

- **Fork-only, gated at run not emit.** Emission (`ops.tcl` / `ops.py`) works on
  any build; the live (`ops.run()`) path raises a clear *"requires the Ladruno
  fork build"* error on stock openseespy (added to `_FORK_ONLY_ELEMENTS`).
- **Fail-loud parse-guard parity:** `formulation='eas'` is rejected with a
  targeted "reserved (ADR 25 Phase 3)" message; `formulation='bbar'` +
  `plane_type='PlaneStress'` is rejected (volumetric locking is a plane-strain
  phenomenon) — mirroring the fork's `OPS_LadrunoQuad.cpp` guards. `thickness`
  is required and validated `> 0`.
- **Builder-ndf bracket:** the fork parser hard-gates on `ndm/ndf == 2/2`, so
  `LadrunoQuad` joins `_BUILDER_NDF_GATED` — the emit orchestrator brackets the
  block with `model -ndf 2` so it survives a mixed-ndf envelope (like `quad` /
  `tri6n`).
- **Result reads:** registered in `RESPONSE_CATALOG` as `Quad_GL_2` (4 GPs,
  `stress_*`/`strain_*`) for every formulation — `ssp` mirrors its centroid
  onto all 4 GP blocks, so the layout matches `FourNodeQuad`.

`std`/`PlaneStrain` defaults are elided from the deck; the required `-thick` is
always emitted. `LadrunoCST` (the 3-node sibling, tag 33008) and the RBE2/RBE3
coupling elements are follow-up PRs.

### CHANGED — viewer state-contract V4: the model viewer joins the dispatcher (ADR 0056)

The last viewer joins; the contract now covers all three. Plus the burn-downs:

- **Model viewer dispatcher**: constructed next to its `VisibilityManager`, `entities` pump bound to `rebuild_now()`. The 8 call-site `plotter.render()` calls after visibility mutators (tree hide/isolate/reveal, parts hide/isolate, toolbar H/I/R) **and** the `vis_mgr.on_changed` render subscriber are deleted — the model viewer used to **double-render** every visibility gesture (subscriber + call-site); it now renders once via the dispatcher.
- **`VisibilityManager`'s "until V4" transitional comment retired**: both production viewers inject a dispatcher; the no-dispatcher mode is documented as standalone/unit-test inline reconcile (it contains no `render()` — it was never a render fallback).
- **ActiveObjects disposition resolved (ADR 0056 OQ3)**: **kept** as the dedicated per-viewer focus-state owner (active layer/geometry/stage/step/pick-mode/selection snapshot) — the V4 census showed it conforms to the contract (owns focus state, fires owner-side, never touches artifacts or render), so the Part-6 "fold or delete" clause is superseded; dated resolution recorded in the ADR. One mechanism per *concern*: dispatcher → reconciler, ActiveObjects → UI projections.
- **Guard widened to `model_viewer.py`** (renders 8 = 1 dispatcher binding + 7 out-of-scope subsystems; artifacts 3; imports 0 — live-fire measured).

### CHANGED — viewer state-contract V3: the mesh viewer joins the dispatcher (ADR 0056)

The mesh viewer now runs the same owner-fires / dispatcher / reconciler contract as the results viewer (one shared `Dispatcher` class — ADR 0056 open question 1 resolved: kinds are data, the module is shared):

- **Two mesh primitives** on the shared dispatcher: `entities` (the `VisibilityManager` actor rebuild, re-homed as the pump — its mutators now owner-fire `MESH_ENTITY_VISIBILITY_CHANGED` instead of rebuilding inline; legacy inline path kept for the model viewer until V4) and `overlays` (keyed overlay rebuild — `MESH_OVERLAY_CHANGED` carries the overlay key as a pass-through scope, `None` = rebuild all, which is exactly what a `gesture_batch` replay produces).
- **`OverlayVisibilityModel` owner-fires** with the affected overlay key; the four per-overlay observer callbacks (each ending in its own `plotter.render()`) became the single overlays pump. Its plain observer list survives for UI-sync subscribers (outline tree).
- **Overlay glyph scales are owned state**: mesh_viewer's module-private `_overlay_scales` dict moved into `OverlayVisibilityModel` (`scale(key)` / `set_scale(key, value)` — idempotent, owner-fired, unknown keys fail loud). The scale UI callbacks are now thin mutator calls.
- **14 scattered `plotter.render()` calls deleted** from mesh_viewer (visibility subscriber, all overlay-rebuild renders incl. the early-return ones, hide/isolate/reveal handlers) — the dispatcher's coalesced render is the render path. The 9 that remain (labels, wireframe, edges, dim filter, prefs, hover/selection recolor) belong to subsystems outside V3 scope and are ratcheted in the guard.
- **Guard scope widened** (`test_viewer_state_contract.py`): `mesh_viewer.py` + `overlays/**` now guarded with measured count-ratchet allowlists (the AST count again beat the survey — overlay `SetVisibility` property calls the grep missed).

### ADDED — viewer state-contract V2: the AST guard (ADR 0056 INV-5)

`tests/viewers/test_viewer_state_contract.py` machine-enforces the contract over `src/apeGmsh/viewers/ui/**`: **G-RENDER** (no `.render()` call expressions), **G-ARTIFACT** (no `SetVisibility` / `set_layer_visible` / `SetPickable` / `add_mesh` / `remove_actor` — zero baseline, hard gate from day one), **G-IMPORT** (no `pyvista` / `vtk*` / `pyvistaqt` / `apeGmsh.viewers.backends` imports, absolute or relative). Allowlists are per-file violation *counts* with a two-way ratchet — exceeding fails, and undershooting fails too with a "ratchet down" message, so the allowlist can only shrink. The durable carve-out is `viewer_window.py` (5 control-layer renders: camera presets / projection toggle / fit-view / theme refresh; 2 imports: it constructs the `QtInteractor` and applies pyvista theme defaults). The guard out-performed the grep survey on its first run — it caught a `pyvistaqt` import the regex baseline missed. Scope widens with each adoption slice (V3: `mesh_viewer.py` + `overlays/`; V4: `model_viewer.py`).

### FIXED — ActiveObjects initial-state seed + the never-run Qt window tests now run (per-file)

Installing `pytest-qt` locally un-skipped the four `@pytest.mark.qt` full-window lifecycle tests in `test_results_viewer_smoke.py` (plus a fifth in `test_stage_activation.py`) — they had **never executed**, and running them surfaced two real issues:

- **`ActiveObjects` never learned the initial stage/step.** The director auto-picks a stage at `__init__` — before the viewer wires the ActiveObjects bridge — so on single-stage results no change event ever fires and `active_stage` stayed `None` forever (a projection that can't rebuild from its owner; ADR 0056 INV-1). `ResultsViewer._show_impl` now seeds ActiveObjects from the director's current stage/step right after wiring the bridge.
- **Bit-rot in the skipped test**: `Results.stage_ids()` no longer exists; the test now reads the `Results.stages` property.
- **`qt` marker registered + deselected by default** (`addopts -m "not qt"`): mixing real `QtInteractor` windows with the suite's offscreen plotters (or with another qt file in the same process) hits a native access violation in interactor init. Every qt test passes per-file in a fresh process — `pytest -m qt <file>`. CI is unaffected (no pytest-qt there; they skip).

Verified on a desktop session with working GL: live-viewer visual gallery (deformed shape + ghost + composition gate + scale + `gesture_batch`) screenshotted and pixel-diffed — the #593 composition-gate and ghost fixes and the #597 one-render-per-cascade are confirmed on real pixels.

### CHANGED — viewer state-contract V1: dispatcher-always + owners-fire (ADR 0056)

The load-bearing slice of ADR 0056. Three structural changes to the results-viewer event pipeline, all behavior-preserving for single gestures and strictly better for cascades:

- **The dispatcher always exists (Part 3).** `ResultsDirector.__init__` constructs the `Dispatcher` with no-op pump defaults; `ResultsViewer.show()` rebinds the real pumps via the new `Dispatcher.bind(...)`. All seven `getattr(director, "dispatcher", None)` defenses and both raw-render fallback branches (outline `_fire_layer_visibility`/`_fire_render`, settings-tab `_fire_render`) are **deleted** — headless contexts now exercise the same event path as the live viewer. Bonus: the settings tab's per-applier `plotter.render()` is gone, so a commit renders once instead of once per applier.
- **Owners fire events (Part 2).** `DiagramRegistry.set_visible` now fires `LAYER_VISIBILITY_CHANGED` itself (idempotent per call — a no-op write skips notify + fire); the call-site fires in the settings-tab checkbox and outline eye-toggle are removed. The remaining UI fires (`DIAGRAM_MODIFIED`/`ATTACHED`/`DETACHED`, `LAYER_REORDERED`) go direct — no defense.
- **`gesture_batch()` + the matrix as data.** The dispatcher's elif chain is refactored into a declarative `_MATRIX` table (one row per event kind, fixed primitive order step→deform→restack→gate; `tests/viewers/test_dispatcher_contract.py` locks every row against the legacy behavior). The new `Dispatcher.gesture_batch()` replays the **matrix-row union** of kinds fired inside the block — the outline's composition/geometry eye cascades now wrap in it, so an N-layer cascade costs one gate pump + one render (owner-fires would otherwise have made it N of each).

### REMOVED — deprecated standalone `apeGmshViewer/` app

The top-level `apeGmshViewer/` package (~3,800 lines: `MainWindow`, VTU/PVD loaders, panels, its own theme/renderer/probes) is deleted. It was the pre-rebuild standalone post-processing viewer, fully superseded by the integrated `ResultsViewer` (`results.viewer()` / `results.show_web()`); the integrated probe overlay was mined from it back in the viewer rebuild and nothing in `src/apeGmsh/` imported it except one legacy convenience wrapper:

- **`g.mesh.results_viewer(...)` is removed with it** — it was a thin wrapper that spawned the old app on a `.vtu`/`.pvd` path (its `point_data=`/`cell_data=` branch already raised `NotImplementedError` pointing at the rebuilt viewer). Post-solve visualization goes through `Results(...).viewer()` / `results.show_web()`. `g.mesh.viewer()` (authoring viewer) is unchanged.
- Packaging follows: the `apegmsh-viewer` console script, the `apeGmshViewer*` setuptools include (and the root `where=["."]` entry that existed only for it), and the package's mypy skip-override are gone. `tests/test_vtu_loader.py` (tested the deleted loaders) is deleted.
- Docs/skill scrubbed: README viewers section + repo layout, `docs/api/viewers.md`, the apegmsh skill (canonical + derived mirror resynced via `scripts/sync_skill.py`). Historical mentions in `internal_docs/` plans, `architecture/*.md` design docs, and old example notebooks (`example_plate_viewer.ipynb` etc., which demo the removed app) are left as historical record.

Four event/state bugs behind the chronic "diagrams toggle inconsistently / hide retains nodes" class, all traced to mutations that bypassed the dispatcher or poked dead state:

- **Composition gate was a silent no-op for every diagram.** `pump_gate` flipped `d._actors` directly, but the ADR 0042 R-B migration moved all 11 diagram kinds onto backend layer handles — `_actors` is empty for all of them, so composition-based show/hide never reached the screen. The gate now routes through a new polymorphic `Diagram.apply_effective_visibility(effective)` channel which reuses each subclass's `set_visible` artifact path **without clobbering `is_visible`** (the user-intent flag the gate itself reads — writing the gated value back would corrupt the next gate run).
- **Outline eye-toggles bypassed the dispatcher.** The outline tree's layer / composition / geometry eye icons mutated visibility and called `plotter.render()` directly — `pump_gate` never re-ran, so an eye-on could leak a layer past the active-composition filter, renders didn't coalesce, and the settings-tab checkbox (which fires `LAYER_VISIBILITY_CHANGED`) and the outline drifted apart. All three paths now fire `LAYER_VISIBILITY_CHANGED` (raw-render fallback kept for headless contexts without a dispatcher).
- **Deformed-shape undeformed ghost resurrected.** The runtime "show undeformed" toggle was write-only state on the backend handle: `set_visible(True)` (and now every gate pump) flipped *both* handles back on, stage-change re-attach rebuilt the ghost from the attach-time style, and the settings tab read a `_runtime_show_undeformed` attribute nobody ever wrote (checkbox state lost on every tab rebuild). The toggle is now recorded on the diagram, honored by `set_visible` / the gate / re-attach, and visible to the settings tab.
- **Ghost nodes after hide are now loud.** When the mesh-viewer scene build's per-entity `getNodes` pass fails, the node cloud has no ownership data and a later hide deliberately retains all nodes (locked fallback) — but both the scene-build failure and the fallback were silent, indistinguishable from a visibility bug. Both now log warnings (`scene.node_centroid_pass_failed`, `visibility.node_cloud_no_ownership_data`).

This is the surgical slice of the viewer state-ownership work; the state/event contract ADR (single store per viewer, dispatcher-only writes, reconciler-only backend calls, AST guard) follows separately.

### ADDED — staged-model H5 archival, write + read (ADR 0055 Phase 2, schema 2.18.0)

Non-partitioned staged builds (`ops.stage(...)`) now **archive to and load from `model.h5`** — previously `apeSees.h5()` failed loud on any staged build, and the whole `Results`/viewer chain (which requires `model=` per ADR 0020 INV-1) was unreachable for staged runs.

- **Write (P2.1):** the H5 emitter captures the per-stage emit stream in-band into per-stage buckets (`stage_open`…`stage_close` bracket) — owned nodes/elements (emit order == replay order), stage-bound fix/mass, regions (with `kind` + `emit_index` provenance — the rayleigh-vs-region interleaving carries OpenSees overwrite semantics), MP constraints, patterns incl. the ADR 0052 HOLD pattern (`role="hold"`, `sp_holds` pairs), recorders, global-form rayleigh + emit-order stamps, SSI-2.E mutators, and the per-stage analysis chain + analyze loop — and persists them under `/opensees/stages/stage_NNN`. The declarative complement (`activated_pgs`, per-stage `initial_stress`, `activate_absorbing`) rides a `set_stage_records` side-channel with fail-loud capture↔record cross-checks. Tri-state is presence-encoded (never-set ⇒ no attr). Staged files carry **no** global `/opensees/analysis` (each stage's chain is scoped — the phantom-leak class of bug is eliminated by construction). Vanilla files are byte-identical; the zone folds into `model_hash`. `ops.h5` now raises only for **partitioned** staged builds (ADR 0055 Phase 5).
- **Read (P2.2):** `OpenSeesModel.from_h5` loads staged archives; `.stages()` returns value-form `StageRecordRO` records; `to_h5` echoes them back (`from_h5 → to_h5` is `model_hash`-stable — the store-and-echo acceptance gate). A structurally inconsistent stages zone fails loud (`MalformedH5Error`) rather than loading as a different staged program. The tcl/py/live re-emit targets fail loud until the staged replay lands (P2.3); `ModelData.from_h5` warns on staged archives (a `write()` re-save would strip them). Recorder group names (`{kind}_{idx}`, unpadded) are now read back in numeric emit order on both the staged and flat paths (alphabetical iteration scrambled mixed kinds and `_10` before `_2`).

Both slices were adversarially panel-reviewed pre-PR (plan gate caught 4 fatal design defects; diff gates caught the phantom-node coordinate gap, the direct-`write()` bypass, recorder-order hash drift, equalDOF pad leakage, and the silent-default corruption-laundering holes — all fixed before merge).

### FIXED / INTERNAL — `apeGmsh.opensees` is mypy-clean (178 → 0); `MYPY_BASELINE: 0`

The bridge package now passes `mypy src/apeGmsh/opensees` with zero errors,
and the `static-gates` CI ratchet is lowered to a hard gate (`MYPY_BASELINE: 0`).
Two of the 178 were real API-surface bugs:

- **`ops.pattern.Plain/UniformExcitation(series=…)` falsely rejected newer
  TimeSeries.** The `series=` annotation was a concrete union frozen at
  Linear/Constant/Path/Trig/Pulse — the Ricker wavelet and the cyclic
  protocols (PR #558) never made it in, so type-checked callers were
  rejected and the internal stage-pattern helpers (typed against the
  abstract base) didn't type-check at all. `series=` now accepts
  `TimeSeries | str` (any subclass; runtime resolution is unchanged).
- **Quoted-annotation names that were never imported** (`Plain`,
  `ConstraintRecord`, `Any`) blinded mypy at ~20 signatures (fixed in the
  ruff sweep, completed here).

Everything else is annotation/narrowing only — zero behavior change:
renamed reused loop variables, fail-loud guards on Optionals whose
invariants are now stated in the error message (e.g. `split='parts'`
requires a buffered Tcl/Py emitter), removed stale `type: ignore`s, and
`type-abstract` disabled package-wide (the `_resolve(ref, base=…)`
convention only isinstance-checks `base`; direct abstract instantiation
is still caught by the `abstract` code). Docs sweep alongside: README
viewer callout de-versioned (was "v1.5.0"), `guide_loads.md` broken TOC
anchor, `docs/api/loads.md` autorefs updated `pattern` → `case`
(ADR 0051 vocabulary).

### ADDED / FIXED — absorbing-skin aspect warning, centred-box fix, rotation guard (ADR 0054, AB-1c close-out)

Closes AB-1c. Three things, all surfaced by source/run verification:

- **Aspect-ratio warning.** A new fail-soft `WarnAbsorbingSkinAspect` (from
  `apeGmsh.parts.plane_wave_box`) fires when the absorbing skin is much thicker
  than its adjacent soil element (ratio > ~4×; STKO ships ~2:1, the matched
  default is 1× and silent) — elongated boundary hexes absorb poorly. Both entry
  points (`add_plane_wave_box`, `add_absorbing_shell`).
- **Centred-box mesh fix.** `add_plane_wave_box(center=…)` translates the block
  via an OCC `translate` + `synchronize`, which renumbers entities and stranded
  the slice's `_metadata` keys — a later `g.mesh.generate` then tripped the
  pre-mesh stale-metadata validator. The builder now reaps them
  (`remove_orphans()`) after the move, so a centred box meshes cleanly (latent
  since AB-1a; no prior test combined `center != 0` with `generate`).
- **Rotation is unsupported — by the element, not apeGmsh.** The OpenSees
  `ASDAbsorbingBoundary3D` element requires boundary-face normals along global X
  or Y (`ASDAbsorbingBoundary3D.cpp:2135`: *"normal vector can be only X or
  Y"*), so a rotated absorbing box is rejected by the solver. `rotation_z_deg`
  now raises a clear `ValueError` explaining this (was a vague "later slice"
  `NotImplementedError`). Per-axis `skin_thickness` was already supported since
  AB-1a. **ADR 0054 AB-1c is complete; rotation is closed as infeasible.**

### ADDED — layered (stratified) absorbing boxes + per-layer material (ADR 0054, AB-1c)

Stratified soil support for both `ASDAbsorbingBoundary` entry points, so a layered
site gets the **correct per-layer impedance** on its lateral skin (a uniform skin
on a layered column spuriously reflects at the quiet boundary).

`g.parts.add_plane_wave_box(z=[(15, 3), (25, 5)])` (a top → bottom list of
`(depth, n_elements)` layers) and `g.parts.add_absorbing_shell(box=…,
element_size=…, layers=[(15, 3), (25, 5)])` (depths summing to the box's z-extent)
now stratify the soil **and** split the lateral skin (`L/R/F/K`) per layer; the
base skin (`B*`) sits on the bottom layer. `AbsorbingSkinResult` gains `n_layers`,
`soil_pgs` (per-layer soil PGs, top → bottom — emit one `stdBrick` per entry) and
`skin_pgs_by_layer` (`layer → {btype → PG}`); the existing fields are unchanged and
a single-layer build is byte-identical to before. The bridge gains
`ops.element.absorbing_boundary(skin=res, materials=[m0, m1, …])` — one material per
layer (top → bottom, `len == n_layers`, mutually exclusive with the homogeneous
`material=`), each layer's skin cells getting that layer's derived `G = E/2(1+ν)`;
`base_series` still rides the bottom (`B`-containing) cells only. The layering lives
entirely in the shared classify/tag core (`_tag_and_structure` per `(btype, layer)`)
plus a `_layered_axis_z` helper; the BYO path slices the box at the layer interfaces
before the weld. Rotation, grading, and per-axis thickness remain deferred (rest of
AB-1c). Tests: turnkey + BYO 2-layer structure, per-layer-material deck (each layer's
`G`), and guards; `tests/parts` green; a live 2-layer transient solves with each
layer's impedance applied. Builds on AB-1b (#573).

### ADDED — `g.parts.add_absorbing_shell` — bring-your-own-box absorbing skin (ADR 0054, AB-1b)

The second entry point for the `ASDAbsorbingBoundary` skin (companion to
`add_plane_wave_box`): you build the soil box (its placement, PGs, the material /
structure you put on it later) and this welds a one-element-thick absorbing skin
onto its five truncation faces, returning the **same** `AbsorbingSkinResult` so the
bridge element (`ops.element.absorbing_boundary`) and the staged flip
(`s.activate_absorbing`) consume it identically.

`g.parts.add_absorbing_shell(*, box, element_size, skin_thickness=None,
faces=None, name=None, names=None, apply_transfinite=True)` resolves `box` to a
single **axis-aligned rectangular** volume (PG / label name or handle; fail-loud on
multi-volume / rotated / curved geometry via a mass-vs-bounding-box check), builds
the ≤17 skin slabs around it, and boolean-`fragment`-welds them on (conformal
shared faces). It then reuses the AB-1a classify → PG → transfinite tail (extracted
into a shared `_tag_and_structure`). The mesh contract is **size-based**: gmsh
cannot report transfinite counts back and the weld renumbers entities, so the box's
prior mesh state is irrelevant — `element_size` (scalar or `(sx,sy,sz)`)
(re)structures box + skin together after the weld. `skin_thickness` defaults to
`element_size`; `faces=` restricts the skin to a subset of `("L","R","F","K","B")`
(e.g. omit a symmetry plane). When `box` is a name its soil PG is reported as-is
(no duplicate is synthesised). The slabs are synchronised **before** the weld — a
synced box fragmented against unsynced slabs leaves coincident-but-separate faces
(duplicate interface nodes → a disconnected, singular model); a node-sharing
invariant test locks this. 13 tests in `tests/parts/test_absorbing_shell.py`
(btype distribution, conformal all-hex, `faces=` restriction, soil-PG handling,
fail-loud guards, and a bridge-deck plug-in proving drop-in for AB-2/AB-3); full
`tests/parts` 72/72. A live transient over a welded box is byte-identical to the
AB-4 plane-wave example (surface arrival at `H/Vs`, late motion 0.93 % of peak).
Rotation / layered-Z / graded skins remain AB-1c.

### ADDED — plane-wave SSI worked example (ADR 0054, AB-4)

Closes the `ASDAbsorbingBoundary` arc (AB-1a → AB-4) with a run-verified,
end-to-end docs example, [`docs/examples/plane-wave-ssi.md`](examples/plane-wave-ssi.md):
a soil column built by `g.parts.add_plane_wave_box`, wrapped by an
`ASDAbsorbingBoundary3D` skin via `ops.element.absorbing_boundary` (a base
shear-velocity `Ricker` injected on the bottom faces), flipped to absorbing in a
staged block with `s.activate_absorbing`, and driven through an implicit Newmark
transient. The example checks the physics directly: the base pulse reaches the
free surface at the shear-wave traveltime `H/Vs` (0.198 s measured vs 0.200 s),
then **radiates out** the quiet base instead of reflecting (late-window surface
velocity 0.93 % of peak). Registered in the examples index and the mkdocs nav,
with a surface-velocity time-history figure. Docs-only — no library code changed.

### FIXED — loads / masses now fit the per-node `ndf`, not the model envelope

Nodal loads and masses are now sized to **each node's own (per-node) `ndf`** at emit, not the model-wide `ops.model(ndm, ndf)` envelope. Per ADR 0048 the per-node `ndf` (inferred from the declared elements, ∪ the `ops.ndf` overlay) is authoritative and every node already emits `-ndf <its value>`; a load/mass vector that did not match that count was silently dropped by OpenSees (`Node::addUnbalancedLoad` / `setMass` warn-and-return on a size mismatch).

The user-visible bug: in a mixed-`ndf` model (e.g. a 6-DOF envelope holding 3-DOF solid/truss nodes), a `p.from_model(case)` import mapped its DOF-agnostic spatial force onto the **envelope** `ndf` — emitting a 6-component `load` on a 3-DOF node, which OpenSees then dropped. The tip force just vanished, with only an stderr warning. This affected the flat, staged, split, and partitioned `from_model` paths.

The fix introduces `fit_dof_vector` (`opensees/_internal/build.py`): every `load` / `mass` vector is fitted to the target node's `ndf` — a **short** vector is zero-padded on the trailing DOFs (so `ops.mass(pg, (m, m, m))` on a 6-DOF beam node is the natural translational-only mass), and a vector with a **non-zero** component beyond the node's `ndf` (e.g. a moment on a 3-DOF solid node, or `Fz` on a 2-DOF planar node) fails loud with a `BridgeError` naming the lost component. `from_model` loads map through `broker_load_components` at the **per-node** `ndf`, so an uncarriable spatial component likewise fails loud per node.

**Behavior change (G3):** `validate_record_ndf_consistency` (ADR 0049 gate G3) previously required `load` / `mass` vectors to **exactly** equal the node `ndf`; it now accepts a short vector (padded) and raises only on a non-zero overflow. `fix` / `support` mask-length and `sp` DOF-index checks are unchanged. The user sets the `ndf` (via elements / `ops.ndf`) and the bridge makes the loads and masses compatible with it, failing loud only when a real component cannot land.

### ADDED — node-pair `ops.element.<spring>(nodes=(node_i, node_j))` (ADR 0049)

Completes the SSI spring-to-ground story opened by ADR 0049: `ops.element.ZeroLength`, `CoupledZeroLength`, and `TwoNodeLink` now accept **`nodes=(node_i, node_j)`** as an alternative to `pg=`, wiring a single spring directly to two explicit endpoints — at least one typically a `g.decouple_node` ground — **without a meshed 2-node "line" physical group**. Each endpoint (`NodeRef`) is a `g.decouple_node` handle, a node-label string resolving to exactly one node, or an int tag (escape hatch, not compose-safe). `pg=` and `nodes=` are mutually exclusive (exactly one, fail-loud). `ZeroLengthSection` keeps `pg=`-only — it is non-adaptive (both nodes must carry exactly 3/6 dof), so a decoupled ground cannot be sized for it; passing `nodes=` to it fails loud pointing at plain `ZeroLength`.

The endpoint `ndf`s are validated by the existing **G1** equal-endpoint gate (the spring's two ends must carry equal `ndf`, fed the inferred ∪ `ops.ndf` overlay map), and distinct resolved tags are required (OpenSees has no same-node guard — an `i == j` pair would assemble a singular element silently). Connectivity round-trips `model.h5` via a new optional `inline_connectivity` dataset under `/opensees/element_meta/{type}/` (the spring has no neutral gmsh cell to source it from), folding into `model_hash` (schema **2.17.0**, additive). Node-pair springs are **global-only** in v1 (no stage binding) and **fail loud under partitioned (MPI) emit** (per-rank node-ownership routing of an explicit node-pair is deferred); they are also not drawn in the mesh/model viewer (no neutral cell). Plan hardened by an adversarial Opus panel before coding (caught the false "H5 round-trip is free" claim, the `ZeroLengthSection` non-adaptive trap, the `expand_pg_to_elements(None)`-returns-whole-mesh trap, and 2 more `fem_eid` sentinel-collision sites).

### ADDED — `s.activate_absorbing()` — staged absorbing-boundary stage flip (ADR 0054, AB-3)

Completes the gravity→dynamic SSI cycle for the absorbing boundary. Inside a
staged block, `s.activate_absorbing(pg=<skin_all_pg> | elements=[...])` emits the
one-way `ASDAbsorbingBoundary` stage switch (0→1) — the OpenSees one-shot
`parameter $pid` / `addToParameter $pid element $eid stage` (one per element) /
`updateParameter $pid 1` / `remove parameter $pid` sequence — once, after the
stage's analysis chain is established and **before** its `analyze` loop, so a
prior gravity stage has already held the boundary by penalty. Targets resolve by
PG (typically `AbsorbingSkinResult.skin_all_pg`) or an explicit element list;
exactly one is required. Reuses the `s.initial_stress`
`parameter`/`addToParameter` plumbing and the `fem_eid -> ops_tag` map, and is
emitted **per partition** under MP (each rank flips only its owned elements; an
eid absent from a rank's tag map is silently skipped, while in single-partition
mode an unregistered eid is a fail-loud `BridgeError`). New `flip_element_stage`
emitter method on the Tcl / openseespy / live / recording backends (no-op for
H5 — analysis directives aren't archived). End-to-end over a plane-wave box
emits one `addToParameter ... stage` per skin element, before the transient
`analyze`.

### ADDED — `ASDAbsorbingBoundary3D` bridge element + `ops.element.absorbing_boundary` (ADR 0054, AB-2)

The OpenSees-side counterpart to the AB-1a plane-wave box: a typed
`ASDAbsorbingBoundary3D` element primitive
(`opensees/element/absorbing.py`) and two namespace facades. The element emits
`element ASDAbsorbingBoundary3D $tag $n1..$n8 $G $v $rho $btype <-fx $ts> <-fy $ts> <-fz $ts>`
— **raw** `G`/`v`/`rho` doubles (not a matTag) plus the fixed `btype` string;
optional base-input time series are fail-loud-guarded to bottom (`B`-containing)
boundaries only, and opposite-face / illegal / repeated `btype` letters are
rejected at construction. `ops.element.ASDAbsorbingBoundary3D(pg=, btype=, ...)`
takes the soil properties either as `material=ElasticIsotropic(...)` (derives
`G = E/(2(1+v))`, reuses `v`, `rho` — read at construction, **never emitted** and
not a dependency) or as raw `G=/v=/rho=`. The convenience
`ops.element.absorbing_boundary(skin=<AbsorbingSkinResult>, material=…, base_series=…, base_dirs=…)`
fans one declaration over every btype PG of a plane-wave skin in a single call,
attaching the base series to the bottom PGs only. Registered in
`_ELEM_REGISTRY` (`mat_family="none"`, `ndf_ok={3}`) so ADR-0048 ndf inference
gives the skin nodes solid DOFs. End-to-end (box → mesh → `apeSees` → deck)
reproduces the closed-form btype tally with the base series on every bottom cell
and a single `nDMaterial` line (the soil's; the skin carries raw `G/v/rho`). AB-3
(the staged `s.activate_absorbing()` flip) follows.

### ADDED — `g.parts.add_plane_wave_box` — structured soil box + absorbing skin (ADR 0054, AB-1a)

First slice of the `ASDAbsorbingBoundary3D` track (ADR 0054 /
`internal_docs/plan_absorbing_skin_ab1.md`). `g.parts.add_plane_wave_box(x=(Lx,nx), y=(Ly,ny), z=(Lz,nz), ...)`
builds, **in the live session** (no Part/STEP round-trip), an axis-aligned
structured soil box wrapped by a one-element-thick absorbing **offset shell** on
its five truncation faces — the local `+Z` top is the free surface and is never
shelled. Soil + shell are one rectangular block sliced only at the region
breakpoints into **18 sub-volumes** (1 soil + up to 17 skin regions: 5 face
panels, 4 vertical edges, 4 bottom edges, 4 bottom corners). Each skin region is
tagged with its OpenSees `btype` (the OR-combined set of truncation faces it lies
outside of, canonical order `BLRFK`) as a volume physical group, so the
forthcoming bridge element fans out one `ASDAbsorbingBoundary3D` per skin hex with
the shared btype. Returns an `AbsorbingSkinResult` (`soil_pg`, `skin_pgs` keyed by
btype, `skin_all_pg` roll-up, `bottom_pgs`, `free_surface_pg`, `axes`, placement).
`skin_thickness` defaults to the adjacent soil element size per face; the block is
built in the local frame then translated to `center` (the slice cutting-plane is
sized around the origin). Fail-loud guards: `rotation_z_deg != 0`, layered-`z`,
and non-positive sizes/thickness are rejected (rotation + stratigraphy are later
slices). The btype→axis mapping (`L`=min-X, `R`=max-X, `F`=min-Y, `K`=max-Y,
`B`=min-Z) is validated against the OpenSees element source and a real STKO
export; the golden test reproduces that deck's exact btype tally. Does **not** use
or modify `DRMBox` (that serves the Domain Reduction Method).

### ADDED — `g.model.geometry.add_rectangle(plane="xy"|"yz"|"xz")`

`add_rectangle` was hardwired to the XY plane (it wrapped `gmsh.occ.addRectangle`, which only builds at a given `z`). It now takes a `plane` keyword so a rectangle can be authored directly on any of the three canonical planes, **corner-anchored**: `(x, y, z)` is always the corner and `dx` / `dy` run along the plane's two in-plane axes — `xy` → (X, Y) at constant `z` (default, unchanged), `xz` → (X, Z) at constant `y`, `yz` → (Y, Z) at constant `x`. Backward-compatible: omitting `plane` reproduces the old XY behavior bit-for-bit, and the existing `angles_deg` / `angles_rad` in-place rotation still composes on top (pivoting about the rectangle centre in the chosen plane) for fully arbitrary orientation. Implemented by building in local XY at the origin, rotating onto the target plane, then translating the corner — so `rounded_radius` is preserved on every plane. An unknown `plane` fails loud. (For a centre-anchored square with an arbitrary normal, `add_cutting_plane(point, normal_vector)` already exists.)

### CHANGED — `g.model.geometry.slice` is now dimension-generic and accepts `point=`

`slice` is no longer hardwired to volumes. Two changes:

- **`dim=1|2|3|'all'` (default `'all'`)** — slices entities of the chosen dimension instead of only volumes. `'all'` (the default) slices every *maximal* entity in the model — volumes in a solid model, surfaces in a shell model, curves in a frame model — which collapses to the historical volume-only behaviour for solid models (so existing calls are unchanged). The OCC `fragment` result map is used to keep only the operand's own fragments, so slicing a dim-2 surface with the dim-2 cutting plane no longer miscounts the plane's pieces. Sliced surfaces / curves (which bound no volume) are registered before the orphan sweep, so they survive.
- **`point=` as an alternative to `offset=`** — give the plane location directly as a point it passes through (only the `axis` coordinate matters) instead of a signed distance from the origin. The two are **mutually exclusive** (passing a non-zero `offset` together with `point` raises).
- The first positional argument was **renamed `solid` → `target`** to match the dimension-agnostic meaning. Positional callers are unaffected; the keyword form `slice(solid=…)` must become `slice(target=…)`.

### ADDED — `ops.ndf` for element-less decoupled nodes + per-node ndf gates G1–G3 (ADR 0049 DOF half)

Completes ADR 0049: after ADR 0048 made per-node `ndf` **inferred** from element
classes, an *element-less* decoupled node — an SSI spring/dashpot **ground**, a
control node, a mass anchor declared via `g.decouple_node(...)` that no element
touches — had no way to state its DOF count. New **`ops.ndf(target, ndf=…)`**
(the sole explicit per-node ndf channel) takes a `g.decouple_node` handle or its
int tag and is resolved at build into an overlay merged over the inferred map
(`{**inferred, **overlay}`). It **fails loud** on a mesh node, an element-touched
node (inference owns those — no two-headed model), or an unresolved handle. A
`label=`/`pg=` grammar is deferred (decoupled labels are not yet registered into
the FEM).

Three fail-loud build-time gates close silent-failure modes OpenSees would
otherwise swallow:

- **G1** (existing `validate_adaptive_element_endpoints`, now fed the *effective*
  inferred ∪ overlay map) — both ends of a `zeroLength`-family element must carry
  equal ndf, so a correct `ops.ndf(ground, K)` spring passes instead of falsely
  raising.
- **G2** (`validate_constraint_master_ndf`) — a `rigidDiaphragm`/`rigidLink`
  master must carry the **exact** ndf (6 in 3D / 3 in 2D, `RigidDiaphragm.cpp:
  94-100`), and every DOF an `equalDOF`/`rigidLink`/`kinematic_coupling`
  references must fit both endpoints; covers broker **and** stage-claimed
  constraints.
- **G3** (`validate_record_ndf_consistency`) — a `mass`/nodal-`load` vector must
  **EQUAL** the node ndf (`Node.cpp:940`/`1272` reject any mismatch and silently
  drop the whole record); a `fix`/`support` mask must not exceed it; an `sp` DOF
  index must fit. (`p.from_model(case)` loads are emit-synthesized per node and
  out of G3's reach.)

The resolved per-node ndf (inferred ∪ stated) persists to `/opensees/nodes_ndf`
(schema 2.14.0; current `SCHEMA_VERSION` 2.15.0, **no bump**) and round-trips
through `model.h5`. **Migration:** a nodal `mass`/`load` whose vector length did
not match the node ndf was silently dropped by OpenSees before and now fails
loud at build — pass a full-length (ndf-sized) vector.

### IMPROVED — `LadrunoBrick` rejects a finite-strain material under a non-finite geometry

`ops.element.LadrunoBrick(material=…, geom=…)` now fails loud at construction when a **finite-strain material** (`LogStrain` / `LadrunoJ2Finite` / `InitDefGrad`) is paired with `geom != "finite"`. Those materials are driven by `setTrialF(F)`; under `linear`/`corot` the element never calls the F-interface, so it would silently integrate **zero stress** — the fork rejects this at run, and apeGmsh now catches it earlier with a clear message pointing to `geom="finite"`. Implemented via an `is_finite_strain` class marker on `NDMaterial` (mirroring the fork's `FiniteStrainNDMaterial` base, and the existing `is_rate_dependent` marker pattern), set `True` on the three finite materials. The converse direction (a non-finite material under `geom="finite"`) is left to the fork's run-time `dynamic_cast` check, since apeGmsh only marks the finite materials it models.

### ADDED — Ladruno-fork live Monitor recorder (`ops.recorder.Monitor` + `read_monitor` / `tail_monitor`)

The Ladruno fork's **Monitor** recorder is now emittable and readable. Unlike the canonical `.ladruno` recorder, the Monitor is a *lightweight live-telemetry sidecar*: it streams a few selected nodal scalars to a small SWMR-HDF5 file (`FORMAT="ladruno-monitor"` — `COLUMNS` / `STEP` / `TIME` / `FRAMES`) that a viewer process can **tail while the analysis is still running**; the same file is a valid at-rest result once the run ends. Fork-only — the `recorder Monitor …` line emits on any build, the fork is needed only to *run*.

- **Emit** — `ops.recorder.Monitor(sink=, nodes=|pg=, dofs=, resp="disp", every=, hz=)`. Channels are nodes × dofs, labelled `node<N>.<resp>.dof<D>` in node-major order; `resp ∈ disp|vel|accel|reaction`; `every=K` (step decimation) and `hz=H` (wall-clock throttle) bound the stream. `pg=` resolves to node tags against the FEM snapshot at emit (mirrors the `Node` recorder).
- **Read** — **not** a `Results` object (the sink carries no FEM), a thin time-history instead: `apeGmsh.results.read_monitor(path) → MonitorData` (`.columns` / `.step` / `.time` / `.frames`, `.channel(label)`, `.to_dataframe(index="time"|"step")`). `apeGmsh.results.tail_monitor(path, timeout=…)` *follows* a still-growing sink in SWMR mode, yielding `(step, time, frame_row)` per frame.
- Verified against a real fork-built `monitor.h5` fixture (build `605affeb`) + a live parity test (sink channel == `ops.nodeDisp` to 1e-9) + a thread-based concurrent-tail test.

### ADDED — Ladruno-fork material wrappers (`LogStrain` / `InitDefGrad` / `StagedStrain` / `LadrunoRebarBuckling`)

apeGmsh can now emit the Ladruno fork's constitutive **wrappers** — materials that hold an inner material and modify its input/output — as typed primitives (mirroring the existing `PlaneStrain` / `InitialStress` wrapper pattern: the inner is held by reference, its tag resolved at emit, and the bridge emits it **before** the wrapper):

- **`ops.nDMaterial.LogStrain(inner=)`** (`ND_TAG` 33010) — the Hencky finite-strain lift: lifts an isotropic small-strain 3-D law to a `FiniteStrainNDMaterial` for `LadrunoBrick … -geom finite`. Pair with an isotropic inner (`LadrunoJ2(-kin 0)`, `ElasticIsotropic`); for combined hardening at finite strain use `LadrunoJ2Finite`.
- **`ops.nDMaterial.InitDefGrad(inner=, no_init_f=, F0=)`** (`ND_TAG` 33013, also `StagedDefGrad`) — finite staged stress-free birth: a continuum element born neutral at the deformed geometry in a staged build. Inner must be a finite-strain material; `F0` is an optional **9 row-major** birth gradient.
- **`ops.nDMaterial.StagedStrain(inner=, no_init=, eps0=)`** (`ND_TAG` 33014) — the small-strain analog (2-D or 3-D everyday staged build); `eps0` is an optional **6-component Voigt** birth strain (required to be 6 — the parser reads it greedily and silently discards a mismatched length).
- **`ops.uniaxialMaterial.LadrunoRebarBuckling(material=, lsr=, model=, …)`** (`MAT_TAG` 33001) — reinforcing-bar buckling overlay around any tension-compression uniaxial (Dhakal-Maekawa `dm` / Gomes-Appleton `ga`); `lsr=0` is the identity gate.

Grammars verified against the shipped fork parsers. Construction guards mirror the parser hard-rejects: `InitDefGrad` `F0` exactly 9 (row-major), `StagedStrain` `eps0` exactly 6, and `LadrunoRebarBuckling`'s post-parse rejects (`lsr>0 ⇒ E>0`; `lsr>0 & model=dm ⇒ fy>0`; `reduction ∈ [0,1]`). **Fork-gated at run, not emit.** Wrappers nest (e.g. `InitDefGrad(LogStrain(LadrunoJ2))`). Second of three slices (after the J2 materials; the `LadrunoBrick -geom/-formulation` element transformations landed separately).

### ADDED — Ladruno-fork J2 plasticity materials (`LadrunoJ2` / `LadrunoUniaxialJ2` / `LadrunoJ2Finite`)

apeGmsh can now emit the Ladruno fork's combined-hardening (Voce + Chaboche) von Mises plasticity family as first-class typed primitives — the OpenSees analogue of Abaqus `*PLASTIC, COMBINED`:

- **`ops.nDMaterial.LadrunoJ2(K=, G=, sig0=, Qinf=, b=, Hiso=, backstresses=[(C,γ),…], rho=, lch_ref=, damage=(r,s,pD,Dc), implex=)`** — the flagship 3-D continuum law (`ND_TAG` 33011); one class serves all five dimensional views.
- **`ops.uniaxialMaterial.LadrunoUniaxialJ2(E=, sig0=, Qinf=, b=, Hiso=, backstresses=, damage=, implex=)`** — the 1-D twin (`MAT_TAG` 33000) for fiber sections / trusses / zeroLength; true multi-backstress ratcheting Menegotto–Pinto can't do. No `-rho` / `-autoRegularization` (the parser rejects them on the uniaxial).
- **`ops.nDMaterial.LadrunoJ2Finite(K=, G=, sig0=, Qinf=, b=, Hiso=, backstresses=, rho=, implex=)`** — finite-strain-native combined J2 (`ND_TAG` 33012, a `FiniteStrainNDMaterial`) for combined hardening **with** large rotation; the consumer is `LadrunoBrick … -geom finite`. No `-damage` / `-autoRegularization` here.

The shared `-iso voce sig0 Qinf b Hiso` / `-kin N C₁ γ₁ …` / `-damage lemaitre r s pD Dc` / `-implex` grammar is verified line-by-line against the shipped fork parsers (`OPS_LadrunoJ2` / `OPS_LadrunoJ2Finite` / `OPS_LadrunoUniaxialJ2`) and centralized in one helper (`material/_ladruno_j2.py`) so the three classes never drift — mirroring the fork's single-kernel design. Construction guards mirror the parser: `sig0 > 0`, ≤ 8 backstress pairs (the fork `MAXBACK`), `C > 0` / `γ ≥ 0`, and the Lemaitre `r > 0` / `0 < Dc ≤ 1` hard guard. **Fork-gated at run, not emit:** the deck line is produced on any build; the material errors at `ops.run()` on stock `openseespy`. Materials round-trip through `model.h5` via the generic `MaterialRecord` (no per-type persistence). First of three slices implementing the fork's materials / wrappers / element-transformations catalog.

### ADDED — Damping definition on the apeSees bridge (`ops.damping` / `s.damping`, ADR 0053)

The bridge had **no way to set damping coefficients** — the per-element `do_rayleigh` flags were inert (nothing to opt into), and there was no `rayleigh` command, region `-rayleigh`, `modalDamping`, or `damping` factory. ADR 0053 adds one domain-level namespace `ops.damping` (a sibling of `fix` / `mass` / `region`, not part of the analysis chain) that owns four of OpenSees' damping channels; material dashpots stay in `ops.uniaxialMaterial.*` and numerical damping in `ops.integrator.*`. Every member is a declaration resolved at emit — no `assign`, no user-held tag.

- **Rayleigh** — `ops.damping.rayleigh(...)` takes either the four raw coefficients or a two-target ratio fit (`ratio=`, `f_i=`, `f_j=` in Hz; β placed by `stiffness=`, default `"initial"`/βK0, the nonlinear-safe choice). `on=` is optional: absent → global `rayleigh`; a physical group (or list) → `region $tag -ele … -rayleigh …`. Because OpenSees overwrites element Rayleigh per element, globals emit before regions ("region refines global") and a `RayleighOverwriteWarning` fires on overlap.
- **Modal** — `ops.damping.modal(ratios, *, modes)` bundles its own `eigen` then `modalDamping` (scalar → uniform; sequence → per-mode). Domain-wide (no `on=`). There is intentionally **no `modal_q`** — `modalDampingQ` is a verified upstream anti-damping bug.
- **Tagged objects** — `ops.damping.uniform` / `sec_stif` / `urd` / `urd_beta` (the four registered OpenSees `damping` types; URD/URDbeta take N≥2 ascending `(freq, value)` points). `uniform`'s `ratio=` is the physical ζ (OpenSees doubles internally). All four take `activate_time` / `deactivate_time` (the "no damping during the gravity stage" lever) and `factor=` (an `ops.timeSeries.*` object → `-factor`). `on=` attaches them via `region -damp`; alternatively omit `on=` and hand the returned handle to a `-damp`-capable element's `damp=` kwarg (`elasticBeamColumn` / `forceBeamColumn` / `dispBeamColumn` / `stdBrick` / `FourNodeQuad` / the Shell family / `ZeroLength`). An object that attaches to nothing fails loud at `build()`.
- **Persistence** — damping objects persist to `/opensees/dampings/` (bridge schema 2.14.0 → **2.15.0**) and fold into `model_hash`; they replay on `OpenSeesModel.from_h5 → build`, with element-flag attachments round-tripping (region attaches share the archival-only `/opensees/regions` limitation).
- **Staged** — the same verbs live on `s.damping.*` inside an `ops.stage(...)` block and resolve inside that stage (after `domainChange`). `s.damping.modal` is deferred (per-stage eigen / `wipeAnalysis` interaction).

Grammar verified line-by-line against the upstream OpenSees parser source. Shipped across D1 (#527), D2 (#528), D3a (#529), D4 (#530), D3b-1 (#531), D3b-2 (#532), D3b-3 (#534), D5 (#536).

### ADDED — `ops.element.BezierTri6` typed primitive (Ladruno-fork Bézier triangle)

apeGmsh can now emit the Ladruno fork's `BezierTri6` — a 6-node quadratic Bézier (Bernstein) plane element (Kadapa 2018) — as a first-class typed primitive: `ops.element.BezierTri6(pg=…, thickness=…, material=…, plane_type=…, bbar=…, consistent_mass=…, pressure=…, rho=…, body_force=…)`. On a straight-sided mesh the Gmsh `tri6` (etype 9) nodes coincide with the element's control points, so connectivity is used verbatim (identity reorder, same basis seam as `SixNodeTri`).

Unlike `SixNodeTri`'s positional `<pressure rho b1 b2>` tail, the fork grammar is **flag-prefixed**: `element BezierTri6 $tag $n1..$n6 $thick $type $matTag [-bbar] [-cMass] [-pressure $p] [-rho $r] [-bodyForce $b1 $b2]`. Each option is independently optional. The `plane_type` validator accepts **only** the 2-value canonical pair (`PlaneStrain`/`PlaneStress`) — not the `*2D` spellings `SixNodeTri` tolerates — matching the fork factory. Mirroring the fork's D5 guard, requesting `bbar=True` under `PlaneStress` emits a `BezierBBarPlaneStressWarning` and drops the `-bbar` flag (the run proceeds).

**Fork-gated at run, not emit:** the `element BezierTri6 …` line is produced on **any** build; the fork is required only to *run* the deck (a clear fork-build error and direct-drive fallback land in a later slice). Result reads are wired in the `_response_catalog` (`ELE_TAG_BezierTri6 = 33000`, live from `classTags.h`; the dead pre-fork 272/273 are explicitly rejected) under both the real `Triangle_GL_2` rule and `Custom` — the recorder serves it via the element's self-declared `basisInfo`. The catalog row carries layout/`n_gp` metadata only; canonical per-GP coords come from the file's `QUADRATURE/GP_PARAM` (the fork integrates the 3 GPs in a permuted index order vs `SixNodeTri`).

### ADDED — `ops.element.BezierTet10` typed primitive (Ladruno-fork Bézier tetrahedron)

The 3D sibling of `BezierTri6` — a 10-node quadratic Bézier (Bernstein) tetrahedron: `ops.element.BezierTet10(pg=…, material=…, bbar=…, consistent_mass=…, rho=…, body_force=…, pressure=…)`, emitting `element BezierTet10 $tag $n1..$n10 $matTag [-bbar] [-cMass] [-rho $r] [-bodyForce $b1 $b2 $b3] [-pressure $p]` (flag-prefixed tail). On a straight-sided mesh the Gmsh `tet10` (etype 11) nodes coincide with the control points, so connectivity is verbatim — the 10 nodes are 4 corners then 6 mid-edge nodes in `TenNodeTetrahedron` order `(1-2, 2-3, 1-3, 1-4, 3-4, 2-4)`. Unlike `BezierTri6` there is no plane-stress degeneracy, so **B-bar is always valid** (no warn-and-drop guard).

The **O11 node-order identity is locked by a machine-precision test** (`tests/opensees/integration/test_bezier_tet10_o11.py`): a straight-sided box meshed to `tet10` has every mid-edge node within `2.2e-16` (relative) of its corner-pair midpoint — confirming the Gmsh tet10 order is byte-identical to the element's control-point order (the `node_reorder={11: identity}` decision; a wrong order would silently yield a wrong stiffness). Reads wired in `_response_catalog` (`ELE_TAG_BezierTet10 = 33001`) under `Tet_GL_2` + `Custom`; the Tet10 GP index order is clean (matches `_TET_GL_2_COORDS` — no permutation, unlike the Tri6 sibling).

### ADDED — Bézier elements: clear fork-build error when run on a stock build

Running a deck that contains a `BezierTri6` / `BezierTet10` **in-process** on a non-fork (stock) openseespy build now raises a clear *"element BezierTri6 requires the Ladruno fork build … only running the deck in-process needs the fork … use the direct-drive fallback"* error, instead of a cryptic openseespy error or a silent no-op that fails much later. `LiveOpsEmitter.element` verifies the fork-only element actually built (catching both the raise and the silent-warn-and-drop stock behaviors via `getEleTags`), caches the verdict after the first success (O(1) overhead), and skips the probe inside non-zero partition blocks. **Deck emission (`ops.tcl` / `ops.py`) is unaffected** — only the in-process run is gated; direct-drive remains the supported fallback on stock builds.

### IMPROVED — Bézier Gauss-point world coordinates reconstructed via the Bernstein basis

`results.elements.gauss.get(...).global_coords(fem)` now reconstructs the **world** coordinate of a Bézier element's Gauss points as `x = B(ξ)·X` over all control points — routing `BezierTri6` / `BezierTet10` through the neutral `apeGmsh._basis` Bernstein evaluator, with ξ taken from the file's `QUADRATURE/GP_PARAM` (never a catalog GP order — respecting the Tri6 GP-index permutation). Previously these higher-order families had no entry in the linear Gmsh shape-function catalog and fell through to a centroid+bbox approximation (visibly wrong — off by ~0.5 on the test element). Every other element type keeps its existing linear-catalog / bbox path unchanged. Verified fork-free against committed fixtures: the Tri6 reconstruction matches an independent affine-barycentric corner map to `2.2e-16`, and the Tet10 reconstruction matches the element's own `GLOBAL_GP_COORDS` to `2.2e-16`. New fixture `tests/fixtures/ladruno/bezier_tet10.ladruno`.

### ADDED — `ops.profiler.*` (Ladruno-fork stack profiler) + `analyze(profile=…)`

apeGmsh can now emit the Ladruno fork's stack-profiler control command, which brackets the analyze loop and writes one `profile.h5`. It is a *control* command, not a model primitive or recorder — no class tag, no `_response_catalog` entry, no reader (read `profile.h5` with the fork's out-of-tree `Ladruno_tools/profiler_viewer`).

The new `ops.profiler` namespace exposes the five shipped fork verbs 1:1 — `start(deep=, memory=, per_step=)` / `stop()` / `reset()` / `report(file, run=)` / `memory()` (mapping to `profiler start [-deep] [-memory] [-perStep]` / `stop` / `reset` / `report <file> [-run <id>]` / `memory`). There is no `config` verb / `-warmupSteps` — the design doc showed them but `OPS_profiler()` never wired them.

**Deck emit (Tcl / Py):** record the verbs before `ops.tcl(...)` / `ops.py(...)`; the bridge brackets the appended `analyze` line, with side chosen by verb (`start` / `reset` before, `stop` / `report` / `memory` after). **Live:** `ops.analyze(steps=…, profile='profile.h5', profile_run=…, profile_deep=…, profile_memory=…, profile_per_step=…)` wraps the in-process run.

Fork-gated at run time: emitting deck text works on any build; the live emitter re-raises a clear *"requires the Ladruno fork build"* error when stock openseespy lacks the `profiler` command. `ops.tcl(run=True)` is the recommended profiled path. One new `Emitter.profiler(*args)` Protocol method (Tcl/Py emit the line, live forwards + gates, h5 no-ops, recording captures).

**Reading the output** — `apeGmsh.profiler.open(path)` / `apeGmsh.profiler.show_web(path)` are a thin, fork-free-at-import bridge to the fork's out-of-tree `Ladruno_tools/profiler_viewer`: `open` re-exports its `ProfilerResults` loader (`manifest` / `rollup` / `series` / `diff`; `series` is the per-step "monitor"), `show_web` launches the one-process React UI. apeGmsh re-exports, never re-implements. The viewer dir must be importable (`viewer_dir=` kwarg, `LADRUNO_PROFILER_VIEWER` env var, or `sys.path`); otherwise a clear install-hint error fires.

### ADDED — Ladruno `.ladruno` recorder: emit + canonical read (`ops.recorder.Ladruno` / `Results.from_ladruno`)

apeGmsh now emits and reads the Ladruno fork's **canonical** HDF5 recorder, the self-describing sibling of STKO `.mpco`.

**Emit** — `ops.recorder.Ladruno(file, nodes=…, elements=…, dt=… | nsteps=…)` produces a `recorder ladruno …` line on **any** build (the fork is required only to *run* the deck). Whole-model value channels via `-N`/`-E`/`-T`.

**Read** — `Results.from_ladruno(path, *, fem=None, merge_partitions=True, model_h5=None)`. Unlike every other constructor, `model_h5=` is **optional**: a `.ladruno` is self-describing (geometry, regions, beam local axes all in-file), so the broker is built from the file itself (schema Principle 0). The reader keys on `INFO/GENERATOR="Ladruno"` + a windowed `FORMAT_VERSION` (ADR 0023 spirit) and needs only `h5py` — no fork at read time. Multi-partition runs (`<stem>.part-N.ladruno`) auto-discover siblings and merge (node-union + element-concat), like `from_mpco`.

**Result reads** go through the usual `results.*` API:
- `results.nodes.get(component="displacement_x")` — chunked nodal channels.
- `results.elements.gauss.get(component="stress_xx")` — continuum stress/strain, **neutral** vocabulary (accepts both the `sigma11` and `sigma_xx`/`eps_xx`/`gamma_xy` token forms different element classes emit). Gauss-point natural coords come from the file's `QUADRATURE/GP_PARAM`.
- `results.elements.line_stations.get(component="axial_force")` — beam internal-force diagrams, **neutral** (`axial_force`/`shear_y`/…); `localForce` end forces get the sign-continuity flip, `basicForce` is one station at ξ=0.
- `results.elements.get(component="localForce")` — **token-driven**: the component is the file's `ON_ELEMENTS/<token>` key (`basicForce`/`localForce`/`force`/`globalForce`), returning the raw `(T, E, NUM_COLUMNS)` block in the file's column order. (The one place the Ladruno element API differs from MPCO's neutral `nodal_resisting_force_*` — Ladruno is file-driven; the neutral beam view is `line_stations`.)

**Beam orientation** — a `.ladruno` writes `MODEL/LOCAL_AXES` (per-class quaternion frames) that `.mpco` omits. `results.elements.local_axes(...)` surfaces a `LocalAxes` (scalar-first quaternions + `.matrices`/`.x_axis`/`.y_axis`/`.z_axis`, axes are the matrix **rows**), and `results.plot.line_force(...)` now orients diagrams from the recorder frame (true cross-section roll) instead of guessing from node geometry — retiring the `.mpco` "no beam vecxz" workaround for wired classes.

**Energy balance** — `results.energy(region=None)` → a pandas DataFrame `KE/IE/DW/ULW/RES/ERR` indexed by time (recorder `-G energy`; whole-domain or per-region), subsuming the standalone EnergyBalance recorder.

**Self-describing geometry** — higher-order / Bézier element groups carry a `BASIS` descriptor (family/topology/order + `GP_PARAM`) instead of a per-class shape-function table; GP world coords are reconstructed via the new neutral `apeGmsh._basis` evaluator (`B(ξ; family, order, topology)` — delegates Lagrange to the existing shape-function library, adds the Bézier/Bernstein bases, validated against the reference elements). Shared with the upcoming Bézier read path.

### ADDED — `g.model.geometry.add_arch(start, apex, end, *, label=)`

A circular arch built as **two tangent arcs that share the apex as a topological vertex**, so the crown survives meshing as a conforming node.

`add_arc(..., through_point=True)` fits a *single* arc through `start`/`apex`/`end` and leaves the apex as a floating construction point that the mesher discards — a physical group placed on the apex then resolves to a node that never lands in the mesh (a gmsh quirk). `add_arch` instead computes the circle centre, emits two `addCircleArc` halves (`start→apex`, `apex→end`) of the *same* circle, then removes the construction centre so it leaves no stray node at the centre of curvature. Both halves are tangent-continuous at the apex (no kink), and the apex becomes a real vertex — guaranteeing a mesh node exactly at the crown for a crown load, monitoring point, or midspan physical group.

Returns `list[Tag]` = `[start→apex, apex→end]` (hand straight to `g.physical.add_curve`); `label=` is applied to **both** halves. Fails loud (`ValueError`) on collinear/coincident points. `add_arc` is unchanged.

The general declarative `g.model.geometry.embed_node()` (for interior points of surfaces/volumes, where the split-into-parts trick has no clean equivalent) was designed alongside this but deferred.

### ADDED — CAD-import health diagnostics + scale-aware healing

`g.model.io.diagnose(*, warn=False) -> ImportHealth` is a new **non-mutating** health check for the current OCC geometry: per-dimension entity counts, sliver tallies (edges/faces below `1e-4 · bbox_diagonal`), the bbox diagonal, and a suggested `heal=` tolerance. It never heals, dedupes, or renumbers — the look-before-you-leap counterpart to `heal_shapes` (which does mutate). `ImportHealth.is_suspect` keys on slivers only, so a surface-only import (shell models) does not false-positive.

A new typed `WarnGeomImportHealth` advisory is auto-emitted by `load_step` / `load_iges` / `g.parts.import_step` on a **raw** (un-healed) import when slivers are present — it names the counts and suggests `heal='auto'`. The import is never healed on the user's behalf (healing renumbers entities, so it stays opt-in).

`g.parts.import_step(...)` gains `heal=` / `dedupe=` kwargs (same semantics as `g.model.io.load_step`), closing the gap where the assembly path — the workflow most likely to ingest real external CAD — could not heal at all.

### CHANGED — `heal=True` / `heal="auto"` on import is now scale-aware

`load_step` / `load_iges` (and the new `parts.import_step`) `heal=True` now derives a **scale-aware** tolerance (`≈ 1e-6 · bbox_diagonal`) instead of the legacy absolute `1e-8`. `"auto"` is an explicit alias; a float still overrides. The old `1e-8` default was effectively a no-op heal on any real-world (mm/m-scale) model — `heal=True` now actually heals. `heal=False` (the import default) is unchanged.

### CHANGED — raw user physical groups now protect entities from the orphan sweep

`apeGmsh.core._geometry_topology._user_intentional` gains a third channel: an entity that participates in ANY physical group (including raw user PGs created via `gmsh.model.addPhysicalGroup` directly) is treated as user-intentional and survives `sweep_dangling` / `find_orphans` / `validate_pre_mesh(strict=True)`.

Closes PR #378 reviewer follow-up #2. The new channel lowers the false-positive rate of the open-world (`strict=True`) check substantially: raw-gmsh frame workflows that tag their lines with PG names like `"Columns"` / `"Beams"` are now protected, even though they bypass apeGmsh's `_metadata` and `g.labels` channels entirely.

Order of channels in `_user_intentional`:

1. `model._metadata` (closed-world, populated by apeGmsh `add_*`)
2. apeGmsh label PGs via `Labels.labels_for_entity` (tier-1 `_label:*`-prefixed PGs only)
3. **Raw user PGs via `gmsh.model.getPhysicalGroupsForEntity`** (new — catches any PG, no prefix filter)

`Mesh.generate`'s default `strict=False` auto-validation is unaffected (it only inspects `_metadata`).  Users explicitly calling `validate_pre_mesh(strict=True)` or `remove_orphans()` get the broader protection.

### ADDED — `g.model.geometry.find_stale_metadata()` + `validate_pre_mesh(strict=...)` split

- **`g.model.geometry.find_stale_metadata() -> list[(dim, tag)]`** — closed-world inspection. Walks only the keys apeGmsh primitives recorded in `model._metadata` and returns those whose tag is no longer in OCC. Cannot false-positive on raw `gmsh.model.geo.*` / `gmsh.model.occ.*` workflows because those workflows don't populate `_metadata` in the first place.
- **`g.model.geometry.validate_pre_mesh(*, strict=False)`** gains the `strict` kwarg:
  - `strict=False` (default; auto-fired by `Mesh.generate`) — runs `find_stale_metadata()` only. Closed-world. Catches the actual leak class the orphan-sweep PR was chasing: an apeGmsh boolean / cut / fragment op consumed an entity without cleaning its `_metadata` key.
  - `strict=True` (opt-in) — runs `find_orphans()` and raises on any orphan dim≤2 entity. Open-world. Users opt in when they know their build script stays inside the apeGmsh facade (`_metadata` + `g.labels` channels).
- **`Mesh.generate` now auto-invokes `g.model.geometry.validate_pre_mesh()`** alongside the loads / constraints / masses validators. Because the default is `strict=False`, raw-gmsh workflows continue to mesh cleanly — only stale apeGmsh-managed metadata trips the auto-check.

The split is the follow-up the PR #378 review backlog asked for. The earlier attempt to auto-wire the full open-world check broke 63 tests across `test_partition_*`, `test_loads_physical_outward`, `test_mesh_editing_crack`, `test_embedded_decomposition`, `test_constraint_emission`, `test_partition_pipeline_e2e` because those workflows build geometry via raw `gmsh.model.geo.addPoint/addLine/addPhysicalGroup`. Splitting the validator into closed-world (auto-fires) and open-world (opt-in) gets the auto-validation back without the false positives.

### FIXED — coincident-face orphan-geometry leak in slice / cut / fragment

- **`g.model.geometry.slice(...)` / `cut_by_surface(...)` / `cut_by_plane(...)` and `g.model.boolean.fragment(...)`** now share a single topology-driven cleanup pass (`apeGmsh.core._geometry_topology.sweep_dangling`), eliminating the three pre-existing definitions of "orphan" the cleanup paths used. The leak was reliably reproducible when a cutting plane coincided with an existing face of the operand (e.g. slicing a swiss-cheese solid at its cavity-bottom z-coordinate, or fragmenting two abutting solids at their shared face): OCC consumed the tool but left at least one free-floating sub-piece behind. Plus matching dim=1 and dim=0 leaks and stale `model._metadata` entries.

- **The sweep keeps anything that (a) bounds a registered volume at any depth, or (b) is user-intentional — in `model._metadata` (every `add_*` primitive registers there) or carrying a label.** Everything else dim ≤ 2 is removed, then stale metadata keys whose tags no longer exist in OCC are reaped. Mirrors the audit's spec verbatim.

### CHANGED — `boolean.fragment(cleanup_free=...)` default flipped back to `True`

- **`g.model.boolean.fragment(...)` default `cleanup_free` flipped from `False` back to `True`**, but the cleanup is now the topology-driven sweep above, not the previous centroid-in-bbox heuristic. The centroid heuristic over-collected shell-on-solid geometry whose centroid happened to fall outside the volume bbox — the topology sweep preserves any standalone shell the user explicitly created (`add_rectangle`, `add_plane_surface`, etc.) because those entities live in `_metadata`. Pass `cleanup_free=False` only when you need OCC's raw output (no sweep, no stale-metadata reap) for downstream inspection.

- **`_bool_op` registration narrowed.** Previously every dimtag in a boolean result (including all sub-surface byproducts of a 3D fragment) was registered in `_metadata`. The sweep would then mis-protect those byproducts as user-intentional. Now `_bool_op` registers only result dimtags whose dimension matches `default_dim` (typically 3). `cut_by_surface(keep_surface=True)` re-registers cut interfaces as `'cut_interface'` explicitly — unchanged for users.

### ADDED — `g.model.geometry.find_orphans()` / `remove_orphans()` / `validate_pre_mesh()`

- **`find_orphans() -> dict[int, list[int]]`** — inspect the model for orphan geometry without modifying it. Returns the dimtags the post-op sweep would reap.
- **`remove_orphans(*, dry_run=False) -> dict[int, list[int]]`** — manual sweep entry point. Same algorithm slice / cut_by_* / fragment run internally.
- **`validate_pre_mesh()`** — raise `GeometryValidationError` if any orphans exist. Mirrors `MassesComposite.validate_pre_mesh` / `LoadsComposite.validate_pre_mesh` / `ConstraintsComposite.validate_pre_mesh`. **Opt-in**: users call it explicitly. `Mesh.generate` does NOT auto-invoke it because raw `gmsh.model.geo.*` / `gmsh.model.occ.*` workflows bypass the `_metadata` channel and raw user PGs bypass the label channel — both would trigger false positives. Auto-wiring stays on the follow-up backlog until those channels are unified.

### ADDED — coincident-face advisory + one-sided-cut warning

- **`WarnGeomCoincidentFace`** fires from `add_axis_cutting_plane` when the requested plane sits within OCC tolerance of an existing axis-aligned face of an operand. The sweep cleans up the orphan regardless; the warning lets users refactor the offset to avoid the OCC fragility entirely.
- **`WarnGeomOneSidedCut`** fires from `cut_by_plane` when the plane offset sits outside the operand's bounding box (only one side has fragments). Previously a silent log line; now a `UserWarning` subclass so test-time `pytest -W error` catches it.

### REMOVED — dead code on `cut_by_surface`

- **`cut_by_surface(sync=...)` parameter dropped.** The argument was accepted but never honored (the function always synced unconditionally). Internal callers (`cut_by_plane`, `slice`) updated accordingly.
- **Dead `inherited_label` block removed** (computed but never read).
- **`_cleanup_slice_orphans` deleted** — replaced by `sweep_dangling`.

### ADDED — higher-order line broker split (ADR 0037, #349)

- **`g.mesh.editing.split_higher_order_lines(physical_group, *, policy, dim=1)`** — broker-side resolution to the 2nd-order-continuum + frame hard-stop. When a quadratic continuum part (shell, tet10, etc.) propagates Gmsh's global mesh order to every line entity in the model, frame PGs end up with 3-node Line3 elements that OpenSees beam-columns refuse at `_check_two_nodes`. The new verb demotes them in place to 2-node Line2s before the bridge ever sees them.

  Three policies:

  - **`"forbid"`** — raise `RuntimeError` if any Line3 present, naming the PG and count. Use as a build-time invariant lock when a PG must remain 1st-order through meshing.
  - **`"split"`** — for each Line3 `(i, j, mid)`, remove the Line3 from its dim=1 entity via `gmsh.model.mesh.removeElements` and add two Line2 pairs `(i, mid)` + `(mid, j)` to the **same** entity via `gmsh.model.mesh.addElements` with type=1. The mid-side node, formerly a Gmsh side-node carrying no FE DOFs, becomes an endpoint of two Line2 elements and acquires DOFs in the OpenSees domain. PG membership tracks at entity level (no rebinding); no new gmsh nodes are minted.
  - **`"constrain"`** — RESERVED, raises `NotImplementedError`. The kinematically clean answer (mid-node linearly interpolated from i and j) requires an OpenSees primitive that doesn't exist today: `ASDEmbeddedNodeElement` accepts exactly 3 or 4 retained nodes per ADR 0036, so a 2-master Line2 pair can't be expressed. Gated on upstream OpenSees work on the same future track as ADR 0036's HostProjector RFC.

- **Sequencing.** Call **after** `g.mesh.generation.generate(...)`, **before** `g.mesh.queries.get_fem_data(...)` and `g.mesh.partitioning.partition(...)`. Never inside a stage block — the mesh edit is global and must complete before the bridge builds.

- **Concentrated-plasticity trap (documented, not enforced).** `policy="split"` on a PG that hosts a `forceBeamColumn` / `dispBeamColumn` with `HingeRadau` / `HingeRadauTwo` / `HingeMidpoint` / `HingeEndpoint` integration places the calibrated end-region hinges in the wrong locations (each sub-element inherits the parent rule → four hinges per parent at the wrong stations). The docstring and ADR 0037 §Consequences flag this; runtime bridge-side detection deferred until it bites.

- **`_check_two_nodes` sharpened.** The bridge guard at [`opensees/element/beam_column.py`](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/element/beam_column.py) gains explicit 3-node and 4-node branches that name the new verb in the error message. Per ADR 0037 INV-1: this is a *friendlier loud-fail message*, not bridge awareness of higher-order topology — the bridge still refuses to emit a beam with three nodes; it just tells the user where to go.

- **`tests/test_mesh_editing_split_higher_order_lines.py`** locks the new invariants: policy dispatch, `dim` guard, unknown-PG `KeyError`, empty-iterable `ValueError`, `policy="forbid"` raise message + Line2 no-op, `policy="split"` count + PG membership + mid-node preservation + Line2 connectivity assertion (the load-bearing invariant: `(i, mid)` + `(mid, j)` pairs), idempotency on Line2-only PG, iterable input, multi-PG split, mixed-order-per-entity surgery (the persistent record of the one-off gmsh spike), and end-to-end through the bridge's `elasticBeamColumn` fan-out.

- **No schema bump anywhere.** Per ADR 0037 INV-3: bridge surface (`/opensees/*` zones, `element_meta`, `tag_recorder`, transform fan-out) is untouched; the FEMData snapshot is consistent topology after split (no parallel "macro-origin" state).

- **Deferred** (ADR 0037 §Future work): `policy="constrain"` (gated on upstream OpenSees primitive); `dim=2` / Line4 cubic edges (additive once needed); bridge-time concentrated-plasticity enforcement.

### FIXED — RecorderDeclaration element fan-out resolves FEM eids to ops_tags (#348)

- **`ops.recorder.declare(elements=...)` / `gauss=...` / `line_stations=...`** now translates FEM eids through the bridge's `fem_eid_to_ops_tag` map before writing `-ele` arg lists. Previously the RecorderDeclaration emit path (`_emit_recorder_declaration` → `_emit_element_level_record` → `_resolve_element_targets` in [`opensees/_internal/build.py`](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/_internal/build.py)) passed raw FEM eids straight into `-ele`, silently targeting the wrong OpenSees tags whenever any `ops.element.X(pg=...)` spec consumed an allocator slot in `_register` (which is always — `_kind_of` groups specs and fan-out instances under the same `"element"` kind).

- **Consistency restored.** The typed-`Element` recorder path ([`opensees/recorder.py:286-305`](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/recorder.py)) already had this translation; this fix brings `RecorderDeclaration` to parity. Both routes now resolve identically and raise `BridgeError` when a target eid maps to no ops_tag, with the eid in the message.

- **New `_translate_to_ops_tags` helper** centralizes the translation so the same fail-loud message fires from all three element-recorder emit sites (canonical Element, raw Element, integrationPoints pairing for line_stations).

- **Backward compatible.** Legacy direct callers (no bridge, e.g. unit tests of `materialize()`) pass `fem_eid_to_ops_tag=None` and get raw eids — only the through-bridge emit path applies the translation.

- **12 pre-existing tests updated** in `tests/opensees/integration/test_recorder_declaration_emit.py` (they had asserted the buggy passthrough by targeting FEM eid 1 with no Element primitive registered). Now register an `elasticBeamColumn` via a new `_register_dummy_beam` helper and assert the translated ops_tag (FEM eid 1 → ops_tag 2, because the spec consumes element-kind slot 1). **2 new regression tests**: `test_elements_pg_translates_fem_eids_to_ops_tags` (positive) and `test_elements_missing_eid_raises_bridge_error` (loud-fail).

### ADDED — Phase SSI-2.E between-stage Domain mutators

Five new `_StageBuilder` verbs lift the append-only restriction on
stage-bound BCs declared in Phase SSI-2.D.  Closes the
`_DEFERRED.md` §"`remove sp` / mass-zero-out across stages" item.

- **`s.remove_sp(*, pg=None, nodes=None, dofs)`** — releases prior-
  tier SP constraints. Emits `remove sp $node $dof` per resolved
  `(node, dof)` pair, INSIDE the stage block and BEFORE any new
  stage-bound `fix` / `mass` / `region` / MP-constraint lines. The
  emit position locks the canonical atomic-replace pattern:
  release prior + re-fix in the same stage works by construction.

- **`s.remove_element(*, pg=None, elements=None)`** — drops elements
  from the Domain mid-analysis. Emits `remove element $tag`. The
  `elements=` parameter takes FEM eids (matching the
  `recorder.Element` convention); the bridge translates to
  OpenSees ops tags via `fem_eid_to_ops_tag` at emit time so the
  emitted line carries the same tag the rest of the deck uses.

- **`s.mass(..., overwrite=True)`** — opts the record out of
  validator V2's cross-tier duplicate-mass refusal. The emitted
  `mass` line is byte-identical with or without the flag (OpenSees
  `Domain::setMass` silently overwrites); the flag is purely a
  build-time validator-bypass marker acknowledging the
  intentional overwrite. `apeSees.mass(...)` accepts the same
  kwarg for symmetry.

- **`s.set_time(t)` / `s.set_creep(on)`** — emit `setTime $t` /
  `setCreep 0|1` right after `stage_open`. Useful for stages whose
  pseudo-time should begin at a non-zero value (overriding the
  prior `stage_close`'s `loadConst -time 0.0` reset) or to toggle
  creep for time-dependent concrete materials.

- **`s.reset()`** — emits the bare OpenSees `reset` command
  between the stage's recorder declarations and its analyze loop.
  Rarely needed; kept for parity with the OpenSees surface.

#### Validators

- **V5** — `s.remove_sp` target must reference an SP declared in
  the global `apeSees.fix` pool OR in a strictly-earlier stage's
  `s.fix` pool, AND not already removed by an earlier stage.
  Same-stage `s.fix` does NOT count (fix emits AFTER remove_sp in
  the stage block).

- **V6** — `s.remove_element` target must reference an element
  emitted globally OR activated by this stage / a strictly-earlier
  stage AND not already removed. PG-typo case (`pg=` resolves to
  nothing on the FEM snapshot) surfaces a dedicated offender line.

- **V2 widened** — the existing duplicate-fix-mass validator now
  subtracts `s.remove_sp` targets from the fix alive set on
  encountering each stage, so the atomic-replace pattern
  (release prior + re-fix same DOF in same stage) passes both V5
  and V2.

#### Protocol

`Emitter` protocol gains five new methods (`set_time`,
`set_creep`, `reset`, `remove_sp`, `remove_element`). Per-emitter
implementations:

- **Tcl / Py**: emit the corresponding OpenSees command (`setTime
  $t`, `setCreep 1|0`, `reset`, `remove sp $node $dof`, `remove
  element $tag`).
- **Live**: forwards to `self._ops.{setTime, setCreep, reset,
  remove("sp", ...), remove("element", ...)}`. Unreachable on the
  staged path (which raises at `stage_open` in live), but works
  for non-staged custom workflows.
- **H5**: no-op (mirrors the existing `stage_open` / `stage_close`
  no-ops; the `apeSees.h5(path)` guard on staged models still
  applies).
- **Recording**: tuple capture for tests.

#### Build pipeline

`_emit_stages_flat` and `_emit_stages_partitioned` widen the
unified `domain_change` gate to fire on removals (a stage that
ONLY does `s.remove_sp` still emits `domain_change` so the next
analysis chain bind sees a fresh DOF map). Element tag allocation
moves earlier in the staged path so V6 can resolve
`elements=[fem_eid]` user inputs against the live
`fem_eid_to_ops_tag` map.

#### Tests

`tests/opensees/unit/test_stage_ssi_2e_mutators.py` — 34 tests
covering builder positive/negative, dataclass field shapes,
single-partition emit position, Tcl emit text, V5 / V6 ownership-
tier rules, V2 relaxation with `overwrite=True`, and the
end-to-end atomic-replace pattern.

### ADDED — embedded-host decomposition for non-simplex / higher-order hosts (ADR 0036)

- **`g.constraints.embedded(...)` now accepts non-simplex and
  higher-order hosts** without remeshing.  Previously the collector
  raised on any host element type other than tri3 / tet4, forcing
  users to either remesh their hex/quad concrete blocks as tets or
  hand-build the `ASDEmbeddedNodeElement` directly in Tcl/Python.
  The decomposition runs apeGmsh-side only — no OpenSees changes,
  no H5 schema bump — by virtualising the host into linear sub-tris
  / sub-tets that the C++ element already accepts (it stores
  retained tags without validating their source; see
  [`ASDEmbeddedNodeElement.cpp:293-322`](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp)).

  Supported host etypes (2D): tri3 (CST), tri6 (LST), quad4,
  quad8, quad9.  Supported (3D): tet4, tet10, hex8 (6 Kuhn tets),
  hex20, prism6 (3 tets), prism15, pyramid5 (2 tets), pyramid13.
  Prism and pyramid use Kuhn-style decompositions verified
  positive-volume against
  `gmsh.model.mesh.getElementProperties(etype)`.

- **Reserved `host_coupling="linear"` keyword on `EmbeddedDef` and
  `embedded(...)`** pins the coupling kinematics so a future
  `"trilinear"` / `"biquadratic"` option (requiring a new OpenSees
  element class that supports N-node retained sets) can be added
  without breaking existing models.  Pre-existing models keep
  producing identical results because `"linear"` is the default.

- **One `UserWarning` per (etype, entity)** when a midside-bearing
  host (tri6, tet10, quad8, quad9, hex20, prism15, pyramid13) hits
  the decomposition path.  The warning surfaces the linear-coupling
  consequence so a user who chose LST for bending curvature knows
  the embed only sees the corner-to-corner linear stretch.  Set
  `host_coupling="linear"` explicitly on the call to acknowledge.

- **Mixed-dim host fail-loud.**  A host PG that combines 2D and 3D
  entities raises a clear error at collection time — the linear
  coupling cannot pick between them deterministically (kNN
  centroid search would dispatch based on opaque proximity).
  Split the host PG into two `embedded(...)` calls instead.

- **`tests/test_embedded_decomposition.py`** locks the new
  invariants: Kuhn / prism / pyramid orientation tables match
  Gmsh's canonical reference coords; each decomposition partitions
  its reference cell with no gaps or overlaps; the quad4 split
  covers any convex quad; end-to-end resolver run on a hex8 mesh
  produces an `InterpolationRecord` with 4 master nodes drawn
  from the host's 8 corners; sliver-tet hosts either resolve with
  bounded weights or fail-loud (never silent nonsense); the
  higher-order warning fires exactly once per (etype, entity); the
  `host_coupling` keyword reservation rejects unimplemented
  values.

- **Renames `_collect_host_elems` → `_collect_host_subelements`.**
  The collector now returns virtual sub-element rows (not real
  gmsh elements), so the name change makes the new contract
  explicit.  Private API only — no downstream user surface
  affected.  ADR 0027 §"Mixed-host silent drop" historical
  reference updated; `docs/api-flows/*` regenerated.

### ADDED — topology safety nets and coincident-node diagnostic

- **`fem.inspect.find_coincident_node_pairs(tol=, pg=)`** — opt-in
  diagnostic that surfaces every pair of distinct nodes sharing an XYZ
  within tolerance and lists which elements / constraints (if any)
  bridge them. An empty refs list is the smoking gun for an unbridged
  duplicate — the classic OCC arc-line-junction failure mode where a
  wire built as `add_ellipse(angle1, angle2) + lines` produces two
  nodes at every corner with no moment continuity. Reuses the
  resolver's `_SpatialIndex` (SciPy `cKDTree` with NumPy fallback);
  no new dependencies.

- **Tuple-uniqueness check at the OpenSees bridge boundary** — every
  element's connectivity tuple is now validated by
  `emit_element_spec` before emission. Repeated node tags in a single
  element fail loud with a `BridgeError` carrying `(fem_eid, type,
  tuple)`. The check is tag-level, so `zeroLength` (two *distinct*
  tags at coincident XYZ) still passes; the only behaviour change is
  that previously-silent resolver bugs now stop at the bridge instead
  of confusing OpenSees downstream.

### CHANGED — `mesh.editing.remove_duplicate_nodes()` always prints

- Dropped the `verbose=` parameter. Node removal is now unconditional
  on stdout — both branches (`merged N node(s)` / `no duplicates
  found`) always announce themselves. Deleting nodes from a meshed
  model is destructive; the visibility floor is intentional so an
  unexpected dedup never hides in a long pipeline log. Callers passing
  `verbose=False` will now `TypeError` — drop the kwarg.

### DOCS — `make_conformal()` canonical fix for arc-line junctions

- **`g.model.queries.make_conformal()` docstring** now calls out two
  flavours of disjoint topology it addresses: IGES/STEP imports AND
  partial-arc-built wires (`add_arc` / `add_circle(angle1,angle2)` /
  `add_ellipse(angle1,angle2)` joined to lines). Adds a **Warnings**
  block on ordering: fragment renumbers entities, so any pre-built
  `Part` / `Assembly` holds stale `Instance.entities` dicts — call
  `make_conformal()` before constructing Parts, or rebuild Parts
  afterward.

- **Wire-builder docstrings** (`add_arc`, `add_circle`, `add_ellipse`)
  now carry a See-Also pointing at `make_conformal(dims=[1])` so the
  next person who hits the cimbra pattern finds the fix from the
  symbol they're already on. `add_ellipse` also references the new
  `find_coincident_node_pairs` diagnostic for post-mesh verification.

### ADDED — stage-bound constraints (`s.embedded` et al.) + `s.initial_stress` PUSH path (Phase SSI-2.D extension)

- **Nine new builder methods on `_StageBuilder` claim resolved constraint
  records by name** so the constraint emits inside the owning stage's
  block rather than in the global pre-stage MP-constraint pass:
  `s.embedded(name=...)`, `s.equal_dof(name=...)`, `s.rigid_link(name=...)`,
  `s.rigid_diaphragm(name=...)`, `s.kinematic_coupling(name=...)`,
  `s.tie(name=...)`, `s.distributing(name=...)`, `s.node_to_surface(name=...)`,
  `s.node_to_surface_spring(name=...)`. The user names the constraint
  at apeGmsh time (`g.constraints.embedded(..., name="cimbra_embed")`)
  and claims it inside the stage block by the same name. Claim-by-name
  semantics (not direct-create) because the kernel resolver requires a
  live `gmsh` model + parts registry that are typically gone by bridge
  time.

- **`s.initial_stress(name=, pg=, sigma_*, ramp_steps=, ...)`** —
  PUSH-create mirror of `ops.initial_stress(...)`. Builds the record
  directly in the stage's pool, no intermediate `s.add(record)` step.
  Coexists with the existing PULL path; pick by style. A byte-identical
  parity test locks the equivalence.

- **Forcing function: Cerro Lindo SSI V5.** STKO's canonical SSI
  workflow installs lining (cimbra) in Stage 3 via `domainChange` onto
  an already-equilibrated rock state. With embed records always
  emitting globally pre-extension, the stiff penalty constraint
  (K=1e8) was active from t=0 and Newton had to equilibrate
  rock + lining + embed simultaneously from zero — diverged on step 2.
  The extension defers the embed to stage 2's block with `domainChange`
  AFTER constraint emit and BEFORE the stage's analysis chain, exactly
  mirroring STKO's behaviour.

- **Emit pipeline:** new `emit_stage_mp_constraints` /
  `emit_stage_mp_constraints_partitioned` orchestrators in
  [`_internal/build.py`](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/_internal/build.py) wrap a
  flat list of stage records via `_StageConstraintAdapter` and reuse
  the six per-kind helpers unchanged. The global emit orchestrators
  receive a `claimed_ids=` set and wrap the FEMData broker in
  `_ExcludeClaimedConstraints`, so claimed records never double-emit.
  `domain_change()` gate widens to include
  `stage.stage_constraint_records`.

- **Out of scope (deferred):** `s.tied_contact` /
  `s.mortar` — `tied_contact` wraps slave records inside a
  `SurfaceCouplingRecord` and the global exclusion filter operates on
  outer-record identity; `mortar` is not kernel-implemented. Both
  tracked in `_DEFERRED.md`. Implicit promotion of `g.constraints.*`
  records to stages (Path A from the scoping conversation) — users
  with pre-existing constraints migrate by adding `name=` and claiming
  in the right stage block.

- **ADR 0034 extended** with §5a (stage-bound constraints via CLAIM-by-
  name), §5b (`s.initial_stress` PUSH justification — PULL was
  forward-looking; side effects fire at emit time), §5c (Cerro Lindo
  forcing function).

### ADDED — ASDEmbeddedNodeElement optional flags now reach the deck (ADR 0035)

- **`g.constraints.tie(...)` / `embedded(...)` / `tied_contact(...)`
  gain four new kwargs** mapping 1:1 to the OpenSees `element
  ASDEmbeddedNodeElement` optionals documented at
  [ASDEmbeddedNodeElement.cpp:201](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp):

  ```python
  g.constraints.tie(
      master_label="shell", slave_label="solid",
      stiffness=1.0e8,        # -K  (penalty stiffness)
      stiffness_p=None,        # -KP (pressure stiffness; only with pressure=True)
      rotational=False,        # -rot (constrain Cnode rotations too)
      pressure=False,          # -p  (u-p / saturated-soil coupling)
  )
  ```

  Defaults (`stiffness=1.0e18`, `stiffness_p=None`, `rotational=False`,
  `pressure=False`) match the C++ parser at
  [ASDEmbeddedNodeElement.cpp:222](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp).
  Scripts that don't pass the new kwargs emit semantically-identical
  models — the visible difference is that the `-K 1e+18` token now
  appears in the Tcl/Py output, where previously it fell through to
  the silent C++ default.

- **Fail-loud `__post_init__` on every Def** mirrors the parser's
  mutual-exclusion check ([cpp:276](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp)):
  `rotational=True, pressure=True` raises `ValueError`. Setting
  `stiffness_p=...` without `pressure=True` also raises — `-KP` is
  ignored by OpenSees outside the u-p path, so the no-op is treated
  as a user mistake rather than silently absorbed.

- **Emitter Protocol** `embeddedNode` widens from
  `(ele_tag, cnode, *args)` to `(ele_tag, cnode, *master_nodes,
  stiffness=1.0e18, stiffness_p=None, rotational=False,
  pressure=False)`. All five concrete emitters (Tcl, Py, LiveOps,
  H5, Recording) honour the same kwargs; a shared
  `_build_embedded_flag_args` helper in
  [emitter/base.py](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/emitter/base.py) materialises
  the flag tokens in parser order for the text-based emitters.

- **H5 schema 2.11.0 → 2.12.0 (additive).**
  `/opensees/constraints/embeddedNode` gains five typed columns —
  `stiffness` (float64), `stiffness_p` (float64) +
  `has_stiffness_p` (uint8 sentinel for `None`), `rotational`
  (uint8), `pressure` (uint8). Per ADR 0023 two-version reader
  window, the bridge accepts both 2.11.x and 2.12.x files. Old
  2.11.x readers ignore the new columns; new readers default the
  columns to the C++ values when a 2.11.x file lacks them.

- **Motivation.** Diffing apeGmsh-emitted decks against STKO surfaced
  a 10-order-of-magnitude K-divergence: STKO emitted `-K 1e8`
  explicitly while apeGmsh fell through to the C++ default `1e18`.
  Penalty conditioning of every tie / embedded / tied_contact
  constraint differed between the two pipelines. The exposure
  closes that gap and gives users the knob to tune K-conditioning
  on stiff-penalty models. See
  [ADR 0035](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0035-asd-embedded-node-element-option-exposure.md).

### CHANGED — boolean.fragment default (BREAKING)

- **`g.model.boolean.fragment(...)` default `cleanup_free` flipped from
  `True` to `False`.** The previous default silently destroyed shell
  surfaces sitting on top of solids in shell-on-solid coupling
  scenarios (and any other dim-2 entity without an upward-volume
  adjacency). Existing scripts that relied on the destructive cleanup
  must now pass `cleanup_free=True` explicitly. The viewer's Fragment
  panel checkbox also defaults unchecked to match.

### ADDED — top-level `g.node_ndf` composite, explicit-only per-node `ndf` (S1b, ADR 0032)

- **New top-level composite `g.node_ndf`** (sibling to `g.constraints`
  / `g.loads` / `g.masses`) for explicit per-node DOF count
  declarations. Required for any model that mixes ndf — most
  importantly shell-on-solid coupling where shared nodes need
  `ndf=6`. API surface:

  ```python
  g.node_ndf.set_default(ndf=3)             # uniform fallback
  g.node_ndf.set("ShellRegion", ndf=6)      # targeted override
  g.node_ndf.list()                         # registered defs
  g.node_ndf.clear()                        # drop all defs
  ```

  Targets follow the same flexible scheme loads and masses accept
  (label / PG / part / mesh-selection / raw DimTag list). Values
  must lie in `[1, 6]`.

- **Fail-loud `fem.nodes.ndf_for(nid)`** raises `LookupError` for
  any node not covered by a declaration or default; the error
  message names both fixes so the user picks the right one. apeGmsh
  deliberately does **not** infer `ndf` from element class — the
  user is the single source of truth (ADR 0032).

- **H5 schema 2.6.0 → 2.7.0** writes an optional `/nodes/ndf` int8
  dataset; readers tolerate 2.6.x absence (two-version window per
  ADR 0023) and raise `MalformedH5Error` on length mismatch with
  `/nodes/ids`. The snapshot_id digest folds `_ndf` when present so
  identical geometry with different declarations hashes differently
  (presence-gated to preserve legacy / direct-test FEM digests).

- **PR #321 hardening:** declaration-order invariant hash (the same
  resolved ndf array hashes identically regardless of `set()`
  ordering); `_fem_built` resets at the top of each FEM build so
  post-extract warnings only fire on genuine post-cache mutations;
  `g.node_ndf.clear()` warns when called after extraction with the
  same once-per-batch semantics as `set` / `set_default`; real
  2.6.0 back-compat fixture replaces the synthetic version-rewrite
  test.

### ADDED — S2: per-node `ndf` flows into OpenSees emit (override-only, ADR 0033)

- **`g.node_ndf` declarations now reach the emitted deck** (PR #325).
  S1 stored per-node `ndf` on the broker; S2 wires it into every
  OpenSees emit path. **Override-only semantics**: `g.node_ndf` is
  the override channel; the model envelope set via
  `apeSees(fem).model(ndm, ndf=K)` is the default. The emitter passes
  `-ndf K` on `ops.node(...)` only when the broker carries a
  non-sentinel value at that nid; sentinel slots elide `-ndf` and
  OpenSees applies the envelope. This mirrors OpenSees-native
  `model BasicBuilder -ndf K` + per-node `-ndf J` override semantics.

  **Zero user-facing migration cost.** Existing scripts that never
  touched `g.node_ndf` emit byte-identical decks — the ~285 existing
  `apeSees(fem)` test sites and every example notebook keep working
  without rewrite.

- **Five wiring sites** consult the broker via
  `fem.nodes.ndf_for(tag)` wrapped in `try/except LookupError` (the
  miss falls back to the envelope, preserving ADR 0032's fail-loud
  broker contract): four owned-node emit sites in
  `apesees.py` (flat global, flat staged-owned, partitioned global,
  partitioned staged-owned) plus one foreign-node site in
  `_internal/build.py::emit_mp_constraints_partitioned`. The
  `_replay_into` helper in `_internal/compose.py` widens its per-node
  tuple to `(tag, coords, ndf|None)` so per-node declarations survive
  an H5 round-trip without truncating to the envelope.

- **Three-site validator** (`validate_envelope_covers_broker_ndf`)
  fires at every `OpenSeesModel` materialisation path —
  `apeSees.model()`, `OpenSeesModel.from_compose_buffers()`, and
  `OpenSeesModel.from_h5()`. A misconfigured envelope
  (e.g. `apeSees(fem).model(ndf=3)` after
  `g.node_ndf.set("Shells", ndf=6)`) raises `BridgeError` at the
  call site, naming the offending node and the fix.

- **`from_msh` reverted to `_ndf=None`** (undoes PR #321's
  zero-stamping); `_hash_nodes` skips the `_ndf` fold when None OR
  all-sentinel, preserving hash symmetry across construction paths
  AND the emit-layer envelope fallback on `.msh`-loaded models.

- **OpenSeesMP consistency is hash-guaranteed** (ADR 0021): the
  resolved `_ndf` array folds into `fem_hash`; every rank
  deserialises the same broker, so all ranks agree on per-node `ndf`
  for shared nodes without explicit cross-rank communication.

- **Architecture:** [ADR 0033](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0033-s2-emit-wiring-per-node-ndf.md)
  codifies the wiring, validator sites, phantom carveout, and
  hash-guaranteed cross-rank consistency. Extends
  [ADR 0032](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0032-explicit-only-per-node-ndf.md)
  (the broker contract S2 consumes).

- **PR #328 (S2 follow-up):** stateful `set_phantom_node_mode(emitter,
  bool)` side-channel replaced with a stateless
  `set_phantom_node_tags(emitter, set[int])` predicate set,
  pre-loaded ONCE at the entry of `emit_mp_constraints` /
  `emit_mp_constraints_partitioned`. Phantom tags are guaranteed
  disjoint from real broker tags (the resolver allocates
  `> max(broker_node_tag)`), so the pre-loaded set classifies every
  subsequent `node()` call without flag-flipping or ordering
  constraints. ADR 0033 §"Phantom-node carveout" gained an
  "Alternatives considered" subsection naming the rejected paths
  (mode flag, tag-formula predicate, Protocol-widening kwarg).
  Eight emit-path tests added in `tests/test_node_ndf.py` covering
  the headline mixed-ndf shell-on-solid case, backcompat byte-identity,
  each of the three validator sites, H5 round-trip, the `from_msh`
  envelope path, and the phantom predicate-set regression.

### ADDED — S5: partitioned mixed-ndf shell-on-solid E2E (closes the stream)

- **`tests/opensees/integration/test_emit_partitioned_mixed_ndf_shell_on_solid.py`**
  (PR #330) — six OpenSeesMP integration tests proving the
  per-foreign-node `ndf` lookup added in PR #325 fires per-tag across
  partition boundaries. The S2 merge unit-tested the flat case; this
  PR closes the partitioned half:
  1. headline mixed-ndf partitioned emit — shell on rank 0 emits
     `ndf=6`, solid on rank 1 emits `ndf=3`, foreign-node decls on
     each rank carry the *broker-sourced* peer `ndf`;
  2. INV-2 preserved (foreign node decl precedes `equalDOF`);
  3. `ATTR_PHANTOM_NODE_TAGS` is empty `frozenset` when no
     `NodeToSurfaceRecord` exists (guards against real broker tags
     being mis-classified as phantoms);
  4. cross-rank consistency — rank 0's foreign-decl ndf for tag T
     equals rank 1's owned-decl ndf for tag T (observable surface of
     ADR 0033's hash-guaranteed agreement);
  5. byte-identical backcompat on uniform-ndf partitioned models;
  6. envelope validator fires at `apeSees.model(...)`, not deep in
     the per-rank fan-out.

  Assertion strategy follows the precedent in every other
  `tests/opensees/integration/test_emit_partitioned_*` file:
  in-process deck capture via `RecordingEmitter`, bucketed per-rank
  by `partition_open` / `partition_close` brackets — no subprocess
  MPI runtime needed for emit-semantics assertions. Test-only PR;
  zero production code changes. Closes the S1 + S2 + S5
  shell-to-solid coupling stream.

### FIXED — Embedded-element pipeline: tag namespace + intersection host-rank + fail-loud guards

- **PR #329 — canonical TagAllocator + intersection host-rank rule
  for partitioned `ASDEmbeddedNodeElement`.** Two latent partition-
  emit bugs in the surface-coupling fan-out:

  1. **Global tag collision under partitioning.** The retired
     `_allocate_embedded_tag_base` returned a static `1_000_000`
     and `_emit_surface_couplings_for_rank` restarted its per-call
     counter from that base on every rank — so two distinct embedded
     records emitted on different ranks both received tag
     `1_000_000`, violating [ADR 0027](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0027-cross-partition-mp-constraints.md) §"Tag determinism".
  2. **Duplicate emit on boundary-shared masters.**
     `_plan_rank_constraints` used `if partition_rank in
     node_owners[masters[0]]` — when `masters[0]` was a partition-
     boundary node owned by multiple ranks, every owning rank's plan
     included the record and the `embeddedNode` line emitted on each
     (duplicate stiffness contribution at solve time).

  Fix: thread the bridge's canonical `TagAllocator` through
  `emit_mp_constraints` / `emit_mp_constraints_partitioned` /
  `_emit_surface_couplings` / `_emit_surface_couplings_for_rank` —
  embedded element tags now come from `tags.allocate("element")`,
  sharing the global element-tag namespace and guaranteed unique
  across ranks. Intersection host-rank rule: the unique owner of
  an embedded record is the host element's owning rank, NOT every
  rank in `node_owners[masters[0]]`. New integration test
  `tests/opensees/integration/test_emit_partitioned_embedded.py`
  (~435 lines) locks both fixes end-to-end. ADR 0027 §"ASDEmbeddedNodeElement
  ownership" + §"Tag determinism" extended to document the canonical-
  allocator share and the intersection rule so future readers don't
  reintroduce either bug.

- **PR #331 — three fail-loud guards on the embedded resolver / emit
  path.** Three resolver/emit-time silent-wrongs that produced
  "valid-looking" decks with broken physics:

  1. **Off-host extrapolation.** `resolve_embedded` accepted the
     closest host candidate regardless of barycentric excess, so an
     embedded node OUTSIDE every host element produced an
     `InterpolationRecord` with NEGATIVE shape-function weights —
     extrapolation, not interpolation. `EmbeddedDef.tolerance` was
     documented as "not currently enforced" and did nothing. New
     `excess: float | None` field on `InterpolationRecord` carries
     the barycentric excess from the resolver; the emit-time guard
     in `build.py` raises `BridgeError` naming the offending slave
     node + its excess when the resolver returns `excess > tolerance`.
  2. **Mixed-host silent drop.** `_collect_host_elems` kept only
     Gmsh element types 2 (tri3) and 4 (tet4); quad4, hex8, prism,
     pyramid, tri6, tet10, and any other type were silently dropped.
     A user meshing the host with hex + a few tets (transition
     region) saw only the tet subset, with embedded nodes in the hex
     region projecting onto distant tets (extrapolation again). The
     collector now retains all supported host element types; mixed-
     host configurations that the C++ ASDEmbeddedNodeElement parser
     cannot consume raise at build time, not at OpenSees runtime.
  3. **Bad Rnode count reaching the C++ parser.** The C++
     `ASDEmbeddedNodeElement` only accepts 3 (tri host) or 4 (tet
     host) Rnodes; 5+ are misread as flag positions, 2 aborts in
     `setDomain`. Hand-built records bypassing the resolver could
     deliver any count and crash OpenSees at runtime. New build-time
     guard raises `BridgeError` naming the offending record's Rnode
     count.

  Combined coverage: 81 new lines in `tests/test_constraint_emission.py`
  + 80 new lines in `tests/test_constraint_resolver.py` lock the
  three guards on positive + negative cases. Full suite: 6733
  passed.

### ADDED — Phase SSI-2.D: stage-bound BCs and recorders (ADR 0034)

- **`_StageBuilder` gains four new verbs** for binding boundary
  conditions and recorders to a specific stage of a multi-stage
  analysis (PRs #323 / #324 / #326):

  ```python
  with ops.stage("excavate") as s:
      s.activate(pgs=["Lining"])                       # SSI-2.B
      s.fix(pg="LiningAnchor", dofs=(1, 1, 1))         # NEW (SSI-2.D)
      s.mass(pg="Lining", values=(100.0, 100.0, 100.0)) # NEW
      s.region(name="lining_rayleigh", pg="Lining")    # NEW
      s.recorder(lining_recorder_spec)                 # NEW
      s.analysis(...)
      s.run(n_increments=20, dt=0.05)
  ```

  Replaces the SSI-2.B workaround of "keep the BC on a globally-
  emitted node" for stage-bound topology. `s.fix` / `s.mass` /
  `s.region` use PUSH (inert dataclasses on the stage directly);
  `s.recorder` uses PULL (claims a `Recorder` already registered via
  `ops.recorder.Node` / `Element` / `MPCO`).

- **Five-validator ownership-tier surface** in
  `BuiltModel._run_staged_bc_validators` — orchestrates H1 (global
  pool targets stage-bound nodes; refactored to share helpers with
  V1) + V1 (stage N's BC targets stage M > N) + V2 (cross-tier
  duplicate fix / mass) + V3 (region `name=` collision across
  scopes) + V4 (stage N's recorder targets stage M > N). Each
  emits a `BridgeError` with an offender list naming the offending
  scopes; the orchestrator runs in fixed order so error-message
  stability holds.

- **PR #323 also fixes a pre-existing latent H1 bug**: the
  partitioned emit path previously skipped H1 entirely — a global
  `fix` on a stage-bound node would slip through under MP and
  crash OpenSees at parse time. H1 now invoked from both
  `_emit_flat` and `_emit_partitioned`.

- **Emit pipeline extensions:** stage-bound `fix` / `mass` /
  `region` emit inside the stage block alongside topology and
  before a single unified `domain_change` (gated on
  `activation OR fix_records OR mass_records OR region_records`);
  stage-bound recorders emit after the chain and before `analyze`
  so they capture the stage's analyze steps. Under MP: per-rank
  fan-out with empty-bracket skip on non-contributing ranks
  (prevents Py-emitter `SyntaxError`); per-stage
  `region_tag_cache` keyed by name guarantees all contributing
  ranks emit the same scalar tag for a given region.

- **Bridge introspection symmetry:** `bridge.all_fix_records` /
  `all_mass_records` / `all_region_records` / `all_recorder_specs`
  read-only properties combining global + per-stage pools, tagged
  by origin. Tooling that previously inspected `bridge._fix_records`
  to count fix declarations would have silently missed stage-bound
  entries.

- **Source-side basis:** a four-agent cross-check against the
  OpenSees C++ source preceded implementation. Key verifications:
  `Domain::addRegion` silently appends on duplicate tag
  (Domain.cpp:2679-2697 — `getRegion` returns only the first);
  `Domain::addSP_Constraint` rejects duplicate `(node, DOF)` pairs
  (Domain.cpp:589-605); recorders cache region members at TCL
  PARSE TIME (TclRecorderCommands.cpp:276); `Domain::setMass`
  silently overwrites. These findings shaped V3 (mandatory),
  V2-fix branch, recorder emit position (after region declarations),
  and V2-mass branch respectively.

- **Architecture:** [ADR 0034](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/decisions/0034-stage-bound-bcs-and-recorders.md),
  [staged-analysis.md](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/staged-analysis.md)
  (refreshed with the SSI-2.D slot order + V1-V4 + per-stage tag
  cache), [api-design.md §"Staged analysis"](https://github.com/nmorabowen/apeGmsh/blob/main/src/apeGmsh/opensees/architecture/api-design.md)
  (refreshed with the four new verbs + PUSH vs PULL note).

- **Test coverage:** 38 new tests across four files (15 unit V1-V3
  + 13 unit fix/mass + 6 integration partitioned fix/mass + 19
  unit region/recorder/V4 + 4 integration partitioned regions).
  Full opensees suite: 2746 passed, 2 skipped, 0 regressions.

## v2.0.0 — Three-broker chain: Results carries OpenSeesModel carries FEMData (BREAKING) · Composed file pattern · lineage chain · MP constraint emission shipped · per-zone schemas

Major architectural refactor establishing the **FEM ⊂ Model ⊂
Results** chain with bidirectional H5 round-trip and a git-style
lineage DAG. 8 phases shipped sequentially over May 2026; 5864 → 6128
baseline tests passing (+264 net new, 0 failed). Five new ADRs lock
the architecture (0019–0023); three prior ADRs (0011, 0014, 0018)
preserved verbatim, AST-tested.

This is a **BREAKING** release — the additive-then-prune migration
shipped all new surfaces alongside the old ones through Phases 4–7,
then Phase 8 pruned the deprecated paths in one go. External users
must update call sites (see Migration below).

### ADDED — OpenSeesModel read-side broker (ADR 0019)

A new immutable, queryable Python view over a `model.h5`'s
`/opensees/` zone:

```python
from apeGmsh.opensees import OpenSeesModel

# Standalone model.h5 (apeSees(fem).h5(p) output):
om = OpenSeesModel.from_h5("model.h5")

# Composed file (results.h5 carries /model + /opensees):
om = OpenSeesModel.from_h5("results.h5", fem_root="/model")

om.fem                  # → FEMData (lazy-imported per INV-4)
om.materials()          # → list[MaterialRecord]    (typed records)
om.sections()           # → list[SectionRecord]
om.transforms()         # → list[TransformRecord]   (carries vecxz)
om.beam_integration()   # → list[BeamIntegrationRecord]
om.patterns()           # → list[PatternRecord]
om.recorders()          # → list[RecorderRecord]
om.cuts() / om.sweeps() # → list[CutRecord / SweepRecord]
om.lineage              # → Lineage(fem_hash, model_hash, warnings)

# Re-emit through any target:
om.build("tcl", "deck.tcl")
om.build("py", "deck.py")
om.build("live")     # in-process openseespy
om.to_h5("copy.h5")  # round-trip via _compose_model_h5
```

`OpenSeesModel` is the **third role** alongside `apeSees(fem)` (the
bridge — write side, typed primitives) and `ModelData(fem)`
(orientation-only side-feeder). ADR 0011 preserved verbatim —
`apeSees.from_h5` does NOT exist; the read goes through a
separate class.

### ADDED — Results carries OpenSeesModel via Composed file (ADR 0020)

The chain forward — `Results.model.fem`. Composed file pattern: one
`results.h5` carries both the model and the results data:

```
results.h5
├── /meta            envelope + per-zone schema versions + lineage
├── /model/          rich FEMData neutral zone
├── /opensees/       bridge zone (transforms, materials, constraints, ...)
└── /stages/...      results data
```

Viewer reads `results.model` directly; no separate `model_h5=` kwarg
needed (preserves ADR 0014 AST guard — viewer remains a pure h5
consumer).

### ADDED — Lineage chain replaces snapshot_id binding (ADR 0021)

Git-style content-hash DAG, warn-not-raise on mismatch:

```
fem_hash      = blake2b(canonical_neutral_zone_bytes)
model_hash    = blake2b(fem_hash || canonical_opensees_zone_bytes)
results_hash  = blake2b(model_hash || canonical_run_zone_bytes)
```

Stored at `/meta/lineage`. `results.lineage.warnings` surfaces
mismatches (list[str]) but never raises from a constructor.
`results.lineage.assert_clean()` is the opt-in loud-fail.

### ADDED — MP constraint emission (ADR 0022) — closes §3.3 deferral

MP constraints now emit automatically into runnable Tcl/Py/Live
decks via 5 new `Emitter` Protocol methods (`equalDOF`, `rigidLink`,
`rigidDiaphragm`, `embeddedNode`, `mp_constraint_comment`). The
build-time fan-out (`opensees/_internal/build.py::emit_mp_constraints`)
walks `fem.nodes.constraints` + `fem.elements.constraints` in
phantom-node-first order. Three integration tests run actual
`ops.analyze()` on rigid_diaphragm / rigid_link / equalDOF / tied_contact
fixtures to prove emitted decks converge (INV-1).

The bridge auto-emits `ops.constraints.Transformation()` when MP
constraints are present and the user has not declared a constraint
handler — closes the "Plain silently ignores MP constraints"
footgun. Users who explicitly want Plain still get respected; a
UserWarning fires.

### ADDED — Per-zone schema versioning (ADR 0023)

Three independent semver stamps replace the racing single envelope:

- `/meta/neutral_schema_version` = `"2.6.0"`
- `/meta/opensees_schema_version` = `"2.8.0"`
- `/meta/results_schema_version` = `"1.1.0"`

Envelope `/meta/schema_version` retained for single-stamp legacy
back-compat. Two-version reader window: reader at X.Y.Z accepts
X.Y.* and X.(Y-1).*; refuses everything else with `SchemaVersionError`.

The opensees zone shipped at `2.7.0` for the additive
`/opensees/constraints/` group, then bumped to `2.8.0` for a
follow-up field rename: the second compound-dtype column of
`/opensees/constraints/embeddedNode` was `embedding_ele` in 2.7.0
(a misnomer — the stored value is the constrained / slave node id,
not an element id) and is `cnode` from 2.8.0 onward (matches the
OpenSees `$Cnode` vocabulary). Same rename rippled through the
`Emitter.embeddedNode` Protocol parameter name and the
`EmbeddedNodeRecord` dataclass field. Two-version reader window
accepts both 2.7.x and 2.8.x files; the column name is
version-dependent.

### ADDED — `FEMData.from_h5(path, *, root="/")` parameterization

The same reader handles both standalone `model.h5` (FEM at root)
and Composed `results.h5` (FEM at `/model/`) via the `root=` kwarg.
`OpenSeesModel.from_h5(path, *, fem_root="/", opensees_root="/opensees")`
extends the same idea.

### CHANGED — FEMData round-trip is now lossless on every record field

Phase 2 closed five audit gaps:
- `name` field on every constraint/load/mass record now round-trips
  (was silently dropped — affected `fem.inspect.constraint_summary()` etc.)
- `partitions` + `part_node_map` / `part_elem_map` now round-trip
  (was lost — `fem.nodes.select(partition=k)` / `select(target=part_label)`
  raised `KeyError` after `from_h5`)
- `info.bandwidth` recomputed on reload (was hardcoded to 0)
- `/meta/snapshot_id` now VERIFIED on read (was written but not
  checked — tampered bytes now raise `MalformedH5Error`)
- New `_assert_fem_equivalent(rebuilt, original)` parity test exercises
  five canonical fixtures (frame, plate, mixed-dim assembly,
  partitioned, mesh-selection) — the meta-gap that allowed B1-B4 to
  slip through.

### REMOVED (BREAKING — Phase 8 prune)

- **`Results.from_native/from_mpco/from_recorders` REQUIRE `model=` /
  `model_h5=`** — `TypeError` on missing. No more silent auto-resolve.
- **`Results.viewer(model_h5=...)` kwarg removed** — chain flows
  through `results.model`. CLI: `python -m apeGmsh.viewers run.h5`
  for native files (auto-resolves); `--model-h5 PATH` required ONLY
  for `.mpco` files (sibling pointer).
- **`BindError` class DELETED** — lineage chain replaces it.
- **`Director.set_model_h5(path)` public method removed** — use
  `set_model(opensees_model)` or pass via `Results.model`. Internal
  `_bind_model_h5(path)` private helper retained for cuts auto-load
  and session restore.
- **`h5_reader.materials()` / `sections()` / etc. now return typed
  records** — the dict-style versions were deleted. Use
  `materials_by_family()` for the family-keyed view.
- **`EXPECTED_SCHEMA_MAJOR` constant removed** — readers use
  per-zone validation via `_internal/schema_version.reader_version(zone)`.
- **`_femdata_native_io.py` deleted** (439 LOC) — production paths
  use the rich neutral-zone layout under `/model/` via the parameterized
  `read_neutral_zone_from_group` / `write_neutral_zone_into_group`.

### Migration

For users with notebooks / scripts that call the old API:

```python
# 1. Add the OpenSeesModel import where you load Results
from apeGmsh.opensees import OpenSeesModel

# 2. For Composed files (apeSees(fem).h5(path) output):
results = Results.from_native(
    path,
    model=OpenSeesModel.from_h5(path, fem_root="/model"),
)

# 3. For standalone model.h5 references:
om = OpenSeesModel.from_h5(model_path)  # default fem_root="/"
results = Results.from_native(results_path, model=om)

# 4. For MPCO + sibling model.h5:
results = Results.from_mpco(mpco_path, model_h5=sibling_model_path)

# 5. Viewer — drop the model_h5= kwarg:
results.viewer()   # NOT results.viewer(model_h5=...)

# 6. Inspect lineage instead of catching BindError:
for w in results.lineage.warnings:
    print(f"lineage warning: {w}")
# or opt into loud-fail:
results.lineage.assert_clean()
```

### Deferred follow-ups (not in this release)

- `embeddedNode` per-kind args formalization (`tie` / `mortar` /
  `tied_contact` / `embedded` share one Protocol method with positional
  packing; works for all known cases because ASDEmbeddedNodeElement
  uses internal isoparametric interpolation).
- Migration tool for archives outside the two-version reader window
  (per ADR 0023 INV-5 — owed but not urgent).

---

## v1.6.0 — Selection-unification v2: one `.select()` idiom · legacy selection surface REMOVED (BREAKING) · half-open mesh box · loads/masses fail-loud

Selection / resolution unification (**v2**). A single
**daisy-chainable `.select()` idiom** is now the *only* selection idiom
at all four levels — geometry, the live mesh, the FEM broker, and
results. This is a **BREAKING** change: the legacy selection surface
was **removed with no deprecation shim** (project-owner-ratified full
removal — v1's backward-compat constraint was explicitly dropped).
Removed: `fem.nodes.get` / `fem.elements.get` / `.resolve` (the
selection accessors), the `results.*.select(...).values()` chain path,
`g.mesh_selection.add_nodes` / `add_elements` / `from_geometric`,
`g.model.queries.select` / `queries.line` / `select_all*`,
`g.model.selection` (`SelectionComposite`), the four legacy `*Chain`
modules + `GeometryChain`, and the `Selection` / `SelectionComposite`
package exports. The classes `core._selection.Selection` and
`viz.Selection` are **retained by architecture** as internal
terminal-payload / viewer-pick-result types — **not** user entry
points; only their exports were dropped.

**Migrate every legacy call** per the old→v2 table and the two
**incomplete-unification gaps** in
[`api/selection.md`](api/selection.md) (ADR 0017 supersedes the earlier
ADR-0016 "accepted gap" framing — these are *owed v2 successors*, not
WONTFIX). Gap 1 (geometric-selection → named mesh-selection): the
capability is **intact** via the retained 2-call route
(`g.model.select(...).to_physical(name)` then
`g.mesh_selection.from_physical(...)`) — only the one-call ergonomic
was lost. Gap 2 (the `SelectionComposite` filter grammar): a genuine
unique-capability loss for which a **v2-native `EntitySelection`
successor is owed/planned** (not a resurrected `SelectionComposite`).

Two behavior changes ride along. The `g.mesh_selection` box filter
moves **closed → half-open** by default to match the results side
(**breaking** — see below; `inclusive=True` restores the old closed
box). And the loads/masses `__ms__` consumer now **fails loud**
instead of silently binding to zero nodes — the one remaining member
of a three-path fail-loud end-state whose other two paths landed
earlier (see below).

This release is selection plumbing only — no viewer or solver-bridge
surface changed.

### ADDED — One fluent `.select()` idiom at all four levels

A new canonical, daisy-chainable selection chain. Entry points,
returning a chain that composes fluently:

| Entry point | Terminal | Family | Result |
|---|---|---|---|
| `g.model.select(target, *, dim=)` | `EntitySelection` | entity | `.to_label`/`.to_physical`/`.to_dataframe`/`.result()` (→ the retained `Selection` payload) |
| `fem.nodes.select(...)` | `MeshSelection` | point | `.result()` → `NodeResult`; `.ids`/`.coords` |
| `fem.elements.select(...)` | `MeshSelection` | point | `.groups()`/`.result()`/`.resolve()` → `GroupResult` |
| `results.nodes.select(...)` / `results.elements.select(...)` | `MeshSelection` | point | `.values(component=, time=, stage=)` → slab (forwards onto the retained `results.<level>.get(...)` reader) |
| `g.mesh_selection.select(*, level=, dim=, ids=, name=)` | `MeshSelection` | point | `.result()` → same dict as `get_nodes/get_elements`; `.ids`; `.save_as(name)` (live-mesh only) |

- **Refining verbs** (identical names on every chain): `.in_box(lo,
  hi, *, inclusive=False)`, `.in_sphere(center, radius)`,
  `.on_plane(point, normal, *, tol=)`, `.nearest_to(point, *,
  count=1)`, `.where(predicate)`. Each returns a new chain of the
  same concrete type, so they daisy-chain:
  ```python
  sel = (fem.nodes.select(pg="Body")
             .in_box((0, 0, 0), (1, 1, 1))
             .on_plane((0, 0, 0), (0, 0, 1), tol=1e-6))
  result = sel.result()          # the existing NodeResult
  ```
- **Set algebra**: `|` `&` `-` `^` (and the named aliases `.union`
  / `.intersect` / `.difference`), insertion-order preserving with
  one fixed dedup law. Combining two chains of different type or
  bound to different engines (different `FEMData` / `Results` /
  session) raises `TypeError` — cross-level mixing is loud, never a
  silent empty set.
- **Seeding** reuses the existing contract-locked resolvers, never a
  re-implementation. Broker chains take the same selectors as `.get`
  (`target` / `pg` / `label` / `tag` / `dim` / `partition`,
  `element_type` for elements) plus `ids=`; no-arg seeds the whole
  domain. Results chains take `pg=` / `label=` / `selection=` /
  `ids=`. `g.mesh_selection.select` takes `ids=` or `name=` (an
  **existing** `g.mesh_selection` set, seeded id-for-id by delegating
  verbatim to the existing `get_tag`/`get_nodes`/`get_elements`
  surface — no new resolver, read-only, fail-loud on an unknown
  name); no-arg seeds the full live-mesh universe. `g.model.select`
  delegates string resolution to the same label → PG → part tier as
  everywhere else.
- **Two spatial families, honestly different — not interchangeable:**
  - **point family** (`fem.*`, `results.*`, `g.mesh_selection`):
    `.in_box` is half-open `[lo, hi)` by default; `inclusive=True`
    gives the closed box `[lo, hi]`. Operates on node coordinates
    (node chains) or element **centroids** (element chains).
  - **entity family** (`g.model.select`): `.in_box` delegates to
    Gmsh's `getEntitiesInBoundingBox` — BRep bounding-box
    **containment** (the whole entity bbox must lie inside the query
    box, expanded by `Geometry.Tolerance` ≈ 1e-8). There is no
    half-open notion, so passing `inclusive=` (or any keyword)
    **raises `TypeError`** rather than being silently ignored. For
    an exact geometric on/crossing predicate use
    `g.model.select(target).crossing_plane(spec, mode="on"|"crossing")`
    (the v2 successor; `g.model.queries.select` was removed).

  The two families share verb *names* and set algebra but never
  share `.in_box` behavior; do not assume a cross-family result is
  identical.
- The classes `core/_selection.Selection` and `viz/Selection.Selection`
  are **retained by architecture** (the `EntitySelection.result()`
  terminal payload and the viewer pick-result type, respectively) —
  structurally distinct internal types, **not** user entry points;
  only their package exports were dropped.
  `g.model.select(...).result()` returns the retained `core`
  `Selection` payload (`.to_label` / `.to_physical` / `.to_dataframe`
  are also direct terminals).
- **Named persistence is `.save_as(name)`** on a `MeshSelection`
  (**live-mesh engine only**), or the retained explicit-ids registrar
  `g.mesh_selection.add(dim, ids, name=)` → FEMData snapshot →
  `results(selection=...)`. The removed
  `g.mesh_selection.add_nodes(..., name=...)` / `from_geometric`
  two-step keeps its capability via the retained 2-call route
  (`g.model.select(...).to_physical(name)` →
  `g.mesh_selection.from_physical(...)`); only the one-call ergonomic
  was lost (incomplete-unification Gap 1 — see
  [`api/selection.md`](api/selection.md)).

### BREAKING — `g.mesh_selection` box is now half-open by default

`g.mesh_selection.add_nodes(in_box=)`, `add_elements(in_box=)`, and
`filter_set(in_box=)` (the `_mesh_filters.nodes_in_box` engine)
flipped from **closed `[lo, hi]`** to **half-open `[lo, hi)`** on the
upper side, per axis. This matches the `results` side, which was
already half-open — the two diverged on `main`; this reconciles them
on the canonical (half-open) semantics.

**Effect:** a node/element coordinate (or centroid) lying exactly on
an *upper* box face is now **excluded** by default. Adjacent boxes no
longer double-count a shared face.

**Migration** — pass `inclusive=True` to restore the old closed box:

```python
# Before (pre-v1.6.0): closed [lo, hi] — upper face INCLUDED
sid = g.mesh_selection.add_nodes(in_box=(0, 0, 0, 1, 1, 1))

# v1.6.0 default: half-open [lo, hi) — upper face EXCLUDED.
# On a 3x3x3 unit-cube lattice this drops 27 nodes -> 8.
sid = g.mesh_selection.add_nodes(in_box=(0, 0, 0, 1, 1, 1))

# To keep the exact pre-v1.6.0 result, opt back into the closed box:
sid = g.mesh_selection.add_nodes(
    in_box=(0, 0, 0, 1, 1, 1), inclusive=True,   # 27 nodes (closed)
)
```

`inclusive=` is also accepted on `add_elements` and `filter_set`.
Audit any `g.mesh_selection` box query that intentionally relied on
catching on-the-upper-face nodes; pad the upper corner outward or pass
`inclusive=True`. Pure-interior boxes are unaffected.

### CHANGED — Loads/masses `__ms__` now fails loud (completing a three-path end-state)

Three paths that once returned a quietly-wrong result now raise; each
is safer (a plausible-looking wrong answer becomes an explicit,
located error). **This release ships only path 2.** Paths 1 and 3
were already loud / merged ahead of it and are described here for the
complete end-state:

1. **`results(...)` `selection=` on an import-origin FEM (already
   loud — locked).** A `FEMData` built from `from_msh` / MPCO /
   native input has no `mesh_selection` (it is `None`). Passing
   `selection=` against such a FEM raises `RuntimeError`
   (`"selection= requires fem.mesh_selection to be present."`) on
   both the node and element resolution arms instead of resolving to
   an empty set. This path was already loud; it is held by a
   characterization pin and is **not** changed by this release. Build
   the selection on the session (`g.mesh_selection`) so it travels
   into the snapshot.
2. **Loads / masses `__ms__` consumer binding to zero nodes (the
   change in this release).** `LoadsComposite._target_nodes` (and the
   `MassesComposite` counterpart) hit a `if info is None: return
   set()` arm that **silently bound a load/mass to zero nodes** when
   a named mesh selection it referenced was gone or the store was
   inconsistent. It now raises `KeyError` (`"... Refusing to silently
   bind this load to zero nodes (fail loud)."`). A load/mass that
   resolves to nothing is a model error, not a no-op. This is the one
   code behavior shipping with this release (with a flipped
   characterization pin and a dedicated regression test).
3. **`results._element_centroids` corrupting a centroid (already
   merged separately — not this release).** This routine used
   `np.clip` to map a connectivity node id that is absent from the
   FEM node set to the *last* node — silently corrupting that
   element's centroid. It now raises `KeyError` (detecting both
   past-the-end and in-range-wrong-id, strictly stronger than the old
   clip), which also makes the legacy `results.elements.in_box` /
   `nearest_to` / `on_plane` helpers fail loud on a broken
   connectivity instead of returning garbage-located elements. The
   new chain centroids were already fail-loud; this brought the
   legacy helper to parity. This fix **merged independently, ahead
   of this release** — it is not introduced here.

### INTERNAL — deduped Loads/Masses target resolver

`MassesComposite._resolve_target` was a byte-for-byte clone of
`LoadsComposite._resolve_target`. Both now delegate to one shared
`core/_resolution.resolve_target` engine. Behavior is byte-identical
(tier order, the `("__ms__", dim, tag)` sentinel, `expected_dim`
scoping, multi-dim-PG `ValueError`, the verbatim per-noun error
messages). No public API change; not a migration. The library-wide
resolvers (`FEMData._resolve_one_target`, `_helpers._resolve_string`)
are deliberately untouched.

### Follow-ups (tracked, not yet shipped)

Consciously deferred; mentioned so they are not mistaken for
available surface:

- Results sub-composite `.select()` — `results.nodes` / `.elements`
  have `.select()`; the five element sub-composites
  (`gauss` / `fibers` / `layers` / `line_stations` / `springs`) do
  **not** yet (they need per-terminal kwarg forwarding).

(The `g.mesh_selection.select()` name-seed that previously sat here
has **shipped** — see the `.select()` ADDED section above.)

## v1.5.0 — Applied loads + reactions diagrams · geometry-scoped gate

One PR landing on top of v1.4.0's post-release work. The
ResultsViewer gains two new diagram kinds in the Add layer dropdown
— **Applied loads** (constant force arrows, one diagram per load
pattern) and **Reactions** (recorded reaction forces and moments,
auto-scaling per step with the time slider). The composition gate
is also tightened so adding a brand-new Geometry no longer leaves
the previous Geometry's diagrams visible.

PR in this release: [#92].

### ADDED — Applied loads diagram ([#92])

`Add layer → Applied loads` lists every `fem.nodes.loads` pattern
that carries at least one non-zero force record. Each diagram
renders force arrows at the resolved nodes for one pattern. Reference
magnitudes only — the broker does not carry the OpenSees
`timeSeries` function, so step scaling is intentionally deferred
(the diagram's `update_to_step` is a documented no-op until
timeSeries metadata lands in the broker). Moments are not drawn
yet; they need a different glyph and will follow.

### ADDED — Reactions diagram ([#92])

`Add layer → Reactions` (enabled iff the file has any
`reaction_force_*` or `reaction_moment_*` recordings). Forces render
as straight arrows; moments use the existing `moment_glyph` curved
arrow. Each family auto-fits its own scale because forces and
torques have different units. The diagram is step-resolved — every
time-slider move re-reads the slab and rebuilds the glyphs. An
auto-filter drops nodes whose magnitude max-over-time is below
`zero_tol × global_max`, so free-interior nodes don't pollute the
scene with near-zero arrows.

### FIXED — Composition gate scoped to active Geometry ([#92])

The gate previously hid layers using the flat diagram registry, so
adding a new (empty) Geometry kept the previous Geometry's diagrams
visible — `active_comp is None` set `show_all=True` and turned every
actor on regardless of which Geometry owned it. The visible-layer
set is now restricted to compositions of the active Geometry; an
empty Geometry shows nothing, an existing one with no active
composition still shows its own layers (preserving the prior
single-Geometry "show all" intent within the active Geometry).

[#92]: https://github.com/nmorabowen/apeGmsh/pull/92

## v1.4.0 — ResultsViewer dock split · new-layer attach + lifecycle fixes · import banner

Six PRs landing on top of v1.3.0. The post-solve viewer's right rail
splits into dedicated **Diagram** and **Geometry** docks (tabified
with Details and Session), with a new floating "?" shortcut HUD on
the viewport. New-layer attach now pushes the active step + re-fires
deformation sync so freshly-added line-force / vector-glyph layers
land aligned with the rest of the scene instead of paint-then-drift.
A series of selection-related composition-gate fixes (Esc, outline
Geometry-row click, stale session restore) stop diagrams from
silently disappearing when the user navigates the outline. The HDF5
reader is now released on viewer close, so re-running a capture
script in the same kernel no longer hits `PermissionError`. The
package prints an ASCII banner + `__version__` on import (suppress
with `APEGMSH_QUIET=1`) so the running version is unambiguous.

PRs in this release: [#69], [#70], [#71], [#72], [#73], [#74].

### ADDED — Diagram / Geometry dock split ([#69])

The right rail's single Details dock — which previously stacked the
DiagramSettingsTab and GeometrySettingsPanel inside a QStackedWidget
— is split into two dedicated docks:

- **Diagram** dock hosts `DiagramSettingsTab.widget` directly.
- **Geometry** dock hosts `GeometrySettingsPanel.widget` directly.
- **Details** dock is reserved for future canvas-driven contextual
  content (contour scalebar edits, picked-node readouts).
- Outline routing: clicking a Composition row raises the Diagram
  dock; clicking a Geometry row raises the Geometry dock.
- The "+ Add layer" button is now on the Diagram dock's title row.
- Layout schema bumped to v5 — saved v4 layouts are discarded so
  users land in the new arrangement on first launch.

### ADDED — Floating shortcut help HUD ([#69])

`ShortcutHelpHUD` — small "?" button in the viewport's bottom-right
corner. Click pops a list of mapped keyboard shortcuts (Esc, Ctrl+H,
Q, N/E/G, Shift+LMB, Shift+click, F2).

### ADDED — Banner + `__version__` on import ([#73])

`import apeGmsh` prints the ASCII banner + the installed version to
stderr. `__version__` is now exposed at the package root, sourced
from `importlib.metadata` (single source of truth = `pyproject.toml`).
Set `APEGMSH_QUIET=1` to suppress for tests / CI / piped scripts.

### FIXED — New-layer attach ([#69])

After `registry.add(...)`, the director now pushes the active step
to the new layer and re-fires `_apply_deformation`. Resolves:

- Line-force diagrams that rendered as collapsed slivers because
  step-0 internal forces are zero (the polydata was correct, the
  values were wrong).
- Vector glyphs that landed at undeformed positions until the user
  manually scrubbed the time slider.

### FIXED — HDF5 reader released on viewer close ([#70])

`ResultsViewer._on_close` now calls `self._results.close()` so the
NativeReader's file handle is released. Re-running a capture script
in the same Jupyter kernel — which deletes and recreates the same
`.h5` path — no longer hits `PermissionError: [WinError 32] The
process cannot access the file because it is being used by another
process`.

### FIXED — Composition gate no longer silently hides diagrams ([#71], [#72], [#74])

Three trigger paths surfaced the same symptom — layers visible in
the dock with checkboxes still checked, but nothing painting in the
viewport — because the composition gate hides every actor when no
composition is "active":

- **#71**: Esc previously called `compositions.set_active(None)`,
  which fired the gate. Esc now only clears probe markers + element
  / GP highlights and leaves composition state alone.
- **#72**: Clicking a Geometry row in the outline previously called
  `compositions.set_active(None)` for the same reason. Selecting a
  Geometry row is now a navigation gesture; composition state is
  left unchanged.
- **#74**: The session JSON persists `active_composition_id`. Pre-#71
  / pre-#72 sessions easily saved `null`. On restore, the gate then
  hid every layer until the user clicked a Composition row. Two-part
  fix: heal stale sessions by defaulting to the first composition
  when the saved active id is null; relax the gate to "show all"
  when no composition is active anywhere.

## v1.3.0 — ResultsViewer B++ redesign · live recorder + MPCO emission · spatial filters

Nine PRs landing on top of v1.2.0. The ResultsViewer ships a full B++
redesign — outline tree, plot pane, viewport HUDs (probe palette top-
right, pick-readout top-left), inline kind picker, style presets,
density toggle — replacing the right-dock tab strip. The recorder
spec gains two new in-process execution strategies (`emit_recorders`
and `emit_mpco`), so one declarative spec now drives five backend
paths. The read side picks up `nearest_to` / `in_box` / `in_sphere` /
`on_plane` spatial filters plus an `element_type=` selector, all
composing additively with the existing `pg=` / `label=` /
`selection=` / `ids=` vocabulary. Elastic beams (`ElasticBeam{2d,3d}`,
`ElasticTimoshenkoBeam{2d,3d}`, `ModElasticBeam2d`) gain a synthesised
2-station line-stations slab via the live capture path, matching the
existing MPCO behaviour. Documentation gets a 6-card landing, grouped
navigation, an 8-notebook curated examples gallery rendered inline
via mkdocs-jupyter, a Recorder reference page, and a Reading &
filtering results guide. **All 9 PRs merged green.**

PRs in this release: [#43], [#44], [#46], [#47], [#48], [#50], [#51],
[#52], [#49].

### ADDED — ResultsViewer B++ redesign ([#43], [#46])

Closes the [B++ Implementation Guide](architecture/apeGmsh_results_viewer.md).
The right-dock tab strip (Stages / Diagrams / Settings / Inspector /
Probes) is retired in favour of a 3×3 grid layout: title-bar row
(40 px), three-column body (left rail · viewport · right rail),
scrubber row (84 px).

- **`ResultsWindow` shell** ([#43] B0). Wraps `ViewerWindow` with the
  3×3 grid central widget. Hidden left (260 px) and right (380 px)
  columns reserve space for the upcoming widgets.
- **`OutlineTree`** ([#43] B1). Left-rail single navigator with four
  groups: Stages, Diagrams, Probes, Plots. Replaces the StagesTab and
  DiagramsTab. Visibility checkboxes toggle render; clicks drive the
  details panel.
- **`PlotPane` + `DetailsPanel`** ([#43] B2). Right-rail vertical-list
  tabs (dot · label · ×). Re-homes the fiber-section, layer-
  thickness, and time-history panels that previously floated as
  `QDockWidget`s on the main-window edges. The DetailsPanel below
  hosts contextual content (DiagramSettingsTab when a diagram row is
  selected).
- **`ProbePaletteHUD`** ([#43] B3). Floating panel in the viewport's
  top-right corner with three mode buttons (Point / Line / Slice) +
  Stop / Clear. Repositions on viewport resize via a Qt event filter.
  Retires `ProbesTab`.
- **`PickReadoutHUD`** ([#46]). Floating glass card in the viewport's
  top-left corner. Subscribes to `ProbeOverlay.on_point_result` and
  Director step / stage changes; renders the picked node id, snapped
  coords, and one mono-typed line per active component value.
  Retires `InspectorTab`.
- **Shift-click → time-history plot** ([#46]). `ShiftClickPicker`
  registers a low-priority VTK observer on `LeftButtonPressEvent`
  that fires only when shift is held; opens (or focuses) a
  `TimeHistoryPanel` as a closable plot-pane tab. The default
  component prefers the active diagram's selector.
- **Title-bar utility strip** ([#46]). Three decorative stop-light
  dots, breadcrumb label, right-aligned icon strip with theme cycle,
  clipboard screenshot, density toggle, help dialog. Theme cycles
  through every palette in `PALETTES`.
- **Density toggle** ([#46]). New `DensityManager` singleton mirroring
  `ThemeManager`. Persists via `QSettings`. `DensityTokens` carry
  `row_h`, `pad_x`, `pad_y`, `gap`, `fs_body`, `fs_head`. The
  global stylesheet picks them up; toggling triggers a full restyle.
- **Two-way tree ↔ plot-tab binding** ([#46]). The Plots group in
  the outline tree mirrors the plot-pane tab list; clicking a Plots
  row activates the matching tab. Empty Plots group falls back to a
  hint placeholder.
- **Inline 2×4 kind picker** ([#46]). Clicking the outline tree's
  "+ Insert" button reveals a 2×4 grid of diagram-kind shortcuts
  directly under the header. Selecting a kind opens
  `AddDiagramDialog` pre-selected for that kind.
- **Diagram picker pre-flight** ([#46]). The Add Diagram dialog now
  greys out kinds whose topology has no data anywhere in the
  Results file (`— no data` suffix). The Component combo
  placeholder distinguishes "no data in file" from "no data in
  selected stage".
- **Style presets** ([#46]). New module
  [`viewers/diagrams/_style_presets.py`] with `style_to_dict` /
  `style_from_dict` codec, `KIND_TO_STYLE_CLASS` registry, and a
  `StylePresetStore` (CRUD under `<QSettings AppConfigLocation>/
  apeGmsh/style_presets/`). Add Diagram dialog gains a Preset combo;
  `DiagramSettingsTab` gains a Save…/Apply footer. Path-traversal
  sanitiser refuses unsafe names.
- **Theme + global-preferences reachability** ([#46]). The
  ResultsWindow help dialog promotes from `QMessageBox` to a proper
  `QDialog` with footer buttons that open the Theme editor and
  Global preferences dialogs (the dock-strip path the other viewers
  use is gone in B5).
- **Theme integration for the new shell** ([#43] B4). Hardcoded
  inline stylesheets removed; the global `build_stylesheet` picks
  up object-name selectors for every new widget (`#ResultsTitleBar`,
  `#OutlineHeader`, `#PlotPaneHeader`, `#DetailsPanel`, `#ProbeHUD`,
  `#OutlineKindPicker`, `#OutlineKindBtn`, etc.). All four palettes
  (catppuccin_mocha, neutral_studio, catppuccin_latte, paper) render
  cleanly.

### ADDED — Live recorder + MPCO emission strategies ([#48])

Two new in-process consumers on the recorder spec — same seam, two
new code paths:

- **`spec.emit_recorders(out_dir)`** — classic recorders pushed live
  into the `ops` domain via `ops.recorder()` calls, with
  `begin_stage` / `end_stage` scoping and per-stage filename
  prefixes (`<stage>__<record>_<token>`).
- **`spec.emit_mpco(path)`** — single in-process MPCO recorder with
  a build-gate that raises with a clear remediation pointer when the
  active openseespy build doesn't include MPCO.
- Threads `stage_id` through emit → cache → transcoder →
  `from_recorders` (default `None` preserves byte-for-byte
  `export.tcl/py` compatibility).
- `to_ops_args` / `mpco_ops_args` are the live-emit equivalents of
  `format_python` / `emit_mpco_python`. Both flow through the
  existing `LogicalRecorder` dataclass so source-form and
  tuple-form share one source of truth.
- Architecture doc rewritten:
  [`apeGmsh_results_obtaining.md`](architecture/apeGmsh_results_obtaining.md)
  covers the spec-as-seam pattern with the five-strategy
  comparison table.
- New user-facing guide:
  [`guide_obtaining_results.md`](internal_docs/guide_obtaining_results.md)
  with worked recipes per strategy + decision flowchart + pitfalls.
- 46 new tests; full recorder/live/mpco sweep at 495 passing.

### ADDED — Spatial filters on every read-side composite ([#51])

Ergonomic spatial selection lands on every composite that returns
slabs:

| Filter | Semantics |
|---|---|
| `nearest_to(point, component=…)` | Single nearest entity to the query point |
| `in_box(box_min, box_max, …)` | Half-open on the upper side: `[box_min, box_max)` so adjacent boxes don't double-count shared faces. Use `np.inf` to relax an axis |
| `in_sphere(center, radius, …)` | Closed ball |
| `on_plane(point_on_plane, normal, tolerance, …)` | Absolute distance ≤ tolerance |

Available on `results.nodes` plus the six element-level composites
(`results.elements`, `.elements.gauss`, `.line_stations`, `.fibers`,
`.layers`, `.springs`) via the shared `_ElementGeometryMixin`.
Element-side queries use centroids computed lazily from the FEM's
node coordinates + per-type connectivity, robust to mixed-type
meshes.

- **Filters compose additively.** Spatial primitives intersect with
  each named selector (`pg=` / `label=` / `selection=` / `ids=` /
  `element_type=`) the same way:
  ```python
  results.nodes.in_box(
      (-1, -1, 0), (1, 1, 5),
      component="displacement_z", pg="Top",
  )
  ```
- **`element_type=` selector** on every element-level composite.
  Restricts the candidate set by broker element-type name
  (`"Tet4"`, `"Hex8"`, `"Quad4"`, etc.). Resolves via
  `fem.elements.types` and `fem.elements.resolve(element_type=…)`.
- **Verbose parameter names per project preference**: `point` (was
  `xyz`), `box_min` / `box_max` (was `p_min` / `p_max`), `center`,
  `radius`, `point_on_plane`, `normal`, `tolerance`.
- New user-facing guide:
  [`guide_results_filtering.md`](internal_docs/guide_results_filtering.md)
  — 12 sections covering the composite tree, selectors menu,
  geometric helpers, additive composition, slab shapes, time
  slicing, stage scoping, discovery API, worked recipes, pitfalls,
  and what's queued.
- 21 spatial tests + 18 existing composite tests pass.

### ADDED — Elastic-beam line-stations synthesis ([#47])

The Phase 11b live-capture path previously required a force- or
disp-based beam-column section.force probe; closed-form elastic
beams (`ElasticBeam{2d,3d}`, `ElasticTimoshenkoBeam{2d,3d}`,
`ModElasticBeam2d`) were silently dropped because they have no
integration points. The MPCO read path already synthesises a
2-station slab from the `localForce` bucket — this commit ports the
same synthesis to the live `DomainCapture` path.

- New `synthesize_line_station_layout_for_elastic_beam(class_name)`
  in [`solvers/_element_response.py`] — builds a 2-station
  `ResponseLayout` at ξ ∈ {-1, +1} for any class with a
  `NODAL_FORCE_CATALOG` entry, using canonical line-station
  component names. Companion
  `is_line_station_synthesis_catalogued` predicate.
- `_LineStationGroup` gains a `mode` field; `_probe_element` splits
  into `_probe_section_force` (existing) and
  `_probe_local_force_synthesis` (new).
- `_step_local_force` reads `ops.eleResponse(eid, "localForce")` once
  per step and applies the standard sign flip on station 2 so the
  slab matches section-force convention (mirrors the MPCO path).
- **Loud skip warning.** `DomainCapture.end_stage` now emits a single
  consolidated `UserWarning` if any elements were dropped from
  line-stations recording, listing the breakdown by reason and a
  sample of element IDs. Steers the user to MPCO or to rebuilding
  as `ForceBeamColumn` instead of letting the silent skip turn into
  a confusing empty diagram.
- `examples/EOS Examples/example_buckleUP_v2.ipynb` gains a
  `recs.line_stations(...)` call so its `elasticBeamColumn` braces /
  columns / beams produce a real line-stations slab the
  `LineForceDiagram` can render.

### ADDED — Recorder vocabulary discovery ([#50])

The recorder vocabulary is now discoverable from three surfaces, all
sharing one source of truth:

- **Method docstrings** on `Recorders.{nodes, elements, line_stations,
  gauss, fibers, layers, modal}` enumerate every canonical component,
  every shorthand expansion, the selector vocabulary, the cadence
  options, coverage caveats per execution strategy, and a worked
  example. mkdocstrings auto-renders these on
  [`api/opensees.md`](api/opensees.md), so the API page now shows
  the full menu.
- **Static introspection methods**:
  ```python
  Recorders.categories()
  Recorders.components_for(category)
  Recorders.shorthands_for(category)
  ```
  Useful at the REPL, in notebooks, for validation messages.
- **New reference page** at *Guides › Results › Recorder reference*
  ([`guide_recorders_reference.md`](internal_docs/guide_recorders_reference.md))
  — single-page menu with categories at a glance, shared selectors
  and cadence, then per-method tables of components and shorthands.
- 16 new introspection tests; full recorder sweep at 158 passing.

### DOCS — 6-card landing, grouped nav, examples gallery ([#49], [#52], [#44], [`affa81d`])

- **6-card landing** ([#49]) replaces the 2-card grid on
  the docs home page. Cards are organised by user
  intent: First steps, Quickstart & Examples, Build a model, Run &
  read results, Architecture, API reference. Plus a 3-card
  "What's new" band.
- **Grouped navigation** ([#49]). `mkdocs.yml` reorganises the
  bloated Guides and Architecture sections into collapsible
  sub-headings: *Getting started* / *Building models* / *Physics* /
  *Solver bridge* / *Results* / *Reference* under Guides;
  *Foundations* / *Subsystems* / *Gmsh background* under
  Architecture. No content moves.
- **Curated examples gallery** ([#49]) — 8 EOS notebooks rendered
  inline via mkdocs-jupyter at `/examples/notebooks/<name>/`.
  `docs/_hooks.py` copies notebooks from `examples/EOS Examples/`
  to `docs/examples/notebooks/` on every build (mtime-aware so
  incremental rebuilds are fast); source of truth stays in
  `examples/EOS Examples/`.
- **8 hero notebooks modernised** ([#52]) — single-flow
  apeGmsh-Results pedagogical template:
  1. Imports + parameters
  2. Geometry (apeGmsh)
  3. Physical groups
  4. Mesh
  5. OpenSees model — vanilla openseespy
  6. Declare recorders — apeGmsh
  7. Run analysis with `spec.capture(...)` *or*
     `spec.emit_recorders(...)`
  8. Read results back via `Results`
  9. Plot in-notebook
  10. Optional viewer (subprocess, `APEGMSH_SKIP_VIEWER` honoured)

  Strategy assignments mixed across the curriculum:
  `01_hello_plate`, `04_portal_frame_2D`, `05_labels_and_pgs`,
  `12_interface_tie`, `17_modal_analysis`, `19_pushover_elastoplastic`
  use `spec.capture`; `02_cantilever_beam_2D`, `10b_part_assembly`
  use `spec.emit_recorders`. All 8 verified end-to-end via
  `nbconvert --execute` against closed-form solutions.
- **Gallery page refresh** (`affa81d`). Each card calls out the
  strategy used (`spec.capture` vs `spec.emit_recorders`), names the
  verification target, and points at the specific pedagogical
  moment that notebook teaches that others don't. New "Common
  shape across every notebook" section makes the unified template
  visible at the gallery level.
- **31 EOS notebooks wired** ([#44]) with a "Capture results"
  section before `g.end()`, providing two pedagogical paths:
  manual `NativeWriter` and declarative
  `Recorders().nodes(...) → DomainCapture` context. New
  `scripts/wire_eos_notebook.py` (auto-wiring) and
  `scripts/migrate_eos_legacy_api.py` (API-drift migrator) for
  future notebooks. Both paths gate the viewer launch on
  `APEGMSH_SKIP_VIEWER` for headless / nbconvert / CI runs.
  Notebooks with pre-existing breakage moved to `to_review/` with a
  README explaining each.

### Test coverage

| PR | New tests | Notes |
|----|----------:|-------|
| [#43] | — | Reorganises existing test infra (B0–B4 widget tests track the migration) |
| [#46] | ~30 | HUD construction + callbacks, density manager, style presets CRUD, picker pre-flight |
| [#47] | 4  | Elastic-beam round trip 3D / 2D, skip-warning fires once, clean-recording no-warning |
| [#48] | 46 | Recorder/live/mpco sweep at 495 passing |
| [#50] | 16 | Static introspection — categories / components_for / shorthands_for |
| [#51] | 21 | nearest_to / in_box / in_sphere / on_plane / element_type semantics + additive intersection |
| **Total** | **~117** |  |

[#43]: https://github.com/nmorabowen/apeGmsh/pull/43
[#44]: https://github.com/nmorabowen/apeGmsh/pull/44
[#46]: https://github.com/nmorabowen/apeGmsh/pull/46
[#47]: https://github.com/nmorabowen/apeGmsh/pull/47
[#48]: https://github.com/nmorabowen/apeGmsh/pull/48
[#49]: https://github.com/nmorabowen/apeGmsh/pull/49
[#50]: https://github.com/nmorabowen/apeGmsh/pull/50
[#51]: https://github.com/nmorabowen/apeGmsh/pull/51
[#52]: https://github.com/nmorabowen/apeGmsh/pull/52

---

## v1.2.0 — Results viewer: Gauss contour, scrubber animation, shape-function catalog

Six PRs landing on top of v1.1.0's Results rebuild. `ContourDiagram`
gains two new rendering paths so element-level Gauss data flows
straight through the dialog → diagram → substrate pipeline; the time
scrubber gets real Play / FPS / loop controls; the shape-function
catalog grows from 5 to 12 element types so quadratic and prism
meshes are first-class. **All 6 PRs merged green** — 377 viewer
tests + 1876 non-viewer tests pass on main.

PRs in this release: [#35], [#36], [#37], [#38], [#39], [#42].

### ADDED — `ContourDiagram` Gauss paths

- **Element-constant Gauss contour** ([#37]). New `gauss_cell` rendering
  path activated when `n_gp == 1` per element (CST / tri31, hex8 with
  one-point integration, etc.). Reads via
  `results.elements.gauss.get(component=...)`, paints `cell_data` on
  a substrate submesh extracted by element IDs. Removes the manual
  nodal-averaging step the plate notebook was using.
- **GP→nodal extrapolation for higher-order integration** ([#39]).
  New `gauss_node` path activated when `n_gp > 1`. The slab is
  projected onto the linear-corner shape functions via the
  Moore–Penrose pseudo-inverse (`pinv(N)`), then accumulated into a
  per-node sum + count for cross-element averaging. New module
  [`apeGmsh.results._gauss_extrapolation`] with two public entry
  points: `extrapolate_gauss_slab_to_nodes(slab, fem)` and
  `per_element_max_gp_count(slab)`.
- **`ContourStyle.topology` field** ([#37]). User-facing knob with
  three values:
    * `"auto"` (default) — prefer nodal data when both composites have
      the requested component; fall through to Gauss otherwise.
    * `"nodes"` — force the nodal-scalar path (point data).
    * `"gauss"` — force the Gauss path; cell-vs-node sub-decision is
      made internally based on `n_gp`.
- **Topology dropdown in the Add Diagram dialog** ([#38]). Visible
  only when the selected kind is Contour. The Component combo
  populates from the union of nodes + gauss components under
  `"auto"`, and from the picked composite under `"nodes"` / `"gauss"`.

### ADDED — Time scrubber animation ([#36])

- **Play button** drives a `QTimer` at `1000 / fps` ms; each tick
  advances one step via `director.set_step(...)`. The scrubber stays
  slider-passive — it only updates the slider via the Director's
  `on_step_changed` callback, never directly.
- **FPS spinner** (1–60, default 30). Live — changing it while
  playing updates the timer interval without disturbing the run.
- **Loop modes** combo: `"once"` (stop at last step), `"loop"`
  (wrap to step 0), `"bounce"` (reverse direction at boundaries,
  never wraps).
- **Stops on stage change** automatically; a fresh stage may have
  a different step count, so the scrubber refreshes and waits for
  the user to press Play again.

### ADDED — Shape-function catalog expansion ([#42])

`SHAPE_FUNCTIONS_BY_GMSH_CODE` grows from 5 entries to 12. New types
covering everything you'd hit by setting
`gmsh.model.mesh.setOrder(2)` plus `wedge6` for extruded / layered
meshes:

| Code | Type   | Notes                                          |
|------|--------|------------------------------------------------|
| 6    | wedge6 | Linear prism (tri × line tensor product)       |
| 9    | tri6   | Quadratic triangle                             |
| 10   | quad9  | Lagrangian biquadratic quad                    |
| 11   | tet10  | Quadratic tet                                  |
| 12   | hex27  | Lagrangian triquadratic hex                    |
| 16   | quad8  | Serendipity quadratic quad                     |
| 17   | hex20  | Serendipity quadratic hex                      |

Node orderings match Gmsh's published convention (cross-checked
against the ASCII diagrams in `gmsh-4.15.1/src/geo/M{Triangle,
Quadrangle,Tetrahedron,Hexahedron,Prism}.h`), so a connectivity row
read straight from a Gmsh-generated mesh works without any
reordering. Pyramids (`pyr5` / `pyr14`) and `line3` are deliberately
out of scope — pyramids have a known apex singularity worth avoiding
for a first pass, and `line3` is rare in OpenSees output.

For GP→nodal extrapolation the higher-order types fall back to
their **linear counterpart** (tri6 → tri3, quad8/9 → quad4,
tet10 → tet4, hex20/27 → hex8). Reasons: the substrate is built
from linear cells (mid-side / face / center nodes are dropped in
`build_fem_scene`), so non-corner extrapolations are never painted;
and `pinv` on the full higher-order N matrix produces a
non-constant nodal field for a constant GP input
(minimum-norm regularization of the under-determined system),
which is wrong for visualization.

### REFACTORED — single-source diagram topology routing ([#35])

- New `Diagram.topology: str` class attribute; each subclass declares
  it next to `kind`. The Add Diagram dialog's `_KIND_TO_TOPOLOGY`
  table is now derived from those attributes:
  ```python
  _KIND_TO_TOPOLOGY = {
      entry.kind_id: entry.diagram_class.topology for entry in _KINDS
  }
  ```
  Previously the dict was hand-maintained alongside the per-class
  composite-reader calls and could drift.

### CHANGED — `ContourDiagram` internals

- Three internal effective-topology values replace the previous two:
  `"nodes"` / `"gauss_cell"` / `"gauss_node"`. Dispatch is decided
  at attach time after a single step-0 read used both for the n_gp
  probe and the initial scatter.
- Cross-element discontinuities are smoothed by the nodal averaging
  in the `gauss_node` path. Standard post-processor behaviour
  (STKO, ParaView). A future per-element subdivision path can
  preserve discontinuities — out of scope here, the `gauss_cell`
  scaffolding stays in place to keep that door open.

### Test coverage

| PR | New tests | Notes |
|----|----------:|-------|
| [#35] | 2  | Pinning test for the eight kind→topology mappings |
| [#36] | 10 | State-machine + timer + stage-change-stop coverage |
| [#37] | 10 | Auto resolution, attach, in-place mutation, multi-GP rejection (later refactored to extrapolation in [#39]) |
| [#38] | 11 | Visibility per kind, component listing per topology, end-to-end run() spec construction |
| [#39] | 11 | Linear-field round-trip on hex8 + 2×2×2 GPs (`atol=1e-12`), shared-face averaging, time-axis preservation, in-place mutation on the new path |
| [#42] | 37 | 5 invariants × 7 types: Kronecker delta, partition of unity, linear precision, dN-sum, FD cross-check |
| **Total** | **81** |  |

### Known follow-ups (not scheduled)

- **Discontinuity-preserving Gauss contour** — subdivides each
  multi-GP element into linear sub-cells, samples at sub-vertices,
  renders cell-data per sub-cell. Preserves jumps at material
  interfaces. ~500–800 LOC, design-discussion-first decision.
- **Hex27 face/center node ordering** — verified self-consistent in
  the shape-function math, but the assumed Gmsh ordering for nodes
  20–26 isn't independently validated against a real Gmsh hex27
  mesh. One-row fix in `_HEX27_LAGRANGE_INDEX` if a real mesh
  surfaces a mismatch.
- **Pyramid shape functions** — `pyr5` and `pyr14`. Add when needed.

[#35]: https://github.com/nmorabowen/apeGmsh/pull/35
[#36]: https://github.com/nmorabowen/apeGmsh/pull/36
[#37]: https://github.com/nmorabowen/apeGmsh/pull/37
[#38]: https://github.com/nmorabowen/apeGmsh/pull/38
[#39]: https://github.com/nmorabowen/apeGmsh/pull/39
[#42]: https://github.com/nmorabowen/apeGmsh/pull/42

## v1.1.0 — Results: backend-agnostic FEM post-processing system rebuild

Wholesale rebuild of the `apeGmsh.results` module. The legacy
in-memory `Results` carrier (a thin VTK-feeder bound to live numpy
arrays) is replaced by a lazy disk-backed reader plus a unified
composite API that mirrors `FEMData`. Recording flows through three
execution strategies — Tcl/Py recorders, in-process domain capture,
and an MPCO bridge — all driven by one declarative
`g.opensees.recorders` spec. **987 tests pass** end-to-end including
the El Ladruno OpenSees Tcl subprocess integration. See PR #12 and
`internal_docs/Results_architecture.md` for full design.

### ADDED — `apeGmsh.results`

- **Native HDF5 schema + I/O.** `NativeWriter` / `NativeReader`
  round-trip nodes, Gauss points, fibers, layers, line stations, and
  per-element forces. Stages are first-class (`kind="transient"` /
  `"static"` / `"mode"`). Multi-partition stitching transparent to
  the reader. Embedded FEMData snapshot in `/model/` — including
  `MeshSelectionStore` — so result files are self-contained.
- **`Results` composite class** mirroring `FEMData`. Selection
  vocabulary `pg=` / `label=` / `selection=` / `ids=`. Stage scoping
  via `results.stage(name)`; mode access via `results.modes`.
  Soft FEM coupling with hash-validated `bind()`.
- **`compute_snapshot_id(fem)`** deterministic content hash —
  ties recorder specs ↔ result files ↔ FEMData snapshots.
- **MPCO reader.** `Results.from_mpco(path)` reads existing STKO
  `.mpco` files through the same composite API. Partial FEMData
  synthesis from MPCO `MODEL/` group (nodes + elements +
  region-derived PGs).
- **`g.opensees.recorders`** declarative spec composite.
  Standalone class, no parent ref, no gmsh dependency. `.nodes`,
  `.elements`, `.line_stations`, `.gauss`, `.fibers`, `.layers`,
  `.modal` declaration methods. `spec.resolve(fem, ndm, ndf)`
  expands shorthand components, validates per category, locks
  `fem_snapshot_id`.
- **Three execution strategies driven by the spec:**
  - **Strategy A** — `g.opensees.export.tcl/py(..., recorders=spec)`
    emits `recorder Node/Element ...` commands + HDF5 manifest
    sidecar. `Results.from_recorders(spec, output_dir, fem=fem)`
    parses output `.out` files into native HDF5 with a cache layer
    at `<project_root>/results/`.
  - **Strategy B** — `with spec.capture(path, fem) as cap:` wraps
    an openseespy analyze loop, querying `ops.nodeDisp` etc. per
    record. Multi-record merge with NaN-fill when records target
    disjoint node sets. `cap.capture_modes()` writes one mode-kind
    stage per `ops.eigen` mode.
  - **Strategy C** — `recorders_file_format="mpco"` dispatches to
    a single `recorder mpco` line aggregating all records.
    Validated via subprocess against the El Ladruno OpenSees Tcl
    build.
- **Element capability flags** on `_ElemSpec`: `has_gauss`,
  `has_fibers`, `has_layers`, `has_line_stations` plus a
  `supports(category)` helper. All 16 entries in `_ELEM_REGISTRY`
  annotated.
- **`MeshSelectionStore`** name-based lookups: `names()`,
  `node_ids(name)`, `element_ids(name)` — mirrors
  `PhysicalGroupSet`'s API.
- **Architecture doc** `internal_docs/Results_architecture.md`
  (single canonical reference).

### CHANGED — Results module API (BREAKING)

- **`Results.from_fem(fem, point_data=..., cell_data=...)` removed.**
  Use `Results.from_native(...)`, `from_mpco(...)`,
  `from_recorders(...)`, or hand-construct via `NativeWriter` for
  the in-memory case.
- **`fem.viewer()`** raises `NotImplementedError` until the viewer
  rebuild project. The new flow will go through the rebuilt
  composite API.
- **`g.mesh.viewer(point_data=..., cell_data=...)`** raises
  `NotImplementedError`. The mesh-only paths (`g.mesh.viewer()` and
  `g.mesh.viewer(results=path)` for a `.vtu`/`.pvd` file) still
  work unchanged.
- **Public exports** under `apeGmsh.results`: `Results`,
  `ResultsReader`, `NativeReader`, `MPCOReader`, `ResultLevel`,
  `StageInfo`, `BindError`, and the slab dataclasses
  (`NodeSlab`, `ElementSlab`, `LineStationSlab`, `GaussSlab`,
  `FiberSlab`, `LayerSlab`).

### DEFERRED — element-level transcoding

Element-level records (`gauss` / `fibers` / `layers` /
`line_stations` / `elements`) work end-to-end on the **declaration
and emission** side. The **read/decode** side is stubbed:

- `MPCOReader.read_gauss/fibers/layers/...` returns empty slabs.
- `RecorderTranscoder` skips element records silently.
- `DomainCapture.step()` raises `NotImplementedError` for element
  categories.

All three need the same missing piece: a per-element-class
response-metadata catalog. Plan in
`internal_docs/plan_element_transcoding.md` (Phase 11a).
**Nodal results work everywhere today.**

### NEW FIXTURE

- `tests/fixtures/results/elasticFrame.mpco` — 400 KB binary,
  12 nodes / 11 elastic frame elements / 10 transient steps /
  2 model stages. Used by the MPCO reader + integration tests.

---

## v1.0.9 — Viewer: higher-order rendering + filter overhaul (WIP)

Lands the viewer fixes and refactor scaffolding from PR #11.  Higher-
order elements (Q8/Q9, Tri6, Tet10, etc.) no longer render as VTK's
sub-triangle tessellation fans.  The dim filter actually hides actors
now, and node display scopes to the dim filter.  Step 5 (corner /
midside / bubble node differentiation) is deferred to the next release.

### FIXED — viewer

- **Q9 / higher-order surface fill** — the fill layer is now built from
  linearized corner-only cells (`mesh_scene.GMSH_LINEAR`), so a Q9 quad
  renders as a single quad and a Tri6 as a single triangle.  31 element
  types covered including P3 / P4 and bubble variants; unknown types
  warn instead of being silently dropped.
- **Dim filter (1D/2D/3D checkboxes)** — was overridden in
  `mesh_viewer._on_mesh_filter` setup so it only set the pick mask;
  now also flips fill / wire / node-cloud actor visibility per dim.
- **Phantom wireframe on Reveal** — `VisibilityManager._rebuild_actors`
  now rebuilds the wire actor alongside the fill on hide / reveal, so
  hidden entities lose their edges and revealed entities regain them.
- **BRep surface fill for higher-order meshes** — `brep_scene` got the
  same linearization treatment for Tri6 / Quad8 / Quad9.

### CHANGED — viewer

- New **wireframe layer** built via `extract_all_edges()` per dim>=2,
  registered as `EntityRegistry.dim_wire_actors`.  Replaces VTK's
  built-in `show_edges` (which rendered the higher-order cell
  tessellation, not the FE element boundary).  Clipping plane,
  visibility manager, and dim filter all participate.
- **Per-dim node clouds** — single global `node_actor` replaced by
  `EntityRegistry.dim_node_actors` keyed by dim, each containing the
  nodes used by entities of that dim (with `includeBoundary=True`).
  The dim filter now scopes node display: unchecking 1D drops 1D-only
  nodes, but boundary nodes shared with a visible 2D dim stay.
- **Tree right-click Hide / Isolate / Reveal-all** — added to BRep
  `SelectionTreePanel` and `BrowserTab` (group + entity rows).  Backed
  by new `VisibilityManager.hide_dts(dts)` / `isolate_dts(dts)`
  programmatic counterparts of the pick-driven `hide()` / `isolate()`.

### ADDED — `viewers.core.visibility` doc

- Spelled out the **filter state model** in the module docstring:
  cosmetic dim toggle (`SetVisibility`), entity hide
  (`VisibilityManager._hidden`), and clipping (render-time) are three
  independent layers, intentionally not unified.

## v1.0.8 — Embedded-node constraint resolver (`ASDEmbeddedNodeElement`)

Closes Phase 11b.  Replaces the `NotImplementedError` on
`ConstraintsComposite._resolve_embedded` with a working resolver, so
embedded-rebar and similar non-conforming inclusions can be expressed
without fragmenting the host mesh.

### ADDED — `solvers/_constraint_resolver.py`

- `_barycentric_tri3(p, p0, p1, p2)` — barycentric coordinates of `p`
  inside a 2D triangle, with projection onto the triangle's plane for
  off-plane points.
- `_barycentric_tet4(p, p0, p1, p2, p3)` — same for a 3D tetrahedron.
- `ConstraintResolver.resolve_embedded(...)` — given embedded nodes and
  host element connectivity, locates each embedded node in its host
  via KD-tree spatial indexing + barycentric coordinates, then emits
  `InterpolationRecord` shape-function couplings that match
  `ASDEmbeddedNodeElement` kinematics.

### ADDED — integration

- `ConstraintsComposite._resolve_embedded` now dispatches to the new
  resolver, collects host element connectivity from a labeled master
  region (tri3 / tet4), filters out embedded nodes that coincide with
  host corners, and returns the coupling records.
- `examples/EOS Examples/15_embedded_rebar.py` rewritten to use the
  embedded path instead of the old fragment-based conformal rebar.

### ADDED — tests

- `tests/test_constraint_resolver.py` — 4 cases (tri3 interior + corner,
  tet4 centroid, multi-element search).

### ADDED — regression coverage

- `tests/test_target_resolution.py` — locks in `FEMData.nodes.get()` /
  `.elements.get()` `target=` precedence (`label > PG`) and raw
  `[(dim, tag)]` passthrough.
- `tests/test_boolean_ops.py` — guards the 2D opt-in
  `fragment(cleanup_free=True)` bug so it can't regress (and now also
  pins that the default `cleanup_free=False` preserves surfaces).
- `tests/test_parts_advanced.py` — covers `g.parts.add(part, label=...,
  translate=...)` on an unlabeled Part (no sidecar).

### ADDED — infrastructure

- `pyproject.toml` `[tool.pytest.ini_options]` with
  `pythonpath = ["src"]`, so pytest run from a worktree picks up the
  worktree's source instead of the editable install pointing at the
  main checkout.

---

## v1.0.7 — Selection upgrades + `set_transfinite_box`

Polish pass on the v1.0.6 selection API.  Eliminates the hand-rolled
patterns that kept showing up in scripts (two-step boundary queries,
`_apply_hex` helpers, manual node-count-by-axis loops) and adds the
predicates and combinators users were reaching for.

### ADDED — boundary helpers

- `g.model.queries.boundary_curves(tag)` — returns every unique
  curve on the boundary of an entity.  Encapsulates the two-step
  query (faces → individual face boundaries with `combined=False` →
  deduplicate) that's needed because Gmsh's `getBoundary(vol,
  recursive=True)` skips dim=1 and goes straight to vertices.
  Accepts a label, PG name, int tag, dimtag, or list of any.
- `g.model.queries.boundary_points(tag)` — symmetric helper for
  the eight corner points of a volume.

### ADDED — `select()` upgrades

- `select()` now accepts a label string with a `dim=` keyword:
  `select('box', dim=2, on={'z': 0})` resolves the label, walks
  to dim 2, and applies the predicate — no manual `boundary()`
  call beforehand.
- `not_on=` and `not_crossing=` negation predicates.  Same
  signed-distance computation as the positive forms; useful for
  *all faces except the bottom* style queries.
- The `Selection.to_label()` call on a mixed-dim selection no
  longer triggers the labels-composite collision warning — using
  the same name across multiple dims is the documented intent
  here.

### ADDED — `Selection` ergonomics

- Set operations: `selection | other` (union with deduplication),
  `selection & other` (intersection), `selection - other`
  (difference).  All three preserve the back-reference to
  `_Queries` so chaining keeps working.
- `selection.partition_by(axis=None)` groups entities by their
  dominant bounding-box axis.  Returns a `dict[str, Selection]`
  keyed by `'x'`, `'y'`, `'z'`, or — if `axis=` is given — a
  single `Selection`.  Semantics are dim-aware:
  - **curves** group by the *largest* extent (curve direction);
  - **surfaces** group by the *smallest* extent (perpendicular /
    normal direction).

### ADDED — primitive factories

- `g.model.queries.plane(z=0)`, `plane(p1, p2, p3)`, and
  `plane(normal=..., through=...)` build a `Plane` you can pass to
  any `select(on=..., crossing=...)` call (positive or negated).
- `g.model.queries.line(p1, p2)` builds a `Line`.
- Define a primitive once, reuse across many selections — useful
  when the same cutting plane appears in several queries.

### ADDED — `set_transfinite_box`

- `g.mesh.structured.set_transfinite_box(vol, *, size=None, n=None,
  recombine=True)` — collapses the full transfinite-hex setup
  (curve node counts, face transfinite + recombine, volume
  transfinite) into a single call.  Accepts either `size=` (target
  element size; node counts derived per edge from
  `round(length / size) + 1`) or `n=` (uniform node count per
  edge).  Pass `recombine=False` for a transfinite tet mesh
  instead of hex.

### CHANGED

- `examples/EOS Examples/22_geometric_selection.ipynb` rewrites
  the 3-D section to use `set_transfinite_box`,
  `select('box', dim=2, ...)`, the `plane()` factory,
  `not_on=`, set operations, `partition_by`, and chained
  `to_label / to_physical` — every v1.0.7 feature is exercised.

### FIXED

- `g.model.queries.bounding_box(tag, dim=N)`,
  `center_of_mass(tag, dim=N)`, and `mass(tag, dim=N)` now honour
  `dim` as an explicit hint when ``tag`` is a bare integer.
  Previously these went through `resolve_to_single_dimtag` →
  `resolve_dim`, which always searches dimensions 3 → 0 and
  returns the first match — so on a model containing both volume
  1 and curve 1, `bounding_box(1, dim=1)` silently returned the
  volume's bounding box.  Bare ints are now passed straight to
  the corresponding Gmsh OCC call at the requested dim.  String
  labels and `(dim, tag)` tuples still go through resolution.
- `g.model.geometry.slice` now passes its plane reference as an
  explicit `(2, plane_tag)` dimtag to the downstream
  `cut_by_surface` / `cut_by_plane` calls.  Previously it passed
  a bare int, which triggered `resolve_dim` to scan the live
  Gmsh model — and because `add_axis_cutting_plane` is called
  with `sync=False`, the new plane wasn't yet visible to
  `getEntities(2)`, causing the resolver to fall through to the
  curves and fail with `"surface ref N resolved to dim=1"`.
  Together these two fixes recover ≈14 previously-failing tests
  in `test_geometry_cutting`, `test_sections`, and
  `test_part_anchors`.

### INTERNAL

- New `_Queries._resolve_to_dimtags(tag)` helper consolidates the
  string / int / dimtag resolution path used by `boundary_curves`,
  `boundary_points`, and the new `select()` label-string branch.
- `_select_impl` now takes `not_on` / `not_crossing` and inverts
  the predicate result (`hit ^ invert`).  The four kwargs are
  mutually exclusive — exactly one must be passed.
- `Selection` is parameterised on `DimTag` (a `tuple[int, int]`
  alias).  Method signatures use proper type hints throughout.
- 29 new test cases in `tests/test_selection.py` covering the new
  helpers, the negation predicates, set operations, `partition_by`
  for both curves and surfaces, the primitive factories, and
  `set_transfinite_box`.  Total: 55 cases passing.

---

## v1.0.6 — Geometric selection API (`g.model.queries.select`)

### ADDED

- `g.model.queries.select(entities, on=..., crossing=...)` filters
  curves, surfaces, or volumes by a geometric predicate. Replaces
  the noisy `entities_in_bounding_box(xmin,ymin,zmin,xmax,ymax,zmax)`
  pattern with a readable description of *what* you want.
- Predicates work on the bounding-box corners of each candidate:
  - `on=` — every corner within `tol` of the primitive (entity lies
    on it).
  - `crossing=` — corners exist on both sides of the primitive
    (entity straddles it). Same signed-distance computation
    underlies both.
- Primitive formats — no imports needed:
  - `{'z': 0}` → axis-aligned plane z = 0.
  - `[(p1), (p2)]` (2 points) → infinite line, for cutting 2-D
    geometry.
  - `[(p1), (p2), (p3)]` (3 points) → infinite plane through 3
    non-collinear points. Use for surfaces and volumes.
- `select()` returns a `Selection` — a `list` subclass with three
  chainable methods:
  - `.select(...)` — filter further (AND logic when stacked).
  - `.to_label(name)` — register every entity as a label, grouped
    by dimension.
  - `.to_physical(name)` — register every entity as a physical
    group, grouped by dimension.
  Each returns `self` so you can keep chaining: `select → label →
  select again → physical`.
- `Selection.__repr__` describes the count by dimension and reminds
  the user how to chain — IDE autocomplete + the repr are the only
  discovery surface needed.
- New curriculum notebook
  `examples/EOS Examples/22_geometric_selection.ipynb` walks the
  full workflow: predicate intro → stacking → unstructured baseline
  → transfinite quad mesh of a plate → 3-D hex of a box.
- Companion script
  `examples/example_unstructured_and_transfinite.py` shows the
  unstructured-vs-transfinite contrast for two adjacent boxes.
- API docs page extended at `docs/api/model.md` with `Selection`,
  `Plane`, and `Line` (the latter two documented as internal but
  exposed so the format reference is auto-generated from
  docstrings).

### INTERNAL

- New module `src/apeGmsh/core/_selection.py` holds `Plane`, `Line`,
  the `_parse_primitive` dispatcher, the `_select_impl` core, and
  the `Selection` class. `Plane` and `Line` are not part of the
  public API — they are only constructed by `_parse_primitive` from
  raw user input passed to `select()`.
- `Selection` carries a back-reference to the originating `_Queries`
  so `.select()`, `.to_label()`, and `.to_physical()` can route to
  the session's `labels` / `physical` composites without the user
  having to thread context.
- Tests in `tests/test_selection.py` (26 cases) cover predicates in
  2-D and 3-D, primitive parsing (including degenerate / collinear /
  coincident input), the Selection chain, label / PG registration
  for both single-dim and mixed-dim selections, and error paths.

---

## v1.0.5 — Line loads with `normal=True` (radial / curve-perpendicular pressure)

### ADDED

- `g.loads.line(..., normal=True)` applies a pressure perpendicular
  to each edge instead of along a fixed direction.  Useful for
  internal/external pressure on curved 2-D boundaries (Lamé-style
  problems, fluid-loaded arcs, etc.) where the cartesian direction
  varies along the curve.
- Default sign comes from Gmsh's surface boundary orientation —
  `gmsh.model.getBoundary([(2, surface)], oriented=True)` tells the
  composite which side of the curve the structure sits on, so
  `magnitude > 0` always pushes *into* the structure (matching
  `g.loads.surface(..., normal=True)`).
- Optional `away_from=(x0, y0, z0)` reference point overrides the
  Gmsh path — flips the in-plane normal so it points away from that
  point.  Use for ambiguous cases (a curve bounding two surfaces, or
  a free curve not bounding any surface) or when you want to be
  explicit.
- Both `reduction="tributary"` (default) and `reduction="consistent"`
  honour `normal=True`.
- New worked example
  `examples/EOS Examples/example_plate_pyGmsh_v2.ipynb` rewrites the
  thick-walled-cylinder Lamé problem on top of the new API —
  replaces ~30 lines of manual consistent-force integration on the
  inner arc with a single `g.loads.line(pg='Pressure', magnitude=p,
  normal=True)` declaration.

### INTERNAL

- New resolver methods `LoadResolver.resolve_line_per_edge_tributary`
  and `resolve_line_per_edge_consistent` accept a list of
  `(edge, q_xyz)` items so the composite can pre-compute per-edge
  force-per-length vectors (which vary along curved boundaries).
  Constant-direction line loads still use the original
  `resolve_line_*` methods unchanged.
- Per-edge normal computation lives in `LoadsComposite` (it needs
  Gmsh queries for the boundary-orientation path); the resolver
  stays pure-math.

---

## v1.0.4 — Low-level booleans preserve Instances + accept label/PG refs

### FIXED

- `g.model.boolean.{fuse,cut,intersect,fragment}` now keep
  `Instance.entities` consistent when called directly on tags that
  happen to belong to a tracked Instance.  Previously the remap only
  ran inside `g.parts.fragment_all` / `fragment_pair` / `fuse_group`,
  so a low-level boolean left Instance entries pointing at consumed
  tags.  The remap-from-result walk has been extracted into
  `PartsRegistry._remap_from_result` and every OCC boolean call site
  (both `_bool_op` and the Parts-level methods) now routes through
  that single implementation.

### ADDED

- `g.model.boolean.*` accepts label names and user physical-group
  names in `objects=` / `tools=`, matching the input shape of
  `g.physical.add`.  Strings resolve via the shared resolver: label
  (Tier 1) first, then user PG (Tier 2).  Raw tags, dimtags, and
  mixed lists still work.

### INTERNAL

- New `resolve_to_dimtags` helper in `apeGmsh.core._helpers` —
  companion to `resolve_to_tags` that emits `(dim, tag)` pairs.
  Handles labels / PGs that span multiple dimensions without the
  caller having to coerce a single dim.
- Plan B (`Instance.entities` as a computed label-backed property)
  was weighed against this conservative fix and deferred; see
  `internal_docs/plan_instance_computed_view.md` for the signals that
  would trigger revisiting it.

---

## v1.0.0 — Clean Architecture (breaking)

v1.0 bundles two breaking changes: the package rename and the Model
composition refactor. A full find-replace migration guide is at
[`internal_docs/MIGRATION_v1.md`](internal_docs/MIGRATION_v1.md).

### BREAKING

- **Package renamed**: `pyGmsh` → `apeGmsh`
  - `from pyGmsh import pyGmsh` → `from apeGmsh import apeGmsh`
  - `class pyGmsh(_SessionBase)` → `class apeGmsh(_SessionBase)`
  - Companion app `apeGmshViewer` keeps its name (only its internal
    imports of our theme module were updated)

- **Model methods split into five sub-composites** (composition replaces
  mixin inheritance):
  - `g.model.geometry.*` — 19 primitive builders (add_point, add_line,
    add_box, add_cylinder, etc.)
  - `g.model.boolean.*` — fuse, cut, intersect, fragment
  - `g.model.transforms.*` — translate, rotate, scale, mirror, copy,
    extrude, revolve, sweep, thru_sections
  - `g.model.io.*` — load/save STEP, IGES, DXF, MSH, heal_shapes
  - `g.model.queries.*` — bounding_box, center_of_mass, mass, boundary,
    adjacencies, entities_in_bounding_box, remove, remove_duplicates,
    make_conformal, registry
  - `g.model.sync()`, `g.model.viewer()`, `g.model.gui()`,
    `g.model.launch_picker()`, `g.model.selection` stay flat on Model

- **Rename `g.mass` → `g.masses`** for consistency with the other
  plural composites (`g.loads`, `g.parts`, `g.physical`, `g.constraints`,
  `g.mesh_selection`)
  - `fem.mass` → `fem.masses`
  - Class names (`MassesComposite`, `MassSet`, `MassDef`, `MassRecord`)
    unchanged

- **Removed legacy aliases**:
  - `g.initialize()` / `g.finalize()` → use `g.begin()` / `g.end()`
  - `g._initialized` → use `g.is_active`
  - `g.model_name` → use `g.name`

- **Removed deprecated methods**:
  - `g.model.viewer_fast()` / `g.mesh.viewer_fast()` → use `viewer()`
    (always fast now)
  - `g.parts.add_physical_groups()` → explicit
    `g.physical.add_volume(inst.entities[3], name=...)`
  - `g.opensees.add_nodal_load()` → use `g.loads.point()` in a
    `g.loads.case()` block
  - `g.mesh_selection.add_nodes(nearest_to=...)` → `closest_to=`

- **Removed convenience delegates on the session**:
  - `g.remove_duplicates()` → `g.model.queries.remove_duplicates()`
  - `g.make_conformal()` → `g.model.queries.make_conformal()`

- **Removed property on `_SessionBase`**:
  - `_parent.model_name` → `_parent.name`

### FIXED

- Pylance / static analyzers no longer lose track of Model methods.
  Composition makes every method statically discoverable through
  the sub-composite classes (`_Geometry`, `_Boolean`, `_Transforms`,
  `_IO`, `_Queries`), each of which is a concrete class with explicit
  methods. No MRO walking across 5 mixin files.

### INTERNAL

- `_GeometryMixin` → `_Geometry`
- `_BooleanMixin` → `_Boolean`
- `_TransformsMixin` → `_Transforms`
- `_IOMixin` → `_IO`
- `_QueriesMixin` → `_Queries`

Each sub-composite now takes a `model` reference in `__init__` and
accesses Model state via `self._model._log(...)`,
`self._model._register(...)`, `self._model._as_dimtags(...)`,
`self._model._registry`, instead of inheriting state.

### MIGRATION

See [`internal_docs/MIGRATION_v1.md`](internal_docs/MIGRATION_v1.md) for the complete
find-replace table and an automated migration script.

**v0.3.0** is the last `pyGmsh` release (pre-rename safety tag).
**v0.3.1** is the transitional `apeGmsh` release (rename only).
**v1.0.0** is the new architecture (composition + cleanups).

---

## v0.3.1 — Package rename (transitional)

- Renamed `pyGmsh` package directory to `apeGmsh/` on disk
- All internal imports, class name, config entries updated
- Examples and docs still reference old API (deferred to v1.0)
- Safety release — rename is isolated from the architectural
  refactor that follows in v1.0

## v0.3.0 — Last pyGmsh release (safety tag)

Safety checkpoint before the package rename and v1.0 refactor. This
is the final tag under the `pyGmsh` name. If you need the old API,
pin to this version.

---

## v0.2.x — Loads, Masses, Viewer overlays

- New `g.loads` composite with pattern context managers
- New `g.mass` composite (renamed to `g.masses` in v1.0)
- `fem.loads` / `fem.mass` auto-resolved by `get_fem_data()`
- Loads/masses/constraints overlays live on `g.mesh.viewer(fem=...)` —
  they are mesh-resolved concepts and never landed on `g.model.viewer()`

## v0.2.0 — Composites architecture

- Assembly absorbed into `apeGmsh` as composites:
  - `g.parts` (PartsRegistry)
  - `g.constraints` (ConstraintsComposite)
- MeshSelectionSet + \_mesh\_filters spatial query engine
- Viewer rebuild: BRep / mesh viewers unified around EntityRegistry,
  PickEngine, ColorManager, VisibilityManager
- Catppuccin Mocha theme across all viewers
