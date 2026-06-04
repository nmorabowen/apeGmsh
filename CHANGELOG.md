# Changelog

## Unreleased — shell-on-solid conformity (S1a + S1b + S2 + S5) · Phase SSI-2.D stage-bound BCs and recorders · embedded-element pipeline hardening (#329 / #331) · ASDEmbeddedNodeElement option exposure (ADR 0035) · stage-bound constraints + `s.initial_stress` PUSH (Phase SSI-2.D extension) · **Phase SSI-2.E between-stage Domain mutators** · topology safety nets (P1/P3) + arc-line wire docs · embedded-host decomposition (ADR 0036) · **higher-order line broker split (ADR 0037)** · RecorderDeclaration element fan-out fix · **orphan-geometry sweep unification + `g.model.geometry` validation API** · **split-sweep auto-validation (closed-world / open-world)** · **raw-PG channel for `_user_intentional`** · **`g.model.geometry.add_arch` (apex-as-vertex two-arc arch)** · **damping definition `ops.damping` / `s.damping` (ADR 0053, D1–D5)** · **Ladruno J2 plasticity materials (`LadrunoJ2` / `LadrunoUniaxialJ2` / `LadrunoJ2Finite`)** · **Ladruno material wrappers (`LogStrain` / `InitDefGrad` / `StagedStrain` / `LadrunoRebarBuckling`)** · **Ladruno live Monitor recorder (`ops.recorder.Monitor` + `read_monitor` / `tail_monitor`)**

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
