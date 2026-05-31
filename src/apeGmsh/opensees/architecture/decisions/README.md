# Architecture Decision Records (ADRs)

Each ADR captures one significant decision: the **context** (what
was at stake), the **decision** itself, the **alternatives** we
considered and rejected, and the **consequences**.

ADRs are append-only. If a later decision reverses or amends an
earlier one, write a new ADR that supersedes it; do not edit history.

| # | Title | Status |
|---|---|---|
| [0001](0001-decouple-from-gmsh-session.md) | Decouple the bridge from the gmsh session | Accepted |
| [0002](0002-typed-primitives-with-capabilities.md) | Typed primitives carry capabilities | Accepted |
| [0003](0003-namespace-api.md) | Namespace API + static typing | Accepted |
| [0004](0004-section-separated-from-material.md) | Sections live outside `material/` | Accepted |
| [0005](0005-patterns-explicit.md) | Patterns are explicit context managers | Accepted |
| [0006](0006-class-name-apesees.md) | Bridge class is `apeSees` | Accepted |
| [0007](0007-time-series-separated-from-pattern.md) | Time series live outside `pattern/` | Accepted |
| [0008](0008-three-emit-targets.md) | Three emit targets via Emitter Protocol | Accepted |
| [0009](0009-no-backwards-compat-with-solvers.md) | No back-compat with `apeGmsh.solvers` | Accepted |
| [0010](0010-csys-for-frame-orientation.md) | Orientation fields for frame orientation (originally "csys") | Accepted |
| [0011](0011-h5-as-fourth-emit-target.md) | HDF5 as a fourth emit target | Accepted |
| [0013](0013-records-in-mesh-not-solvers.md) | Resolved records live in `apeGmsh.mesh.records`, not `apeGmsh.solvers` | Accepted |
| [0014](0014-viewer-is-pure-h5-consumer.md) | `apeGmsh.viewers` is a pure `model.h5` consumer | Accepted |
| [0015](0015-label-pg-separate-registries-kernel-leaf.md) | Tier-1 labels / Tier-2 physical groups are separate registries; `apeGmsh/_kernel` is a downward-only leaf | Accepted |
| [0016](0016-selection-unification-v2-complete.md) | selection-unification-v2 complete: legacy surface removed, two terminals on one spatial kernel, two ratified capability gaps | Accepted (supersedes the P2-I-transient framing of 0015; §4 amended by 0017) |
| [0017](0017-selection-gaps-are-incomplete-unification.md) | The two v2 capability gaps are *incomplete unification*, not accepted permanent gaps — v2-native successors owed/planned | Accepted (amends 0016 §4) |
| [0018](0018-modeldata-vanilla-opensees-enrichment.md) | `ModelData` — declarative `model.h5` orientation enrichment for hand-written OpenSees | Accepted (complements 0011) |
| [0019](0019-opensees-model-read-side-broker.md) | `OpenSeesModel` — read-side broker for `model.h5`, distinct from `apeSees` | Accepted (complements 0011 and 0018) |
| [0020](0020-results-carries-opensees-model.md) | `Results` carries `OpenSeesModel` via the Composed-file pattern; viewer stays file-mediated | Accepted (preserves 0014) |
| [0021](0021-lineage-chain-replaces-snapshot-id.md) | Lineage chain replaces `snapshot_id` binding; warn-not-raise | Accepted (ratifies the May 2026 bind-contract decision) |
| [0022](0022-mp-constraint-emission-fanout.md) | MP constraint emission via `/neutral/` fan-out — closes the §3.3 deferral | Accepted (widens the `Emitter` Protocol) |
| [0023](0023-per-zone-schema-versioning.md) | Per-zone schema versioning; two-version reader window | Accepted (replaces single-envelope policy) |
| [0024](0024-emitter-protocol-widen-region.md) | `Emitter.region` Protocol widening for MPCO `pg=` filtering | Accepted (widens the `Emitter` Protocol; schema 2.8.0 → 2.9.0) |
| [0025](0025-emitter-protocol-widen-eigen.md) | `Emitter.eigen` Protocol widening for one-shot modal solves; `apeSees.eigen` bridge method + `EigenResult` | Accepted (widens the `Emitter` Protocol; no schema bump) |
| [0026](0026-h5modelreader-protocol-contract.md) | `H5ModelReader` Protocol — formalise the viewer-side model read contract; unblock director path-survivor elimination and foreign-format adapters | Accepted (May 2026; successor to 0014/0019/0020; adoption via PR7-a through PR7-d, shipped in PRs #282/#286) |
| [0027](0027-cross-partition-mp-constraints.md) | Cross-partition MP-constraint emission policy: replicate on every owning rank, broker-deterministic phantom tags, auto ParallelPlain + Mumps | Accepted (extends ADR 0022; schema 2.9.0 → 2.10.0 in P4) |
| [0028](0028-initial-stress-via-parameter-ramping.md) | Initial-stress injection via parameter ramping (Phase SSI-1): `ops.initial_stress(...)` + Protocol `addToParameter` / `step_hook_ramp` + `analyze` hook-wrap | Accepted (widens the `Emitter` Protocol; H5 archival deferred) |
| [0029](0029-staged-analysis-context-manager.md) | Staged analysis via `ops.stage(name)` context manager (Phase SSI-2.A): per-stage analysis chain + Protocol `stage_open` / `stage_close` + `loadConst`/`wipeAnalysis`/hook-list cleanup contract | Accepted (widens the `Emitter` Protocol; live execution + H5 archival deferred) |
| [0030](0030-stage-bound-topology-activation.md) | Stage-bound topology activation via `s.activate(pgs=)` (Phase SSI-2.B): per-PG ownership + Protocol `domain_change` + lowest-stage-index node ownership + cross-stage tag-identity preservation | Accepted (widens the `Emitter` Protocol; extended in Phase SSI-2.C / PR #315 with `_emit_stages_partitioned` for MP + staged combo) |
| [0031](0031-ssi-convenience-helpers.md) | SSI convenience helpers (Phase SSI-3): `convergence_confinement` (wraps `initial_stress`) and `imposed_displacement` (wraps `pattern.Plain` + `sp`) | Accepted (no Protocol widening — composes over existing primitives) |
| [0032](0032-explicit-only-per-node-ndf.md) | Explicit-only per-node `ndf` via top-level `g.node_ndf` composite | Accepted (May 2026; codifies the doctrine shipped in PR #317 and hardened by PR #321) |
| [0033](0033-s2-emit-wiring-per-node-ndf.md) | S2: Wire per-node `ndf` into OpenSees emit paths (override-only) | Accepted (May 2026; PR #325) |
| [0034](0034-stage-bound-bcs-and-recorders.md) | Stage-bound BCs and recorders (Phase SSI-2.D): `s.fix` / `s.mass` / `s.region` / `s.recorder` builder methods + five-validator surface (H1 / V1 / V2 / V3 / V4) + recorder claiming + per-stage region tag cache | Accepted (May 2026; PRs #323 / #324 / #326) |
| [0035](0035-asd-embedded-node-element-option-exposure.md) | `ASDEmbeddedNodeElement` C++ optionals (`-rot` / `-p` / `-K` / `-KP`) exposed end-to-end through the constraint pipeline | Accepted (schema 2.12.0; widens the `Emitter` Protocol) |
| [0036](0036-embedded-host-decomposition.md) | Embedded-host decomposition: linear coupling over corner nodes for quad / hex / prism / pyramid / higher-order hosts; reserved `host_coupling` keyword; four-mitigation invariant set | Accepted (May 2026; no OpenSees changes, no H5 schema bump) |
| [0037](0037-higher-order-line-broker-split.md) | Higher-order lines: broker-side split via mesh editing (`split_higher_order_lines`); bridge surfaces untouched | Accepted (May 2026; no OpenSees changes, no H5 schema bump) |
| [0038](0038-compose-model-composition.md) | `g.compose()` model composition: flat-after-merge with namespaced tag-offset import | Accepted (2026-05-26; Phase 3, builds on the three-broker refactor 0019–0023) |
| [0041](0041-chain-phase-geometry-constraints.md) | Chain-phase routing for geometry-intensive constraints (`embedded`, `tied_contact`) | Accepted (2026-05-27; 8 open questions resolved) |
| [0042](0042-render-backend-seam.md) | Rendering-backend seam: `SceneLayer` IR + `RenderBackend` Protocol — render-side mirror of ADR 0026; decouples domain logic from PyVista so `trame` web/Jupyter (and a future ParaView export adapter) drop in behind one seam | Accepted (May 2026; R-A → R-C shipped, R-D in ADR 0047) |
| [0043](0043-connectivity-graph-and-flexible-emit.md) | Connectivity graph (parts=nodes, interfaces=edges) + flexible split emit (driver + per-part sourced fragments) for model chaining; phase-bucket order as the unified deck/compose contract; monolithic (A) vs sequential/Workbench (B) coupling | **Proposed** (discussion draft; open questions drive the agenda) |
| [0045](0045-selection-and-pick-contract.md) | Selection & pick contract: `SelectionTarget` IR + `Substrate` + canonical `BBox` + `FilterController` (the `0/1/2/3/4` dim spine) + `SelectableSubstrate` Protocol + serialized `SelectionLog` over an additively-widened `PickBackend` — unifies picking across the model/mesh/results viewers, fixes volume picking + the six-bbox mess, lands above ADR 0047 | Accepted (2026-05-30; from a 13-agent design workflow; 4 forks ratified; slices S0 → S6) |
| [0047](0047-pick-backend-and-export.md) | `PickBackend` Protocol (optional capability gated by `supports_picking()`) + VTK-free pick IR + server-side web picking + `ParaViewExportBackend` — Phase R-D of the render seam; brings the last VTK-bound domain surface (picking) behind a Protocol | Accepted (2026-05-30; 4 open questions resolved; R-D.1 + shared backend shipped in #480) |
