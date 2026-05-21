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
