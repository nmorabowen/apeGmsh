# ADR 0033 — S2: Wire per-node `ndf` into OpenSees emit paths

**Status:** Accepted (2026-05-24). Ships in branch
`feat/s2-ndf-emit-wiring` (PR TBD-S2-PR). Extends
[ADR 0032](0032-explicit-only-per-node-ndf.md) (the S1 broker contract
this consumer wires). Related to
[ADR 0021](0021-lineage-chain-replaces-snapshot-id.md),
[ADR 0022](0022-mp-constraint-emission-fanout.md),
[ADR 0019](0019-opensees-model-read-side-broker.md), and
[ADR 0026](0026-h5modelreader-protocol-contract.md).

## Context

[ADR 0032](0032-explicit-only-per-node-ndf.md) codified the per-node
`ndf` user contract: explicit declarations via `g.node_ndf.set(...)`
and `g.node_ndf.set_default(...)`, fail-loud `fem.nodes.ndf_for(nid)`
on undeclared nodes. The broker stored the data; the OpenSees emit
layer ignored it. Every `ops.node(...)` call passed the bridge's
model-wide envelope `OpenSeesModel.ndf`, regardless of what the user
declared per-region. Mixed-ndf models (shell on solid: shell nodes
need `ndf=6` for rotational DOFs; solid-only nodes need only `ndf=3`)
were correct in apeGmsh's broker but wrong in the emitted OpenSees
deck. S2 closes that gap without breaking any of the ~285 existing
`apeSees(fem)` test sites and example notebooks that declare a single
envelope `ndf` and expect every node to inherit it.

## Decision

**Override-only semantics.** `g.node_ndf` is the override channel;
the OpenSees model envelope (`apeSees(fem).model(ndm, ndf=K)`) is the
default. The emitter passes `-ndf K` to `ops.node(...)` only when the
broker has a non-sentinel value at that nid; sentinel slots emit
without `-ndf` and OpenSees applies the envelope. This mirrors
OpenSees-native semantics — `model BasicBuilder -ndf K` sets the
per-node default; per-node `-ndf J` is the override.

**`ndf_for(nid)` stays fail-loud.** Per
[ADR 0032](0032-explicit-only-per-node-ndf.md)'s contract,
`fem.nodes.ndf_for(nid)` raises `LookupError` when the slot is
sentinel or `_ndf is None`. The emitter wraps the call in
`try/except LookupError`; on miss it emits without `-ndf`, letting
the envelope win. This keeps `NodeComposite` decoupled from
`OpenSeesModel` and preserves the absence-vs-declared distinction at
the broker boundary.

**Validator at three sites.** `apeSees.model()`,
`OpenSeesModel.from_compose_buffers()`, and `OpenSeesModel.from_h5()`
all assert `envelope >= max(non-sentinel _ndf)` via
`apeGmsh.opensees._internal.build.validate_envelope_covers_broker_ndf`.
On mismatch, raise `BridgeError` naming the offending node and the
fix (raise `ndf` in `apeSees(fem).model(...)`). Three sites because
`OpenSeesModel` materialises through three distinct paths (live
session via `apeSees(fem)`, compose-buffer publish, H5
deserialisation); each must guard against a downsized envelope
silently truncating per-node declarations.

**Phantom-node carveout.** Broker-synthetic phantom nodes (per
[ADR 0022](0022-mp-constraint-emission-fanout.md) INV-3 — emitted
before constraints under MP so the foreign-rank tag exists) are never
in `fem.nodes.ids` and never resolve through `g.node_ndf`. They
continue to emit with hardcoded `ndf=6`. The H5 emitter previously
used `ndf is not None` as the phantom-vs-real-node discriminator —
that signal is now ambiguous (real broker nodes legally carry `ndf=`
under S2), so the phantom-emit helpers in
`apeGmsh.opensees._internal.build` flip an explicit
`_in_phantom_node_emit` side-channel flag (see
`apeGmsh.opensees._internal.tag_resolution.set_phantom_node_mode`)
around their node calls. The H5 emitter consults the flag to
populate `/opensees/constraints/phantom_node_tags`; other emitters
ignore it. This is a broker-internal exception to the explicit-only
doctrine; the user never authors phantom-node declarations.

**OpenSeesMP consistency is hash-guaranteed.** Per
[ADR 0021](0021-lineage-chain-replaces-snapshot-id.md), the resolved
`_ndf` array folds into `fem_hash`. Every rank deserialises the same
broker, so all ranks agree on per-node `ndf` for shared nodes
without explicit cross-rank communication.

**`from_msh` carries `_ndf=None`** (revert of PR #321's zero-stamping).
Combined with `_hash_nodes` skipping the fold when `_ndf is None` OR
all-sentinel, hash symmetry is preserved across construction paths
AND the emit layer's envelope fallback works on `.msh`-loaded models.

## Consequences

**Zero user-facing migration cost.** Existing scripts that never
touched `g.node_ndf` emit byte-identical OpenSees decks — sentinel
slots elide `-ndf`, envelope is unchanged. The ~285 existing
`apeSees(fem)` test sites and every example notebook keep working
without any rewrite. Red team flagged this risk during design review;
the override-only resolution mirrors what
[ADR 0032](0032-explicit-only-per-node-ndf.md) §Consequences already
documented as the broker contract's downstream behaviour.

**Mixed-ndf shell-on-solid models now emit correctly** under both
single-process and OpenSeesMP paths. The four owned-node emit sites
in `apesees.py` (flat global, flat staged-owned, partitioned global,
partitioned staged-owned) and the foreign-node site in
`build.py::emit_mp_constraints_partitioned` all route through the
single helper `_emit_node_with_broker_ndf`. The
`OpenSeesModel.build('tcl'|'py'|'live'|'h5')` replay path widens its
per-node tuple to `(tag, coords, ndf|None)` so the per-node ndf
declarations survive an H5 round-trip without truncating to the
envelope.

**`OpenSeesModel.ndf` semantics shift** from "uniform per-node value"
to "envelope default for nodes without override." Existing accessors
(`.ndf` property) keep returning the envelope scalar — there is no
behavioural change for callers that already treat it as the model
default.

**Misconfigured envelopes fail at the call site, not deep in emit.**
A user who calls `apeSees(fem).model(ndf=3)` after declaring
`g.node_ndf.set("Shells", ndf=6)` sees `BridgeError` at the `model()`
call, naming the offending node and the fix. The same guard fires
again on `OpenSeesModel.from_h5(...)` and
`OpenSeesModel.from_compose_buffers(...)` so corrupted H5 files and
programmatic compose flows that bypass `apeSees.model()` also fail
loud.

**`_replay_into` (compose H5 path) consults the broker directly.**
The helper at `apeGmsh.opensees._internal.compose._replay_into` walks
`self._fem.nodes.ndf_for(tag)` per node and emits `-ndf K` only when
a declaration covers the node; legacy 2-tuple `(tag, coords)` callers
are tolerated for byte-stability of the existing replay shape.

## Related

- [ADR 0032](0032-explicit-only-per-node-ndf.md) — the broker
  contract this consumer wires. `ndf_for` stays fail-loud; the emit
  layer absorbs the `LookupError` and lets the envelope win.
- [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) — lineage
  chain. The resolved `_ndf` array folds into `fem_hash`, guaranteeing
  cross-rank consistency under OpenSeesMP without explicit
  coordination.
- [ADR 0022](0022-mp-constraint-emission-fanout.md) — MP-constraint
  emission. Phantom nodes (INV-3) are exempt from `g.node_ndf` and
  keep their hardcoded `ndf=6`; the new
  `set_phantom_node_mode` side-channel disambiguates phantom-emit
  from real-broker-emit for the H5 emitter.
- [ADR 0019](0019-opensees-model-read-side-broker.md) — read-side
  broker. The `OpenSeesModel.from_h5` and `from_compose_buffers`
  paths validate the envelope against the rehydrated broker's
  per-node ndf.
- [ADR 0026](0026-h5modelreader-protocol-contract.md) — H5ModelReader
  Protocol. Foreign-format adapters that surface per-node ndf through
  the Protocol's `nodes()` mapping flow through the same emit-side
  fallback unchanged.
- `apeGmsh.opensees._internal.build.validate_envelope_covers_broker_ndf`
  — the single validator called at the three materialisation sites.
- `apeGmsh.opensees._internal.build._emit_node_with_broker_ndf` — the
  single helper that wraps `fem.nodes.ndf_for(tag)` in `try/except
  LookupError` and elides `-ndf` on the miss.
- `apeGmsh.opensees._internal.tag_resolution.set_phantom_node_mode`
  / `is_phantom_node_mode` — the explicit phantom-emit side-channel.
- `apeGmsh.mesh._fem_factory._from_msh` — reverted to leave
  `_ndf=None`; combined with the `_hash_nodes` empty-channel gate,
  hash symmetric across `from_msh` and `from_gmsh-no-declarations`.

## References

PR `feat/s2-ndf-emit-wiring` (to be merged at `<sha>`).
