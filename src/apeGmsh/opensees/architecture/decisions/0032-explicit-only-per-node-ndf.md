# ADR 0032 — Explicit-only per-node `ndf` via top-level `g.node_ndf` composite

**Status:** Accepted (May 2026). Codifies the explicit-only doctrine
introduced in PR #317 and hardened by PR #321. Sibling to the
[ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) lineage chain
and the [ADR 0023](0023-per-zone-schema-versioning.md) two-version
reader window; informs future foreign-format adapters covered by
[ADR 0026](0026-h5modelreader-protocol-contract.md).

## Context

The OpenSees solver assigns a per-node DOF count (`ndf`) at node
creation time — `ops.node tag x y z -ndf K`. Until S1b, apeGmsh stored
only a single model-wide `ndf` on the `OpenSeesModel` envelope; every
emitted node inherited that value. The assumption worked for the
homogeneous-DOF models apeGmsh historically targeted (3D frame:
`ndf=6` everywhere; 3D continuum: `ndf=3` everywhere), but it falls
over the moment a single broker carries nodes that need different DOF
counts.

The S1a shell-on-solid conformity work (PR #306) is the first such
case to ship: a shell region sits on top of a continuum region, the
two meshes are made conformal via `fragment_all()`, and the shared
nodes are simultaneously corners of `ShellMITC4` elements (which need
three displacement + three rotational DOFs) and corners of
`SSPbrickUP` / `stdBrick` elements (which need only three
displacement DOFs). OpenSees represents this by declaring the shared
nodes at `ndf=6`; the solid elements see the extra rotational DOFs
as ghost slots, the shell elements see all six. With one
model-wide `ndf`, the broker has no way to declare such a node
correctly — bumping the whole model to `ndf=6` works for shell-on-
solid but silently inflates DOF count on every other 3D continuum
model.

The S1b work introduces a per-node `ndf` array on the FEM broker.
The remaining design question — addressed by this ADR — is **how
that array is populated**. The candidate spectrum spans
*implicit-by-dim* (a `{1: 6, 2: 6, 3: 3}` table picks defaults from
element dim) through *hybrid* (implicit defaults + an explicit
override hook) to *explicit-only* (every node covered by a user
declaration; uncovered nodes fail loud). This ADR records the
principle that drives the choice and the explicit-only API the
broker ships.

## Decision

Per-node `ndf` is a **user contract**, not an inferable property.
Every node that the user expects to emit through OpenSees must be
covered by an explicit declaration. The declarations live on a new
top-level composite `g.node_ndf`, sibling to `g.constraints` /
`g.loads` / `g.masses` in the apeGmsh session.

### API surface

```python
# Uniform-ndf models — exactly one call.
g.node_ndf.set_default(ndf=3)

# Shell-on-solid mixed-ndf — one targeted call + one default.
g.node_ndf.set_default(ndf=3)              # solids
g.node_ndf.set("ShellRegion", ndf=6)       # shells override the default

# Targeted-only — every node must be in the target.
g.node_ndf.set("Trusses", ndf=3)
```

`g.node_ndf.set(target, *, ndf, name=None)` accepts the same target
shapes the load and mass composites do: a label name, a physical
group name, a part label, a raw `[(dim, tag), ...]` list, or a
mesh-selection name. Targets are resolved at FEM-build time via the
shared `core._resolution.resolve_target` helper; unresolvable targets
raise `KeyError` (the dimensional resolution contract).

`g.node_ndf.set_default(ndf=K)` declares the fallback for every node
not covered by a targeted call. Re-calling replaces the default
in-place (does not append a second default).

`g.node_ndf.list()` returns the registered defs in declaration order;
`g.node_ndf.clear()` drops every def (warning if the cache is built).

### Fail-loud surface

At FEM-build time, the factory walks the declarations and writes
resolved values into a per-node `int8` array (`NodeComposite._ndf`),
sentinel `0` for undeclared. The broker query is `fem.nodes.ndf_for(nid)`;
on the sentinel it raises:

```
LookupError: node 42: ndf not declared — call
  g.node_ndf.set(target, ndf=K) covering this node, or
  g.node_ndf.set_default(ndf=K) for the uniform case.
```

The message names both fixes so the user picks the right one without
re-reading the docs. The first emit of an undeclared node throws —
the bug surfaces immediately at model construction time, not hours
later in a converged-but-wrong analysis.

### Composite at the top level

`g.node_ndf` is a sibling of `g.constraints` / `g.loads` / `g.masses`,
not a sub-namespace of `g.model`. The composite owns its own def list,
its own validators, and its own H5 round-trip; it is not a thin shim
over the OpenSees envelope value. Future ndf-related APIs (per-element
class overrides, per-material defaults, per-stage activation toggles)
hang off `g.node_ndf` without needing to wedge new vocabulary into
`g.model`.

## Rationale

**Implicit defaults silently lie.** Any dim-keyed default table —
`{line: 6, surface: 6, volume: 3}` is the obvious candidate — fills
in numbers the user did not author. A 3D truss model would receive
`ndf=6` on every chord (because the element is line-dim); the
analyst expecting `ndf=3` displacement-only nodes never sees a
diagnostic. An embedded-mesh joint where a beam tip lands inside a
solid host gets `ndf=6` for the beam neighbour and `ndf=3` for the
host — at the shared node the table cannot decide between them.
Element dim alone does not carry enough information to resolve mixed
cases; any dim-keyed table quietly picks the wrong one.

**Explicit-only matches the existing architectural precedent.** The
constraints, loads, and masses composites all require the user to
declare per-region — there is no `IMPLICIT_LOAD_BY_DIM` table that
infers point-loads from surface elements. The dimensional resolution
contract codified in `tests/test_resolution_contract.py` is the
existing law of the broker: name → dim-scoped-entity, fail loud on
missing coverage, never silently fall back. `ndf` follows the same
law. The user owns intent; the broker owns the mechanics of resolution.

**Fail-loud surfaces missing declarations where they cost the least
to fix.** The `LookupError` fires the moment a downstream consumer
asks for `ndf_for(nid)`. The error message names the user's actual
model (`node 42`, not `internal entity 0x7fa0...`) and names both
fixes (`set` for partial coverage, `set_default` for uniform). The
alternative — silent default plus runtime divergence — is the bug
class that took the longest to find in pre-S1b mixed-ndf prototypes.

## Consequences

**Every model that emits to OpenSees needs at least one `g.node_ndf`
call.** Uniform models need exactly one
(`g.node_ndf.set_default(ndf=K)`); mixed-ndf models need one default
plus per-region overrides. Models that bypass apeGmsh's broker (e.g.
direct `FEMData(...)` construction in tests) carry `_ndf = None` and
fall back to the bridge's model-wide `ndf` — the sentinel-undeclared
LookupError fires only when an explicit declaration was started but
not completed.

**The FEM hash (per [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md))
folds the resolved `_ndf` array.** Identical geometry with different
ndf declarations produces different `fem_hash` values; the lineage
chain detects ndf-only changes the same way it detects geometry
changes. The fold is gated on `_ndf is not None` so legacy / direct-
test FEMs preserve their digest.

**H5 schema 2.6.0 → 2.7.0.** The neutral zone gains an optional
`/nodes/ndf` int8 dataset. Per the
[ADR 0023](0023-per-zone-schema-versioning.md) two-version reader
window, readers at neutral=2.7.0 accept 2.6.x files (no `/nodes/ndf`
present → `_ndf = None`, falls back to bridge envelope) and 2.7.x
files (dataset present, length must match `/nodes/ids` or
`MalformedH5Error`). The bump is additive; no migration is required
for files written before S1b.

**`g.node_ndf` is the home for future ndf-related APIs.** Per-element-
class overrides, per-material defaults, and per-stage activation
toggles (relevant to [ADR 0030](0030-stage-bound-topology-activation.md))
all belong on this composite. Future widenings stay self-contained
inside `core/NodeNDFComposite.py` and `mesh/_fem_factory.py::_populate_node_ndf`
without touching `g.model` or the OpenSees bridge.

**Foreign-format readers conforming to the
[ADR 0026](0026-h5modelreader-protocol-contract.md) `H5ModelReader`
Protocol** must surface per-node `ndf` if their format encodes it
(LS-DYNA `d3plot` carries it; xDMF and Exodus typically do not), or
declare the sentinel-undeclared fallback. The Protocol's
`nodes()` method already returns a `Mapping[str, Any]` covering
`ids` / `coords`; adapters may add an `ndf` key when present. A
viewer that opens such a reader and finds no `ndf` key applies the
bridge envelope value, mirroring the apeGmsh 2.6.x back-compat
behaviour.

**Post-extraction mutations warn.** `g.node_ndf.set(...)` after
`g.mesh.queries.get_fem_data()` does not retroactively rewrite the
cached FEM. The composite warns on the first such call and clears
its post-extract flag so a batch of re-declarations does not produce
N redundant warnings; the next `get_fem_data()` re-stamps the flag.
The warning text names the re-extract path so the user knows how to
make the declaration take effect.

## Related

- [ADR 0021](0021-lineage-chain-replaces-snapshot-id.md) — lineage
  chain. The `_ndf` fold into `fem_hash` is gated by the
  presence-aware rule INV-1 establishes; ndf-only changes produce
  lineage warnings under the warn-not-raise rule.
- [ADR 0023](0023-per-zone-schema-versioning.md) — per-zone schema
  versioning. The neutral-zone bump 2.6.0 → 2.7.0 for the additive
  `/nodes/ndf` dataset is the canonical example of the additive-minor
  rule + two-version reader window.
- [ADR 0026](0026-h5modelreader-protocol-contract.md) — `H5ModelReader`
  Protocol. Foreign-format adapters surface per-node ndf through the
  same Protocol's `nodes()` method when their format encodes it.
- `src/apeGmsh/core/NodeNDFComposite.py` — the composite shipped by
  PR #317 (hardening in PR #321).
- `src/apeGmsh/_kernel/defs/node_ndf.py` — the `NodeNDFDef`
  dataclass.
- `src/apeGmsh/mesh/_fem_factory.py::_populate_node_ndf` — the
  resolver that walks the def list at FEM-build time.
- `src/apeGmsh/mesh/FEMData.py::NodeComposite.ndf_for` — the
  fail-loud broker query.
- `tests/test_node_ndf.py` — 15 tests covering fail-loud, override
  + default, targeted-only uncovered raise, H5 round-trip + hash,
  2.6.0 forward-compat, length validation, hash regression, post-
  extraction warn, from_msh sentinel, resolver KeyError propagation,
  and API guards.
- `tests/test_resolution_contract.py` — the existing dimensional
  resolution contract this ADR aligns ndf with.
- PR #317 — top-level `g.node_ndf` composite + fail-loud `ndf_for`
  + H5 schema 2.7.0 + 15 new tests.
- PR #321 — hardening: `_fem_built` reset at the top of each build,
  `clear()` post-extract warning semantics, hash declaration-order
  invariance test, real 2.6.0 back-compat test.
