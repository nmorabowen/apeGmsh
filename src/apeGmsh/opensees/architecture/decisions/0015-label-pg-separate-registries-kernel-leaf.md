# ADR 0015 — Tier-1 labels and Tier-2 physical groups are separate registries; `apeGmsh/_kernel` is a downward-only leaf

**Status:** Accepted (selection-unification-v2 Phase P2-I, May 2026)

## Context

The selection-unification-v2 work
([docs/plans/selection-unification-v2.md](../../../../../docs/plans/selection-unification-v2.md))
collapses ~11 divergent terminal selection types, two classes both
named `Selection`, four name-resolvers and six spatial copies into
**one fluent idiom with two terminals**: `EntitySelection` (CAD
dimtags) and `MeshSelection` (node / element ids). P1-K (ADR-adjacent;
the keystone relayer) relocated the pure data / algorithm layer into a
new root-leaf package `apeGmsh/_kernel`. P2-I introduces the two v2
terminals on that kernel and repoints the five `.select()` host hooks
to return them.

Two structural decisions made during P2-I are load-bearing enough to
record as an ADR, because a future refactor that "simplifies" either
one would silently reintroduce a class of bug the v2 design exists to
remove:

1. **`EntitySelection` exposes two distinct promotion terminals** —
   `.to_label(name)` and `.to_physical(name)`. They write to two
   *different* registries with different identity semantics. The
   temptation (they both "name a set of entities") is to unify them
   into one registry or one method. That is exactly the merge the v2
   mandate forbids.
2. **`apeGmsh/_kernel` is a leaf with a one-directional dependency
   edge.** P1-K froze the import-DAG polarity (the latent
   `core ↔ mesh` cycle survives only because the cross-package edges
   have a specific eager/deferred polarity). P2-I's new module
   `mesh/_mesh_selection.py` must not be the seam that re-inverts it.

### Tier-1 vs Tier-2 — the concrete divergence

`g.labels` (Tier-1, `session.labels.add`) and `g.physical` (Tier-2,
`session.physical.add`) are **both** backed by Gmsh physical groups,
but Tier-1 names are stored with a `_label:` prefix
(`apeGmsh/_kernel/_label_prefix.py`: `LABEL_PREFIX`, `is_label_pg`,
`add_prefix`, `strip_prefix` — relocated there by P1-K, HT4). The
prefix is not cosmetic: it is what makes Tier-1 labels survive boolean
geometry operations (fragment / fuse / cut) with stable identity,
while Tier-2 raw physical groups have the unprefixed user name and the
raw-PG lifetime.

Because the prefix namespaces them, a user name `Clash` can
legitimately exist **simultaneously** as:

- `(d, 'Clash')`        — a Tier-2 raw physical group, and
- `(d, '_label:Clash')` — a Tier-1 boolean-op-stable label.

These are *different sets with different lifetimes*. Silently merging
them (e.g. "a `to_physical('Clash')` should also satisfy a
`label='Clash'` lookup", or one combined registry) destroys the
Tier-1 boolean-op identity guarantee — the bug class HT2/HT3 and the
resolution contract (`tests/test_resolution_contract.py`,
`tests/test_target_resolution.py`) were written to lock out.

## Decision

### 1. Tier-1 labels and Tier-2 physical groups are separate registries, never merged

`EntitySelection.to_label(name)` delegates **per dimension** to
`session.labels.add(d, tags, name=name)` (Tier-1, `_label:`-prefixed,
boolean-op-stable). `EntitySelection.to_physical(name)` delegates
**per dimension** to `session.physical.add(d, tags, name=name)`
(Tier-2, raw gmsh physical group). They are **two distinct terminals
on the same selection** and write to **two distinct registries**. No
code path merges, aliases, or cross-resolves the two. The resolution
contract's tier order (label → physical group → part) keeps them
ordered but **never collapsed**; `EntitySelection` re-implements none
of that tier logic — it delegates verbatim to the contract-locked
resolver, exactly as the legacy terminal did.

`EntitySelection` additionally exposes `.to_dataframe()`. This is
**new** (the legacy `core/_selection.Selection` has no `to_dataframe`;
only the frozen `viz/Selection` does). It is implemented **locally**
in `core/_selection.py` — it does **not** import `apeGmsh.viz`. It
mirrors `viz/Selection.py`'s column set (`dim, tag, kind, label, x, y,
z, mass`) using the gmsh bounding-box centre + `gmsh.model.occ.getMass`
and the session label reverse-map reached through the chain's engine
(`self._engine._model._parent`). Reaching `viz` from `core` would add
an eager cross-package edge the import-DAG tripwire forbids (see
decision 2).

### 2. `apeGmsh/_kernel` is a downward-only leaf

`apeGmsh/_kernel` depends only on the standard library, numpy, gmsh,
and the leaf-pure sibling `apeGmsh.fem` (HT10). Every other package
(`core`, `mesh`, `viz`, `results`) imports **downward into**
`_kernel`; `_kernel` never imports back. The
`tests/test_import_dag_polarity.py` tripwire — its `PKGS` widened to
include `_kernel` and `fem` (P0-T, HT5/HT10) — freezes the exact set of
eager cross-package edges and fails on any new one.

P2-I's new module `mesh/_mesh_selection.py` (the home of
`MeshSelection`) obeys this by construction: its only **module-level**
import is `from .._kernel.chain import SelectionChain`. Everything
engine-specific — the relocated payloads and the four legacy chains it
delegates to for byte-faithful per-engine behaviour — is imported
**deferred inside method bodies**, identical to the established
`mesh/_node_chain.py` idiom. The result is exactly **one** new frozen
BASELINE triple, `('mesh', '_kernel', 'mesh/_mesh_selection.py')` — the
same downward polarity already frozen for every other `mesh→_kernel`
leaf. No `core ↔ mesh` triple is added; no previously-deferred edge is
flipped eager. The five host-hook repoints reuse the existing
deferred-import idiom, so the `results→mesh` / `mesh→results` delegate
imports are function-body-only and invisible to the eager tripwire.

`EntitySelection` is defined **beside** `GeometryChain` in
`core/_selection.py` (same module, no new module, no new import edge).
`MeshSelection` is the **only** new leaf and it points one way: down.

## Alternatives considered

1. **One registry / one promotion method.** Make `.to_label` and
   `.to_physical` write to a single namespace, or expose only a
   `.promote(name)`. Rejected — destroys the Tier-1 boolean-op-stable
   identity (the `_label:` prefix exists precisely so `Clash` can be a
   raw PG *and* a stable label at once). This is the HT2/HT3 bug class
   the resolution contract locks; merging would silently fail under
   any boolean geometry op.

2. **`EntitySelection.to_dataframe()` imports `viz/Selection`'s
   implementation** (don't duplicate the column logic). Rejected —
   `core → viz` is an eager cross-package edge the import-DAG polarity
   tripwire forbids; it would risk the FP-1 cycle the whole P1-K
   relayer exists to delete. Re-implementing the (small, gmsh-only)
   column build locally in `core` is the same trade the viewer made in
   ADR 0014 (`viewers/data/_records.py` duplicates field names, not
   logic, to keep the package boundary clean).

3. **`MeshSelection` re-implements each engine's coordinate /
   centroid / materialise logic** instead of delegating to the four
   legacy chains. Rejected for P2-I — four hand-rewritten copies risk
   silent semantic drift from the legacy behaviour P2-I must preserve
   invisibly. Delegating to a freshly-constructed legacy chain of the
   engine-appropriate class makes the per-engine behaviour
   byte-faithful *by construction* (it literally runs the legacy
   code). The legacy chains stay defined-and-importable through P2-I;
   P3 deletes them and folds their logic into `_kernel/spatial.py` as
   reviewed pinned flips (out of P2-I scope).

4. **Put `MeshSelection` in an existing `mesh` module** (e.g. extend
   `mesh/_node_chain.py`). Rejected — it would entangle the v2 terminal
   with a legacy chain module that P3 deletes, and muddy the
   single-new-leaf import story. A dedicated leaf with one downward
   edge is the cleaner, separately-reviewable unit.

## Consequences

**Positive:**

- The Tier-1 / Tier-2 identity guarantee is ADR-locked: a future
  "let's just merge labels and physical groups" must read this ADR and
  the resolution contract first. The two terminals stay independently
  meaningful (`to_label` for boolean-op-stable named sets, `to_physical`
  for raw solver-facing groups).
- The `_kernel` leaf boundary is ADR-locked as downward-only. The
  import-DAG tripwire enforces it mechanically every commit; this ADR
  records *why* the single P2-I BASELINE triple is the only allowed
  delta and why `EntitySelection` lives in `core/_selection.py` rather
  than a new module.
- `MeshSelection`'s delegate-to-legacy mechanism gives a concrete
  invisibility proof (`tests/test_p2i_parity.py`): the v2 terminal and
  the still-present legacy chain are asserted equivalent on
  `_items` / set-algebra / spatial / `.result()` / `.values()` for all
  four point host contexts and the entity context.

**Negative:**

- `EntitySelection.to_dataframe()` duplicates `viz/Selection`'s column
  set (names, not logic). The duplication is the deliberate cost of
  the package-boundary invariant (same trade as ADR 0014). If the
  column contract changes it must change in both places; that risk is
  covered by the parity / idiom tests.
- Through P2-I the four legacy point chains and `GeometryChain` remain
  defined (unwired) solely so `MeshSelection` / `EntitySelection` can
  delegate / mirror them and the parity proof can run. They are dead
  weight until **P3** deletes them; this is an intentional, time-boxed
  transient (the v2 plan sequences removal last and gated, never folded
  into the invisible relayer).
- `.save_as(name)` on `MeshSelection` is only functional on the
  live-mesh engine context. The mutable mesh-selection store
  (`g.mesh_selection` / `MeshSelectionSet`, whose `add()` →
  `_snapshot()` → `MeshSelectionStore` → FEMData HDF5 round-trips as
  `selection=`) is reachable **only** from the live-mesh engine
  (`_LiveMeshEngine.ms`). The broker-node / broker-element / results
  engines hold only the immutable read-only `MeshSelectionStore`
  snapshot (no registration surface) and are routinely detached /
  import-origin objects with no live gmsh session. `.save_as` is
  therefore **present-but-loud** on those engines (the
  `EntitySelection.in_box` `inclusive=`→`TypeError` precedent: an
  explicit fail, never a silent no-op or a reinvented parallel store).
  The legacy `MeshSelectionChain` had no `.save_as` at all, so this is
  strictly additive and breaks no P2-I parity.

## References

- [docs/plans/selection-unification-v2.md](../../../../../docs/plans/selection-unification-v2.md)
  — the hardened plan; §1 (two terminals), §3 (HT2/HT3/HT4/HT5/HT10),
  §4 (architecture), §5 (R-v2-3 the label/PG non-merge), §6 P1-K /
  P2-I, §6.1 (P2 pre-flight corrections / STOP-2).
- [tests/test_import_dag_polarity.py](../../../../../tests/test_import_dag_polarity.py)
  — the eager cross-package edge tripwire; `_kernel` + `fem` in `PKGS`;
  the single P2-I BASELINE triple.
- [tests/test_resolution_contract.py](../../../../../tests/test_resolution_contract.py),
  [tests/test_target_resolution.py](../../../../../tests/test_target_resolution.py)
  — the dimensional / tiered resolution contract the
  label-vs-physical-group separation upholds (byte-unchanged through
  P1-K and P2-I).
- [decisions/0014-viewer-is-pure-h5-consumer.md](0014-viewer-is-pure-h5-consumer.md)
  — the same package-boundary "duplicate field names, not logic"
  trade, on the viewer side.
- [decisions/0013-records-in-mesh-not-solvers.md](0013-records-in-mesh-not-solvers.md)
  — the producer-side relocation precedent for breaking a cross-package
  cycle by moving pure types to a neutral layer.
