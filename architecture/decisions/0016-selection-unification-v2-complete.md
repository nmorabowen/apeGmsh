# ADR 0016 — selection-unification-v2 complete: legacy surface removed, two terminals on one spatial kernel, two ratified capability gaps

**Status:** Accepted (selection-unification-v2 COMPLETE — P3-K / P3-R /
P3-S / P4, May 2026). **Supersedes the P2-I-transient framing of
[ADR 0015](0015-label-pg-separate-registries-kernel-leaf.md)** (its two
core Decisions — Tier-1/Tier-2 separate registries; `apeGmsh/_kernel`
downward-only leaf — **remain in force unchanged**; only 0015's
time-boxed "the legacy chains remain defined until P3 / `MeshSelection`
delegates to them / `tests/test_p2i_parity.py` is the invisibility
proof" framing is now historical).

## Context

ADR 0015 recorded two P2-I structural decisions and explicitly flagged a
transient: *"Through P2-I the four legacy point chains and
`GeometryChain` remain defined (unwired)… dead weight until **P3**
deletes them"* and *"P3 deletes them and folds their logic into
`_kernel/spatial.py`"*. The v2 plan
([docs/plans/selection-unification-v2.md](../../../../../docs/plans/selection-unification-v2.md))
sequenced that removal **last and gated**. It is now done:

- **P3-K** (behaviour-invisible) collapsed `MeshSelection`'s
  delegate-to-legacy-chains shell into a self-contained terminal and
  lifted the one shared box/sphere/plane mask kernel into the pure leaf
  `apeGmsh/_kernel/spatial.py` (4 byte-identical copies → 1).
- **P3-R** (BREAKING, owner-ratified full removal, no shim) hard-removed
  the legacy surface, migrated the 15 production callers, deleted the
  four legacy `*Chain` modules + `GeometryChain`, relocated the pure
  engines, and flipped the two `mesh/_mesh_filters.py` silent-row-0
  centroid sites to fail-loud (a reviewed pinned production+assertion
  diff).
- **P3-S** (additive) added the new-idiom spatial regression
  successors (`tests/test_pin_spatial_kernel_v2.py`) pinning the unified
  kernel's `(point,normal,tol)` plane / box boundary / element-centroid
  behaviour through `g.mesh_selection.select(...)` with frozen literals.
- **P4** (this ADR + docs/skill/memory) stops the library's own docs
  from describing the removed surface and records the two ratified
  capability gaps.

The end-state diverges from 0015's transient prose enough that, per the
append-only ADR rule (`README.md`), it is recorded as a new ADR rather
than an edit to 0015.

## Decision

### 1. ADR 0015's two core Decisions remain in force

Tier-1 labels (`session.labels.add`, `_label:`-prefixed,
boolean-op-stable) and Tier-2 physical groups (`session.physical.add`,
raw) are **separate registries, never merged**;
`EntitySelection.to_label` / `.to_physical` / `.to_dataframe` are the
unchanged distinct terminals. `apeGmsh/_kernel` remains a
**downward-only leaf** (stdlib / numpy / gmsh / `apeGmsh.fem` only); the
`tests/test_import_dag_polarity.py` tripwire still freezes the eager
cross-package edge set. v2 completion changed neither.

### 2. The legacy selection surface is removed, with no backward-compat

Hard-removed (no shim, no deprecation window — owner-ratified R-v2-1):
`fem.*.get` / `get_ids` / `get_coords` / `fem.elements.resolve` (the
selection accessors); the chain `results.*.select(...).values()` /
`ResultChain.get` path; `g.mesh_selection.add_nodes` / `add_elements` /
`from_geometric`; `g.model.queries.select` / `queries.line` /
`select_all*`; `SelectionComposite` (`g.model.selection`); the four
modules `mesh/_node_chain.py`, `mesh/_elem_chain.py`,
`results/_result_chain.py`, `mesh/_mesh_selection_chain.py` + the
`GeometryChain` class; and the `Selection` / `SelectionComposite`
**package exports**. The only standing alias is the zero-cost
`.result()` identity terminal (R-v2-2).

### 3. Two terminals on one spatial kernel; two classes retained by architecture

`g.model.select(...)` returns **`EntitySelection`** (entity family,
`(dim,tag)`); the five point-level entry points (`fem.nodes/elements`,
`results.nodes/elements`, `g.mesh_selection`) return **`MeshSelection`**
(point family, ids). Both are self-contained on the one
`apeGmsh/_kernel/spatial.py` mask kernel (`box_mask` / `sphere_mask` /
`plane_mask`); the element-centroid is per-engine and **fail-loud**
(a connectivity id absent from the node set raises `KeyError`, never a
silent row-0 substitution).

The classes **`core/_selection.Selection`** (the `.result()` /
`EntitySelection` terminal payload, R-v2-8) and **`viz.Selection`**
(the viewer pick-result type, SC-8) are **retained by architecture** —
they are internal payload / pick-result types, structurally distinct
(HT6), reached via deferred in-method imports. They are **not** removed
and **not** a backward-compat facade; only their *package exports* were
dropped. The four name-resolvers (FEMData node / element /
`_group_set._resolve`) stay **separate** (HT2/HT3) — "one engine" is the
*spatial* kernel only (R-v2-5), never the name-resolvers.

### 4. Two ratified capability gaps (the SC-12 disposition class)

Full removal removed two capabilities that have **no v2 successor**.
Per the ratified SC-12 precedent (a removed capability with no successor
is *documented as a known gap*, not re-introduced; head-resolved,
owner-informed via the P3-R PR #255 + this ADR + the user docs — no
owner re-ratification owed):

1. **Geometric-selection → named mesh-selection**
   (`g.mesh_selection.from_geometric` + `viz.Selection.to_mesh_*`):
   both ends removed; `.save_as` is live-mesh-engine-only and persists
   the current chain's ids, not a geometry round-trip. No v2 successor.
2. **The `SelectionComposite` filter grammar**
   (`g.model.selection.select_*(labels= / kinds= /
   length|area|volume_range= / predicate= / exclude_tags= / physical= /
   at_point=)`): `EntitySelection` exposes only spatial verbs + set
   algebra + `to_label`/`to_physical`/`to_dataframe`/`result` — no
   declarative filter-grammar equivalent. (The retained
   `viz.Selection.filter()` exposes a similar grammar but is the
   viewer-pick-result type, **not** a `g.model.select(...)` path.)

## Alternatives considered

1. **A deprecation shim / aliasing `fem.*.get → .select`.** Rejected —
   the project owner explicitly ratified full removal ("do it right"),
   not a compat window; v2 exists precisely to delete the
   backward-compat constraint v1 carried.
2. **Edit ADR 0015 in place to the v2-complete state.** Rejected — the
   ADR `README.md` mandates append-only ("write a new ADR that
   supersedes it; do not edit history"). 0015's two core Decisions are
   still correct and must remain readable as the P2-I record; only its
   transient framing is superseded — exactly the append-superseding case
   the rule is for.
3. **Re-introduce the `SelectionComposite` filter grammar on
   `EntitySelection`** so there is no capability gap. Rejected for v2 —
   it would re-grow the divergent surface v2 removed; the gap is the
   accepted, documented consequence of full removal (the SC-12 class),
   recorded honestly rather than papered over.
4. **Keep the four legacy chains as "dead but importable"** (0015's
   P2-I transient) indefinitely. Rejected — that transient was
   explicitly time-boxed to P2-I; carrying dead chain modules forever is
   the divergent-surface debt v2 exists to clear, and they
   collection-error the ~10 chain-test files once unwired.

## Consequences

**Positive:**

- One fluent idiom, two terminals, one spatial kernel, no import cycle,
  legacy surface gone — the v2 goal, ADR-locked complete.
- The retained-by-architecture status of `core/_selection.Selection`
  and `viz.Selection` is recorded: a future "delete the other
  Selection" must read this ADR + HT6 + R-v2-8/SC-8 first (they are
  payload / pick-result types, structurally irreconcilable, *not* a
  removed-but-forgotten facade).
- The two capability gaps are discoverable (this ADR + `docs/api/
  selection.md` "Known capability gaps" + the changelog): a user hitting
  the removed `from_geometric` / `select_*(labels=…)` finds the
  rationale and the nearest workaround, not silence.

**Negative:**

- No migration shim: code on the legacy surface breaks at once and must
  be ported per the `docs/api/selection.md` migration table. This is the
  deliberate, owner-ratified cost of removing the v1 backward-compat
  debt.
- The two capability gaps are real losses with no drop-in replacement.
  Documented, not mitigated, by design (the SC-12 class).
- `core/_selection.Selection` / `viz.Selection` remain defined but
  unexported — a reader grepping exports will not find them; this ADR is
  the pointer that they are intentionally retained internals.

## References

- [docs/plans/selection-unification-v2.md](../../../../../docs/plans/selection-unification-v2.md)
  — the hardened plan; §1 (goal), §3 (HT1–HT10), §5 (R-v2-1..R-v2-8),
  §6.2 (P3-K/P3-R/P3-S + the 2026-05-19 P3-S & P4 M-CORRECTION notes),
  §7 (invariants), §8 (out of scope).
- [docs/plans/selection-unification-v2-p3r-callers.md](../../../../../docs/plans/selection-unification-v2-p3r-callers.md)
  — the committed P3-R caller-migration contract + §0 M-CORRECTION /
  M-CORRECTION-P3S / M-CORRECTION-P4.
- [decisions/0015-label-pg-separate-registries-kernel-leaf.md](0015-label-pg-separate-registries-kernel-leaf.md)
  — the P2-I ADR whose two core Decisions this ADR keeps in force and
  whose transient framing it supersedes.
- [tests/test_resolution_contract.py](../../../../../tests/test_resolution_contract.py)
  — byte-unchanged through P3-K/P3-R/P3-S (the through-removal proof).
- [tests/test_pin_spatial_kernel_v2.py](../../../../../tests/test_pin_spatial_kernel_v2.py)
  — the P3-S new-idiom spatial regression successors;
  [tests/test_pin_spatial_v2.py](../../../../../tests/test_pin_spatial_v2.py)
  — the P3-R SC-11 `_mesh_filters` fail-loud reviewed pin.
- [tests/test_import_dag_polarity.py](../../../../../tests/test_import_dag_polarity.py)
  — the still-frozen `_kernel` downward-leaf tripwire (ADR 0015 §2).
