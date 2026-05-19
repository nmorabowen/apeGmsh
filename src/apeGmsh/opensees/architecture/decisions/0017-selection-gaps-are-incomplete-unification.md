# ADR 0017 — the two selection-unification-v2 capability gaps are *incomplete unification*, not accepted permanent gaps (amends ADR 0016 §4)

**Status:** Accepted (owner decision, 2026-05-19). **Amends
[ADR 0016](0016-selection-unification-v2-complete.md) §4 only** — the
rest of ADR 0016 and all of ADR 0015 remain in force. Append-only:
this ADR supersedes 0016 §4's *disposition*, it does not edit 0016
(per `README.md`).

## Context

ADR 0016 §4 dispositioned the two capabilities dropped by the v2 full
removal — (1) geometric-selection → named mesh-selection
(`g.mesh_selection.from_geometric` / `viz.Selection.to_mesh_*`) and
(2) the `SelectionComposite` declarative filter grammar
(`select_*(labels=/kinds=/*_range=/predicate=/exclude_tags=/physical=/
at_point=)`) — as **"ratified capability gaps, no v2 successor"** via
the **SC-12 precedent** (a removed capability with no successor →
documented, not re-introduced; head-resolved, owner-*informed*).

The project owner has corrected the framing. v2's mandate
(`docs/plans/selection-unification-v2.md` §1) was **unification** —
collapse ~11 divergent terminals / 4 resolvers / 6 spatial copies /
two `Selection` classes into one idiom — **not capability reduction**.
The SC-12 precedent is sound for **redundant / duplicative** removals
(removing them *is* the unification; zero capability lost). It was
**over-applied** to a **unique-capability** removal (the filter
grammar): a functional regression wearing a unification costume.
Tell-tale: the filter grammar still exists on the *viewer-pick* path
(`viz.Selection.filter()`) but not the *programmatic* path
(`g.model.select(...)`) — that asymmetry **is exactly the kind of
inconsistency v2 exists to eliminate**, so it argues *for* finishing
the unification, not for accepting the gap.

## Decision

1. **Policy — distinguish removed surface by kind.** A *redundant /
   duplicative* removal owes no successor (true unification). A
   *unique capability with no v2-idiom equivalent* is **incomplete
   unification** and **owes a v2-native successor**; the successor's
   form / scope / priority is a separate planning decision, but
   "documented permanent gap / WONTFIX" is **not** the default and is
   admissible only if planning proves the capability vestigial
   (provably zero real programmatic use) — and even then it is a
   *deferral*, not a goal. The SC-12 precedent is hereby scoped to
   redundant removals only.

2. **Gap 2 (filter grammar) — reclassified: unique-capability loss →
   successor owed.** A v2-native successor on `EntitySelection`
   (composing with the existing spatial verbs + set algebra; a
   cleaner-than-1:1 shape is allowed if it improves consistency) is
   **planned**, tracked in `docs/plans/selection-gaps-v3.md`. The
   deleted `tests/test_selection_filters.py` (33 tests, git-recoverable)
   is its behavioural floor.

3. **Gap 1 (geometry → named mesh-selection) — reclassified:
   capability intact, ergonomics-only.** It survives via the retained
   two-call route (`g.model.select(...).to_physical(name)` →
   `g.mesh_selection.from_physical(dim, name, ms_name=)`, or
   `g.mesh_selection.add(dim, ids, name=)`). Only the *one-call*
   geometry→named-set shorthand was lost. A v2-idiom one-liner is an
   **ergonomics** decision, not a functionality verdict; not a
   WONTFIX-of-a-loss because nothing was lost.

4. **The successor is a NEW v2-consistent feature, not a regression
   undo.** It must respect every v2 invariant: ADR 0015 (separate
   Tier-1/Tier-2/mesh-selection registries; the `_kernel`
   downward-only leaf and the forbidden eager `core → viz` edge — the
   filter engine lives in `viz/Selection.py`, so exposing it on the
   `core`-side `EntitySelection` is the real engineering problem, the
   same boundary `EntitySelection.to_dataframe()` already paid for);
   the import-DAG tripwire; the resolution contract; no resurrection
   of `SelectionComposite` / `g.model.selection`. It **completes** the
   consistency goal by unifying the capability onto the single
   terminal; it does not soften it.

## Alternatives considered

1. **Leave ADR 0016 §4 as the final word (permanent documented
   gaps).** Rejected — it mischaracterises *collateral capability
   loss* as *program intent*. The mandate was consistency, not
   reduction; under-delivering the mandate and calling it "done" is
   the error the owner caught.
2. **Edit ADR 0016 §4 in place.** Rejected — ADRs are append-only
   (`README.md`). 0016's other decisions stay correct; this is the
   textbook supersede-by-new-ADR case (exactly as 0016 superseded
   0015's transient framing).
3. **Resurrect `SelectionComposite` / `g.model.selection` verbatim to
   "close the gap" fast.** Rejected — re-grows the divergent surface
   v2 removed. The consistency goal is *finished* by unifying the
   capability onto `EntitySelection`, not by restoring the old
   divergent class.

## Consequences

**Positive:**
- The recorded disposition now matches the program's actual mandate.
  Future sessions/users read "successor planned (incomplete
  unification)", not "accepted permanent gap" — `docs/api/selection.md`
  is updated to match.
- The redundant-vs-unique policy prevents future over-application of
  the SC-12 precedent to genuine capability removals.

**Negative:**
- The planned Gap-2 successor partially re-grows entity-selection
  surface. This is the deliberate, owner-accepted cost of not
  under-delivering the consistency mandate; bounded by "v2-idiom-native,
  composes with the existing verbs/set-algebra, no class resurrection."
- Until the successor ships, the capabilities are reachable only via
  the documented workarounds (Gap 1: the two-call route; Gap 2: the
  viewer-pick `viz.Selection.filter()`, or a manual predicate over
  `g.model.select(...).result()`).
- The v2 program itself remains **COMPLETE and merged**; this ADR
  opens a *follow-on feature track* (selection-gaps-v3). It does **not**
  reopen v2.

## References

- [decisions/0016-selection-unification-v2-complete.md](0016-selection-unification-v2-complete.md)
  — §4 disposition amended here; §1–§3 + Consequences remain in force.
- [decisions/0015-label-pg-separate-registries-kernel-leaf.md](0015-label-pg-separate-registries-kernel-leaf.md)
  — Decision 2 (the `core → viz` / import-DAG boundary the successor
  must respect; the `to_dataframe` re-implement-locally precedent).
- [docs/api/selection.md](../../../../../docs/api/selection.md)
  — "Incomplete unification — pending v2 successors" (updated to match).
- [docs/plans/selection-unification-v2.md](../../../../../docs/plans/selection-unification-v2.md)
  — the completed program (§5 R-v2-1, §8 out-of-scope).
- `docs/plans/selection-gaps-v3.md` — the follow-on planning doc (to be
  authored by the planning session that scopes the successors).
- `tests/test_selection_filters.py` — the deleted 33-test Gap-2
  behavioural spec; recover via
  `git log --all --diff-filter=D -- tests/test_selection_filters.py`
  then `git show <commit>^:tests/test_selection_filters.py`.
