# ADR 0041 — Chain-phase routing for geometry-intensive constraints (`embedded`, `tied_contact`) (summary)

**Status:** ACCEPTED 2026-05-27. Authored as DRAFT during a multi-agent design review session;
the 8 open questions were resolved with the worker's recommendations adopted verbatim. […]

## Decision

Compose v1.1-A.2: route `EmbeddedDef` and `TiedContactDef` in chain phase (after
`apeGmsh.from_h5` / `g.compose`) instead of leaving them on the bump-counter fallback. Builds on
ADR 0036 (build-phase Kuhn decomposition) and ADR 0038 (chain-phase router, `FEMDataSource`).

1. **Lift Kuhn decomposition into a geometric utility.** New module
   `src/apeGmsh/_kernel/geometry/_host_decomposition.py` holds `HEX8_TO_6_TETS`,
   `PRISM6_TO_3_TETS`, `PYRAMID5_TO_2_TETS` and the pure-numpy, gmsh-free
   `decompose_hosts_to_subelements(groups, *, warn_higher_order=None)`, which takes
   `(etype, conn)` pairs from any source and returns `ndarray(F, 3 | 4)` sub-element rows.
   `ConstraintsComposite._collect_host_subelements` becomes a thin gmsh-backed adapter.
   Not placed in `_constraint_resolver/`: the tables are pure geometry (ADR 0015 leaf rule).
2. **`FEMDataSource` grows two concrete-class methods**: `host_subelements_for(target)`
   (element-side Tier 1 → Tier 2 walk, then decomposition; `KeyError` / `ValueError`) and
   `boundary_faces_for(target)` (dim=2 `ElementGroup` rows filtered by node ownership, the
   `PartsRegistry.build_face_map` semantics). `GmshSource` does not grow them.
3. **Two router branches** in `route_def_to_fem` (`_chain_phase_router.py`):
   `_route_embedded` (drops embedded nodes that coincide with host corners, then
   `resolve_embedded` → `InterpolationRecord`s) and `_route_tied_contact`
   (`resolve_tied_contact` → one `SurfaceCouplingRecord`), each appended with
   `fem.with_constraint(rec)`. `try_chain_phase_route` still catches `KeyError`/`TypeError`
   for the fallback; resolver `ValueError`s propagate. `_build_resolver` does not grow.
4. **No face synthesis.** Default `get_fem_data(dim=None)` extraction and the H5 round-trip
   keep dim=2 groups, so filtering suffices. A broker saved with `dim=3` only raises a clean
   `ValueError` naming the remedy (re-extract with `dim=None` and re-save).

Invariants:
- **INV-1** the ADR 0038 callable contract is unchanged: both verbs work in build and chain
  phase; the bump-counter path stays the safety net. No public API change.
- **INV-2** build phase is byte-identical (locked by `tests/test_embedded_decomposition.py`).
- **INV-3** `FEMDataSource` stays narrow: the `ResolverSource` Protocol keeps four methods
  (`node_ids`, `node_coords`, `nodes_for`, `has_target`).
- **INV-4** geometry mutation stays frozen in chain phase (`ChainPhaseError`); the branches
  are read-only against FEMData.
- **INV-5** no H5 schema bump (existing record types, ADR 0023).
- **INV-6** ADR 0036's higher-order warning and mixed-dim fail-loud are preserved; in chain
  phase the warning fires once per `(etype, target)`.

Delivery: Phases 1–3 (embedded) in one PR, Phases 4–5 (tied_contact) in a second.

## Amendments

No dated amendment sections. The 2026-05-27 "Decisions (resolved)" section locks the eight
answers: module location, the renumber 0039 → 0041, the two-method split, concrete class
only, a clean `ValueError` for the dim=3-only broker, two PRs, the audit-then-move rule for
the Kuhn constants, and warnings per `(etype, target)`. Follow-ups (not blocking):
volume-to-face synthesis (Phase 6, on demand), `host_subelements_for` caching.

Full text: [../0041-chain-phase-geometry-constraints.md](../0041-chain-phase-geometry-constraints.md).
Precedence (AGENTS.md): the code wins over the ADR, and the ADR wins over this summary.
