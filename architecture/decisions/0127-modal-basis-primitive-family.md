# ADR 0127 — Admit the `ModalBasis` primitive family for the fork's mode-set commands

**Status:** Proposed (2026-10-29). Admits one entry to the frozen
"primitive families" registry under ADR 0121 D4. Program issue
nmorabowen/OpenSees#1024, chain A link A3a (apeGmsh PR #2620).

## Context

Chain M of the Dynamics program adds two commands to the Ladruno fork:

- `residualModes -dir … <-period T> <-replace> <-tol t>` (WP-213, fork
  ADR 213) appends missing-mass pseudo-modes to the domain's mode set.
- `ritzVectors n <-dir …> <-pattern …> <-cycles …> <-tol t>` (WP-215, fork
  ADR 213 D7, D8) installs load-dependent Ritz vectors as the mode set, in
  place of `eigen`.

Both run after the model is built, like `eigen`, and change the mode set
that `modalProperties` and `responseSpectrumAnalysis` then read. Link A3a
wraps them as `apeSees.residual_modes` and `apeSees.ritz_vectors`.

Two locks decide how the bridge emits a new verb:

- ADR 0114 D2 freezes the emitter Protocol at 75 methods, so a new verb is a
  `command()` token with a `via="command"` row in `emitter/verbs.py`, never
  a new method.
- ADR 0114 D3 lets only a `Primitive`'s `_emit` call `command()` (the verbs
  lock, check (g)).

So each command needs a concrete `Primitive` class. Every concrete primitive
must also belong to a family: the registry lock (`test_verbs_lock.py`
check (h)) needs its allocator kind from `apesees._KIND_BY_FAMILY`, and
`tests/test_family_completeness.py` needs it on a family's contract list.
Under ADR 0121 D4, a new family is an admission to the frozen "primitive
families" registry, and needs an Accepted ADR.

None of the existing families fits:

- **`Analysis`** (and the other chain bases: `ConstraintHandler`,
  `Numberer`, `LinearSystem`, `ConvergenceTest`, `SolutionAlgorithm`,
  `Integrator`). These are the analysis chain. The bridge selects them by
  `isinstance` to emit the chain, re-bind it per stage
  (`_ANALYSIS_CHAIN_BASES`), and gate `analyze` and `ritz_vectors` on a
  declared analysis. A mode-set command in one of these families would
  be emitted with the chain at build time, before any mode exists, which
  the fork refuses ("run eigen first").
- **The model families** (`UniaxialMaterial`, `NDMaterial`, `Section`,
  `GeomTransf`, `BeamIntegration`, `Element`, `TimeSeries`, `Pattern`,
  `Damping`, `Recorder`). Each is a tagged model declaration that a deck
  and `model.h5` carry. The mode-set commands declare nothing and have no
  tag.

The command side needs no admission. The two rows use the existing verb
family `modal`, the family of `eigen`, `modal_properties` and
`response_spectrum_analysis`.

## Decision

### D1 — The `ModalBasis` family

`apeGmsh.opensees._internal.types.ModalBasis(Primitive)` is the abstract
base for commands that replace or extend the domain's mode set. Its
allocator kind is `modalBasis` (one row in `apesees._KIND_BY_FAMILY`). Its
two members are in `apeGmsh.opensees.analysis.modal`:

- `ResidualModes(dirs, period="rayleigh", replace=False, tol=None)` emits
  `command("residualModes", …)`.
- `RitzVectors(n, dirs=(), patterns=(), cycles=None, tol=None)` emits
  `command("ritzVectors", …)`.

Each validates its arguments the way the fork's parser does.

### D2 — Members are never registered on a bridge

A `ModalBasis` primitive is a value object that a procedure emits on its
own live domain, after the basis it extends (`record._emit(live, 0)`). It
never goes through `apeSees._register`, so it has no tag and no
declaration, and decks and `model.h5` do not carry it. A deck has no slot
for it: `eigen` reaches a deck only ahead of `modalDamping` (ADR 0053 D4).
If a member ever needs to reach decks, that is a new decision.

### D3 — The census entry

`tests/families.py` gains `Family("modal basis", ModalBasis, …:ALL_MODAL_BASIS)`.
Its contract list is in `tests/opensees/parity/test_chain_m_command_parity.py`,
which checks every member's fork command line on all five emitters.
`tests/freeze_census.py` gains
`ADMITTED["primitive families"]["modal basis"] = "0127"`.

### D4 — New members

Under ADR 0121 D4, members of an admitted family are not frozen. A later
mode-set command (for example a static-correction variant) joins
`ModalBasis` with its own fork refusal and warrant-row test, and needs no
further ADR.

## Consequences

- `lock-tests` (`tests/test_freeze_lock.py`) passes only once this ADR is
  Accepted. The maintainer accepts it: `scripts/land_pr.py` refuses a diff
  that sets an ADR to Accepted, and one that edits the census without the
  `human-gate` label.
- `params_names_ledger.txt` lists both members as `nostore`, because the
  family has no flat argv store.
- The frozen emitter Protocol, the D3 caller lock and the `modal` verb
  family are unchanged.
