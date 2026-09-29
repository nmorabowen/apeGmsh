# ADR 0035 — ASDEmbeddedNodeElement option exposure

**Status:** Accepted. Bumps the OpenSees zone H5 schema to 2.12.0 and
widens the [Emitter Protocol](../../emitter/base.py)'s
`embeddedNode` method.

## Context

`element ASDEmbeddedNodeElement` is the OpenSees primitive every
non-matching-mesh tie / embedded-element / tied-contact constraint
fans out to (ADR 0022). Its C++ parser ([SRC/element/CEqElement/
ASDEmbeddedNodeElement.cpp](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp))
documents the full vocabulary at line 201:

```
element ASDEmbeddedNodeElement $tag $Cnode $Rnode1 $Rnode2 $Rnode3 \
                               <$Rnode4> <-rot> <-p> <-K $K> <-KP $KP>
```

Pre-ADR-0035, the apeGmsh bridge emitted only the mandatory
positionals: `element ASDEmbeddedNodeElement $tag $Cnode $Rnode1
$Rnode2 $Rnode3 <$Rnode4>`.  None of the four optionals were
exposed, so OpenSees fell through to the parser defaults:

| Flag  | C++ default                                 |
|-------|---------------------------------------------|
| `-rot`| absent (rotational DOFs ignored)            |
| `-p`  | absent (no pressure coupling)               |
| `-K`  | `1.0e18` ([cpp:222](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp))  |
| `-KP` | `1.0e18` (falls back to `K` when only `-K` given) |

The bug surfaced via deck diffs against STKO, which emits `-K 1.0e8`
explicitly. The K-divergence (1e18 vs 1e8 = 10 orders of magnitude)
materially shifts the penalty conditioning of the coupled system —
two pipelines that should match physically did not. The deeper
problem was that **the user had no way to set K** even when they
wanted to: the option simply was not in the API.

## Decision

Expose all four optionals as kwargs on every layer of the constraint
pipeline:

```
g.constraints.tie(..., stiffness=1e8, rotational=True)
                  │
                  ▼
              TieDef(stiffness, stiffness_p, rotational, pressure)
                  │  __post_init__ validates rot xor p, stiffness_p ⇒ p
                  ▼
              ConstraintResolver.resolve_tie(...)
                  │
                  ▼
              InterpolationRecord(stiffness, stiffness_p, rotational, pressure)
                  │
                  ▼
              _emit_surface_couplings → emitter.embeddedNode(
                      ele_tag, cnode, *master_nodes,
                      stiffness=..., stiffness_p=..., rotational=..., pressure=...)
                  │
                  ▼
              tcl / py / live → -rot -p -K $K -KP $KP (parser order)
              h5             → typed columns on /opensees/constraints/embeddedNode
              recording      → kwargs dict
```

### Defaults match the C++ parser

`stiffness=1.0e18`, `stiffness_p=None`, `rotational=False`,
`pressure=False`. Scripts that never touched the new kwargs emit
semantically-identical decks to the pre-ADR-0035 implementation.
The visible difference is that `-K 1e+18` now appears explicitly in
the emitted line — this is intentional. Hiding the value behind a
silent C++ default was the original bug.

### Fail-loud `__post_init__` validation on every Def

Two invariants enforced at Def construction (before any meshing /
resolution work):

1. `not (rotational and pressure)` — mirrors the C++ check at
   [ASDEmbeddedNodeElement.cpp:276](https://github.com/OpenSees/OpenSees/blob/master/SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp)
   which rejects both `-rot` and `-p` together.
2. `stiffness_p is None or pressure is True` — `-KP` is only consulted
   by the C++ when `m_p_flag` is on (see `iKP` references in
   `getTangentStiff()`); passing `stiffness_p` without `pressure=True`
   is a no-op the user almost certainly didn't intend.

### Emitter Protocol widening (INV-4)

`embeddedNode` signature changes from
`(ele_tag, cnode, *args)` to
`(ele_tag, cnode, *master_nodes, stiffness=1.0e18, stiffness_p=None,
rotational=False, pressure=False)`. All five concrete emitters (Tcl,
Py, LiveOps, H5, Recording) implement the same kwargs to preserve
INV-4 (Protocol shape uniform across emitters). The flag tokens are
materialised in parser order via the shared
[`_build_embedded_flag_args`](../../emitter/base.py) helper for
tcl / py / live; H5 stores them as typed compound-dtype columns and
bypasses the helper.

### H5 schema 2.11.0 → 2.12.0 (additive)

`/opensees/constraints/embeddedNode` gains five typed columns —
`stiffness` (float64), `stiffness_p` (float64) + `has_stiffness_p`
(uint8 sentinel for `None`), `rotational` (uint8), `pressure`
(uint8). Per [ADR 0023](0023-per-zone-schema-versions.md) two-version
reader window, the bridge accepts both 2.11.x and 2.12.x files. Old
2.11.x readers ignore the new columns; new readers default the
columns to the C++ values when a 2.11.x file lacks them.

## Consequences

**Positive.**
- Decks emitted by apeGmsh and STKO can be byte-equivalent (modulo
  unrelated headers) for the embedded-element family — pass
  `stiffness=1.0e8` and the `-K` token matches.
- K-conditioning is now a knob the user can turn for solver
  diagnostics on stiff-penalty models.
- `-rot` makes shell-on-solid embedding genuinely well-posed when the
  embedded node has rotations — previously the rotations were silently
  unconstrained.

**Neutral / forward-looking.**
- The `-p` (u-p coupling) path is now wire-up complete but covers only
  a subset of OpenSees u-p elements out of the box. Users emitting
  saturated-soil models with embedded reinforcement get the API
  surface; correctness depends on the host element being u-p (e.g.
  `SSPbrickUP`).

**Breaking — none under default kwargs.** The defaults match the C++
parser, so legacy scripts emit semantically-identical models. The
emitted Tcl / Py text now includes `-K 1e+18` explicitly — string
matching against the old pre-flag form needs updating (the apeGmsh
test suite already absorbed this change).

## Cross-references

- [ADR 0022](0022-mp-constraint-emission-fanout.md) — the fan-out
  pass that routes tie / embedded / tied_contact onto
  ASDEmbeddedNodeElement.
- [ADR 0023](0023-per-zone-schema-versions.md) — two-version reader
  window discipline.
- [ADR 0027](0027-cross-partition-mp-constraints.md) — partitioned
  embedded-element emission (also threaded by this ADR via
  `_emit_surface_couplings_for_rank`).
- [ADR 0033](0033-s2-emit-wiring-per-node-ndf.md) — sibling discipline
  for threading typed metadata from broker through resolver into the
  emitted deck.
