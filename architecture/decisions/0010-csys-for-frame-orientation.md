# ADR 0010 — Orientation fields for frame orientation

**Status:** Accepted (shipped). Originally landed as `csys=` in
`apeGmsh.solvers`; renamed to `orientation=` in the post-Phase-8.6
naming pass (the OpenSees `geomTransf` is itself called a "geometric
transformation", so `csys` overloaded that vocabulary).

## Context

OpenSees `geomTransf <type> <tag> vx vy vz` requires a single
Cartesian vector to define the local x-z plane. For curved members
(arches, ring beams, dome ribs), users would have to compute one
vector per element by hand, or accept misaligned strong axes.

Other software (SAP, ETABS, Abaqus) lets the user pick a coordinate
system (Cartesian, cylindrical, spherical) and derives the local
orientation per element from it.

## Decision

Introduce `Cartesian`, `Cylindrical`, `Spherical` orientation classes.
Each returns an orthonormal triad `(e1, e2, e3)` at a queried point.
The `reference_axis` (`e3`) is the in-plane reference for the
orientation rule:

```
local_y = unit(e3 × tangent)        if tangent not parallel to e3
        = e2                         otherwise (degenerate)
vecxz   = tangent × local_y
```

`ops.geomTransf.<Type>(orientation=Cartesian())` accepts an
orientation field; the build step computes per-element vecxz. Curved
beams emit one `geomTransf` line per distinct vecxz observed,
automatically.

## Alternatives considered

1. **One vector per geometric transform (today's behavior only).**
   Rejected — forces users to compute per-element orientations
   manually for curved beams.
2. **Per-element overrides on `assign`.** Rejected — two ways to
   specify orientation creates ambiguity. The orientation field
   encodes intent; per-element overrides become noise.
3. **Surface normal of an adjacent face as the reference.**
   Considered. Real but limited — fails at edges where two faces
   meet (e.g. rib joining a dome to a meridional rib). Defer to a
   future `AlongSurface` orientation if the use case appears.

## Consequences

**Positive:**

- Cartesian (default Z-up) reproduces today's hardcoded behavior
  exactly.
- Tank ring beams, dome ribs, and curved arches "just work" with
  one orientation declaration.
- Roll-about-axis composes via `roll_deg=` parameter.
- 28 unit tests, no regressions on 82 prior solver tests.

**Negative:**

- Curved members emit N `geomTransf` lines (one per distinct
  vecxz). Document.
- Sign-flip on legs of arches when tangent direction reverses
  (continuous traversal). Inherent to OpenSees vecxz semantics;
  not papered over. Mesh-side fix: `g.mesh.editing.reverse(...)`
  on the affected PG.
- Asymmetric sections (channels, angles) need explicit
  `roll_deg` to align consistently across legs of arches.
  Documented.

## Naming amendment (post-Phase 8.6)

The original ADR used `csys=` as the construction kwarg and called
the classes "coordinate systems." Both clashed with OpenSees's own
"geometric transformation" (`geomTransf`) vocabulary — readers
asked how `csys=PDelta` related to `geomTransf=PDelta`. The kwarg
was renamed to `orientation=` and the classes are described as
"orientation fields" (they produce a triad at every point); the
class names themselves (`Cartesian`, `Cylindrical`, `Spherical`)
stay the same because those are literal geometric names.

The implementation file moved `solvers/_opensees_csys.py` →
`opensees/_csys.py` (Phase 8.2) → `opensees/_orientation.py` (this
rename pass). A deprecation shim at `solvers/_opensees_csys.py`
re-exports for one cycle.

## Reference

- Implementation: `apeGmsh/opensees/_orientation.py`.
- Tests: `tests/test_opensees_orientation.py`,
  `tests/opensees/unit/primitives/test_geom_transf.py`,
  `tests/opensees/contract/test_geom_transf_contract.py`.
- Walked design discussion: see conversation transcript and the
  shoe-buckle arch example.
