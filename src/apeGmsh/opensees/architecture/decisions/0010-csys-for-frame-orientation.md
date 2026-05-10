# ADR 0010 — Coordinate systems for frame orientation

**Status:** Accepted (already shipped in `apeGmsh.solvers`)

## Context

OpenSees `geomTransf <type> <tag> vx vy vz` requires a single
Cartesian vector to define the local x-z plane. For curved members
(arches, ring beams, dome ribs), users would have to compute one
vector per element by hand, or accept misaligned strong axes.

Other software (SAP, ETABS, Abaqus) lets the user pick a coordinate
system (Cartesian, cylindrical, spherical) and derives the local
orientation per element from it.

## Decision

Introduce `Cartesian`, `Cylindrical`, `Spherical` CS classes. Each
returns an orthonormal triad `(e1, e2, e3)` at a queried point. The
`reference_axis` (`e3`) is the in-plane reference for the orientation
rule:

```
local_y = unit(e3 × tangent)        if tangent not parallel to e3
        = e2                         otherwise (degenerate)
vecxz   = tangent × local_y
```

`add_geom_transf(name, transf_type, csys=Cartesian())` accepts a CS;
the build step computes per-element vecxz. Curved beams emit one
`geomTransf` line per distinct vecxz observed, automatically.

## Alternatives considered

1. **One vector per geometric transform (today's behavior only).**
   Rejected — forces users to compute per-element orientations
   manually for curved beams.
2. **Per-element overrides on `assign`.** Rejected — two ways to
   specify orientation creates ambiguity. The CS encodes intent;
   per-element overrides become noise.
3. **Surface normal of an adjacent face as the reference.**
   Considered. Real but limited — fails at edges where two faces
   meet (e.g. rib joining a dome to a meridional rib). Defer to a
   future `AlongSurface` CS if the use case appears.

## Consequences

**Positive:**

- Cartesian (default Z-up) reproduces today's hardcoded behavior
  exactly.
- Tank ring beams, dome ribs, and curved arches "just work" with
  one CS declaration.
- Roll-about-axis composes via `roll_deg=` parameter.
- 28 unit tests, no regressions on 82 prior solver tests.

**Negative:**

- Curved members emit N `geomTransf` lines (one per distinct
  vecxz). Document.
- Sign-flip on legs of arches when tangent direction reverses
  (continuous traversal). Inherent to OpenSees vecxz semantics;
  not papered over.
- Asymmetric sections (channels, angles) need explicit
  `roll_deg` to align consistently across legs of arches.
  Documented.

## Reference

- Implementation: `apeGmsh/solvers/_opensees_csys.py` (already in
  `solvers/`; will be re-exported from
  `apeGmsh.opensees.transform` once skeletons land).
- Tests: `tests/test_opensees_csys.py`
- Walked design discussion: see conversation transcript and the
  shoe-buckle arch example.
