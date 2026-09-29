# ADR 0004 — Sections live outside `material/`

**Status:** Accepted

## Context

OpenSees source organizes sections under `SRC/material/section/`,
alongside `material/uniaxial/`, `material/nD/`, and
`material/yieldSurface/`. The literal mirror would put our sections
in `apeGmsh.opensees.material.section`.

But sections and materials play different roles in user code:

| | Material | Section |
|---|---|---|
| Role | constitutive law (σ–ε) | aggregator of constitutive law over a cross-section |
| Dependencies | leaf node | composes materials |
| Mental model | "what's the steel made of" | "what's the cross-section like" |
| Capabilities | backbone, cyclic | fiber plot, M-φ, geometry queries |
| Recipes apply | rarely | constantly (RectangularConfinedColumn, IShape, …) |
| Used by | sections, zero-length, truss | beam-columns, shells |

In structural-engineering practice, "the column section" and "the
rebar material" are different categories. Lumping them under
`material/` makes the user mental model fuzzy.

## Decision

Separate `section/` at the top level of the package:

```
apeGmsh/opensees/
├── material/
│     uniaxial.py
│     nd.py
│     yield_surface.py
├── section/
│     fiber.py
│     plate.py
│     beam.py
│     aggregator.py
```

Class names still match OpenSees tokens (`Fiber`, `ElasticMembranePlateSection`,
`LayeredShell`).

## Alternatives considered

1. **Mirror OpenSees: `material/section.py`.** Rejected on
   user-mental-model grounds.
2. **Three-tier hierarchy: `material.uniaxial`, `material.nd`,
   `material.section`.** Rejected — `material.section.Fiber` is
   awkwardly nested when sections are first-class user concepts.
3. **`section/` containing only fiber sections; plate sections
   under `material/`.** Rejected — splits an already-cohesive
   category by accident of OpenSees implementation.

## Consequences

**Positive:**

- Section is first-class in import paths and IDE navigation.
- Recipes (which target sections almost exclusively) live in
  parallel: `section_recipes.py` is conceptually adjacent to
  `section/`.
- Material's `dependencies()` doesn't return sections (always
  leaf), reducing graph complexity.

**Negative:**

- Departs from OpenSees source folder layout. Acceptable —
  the *type token* match (Fiber, ElasticMembranePlateSection) is
  what gives OpenSees fluency, not the folder name.
- Users porting Tcl scripts may look for `material.section.Fiber`
  first. Mitigated by docstring redirects and clear docs.

## Reference

- [charter.md](../charter.md)
- [layout.md](../layout.md)
