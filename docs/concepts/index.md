# Concepts

**Understanding-oriented. The *why* behind apeGmsh, not the *how*.**

apeGmsh makes a handful of deliberate choices — a session owns one Gmsh
kernel, composites split by concern, you address geometry by *name* and
never by raw tag, the `FEMData` snapshot is the solver contract, and the
OpenSees bridge is *typed* rather than string-templated. None of that is
arbitrary. Once the mental model clicks, the API stops feeling like a pile
of methods to memorize and starts feeling like one idea applied
consistently.

These pages are for reading in an armchair, not for copy-pasting. When you
want to *do* a specific thing, the [How-to recipes](../how-to/index.md)
are faster. When you want to learn the workflow hands-on, start with the
[Tutorials](../tutorials/index.md). Come here when you want to understand
*why it's built this way* — so you can predict how it behaves instead of
guessing.

## Start here

→ **[The apeGmsh mental model](mental-model.md)**

One short page — about a screen and a half — covering the few ideas the
whole library rests on: the session and its kernel, composites by concern,
the tag / label / physical-group distinction, `.select()`, the immutable
`FEMData` snapshot, declare-then-resolve, and the typed bridge. **If you
read one Concepts page, read this one.** Everything else fills in detail.

## Going deeper

The topic guides below explain individual subsystems. They're the
de-blended descendants of the old long-form walkthrough — each one leads
with the idea, then shows it in code.

### The session and its abstractions

- **[Building a model — the basics](../internal_docs/guide_basics.md)** — the session lifecycle, composites, and the shape of a typical script.
- **[Naming things: tags, labels & queries](../internal_docs/guide_queries.md)** — why you address geometry by name, and how queries resolve to entities.
- **[Selection & the selection chain](../internal_docs/guide_selection.md)** — the `.select()` vocabulary and how chained selections compose.
- **[Parts vs. the session](../internal_docs/guide_parts_vs_session.md)** — when geometry belongs to a reusable Part and when it belongs to the session itself.

### Geometry, meshing & assembly

- **[CAD import & healing](../internal_docs/guide_cad_import.md)** — bringing STEP/IGES in, healing dirty geometry, and naming imported faces by query.
- **[Meshing](../internal_docs/guide_meshing.md)** — mesh sizing, fields, structured meshing, and editing.
- **[Parts & assembly](../internal_docs/guide_parts_assembly.md)** — Part templates, placement, and fragmenting an assembly into a conformal whole.
- **[Transforms](../internal_docs/guide_transforms.md)** — translate / rotate / mirror and how they interact with Parts.

### Physics: loads, masses & constraints

- **[Loads](../internal_docs/guide_loads.md)** — the declare-then-resolve pipeline for forces, pressures, gravity, and prescribed displacements.
- **[Masses](../internal_docs/guide_masses.md)** — lumped and distributed mass, resolved the same way loads are.
- **[Constraints](../internal_docs/guide_constraints.md)** — equalDOF, ties, rigid links and diaphragms — and the v2.0 rule that MP constraints **emit automatically** through the bridge.
- **[Sections](../internal_docs/guide_sections.md)** — declaring sections and integration, and how they reach the solver.

### The solver contract & results

- **[The FEM broker (`FEMData`)](../internal_docs/guide_fem_broker.md)** — the immutable snapshot every solver bridge consumes, and why it's frozen.
- **[The OpenSees bridge](../internal_docs/guide_opensees.md)** — the typed `apeSees(fem)` surface: typed primitives instead of raw `ops.*` strings.
- **[Obtaining results](../internal_docs/guide_obtaining_results.md)** — the deferred fork between `from_native`, `from_recorders`, and `from_mpco`, and how to choose.
- **[Reading & filtering results](../internal_docs/guide_results.md)** — the slab-based read API and selecting result data by `pg=` / `label=` / `selection=`.

---

*Next: [Core concepts (the mental model)](mental-model.md).*
