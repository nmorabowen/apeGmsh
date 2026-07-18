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

- **[The session](session.md)** — the session lifecycle, composites, the shape of a typical script, and when geometry belongs to a reusable Part rather than the session itself.
- **[Selection & queries](selection.md)** — why you address geometry by name, the `.select()` vocabulary, and how queries resolve to entities.

### Geometry, meshing & assembly

- **[Geometry & CAD](geometry-and-cad.md)** — authored geometry, STEP import and healing, naming imported faces by query, and transforms.
- **[Meshing](meshing.md)** — mesh sizing, fields, structured meshing, and what partitioning is for.
- **[Parts & assembly](parts-and-assembly.md)** — Part templates, placement, and fragmenting an assembly into a conformal whole.

### Physics: loads, masses & constraints

- **[Loads & masses](loads-and-masses.md)** — the declare-then-resolve pipeline for forces, pressures, gravity, prescribed displacements, and mass.
- **[Constraints](constraints.md)** — equalDOF, ties, rigid links, diaphragms, and contact — and the rule that MP constraints **emit automatically** through the bridge.
- **[Sections](sections.md)** — declaring sections, the in-process section-property analyzer, and how both reach the solver.

### The solver contract & results

- **[The FEM broker (`FEMData`)](fem-broker.md)** — the immutable snapshot every solver bridge consumes, and why it's frozen.
- **[The OpenSees bridge](opensees-bridge.md)** — the typed `apeSees(fem)` surface: typed primitives instead of raw `ops.*` strings.
- **[Obtaining results](../internal_docs/guide_obtaining_results.md)** — the deferred fork between `from_native`, `from_recorders`, and `from_mpco`, and how to choose.
- **[Reading & filtering results](../internal_docs/guide_results.md)** — the slab-based read API and selecting result data by `pg=` / `label=` / `selection=`.

---

*Next: [Core concepts (the mental model)](mental-model.md).*
