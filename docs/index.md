---
hide:
  - navigation
---

<div class="ape-hero" markdown>
<img class="ape-hero__mark" src="assets/logo.svg" alt="apeGmsh mark" />
<div>
  <div class="ape-hero__word">apeGmsh</div>
  <div class="ape-hero__sub">LADRUÑO</div>
</div>
</div>

!!! info "Built on Gmsh"
    apeGmsh is a wrapper built on top of the (awesome) [Gmsh](https://gmsh.info)
    Python API. It adds a set of abstractions over the main API to fit an
    intended structural-FEM workflow — parts, constraints, loads, masses,
    and an OpenSees bridge. You still have the full Gmsh API underneath
    whenever you need it.

## Where do you want to start?

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } &nbsp; __[First steps](internal_docs/first_steps.md)__

    ---

    *I'm new — orient me.*

    A conversational walkthrough of the session model, naming
    (tags / labels / physical groups), queries, and CAD import.
    The right place to start.

-   :material-rocket-launch:{ .lg .middle } &nbsp; __[Quickstart & Examples](examples/index.md)__

    ---

    *Show me a working model.*

    Hello-plate, cantilever, portal frame, modal, pushover — a
    curated gallery of notebooks rendered inline.

-   :material-cube-outline:{ .lg .middle } &nbsp; __[Build a model](internal_docs/guide_basics.md)__

    ---

    *I'm meshing, constraining, and loading.*

    The composite-by-composite reference: parts, mesh sizing, sections,
    constraints, loads, masses.

-   :material-chart-line:{ .lg .middle } &nbsp; __[Run & read results](internal_docs/guide_obtaining_results.md)__

    ---

    *I'm post-processing.*

    Five strategies for obtaining results, the slab-based read API,
    and the post-solve viewer.

-   :material-bank:{ .lg .middle } &nbsp; __[Architecture](architecture/apeGmsh_architecture.md)__

    ---

    *Why is it built this way?*

    Principles, the broker freeze, parts & assembly, the spec-as-seam
    pattern for the OpenSees bridge.

-   :material-book-open-variant:{ .lg .middle } &nbsp; __[API reference](api/index.md)__

    ---

    *Look up a method.*

    Complete API surface — session composites, mesh, OpenSees bridge,
    parts, constraints, loads, masses, results, viewers.

</div>

## What's new

<div class="grid cards" markdown>

-   :material-new-box: &nbsp; **Five-strategy results pipeline**

    ---

    A single declarative spec drives five execution paths — script
    export, live recorders, domain capture, MPCO export, live MPCO.

    [Architecture →](architecture/apeGmsh_results_obtaining.md) ·
    [Guide →](internal_docs/guide_obtaining_results.md)

-   :material-monitor-eye: &nbsp; **ResultsViewer redesign (B0–B5)**

    ---

    Post-solve viewer: diagrams (contour, vector glyph, line force,
    fiber section, layer stack, gauss markers, spring force,
    **applied loads**, **reactions**), scrubber, persistent
    sessions, multi-stage navigation.

    [Architecture →](architecture/apeGmsh_results_viewer.md)

-   :material-cog-transfer: &nbsp; **Native + MPCO + transcoder readers**

    ---

    One slab-based composite API across three on-disk formats.
    `pg=` / `label=` / `selection=` selection vocabulary all the way.

    [Reference →](api/results.md)

-   :material-package-variant: &nbsp; **v1.4 – v1.5 polish**

    ---

    Import banner with `APEGMSH_QUIET=1` opt-out, event-loop
    dispatcher for the viewers, per-card Apply on diagram layers,
    per-Geometry display fix, and applied-loads & reactions
    diagrams.

    [Changelog →](changelog.md)

</div>

--8<-- "README.md"

---

## Credits

**Developed by:** Nicolás Mora Bowen · Patricio Palacios · José Abell · Guppi

Part of José Abell's *El Ladruño Research Group*.
