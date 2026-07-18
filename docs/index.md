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

## Install

```bash
pip install apeGmsh[all]
```

The `[all]` extra pulls in the OpenSees bridge (via
[openseespy](https://pypi.org/project/openseespy/)), the web viewer, and
plotting — everything the tutorials use. Want just the modelling core?
`pip install apeGmsh`.

!!! tip "New here? Build a model in 10 minutes"
    The fastest way in is to **[build your first model →](tutorials/first-model.md)**:
    a steel cantilever you solve and check against `PL³/3EI`, end to end, in
    under 40 lines. When you want the whole staircase — every tutorial and
    worked example in reading order, each with its verification check — it's
    on the **[learning path](tutorials/learning-path.md)**.

## Where do you want to start?

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } &nbsp; __[Tutorials](tutorials/index.md)__

    ---

    *Teach me, step by step.*

    Hand-held journeys that end in a number you can trust. Start with
    [**your first model in 10 minutes**](tutorials/first-model.md).

-   :material-tools:{ .lg .middle } &nbsp; __[How-to recipes](how-to/index.md)__

    ---

    *I know the basics — how do I X?*

    Task-titled recipes: fix supports, apply a pressure, tie meshes,
    run a pushover, read a displacement, save & reload, compose modules.

-   :material-lightbulb-on:{ .lg .middle } &nbsp; __[Concepts](concepts/mental-model.md)__

    ---

    *Help me build a mental model.*

    The six ideas behind everything — session, composites, naming,
    `FEMData`, declare-then-resolve, the typed bridge — then the topic guides.

-   :material-rocket-launch:{ .lg .middle } &nbsp; __[Examples](examples/index.md)__

    ---

    *Show me a worked model.*

    Recognizable structural problems — frames, modal, pushover —
    built end to end and checked against known answers.

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
