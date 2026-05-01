# Examples

A curated, in-progress gallery — a working tour through the apeGmsh
API by example. Each notebook is rendered inline; you can also open
the source on GitHub.

> **Looking for the full repo?** All 60+ examples live at
> [apeGmsh/examples](https://github.com/nmorabowen/apeGmsh/tree/main/examples)
> on GitHub. The gallery below is the curated subset we surface in
> the docs.

## Getting started

<div class="grid cards" markdown>

-   :material-numeric-1-circle:{ .lg .middle } &nbsp; __[Hello plate](notebooks/01_hello_plate.ipynb)__

    ---

    The smallest viable apeGmsh model — a square plate, fixed on one
    edge, loaded on the opposite. Covers the session, geometry,
    physical groups, meshing, and the OpenSees bridge in ~40 lines.

-   :material-numeric-2-circle:{ .lg .middle } &nbsp; __[Cantilever beam (2D)](notebooks/02_cantilever_beam_2D.ipynb)__

    ---

    The canonical FEM teaching example: a 2D cantilever under
    end-load. Walks through the linear-static analysis pipeline and
    pulls displacements, reactions, and stresses out of `Results`.

</div>

## Building richer models

<div class="grid cards" markdown>

-   :material-numeric-4-circle:{ .lg .middle } &nbsp; __[Portal frame (2D)](notebooks/04_portal_frame_2D.ipynb)__

    ---

    Multi-element frame with beam-column elements, shared nodes, and
    distributed loads. Introduces section assignment and load
    patterns.

-   :material-tag-multiple:{ .lg .middle } &nbsp; __[Labels & physical groups](notebooks/05_labels_and_pgs.ipynb)__

    ---

    The naming system in practice — Tier 1 labels (geometry-time,
    boolean-survivable) vs Tier 2 physical groups (solver-facing).
    The selection vocabulary used everywhere downstream.

-   :material-puzzle-outline:{ .lg .middle } &nbsp; __[Part assembly](notebooks/10b_part_assembly.ipynb)__

    ---

    Build a part once, instantiate it many times, fragment into a
    conformal assembly. Labels survive the STEP round-trip.

-   :material-vector-link:{ .lg .middle } &nbsp; __[Interface tie](notebooks/12_interface_tie.ipynb)__

    ---

    Non-matching meshes joined via `ASDEmbeddedNodeElement` —
    apeGmsh's tie constraint emitted as a penalty element. Useful
    when two meshed regions need to act as one.

</div>

## Analysis types

<div class="grid cards" markdown>

-   :material-sine-wave:{ .lg .middle } &nbsp; __[Modal analysis](notebooks/17_modal_analysis.ipynb)__

    ---

    Eigenvalue analysis with `DomainCapture.capture_modes()` — one
    stage per mode in a single results file. Reads back as
    `results.modes[i]` with frequency, period, and mode shape.

-   :material-chart-bell-curve:{ .lg .middle } &nbsp; __[Pushover (elastoplastic)](notebooks/19_pushover_elastoplastic.ipynb)__

    ---

    Nonlinear pushover analysis using fiber sections. Demonstrates
    iterative load stepping, capacity tracking, and pulling section
    forces along the length via line-station results.

</div>

## What's not in the gallery (yet)

The repo's `examples/` directory has many more — buckling, contact
springs, tunnel meshes, lateral-torsional buckling, embedded rebars,
soil-structure interaction, etc. Browse them on
[GitHub](https://github.com/nmorabowen/apeGmsh/tree/main/examples)
or [examples/EOS Examples/](https://github.com/nmorabowen/apeGmsh/tree/main/examples/EOS%20Examples)
for the structured course material (numbered 01–22+).

If you'd like a particular example promoted to the gallery — better
narrative, screenshots, prose — open an issue with the notebook
filename and a one-line "why this one matters."
