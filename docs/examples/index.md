# Examples

Recognizable structural problems, built end to end through the **typed
`apeSees` bridge** and checked against a known answer.

<div class="grid cards" markdown>

-   __[Portal frame (2D)](portal-frame.md)__ — two columns + a beam under
    gravity and lateral load; multiple element groups, drift, and a base
    shear that checks to exactly the applied load.

-   __[Modal analysis (cantilever)](modal-analysis.md)__ — `g.masses` +
    `ops.eigen`, the first three natural frequencies read back through
    `Results` and matched to the Euler–Bernoulli closed form.

-   __[Fiber sections & moment–curvature](fiber-moment-curvature.md)__ — a
    `W14×90` fibre section (dimensions from **apeSteel**) driven through
    yield with `ZeroLengthSection`; the M–κ curve passes through
    $M_y=F_yS_x$ and plateaus at $M_p=F_yZ_x$, shape factor 1.10.

-   __[Multi-part assembly](multipart-assembly.md)__ — build a column once
    as a reusable `Part`, stamp it three times with `g.parts.add`, and read
    each copy's deflection back by its own label (each exactly $PL^3/3EI$).

-   __[Tie non-matching meshes](tie-non-matching-meshes.md)__ — two solid
    blocks meshed at different sizes, joined by a `g.constraints.tie` the
    bridge **auto-emits**; the load transmits exactly and the column matches
    the monolithic bar to ~3%.

-   __[STEP import: plate with a hole](step-plate-with-hole.md)__ — import
    a CAD part, heal it, name edges by geometric query, refine at the
    hole, and recover the classic stress concentration $K_t\approx3$.

</div>

More rungs (multi-part assembly, mesh ties, shell-on-solid, compose,
pushover, staged SSI) are landing wave by wave.

## Legacy notebook gallery

The notebooks below predate the typed-bridge curriculum — they drive OpenSees
with vanilla `openseespy` and split the `spec.capture` / `spec.emit_recorders`
strategies across the gallery. They still run, but the worked examples above
are the recommended starting point. Each notebook follows the same template:

* Build geometry / mesh / FEM with apeGmsh.
* Drive OpenSees with **vanilla `openseespy`** calls.
* Declare **apeGmsh recorders** by physical-group / label name.
* Wrap the analysis in **`spec.capture(...)`** or
  **`spec.emit_recorders(...)`** — the two strategies are split across
  the gallery so you see both in action.
* Read results back via **`Results.from_native(...)` /
  `Results.from_recorders(...)`**, verify against closed-form
  references, and plot in-notebook from the slabs.

> **Looking for the full repo?** 60+ examples live under
> [apeGmsh/examples](https://github.com/nmorabowen/apeGmsh/tree/main/examples)
> on GitHub. The 8 below are the curated subset rendered inline in
> the docs.

> **Strategy at a glance.** Each card calls out which results-acquisition
> strategy that notebook uses — `spec.capture` for the apeGmsh-native
> HDF5 path with broadest coverage, `spec.emit_recorders` for live
> classic OpenSees recorders. Same `Results` API on the read side
> regardless.

## Getting started

<div class="grid cards" markdown>

-   :material-numeric-1-circle:{ .lg .middle } &nbsp; __[Hello plate](notebooks/01_hello_plate.ipynb)__

    ---

    *Strategy: `spec.capture`*

    The smallest viable apeGmsh ↔ openseespy ↔ Results loop. A unit
    plate in plane stress under uniaxial tension, verified against
    $u_x = \sigma L / E$ to machine precision. Covers the full
    pipeline in ~26 cells: geometry → mesh → vanilla `ops.*` model →
    recorder declaration → capture → read back → in-notebook plot.

-   :material-numeric-2-circle:{ .lg .middle } &nbsp; __[Cantilever beam (2D)](notebooks/02_cantilever_beam_2D.ipynb)__

    ---

    *Strategy: `spec.emit_recorders`*

    The other results-acquisition strategy. Same `begin_stage` /
    `end_stage` lifecycle as `capture`, but classic recorder `.out`
    files on disk that `Results.from_recorders(stage_id=...)` reads
    back. Verified against Euler-Bernoulli $\delta = -PL^3/(3EI)$
    plus the deflected-shape cubic.

</div>

## Building richer models

<div class="grid cards" markdown>

-   :material-numeric-4-circle:{ .lg .middle } &nbsp; __[Portal frame (2D)](notebooks/04_portal_frame_2D.ipynb)__

    ---

    *Strategy: `spec.capture`*

    Multi-element frame with **two element groups of different
    cross-section** (columns + rigid beam) driven from named
    physical groups. The first example where the FEM is *more*
    correct than the classical drift formula — the residual ~0.75%
    is inherent (axial deformation + joint kinematics).

-   :material-tag-multiple:{ .lg .middle } &nbsp; __[Labels & physical groups](notebooks/05_labels_and_pgs.ipynb)__

    ---

    *Strategy: `spec.capture` (small)*

    The two naming namespaces and how both flow through to `Results`.
    Declare recorders in **both** namespaces (`pg=` and `label=`),
    cross-check that the read side gives identical answers either
    way. Demonstrates the `target=` precedence order
    (label → PG → part).

-   :material-puzzle-outline:{ .lg .middle } &nbsp; __[Part assembly](notebooks/10b_part_assembly.ipynb)__

    ---

    *Strategy: `spec.emit_recorders`*

    Build a column once as a `Part`, instantiate three times via
    `g.parts.add(part, label=..., translate=...)`. Declare **one
    recorder per part label**, read each part's slab independently —
    part labels survive end-to-end as first-class selectors.

-   :material-vector-link:{ .lg .middle } &nbsp; __[Interface tie](notebooks/12_interface_tie.ipynb)__

    ---

    *Strategy: `spec.capture`*

    Two beams meeting at a shared point with **non-matching meshes**,
    joined via `g.constraints.equal_dof`. The notebook verifies the
    constraint contract directly through `Results` —
    `u_master` and `u_slave` at the junction are equal to round-off.
    Tip deflection matches the single-cantilever closed form.

</div>

## Analysis types

<div class="grid cards" markdown>

-   :material-sine-wave:{ .lg .middle } &nbsp; __[Modal analysis](notebooks/17_modal_analysis.ipynb)__

    ---

    *Strategy: `spec.capture` (`capture_modes`)*

    Eigenvalue analysis — `cap.capture_modes(N)` writes one stage per
    mode with `kind="mode"`. Read back via `results.modes`, with
    `mode_index`, `eigenvalue`, `frequency_hz`, `period_s` exposed as
    scoped properties. First three bending modes verified against
    Euler-Bernoulli to <0.3% error. **Modal is capture-only** — the
    classic recorder path can't drive `ops.eigen()`.

-   :material-chart-bell-curve:{ .lg .middle } &nbsp; __[Pushover (elastoplastic)](notebooks/19_pushover_elastoplastic.ipynb)__

    ---

    *Strategy: `spec.capture`*

    The first **multi-step nonlinear** notebook — a Steel01 bar
    pushed to 4× yield via `DisplacementControl`. The recorder fires
    at every converged step; the capacity curve drops out of two
    `Results.nodes.get(...)` calls without any manual `nodeDisp` /
    `nodeReaction` accumulation. Three closed-form checks (slope,
    plateau, yield onset) all match to round-off.

</div>

## Common shape across every notebook

```
1.  Imports + parameters
2.  Geometry              ← apeGmsh
3.  Physical groups + labels
4.  Mesh
5.  Build OpenSees model  ← VANILLA openseespy (ops.*)
6.  Declare recorders     ← apeGmsh
7.  Run analysis          ← spec.capture(...) or spec.emit_recorders(...)
8.  Read results back     ← Results
9.  Verify + plot         ← matplotlib from slabs
10. Optional viewer       ← results.viewer(blocking=False)
```

The "vanilla openseespy except for recorders" rule is visible in
section 5 (raw `ops.*` calls — nodes, elements, materials, fix,
load) vs section 6 onward (apeGmsh recorders, capture, Results,
viewer). The model setup stays close to OpenSees idioms; the
**results pipeline is where apeGmsh adds value**.

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
