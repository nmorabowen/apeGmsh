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
pip install "apeGmsh[all] @ git+https://github.com/nmorabowen/apeGmsh.git@main"
```

Not on PyPI yet — the line above installs straight from the repo. The
`[all]` extra pulls in the OpenSees bridge (via
[openseespy](https://pypi.org/project/openseespy/)), the web viewer, and
plotting — everything the tutorials use. Want just the modelling core?
Drop the `[all]`.
<video autoplay muted loop playsinline width="100%">
  <source src="assets/anim/moment-tensor.mp4" type="video/mp4">
</video>
<p style="margin-top:0.3em"><em>A double-couple seismic source radiating through a solid block — solved and rendered by the 165-line script <code>scripts/render_showcase/moment_tensor.py</code>.</em></p>


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

-   :material-shape-plus: &nbsp; **Section properties, in-process**

    ---

    Draw any cross-section — built-up plates, an SRC column, a holed
    box — mesh the face, and read A, I, J, shear centre, warping and
    plastic moduli straight off it. Bind the analyzer to the bridge
    and the emitted deck always follows the drawn geometry.

    [Compute section properties →](how-to/section-properties.md)

-   :material-vector-combine: &nbsp; **Contact, declared like everything else**

    ---

    `g.constraints.contact(master, slave)` puts node-to-segment or
    mortar contact — friction, mesh-tying, the lot — on the same
    declare-then-resolve pipeline as every tie and diaphragm.

    [How constraints couple a model →](concepts/constraints.md)

-   :material-waveform: &nbsp; **Staged SSI, end to end**

    ---

    Settle a soil column under gravity, freeze it, flip the boundary
    to absorbing, and shake the equilibrated state — the staged
    workflow that used to take a folder of hand-written decks.

    [The staged SSI example →](examples/staged-gravity-ssi.md)

-   :material-school: &nbsp; **The docs are now a course**

    ---

    Every tutorial and example sits on one ordered staircase, each
    rung checked against a known answer — and the showcase models
    now move.

    [Walk the learning path →](tutorials/learning-path.md)

</div>


---

## Credits

**Developed by:** Nicolás Mora Bowen · Patricio Palacios · José Abell · Guppi

Part of José Abell's *El Ladruño Research Group*.

---

## Citing apeGmsh

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21815489.svg)](https://doi.org/10.5281/zenodo.21815489)

If apeGmsh contributed to published work, please cite it:

> Mora Bowen, N., Palacios, P., Abell, J., & Guppi. *apeGmsh: a structural-FEM
> wrapper around Gmsh with a snapshot FEM broker.* Zenodo.
> <https://doi.org/10.5281/zenodo.21815489>

```bibtex
@software{apeGmsh,
  author    = {Mora Bowen, Nicolás and Palacios, Patricio and Abell, José and Guppi},
  title     = {apeGmsh: a structural-FEM wrapper around Gmsh with a snapshot FEM broker},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21815489},
  url       = {https://doi.org/10.5281/zenodo.21815489}
}
```

!!! tip "Which DOI?"
    The DOI above is the **concept DOI** — it always resolves to the most
    recent release, which is what you want when citing the software itself.
    To pin the exact version you ran, use that release's version DOI instead
    (v2.0.0 is [10.5281/zenodo.21815490](https://doi.org/10.5281/zenodo.21815490)).

## License

apeGmsh is licensed under the
[GNU General Public License v3.0 or later](https://github.com/nmorabowen/apeGmsh/blob/main/LICENSE).

GPL rather than a permissive license because apeGmsh builds on
[Gmsh](https://gmsh.info), which is GPLv2+ — its linking exception covers only
Netgen, METIS, OpenCASCADE and ParaView, not downstream API consumers.
[pygmsh](https://github.com/nschloe/pygmsh) is GPLv3+ for the same reason.

Note the GPL constrains **distribution**, not use: running apeGmsh internally,
including on commercial engineering projects, carries no source-disclosure
obligation.
