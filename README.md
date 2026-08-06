# apeGmsh

[![Docs](https://github.com/nmorabowen/apeGmsh/actions/workflows/docs.yml/badge.svg)](https://nmorabowen.github.io/apeGmsh/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21815489.svg)](https://doi.org/10.5281/zenodo.21815489)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Structural-FEM wrapper around [Gmsh](https://gmsh.info) with a composition-based API and a
snapshot FEM broker. Designed to make it cheap to describe a model
once (geometry + physical groups + loads + constraints) and feed it
to any solver. OpenSees has first-class support; other solvers can be
plugged in through the same `FEMData` contract.

[![A double-couple seismic source radiating through a solid block — see it move on the docs site](https://nmorabowen.github.io/apeGmsh/assets/anim/hero-poster.jpg)](https://nmorabowen.github.io/apeGmsh/)
*A double-couple seismic source radiating through a solid block — one of the
[animated showcase models](https://nmorabowen.github.io/apeGmsh/); the script
that solves and renders it is 165 lines.*

**New to the library?** Start with the
[**learning path**](https://nmorabowen.github.io/apeGmsh/tutorials/learning-path/) —
every tutorial and worked example in reading order, each checked against a
known answer — or jump straight into
[**your first model in 10 minutes**](https://nmorabowen.github.io/apeGmsh/tutorials/first-model/).

**Documentation:** <https://nmorabowen.github.io/apeGmsh/>

> [!NOTE]
> **Built on Gmsh.** apeGmsh is a wrapper built on top of the (awesome)
> [Gmsh](https://gmsh.info) Python API. It adds a set of abstractions over the
> main API to fit an intended structural-FEM workflow — parts, constraints,
> loads, masses, and an OpenSees bridge. You still have the full Gmsh API
> underneath whenever you need it.

## Installation

```bash
pip install apeGmsh
```

Or with every optional dependency — the OpenSees bridge, the Qt and web
viewers, and plotting:

```bash
pip install "apeGmsh[all]"
```

To track unreleased work, install from the repo instead:

```bash
pip install "apeGmsh[all] @ git+https://github.com/nmorabowen/apeGmsh.git@main"
```

Or clone for editable development:

```bash
git clone https://github.com/nmorabowen/apeGmsh.git
cd apeGmsh
pip install -e ".[all]"
```

Requires Gmsh (with Python bindings), NumPy, and Pandas. Optional
extras: `matplotlib` (plotting), `openseespy` (analysis),
`pyvista` + `PySide6` (Qt and web viewers).

## Quick start

```python
from apeGmsh import apeGmsh

with apeGmsh(model_name="plate") as g:
    # 1. Geometry — sub-composites under g.model
    p1 = g.model.geometry.add_point(0,   0,  0, lc=10)
    p2 = g.model.geometry.add_point(100, 0,  0, lc=10)
    p3 = g.model.geometry.add_point(100, 50, 0, lc=10)
    p4 = g.model.geometry.add_point(0,   50, 0, lc=10)
    l1 = g.model.geometry.add_line(p1, p2)
    l2 = g.model.geometry.add_line(p2, p3)
    l3 = g.model.geometry.add_line(p3, p4)
    l4 = g.model.geometry.add_line(p4, p1)
    loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
    surf = g.model.geometry.add_plane_surface(loop)

    # 2. Mesh — sub-composites under g.mesh
    g.mesh.sizing.set_global_size(10)
    g.mesh.generation.generate(dim=2)
    g.mesh.partitioning.renumber(dim=2, method="simple", base=1)

    # 3. Snapshot for the solver
    fem = g.mesh.queries.get_fem_data(dim=2)
    print(fem.info)
```

On import, apeGmsh prints an ASCII banner with the version to
`stderr`. Set `APEGMSH_QUIET=1` to suppress it (useful for tests
and CI).

## How it fits together

You describe a model inside an `apeGmsh` session — one Gmsh kernel
fronted by focused composites (`g.model` for geometry, `g.mesh` for
meshing, plus physical groups, parts, constraints, loads, and masses).
Definitions reference *labels*, not raw mesh tags, so you can declare
a load on "TopFace" before a single node exists. Once the mesh is
generated, `get_fem_data()` resolves every definition against it and
returns a frozen `FEMData` snapshot — nodes, elements, and all loads,
masses, and constraints as plain nodal records. That snapshot is the
single contract between Gmsh and any downstream solver.

For OpenSees, the snapshot feeds `apeSees(fem)`, a typed bridge built
after the session closes: typed constructors for materials, sections,
and elements; MP constraints emitted automatically; loads pulled in
per pattern with `p.from_model(case)`. The bridge emits a runnable
Tcl or Python deck — or runs openseespy in-process — and solver output
comes back as a `Results` object with interactive and web viewers.
The [mental model](https://nmorabowen.github.io/apeGmsh/concepts/mental-model/)
page walks through these ideas properly; the
[design notes](https://nmorabowen.github.io/apeGmsh/design/) cover the
internals; the site's API reference owns the full method inventory.

## Credits

**Developed by:** Nicolás Mora Bowen · Patricio Palacios · José Abell · Guppi

Part of José Abell's *El Ladruño Research Group*.

## License

apeGmsh is licensed under the **GNU General Public License v3.0 or later**
(see [LICENSE](LICENSE)).

GPL rather than a permissive license because apeGmsh builds on
[Gmsh](https://gmsh.info), which is GPLv2+ — its linking exception covers only
Netgen, METIS, OpenCASCADE and ParaView, not downstream API consumers.
[pygmsh](https://github.com/nschloe/pygmsh) is GPLv3+ for the same reason.

Note that the GPL constrains **distribution**, not use: running apeGmsh
internally — including on commercial engineering projects — carries no
source-disclosure obligation.

## Citing

If you use apeGmsh in published work, please cite it via its DOI:

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

That is the **concept DOI** — it always resolves to the most recent release, so
prefer it unless you need to pin the exact version you ran. For that, use the
version DOI shown on the
[Zenodo record](https://doi.org/10.5281/zenodo.21815489) (v2.0.0 is
[10.5281/zenodo.21815490](https://doi.org/10.5281/zenodo.21815490)).

Machine-readable metadata lives in [CITATION.cff](CITATION.cff) — GitHub's
"Cite this repository" button reads it directly.
