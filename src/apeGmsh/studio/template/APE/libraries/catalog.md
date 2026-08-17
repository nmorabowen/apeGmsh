# ape* libraries — Studio toolchain catalog

Portable inventory of libraries that underpin **APE Studio** habitats.
Run:

```text
python APE/libraries/harvest.py
```

to refresh on-disk status (`catalog.json`) for the libraries listed in
`harvest.py`'s `LIBRARIES` list, from what it finds on **your** machine.
This file (`catalog.md`) is the human-edited door — update roles here by
hand when you add or remove libraries.

## Required infra (shipped)

Three libraries are wired directly into this template's code (`tools/`
launchers, `scripts/_habitat.py` PYTHONPATH default) and are genuine
dependencies, not personal recommendations:

| Library | Role |
|---------|------|
| **apeGmsh** | Gmsh FEM wrapper, `FEMData`, OpenSees bridge, Studio habitat (`.apegmsh/`, MCP). The whole habitat is built on it. |
| **apeCAD** | 3D spatial scratchpad / intent document (`tools/apeCAD`). |
| **apeSketch** | Hand-ink / scheme bridge (`tools/apeSketch`). |

## No further library catalog ships with this template

Any additional ape* libraries (steel or concrete design modules, seismic
ground-motion toolkits, BIM/analytical access points, and similar) are the
**template author's own personal stack** — they are not distributed or
required by this template. Add your own entries to `harvest.py`'s
`LIBRARIES` list if you have equivalents.

**Agent guidance:** if any such libraries happen to be installed in your
environment, use them; if not, ignore references to them and proceed —
only apeGmsh, apeCAD, and apeSketch are actually required.
