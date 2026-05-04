# API Reference

Auto-generated from module docstrings by [mkdocstrings](https://mkdocstrings.github.io/).

The reference is organized per top-level composite, mirroring the
session object's layout:

| Page | Access | Purpose |
|---|---|---|
| [Session](session.md)          | `apeGmsh(...)` | Top-level session lifecycle and composite wiring |
| [Model](model.md)              | `g.model`      | OCC geometry (`geometry`, `boolean`, `transforms`, `io`, `queries`) |
| [Parts](parts.md)              | `g.parts`      | Part registry and assembly bookkeeping |
| [Mesh](mesh.md)                | `g.mesh`       | Meshing (`generation`, `sizing`, `field`, `structured`, `editing`, `queries`, `partitioning`) |
| [Constraints](constraints.md)  | `g.constraints`| Solver-agnostic constraint defs, records, resolver |
| [Loads](loads.md)              | `g.loads`      | Load defs, records, resolver |
| [Masses](masses.md)            | `g.masses`     | Mass defs, records, resolver |
| [FEM Broker](fem.md)           | `g.mesh.queries.get_fem_data()` | `FEMData` snapshot container |
| [OpenSees](opensees.md)        | `g.opensees`   | OpenSees bridge (`materials`, `elements`, `ingest`, `inspect`, `export`) |
| [Results](results.md)          | `Results`      | Post-processing container |
| [Viewers](viewers.md)          |                | Qt/PyVista model and mesh viewers |
| [Viz](viz.md)                  | `g.plot`, etc. | Matplotlib, selection, VTK export |

!!! note
    Private members (names starting with `_`) are filtered out by
    default. Sub-composite implementation modules (`_mesh_generation`,
    `_model_geometry`, etc.) are documented under their parent
    composite page.

## Utilities

Top-level helpers re-exported from `apeGmsh` that don't have a
dedicated composite page:

- `apeGmsh.workdir` — context manager for the working directory.
- `apeGmsh.preview` — notebook viewer.
- `apeGmsh.settings`, `apeGmsh.theme_editor` — viewer preferences and
  theme editor (see [Viewers](viewers.md)).
- `SelectionPicker` is a back-compat alias for `ModelViewer`.
