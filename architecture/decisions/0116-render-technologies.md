# ADR 0116 — Render technologies: three.js in apeGmshViewer and matplotlib; VTK in sunset

**Status:** Proposed (2026-10-05). The maintainer accepts it by merging the
X1-c slice (#1483), which is a human gate (PROGRAM.md §4).

**Owner:** nmora

**Evidence:** the expert panel's cut table
(`internal_docs/plan_expert_panel_2026-09.md`, row K21, voted 7–0) cut the
plotly `NotebookPreview` and the three.js `geom_transf_viewer` and asked for
a two-render-technology rule of "VTK + matplotlib". [ADR
0112](0112-files-are-the-model-and-a-read-only-app.md) then replaced that
pair: its table *Relation to the remediation program* supersedes K21 and
says that the X1 ADR records three.js (apeGmshViewer) and matplotlib, with
VTK in sunset. The maintainer restated this on the chain X issue (#1199).
This ADR records that rule and nothing more.

**Builds on** ADR 0112 (D5, D8 and the relation table). **Program link:**
X1, slice c (#1483).

## Context

Before this slice, `src/apeGmsh` drew pictures with four technologies:

| Technology | Where | Purpose |
|---|---|---|
| VTK (pyvista, pyvistaqt) | `viewers/` only | the Qt viewers, the headless stills (`render`, ADR 0094), the trame web viewer |
| matplotlib | `viz/Plot.py` (`g.plot`), `sections/`, `results/plot/`, the Qt panels | static 2-D figures and inline notebook pictures |
| plotly | `viz/NotebookPreview.py` | `apeGmsh.preview`, `g.model.preview`, `g.mesh.preview` |
| three.js | `viewers/geom_transf_viewer.py` | `GeomTransfViewer`, a standalone browser page for beam local axes |

Every technology carries its own colours, camera and frame conventions, and
each one needs its own tests. The cost showed up in the three.js page: it
computed the beam local frame again in JavaScript and drew local y and z
negated relative to OpenSees, until #1186 fixed it. That was a second copy
of a rule that `viewers/diagrams/_beam_geometry.py` already held.

ADR 0112 makes apeGmshViewer (TypeScript and three.js, D5) the one
application that shows a model to a person, and freezes the Qt viewers to
bug fixes until they are deleted at its phase P4 (D8).

## Decision

### D1 — Two target technologies

New rendering code uses one of two technologies:

- **three.js, inside apeGmshViewer**, for every interactive 3-D view of a
  model, mesh or result.
- **matplotlib**, for static figures and for pictures drawn from Python,
  inline in a notebook included.

### D2 — apeGmshViewer is the only three.js surface

No three.js page, widget or HTML template lives in `src/apeGmsh`. A view
that needs three.js is a window of apeGmshViewer. `GeomTransfViewer` is
removed on this basis; a transform-orientation view comes back, if anyone
wants it, as an apeGmshViewer window (ADR 0112, relation table).

### D3 — VTK is in sunset

VTK (pyvista, pyvistaqt, `vtk`) stays allowed only where it is today:
the frozen Qt viewers and the headless stills in `src/apeGmsh/viewers/`.
New code outside that package does not import it. VTK leaves with the Qt
viewers at ADR 0112 P4, and the stills renderer leaves once apeGmshViewer
produces the same stills headlessly (ADR 0112, open questions).

Writing VTK **files** (`viz/VTKExport.py`, `.vtu` for ParaView) is a file
format, not a render technology. This ADR does not cover it.

### D4 — No other technology enters without an ADR

plotly, bokeh, k3d, pythreejs, trame and the like do not enter `src/` without
a superseding ADR. The plotly preview is removed under this rule. trame
leaves with the legacy results viewer (chain X, X3).

## Consequences

- **Removed in X1-c:** `apeGmsh.viz.NotebookPreview`, the top-level
  `apeGmsh.preview`, `Model.preview` and `Mesh.preview`; and
  `apeGmsh.viewers.geom_transf_viewer.GeomTransfViewer` with its re-export.
  The #1186 frame fix is moot. The frame rule lives on in
  `viewers/diagrams/_beam_geometry.py` (`compute_local_axes`) and
  `opensees/_orientation.py`, with their own tests.
- The same slice also cuts `apeGmsh.sensitivity` and the import banner
  (panel P2). Those are scope cuts, not render decisions, and this ADR only
  lists them for traceability.
- Inline pictures in a notebook use `g.plot` (matplotlib) or a `render(...)`
  still.

## Not built

- No lint enforces D3 or D4 yet. Until one exists, review enforces them, and
  ADR 0112's shrink-only ratchet on `src/apeGmsh/viewers/` (chain X, X4)
  keeps the VTK footprint from growing.

## Relation to earlier decisions

- **Panel K21** (VTK + matplotlib, never an ADR): superseded, as ADR 0112
  already decided.
- **ADR 0112:** consistent. This ADR writes down the rule from 0112's
  relation table and changes nothing in 0112.
- **ADR 0094** (agent assess and viewer render): its stills stay under D3
  until apeGmshViewer replaces them.
