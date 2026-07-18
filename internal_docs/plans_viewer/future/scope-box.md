# Future: Scope Box

**Status:** future · **Estimated cost:** ~3d · **Depends on:** plan 04
**Discussed in:** [`../region-tools-comparison.md`](../region-tools-comparison.md)

## Goal

A 3D rectangular region widget that filters visible cells in `mesh.viewer` and
`results.viewer` — show only elements whose centroid is inside the box.

## Why

- Today: nothing. Users navigate around irrelevant geometry by camera tricks or by
  using physical groups to subset the view.
- Standard CAD/BIM/FEM tool: Revit calls them "Scope Boxes." Every serious FEM
  postprocessor has them.

## Mechanism

- PyVista's `add_box_widget(callback)` (or `vtkBoxWidget` directly). User gets 6 drag
  handles + 3 rotation handles in 3D.
- Cell mask: compute centroids of all cells once at attach; on widget callback, recompute
  inside/outside via `vtkBox.EvaluateFunction`; flip per-cell visibility.
- Persisted in session JSON (box origin + bounds + rotation + active state).

## Files

- New: `src/apeGmsh/viewers/overlays/scope_box_overlay.py`
- New: `src/apeGmsh/viewers/ui/_scope_box_panel.py` — toggle + bounds spinners.
- Modify: `src/apeGmsh/viewers/diagrams/_session.py` — persist scope box state.

## Risks

- **Performance**: cell-centroid recompute on every drag may stutter for large meshes.
  Throttle on widget release, or precompute centroids once.
- **Deformation interaction**: filter on undeformed centroids or deformed? Default to
  undeformed (cheaper, more predictable). Document.
- **Widget vs camera controls** can fight in PyVista. Disable orbit during widget drag.

## Done criteria

- User can toggle a scope box in `results.viewer`; only cells inside render.
- Drag handles work; bounds editable via spinners in panel.
- Box state persists across viewer launches.
- Works for both `mesh.viewer` and `results.viewer`.

## Out of scope

- Multiple simultaneous scope boxes (union / intersection).
- Cylinder / sphere scope shapes.
- Scope-box-driven element filter for `section_cut` integration.
- Scope box in `model.viewer` (BRep — different actor source).
