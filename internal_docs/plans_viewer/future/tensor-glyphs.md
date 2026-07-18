# Future: Tensor Glyphs (Principal Directions)

**Status:** future · **Estimated cost:** ~3–4d · **Depends on:** plan 04
**Discussed in:** [`../diagram-catalog-comparison.md`](../diagram-catalog-comparison.md)

## Goal

Render principal stress (or strain) directions at Gauss points or element centers as
glyphs — ellipsoids, three-axis crosses, or arrow triplets.

## Why

- `_kind_catalog.py:113` explicitly skips tensor suffixes. So when a recording has
  `stress_xx`/`yy`/`zz`/`xy`/`yz`/`xz`, the user can color by each component but cannot
  see principal directions.
- Basic expectation for solid stress analysis.

## Mechanism

- `vtkTensorGlyph` filter. Input: `vtkPolyData` of seed points + a 6-or-9 component
  tensor array. Output: glyphed geometry positioned and oriented by the tensor's
  eigenvectors, scaled by eigenvalues.
- Three glyph types selectable in style: ellipsoid (default), three-axis cross, arrow
  triplet.
- Color by `max principal magnitude` or by `deviator`.

## Files

- New: `src/apeGmsh/viewers/diagrams/_tensor_glyph.py`
- Modify: `_styles.py` — add `TensorGlyphStyle` (glyph_type, scale_factor, color_mode).
- Modify: `_kind_catalog.py` — add `tensor_glyph` kind; populate data options from
  tensor-suffix prefixes (e.g., `stress`, `strain`).

## Risks

- **6-vs-9 component convention.** Verify how OpenSees records tensors (typically the
  6 upper-triangle components). vtkTensorGlyph expects 9 — pad with symmetric off-diagonals.
- **Glyph scaling.** Too small → invisible; too large → cluttered. Auto-scale based on
  data range + model diagonal + manual override.

## Done criteria

- New `tensor_glyph` kind available in Add-Diagram dialog when tensor data exists.
- Three glyph types selectable in style.
- Ellipsoids correctly oriented (eigenvectors) and scaled (eigenvalues).
- Tested against a uniaxial tension reference case.

## Out of scope

- Strain energy density visualization.
- Animation of principal direction rotation over time.
- Hyperstreamlines (continuous principal-direction paths through the field).
