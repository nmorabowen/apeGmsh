"""
Two adjacent boxes — unstructured (tet) and transfinite (hex) meshes.

Geometry
--------
  box   : [0,5]  × [0,5]  × [0,5]   (target size 1.0)
  box_2 : [5,7.5]× [0,2.5]× [0,5]   (target size 0.3)

The boxes are adjacent (touching at x=5) but do not overlap.

Unstructured section
  Uses fragment() to create a conformal interface, then per-region
  element-size targets → tetrahedra.

Transfinite section
  Each box is meshed independently as a clean 6-face hex (no fragment,
  so no topology complications).  Node counts per edge are derived from
  edge length / target_size.  Surfaces are recombined to quads so the
  volume becomes all-hex.

  Note: without fragment the interface at x=5 is non-conformal (node
  counts differ between the two regions).  For a conformal transfinite
  interface you need matching node counts on every shared edge, which
  means both regions must use the same effective size along those edges.
"""

from apeGmsh import apeGmsh


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _apply_transfinite_hex(m, vol: int, target_size: float) -> None:
    """Set transfinite + recombine constraints on a single clean-box volume.

    Walks the boundary curves, computes node count from edge length, then
    marks every bounding surface as transfinite+recombined and the volume
    as transfinite.
    """
    # Curves: node count = round(length / target_size) + 1
    faces = m.model.queries.boundary(vol, oriented=False)
    for _, ctag in dict.fromkeys(
        m.model.queries.boundary(faces, combined=False, oriented=False)
    ):
        xmin, ymin, zmin, xmax, ymax, zmax = m.model.queries.bounding_box(ctag, dim=1)
        L = max(xmax - xmin, ymax - ymin, zmax - zmin)
        n = max(2, round(L / target_size) + 1)
        m.mesh.structured.set_transfinite_curve(ctag, n)

    # Surfaces → quads (recombine), then volume → hex
    for _, stag in faces:
        m.mesh.structured.set_transfinite_surface(stag)
        m.mesh.structured.set_recombine(stag, dim=2)

    m.mesh.structured.set_transfinite_volume(vol)


# ─────────────────────────────────────────────────────────────────────────────
# 1) UNSTRUCTURED (tetrahedra) — conformal interface via fragment
# ─────────────────────────────────────────────────────────────────────────────

m = apeGmsh(model_name='box_tet', verbose=False)
m.begin()

m.model.geometry.add_box(0, 0, 0, 5,   5,   5, label='box')
m.model.geometry.add_box(5, 0, 0, 2.5, 2.5, 5, label='box_2')
m.model.boolean.fragment(['box'], ['box_2'])

m.mesh.sizing.set_size('box',   size=1.0)
m.mesh.sizing.set_size('box_2', size=0.3)
m.mesh.generation.generate(dim=3)
m.mesh.viewer()

m.end()


# ─────────────────────────────────────────────────────────────────────────────
# 2) TRANSFINITE (hexahedra) — structured grid per volume
# ─────────────────────────────────────────────────────────────────────────────

m = apeGmsh(model_name='box_hex', verbose=False)
m.begin()

vol_box   = m.model.geometry.add_box(0, 0, 0, 5,   5,   5, label='box')
vol_box_2 = m.model.geometry.add_box(5, 0, 0, 2.5, 2.5, 5, label='box_2')
# No fragment: keeps each box as a clean 6-face hex (required for transfinite volume)

_apply_transfinite_hex(m, vol_box,   target_size=1.0)
_apply_transfinite_hex(m, vol_box_2, target_size=0.3)

m.mesh.generation.generate(dim=3)
m.mesh.viewer()

m.end()
