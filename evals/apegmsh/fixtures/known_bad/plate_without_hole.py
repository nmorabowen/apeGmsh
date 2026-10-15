"""KNOWN-BAD (the circle is drawn and grouped but never cut from the plate, so the plate has no hole).

Rectangular plate with a central circular hole: a labelled 2-D mesh.

A 10 m by 5 m plate with a hole of radius 1 m at its centre. The mesh is
triangles of about 0.25 m. The edges a later analysis will need (the two
short sides and the hole) are named groups, and the model is saved to
model.h5 beside this script.
Units: m.

Procedure:
    1. Data
    2. Geometry: the outline corners and sides, the hole, the plate surface
    3. Groups: the plate, its left and right sides, the hole edge
    4. Mesh: triangles of the target size, saved to model.h5
    5. Checks and report: the meshed area against the plate area
"""
from pathlib import Path

from apeGmsh import apeGmsh

# --- 1. Data
OUT = Path(__file__).parent / "model.h5"

LENGTH = 10.0             # m, plate length along x
HEIGHT = 5.0              # m, plate height along y
HOLE_RADIUS = 1.0         # m, radius of the central hole
ELEMENT_SIZE = 0.25       # m, target element size
CENTRE_X = LENGTH / 2     # m, hole centre
CENTRE_Y = HEIGHT / 2     # m, hole centre

# The hole edge is meshed as a polygon, so the meshed area exceeds the exact area
# by a fraction of the hole area; 0.5 % of the plate is several times that.
TOL_AREA = 0.005          # relative, meshed vs exact plate area

with apeGmsh(model_name="plate_with_hole", save_to=str(OUT)) as g:
    geo = g.model.geometry

    # --- 2. Geometry
    geo.add_point(0.0, 0.0, 0.0, label="corner_bottom_left")
    geo.add_point(LENGTH, 0.0, 0.0, label="corner_bottom_right")
    geo.add_point(LENGTH, HEIGHT, 0.0, label="corner_top_right")
    geo.add_point(0.0, HEIGHT, 0.0, label="corner_top_left")
    geo.add_line("corner_bottom_left", "corner_bottom_right", label="bottom_edge")
    geo.add_line("corner_bottom_right", "corner_top_right", label="right_edge")
    geo.add_line("corner_top_right", "corner_top_left", label="top_edge")
    geo.add_line("corner_top_left", "corner_bottom_left", label="left_edge")
    geo.add_curve_loop(["bottom_edge", "right_edge", "top_edge", "left_edge"], label="outline")
    geo.add_circle(CENTRE_X, CENTRE_Y, 0.0, HOLE_RADIUS, label="hole_edge")
    geo.add_plane_surface("outline", label="plate")

    # --- 3. Groups
    g.physical.add_surface("plate", name="plate")
    g.physical.add_curve("left_edge", name="left_edge")
    g.physical.add_curve("right_edge", name="right_edge")
    g.physical.add_curve("hole_edge", name="hole_edge")

    # --- 4. Mesh
    g.mesh.sizing.set_global_size(ELEMENT_SIZE)
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)

# --- 5. Checks and report
plate_area = LENGTH * HEIGHT   # m2, the outline only
# Node coordinates by id, then each triangle's area from its three corners.
node_xyz = dict(fem.nodes.select(pg="plate").result())
meshed_area = 0.0                                         # m2, sum over the triangles
for corner_a, corner_b, corner_c in fem.elements.select(pg="plate").connectivity:
    ab = node_xyz[corner_b] - node_xyz[corner_a]
    ac = node_xyz[corner_c] - node_xyz[corner_a]
    cross_z = ab[0] * ac[1] - ab[1] * ac[0]
    meshed_area += abs(cross_z) / 2
area_error = meshed_area / plate_area - 1
assert abs(area_error) < TOL_AREA, f"meshed area off by {area_error:.2%}"
print(f"Mesh: {len(fem.nodes)} nodes, {len(fem.elements)} elements")
print(f"Area: meshed {meshed_area:.4f} m2, exact {plate_area:.4f} m2 ({area_error:+.2%})")
