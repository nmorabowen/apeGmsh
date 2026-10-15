"""KNOWN-BAD (the api-cheatsheet.md field fence at 6308b303: distance(edges=), threshold(input_field=); the live keywords are curves= and a positional field).

Graded mesh of a soil block under a strip footing (the staged_footing mesh).

A 2 m wide strip footing sits at the centre of the top of a 20 m wide, 10 m
deep soil block. The mesh is 6-node triangles, 0.25 m along the footing and
coarsening to 1.0 m towards the far boundaries. The model is saved to model.h5
beside this script, with named groups for the soil, its base, its two sides,
the footing strip and the footing centre.
Units: m. x is horizontal, y is up, the ground is y = 0 and the footing centre is x = 0.

Procedure:
    1. Data: geometry and mesh sizes.
    2. Geometry: the block outline, with the footing edges and centre as points.
    3. Groups: soil, base, sides, footing, footing centre.
    4. Mesh: 6-node triangles graded from the footing outwards, saved to model.h5.
    5. Checks and report: the meshed area against the block area.
"""
from pathlib import Path

from apeGmsh import apeGmsh

# --- 1. Data
OUT = Path(__file__).parent / "model.h5"

SOIL_WIDTH = 20.0       # m, block width
SOIL_DEPTH = 10.0       # m, block depth
FOOTING_WIDTH = 2.0     # m, footing width
HALF_SOIL = SOIL_WIDTH / 2         # m, centre to side
HALF_FOOTING = FOOTING_WIDTH / 2   # m, centre to footing edge

SIZE_FOOTING = 0.25     # m, element size along the footing (8 quadratic elements)
SIZE_FAR = 1.0          # m, element size at the far boundaries
DIST_FINE = 1.0         # m, keep the fine size this far from the footing
DIST_COARSE = 8.0       # m, reach the far size this far from the footing

# The block has straight edges, so the triangle corners tile it exactly; only
# round-off separates the meshed area from the block area.
TOL_AREA = 1.0e-9       # relative, meshed vs exact block area

with apeGmsh(model_name="footing_mesh", save_to=str(OUT)) as g:
    geo = g.model.geometry

    # --- 2. Geometry: outline points at z = 0 (a 2-D model), then the seven edges
    geo.add_point(-HALF_SOIL, -SOIL_DEPTH, 0.0, label="base_left")
    geo.add_point(HALF_SOIL, -SOIL_DEPTH, 0.0, label="base_right")
    geo.add_point(HALF_SOIL, 0.0, 0.0, label="ground_right")
    geo.add_point(HALF_FOOTING, 0.0, 0.0, label="footing_right_edge")
    geo.add_point(0.0, 0.0, 0.0, label="footing_centre")
    geo.add_point(-HALF_FOOTING, 0.0, 0.0, label="footing_left_edge")
    geo.add_point(-HALF_SOIL, 0.0, 0.0, label="ground_left")

    # Counter-clockwise; the footing is split at its centre so the centre is a node.
    geo.add_line("base_left", "base_right", label="base")
    geo.add_line("base_right", "ground_right", label="right_side")
    geo.add_line("ground_right", "footing_right_edge", label="ground_right_surface")
    geo.add_line("footing_right_edge", "footing_centre", label="footing_right_half")
    geo.add_line("footing_centre", "footing_left_edge", label="footing_left_half")
    geo.add_line("footing_left_edge", "ground_left", label="ground_left_surface")
    geo.add_line("ground_left", "base_left", label="left_side")
    geo.add_curve_loop([
        "base", "right_side", "ground_right_surface", "footing_right_half",
        "footing_left_half", "ground_left_surface", "left_side",
    ], label="soil_outline")
    geo.add_plane_surface("soil_outline", label="soil")

    # --- 3. Groups
    g.physical.add_surface("soil", name="soil")
    g.physical.add_curve("base", name="base")
    g.physical.add_curve(["left_side", "right_side"], name="sides")
    g.physical.add_curve(["footing_left_half", "footing_right_half"], name="footing")
    g.physical.add_point("footing_centre", name="footing_centre")

    # --- 4. Mesh: 6-node triangles graded from the footing outwards
    near_footing = g.mesh.field.distance(edges=["footing_left_half", "footing_right_half"])
    grading = g.mesh.field.threshold(
        input_field=near_footing, size_min=SIZE_FOOTING, size_max=SIZE_FAR,
        dist_min=DIST_FINE, dist_max=DIST_COARSE,
    )
    g.mesh.field.set_background(grading)
    g.mesh.generation.generate(dim=2)
    # Quadratic: linear triangles lock under the isochoric flow a clay analysis needs.
    g.mesh.generation.set_order(2)
    fem = g.mesh.queries.get_fem_data(dim=2)

# --- 5. Checks and report
block_area = SOIL_WIDTH * SOIL_DEPTH   # m2, exact
# Node coordinates by id, then each triangle's area from its three corner nodes
# (the first three of the six).
node_xyz = dict(fem.nodes.select(pg="soil").result())
meshed_area = 0.0                      # m2, sum over the triangles
for triangle in fem.elements.select(pg="soil").connectivity:
    ab = node_xyz[triangle[1]] - node_xyz[triangle[0]]
    ac = node_xyz[triangle[2]] - node_xyz[triangle[0]]
    cross_z = ab[0] * ac[1] - ab[1] * ac[0]
    meshed_area += abs(cross_z) / 2
area_error = meshed_area / block_area - 1
assert abs(area_error) < TOL_AREA, f"meshed area off by {area_error:.1e}"
print(f"Mesh: {len(fem.nodes)} nodes, {len(fem.elements)} elements")
print(f"Area: meshed {meshed_area:.6f} m2, exact {block_area:.6f} m2 ({area_error:+.1e})")
