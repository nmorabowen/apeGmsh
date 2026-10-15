"""KNOWN-BAD (the parts are placed but never fragmented, so the contact face is meshed twice).

A column on a footing, built as two Parts and assembled into one conformal mesh.

A 0.5 m square column, 3 m tall, stands at the centre of a 2 m square footing,
0.5 m thick. Each is its own Part; the assembly fragments them so the column
and the footing share the nodes of their contact face. The mesh is tetrahedra
of about 0.25 m, saved to model.h5 beside this script with one volume group
per part.
Units: m. z is up; the footing top is z = 0.

Procedure:
    1. Data
    2. Parts: the column and the footing, each in its own session
    3. Assembly: both parts placed, then fragmented into one conformal body
    4. Groups: one volume group per part
    5. Mesh: tetrahedra of the target size, saved to model.h5
    6. Checks and report: the column and footing volumes, the shared nodes
"""
from pathlib import Path

from apeGmsh import Part, apeGmsh

# --- 1. Data
OUT = Path(__file__).parent / "model.h5"

COLUMN_SIDE = 0.5         # m, square column
COLUMN_HEIGHT = 3.0       # m, from the footing top
FOOTING_SIDE = 2.0        # m, square footing
FOOTING_THICKNESS = 0.5   # m
ELEMENT_SIZE = 0.25       # m, target element size

HALF_COLUMN = COLUMN_SIDE / 2     # m, column centre to face
HALF_FOOTING = FOOTING_SIDE / 2   # m, footing centre to edge

# Both bodies are boxes, so the tetrahedra fill them exactly; only round-off remains.
TOL_VOLUME = 1.0e-9       # relative, meshed vs exact volume

# --- 2. Parts
with Part("column") as column:
    column.model.geometry.add_box(
        -HALF_COLUMN, -HALF_COLUMN, 0.0, COLUMN_SIDE, COLUMN_SIDE, COLUMN_HEIGHT, label="column",
    )

with Part("footing") as footing:
    footing.model.geometry.add_box(
        -HALF_FOOTING, -HALF_FOOTING, -FOOTING_THICKNESS,
        FOOTING_SIDE, FOOTING_SIDE, FOOTING_THICKNESS, label="footing",
    )

with apeGmsh(model_name="column_on_footing", save_to=str(OUT)) as g:
    # --- 3. Assembly
    g.parts.add(column, label="column")
    g.parts.add(footing, label="footing")

    # --- 4. Groups
    g.physical.add_volume("column", name="column")
    g.physical.add_volume("footing", name="footing")

    # --- 5. Mesh
    g.mesh.sizing.set_global_size(ELEMENT_SIZE)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

# --- 6. Checks and report
# Node coordinates by id, then each tetrahedron's volume from its four corners.
node_xyz = dict(fem.nodes.select(pg=["column", "footing"]).result())
meshed_volume = {}
for part in ("column", "footing"):
    meshed_volume[part] = 0.0     # m3, sum over the tetrahedra
    for tet in fem.elements.select(pg=part).connectivity:
        edge_a = node_xyz[tet[1]] - node_xyz[tet[0]]
        edge_b = node_xyz[tet[2]] - node_xyz[tet[0]]
        edge_c = node_xyz[tet[3]] - node_xyz[tet[0]]
        triple = edge_a[0] * (edge_b[1] * edge_c[2] - edge_b[2] * edge_c[1])
        triple -= edge_a[1] * (edge_b[0] * edge_c[2] - edge_b[2] * edge_c[0])
        triple += edge_a[2] * (edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0])
        meshed_volume[part] += abs(triple) / 6

column_volume = COLUMN_SIDE**2 * COLUMN_HEIGHT              # m3, exact
footing_volume = FOOTING_SIDE**2 * FOOTING_THICKNESS        # m3, exact
column_error = meshed_volume["column"] / column_volume - 1
footing_error = meshed_volume["footing"] / footing_volume - 1
assert abs(column_error) < TOL_VOLUME, f"column volume off by {column_error:.1e}"
assert abs(footing_error) < TOL_VOLUME, f"footing volume off by {footing_error:.1e}"

# Conformal: the column and the footing share the nodes of their contact face.
column_nodes = set(fem.nodes.select(pg="column").ids)
footing_nodes = set(fem.nodes.select(pg="footing").ids)
shared_nodes = column_nodes & footing_nodes

print(f"Mesh: {len(fem.nodes)} nodes, {len(fem.elements)} elements")
print(f"Column volume {meshed_volume['column']:.6f} m3 ({column_error:+.1e})")
print(f"Footing volume {meshed_volume['footing']:.6f} m3 ({footing_error:+.1e})")
print(f"Nodes shared on the contact face: {len(shared_nodes)}")
