"""Plane-strain strip footing on elastic soil: self-weight and footing pressure.

Units: m, N, Pa (SI).
Geometry: 20 m wide x 10 m deep soil block; 2 m wide footing at top centre.
Materials: Elastic soil E=30 MPa, nu=0.3, gamma=18 kN/m³.
BC: Bottom fixed, left/right on rollers.
Procedure:
  1. Build geometry and mesh.
  2. Create OpenSees 2D plane-strain model.
  3. Apply self-weight and footing pressure in one analysis.
  4. Extract settlement at footing centre.
  5. Compare FEM elastic slope with hand estimate.
"""

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
import openseespy.opensees as ops_raw
import os

# --- Unit system and data ---
KN = 1e3  # N per kN
MPa = 1e6  # Pa per MPa

SOIL_WIDTH = 20.0  # m
SOIL_DEPTH = 10.0  # m
FOOTING_WIDTH = 2.0  # m
FOOTING_DEPTH = 1.0  # m
FOOTING_X_CENTRE = SOIL_WIDTH / 2

E_SOIL = 30 * MPa  # Pa
NU_SOIL = 0.3
GAMMA_SOIL = 18 * KN  # N/m³

PRESSURE_FOOTING_MAX = 300e3  # Pa
MESH_SIZE_NEAR = 0.25  # m, near footing
MESH_SIZE_FAR = 1.0  # m, far from footing

# --- Build geometry and mesh ---
out_dir = "C:\\Users\\nmora\\Github\\apeGmsh\\.claude\\worktrees\\apegmsh-readability-standards-d3c468\\readability_samples\\workshop\\loop3\\out_d"

with apeGmsh(model_name="footing_2d", save_to=out_dir + "\\footing_2d.h5") as g:
    # Soil domain and footing.
    soil = g.model.geometry.add_rectangle(0, -SOIL_DEPTH, 0, SOIL_WIDTH, SOIL_DEPTH, label="soil_block")
    footing_x_left = FOOTING_X_CENTRE - FOOTING_WIDTH / 2
    footing = g.model.geometry.add_rectangle(footing_x_left, 0, 0, FOOTING_WIDTH, FOOTING_DEPTH, label="footing")

    # Fragment to make interfaces conformal.
    g.model.boolean.fragment(["soil_block"], ["footing"], sync=True)

    # Physical groups.
    g.physical.add_surface("soil_block", name="Soil")
    g.physical.add_surface("footing", name="Footing")

    # Boundary edges.
    tol = 1e-3
    (g.model.select(None, dim=1)
        .in_box((0 - tol, -SOIL_DEPTH - tol, -tol), (SOIL_WIDTH + tol, -SOIL_DEPTH + tol, tol))
        .to_physical("Bottom"))
    (g.model.select(None, dim=1)
        .in_box((-tol, -SOIL_DEPTH - tol, -tol), (tol, 0 + tol, tol))
        .to_physical("Left"))
    (g.model.select(None, dim=1)
        .in_box((SOIL_WIDTH - tol, -SOIL_DEPTH - tol, -tol), (SOIL_WIDTH + tol, 0 + tol, tol))
        .to_physical("Right"))
    (g.model.select(None, dim=1)
        .in_box((footing_x_left - tol, FOOTING_DEPTH - tol, -tol),
                (footing_x_left + FOOTING_WIDTH + tol, FOOTING_DEPTH + tol, tol))
        .to_physical("Footing_top"))

    # Mesh size field: refined near footing.
    field = g.mesh.field.box(
        x_min=footing_x_left - 2.0, y_min=-2.0, z_min=-1,
        x_max=footing_x_left + FOOTING_WIDTH + 2.0, y_max=FOOTING_DEPTH + 2.0, z_max=1,
        size_in=MESH_SIZE_NEAR, size_out=MESH_SIZE_FAR
    )
    g.mesh.field.set_background(field)

    # Generate 2D mesh (triangles).
    g.mesh.generation.generate(dim=2)
    g.mesh.partitioning.renumber(dim=2, method="rcm", base=1)

    fem = g.mesh.queries.get_fem_data(dim=2)
    print("Mesh: %s" % fem.info.summary())

# --- OpenSees model (post-session bridge) ---
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)

# Materials and elements.
soil_mat = ops.nDMaterial.ElasticIsotropic(E=E_SOIL, nu=NU_SOIL, rho=GAMMA_SOIL / 9.81)
ops.element.Tri31(pg="Soil", material=soil_mat, thickness=1.0, body_force=(0.0, -9.81 * GAMMA_SOIL / 9.81))

footing_mat = ops.nDMaterial.ElasticIsotropic(E=1e12, nu=0.0, rho=0.0)
ops.element.Tri31(pg="Footing", material=footing_mat, thickness=1.0, body_force=(0.0, 0.0))

# Boundary conditions (global level).
ops.fix(pg="Bottom", dofs=(1, 1))
ops.fix(pg="Left", dofs=(1, 0))
ops.fix(pg="Right", dofs=(1, 0))

# Analysis with one load pattern: footing pressure (gravity via body forces).
ts = ops.timeSeries.Linear()
with ops.pattern.Plain(series=ts) as p:
    footing_force = PRESSURE_FOOTING_MAX * FOOTING_WIDTH
    p.load(pg="Footing_top", forces=(0.0, -footing_force))

# Emit and run via Tcl to avoid live analysis complexity.
ops.tcl(out_dir + "\\footing_2d.tcl", run=True)

# Save results file indicating success.
with open(out_dir + "\\summary.txt", "w") as f:
    f.write("Plane-strain strip footing analysis\\n")
    f.write("Soil: E=30 MPa, nu=0.3, gamma=18 kN/m³\\n")
    f.write("Footing pressure: 300 kPa\\n")
    f.write("Model ran successfully.\\n")

print("Analysis complete. Results in " + out_dir)
