"""
Two-storey RC frame: 2x2 bays, fixed base, modal + lateral static analysis.
Units: m (length), N (force), Pa (stress), s (time).
"""

import os
import numpy as np
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- Data

# Geometry: dimensions (m)
BAY_LENGTH = 6.0  # m, one span
N_BAYS_X = 2  # bays in X
N_BAYS_Y = 2  # bays in Y
STOREY_HEIGHT = 3.2  # m
N_STOREYS = 2  # storeys

# Column cross-section: square 0.5 x 0.5 m
COL_DEPTH = 0.5  # m, in X or Y
COL_WIDTH = 0.5  # m, perpendicular to depth
COL_AREA = COL_DEPTH * COL_WIDTH  # m²

# Beam cross-section: 0.3 x 0.6 m (width x depth in bending plane)
BEAM_WIDTH = 0.3  # m
BEAM_DEPTH = 0.6  # m
BEAM_AREA = BEAM_WIDTH * BEAM_DEPTH  # m²

# Material: elastic concrete
E_CONCRETE = 25e9  # Pa, elastic modulus
NU_CONCRETE = 0.2  # Poisson's ratio
RHO_CONCRETE = 2400  # kg/m³, density

# Gravity & mass
G = 9.81  # m/s²
FLOOR_LOAD = 8e3  # Pa, 8 kPa distributed over floor area
FLOOR_AREA_PER_LEVEL = (N_BAYS_X * BAY_LENGTH) * (N_BAYS_Y * BAY_LENGTH)  # m²
FLOOR_MASS_TOTAL = FLOOR_LOAD / G * FLOOR_AREA_PER_LEVEL  # kg per storey

# Lateral load: 10% of total weight, distributed by storey height
SEISMIC_FRACTION = 0.10  # 10% of weight
TOTAL_WEIGHT = FLOOR_MASS_TOTAL * N_STOREYS * G  # N
SEISMIC_FORCE_TOTAL = SEISMIC_FRACTION * TOTAL_WEIGHT  # N

# Analysis tolerances
TOL_CONVERGENCE = 1e-6  # drift convergence for static analysis
MAX_ITERATIONS = 100  # iterations for static analysis

# Output directory
OUT_DIR = "out_a"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Geometry & Model Setup

with apeGmsh(model_name="frame_2x2", save_to=os.path.join(OUT_DIR, "frame.h5")) as g:

    # Build frame by looping over bays and storeys, creating nodes and members
    # Nodes are labeled by position: N_x_y_z (X bay, Y bay, Z storey)
    # Elements are labeled by role: Col_... for columns, Beam_... for beams

    # Create nodes for all joints
    nodes = {}  # dictionary: (bay_x, bay_y, storey) -> node_label

    for i_x in range(N_BAYS_X + 1):
        x_coord = i_x * BAY_LENGTH
        for i_y in range(N_BAYS_Y + 1):
            y_coord = i_y * BAY_LENGTH
            for i_z in range(N_STOREYS + 1):
                z_coord = i_z * STOREY_HEIGHT
                node_label = f"N_{i_x}_{i_y}_{i_z}"
                nodes[(i_x, i_y, i_z)] = node_label
                g.model.geometry.add_point(x_coord, y_coord, z_coord, label=node_label)

    # Create column members (vertical elements)
    # Columns connect (i_x, i_y, i_z) to (i_x, i_y, i_z+1)
    for i_x in range(N_BAYS_X + 1):
        for i_y in range(N_BAYS_Y + 1):
            for i_z in range(N_STOREYS):
                node_bot = nodes[(i_x, i_y, i_z)]
                node_top = nodes[(i_x, i_y, i_z + 1)]
                col_label = f"Col_{i_x}_{i_y}_{i_z}"
                g.model.geometry.add_line(node_bot, node_top, label=col_label)

    # Create beam members in X direction (horizontal at each level)
    # Beams connect (i_x, i_y, i_z) to (i_x+1, i_y, i_z)
    for i_z in range(N_STOREYS + 1):
        for i_x in range(N_BAYS_X):
            for i_y in range(N_BAYS_Y + 1):
                node_i = nodes[(i_x, i_y, i_z)]
                node_j = nodes[(i_x + 1, i_y, i_z)]
                beam_label = f"BeamX_{i_x}_{i_y}_{i_z}"
                g.model.geometry.add_line(node_i, node_j, label=beam_label)

    # Create beam members in Y direction (horizontal at each level)
    # Beams connect (i_x, i_y, i_z) to (i_x, i_y+1, i_z)
    for i_z in range(N_STOREYS + 1):
        for i_x in range(N_BAYS_X + 1):
            for i_y in range(N_BAYS_Y):
                node_i = nodes[(i_x, i_y, i_z)]
                node_j = nodes[(i_x, i_y + 1, i_z)]
                beam_label = f"BeamY_{i_x}_{i_y}_{i_z}"
                g.model.geometry.add_line(node_i, node_j, label=beam_label)

    # Physical groups for OpenSees assembly
    # All columns as one group
    col_labels = [f"Col_{i_x}_{i_y}_{i_z}"
                  for i_x in range(N_BAYS_X + 1)
                  for i_y in range(N_BAYS_Y + 1)
                  for i_z in range(N_STOREYS)]
    g.physical.add_curve(col_labels, name="Columns")

    # All beams in X direction
    beamx_labels = [f"BeamX_{i_x}_{i_y}_{i_z}"
                    for i_z in range(N_STOREYS + 1)
                    for i_x in range(N_BAYS_X)
                    for i_y in range(N_BAYS_Y + 1)]
    g.physical.add_curve(beamx_labels, name="BeamsX")

    # All beams in Y direction
    beamy_labels = [f"BeamY_{i_x}_{i_y}_{i_z}"
                    for i_z in range(N_STOREYS + 1)
                    for i_x in range(N_BAYS_X + 1)
                    for i_y in range(N_BAYS_Y)]
    g.physical.add_curve(beamy_labels, name="BeamsY")

    # Base nodes (lowest level) for fixities
    base_labels = [nodes[(i_x, i_y, 0)]
                   for i_x in range(N_BAYS_X + 1)
                   for i_y in range(N_BAYS_Y + 1)]
    g.physical.add_point(base_labels, name="Base")

        # Diaphragm constraints: create physical groups for each floor diaphragm
    # All nodes at a floor are constrained to move together (rigid diaphragm)
    for i_z in range(1, N_STOREYS + 1):
        # Master node is the first node at that level
        master_node = nodes[(0, 0, i_z)]
        # All other nodes at that level are slaves
        slave_nodes = [nodes[(i_x, i_y, i_z)]
                      for i_x in range(N_BAYS_X + 1)
                      for i_y in range(N_BAYS_Y + 1)
                      if (i_x, i_y) != (0, 0)]

        diaphragm_master_label = f"DiaphragmMaster_{i_z}"
        diaphragm_slave_label = f"DiaphragmSlaves_{i_z}"
        g.physical.add_point([master_node], name=diaphragm_master_label)
        g.physical.add_point(slave_nodes, name=diaphragm_slave_label)

    # --- Groups & Loads

    # Masses lumped at all nodes of each floor level (inertia at diaphragm)
    mass_per_node = FLOOR_MASS_TOTAL / ((N_BAYS_X + 1) * (N_BAYS_Y + 1))
    for i_z in range(1, N_STOREYS + 1):
        # Lumped mass at both master and slaves
        master_label = f"DiaphragmMaster_{i_z}"
        slave_label = f"DiaphragmSlaves_{i_z}"
        g.masses.point(master_label, mass=mass_per_node)
        g.masses.point(slave_label, mass=mass_per_node)

    # Rigid diaphragm: equal DOF constraints on master and slave nodes
    # All nodes at each floor move together in X-Y directions (horizontal rigidity)
    for i_z in range(1, N_STOREYS + 1):
        master_label = f"DiaphragmMaster_{i_z}"
        slave_label = f"DiaphragmSlaves_{i_z}"
        g.constraints.equal_dof(master_label, slave_label, dofs=(1, 2))

    # --- Mesh

    # Global mesh size: one element per member (for frame analysis)
    # Mesh size = length of smallest member (all members are 6 m or less)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=1)

    # Get solver snapshot
    fem = g.mesh.queries.get_fem_data(dim=1)
    print("\n=== FEM Data ===")
    print(fem.info.summary())

# --- OpenSees Model

ops = apeSees(fem)
# 2D frame in XY plane: lateral motion in X, vertical in Y, rotations about Z
ops.model(ndm=2, ndf=3)

# Material
conc = ops.nDMaterial.ElasticIsotropic(E=E_CONCRETE, nu=NU_CONCRETE, rho=RHO_CONCRETE)

# Section properties for columns (square: 0.5 x 0.5 m)
# About principal axes: both equal for square section
col_ixx = (COL_WIDTH * COL_DEPTH**3) / 12  # m^4, second moment of inertia
col_iyy = (COL_DEPTH * COL_WIDTH**3) / 12  # m^4, same as col_ixx for square
col_g = E_CONCRETE / (2 * (1 + NU_CONCRETE))  # Pa, shear modulus
col_j = col_ixx + col_iyy  # m^4, polar moment (approx for square)

# Section properties for beams (rectangular: 0.3 x 0.6 m)
# Width 0.3 m perpendicular to bending, depth 0.6 m in bending plane
beam_iz = (BEAM_WIDTH * BEAM_DEPTH**3) / 12  # m^4, moment about Z axis
beam_iy = (BEAM_DEPTH * BEAM_WIDTH**3) / 12  # m^4, moment about Y axis
beam_g = E_CONCRETE / (2 * (1 + NU_CONCRETE))  # Pa, shear modulus
beam_j = beam_iz + beam_iy  # m^4, polar moment (approx)

# Geometric transformation (2D frame, no P-Delta)
ops.geomTransf.Linear(name="LinearTransf")

# Create beam-column elements for columns (vertical members)
# For 2D: use Iz only (bending about Z axis)
ops.element.elasticBeamColumn(
    pg="Columns",
    transf="LinearTransf",
    A=COL_AREA,
    E=E_CONCRETE,
    Iz=col_ixx,
)

# Create beam elements in X direction
ops.element.elasticBeamColumn(
    pg="BeamsX",
    transf="LinearTransf",
    A=BEAM_AREA,
    E=E_CONCRETE,
    Iz=beam_iz,
)

# Create beam elements in Y direction
ops.element.elasticBeamColumn(
    pg="BeamsY",
    transf="LinearTransf",
    A=BEAM_AREA,
    E=E_CONCRETE,
    Iz=beam_iz,
)

# Import lumped masses from the geometry definition
ops.mass_from_model()

# Fixities at base: all translations and rotation
ops.fix(pg="Base", dofs=(1, 1, 1))

# --- Analysis: Modal

print("\n=== Modal Analysis ===")

# Eigenvalue analysis: find first 3 natural periods
n_modes = 3
result = ops.eigen(num_modes=n_modes)
assert result is not None, "Eigenvalue analysis failed"

# Retrieve and report natural periods from eigenvalues
# result.frequencies contains the natural frequencies in rad/s
periods = []
for i_mode in range(n_modes):
    omega = result.frequencies[i_mode]  # rad/s
    period = 2 * np.pi / omega if omega > 0 else np.inf
    periods.append(period)
    print(f"Mode {i_mode + 1}: Period = {period:.4f} s")

# --- Analysis: Lateral Static

print("\n=== Lateral Static Analysis ===")

# Distribute lateral force proportionally to storey height
# Force at each storey = total_force * (height of that storey / total height)
total_height = N_STOREYS * STOREY_HEIGHT
forces_by_storey = {}
for i_z in range(1, N_STOREYS + 1):
    # Height contribution: cumulative height up to this level
    height_at_storey = i_z * STOREY_HEIGHT
    force_storey = SEISMIC_FORCE_TOTAL * (height_at_storey / total_height)
    forces_by_storey[i_z] = force_storey

print(f"Total seismic force (10% of weight): {SEISMIC_FORCE_TOTAL / 1e3:.2f} kN")
for i_z in range(1, N_STOREYS + 1):
    print(f"Force at storey {i_z}: {forces_by_storey[i_z] / 1e3:.2f} kN")

# Apply lateral loads at each storey (distributed equally to all diaphragm nodes)
# Each node gets a portion of the storey force
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    for i_z in range(1, N_STOREYS + 1):
        master_label = f"DiaphragmMaster_{i_z}"
        slave_label = f"DiaphragmSlaves_{i_z}"
        n_nodes_diaphragm = (N_BAYS_X + 1) * (N_BAYS_Y + 1)
        force_per_node = forces_by_storey[i_z] / n_nodes_diaphragm
        # Apply horizontal force in X direction only
        p.load(pg=master_label, forces=(force_per_node, 0.0, 0.0))
        p.load(pg=slave_label, forces=(force_per_node, 0.0, 0.0))

# Static analysis with load control
ops.integrator.LoadControl(lambda_increment=1.0)
ops.analysis.Static()
status_static = ops.analyze(steps=1)
assert status_static == 0, f"Static analysis failed: {status_static}"

# Retrieve displacements and compute inter-storey drift ratios
print("\n=== Results: Inter-storey Drift Ratios ===")

# Displacement at each storey (X direction at diaphragm center)
# Use first node of each diaphragm for displacement read
u_by_storey = {}
for i_z in range(1, N_STOREYS + 1):
    node_label = nodes[(0, 0, i_z)]
    # raw-ops: need direct node displacement read (apeSees has no native method)
    ops_raw = ops.raw
    node_tag = fem.nodes.select(label=node_label).tags[0]
    u_x = ops_raw.nodeDisp(node_tag, 1)  # DOF 1 is X direction
    u_by_storey[i_z] = u_x

print(f"Storey displacements (X direction):")
for i_z in range(1, N_STOREYS + 1):
    print(f"  Level {i_z}: u = {u_by_storey[i_z] * 1e3:.4f} mm")

# Inter-storey drift ratio: (u_i - u_i-1) / storey_height
print(f"\nInter-storey drift ratios:")
u_previous = 0.0
for i_z in range(1, N_STOREYS + 1):
    u_current = u_by_storey[i_z]
    drift = (u_current - u_previous) / STOREY_HEIGHT
    drift_ratio_pct = drift * 100
    print(f"  Level {i_z}: drift ratio = {drift_ratio_pct:.4f}%")
    u_previous = u_current

# --- Report & Checks

print("\n=== Summary Report ===")
print(f"Frame: {N_BAYS_X} x {N_BAYS_Y} bays @ {BAY_LENGTH} m, {N_STOREYS} storeys @ {STOREY_HEIGHT} m")
print(f"Column section: {COL_DEPTH} x {COL_WIDTH} m, Area = {COL_AREA:.4f} m²")
print(f"Beam section: {BEAM_WIDTH} x {BEAM_DEPTH} m, Area = {BEAM_AREA:.4f} m²")
print(f"Material: E = {E_CONCRETE / 1e9:.1f} GPa, ρ = {RHO_CONCRETE} kg/m³")
print(f"Total weight: {TOTAL_WEIGHT / 1e3:.2f} kN")
print(f"Floor mass per storey: {FLOOR_MASS_TOTAL / 1e3:.2f} Mg")

print(f"\nNatural periods (first 3 modes):")
for i, period in enumerate(periods, 1):
    print(f"  T{i} = {period:.4f} s")

print(f"\nLateral load case: {SEISMIC_FRACTION * 100:.0f}% of weight, distributed by height")

# Sanity checks
assert len(periods) == 3, "Expected 3 periods"
assert all(p > 0 for p in periods), "All periods must be positive"
assert periods == sorted(periods), "Periods should be in increasing order"

print("\n✓ Analysis completed successfully")
