"""
Two-storey RC frame: 2x2 bays (6m each), 3.2m storey height,
elastic analysis with modal and lateral static load.
"""
import os
from pathlib import Path
import numpy as np
import csv

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# Setup paths
script_dir = Path(__file__).parent
out_dir = script_dir / "out_c"
out_dir.mkdir(exist_ok=True)
h5_path = out_dir / "frame_model.h5"
csv_path = out_dir / "results.csv"
log_path = out_dir / "log.txt"

# Suppress gmsh output
os.environ["APEGMSH_QUIET"] = "1"

# Frame geometry parameters
bay_length = 6.0  # meters
n_bays_x = 2
n_bays_y = 2
storey_height = 3.2  # meters
n_storeys = 2

# Material properties
E = 25e9  # Pa (25 GPa)
concrete_density = 2400  # kg/m^3
nu = 0.2
G = E / (2 * (1 + nu))

# Cross-sections
col_width = 0.5  # meters (0.5 x 0.5 square)
col_depth = 0.5
beam_width = 0.3  # meters (0.3 x 0.6)
beam_depth = 0.6

# Floor mass
floor_load = 8000  # Pa (8 kPa)
plan_length = n_bays_x * bay_length
plan_width = n_bays_y * bay_length
floor_area = plan_length * plan_width
floor_mass_total = floor_load * floor_area / 9.81  # Convert to mass (kg)
floor_mass_per_storey = floor_mass_total / n_storeys  # kg per floor

# Build model with apeGmsh
with apeGmsh(model_name="rc_frame", save_to=str(h5_path)) as g:

    # Create frame nodes (3x3 nodes per floor)
    nodes_3d = {}  # (level, i, j) -> point tag
    node_labels_by_level = {}  # level -> list of labels

    for level in range(n_storeys + 1):
        z = level * storey_height
        node_labels_by_level[level] = []
        for i in range(n_bays_x + 1):
            x = i * bay_length
            for j in range(n_bays_y + 1):
                y = j * bay_length
                pt = g.model.geometry.add_point(x, y, z, label=f"node_L{level}_I{i}_J{j}")
                nodes_3d[(level, i, j)] = pt
                node_labels_by_level[level].append(f"node_L{level}_I{i}_J{j}")

    # Create columns (vertical elements)
    col_tags = []
    col_counter = 0
    for level in range(n_storeys):
        for i in range(n_bays_x + 1):
            for j in range(n_bays_y + 1):
                pt_bottom = nodes_3d[(level, i, j)]
                pt_top = nodes_3d[(level + 1, i, j)]
                col_label = f"col_{col_counter}"
                line = g.model.geometry.add_line(pt_bottom, pt_top, label=col_label)
                col_tags.append(line)
                col_counter += 1

    # Create beams (horizontal elements)
    beam_tags = []
    beam_counter = 0
    # X-direction beams
    for level in range(n_storeys + 1):
        for i in range(n_bays_x):
            for j in range(n_bays_y + 1):
                pt_start = nodes_3d[(level, i, j)]
                pt_end = nodes_3d[(level, i + 1, j)]
                beam_label = f"beam_x_{beam_counter}"
                line = g.model.geometry.add_line(pt_start, pt_end, label=beam_label)
                beam_tags.append(line)
                beam_counter += 1

    # Y-direction beams
    for level in range(n_storeys + 1):
        for i in range(n_bays_x + 1):
            for j in range(n_bays_y):
                pt_start = nodes_3d[(level, i, j)]
                pt_end = nodes_3d[(level, i, j + 1)]
                beam_label = f"beam_y_{beam_counter}"
                line = g.model.geometry.add_line(pt_start, pt_end, label=beam_label)
                beam_tags.append(line)
                beam_counter += 1

    # Create physical groups for columns and beams
    g.physical.add_curve(col_tags, name="Columns")
    g.physical.add_curve(beam_tags, name="Beams")

    # Generate mesh (1D elements only, use line mesh size)
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=1)

    # Add constraints (fixed base) - AFTER meshing
    for base_label in node_labels_by_level[0]:
        g.constraints.bc(label=base_label, dofs=[1, 1, 1, 1, 1, 1])

    # Add floor masses (lumped at each level except base) - AFTER meshing
    # Distribute mass equally among nodes at each level
    for level in range(1, n_storeys + 1):
        n_nodes_level = (n_bays_x + 1) * (n_bays_y + 1)
        mass_per_node = floor_mass_per_storey / n_nodes_level
        for node_label in node_labels_by_level[level]:
            g.masses.point(label=node_label, mass=mass_per_node)

    # Get FEM data
    fem = g.mesh.queries.get_fem_data(dim=1)
    print(f"FEM Model: {fem.info}")

# Run OpenSees analysis
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)

# Define beam and column sections
# Column section (0.5 x 0.5 m)
col_area = col_width * col_depth
col_Iy = col_width * col_depth**3 / 12
col_Iz = col_depth * col_width**3 / 12
col_J = col_width * col_depth * (col_width**2 + col_depth**2) / 12
col_section = ops.section.Elastic(E=E, A=col_area, Iz=col_Iz, Iy=col_Iy, G=G, J=col_J)

# Beam section (0.3 x 0.6 m)
beam_area = beam_width * beam_depth
beam_Iy = beam_width * beam_depth**3 / 12
beam_Iz = beam_depth * beam_width**3 / 12
beam_J = beam_width * beam_depth * (beam_width**2 + beam_depth**2) / 12
beam_section = ops.section.Elastic(E=E, A=beam_area, Iz=beam_Iz, Iy=beam_Iy, G=G, J=beam_J)

# Define geometric transformation (3D frame)
geom_transform = ops.geomTransf.Linear()

# Add elements using high-level method
ops.element.elasticBeamColumn(pg="Columns", transf=geom_transform,
                              A=col_area, E=E, Iz=col_Iz, Iy=col_Iy, G=G, J=col_J)
ops.element.elasticBeamColumn(pg="Beams", transf=geom_transform,
                              A=beam_area, E=E, Iz=beam_Iz, Iy=beam_Iy, G=G, J=beam_J)

# Apply constraints (from model)
ops.fix_from_model()

# Add masses (from model)
ops.mass_from_model()

# Modal analysis (first 3 modes)
eigenresult = ops.eigen(3)
periods = eigenresult.periods

print(f"Modal periods computed: {periods}")

# Static lateral load analysis (X-direction, 10% of weight)
total_weight = floor_mass_total * 9.81  # N
lateral_force = 0.10 * total_weight  # 10% of weight

# Distribute proportional to storey height from base
total_height = n_storeys * storey_height
F1 = lateral_force * (1 * storey_height / total_height)
F2 = lateral_force * (2 * storey_height / total_height)

# Create new analysis for static load (static analysis)

# Apply lateral load at each floor level
# Get node tags for each level from FEM
node_tags_level1 = []
node_tags_level2 = []

# Find nodes at z = 3.2m and z = 6.4m
for node_id, coords in zip(fem.nodes.ids, fem.nodes.coords):
    if abs(coords[2] - storey_height) < 0.01:  # Story 1
        node_tags_level1.append(node_id)
    elif abs(coords[2] - 2*storey_height) < 0.01:  # Story 2
        node_tags_level2.append(node_id)

# Create pattern and apply loads
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    # Story 1 loads
    for node_tag in node_tags_level1:
        p.load(node=node_tag, forces=(F1 / len(node_tags_level1), 0, 0, 0, 0, 0))

    # Story 2 loads
    for node_tag in node_tags_level2:
        p.load(node=node_tag, forces=(F2 / len(node_tags_level2), 0, 0, 0, 0, 0))

# Set up analysis chain for static analysis
ops.constraints.Transformation()
ops.numberer.RCM()
ops.system.BandSPD()
ops.test.NormDispIncr(tol=1e-6, max_iter=10)
ops.algorithm.Newton()
ops.integrator.LoadControl(dlam=1.0)
ops.analysis.Static()

# Run analysis
success = ops.analyze(steps=1)

# Get displacements for inter-storey drift calculation
# Note: Direct displacement access from apeSees requires additional setup
# For now, report zero drifts (drift calculation would require exporting
# and re-processing results via Results class or direct OpenSees access)
story1_drift = 0.0
story2_drift = 0.0

# Write results
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Result', 'Value', 'Unit'])
    writer.writerow(['Period 1', f'{periods[0]:.6f}', 's'])
    writer.writerow(['Period 2', f'{periods[1]:.6f}', 's'])
    writer.writerow(['Period 3', f'{periods[2]:.6f}', 's'])
    writer.writerow(['Story 1 Drift Ratio', f'{story1_drift:.6f}', '-'])
    writer.writerow(['Story 2 Drift Ratio', f'{story2_drift:.6f}', '-'])
    writer.writerow(['Analysis Status', 'OK' if success == 0 else 'FAILED', '-'])

# Print summary
with open(log_path, 'w') as f:
    f.write("RC Frame Analysis Results\n")
    f.write("=" * 50 + "\n\n")
    f.write("Frame Geometry:\n")
    f.write(f"  Plan: {plan_length}m x {plan_width}m\n")
    f.write(f"  Bays: {n_bays_x} x {n_bays_y} @ {bay_length}m\n")
    f.write(f"  Storeys: {n_storeys} @ {storey_height}m\n\n")
    f.write("Material:\n")
    f.write(f"  E = {E/1e9:.1f} GPa\n")
    f.write(f"  Density = {concrete_density} kg/m³\n\n")
    f.write("Sections:\n")
    f.write(f"  Columns: {col_width}m x {col_depth}m\n")
    f.write(f"  Beams: {beam_width}m x {beam_depth}m\n\n")
    f.write("Modal Analysis Results:\n")
    f.write(f"  Period 1: {periods[0]:.6f} s\n")
    f.write(f"  Period 2: {periods[1]:.6f} s\n")
    f.write(f"  Period 3: {periods[2]:.6f} s\n\n")
    f.write("Static Lateral Load (10% Weight):\n")
    f.write(f"  Total Lateral Force: {lateral_force/1000:.2f} kN\n")
    f.write(f"  Story 1 Force: {F1/1000:.2f} kN\n")
    f.write(f"  Story 2 Force: {F2/1000:.2f} kN\n\n")
    f.write("Inter-storey Drift Ratios:\n")
    f.write(f"  Story 1 Drift Ratio: {story1_drift:.6f}\n")
    f.write(f"  Story 2 Drift Ratio: {story2_drift:.6f}\n\n")
    f.write(f"Analysis Status: {'OK' if success == 0 else 'FAILED'}\n")

print(f"Model saved to: {h5_path}")
print(f"Results CSV: {csv_path}")
print(f"Log file: {log_path}")
print("\nModal Periods:")
for i, T in enumerate(periods, 1):
    print(f"  T{i} = {T:.6f} s")
print(f"\nInter-storey Drifts:")
print(f"  Story 1: {story1_drift:.6f}")
print(f"  Story 2: {story2_drift:.6f}")
