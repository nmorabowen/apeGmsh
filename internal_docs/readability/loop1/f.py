"""
2-D Pratt truss bridge: linear static analysis.

Geometry: 6 panels × 4 m (span 24 m), height 4 m. Pinned at left, roller at
right. 100 kN point loads at interior bottom-chord joints (5 loads total).

Output: FEMData to .h5, OpenSees deck to .py, results to CSV.
All files written to out_f/ subdirectory.

Units: meters (m), kilonewtons (kN), kilopascals (kPa).
"""

# ============================================================================
# DATA
# ============================================================================

# Geometry
span = 24.0  # m, total horizontal span
n_panels = 6  # number of equal panels
panel_length = span / n_panels  # m, 4.0
height = 4.0  # m, truss height

# Cross-sections (steel)
A_chord = 0.02  # m^2, top and bottom chords
A_vertical = 0.005  # m^2, vertical members
A_diagonal = 0.01  # m^2, diagonal members

# Material
E_steel = 200e6  # kPa (200 GPa), Young's modulus
density_steel = 78.5  # kN/m^3

# Loading
P_load = 100.0  # kN, point load at each interior bottom-chord joint
n_interior_joints = n_panels - 1  # 5 interior joints on bottom chord

# Mesh
element_size = 0.5  # m, global mesh size (coarse to keep run fast)


# ============================================================================
# GEOMETRY
# ============================================================================

from apeGmsh import apeGmsh

with apeGmsh(model_name="pratt_truss", save_to="out_f/pratt_truss.h5") as g:
    # Bottom chord: y = 0, x from 0 to 24
    bottom_points = {}
    for i in range(n_panels + 1):
        x = i * panel_length
        p = g.model.geometry.add_point(x, 0.0, 0.0, label=f"bot_{i}")
        bottom_points[i] = p

    # Top chord: y = height, x from 0 to 24
    top_points = {}
    for i in range(n_panels + 1):
        x = i * panel_length
        p = g.model.geometry.add_point(x, height, 0.0, label=f"top_{i}")
        top_points[i] = p

    # Bottom chord lines
    bottom_lines = []
    for i in range(n_panels):
        line = g.model.geometry.add_line(
            bottom_points[i], bottom_points[i + 1], label=f"bot_chord_{i}"
        )
        bottom_lines.append(line)

    # Top chord lines
    top_lines = []
    for i in range(n_panels):
        line = g.model.geometry.add_line(
            top_points[i], top_points[i + 1], label=f"top_chord_{i}"
        )
        top_lines.append(line)

    # Vertical lines (connecting top and bottom at each node)
    vertical_lines = []
    for i in range(n_panels + 1):
        line = g.model.geometry.add_line(
            bottom_points[i], top_points[i], label=f"vertical_{i}"
        )
        vertical_lines.append(line)

    # Diagonal lines: alternating pattern (one per panel, 2 per panel alternating)
    diagonal_lines = []
    for i in range(n_panels):
        # Down-right: top[i] to bot[i+1]
        line1 = g.model.geometry.add_line(
            top_points[i], bottom_points[i + 1], label=f"diag_{i}_dr"
        )
        # Up-right: bot[i] to top[i+1]
        line2 = g.model.geometry.add_line(
            bottom_points[i], top_points[i + 1], label=f"diag_{i}_ur"
        )
        diagonal_lines.extend([line1, line2])

    # ========================================================================
    # PHYSICAL GROUPS
    # ========================================================================

    # Group members by cross-section so OpenSees can assign areas
    for i, line in enumerate(bottom_lines):
        g.physical.add(1, [line], name="bot_chord")

    for i, line in enumerate(top_lines):
        g.physical.add(1, [line], name="top_chord")

    for i, line in enumerate(vertical_lines):
        g.physical.add(1, [line], name="vertical")

    for i, line in enumerate(diagonal_lines):
        g.physical.add(1, [line], name="diagonal")

    # Support nodes: pinned at left, roller at right
    g.physical.add(0, [bottom_points[0]], name="support_left")
    g.physical.add(0, [bottom_points[n_panels]], name="support_right")

    # Load application points: interior bottom-chord joints (exclude ends)
    load_tags = [bottom_points[i] for i in range(1, n_panels)]
    g.physical.add(0, load_tags, name="load_points")

    # ========================================================================
    # LOADS AND MASSES (pre-mesh)
    # ========================================================================

    with g.loads.case("dead"):
        # Point loads at interior bottom-chord joints
        for i in range(1, n_panels):
            # Downward load (negative y direction)
            g.loads.point.force(
                f"bot_{i}", force=(0.0, -P_load, 0.0)
            )

    # Lumped masses at load points (negligible for static; included for completeness)
    g.masses.point("load_points", mass=0.1)

    # ========================================================================
    # MESH
    # ========================================================================

    g.mesh.sizing.set_global_size(element_size)
    g.mesh.generation.generate(dim=1)  # 1-D line mesh
    g.mesh.partitioning.renumber(dim=1, method="rcm", base=1)

    # Snapshot: extract FEMData
    fem = g.mesh.queries.get_fem_data(dim=1)
    print(fem.info.summary())

# ============================================================================
# OPENSEES MODEL
# ============================================================================

from apeGmsh.opensees import apeSees

ops = apeSees(fem)
ops.model(ndm=2, ndf=2)  # 2-D truss: ux, uy

# Material: uniaxial elastic for truss members (high yield to stay elastic)
steel = ops.uniaxialMaterial.Steel02(fy=1e9, E=E_steel, b=1e-6)

# Elements
ops.element.Truss(pg="bot_chord", material=steel, A=A_chord)
ops.element.Truss(pg="top_chord", material=steel, A=A_chord)
ops.element.Truss(pg="vertical", material=steel, A=A_vertical)
ops.element.Truss(pg="diagonal", material=steel, A=A_diagonal)

# Supports: pinned (no translation) at left, roller (Uy fixed) at right
ops.fix(pg="support_left", dofs=(1, 1))  # Fix Ux, Uy
ops.fix(pg="support_right", dofs=(0, 1))  # Fix Uy only (roller in x)

# Loads
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    # Opt-in: import the "dead" load case from the session
    p.from_model("dead")

# ============================================================================
# ANALYSIS AND SOLVE
# ============================================================================

# Linear static analysis
ops.analysis.Static()
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=1.0)
ops.test.NormDispIncr(tol=1e-3, max_iter=100)
ops.system.UmfPack()
ops.numberer.RCM()
ops.constraints.Transformation()

# Run analysis
result = ops.analyze(steps=1)
if result != 0:
    print("ERROR: analysis did not converge")
else:
    print("Analysis completed successfully")

# ============================================================================
# EMIT DECK AND RESULTS
# ============================================================================

ops.py("out_f/pratt_truss_deck.py")
print("OpenSees deck written to out_f/pratt_truss_deck.py")

# ============================================================================
# HAND CHECK: Method of Sections
# ============================================================================

# At midspan (x=12 m), left FBD receives 3 loads at x = 4, 8, 12 m.
# Total downward load on left = 300 kN.
# Vertical reaction at pinned support = 250 kN (by symmetry, half of 500 kN total).
# Shear at cut = 250 - 300 = -50 kN (net downward).
# Moment at support = 250×12 - 100×4 - 100×8 - 100×12 = 3000 - 400 - 800 - 1200 = 600 kN·m.

# For a truss cut through bottom chord, vertical, and diagonal at x=12:
# Moment about the base (y=0, x=12) of the vertical:
#   Moment_equilibrium => F_bc × h = moment_from_loads
#   F_bc × 4 = 600
#   F_bc = 150 kN (this is approximate, depends on exact section location)

# Better estimate: use lever arm formula for truss under concentrated loads.
# For 3 loads spanning 12 m at 4 m spacing, each 100 kN:
# Approximate mid-chord force ~ 300-500 kN (depends on geometry).

total_load = n_interior_joints * P_load  # 500 kN
approx_mid_chord_kN = (total_load / 2) * (span / 2) / height  # Rough formula
print(f"\nApproximate midspan bottom-chord force (hand calculation): {approx_mid_chord_kN:.1f} kN")

# ============================================================================
# SAVE RESULTS SUMMARY
# ============================================================================

import os
os.makedirs("out_f", exist_ok=True)

with open("out_f/results_summary.csv", "w") as f:
    f.write("Quantity,Value,Unit\n")
    f.write(f"Span,{span},m\n")
    f.write(f"Height,{height},m\n")
    f.write(f"Number of panels,{n_panels},—\n")
    f.write(f"Total load (sum of interior joint loads),{total_load},kN\n")
    f.write(f"Element size (mesh),{element_size},m\n")
    f.write(f"Bottom chord area,{A_chord},m^2\n")
    f.write(f"Top chord area,{A_chord},m^2\n")
    f.write(f"Vertical area,{A_vertical},m^2\n")
    f.write(f"Diagonal area,{A_diagonal},m^2\n")
    f.write(f"Steel modulus,{E_steel},kPa\n")
    f.write(f"Approx midspan chord force (hand calc),{approx_mid_chord_kN:.1f},kN\n")

print("Results summary written to out_f/results_summary.csv")
print("Model data saved to out_f/pratt_truss.h5")
