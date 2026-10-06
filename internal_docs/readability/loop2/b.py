"""Two-storey RC frame, 2 x 2 bays: modal analysis + equivalent lateral static load.

Elastic beam-column members on fixed bases, a rigid floor diaphragm at each
level, and the floor mass (8 kPa over the floor area) lumped at the diaphragm
master node. Reports the first three periods and the inter-storey drift ratios
under a lateral load in X of 10 % of the seismic weight, distributed in
proportion to storey height.

Units: SI throughout (N, m, kg, s). Z is up.
"""

# --- Imports -----------------------------------------------------------------
import csv
from pathlib import Path

import openseespy.opensees as ops_raw

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# --- Data --------------------------------------------------------------------
OUT = Path(__file__).parent / "out_b"
OUT.mkdir(exist_ok=True)

KN = 1e3            # N per kN
MM = 1e-3           # m per mm
G_ACC = 9.81        # m/s2, gravity

# Frame grid
BAY = 6.0           # m, bay length in both plan directions
N_BAYS = 2          # bays per direction
STOREY = 3.2        # m, storey height
N_STOREYS = 2
LX = N_BAYS * BAY   # m, plan dimension in X
LY = N_BAYS * BAY   # m, plan dimension in Y

# Concrete
E = 25e9            # Pa, modulus of elasticity
NU = 0.2            # Poisson's ratio
G = E / (2 * (1 + NU))   # Pa, shear modulus

# Column 0.5 x 0.5 m (square: both bending axes equal)
B_COL = 0.5         # m, column width
H_COL = 0.5         # m, column depth
A_COL = B_COL * H_COL                   # m2
I_COL = B_COL * H_COL**3 / 12           # m4, about either axis
J_COL = 0.1406 * B_COL**4               # m4, torsion constant of a square (Roark)

# Beam 0.3 m wide x 0.6 m deep
B_BEAM = 0.3        # m, beam width
H_BEAM = 0.6        # m, beam depth
A_BEAM = B_BEAM * H_BEAM                # m2
I_BEAM_STRONG = B_BEAM * H_BEAM**3 / 12  # m4, bending in the vertical plane
I_BEAM_WEAK = H_BEAM * B_BEAM**3 / 12    # m4, bending in plan
J_BEAM = H_BEAM * B_BEAM**3 * (1 / 3 - 0.21 * (B_BEAM / H_BEAM) * (1 - B_BEAM**4 / (12 * H_BEAM**4)))  # m4, Roark rectangle

# Floor mass: 8 kPa over the floor area, lumped at the diaphragm master
FLOOR_PRESSURE = 8e3                    # Pa, seismic floor load
FLOOR_AREA = LX * LY                    # m2
W_FLOOR = FLOOR_PRESSURE * FLOOR_AREA   # N, seismic weight per floor
M_FLOOR = W_FLOOR / G_ACC               # kg, translational mass per floor
I_FLOOR = M_FLOOR * (LX**2 + LY**2) / 12  # kg m2, rotational inertia of a uniform rectangular floor about its centre

# Equivalent lateral static load in X
W_TOTAL = N_STOREYS * W_FLOOR           # N, seismic weight of the building
BASE_SHEAR_RATIO = 0.10                 # fraction of W_TOTAL applied as base shear
V_BASE = BASE_SHEAR_RATIO * W_TOTAL     # N
h_L1 = 1 * STOREY                       # m, height of level 1 above the base
h_L2 = 2 * STOREY                       # m, height of level 2 above the base
F_L1 = V_BASE * h_L1 / (h_L1 + h_L2)    # N, storey force at level 1 (equal floor weights)
F_L2 = V_BASE * h_L2 / (h_L1 + h_L2)    # N, storey force at level 2

# Mesh and analysis settings
MEMBER_MESH_SIZE = BAY                  # m, >= the longest member so each member is one elastic element
DIAPHRAGM_PLANE_TOL = 0.01              # m, slab nodes within this distance of the floor plane join the diaphragm
N_MODES = 3
TOL_SOLVER = 1e-8                       # m, displacement-increment convergence tolerance (linear problem)
TOL_CHECK = 1e-6                        # relative tolerance for the base-shear equilibrium check

# --- Geometry: a 3 x 3 grid of joints on three levels -------------------------
with apeGmsh(model_name="two_storey_frame") as g:

    def joint(ix, iy, level):
        return f"J{ix}{iy}_L{level}"

    for level in range(N_STOREYS + 1):
        for ix in range(N_BAYS + 1):
            for iy in range(N_BAYS + 1):
                g.model.geometry.add_point(ix * BAY, iy * BAY, level * STOREY, label=joint(ix, iy, level))

    columns = []
    for level in range(N_STOREYS):
        for ix in range(N_BAYS + 1):
            for iy in range(N_BAYS + 1):
                label = f"C{ix}{iy}_L{level + 1}"
                g.model.geometry.add_line(joint(ix, iy, level), joint(ix, iy, level + 1), label=label)
                columns.append(label)

    beams_x = []
    beams_y = []
    for level in range(1, N_STOREYS + 1):
        for iy in range(N_BAYS + 1):
            for ix in range(N_BAYS):
                label = f"BX{ix}{iy}_L{level}"
                g.model.geometry.add_line(joint(ix, iy, level), joint(ix + 1, iy, level), label=label)
                beams_x.append(label)
        for ix in range(N_BAYS + 1):
            for iy in range(N_BAYS):
                label = f"BY{ix}{iy}_L{level}"
                g.model.geometry.add_line(joint(ix, iy, level), joint(ix, iy + 1, level), label=label)
                beams_y.append(label)

    # --- Groups and constraints ------------------------------------------------
    g.physical.add_curve(columns, name="columns")
    g.physical.add_curve(beams_x, name="beams_x")
    g.physical.add_curve(beams_y, name="beams_y")

    base_joints = [joint(ix, iy, 0) for ix in range(N_BAYS + 1) for iy in range(N_BAYS + 1)]
    g.physical.add_point(base_joints, name="fixed_base")

    # The central joint of each floor sits at the floor centroid (the centre of
    # mass of a uniform floor), so it is the diaphragm master and the mass node.
    ix_centre = N_BAYS // 2
    iy_centre = N_BAYS // 2
    for level in range(1, N_STOREYS + 1):
        master = joint(ix_centre, iy_centre, level)
        slaves = [
            joint(ix, iy, level)
            for ix in range(N_BAYS + 1)
            for iy in range(N_BAYS + 1)
            if joint(ix, iy, level) != master
        ]
        g.physical.add_point([master], name=f"floor_master_L{level}")
        g.physical.add_point(slaves, name=f"floor_joints_L{level}")
        g.constraints.rigid_diaphragm(
            f"floor_master_L{level}",
            f"floor_joints_L{level}",
            master_point=(ix_centre * BAY, iy_centre * BAY, level * STOREY),
            plane_normal=(0, 0, 1),
            plane_tolerance=DIAPHRAGM_PLANE_TOL,
            name=f"diaphragm_L{level}",
        )

    # --- Mesh: one elastic element per member ---------------------------------
    g.mesh.sizing.set_size_all_points(MEMBER_MESH_SIZE)   # the size is read at the joints, not from the global cap
    g.mesh.generation.generate(dim=1)
    g.mesh.partitioning.renumber(dim=1, base=1)   # dense 1-based node tags for the OpenSees deck
    fem = g.mesh.queries.get_fem_data(dim=None)   # all dims: the joint groups are points
    print(fem.info.summary())

# --- OpenSees model ----------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)

# Local z horizontal for beams, so Iz is the strong axis (bending in the vertical plane)
transf_columns = ops.geomTransf.Linear(vecxz=(1, 0, 0))
transf_beams_x = ops.geomTransf.Linear(vecxz=(0, 1, 0))
transf_beams_y = ops.geomTransf.Linear(vecxz=(-1, 0, 0))

ops.element.elasticBeamColumn(pg="columns", transf=transf_columns,
                              A=A_COL, E=E, G=G, J=J_COL, Iy=I_COL, Iz=I_COL)
ops.element.elasticBeamColumn(pg="beams_x", transf=transf_beams_x,
                              A=A_BEAM, E=E, G=G, J=J_BEAM, Iy=I_BEAM_WEAK, Iz=I_BEAM_STRONG)
ops.element.elasticBeamColumn(pg="beams_y", transf=transf_beams_y,
                              A=A_BEAM, E=E, G=G, J=J_BEAM, Iy=I_BEAM_WEAK, Iz=I_BEAM_STRONG)

ops.fix(pg="fixed_base", dofs=(1, 1, 1, 1, 1, 1))

# Floor mass at the diaphragm master: in-plane translations plus rotation about
# Z; the vertical and rocking terms stay zero because the diaphragm carries only
# the in-plane motion and the lateral modes are the ones asked for.
for level in range(1, N_STOREYS + 1):
    ops.mass(pg=f"floor_master_L{level}", values=(M_FLOOR, M_FLOOR, 0.0, 0.0, 0.0, I_FLOOR))

# --- Analysis settings (shared by both runs) ---------------------------------
ops.constraints.Transformation()       # rigidDiaphragm is a multi-point constraint
ops.numberer.RCM()
ops.system.BandGeneral()
ops.test.NormDispIncr(tol=TOL_SOLVER, max_iter=10)
ops.algorithm.Linear()                 # elastic members: one factorisation solves the step
ops.integrator.LoadControl(dlam=1.0)   # the full load in one step
ops.analysis.Static()

ops.h5(OUT / "two_storey_frame.h5")    # model archive for the viewer

# --- Analysis 1: modal -------------------------------------------------------
modal = ops.eigen(N_MODES)
periods = modal.periods

# --- Analysis 2: equivalent lateral static load in X -------------------------
with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.load(pg="floor_master_L1", forces=(F_L1, 0.0, 0.0, 0.0, 0.0, 0.0))
    p.load(pg="floor_master_L2", forces=(F_L2, 0.0, 0.0, 0.0, 0.0, 0.0))

status = ops.analyze(steps=1)
assert status == 0, f"static analysis failed with status {status}"

# --- Checks and report -------------------------------------------------------
node_L1 = fem.nodes.select(pg="floor_master_L1").ids[0]
node_L2 = fem.nodes.select(pg="floor_master_L2").ids[0]
u_L1 = ops_raw.nodeDisp(node_L1, 1)   # raw-ops: displacement read-back from the live domain
u_L2 = ops_raw.nodeDisp(node_L2, 1)   # raw-ops: displacement read-back from the live domain

drift_L1 = u_L1 / STOREY
drift_L2 = (u_L2 - u_L1) / STOREY

# Equilibrium: the base reactions in X balance the applied base shear
ops_raw.reactions()                                   # raw-ops: reaction read-back from the live domain
R_base_x = sum(ops_raw.nodeReaction(node, 1) for node in fem.nodes.select(pg="fixed_base").ids)  # raw-ops: reaction read-back
err_base_shear = (R_base_x + V_BASE) / V_BASE
assert abs(err_base_shear) < TOL_CHECK, f"base shear mismatch: {err_base_shear:.2e}"

print(f"Floor mass {M_FLOOR:.0f} kg per floor; base shear {V_BASE / KN:.1f} kN "
      f"(F_L1 = {F_L1 / KN:.1f} kN, F_L2 = {F_L2 / KN:.1f} kN)")
for mode, T in enumerate(periods, start=1):
    print(f"Mode {mode}: T = {T:.4f} s")
print(f"Level 1: ux = {u_L1 / MM:.2f} mm, drift ratio = {drift_L1:.5f}")
print(f"Level 2: ux = {u_L2 / MM:.2f} mm, drift ratio = {drift_L2:.5f}")
print(f"Base shear check: sum Rx = {-R_base_x / KN:.2f} kN vs V = {V_BASE / KN:.2f} kN "
      f"(rel. error {err_base_shear:.1e})")

with open(OUT / "periods.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["mode", "period_s"])
    for mode, T in enumerate(periods, start=1):
        writer.writerow([mode, f"{T:.6f}"])

with open(OUT / "drifts.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["level", "storey_force_kN", "ux_mm", "drift_ratio"])
    writer.writerow([1, f"{F_L1 / KN:.3f}", f"{u_L1 / MM:.4f}", f"{drift_L1:.6f}"])
    writer.writerow([2, f"{F_L2 / KN:.3f}", f"{u_L2 / MM:.4f}", f"{drift_L2:.6f}"])
