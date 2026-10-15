"""KNOWN-BAD (the right truss numbers from a hand-written numpy stiffness solve: no apeGmsh, no OpenSees).

2-D Pratt truss bridge under bottom-chord joint loads, solved by direct stiffness in numpy.
Units: N, m, Pa.

Procedure:
    1. Data
    2. Stiffness: one 4 x 4 bar matrix per member, assembled
    3. Solve: the free degrees of freedom under the joint loads
    4. Checks and report: bar forces and the midspan deflection, written to result.json
"""
import json
from pathlib import Path

import numpy as np

# --- 1. Data
RESULT_FILE = Path(__file__).parent / "result.json"
N_PANELS = 6              # number of panels along the span
PANEL = 4.0               # m, panel length
DEPTH = 4.0               # m, truss depth
E_STEEL = 200e9           # Pa, Young's modulus
A_CHORD = 6.0e-3          # m2, chords and end posts
A_VERTICAL = 3.0e-3       # m2, verticals
A_DIAGONAL = 4.0e-3       # m2, diagonals
JOINT_LOAD = 100e3        # N, downward load at each interior bottom-chord joint
DOF_PER_JOINT = 2         # ux, uy
TOL_EQUILIBRIUM = 1e-6    # relative, solved reactions vs applied load

MEMBERS = {
    A_CHORD: ["B0-B1", "B1-B2", "B2-B3", "B3-B4", "B4-B5", "B5-B6", "T1-T2", "T2-T3", "T3-T4",
              "T4-T5", "B0-T1", "T5-B6"],
    A_VERTICAL: ["B1-T1", "B2-T2", "B3-T3", "B4-T4", "B5-T5"],
    A_DIAGONAL: ["T1-B2", "T2-B3", "T4-B3", "T5-B4"],
}
LOADED_JOINTS = ["B1", "B2", "B3", "B4", "B5"]

joint_xy = {}
for i in range(N_PANELS + 1):
    joint_xy[f"B{i}"] = np.array([i * PANEL, 0.0])
for i in range(1, N_PANELS):
    joint_xy[f"T{i}"] = np.array([i * PANEL, DEPTH])
index = {}
for row, joint in enumerate(joint_xy):
    index[joint] = row

# --- 2. Stiffness
n_dof = DOF_PER_JOINT * len(joint_xy)
stiffness = np.zeros((n_dof, n_dof))
bar_data = {}
for area, bars in MEMBERS.items():
    for bar in bars:
        start, end = bar.split("-")
        delta = joint_xy[end] - joint_xy[start]
        length = np.linalg.norm(delta)
        direction = np.concatenate([-delta, delta]) / length
        dofs = [2 * index[start], 2 * index[start] + 1, 2 * index[end], 2 * index[end] + 1]
        stiffness[np.ix_(dofs, dofs)] += E_STEEL * area / length * np.outer(direction, direction)
        bar_data[bar] = (dofs, direction, E_STEEL * area / length)

# --- 3. Solve
load = np.zeros(n_dof)
for joint in LOADED_JOINTS:
    load[2 * index[joint] + 1] = -JOINT_LOAD
fixed = [2 * index["B0"], 2 * index["B0"] + 1, 2 * index["B6"] + 1]
free = np.setdiff1d(np.arange(n_dof), fixed)
displacement = np.zeros(n_dof)
displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], load[free])

# --- 4. Checks and report
reactions = stiffness @ displacement - load
reaction_error = reactions[fixed].sum() / (len(LOADED_JOINTS) * JOINT_LOAD) - 1
assert abs(reaction_error) < TOL_EQUILIBRIUM, "the supports do not carry the load"
axial = {}
for bar, (dofs, direction, axial_stiffness) in bar_data.items():
    axial[bar] = axial_stiffness * direction @ displacement[dofs]
bottom_chord = []
for bar in ["B0-B1", "B1-B2", "B2-B3", "B3-B4", "B4-B5", "B5-B6"]:
    bottom_chord.append(axial[bar])
result = {
    "bottom_chord_max_axial_N": float(max(bottom_chord)),
    "end_post_axial_N": float(axial["B0-T1"]),
    "midspan_uy_m": float(displacement[2 * index["B3"] + 1]),
}
RESULT_FILE.write_text(json.dumps(result, indent=2))
print(result)
