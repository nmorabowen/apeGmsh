"""2-D Pratt truss bridge under gravity joint loads, linear static.

Six 4 m panels (span 24 m), 4 m deep, steel pin-jointed members. Pinned at
the left abutment, roller at the right. A 100 kN vertical load hangs at each
of the five interior bottom-chord joints. The run reports the extreme chord
and diagonal axial forces plus the midspan deflection, and checks the
midspan bottom-chord force against a method-of-sections hand calculation.

Units: N, m, Pa. Compression is negative.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apeGmsh import Results, apeGmsh
from apeGmsh.opensees import apeSees

# -- Data --------------------------------------------------------------------
out_dir = Path(__file__).parent / "out_d"
out_dir.mkdir(exist_ok=True)

n_panels = 6
a = 4.0                         # m, panel length
h = 4.0                         # m, truss depth
L = n_panels * a                # m, span

E = 200e9                       # Pa, structural steel
A_chord = 8.0e-3                # m^2, top and bottom chords
A_vertical = 3.0e-3             # m^2, verticals (hangers)
A_diagonal = 4.0e-3             # m^2, diagonals

P = 100e3                       # N, gravity load per interior bottom joint

nodes_per_member = 2            # end nodes only: one truss element per bar

solver_tolerance = 1e-8         # displacement-increment norm; linear system converges in one pass
solver_max_iter = 10
load_factor_step = 1.0          # the whole gravity case in one static step

# -- Geometry ----------------------------------------------------------------
with apeGmsh(model_name="pratt_truss", verbose=False) as g:
    geometry = g.model.geometry

    # Bottom-chord joints b0..b6 along the deck, top-chord joints t1..t5 above
    # the interior panel points. A Pratt truss has no top node over the supports.
    for i in range(n_panels + 1):
        geometry.add_point(i * a, 0.0, 0.0, label=f"b{i}")
    for i in range(1, n_panels):
        geometry.add_point(i * a, h, 0.0, label=f"t{i}")

    for i in range(n_panels):
        geometry.add_line(f"b{i}", f"b{i + 1}", label=f"bottom_{i + 1}")
    for i in range(1, n_panels - 1):
        geometry.add_line(f"t{i}", f"t{i + 1}", label=f"top_{i}")
    for i in range(1, n_panels):
        geometry.add_line(f"b{i}", f"t{i}", label=f"vertical_{i}")

    # End posts rise from the supports; interior diagonals slope down toward
    # midspan so they carry tension under gravity (the Pratt arrangement).
    geometry.add_line("b0", "t1", label="end_post_left")
    geometry.add_line(f"b{n_panels}", f"t{n_panels - 1}", label="end_post_right")
    geometry.add_line("t1", "b2", label="diagonal_left_1")
    geometry.add_line("t2", "b3", label="diagonal_left_2")
    geometry.add_line("t5", "b4", label="diagonal_right_1")
    geometry.add_line("t4", "b3", label="diagonal_right_2")

    # -- Groups & loads ------------------------------------------------------
    bottom_labels = [f"bottom_{i + 1}" for i in range(n_panels)]
    top_labels = [f"top_{i}" for i in range(1, n_panels - 1)]
    vertical_labels = [f"vertical_{i}" for i in range(1, n_panels)]
    diagonal_labels = [
        "end_post_left", "end_post_right",
        "diagonal_left_1", "diagonal_left_2",
        "diagonal_right_1", "diagonal_right_2",
    ]
    loaded_joint_labels = [f"b{i}" for i in range(1, n_panels)]

    g.physical.add_curve(bottom_labels, name="bottom_chord")
    g.physical.add_curve(top_labels, name="top_chord")
    g.physical.add_curve(vertical_labels, name="verticals")
    g.physical.add_curve(diagonal_labels, name="diagonals")
    g.physical.add_point(["b0"], name="pin")
    g.physical.add_point([f"b{n_panels}"], name="roller")
    g.physical.add_point(loaded_joint_labels, name="loaded_joints")
    g.physical.add_point([f"b{n_panels // 2}"], name="midspan")

    with g.loads.case("gravity"):
        g.loads.point.force("loaded_joints", (0.0, -P, 0.0))

    # -- Mesh ----------------------------------------------------------------
    # A pin-jointed bar must be a single truss element: an intermediate node
    # on a truss has no transverse stiffness and makes the system singular.
    member_labels = bottom_labels + top_labels + vertical_labels + diagonal_labels
    for member in member_labels:
        g.mesh.structured.set_transfinite_curve(member, n_nodes=nodes_per_member)
    g.mesh.generation.generate(dim=1)
    g.mesh.partitioning.renumber(dim=1, base=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

print(fem.info.summary())

# -- OpenSees model ----------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=2, ndf=2)

steel = ops.uniaxialMaterial.ElasticMaterial(E=E)
ops.element.Truss(pg="bottom_chord", A=A_chord, material=steel)
ops.element.Truss(pg="top_chord", A=A_chord, material=steel)
ops.element.Truss(pg="verticals", A=A_vertical, material=steel)
ops.element.Truss(pg="diagonals", A=A_diagonal, material=steel)

ops.fix(pg="pin", dofs=(1, 1))
ops.fix(pg="roller", dofs=(0, 1))

with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
    p.from_model("gravity")

# -- Analysis ----------------------------------------------------------------
ops.constraints.Plain()
ops.numberer.RCM()
ops.system.BandSPD()
ops.test.NormDispIncr(tol=solver_tolerance, max_iter=solver_max_iter)
ops.algorithm.Linear()
ops.integrator.LoadControl(dlam=load_factor_step)
ops.analysis.Static()

mpco_path = out_dir / "pratt_truss.mpco"
model_path = out_dir / "pratt_truss.h5"
deck_path = out_dir / "pratt_truss_deck.py"
member_groups = ["bottom_chord", "top_chord", "verticals", "diagonals"]

# Nodal results go to the MPCO archive; truss axial forces go to one plain
# text recorder per member group, which the report reads back directly.
ops.recorder.MPCO(
    file=str(mpco_path),
    nodal_responses=("displacement", "reactionForce"),
)
for group in member_groups:
    ops.recorder.Element(
        file=str(out_dir / f"axial_{group}.out"), response=("axialForce",), pg=group,
    )
ops.h5(str(model_path))
# The deck runs in its own interpreter so every recorder file is closed and
# readable when the run returns; one load step suffices for linear statics.
ops.py(str(deck_path), run=True, analyze_steps=1)

# -- Checks & report ---------------------------------------------------------
results = Results.from_mpco(str(mpco_path), fem=fem, model_h5=str(model_path))

midspan_deflection = results.nodes.get(pg="midspan", component="displacement_y").values[-1, 0]
reaction_pin = results.nodes.get(pg="pin", component="reaction_force_y").values[-1, 0]
reaction_roller = results.nodes.get(pg="roller", component="reaction_force_y").values[-1, 0]

# Each recorder row holds one axial force per member in the group's
# element-id order (no time column); the member labels name those columns.
member_of_element = {}          # element id -> member label
for member in member_labels:
    element_id = fem.elements.labels.element_ids(member)[0]
    member_of_element[element_id] = member

axial_force = {}                # member label -> axial force, N (tension +)
for group in member_groups:
    record = np.loadtxt(out_dir / f"axial_{group}.out", ndmin=2)
    forces_in_group = record[-1, :]
    element_ids_in_group = fem.elements.physical.element_ids(group)
    for element_id, force in zip(element_ids_in_group, forces_in_group):
        axial_force[member_of_element[element_id]] = float(force)

chord_forces = [axial_force[m] for m in bottom_labels + top_labels]
diagonal_forces = [axial_force[m] for m in diagonal_labels]
max_chord_tension = max(chord_forces)
max_chord_compression = min(chord_forces)
max_diagonal_tension = max(diagonal_forces)
max_diagonal_compression = min(diagonal_forces)

# Hand check, method of sections: cut the panel right of midspan (b3-b4,
# t3-t4, t4-b3) and take moments about t4 for the left free body. Only the
# bottom chord b3-b4 has a lever arm there, equal to the truss depth.
n_loads = n_panels - 1
reaction_hand = n_loads * P / 2
x_t4 = 4 * a
moment_about_t4 = reaction_hand * x_t4 - P * (3 * a + 2 * a + a)
bottom_midspan_hand = moment_about_t4 / h
bottom_midspan_fem = axial_force["bottom_4"]
bottom_midspan_error = (bottom_midspan_fem - bottom_midspan_hand) / bottom_midspan_hand

kN = 1e-3
print(f"reactions: pin {reaction_pin * kN:.1f} kN, roller {reaction_roller * kN:.1f} kN")
print(f"midspan deflection: {midspan_deflection * 1e3:.2f} mm")
print(f"chord axial force: max tension {max_chord_tension * kN:.1f} kN, "
      f"max compression {max_chord_compression * kN:.1f} kN")
print(f"diagonal axial force: max tension {max_diagonal_tension * kN:.1f} kN, "
      f"max compression {max_diagonal_compression * kN:.1f} kN")
print(f"midspan bottom chord: FEM {bottom_midspan_fem * kN:.1f} kN, "
      f"hand {bottom_midspan_hand * kN:.1f} kN, error {bottom_midspan_error:.2%}")
assert abs(bottom_midspan_error) < 1e-3, "midspan bottom chord disagrees with the hand calculation"

with open(out_dir / "member_forces.csv", "w") as csv:
    csv.write("member,axial_force_kN\n")
    for member, force in axial_force.items():
        csv.write(f"{member},{force * kN:.3f}\n")

# Axial-force diagram: members coloured blue (compression) to red (tension).
figure, axes = plt.subplots(figsize=(10, 3))
force_scale = max(abs(f) for f in axial_force.values())
for member, force in axial_force.items():
    end_coords = fem.elements.labels.node_coords(member)
    colour_position = 0.5 + 0.5 * force / force_scale       # 0 = -max, 0.5 = zero, 1 = +max
    colour = plt.cm.coolwarm(colour_position)
    axes.plot(end_coords[:, 0], end_coords[:, 1], color=colour, linewidth=3)
    axes.text(end_coords[:, 0].mean(), end_coords[:, 1].mean(), f"{force * kN:.0f}",
              fontsize=7, ha="center", va="center",
              bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
axes.set_aspect("equal")
axes.set_title("Pratt truss axial forces, kN (tension +)")
axes.set_xlabel("x, m")
axes.set_ylabel("y, m")
figure.tight_layout()
figure.savefig(out_dir / "axial_forces.png", dpi=150)
