"""Plate v2 — exhaustive multi-step capture for ResultsViewer testing.

Companion script to ``example_plate_pyGmsh_v2.ipynb``. Same Lamé
thick-walled cylinder model, but:

- Splits the static solve into ``N_STEPS`` LoadControl substeps so the
  capture has a real time series (not a single t=1.0 point).
- Declares **every applicable recorder** for the problem:
  displacement, reaction_force, per-element global resisting forces,
  Gauss-point stress and strain, von Mises stress.
- Captures everything to a single ``plate_v2_full.h5`` for viewer test.
- Does NOT open a viewer — safe to run unattended.

Run with the venv interpreter::

    python "examples/EOS Examples/_run_plate_v2_full_capture.py"
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh
from apeGmsh.solvers.Recorders import Recorders
from apeGmsh.results.capture._domain import DomainCapture


# ── Constants (same as the notebook) ───────────────────────────────
inner_radius = 100.0
outer_radius = 200.0
lc = 10.0
E = 210.0e3
nu = 0.3
p = 100.0
thk = 1.0
N_STEPS = 10


# ── Build geometry, mesh, FEMData ──────────────────────────────────
g = apeGmsh(model_name="Plate2D_v2_full", verbose=False)
g.begin()

pc = g.model.geometry.add_point(0, 0, 0, lc=lc, label="center")
p1 = g.model.geometry.add_point(inner_radius, 0, 0, lc=lc, label="inner_x")
p2 = g.model.geometry.add_point(outer_radius, 0, 0, lc=lc, label="outer_x")
p3 = g.model.geometry.add_point(0, outer_radius, 0, lc=lc, label="outer_y")
p4 = g.model.geometry.add_point(0, inner_radius, 0, lc=lc, label="inner_y")

l1 = g.model.geometry.add_line(p1, p2, label="bottom")
l2 = g.model.geometry.add_arc(p2, pc, p3, label="outer_arc")
l3 = g.model.geometry.add_line(p3, p4, label="left")
l4 = g.model.geometry.add_arc(p4, pc, p1, label="inner_arc")

loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
surf = g.model.geometry.add_plane_surface(loop, label="plate")

g.physical.add(1, [l1], name="Sym_Y")
g.physical.add(1, [l3], name="Sym_X")
g.physical.add(1, [l4], name="Pressure")
g.physical.add(2, [surf], name="Plate")

g.loads.line(pg="Pressure", magnitude=p, normal=True)

g.mesh.generation.set_order(1)
g.mesh.generation.generate(2)
g.mesh.partitioning.renumber(dim=2, method="rcm", base=1)

fem = g.mesh.queries.get_fem_data(dim=2, remove_orphans=True)
print(f"Mesh: {fem.inspect.summary()}")


# ── Build OpenSees model ───────────────────────────────────────────
ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)

for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
    ops.node(int(nid), float(xyz[0]), float(xyz[1]))

ops.nDMaterial("ElasticIsotropic", 1, E, nu)
for eid, conn in zip(fem.elements.ids, fem.elements.connectivity):
    ops.element(
        "tri31", int(eid), *(int(n) for n in conn),
        thk, "PlaneStrain", 1,
    )

for nid in fem.nodes.get_ids(pg="Sym_Y"):
    ops.fix(int(nid), 0, 1)
for nid in fem.nodes.get_ids(pg="Sym_X"):
    ops.fix(int(nid), 1, 0)

ops.timeSeries("Linear", 1)
ops.pattern("Plain", 1, 1)
for ld in fem.nodes.loads:
    fx, fy, _ = ld.force_xyz or (0.0, 0.0, 0.0)
    ops.load(int(ld.node_id), fx, fy)


# ── Declare every applicable recorder ──────────────────────────────
recs = Recorders()
recs.nodes(components="displacement", pg="Plate")
recs.nodes(components="reaction_force", pg="Plate")
recs.elements(
    components=["nodal_resisting_force_x", "nodal_resisting_force_y"],
    pg="Plate",
)
recs.gauss(components="stress", pg="Plate")
recs.gauss(components="strain", pg="Plate")

spec = recs.resolve(fem, ndm=2, ndf=2)


# ── Run analysis with N_STEPS load substeps ────────────────────────
ops.constraints("Transformation")
ops.numberer("RCM")
ops.system("BandGeneral")
ops.test("NormDispIncr", 1e-8, 10)
ops.algorithm("Newton")
ops.integrator("LoadControl", 1.0 / N_STEPS)
ops.analysis("Static")

results_path = Path(__file__).resolve().parent / "plate_v2_full.h5"
if results_path.exists():
    results_path.unlink()

with DomainCapture(spec, results_path, fem, ndm=2, ndf=2) as cap:
    cap.begin_stage("static_loadup", kind="static")
    for step in range(N_STEPS):
        ok = ops.analyze(1)
        if ok != 0:
            raise RuntimeError(f"analyze failed at step {step+1} (code {ok})")
        cap.step(t=ops.getTime())
    cap.end_stage()

ops.wipe()
g.end()

size_kb = results_path.stat().st_size / 1024
print(f"\nWrote {results_path}")
print(f"  size: {size_kb:.1f} KB")
print(f"  steps: {N_STEPS} (t = 0.1, 0.2, ..., 1.0)")
print(f"  recorders: displacement, reaction_force, "
      f"nodal_resisting_force, stress, strain")
