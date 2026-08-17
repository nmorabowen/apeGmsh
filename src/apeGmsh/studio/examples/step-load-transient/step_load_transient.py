"""Step-load transient — generic minimal cantilever, Newmark dynamics.

ADR 0095 Amendment 7, S8a example library seed. Ports the deck/chain/
recorder pattern of a private out-of-repo transient case (density +
Constant-series step load + Newmark + Ladruno ``-G energy``) onto a
small **generic** cantilever box — the specimen geometry does not
ship (per the amendment).

A small solid cantilever (``FourNodeTetrahedron``, density on the
material) fixed at one end. Two decks are built off the same mesh:

* a **static** twin — the full tip force applied through a ``Linear``
  timeSeries and a single ``LoadControl`` step (``lambda = 1``);
* the **transient** case — the same force applied as a **step** at
  ``t = 0`` (``Constant`` timeSeries) on a deck that carries the
  material density, integrated with Newmark(0.5, 0.25).

Both decks are emitted as Tcl and run through the classic OpenSees
exe (``requires: ladruno``). ``verify.py`` reads the Node-recorder tip
displacement from each and the transient's Ladruno energy-balance
channel to compute the dynamic amplification factor (DAF) and check
the energy closure.

Run (writes outputs to the current directory):

    python step_load_transient.py
    python verify.py

Units: SI throughout — metres, kilograms, seconds, newtons, pascals.
"""
from __future__ import annotations

import os
from pathlib import Path

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# ---------------------------------------------------------------------------
# Parameters (edit here)
# ---------------------------------------------------------------------------

# Generic minimal geometry: a small solid cantilever box.
L, B, H = 1.0, 0.1, 0.1     # m
MESH_SIZE = 0.035           # m, target tet edge length

# Material (density lives on the material — the pattern this example
# ports from a private out-of-repo transient case).
E = 200.0e9      # Pa
NU = 0.30
RHO = 7850.0     # kg/m^3

# Step load: transverse tip force, applied fully at t=0 in the
# transient case (Constant series), ramped in the static twin.
PZ = -1.0e4      # N

# Dynamics (Newmark average-acceleration; unconditionally stable).
DT = 5.0e-4      # s
N_STEPS = 200    # -> 0.10 s (~8 periods of the ~12 ms first mode)

TOL = 1.0e-8
MAX_ITER = 30

STATIC_TCL = Path("step_static.tcl")
TRANSIENT_TCL = Path("step_transient.tcl")
STATIC_MODEL_H5 = Path("step_static_model.h5")
TRANSIENT_MODEL_H5 = Path("step_transient_model.h5")
STATIC_NODE_OUT = Path("step_static_tip.out")
TRANSIENT_NODE_OUT = Path("step_transient_tip.out")
LADRUNO_FILE = Path("step_transient.ladruno")

DEFAULT_BIN = r"C:\Program Files\Ladruno\OpenSees\bin\OpenSees.exe"


# ---------------------------------------------------------------------------
# 1. Geometry + mesh + FEMData
# ---------------------------------------------------------------------------

def build_fem():
    with apeGmsh(model_name="step_load_transient", verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, L, B, H, label="body")
        g.physical.from_label("body", name="Solid")
        g.mesh.sizing.set_global_size(MESH_SIZE)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    print(f"FEM: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")
    return fem


def fixed_and_tip_nodes(fem) -> tuple[list[int], int]:
    """Fixed-end nodes (x=0 face) and the tip node (x=L, on the
    section centreline) — plain coordinate filtering, no physical
    groups needed beyond the whole-body "Solid" group used for the
    element declaration."""
    ids = [int(i) for i in fem.nodes.ids]
    coords = fem.nodes.coords
    tol = 1.0e-9
    fixed = [i for i, xyz in zip(ids, coords) if abs(float(xyz[0])) < tol]
    tip_face = [
        (i, xyz) for i, xyz in zip(ids, coords)
        if abs(float(xyz[0]) - L) < tol
    ]
    assert fixed, "no fixed-end nodes found at x=0"
    assert tip_face, "no tip-face nodes found at x=L"
    cy, cz = B / 2.0, H / 2.0
    tip = min(
        tip_face,
        key=lambda ix: (float(ix[1][1]) - cy) ** 2 + (float(ix[1][2]) - cz) ** 2,
    )[0]
    return fixed, tip


# ---------------------------------------------------------------------------
# 2. apeSees model declarations (static twin + transient)
# ---------------------------------------------------------------------------

def declare_static(fem, fixed: list[int], tip: int) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, rho=RHO)
    ops.element.FourNodeTetrahedron(pg="Solid", material=steel)
    ops.fix(nodes=tuple(fixed), dofs=(1, 1, 1))

    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        pat.load(node=tip, forces=(0.0, 0.0, PZ))

    ops.recorder.Node(
        file=str(STATIC_NODE_OUT), response="disp",
        nodes=(tip,), dofs=(1, 2, 3), time_format="dt",
    )

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=TOL, max_iter=MAX_ITER)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    ops.h5(str(STATIC_MODEL_H5))
    return ops


def declare_transient(fem, fixed: list[int], tip: int) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    steel = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, rho=RHO)
    ops.element.FourNodeTetrahedron(pg="Solid", material=steel)
    ops.fix(nodes=tuple(fixed), dofs=(1, 1, 1))

    # Step load: full PZ from t=0 (Constant series) — excites the
    # cantilever dynamically so KE/IE/DW are all nonzero.
    with ops.pattern.Plain(series=ops.timeSeries.Constant()) as pat:
        pat.load(node=tip, forces=(0.0, 0.0, PZ))

    ops.recorder.Node(
        file=str(TRANSIENT_NODE_OUT), response="disp",
        nodes=(tip,), dofs=(1, 2, 3), time_format="dt",
    )
    ops.recorder.Ladruno(
        file=str(LADRUNO_FILE),
        energy=True,   # -G energy: the point of this case
    )

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=TOL, max_iter=MAX_ITER)
    ops.algorithm.Newton()
    ops.integrator.Newmark(gamma=0.5, beta=0.25)
    ops.analysis.Transient()

    ops.h5(str(TRANSIENT_MODEL_H5))
    return ops


# ---------------------------------------------------------------------------
# 3. Emit + run through the classic OpenSees exe
# ---------------------------------------------------------------------------

def _resolve_bin() -> str:
    return os.environ.get("STEP_LOAD_OPENSEES", DEFAULT_BIN)


def run_static(ops: apeSees) -> None:
    bin_path = _resolve_bin()
    ops.tcl(
        str(STATIC_TCL), analyze_steps=1,
        run=True, bin=bin_path, log=str(STATIC_TCL) + ".log",
    )
    print(f"static:    Tcl={STATIC_TCL}  node_out={STATIC_NODE_OUT}")


def run_transient(ops: apeSees) -> None:
    bin_path = _resolve_bin()
    ops.tcl(
        str(TRANSIENT_TCL), analyze_steps=N_STEPS, analyze_dt=DT,
        run=True, bin=bin_path, log=str(TRANSIENT_TCL) + ".log",
    )
    print(f"transient: Tcl={TRANSIENT_TCL}  node_out={TRANSIENT_NODE_OUT}  "
          f"ladruno={LADRUNO_FILE}  ({N_STEPS} steps x {DT}s)")


def main() -> None:
    fem = build_fem()
    fixed, tip = fixed_and_tip_nodes(fem)
    print(f"fixed nodes: {len(fixed)}  tip node: {tip}")

    ops_static = declare_static(fem, fixed, tip)
    run_static(ops_static)

    ops_transient = declare_transient(fem, fixed, tip)
    run_transient(ops_transient)


if __name__ == "__main__":
    main()
