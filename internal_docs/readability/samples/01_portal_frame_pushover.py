"""2-storey, 2-bay steel moment frame: gravity + displacement-controlled pushover.

Geometry (2-D, x horizontal, y up), SI units (N, m, Pa):

    bays 2 x 6.0 m, storeys 2 x 3.5 m, fixed bases.

Members are forceBeamColumn elements with Steel02 fiber W-sections
(columns W360x110-ish, beams W460x74-ish), P-Delta transformation on the
columns. Gravity is a uniform line load on every beam, applied first and
held; the lateral load is an inverted-triangle pattern at the left-column
joints, pushed under DisplacementControl on the roof until 4 % roof drift.

Outputs (in ./out_01/ next to this script):
    model.h5               apeGmsh/apeSees model archive
    pushover.csv           roof drift, roof displacement, base shear
    pushover.png           base shear vs roof drift
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402
from apeGmsh.opensees.emitter.live import get_ops  # noqa: E402
from apeGmsh.opensees.section.fiber import RectPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "out_01"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BAY = 6.0                 # m
STOREY = 3.5              # m
N_BAYS, N_STOREYS = 2, 2
H_TOTAL = N_STOREYS * STOREY

N_EL_COLUMN = 4           # elements per column
N_EL_BEAM = 6             # elements per beam

E_STEEL = 200e9           # Pa
FY = 345e6                # Pa
B_HARDENING = 0.01

# W-section dims (d, bf, tf, tw) in m.
COLUMN_W = (0.360, 0.256, 0.0199, 0.0114)   # ~W360x110
BEAM_W = (0.457, 0.190, 0.0145, 0.0090)     # ~W460x74

W_GRAVITY = 25e3          # N/m, uniform on every beam

TARGET_DRIFT = 0.04       # roof drift ratio to push to
N_PUSH_STEPS = 200


# ---------------------------------------------------------------------------
# 1. Geometry, groups, gravity load, mesh
# ---------------------------------------------------------------------------
def build_fem():
    with apeGmsh(model_name="portal_frame", verbose=False) as g:
        geo = g.model.geometry

        # Grid of joints: joint[(i, j)] -> point tag, i = column line, j = level.
        joint = {}
        for i in range(N_BAYS + 1):
            for j in range(N_STOREYS + 1):
                joint[i, j] = geo.add_point(i * BAY, j * STOREY, 0.0)

        columns = [
            geo.add_line(joint[i, j], joint[i, j + 1])
            for i in range(N_BAYS + 1) for j in range(N_STOREYS)
        ]
        beams = [
            geo.add_line(joint[i, j], joint[i + 1, j])
            for j in range(1, N_STOREYS + 1) for i in range(N_BAYS)
        ]

        g.physical.add_curve(columns, name="Columns")
        g.physical.add_curve(beams, name="Beams")
        g.physical.add_point([joint[i, 0] for i in range(N_BAYS + 1)], name="Base")
        # Lateral-load points: left-column joint at each floor.
        g.physical.add_point([joint[0, 1]], name="Floor1Left")
        g.physical.add_point([joint[0, 2]], name="RoofLeft")

        with g.loads.case("gravity"):
            g.loads.line("Beams", magnitude=W_GRAVITY, direction=(0.0, -1.0, 0.0))

        for c in columns:
            g.mesh.structured.set_transfinite_curve(c, n_nodes=N_EL_COLUMN + 1)
        for b in beams:
            g.mesh.structured.set_transfinite_curve(b, n_nodes=N_EL_BEAM + 1)
        g.mesh.generation.generate(dim=1)
        g.mesh.partitioning.renumber(dim=1, base=1)

        fem = g.mesh.queries.get_fem_data(dim=1)

    print(fem.info.summary())
    return fem


# ---------------------------------------------------------------------------
# 2. OpenSees model
# ---------------------------------------------------------------------------
def w_patches(material, d, bf, tf, tw):
    """Strong-axis W-shape fiber patches (local y = depth)."""
    h = d / 2.0
    yw = h - tf
    return (
        RectPatch(material=material, ny=4, nz=1, yI=yw, zI=-bf / 2, yJ=h, zJ=bf / 2),
        RectPatch(material=material, ny=12, nz=1, yI=-yw, zI=-tw / 2, yJ=yw, zJ=tw / 2),
        RectPatch(material=material, ny=4, nz=1, yI=-h, zI=-bf / 2, yJ=-yw, zJ=bf / 2),
    )


def declare_model(fem, control_node):
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)

    steel = ops.uniaxialMaterial.Steel02(fy=FY, E=E_STEEL, b=B_HARDENING)
    col_sec = ops.section.Fiber(patches=w_patches(steel, *COLUMN_W))
    beam_sec = ops.section.Fiber(patches=w_patches(steel, *BEAM_W))

    ops.element.forceBeamColumn(
        pg="Columns",
        transf=ops.geomTransf.PDelta(),
        integration=ops.beamIntegration.Lobatto(section=col_sec, n_ip=5),
    )
    ops.element.forceBeamColumn(
        pg="Beams",
        transf=ops.geomTransf.Linear(),
        integration=ops.beamIntegration.Lobatto(section=beam_sec, n_ip=5),
    )

    ops.fix(pg="Base", dofs=(1, 1, 1))

    # Pseudo-time t is the load factor. Gravity ramps 0 -> 1 over t in [0, 1]
    # and is then held; the lateral pattern is zero until t = 1 and then grows
    # as (t - 1). So during the pushover the lateral load factor = t - 1.
    t_end = 1e12
    gravity_ts = ops.timeSeries.Path(time=(0.0, 1.0, t_end), values=(0.0, 1.0, 1.0))
    lateral_ts = ops.timeSeries.Path(time=(0.0, 1.0, t_end), values=(0.0, 0.0, t_end - 1.0))

    with ops.pattern.Plain(series=gravity_ts) as p:
        p.from_model("gravity")

    # Inverted-triangle reference lateral load (sums to 1 N).
    with ops.pattern.Plain(series=lateral_ts) as p:
        p.load(pg="Floor1Left", forces=(1.0 / 3.0, 0.0, 0.0))
        p.load(pg="RoofLeft", forces=(2.0 / 3.0, 0.0, 0.0))

    # Gravity analysis chain; the pushover swaps the integrator below.
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()
    return ops


# ---------------------------------------------------------------------------
# 3. Analysis
# ---------------------------------------------------------------------------
def run(ops, fem, control_node):
    ops.run(wipe=True)
    osi = get_ops()

    if osi.analyze(10) != 0:
        raise RuntimeError("gravity analysis failed")
    base_nodes = [int(n) for n in fem.nodes.physical.node_ids("Base")]
    osi.reactions()
    gravity_reaction = sum(osi.nodeReaction(n, 2) for n in base_nodes)
    print(f"gravity done: sum Ry = {gravity_reaction / 1e3:.1f} kN "
          f"(expected {W_GRAVITY * N_BAYS * BAY * N_STOREYS / 1e3:.1f} kN), "
          f"roof uy = {osi.nodeDisp(control_node, 2) * 1e3:.2f} mm")

    def base_shear():
        osi.reactions()
        return -sum(osi.nodeReaction(n, 1) for n in base_nodes)

    du = TARGET_DRIFT * H_TOTAL / N_PUSH_STEPS
    osi.integrator("DisplacementControl", control_node, 1, du)

    drift, roof_u, shear = [0.0], [0.0], [base_shear()]
    for step in range(N_PUSH_STEPS):
        ok = osi.analyze(1)
        if ok != 0:  # one retry with a more robust algorithm
            osi.algorithm("KrylovNewton")
            ok = osi.analyze(1)
            osi.algorithm("Newton")
        if ok != 0:
            print(f"pushover stopped at step {step}: no convergence")
            break
        u = osi.nodeDisp(control_node, 1)
        roof_u.append(u)
        drift.append(u / H_TOTAL)
        shear.append(base_shear())
    return drift, roof_u, shear


# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------
def report(drift, roof_u, shear):
    with open(OUT / "pushover.csv", "w") as f:
        f.write("roof_drift,roof_disp_m,base_shear_kN\n")
        for d, u, v in zip(drift, roof_u, shear):
            f.write(f"{d:.6f},{u:.6f},{v / 1e3:.3f}\n")

    print("\n roof drift [%]   roof disp [mm]   base shear [kN]")
    rows = sorted(set(range(0, len(drift), max(1, len(drift) // 10))) | {len(drift) - 1})
    for k in rows:
        print(f"   {drift[k] * 100:8.2f}      {roof_u[k] * 1e3:10.1f}      {shear[k] / 1e3:10.1f}")

    k0 = shear[1] / roof_u[1]
    print(f"\ninitial stiffness  = {k0 / 1e3:.0f} kN/m")
    print(f"peak base shear    = {max(shear) / 1e3:.1f} kN")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([d * 100 for d in drift], [v / 1e3 for v in shear], lw=2)
    ax.set_xlabel("Roof drift [%]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("2-storey 2-bay steel frame: pushover")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pushover.png", dpi=150)
    plt.close(fig)


def main():
    fem = build_fem()
    control_node = int(fem.nodes.physical.node_ids("RoofLeft")[0])
    ops = declare_model(fem, control_node)
    ops.h5(str(OUT / "model.h5"))
    drift, roof_u, shear = run(ops, fem, control_node)
    report(drift, roof_u, shear)
    print(f"\noutputs in {OUT}")


if __name__ == "__main__":
    main()
