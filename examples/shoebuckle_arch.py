"""Shoebuckle arch frame — apeGmsh build + live apeSees run, MPCO output.

A planar portal "cimbra": two pinned columns and a circular arch through
the crown, modelled as a 2-D fiber-section beam-column frame
(``ndm=2, ndf=3``) and pushed under a parametrised inward "lining
pressure" until the limit point.

Pipeline
--------
1.  apeGmsh builds the line geometry (2 columns + 1 circular arch),
    meshes it into ~0.1 m beam segments, and emits a ``FEMData``
    snapshot (``dim=1``).
2.  ``apeSees(fem)`` declares the custom ASDSteel1D fiber section,
    ``dispBeamColumn`` elements, pinned bases, the per-node pressure
    pattern, the analysis chain and an MPCO recorder.
3.  ``ops.run()`` pushes the whole deck into the in-process openseespy
    domain (no ``analyze``).
4.  A hand-written driver ramps ``LoadControl`` to ``lambda = 1`` with
    limit-point detection and an optional ``DisplacementControl``
    restart to trace the post-peak branch. The MPCO recorder captures
    every committed step.

Run this with the OpenSees venv interpreter (it needs an openseespy
build that has ASDSteel1D and the MPCO recorder compiled in):

    & "C:\\Users\\nmora\\venv\\opensees_venv\\Scripts\\python.exe" \\
        examples/shoebuckle_arch.py

Units: SI throughout — metres, newtons, pascals.

NOTE ON THE MATERIAL (spec vs. reality)
---------------------------------------
The real ``uniaxialMaterial ASDSteel1D`` takes ``E sy su eu`` and
*derives* its two-term Chaboche hardening internally so the initial
slope is ~E and the stress reaches ``su`` at strain ``eu`` — the
H1/gamma1/H2/gamma2 table in the design note is informational, not an
input. ``su`` was not in the spec; the values below use AISC 341 A36
expected strengths (Ry=1.5 on Fy, Rt~1.2 on Fu). Confirm/override
SY/SU before trusting absolute capacities — ``su - sy`` sets how fast
the auto-derived hardening saturates. The nominal sensitivity bracket
is ``SY=250e6, SU=400e6``.

NOTE ON apeGmsh LAYERING (what comes from where)
------------------------------------------------
* Geometry/topology is apeGmsh: points, lines, the circular arc, and
  the ``LeftColumn`` / ``Arch`` / ``RightColumn`` / ``Frame`` / ``Base``
  groups. Node→region classification reads those groups
  (``classify_regions``) — no raw coordinate tests.
* ``g.sections.*`` builds *mesh geometry* (Parts / solids / shells),
  not an OpenSees fiber section. A ``dispBeamColumn`` fiber section is
  necessarily ``ops.section.Fiber``; its patch corners come from one
  named source (``w_fiber_patches``).
* The load is declared through ``g.loads`` (apeGmsh's loads API), not
  the OpenSees bridge. The per-node parametrised field ``p_i = sum_k
  alpha_k psi_k(s_i)`` is injected with ``g.loads.point_closest`` (the
  one g.loads primitive that places a distinct force per node); region
  + normal come from the apeGmsh-labelled curves. apeSees emits the
  resolved ``fem.nodes.loads`` as a synthesized Linear timeSeries +
  Plain pattern (ADR 0001 — implemented in this change).
"""
from __future__ import annotations

import math

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

# ---------------------------------------------------------------------------
# Parameters (edit here)
# ---------------------------------------------------------------------------

SPAN = 5.9          # m, distance between column centrelines / springings
H_COL = 2.2         # m, column height (base -> springing)
RISE = 2.2          # m, arch rise above the springing line
ELEM_SIZE = 0.1     # m, target beam-element length

# Custom W section (mm in the spec -> m here).
BF = 0.150          # flange width
HT = 0.160          # overall depth
TF = 0.012          # flange thickness
TW = 0.008          # web thickness

# ASDSteel1D — A36 with AISC expected strengths. Bracket: 250e6 / 400e6.
E_STEEL = 200.0e9   # Pa
SY = 375.0e6        # Pa   = Ry * Fy = 1.5 * 250 MPa
SU = 480.0e6        # Pa   ~ Rt * Fu = 1.2 * 400 MPa   (ASSUMPTION)
EU = 0.20           # -    A36 nominal ultimate strain

# Analysis.
DLAM = 0.005        # LoadControl load-factor increment (~200 steps to lambda=1)
TOL = 1.0e-6
MAX_ITER = 30
MPCO_FILE = "shoebuckle.mpco"

# Reference pressure shape. ``pressure(s)`` returns Sum_k alpha_k psi_k(s)
# — this is the seam the future LHS wrapper samples over. For now one
# hard-coded physically-plausible alpha-vector (a near-uniform inward
# lining pressure with a mild asymmetry on the crown).
P0 = 5.0e3          # Pa-equivalent line pressure scale (N per unit s)
ALPHA = (
    1.00,           # uniform component (whole perimeter)
    0.15,           # extra on the left leg
    0.10,           # extra on the right leg
    0.25,           # extra on the left half of the crown
    0.05,           # extra on the right half of the crown
)


# ---------------------------------------------------------------------------
# Derived geometry
# ---------------------------------------------------------------------------

# Circular arch through the crown: chord = SPAN, rise = RISE.
R = (SPAN**2 + 4.0 * RISE**2) / (8.0 * RISE)
CX = SPAN / 2.0
CY = H_COL - (R - RISE)               # circle centre, below the springings
CROWN_Y = CY + R                      # == H_COL + RISE

# Springing polar angles about the centre (atan2, radians).
THETA_L = math.atan2(H_COL - CY, 0.0 - CX)        # left springing
THETA_R = math.atan2(H_COL - CY, SPAN - CX)       # right springing
SWEEP = THETA_L - THETA_R                          # total swept angle (>0)
ARC_LEN = R * SWEEP
PERIM_TOTAL = H_COL + ARC_LEN + H_COL              # left leg + arch + right leg


REGION_LEFT, REGION_ARCH, REGION_RIGHT = "L", "A", "R"


def classify_regions(fem) -> dict[int, str]:
    """Map node id -> region using the apeGmsh-labelled curves.

    Region membership comes from the ``LeftColumn`` / ``Arch`` /
    ``RightColumn`` physical groups built on the model — NOT from raw
    coordinate tests. The two springing nodes are shared by a column
    and the arch; the arch is applied last so it wins them (keeps the
    perimeter coordinate and the normal continuous across springing).
    """
    region: dict[int, str] = {}
    for nid in fem.nodes.get(pg="LeftColumn").ids:
        region[int(nid)] = REGION_LEFT
    for nid in fem.nodes.get(pg="RightColumn").ids:
        region[int(nid)] = REGION_RIGHT
    for nid in fem.nodes.get(pg="Arch").ids:
        region[int(nid)] = REGION_ARCH
    return region


def perimeter_s(x: float, y: float, region: str) -> float:
    """Perimeter coordinate ``s`` (0 at left base, CCW). The region
    is resolved from apeGmsh labels by :func:`classify_regions`; here
    we only parametrise arclength within the known region."""
    if region == REGION_LEFT:
        return y
    if region == REGION_RIGHT:
        return PERIM_TOTAL - y
    theta = math.atan2(y - CY, x - CX)                     # arch
    return H_COL + R * (THETA_L - theta)


def inward_normal(x: float, y: float, region: str) -> tuple[float, float]:
    """Unit normal pointing into the tunnel interior (compression +)."""
    if region == REGION_LEFT:
        return (1.0, 0.0)            # interior is +x of the left leg
    if region == REGION_RIGHT:
        return (-1.0, 0.0)           # interior is -x of the right leg
    # Arch: outward radial is (P - C)/R; interior is toward the centre.
    dx, dy = x - CX, y - CY
    norm = math.hypot(dx, dy)
    return (-dx / norm, -dy / norm)


def pressure(s: float) -> float:
    """Sum_k alpha_k * psi_k(s)  — the load-shape seam for LHS."""
    a0, aL, aR, aCL, aCR = ALPHA
    s_arch0 = H_COL
    s_arch1 = H_COL + ARC_LEN
    s_crown = H_COL + ARC_LEN / 2.0

    p = a0                                              # uniform
    if s < s_arch0:                                     # left leg ramp
        p += aL * (1.0 - s / H_COL)
    elif s > s_arch1:                                   # right leg ramp
        p += aR * (s - s_arch1) / H_COL
    else:                                               # crown halves
        if s <= s_crown:
            p += aCL * (1.0 - abs(s - s_arch0) / (ARC_LEN / 2.0))
        else:
            p += aCR * (1.0 - abs(s_arch1 - s) / (ARC_LEN / 2.0))
    return P0 * p


# ---------------------------------------------------------------------------
# 1. Geometry + mesh + FEMData
# ---------------------------------------------------------------------------

def build_fem():
    with apeGmsh(model_name="shoebuckle", verbose=False) as g:
        gm = g.model.geometry

        p_base_l  = gm.add_point(0.0,  0.0,   0.0, label="base_left")
        p_base_r  = gm.add_point(SPAN, 0.0,   0.0, label="base_right")
        p_spr_l   = gm.add_point(0.0,  H_COL, 0.0, label="spring_left")
        p_spr_r   = gm.add_point(SPAN, H_COL, 0.0, label="spring_right")
        p_centre  = gm.add_point(CX,   CY,    0.0)   # arc construction centre

        col_l = gm.add_line(p_base_l, p_spr_l, label="LeftColumn")
        col_r = gm.add_line(p_base_r, p_spr_r, label="RightColumn")
        # add_arc takes the SHORTER arc -> the 146.8 deg crown arc.
        arch = gm.add_arc(p_spr_l, p_centre, p_spr_r, label="Arch")

        g.physical.add_curve([col_l, col_r, arch], name="Frame")
        g.physical.add_point([p_base_l, p_base_r], name="Base")

        # Per-curve solver groups for label-driven node classification
        # (canonical apeGmsh: promote the curve labels — no raw tags).
        g.labels.promote_to_physical("LeftColumn")
        g.labels.promote_to_physical("RightColumn")
        g.labels.promote_to_physical("Arch")

        # Deterministic element counts (~0.1 m): 22 col elems, ~79 arch.
        n_col = round(H_COL / ELEM_SIZE) + 1
        n_arch = round(ARC_LEN / ELEM_SIZE) + 1
        g.mesh.structured.set_transfinite_curve(col_l, n_nodes=n_col)
        g.mesh.structured.set_transfinite_curve(col_r, n_nodes=n_col)
        g.mesh.structured.set_transfinite_curve(arch, n_nodes=n_arch)

        g.mesh.generation.generate(dim=1)
        g.mesh.partitioning.renumber(dim=1, base=1)

        # First extraction: nodes + label-driven regions, used only to
        # build the pressure field. (FEMData is an immutable query —
        # we re-declare on the session and re-extract.)
        fem0 = g.mesh.queries.get_fem_data(dim=1, remove_orphans=True)
        region_of = classify_regions(fem0)

        # Inward-pressure field through apeGmsh's loads API. The field
        # p_i = sum_k alpha_k psi_k(s_i) varies per node; point_closest
        # is the g.loads primitive that injects a distinct force at the
        # mesh node nearest a coordinate (here, exactly each node).
        # Region + normal come from the apeGmsh-labelled curves.
        with g.loads.case("Pressure"):
            for nid, xyz in zip(fem0.nodes.ids, fem0.nodes.coords):
                x, y = float(xyz[0]), float(xyz[1])
                region = region_of[int(nid)]
                s = perimeter_s(x, y, region)
                nx, ny = inward_normal(x, y, region)
                p = pressure(s) * ELEM_SIZE                # nodal tributary
                g.loads.point.force_closest(
                    (x, y, 0.0), force=(p * nx, p * ny, 0.0),
                )

        # Re-extract: fem now carries the resolved fem.nodes.loads,
        # which apeSees emits per ADR 0001.
        fem = g.mesh.queries.get_fem_data(dim=1, remove_orphans=True)

    # Sanity: two pinned bases, and add_arc gave us the crown arc.
    base_ids = fem.nodes.get(pg="Base").ids
    assert len(base_ids) == 2, f"expected 2 base nodes, got {len(base_ids)}"
    ymax = max(float(c[1]) for c in fem.nodes.coords)
    assert abs(ymax - CROWN_Y) < 1e-3, (
        f"arch apex y={ymax:.4f} != crown {CROWN_Y:.4f}; add_arc picked "
        "the wrong arc — swap spring_left/spring_right."
    )
    n_loaded = sum(1 for _ in fem.nodes.loads)
    assert n_loaded > 0, "g.loads.point_closest produced no NodalLoadRecords"
    print(
        f"FEM: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements, "
        f"{n_loaded} loaded nodes "
        f"(R={R:.4f} m, sweep={math.degrees(SWEEP):.1f} deg, "
        f"arc_len={ARC_LEN:.3f} m, perimeter={PERIM_TOTAL:.3f} m)"
    )
    return fem


# ---------------------------------------------------------------------------
# 2. apeSees model declaration
# ---------------------------------------------------------------------------

def declare_model(fem):
    # default_orientation=None: 2-D model, no vecxz at emit time.
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)

    steel = ops.uniaxialMaterial.ASDSteel1D(
        E=E_STEEL, sy=SY, su=SU, eu=EU, fracture=True,
    )

    # Strong-axis fiber section. g.sections.* builds *mesh geometry*
    # (Parts / solids / shells), not an OpenSees fiber section — a
    # dispBeamColumn fiber section is necessarily ops.section.Fiber.
    # Patch corners come from one named source (no scattered offsets).
    section = ops.section.Fiber(patches=w_fiber_patches(steel))

    transf = ops.geomTransf.Linear()                      # swap Corotational
    integ = ops.beamIntegration.Legendre(section=section, n_ip=2)
    ops.element.dispBeamColumn(pg="Frame", transf=transf, integration=integ)

    ops.fix(pg="Base", dofs=(1, 1, 0))                    # pinned bases

    # No bridge load pattern: the inward-pressure field was declared
    # through g.loads (build_fem). apeSees emits fem.nodes.loads as a
    # synthesized Linear timeSeries + Plain pattern per ADR 0001.

    # Analysis chain (emitted by run(); the loop overrides per-step).
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=TOL, max_iter=MAX_ITER)
    ops.algorithm.KrylovNewton()
    ops.integrator.LoadControl(dlam=DLAM)
    ops.analysis.Static()

    # MPCO output: nodal kinematics/reactions + beam section force &
    # deformation. (STKO -E tokens; adjust to your build's vocabulary
    # if a response is missing in the .mpco.)
    ops.recorder.MPCO(
        file=MPCO_FILE,
        nodal_responses=("displacement", "reactionForce"),
        elem_responses=("force", "deformation"),
        nsteps=1,
    )
    return ops


def w_fiber_patches(steel):
    """Strong-axis W fiber section from the named section dimensions.

    Single source of truth for the patch corners (local y = depth,
    local z = flange width) — top flange, web, bottom flange.
    """
    from apeGmsh.opensees.section.fiber import RectPatch

    h2 = HT / 2.0
    bf2 = BF / 2.0
    y_web = h2 - TF
    return (
        RectPatch(material=steel, ny=1, nz=20,            # top flange
                  yI=y_web, zI=-bf2, yJ=h2, zJ=bf2),
        RectPatch(material=steel, ny=16, nz=1,            # web
                  yI=-y_web, zI=-TW / 2.0, yJ=y_web, zJ=TW / 2.0),
        RectPatch(material=steel, ny=1, nz=20,            # bottom flange
                  yI=-h2, zI=-bf2, yJ=-y_web, zJ=bf2),
    )


# ---------------------------------------------------------------------------
# 3 + 4. Live run + adaptive limit-point driver
# ---------------------------------------------------------------------------

def run_to_limit(ops, fem) -> None:
    ops.run(wipe=True)                       # full deck -> in-process domain
    import openseespy.opensees as osi

    node_ids = [int(n) for n in fem.nodes.ids]
    crown_id = max(
        zip(node_ids, fem.nodes.coords),
        key=lambda nc: float(nc[1][1]),      # highest node = crown
    )[0]

    def crown_disp() -> tuple[float, float]:
        return osi.nodeDisp(crown_id, 1), osi.nodeDisp(crown_id, 2)

    lam = 0.0
    step = 0
    max_steps = int(math.ceil(1.0 / DLAM)) + 5
    print(f"LoadControl ramp (dlam={DLAM}); crown node = {crown_id}")

    # Static + LoadControl + Linear timeSeries(factor=1): the domain
    # pseudo-time is exactly the load factor lambda.
    while lam < 1.0 - 1e-9 and step < max_steps:
        ok = osi.analyze(1)
        if ok == 0:
            step += 1
            lam = osi.getTime()
            if step % 20 == 0:
                ux, uy = crown_disp()
                print(f"  step {step:4d}  lambda={lam:7.4f}  "
                      f"crown=({ux:+.5f}, {uy:+.5f}) m")
            continue

        # Non-convergence: try a sterner solve at a smaller increment.
        recovered = False
        for sub in (0.25, 0.1):
            osi.integrator("LoadControl", DLAM * sub)
            osi.algorithm("ModifiedNewton", "-initial")
            osi.test("NormDispIncr", TOL, MAX_ITER * 4, 0)
            if osi.analyze(1) == 0:
                step += 1
                lam = osi.getTime()
                osi.integrator("LoadControl", DLAM)
                osi.algorithm("KrylovNewton")
                osi.test("NormDispIncr", TOL, MAX_ITER, 0)
                recovered = True
                break
        if recovered:
            continue

        # Genuine limit point.
        ux, uy = crown_disp()
        print(
            f"\nLIMIT POINT reached: lambda_u = {lam:.4f} "
            f"at step {step}\n  crown disp = ({ux:+.5f}, {uy:+.5f}) m"
        )
        _trace_post_peak(osi, crown_id)
        break
    else:
        ux, uy = crown_disp()
        print(
            f"\nReached lambda = {lam:.4f} (target 1.0) in {step} steps "
            f"without a limit point.\n  crown disp = "
            f"({ux:+.5f}, {uy:+.5f}) m"
        )

    print(f"MPCO results written to {MPCO_FILE}")


def _trace_post_peak(osi, crown_id: int, n: int = 80) -> None:
    """Switch to DisplacementControl on the crown to walk past the peak."""
    dof = 2  # vertical crown displacement drives the snap-through
    d_now = osi.nodeDisp(crown_id, dof)
    d_incr = math.copysign(2.0e-4, d_now if d_now != 0.0 else -1.0)
    osi.integrator("DisplacementControl", crown_id, dof, d_incr)
    osi.algorithm("KrylovNewton")
    osi.test("NormDispIncr", TOL, MAX_ITER * 2, 0)
    print(f"  post-peak: DisplacementControl node {crown_id} dof {dof} "
          f"dU={d_incr:+.1e} ({n} steps)")
    traced = 0
    for _ in range(n):
        if osi.analyze(1) != 0:
            break
        traced += 1
    print(f"  traced {traced} post-peak steps; "
          f"lambda_end = {osi.getTime():.4f}")


def main() -> None:
    fem = build_fem()
    ops = declare_model(fem)
    run_to_limit(ops, fem)


if __name__ == "__main__":
    main()
