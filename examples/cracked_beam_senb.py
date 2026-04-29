"""
Simply-supported beam with edge crack — 3-point bending (SENB)
================================================================

Pure apeGmsh model: geometry, mesh, materials, elements, BCs and the
load are all declared via the apeGmsh OpenSees bridge.  apeGmsh
validates the model, emits an openseespy-compatible script, we exec
it to populate the live ops session, then add the analysis steps.

Geometry  : L_BEAM (long) x H (tall) rectangle, 2-D plane stress.
Crack     : vertical edge notch at midspan, depth A_CRK from bottom.
Supports  : pin at bottom-left, roller at bottom-right.
Load      : downward point P at top midspan.
Material  : linear-elastic isotropic (E, NU).
Element   : quad4, PlaneStress.

Crack handling: the crack curve is a SINGLE shared OCC line
(L_crack, in BL right edge = BR left edge).  After meshing,
g.mesh.editing.crack runs Gmsh's Crack plugin and duplicates the
nodes along it.  The crack mouth at (CX, 0) is the open boundary —
its node is duplicated too.  The crack tip at (CX, A_CRK) stays a
single shared node (default plugin behaviour).
"""

from pathlib import Path

import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# ---------------------------------------------------------------------------
# Parameters  (consistent units: mm, N, MPa)
# ---------------------------------------------------------------------------
L_BEAM = 2000.0
H      = 500.0
CX     = L_BEAM / 2
A_CRK  = 100.0
THICK  = 1.0       # plane-stress thickness

E   = 30_000.0     # MPa
NU  = 0.2
P   = 1_000.0      # N (downward at midspan top)

NX       = 41
NY_CRACK = 6
NY_TOP   = 21

EMIT_PY = Path(__file__).with_name("senb_beam_emit.py")

# ---------------------------------------------------------------------------
# Build the model with apeGmsh
# ---------------------------------------------------------------------------
with apeGmsh(model_name="senb_beam", verbose=False) as g:

    geo = g.model.geometry

    # --- Points ---
    p_bl  = geo.add_point(0,      0,     0)
    p_cb  = geo.add_point(CX,     0,     0)   # crack mouth
    p_br  = geo.add_point(L_BEAM, 0,     0)
    p_il  = geo.add_point(0,      A_CRK, 0)
    p_tip = geo.add_point(CX,     A_CRK, 0)   # crack tip
    p_ir  = geo.add_point(L_BEAM, A_CRK, 0)
    p_tl  = geo.add_point(0,      H,     0)
    p_tc  = geo.add_point(CX,     H,     0)   # load point
    p_tr  = geo.add_point(L_BEAM, H,     0)

    # --- Curves ---
    L_bot_L   = geo.add_line(p_bl,  p_cb)
    L_bot_R   = geo.add_line(p_cb,  p_br)
    L_crack   = geo.add_line(p_cb,  p_tip)        # SINGLE crack curve
    L_center  = geo.add_line(p_tip, p_tc)
    L_left_L  = geo.add_line(p_bl,  p_il)
    L_right_L = geo.add_line(p_br,  p_ir)
    L_left_U  = geo.add_line(p_il,  p_tl)
    L_right_U = geo.add_line(p_ir,  p_tr)
    L_mid_L   = geo.add_line(p_il,  p_tip)
    L_mid_R   = geo.add_line(p_tip, p_ir)
    L_top_L   = geo.add_line(p_tl,  p_tc)
    L_top_R   = geo.add_line(p_tc,  p_tr)

    # --- Surfaces (4 conformal patches sharing every interface curve) ---
    S_BL = geo.add_plane_surface(
        geo.add_curve_loop([L_bot_L, L_crack, -L_mid_L, -L_left_L])
    )
    S_BR = geo.add_plane_surface(
        geo.add_curve_loop([L_bot_R, L_right_L, -L_mid_R, -L_crack])
    )
    S_TL = geo.add_plane_surface(
        geo.add_curve_loop([L_mid_L, L_center, -L_top_L, -L_left_U])
    )
    S_TR = geo.add_plane_surface(
        geo.add_curve_loop([L_mid_R, L_right_U, -L_top_R, -L_center])
    )

    # --- Physical groups ---
    g.physical.add_surface([S_BL, S_BR, S_TL, S_TR], name="Domain")
    g.physical.add_curve([L_crack], name="Crack")
    g.physical.add_point([p_cb],    name="CrackBase")   # mouth -> duplicate
    g.physical.add_point([p_bl],    name="Pin")
    g.physical.add_point([p_br],    name="Roller")
    g.physical.add_point([p_tc],    name="LoadPoint")

    # --- Transfinite mesh ---
    st = g.mesh.structured
    for L_curve in (L_bot_L, L_bot_R, L_mid_L, L_mid_R, L_top_L, L_top_R):
        st.set_transfinite_curve(L_curve, NX)
    for L_curve in (L_crack, L_left_L, L_right_L):
        st.set_transfinite_curve(L_curve, NY_CRACK)
    for L_curve in (L_center, L_left_U, L_right_U):
        st.set_transfinite_curve(L_curve, NY_TOP)
    st.set_transfinite_surface(S_BL, corners=[p_bl,  p_cb,  p_tip, p_il])
    st.set_transfinite_surface(S_BR, corners=[p_cb,  p_br,  p_ir,  p_tip])
    st.set_transfinite_surface(S_TL, corners=[p_il,  p_tip, p_tc,  p_tl])
    st.set_transfinite_surface(S_TR, corners=[p_tip, p_ir,  p_tr,  p_tc])
    for S in (S_BL, S_BR, S_TL, S_TR):
        st.set_recombine(S, dim=2)

    # --- Mesh + duplicate nodes along the crack ---
    g.mesh.generation.generate(dim=2)
    g.mesh.editing.crack("Crack", dim=1, open_boundary="CrackBase")

    # ----------------------------------------------------------------
    # OpenSees bridge — declare the model
    # ----------------------------------------------------------------
    g.opensees.set_model(ndm=2, ndf=2)

    g.opensees.materials.add_nd_material(
        "Mat", "ElasticIsotropic",
        E=E, nu=NU,
    )

    g.opensees.elements.assign(
        "Domain", "quad",
        material="Mat",
        thick=THICK, eleType="PlaneStress",
    )

    # Boundary conditions
    g.opensees.elements.fix("Pin",    dofs=[1, 1])  # Ux = Uy = 0
    g.opensees.elements.fix("Roller", dofs=[0, 1])  # Uy = 0

    # Load — pattern + concentrated point load
    with g.loads.pattern("Bending"):
        g.loads.point(pg="LoadPoint", force_xyz=(0.0, -P, 0.0))

    # Pull resolved FEM data, then ingest the loads into the bridge
    fem = g.mesh.queries.get_fem_data(dim=2)
    g.opensees.ingest.loads(fem)

    # Resolve the IDs we'll need for post-processing (BEFORE leaving
    # the with-block — once we exec the emitted .py the live ops
    # session has the same node IDs)
    pin_id    = int(fem.nodes.get(pg="Pin").ids[0])
    roller_id = int(fem.nodes.get(pg="Roller").ids[0])
    load_id   = int(fem.nodes.get(pg="LoadPoint").ids[0])

    # The two duplicated crack-mouth nodes (used to compute CMOD)
    mouth_ids = sorted(
        int(nid) for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords)
        if abs(xyz[0] - CX) < 1e-6 and abs(xyz[1]) < 1e-6
    )
    assert len(mouth_ids) == 2, (
        f"expected 2 crack-mouth nodes, got {len(mouth_ids)}"
    )

    # Validate + freeze
    g.opensees.build()

    # Emit a self-contained openseespy script
    g.opensees.export.py(str(EMIT_PY))

    # Quick summary while we still have the bridge handy
    print(g.opensees.inspect.summary())

# ---------------------------------------------------------------------------
# Populate the live openseespy session by execing the emitted script
# ---------------------------------------------------------------------------
ops.wipe()
exec(EMIT_PY.read_text(), {'ops': ops, '__name__': '__main__'})

# ---------------------------------------------------------------------------
# Linear static analysis
# ---------------------------------------------------------------------------
ops.system('UmfPack')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 1.0)
ops.algorithm('Linear')
ops.analysis('Static')
ok = ops.analyze(1)

print(f"\nAnalysis status: {'OK' if ok == 0 else f'FAIL ({ok})'}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
ux_L, uy_L = ops.nodeDisp(load_id)
print(f"\nLoad-point disp:  Ux={ux_L:+.4e}  Uy={uy_L:+.4e}  mm")

m1, m2 = mouth_ids
ux1, uy1 = ops.nodeDisp(m1)
ux2, uy2 = ops.nodeDisp(m2)
cmod = abs(ux2 - ux1)
print(f"\nCrack-mouth nodes at (CX, 0):")
print(f"  node {m1}:  Ux={ux1:+.4e}  Uy={uy1:+.4e}")
print(f"  node {m2}:  Ux={ux2:+.4e}  Uy={uy2:+.4e}")
print(f"CMOD = |Ux2 - Ux1| = {cmod:.4e} mm")

# Reactions — sanity check (sum vertical reactions should equal P)
ops.reactions()
Rx_pin,  Ry_pin  = ops.nodeReaction(pin_id)
Rx_roll, Ry_roll = ops.nodeReaction(roller_id)
print(f"\nReactions:")
print(f"  Pin    ({pin_id}) : Rx={Rx_pin:+.4e}  Ry={Ry_pin:+.4e}")
print(f"  Roller ({roller_id}): Rx={Rx_roll:+.4e}  Ry={Ry_roll:+.4e}")
print(f"  Sum Ry = {Ry_pin + Ry_roll:+.4e}  vs applied P = {P:+.4e}")

ops.wipe()
