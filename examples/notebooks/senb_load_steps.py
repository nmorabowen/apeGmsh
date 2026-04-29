# %% [markdown]
# # SENB beam — incremental loading + recorded response curves
#
# Single-edge notched beam in 3-point bending, linear-elastic, plane stress.
# We use apeGmsh for the mesh + the OpenSees bridge to declare the model,
# then drive openseespy through `n_steps` load increments and record
# load-vs-displacement and load-vs-CMOD curves at every step.
#
# Because the material is linear-elastic, every recorded curve is a
# straight line — the slope is the secant stiffness of the cracked beam,
# which we use to back out an "effective Young's modulus" and compare to
# Euler-Bernoulli beam theory.

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# %% [markdown]
# ## Parameters (mm, N, MPa)

# %%
L_BEAM = 2000.0    # span
H      = 500.0     # depth
CX     = L_BEAM / 2
A_CRK  = 100.0     # crack depth
THICK  = 1.0

E   = 30_000.0
NU  = 0.2
P   = 1_000.0      # total target load
N_STEPS = 20       # load increments

NX       = 41
NY_CRACK = 6
NY_TOP   = 21

EMIT_PY = Path(__file__).with_name("senb_emit.py") if "__file__" in dir() else Path("senb_emit.py").resolve()

# %% [markdown]
# ## Build the mesh + apeGmsh OpenSees bridge

# %%
with apeGmsh(model_name="senb_beam", verbose=False) as g:

    geo = g.model.geometry

    p_bl  = geo.add_point(0,      0,     0)
    p_cb  = geo.add_point(CX,     0,     0)
    p_br  = geo.add_point(L_BEAM, 0,     0)
    p_il  = geo.add_point(0,      A_CRK, 0)
    p_tip = geo.add_point(CX,     A_CRK, 0)
    p_ir  = geo.add_point(L_BEAM, A_CRK, 0)
    p_tl  = geo.add_point(0,      H,     0)
    p_tc  = geo.add_point(CX,     H,     0)
    p_tr  = geo.add_point(L_BEAM, H,     0)

    L_bot_L   = geo.add_line(p_bl,  p_cb)
    L_bot_R   = geo.add_line(p_cb,  p_br)
    L_crack   = geo.add_line(p_cb,  p_tip)
    L_center  = geo.add_line(p_tip, p_tc)
    L_left_L  = geo.add_line(p_bl,  p_il)
    L_right_L = geo.add_line(p_br,  p_ir)
    L_left_U  = geo.add_line(p_il,  p_tl)
    L_right_U = geo.add_line(p_ir,  p_tr)
    L_mid_L   = geo.add_line(p_il,  p_tip)
    L_mid_R   = geo.add_line(p_tip, p_ir)
    L_top_L   = geo.add_line(p_tl,  p_tc)
    L_top_R   = geo.add_line(p_tc,  p_tr)

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

    g.physical.add_surface([S_BL, S_BR, S_TL, S_TR], name="Domain")
    g.physical.add_curve([L_crack],                  name="Crack")
    g.physical.add_point([p_cb],                     name="CrackBase")
    g.physical.add_point([p_bl],                     name="Pin")
    g.physical.add_point([p_br],                     name="Roller")
    g.physical.add_point([p_tc],                     name="LoadPoint")

    st = g.mesh.structured
    for L_ in (L_bot_L, L_bot_R, L_mid_L, L_mid_R, L_top_L, L_top_R):
        st.set_transfinite_curve(L_, NX)
    for L_ in (L_crack, L_left_L, L_right_L):
        st.set_transfinite_curve(L_, NY_CRACK)
    for L_ in (L_center, L_left_U, L_right_U):
        st.set_transfinite_curve(L_, NY_TOP)
    st.set_transfinite_surface(S_BL, corners=[p_bl,  p_cb,  p_tip, p_il])
    st.set_transfinite_surface(S_BR, corners=[p_cb,  p_br,  p_ir,  p_tip])
    st.set_transfinite_surface(S_TL, corners=[p_il,  p_tip, p_tc,  p_tl])
    st.set_transfinite_surface(S_TR, corners=[p_tip, p_ir,  p_tr,  p_tc])
    for S in (S_BL, S_BR, S_TL, S_TR):
        st.set_recombine(S, dim=2)

    g.mesh.generation.generate(dim=2)
    g.mesh.editing.crack("Crack", dim=1, open_boundary="CrackBase")

    # ---- apeGmsh OpenSees bridge ----
    g.opensees.set_model(ndm=2, ndf=2)
    g.opensees.materials.add_nd_material("Mat", "ElasticIsotropic", E=E, nu=NU)
    g.opensees.elements.assign(
        "Domain", "quad",
        material="Mat", thick=THICK, eleType="PlaneStress",
    )
    g.opensees.elements.fix("Pin",    dofs=[1, 1])
    g.opensees.elements.fix("Roller", dofs=[0, 1])
    with g.loads.pattern("Bending"):
        g.loads.point(pg="LoadPoint", force_xyz=(0.0, -P, 0.0))

    fem = g.mesh.queries.get_fem_data(dim=2)
    g.opensees.ingest.loads(fem)

    pin_id    = int(fem.nodes.get(pg="Pin").ids[0])
    roller_id = int(fem.nodes.get(pg="Roller").ids[0])
    load_id   = int(fem.nodes.get(pg="LoadPoint").ids[0])
    mouth_ids = sorted(
        int(nid) for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords)
        if abs(xyz[0] - CX) < 1e-6 and abs(xyz[1]) < 1e-6
    )
    assert len(mouth_ids) == 2

    g.opensees.build()
    g.opensees.export.py(str(EMIT_PY))
    print(g.opensees.inspect.summary())

# %% [markdown]
# ## Hand the model to live openseespy and run incrementally
#
# `LoadControl(dLambda)` advances the load factor `lambda` by `dLambda`
# every `analyze(1)`.  After `N_STEPS` calls, `lambda = 1.0` and the
# pattern's full nominal load `P` is applied.

# %%
ops.wipe()
exec(EMIT_PY.read_text(), {'ops': ops, '__name__': '__main__'})

ops.system('UmfPack')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 1.0 / N_STEPS)
ops.algorithm('Linear')
ops.analysis('Static')

P_history     = []
uy_load_hist  = []
ux_load_hist  = []
cmod_hist     = []
react_pin_hist    = []
react_roller_hist = []

for step in range(N_STEPS):
    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError(f"step {step}: analyze failed (code {ok})")
    P_now = (step + 1) * (P / N_STEPS)
    P_history.append(P_now)

    ux_L, uy_L = ops.nodeDisp(load_id)
    uy_load_hist.append(uy_L)
    ux_load_hist.append(ux_L)

    ux1 = ops.nodeDisp(mouth_ids[0])[0]
    ux2 = ops.nodeDisp(mouth_ids[1])[0]
    cmod_hist.append(abs(ux2 - ux1))

    ops.reactions()
    react_pin_hist.append(ops.nodeReaction(pin_id)[1])
    react_roller_hist.append(ops.nodeReaction(roller_id)[1])

P_history     = np.array(P_history)
uy_load_hist  = np.array(uy_load_hist)
cmod_hist     = np.array(cmod_hist)
react_pin_hist    = np.array(react_pin_hist)
react_roller_hist = np.array(react_roller_hist)

print(f"\nFinal load   : {P_history[-1]:.1f} N")
print(f"Final Uy@load: {uy_load_hist[-1]:+.4e} mm")
print(f"Final CMOD   : {cmod_hist[-1]:.4e} mm")
print(f"Final react  : pin={react_pin_hist[-1]:.1f}  roller={react_roller_hist[-1]:.1f}  "
      f"sum={react_pin_hist[-1] + react_roller_hist[-1]:.1f}")

# %% [markdown]
# ## Plot the response curves

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].plot(-uy_load_hist, P_history, "o-")
axes[0].set_xlabel("|Uy| at load point  [mm]")
axes[0].set_ylabel("Applied load P  [N]")
axes[0].set_title("Load vs deflection")
axes[0].grid(alpha=0.3)

axes[1].plot(cmod_hist, P_history, "s-", color="C1")
axes[1].set_xlabel("CMOD  [mm]")
axes[1].set_ylabel("Applied load P  [N]")
axes[1].set_title("Load vs crack-mouth opening")
axes[1].grid(alpha=0.3)

axes[2].plot(P_history, react_pin_hist,    "v-", label="Pin Ry")
axes[2].plot(P_history, react_roller_hist, "^-", label="Roller Ry")
axes[2].plot(P_history, react_pin_hist + react_roller_hist,
             "k--", alpha=0.5, label="Sum (= P)")
axes[2].set_xlabel("Applied load P  [N]")
axes[2].set_ylabel("Reaction Ry  [N]")
axes[2].set_title("Reactions vs load")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Effective stiffness — compare to uncracked beam theory
#
# Linear elastic → load-deflection slope is the cracked-beam stiffness.
# Euler-Bernoulli midspan stiffness for the **uncracked** rectangular
# beam (length L, depth H, thickness t):
#
# $$
# K_{\text{EB}} = \frac{48 E I}{L^3}, \quad I = \frac{t H^3}{12}
# $$
#
# The cracked stiffness should be smaller than $K_{EB}$ because the
# notch reduces the effective bending stiffness of the midspan
# cross-section.  We also expect short-beam shear deformation
# (slenderness $L/H = 4$) to soften the response further.

# %%
slope_disp = np.polyfit(-uy_load_hist, P_history, 1)[0]   # K_FEM = dP / d|Uy|
I_uncracked = THICK * H**3 / 12.0
K_EB = 48.0 * E * I_uncracked / L_BEAM**3
print(f"K_FEM (cracked, plane stress + shear) = {slope_disp:.2f}  N/mm")
print(f"K_EB  (uncracked, Euler-Bernoulli)    = {K_EB:.2f}  N/mm")
print(f"Stiffness ratio K_FEM / K_EB = {slope_disp / K_EB:.3f}")

slope_cmod = np.polyfit(cmod_hist, P_history, 1)[0]
print(f"\nLoad/CMOD slope (compliance is 1/this) = {slope_cmod:.2f}  N/mm")
