# %% [markdown]
# # SENB beam — mesh convergence study
#
# Sweep the transfinite mesh densities `(NX, NY_CRACK, NY_TOP)` and
# report:
#
# - mid-span deflection at the load point
# - crack-mouth opening displacement (CMOD)
# - elapsed wall-clock per run
#
# Both quantities should converge to a finite limit as the mesh is
# refined — this notebook also fits a Richardson-style extrapolation
# and plots relative error vs total DOF count.

# %%
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# %% [markdown]
# ## Fixed parameters (mm, N, MPa)

# %%
L_BEAM = 2000.0
H      = 500.0
CX     = L_BEAM / 2
A_CRK  = 100.0
THICK  = 1.0
E      = 30_000.0
NU     = 0.2
P      = 1_000.0

EMIT_PY = Path(__file__).with_name("senb_emit_conv.py") if "__file__" in dir() else Path("senb_emit_conv.py").resolve()

# %% [markdown]
# ## Single SENB run as a function

# %%
def run_senb(nx: int, ny_crack: int, ny_top: int) -> dict:
    """Build, solve, and post-process one SENB run; return summary dict."""
    t0 = time.perf_counter()

    with apeGmsh(model_name="senb_conv", verbose=False) as g:
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
            st.set_transfinite_curve(L_, nx)
        for L_ in (L_crack, L_left_L, L_right_L):
            st.set_transfinite_curve(L_, ny_crack)
        for L_ in (L_center, L_left_U, L_right_U):
            st.set_transfinite_curve(L_, ny_top)
        st.set_transfinite_surface(S_BL, corners=[p_bl,  p_cb,  p_tip, p_il])
        st.set_transfinite_surface(S_BR, corners=[p_cb,  p_br,  p_ir,  p_tip])
        st.set_transfinite_surface(S_TL, corners=[p_il,  p_tip, p_tc,  p_tl])
        st.set_transfinite_surface(S_TR, corners=[p_tip, p_ir,  p_tr,  p_tc])
        for S in (S_BL, S_BR, S_TL, S_TR):
            st.set_recombine(S, dim=2)

        g.mesh.generation.generate(dim=2)
        g.mesh.editing.crack("Crack", dim=1, open_boundary="CrackBase")

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

        load_id = int(fem.nodes.get(pg="LoadPoint").ids[0])
        mouth_ids = sorted(
            int(nid) for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords)
            if abs(xyz[0] - CX) < 1e-6 and abs(xyz[1]) < 1e-6
        )
        n_nodes = len(fem.nodes.ids)
        n_elems = sum(len(grp.ids) for grp in fem.elements.get(pg="Domain"))

        g.opensees.build()
        g.opensees.export.py(str(EMIT_PY))

    ops.wipe()
    exec(EMIT_PY.read_text(), {'ops': ops, '__name__': '__main__'})
    ops.system('UmfPack')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError(f"analyze failed (code {ok}) for nx={nx}, ny_crack={ny_crack}")

    uy = ops.nodeDisp(load_id)[1]
    cmod = abs(ops.nodeDisp(mouth_ids[1])[0] - ops.nodeDisp(mouth_ids[0])[0])
    ops.wipe()

    return {
        "nx": nx, "ny_crack": ny_crack, "ny_top": ny_top,
        "n_nodes": n_nodes, "n_elems": n_elems, "n_dofs": 2 * n_nodes,
        "uy_load": uy, "cmod": cmod,
        "wall_s": time.perf_counter() - t0,
    }

# %% [markdown]
# ## Sweep — five mesh sizes (coarsest → finest)

# %%
mesh_sizes = [
    # (nx,  ny_crack, ny_top)
    ( 11,  4,  6),    # very coarse
    ( 21,  6, 11),    # coarse
    ( 41, 11, 21),    # medium
    ( 81, 21, 41),    # fine
    (121, 31, 61),    # very fine
]

results = []
for (nx, nyc, nyt) in mesh_sizes:
    print(f"running nx={nx:3d}  ny_crack={nyc:3d}  ny_top={nyt:3d}  ...", end=" ", flush=True)
    r = run_senb(nx, nyc, nyt)
    print(f"DOFs={r['n_dofs']:5d}  Uy={r['uy_load']:+.4e}  CMOD={r['cmod']:.4e}  "
          f"({r['wall_s']:.1f}s)")
    results.append(r)

# %% [markdown]
# ## Summary table

# %%
import pandas as pd
df = pd.DataFrame(results)
df["uy_load_abs"] = df["uy_load"].abs()
df

# %% [markdown]
# ## Convergence plots

# %%
ndofs = df["n_dofs"].to_numpy()
uy    = df["uy_load_abs"].to_numpy()
cmod  = df["cmod"].to_numpy()

# Use the finest run as the reference "exact" value
uy_ref   = uy[-1]
cmod_ref = cmod[-1]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(ndofs, uy,   "o-")
axes[0].axhline(uy_ref,   ls="--", color="grey", alpha=0.6,
                label=f"finest = {uy_ref:.4e}")
axes[0].set_xscale("log")
axes[0].set_xlabel("Total DOFs")
axes[0].set_ylabel("|Uy| at load point  [mm]")
axes[0].set_title("Mid-span deflection vs mesh refinement")
axes[0].grid(alpha=0.3, which="both")
axes[0].legend()

axes[1].plot(ndofs, cmod, "s-", color="C1")
axes[1].axhline(cmod_ref, ls="--", color="grey", alpha=0.6,
                label=f"finest = {cmod_ref:.4e}")
axes[1].set_xscale("log")
axes[1].set_xlabel("Total DOFs")
axes[1].set_ylabel("CMOD  [mm]")
axes[1].set_title("Crack-mouth opening vs mesh refinement")
axes[1].grid(alpha=0.3, which="both")
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Relative error vs the finest mesh
#
# Plotted on log-log axes — a straight line indicates a clean
# convergence rate, slope ≈ −1 for linear quad elements on smooth
# fields, less near singularities (the crack tip is a $r^{-1/2}$
# stress singularity for linear-elastic cracks, so we usually see a
# shallower slope here).

# %%
err_uy   = np.abs(uy[:-1]   - uy_ref)   / uy_ref
err_cmod = np.abs(cmod[:-1] - cmod_ref) / cmod_ref
ndofs_e  = ndofs[:-1]

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(ndofs_e, err_uy,   "o-", label="|Uy| error")
ax.loglog(ndofs_e, err_cmod, "s-", label="CMOD error")
ax.set_xlabel("Total DOFs")
ax.set_ylabel("relative error vs finest")
ax.set_title("Mesh convergence (log-log)")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.show()
