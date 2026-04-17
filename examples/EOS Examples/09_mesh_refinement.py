# %% [markdown]
# # 09 — Mesh Refinement and Convergence
#
# **Curriculum slot:** Tier 2, slot 09.
# **Prerequisite:** 01 — Hello Plate.
#
# ## Purpose
#
# For any problem whose exact solution has **non-uniform strain**, a
# linear-triangle (``tri31``) FEM result converges toward the
# analytical answer as the mesh is refined. This notebook runs the
# same problem at a sequence of mesh sizes and plots the error.
#
# ## Problem
#
# A **cantilever plate** of length $L$ and depth $h$, thin ($t = 1$),
# fixed along the left edge, loaded by a vertical shear $V$
# distributed uniformly along the right edge. Plane stress.
#
# ```
#            V ↓ (distributed along right edge)
#    ┌────────────────────────┐
#    │                        ↓       +y
#    │                        ↓        │
#  h │        (plate)         ↓        └── +x
#    │                        ↓
#    │                        ↓
#    ●────────────────────────┘
#     ← uxy = 0 on left edge
#              L
# ```
#
# For an aspect ratio $L/h = 10$ Euler-Bernoulli theory gives a
# very good reference for the tip deflection:
#
# $$
# \delta_{\text{tip}}^{\text{EB}} \;=\; \dfrac{V\,L^{3}}{3\,E\,I},
# \qquad I = \dfrac{t\,h^{3}}{12}.
# $$
#
# Constant-strain triangles are **notoriously stiff in bending**
# (membrane locking on CST). At coarse meshes ``tri31`` will
# dramatically under-predict the tip deflection; the error shrinks
# as the mesh refines. Seeing the error trajectory is the core
# takeaway.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# --- Geometry + loading ---
L = 1.0         # length  [m]
h = 0.1         # depth   [m]
V = 100.0       # tip shear (total)  [N]
t = 1.0         # plane-stress thickness  [m]

# --- Material ---
E  = 2.0e11     # Pa
nu = 0.3

# --- Analytical reference (Euler-Bernoulli) ---
I = t * h**3 / 12.0
delta_EB = -V * L**3 / (3.0 * E * I)


# %% [markdown]
# ## 2. The sweep
#
# Run the same analysis at five mesh sizes and collect tip
# deflections. The ``run(lc)`` function builds a fresh apeGmsh
# geometry, meshes it at characteristic length ``lc``, hands the
# mesh to OpenSees, analyzes, and returns the tip $u_y$.

# %%
def run(lc: float) -> tuple[int, float]:
    """Build + solve at mesh size ``lc``. Return (n_elements, tip_uy)."""
    g_ctx = apeGmsh(model_name=f"09_mesh_ref_{lc:.3f}", verbose=False)
    g = g_ctx.__enter__()
    try:
        p_BL = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=lc)
        p_BR = g.model.geometry.add_point(L,   0.0, 0.0, lc=lc)
        p_TR = g.model.geometry.add_point(L,   h,   0.0, lc=lc)
        p_TL = g.model.geometry.add_point(0.0, h,   0.0, lc=lc)
        l_B = g.model.geometry.add_line(p_BL, p_BR)
        l_R = g.model.geometry.add_line(p_BR, p_TR)
        l_T = g.model.geometry.add_line(p_TR, p_TL)
        l_L = g.model.geometry.add_line(p_TL, p_BL)
        loop = g.model.geometry.add_curve_loop([l_B, l_R, l_T, l_L])
        surf = g.model.geometry.add_plane_surface([loop])
        g.model.sync()

        g.physical.add(1, [l_L],  name="left")
        g.physical.add(1, [l_R],  name="right")
        g.physical.add(2, [surf], name="plate")

        g.mesh.generation.generate(2)
        fem = g.mesh.queries.get_fem_data()
        n_elems = fem.info.n_elems

        # ── OpenSees build ─────────────────────────────────────
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        for nid, xyz in fem.nodes.get():
            ops.node(int(nid), float(xyz[0]), float(xyz[1]))
        ops.nDMaterial("ElasticIsotropic", 1, E, nu)
        for group in fem.elements.get(target="plate"):
            for eid, nodes in zip(group.ids, group.connectivity):
                ops.element("tri31", int(eid),
                            int(nodes[0]), int(nodes[1]), int(nodes[2]),
                            t, "PlaneStress", 1)

        # Left-edge: fully clamp (both DOFs).
        for n in fem.nodes.get(target="left").ids:
            ops.fix(int(n), 1, 1)

        # Right-edge: distribute the total shear V evenly by
        # tributary length over the edge nodes.
        rn_ids = list(fem.nodes.get(target="right").ids)
        # sort by y
        rn_coords = np.array([
            (int(n), float(fem.nodes.coords[i, 1]))
            for i, n in enumerate(fem.nodes.ids)
            if int(n) in set(int(r) for r in rn_ids)
        ])
        order = np.argsort(rn_coords[:, 1])
        rn_sorted = rn_coords[order]
        ys = rn_sorted[:, 1]
        trib = np.zeros(len(ys))
        trib[0]  = (ys[1]  - ys[0])  / 2.0
        trib[-1] = (ys[-1] - ys[-2]) / 2.0
        for i in range(1, len(ys) - 1):
            trib[i] = (ys[i+1] - ys[i-1]) / 2.0
        # Edge length should sum to h
        trib *= h / trib.sum()    # normalise in case of numerical drift

        ops.timeSeries("Constant", 1)
        ops.pattern("Plain", 1, 1)
        for (nid_y, tr) in zip(rn_sorted, trib):
            nid = int(nid_y[0])
            Fy = -V * float(tr) / h    # downward share of V
            ops.load(nid, 0.0, Fy)

        ops.system("BandGeneral"); ops.numberer("Plain"); ops.constraints("Plain")
        ops.test("NormUnbalance", 1e-10, 10); ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0); ops.analysis("Static")
        ops.analyze(1)

        # Mean tip u_y over the right-edge nodes.
        uy = float(np.mean([ops.nodeDisp(int(n), 2) for n in rn_ids]))
        return n_elems, uy
    finally:
        g_ctx.__exit__(None, None, None)


# %% [markdown]
# ## 3. Run the sweep

# %%
lc_values = [h, h / 2, h / 4, h / 8, h / 16]

print(f"{'lc':>10s}  {'n_elems':>10s}  {'FEM tip u_y':>16s}  {'err %':>10s}")
results: list[tuple[float, int, float, float]] = []
for lc in lc_values:
    n_elems, uy = run(lc)
    err = abs(uy - delta_EB) / abs(delta_EB) * 100.0
    results.append((lc, n_elems, uy, err))
    print(f"{lc:>10.4f}  {n_elems:>10d}  {uy:>16.6e}  {err:>9.4f}%")
print()
print(f"Euler-Bernoulli reference : {delta_EB:.6e}  m")


# %% [markdown]
# ## 4. Observation
#
# At the coarsest mesh ``lc = h`` the tri31 solution typically
# under-predicts the tip deflection by ~30-40% — CST membrane
# locking. As ``lc`` halves the error roughly quarters (O(h²)
# convergence for the stress/displacement integrals), and by
# ``lc = h/16`` the FEM tip deflection is within a few percent of
# the Euler-Bernoulli reference.
#
# Two lessons for later notebooks:
#
# 1. **tri31 is a membrane-stiff bending element.** For
#    bending-dominated problems prefer higher-order triangles
#    (tri6) or quadrilaterals (quad, SSPquad), or switch to shell
#    elements entirely (ShellMITC4, ASDShellQ4 — slot 18).
# 2. **Convergence curves are the acceptance test.** Whenever a
#    new apeGmsh geometry or mesh knob is introduced, running the
#    same problem across a refinement sweep and checking that the
#    answer stabilises is the discipline that catches modelling
#    bugs early.


# %% [markdown]
# ## 5. (Optional) log-log convergence plot
#
# Uncomment in Jupyter to see the classical straight-line error-vs-h
# log-log plot. Slope ≈ 2 is the theoretical expectation for the
# displacement norm of a C^0 element.

# %%
# import matplotlib.pyplot as plt
# lcs, ne, uys, errs = zip(*results)
# plt.figure()
# plt.loglog(lcs, errs, "o-")
# plt.xlabel("characteristic mesh size lc [m]")
# plt.ylabel("tip-deflection error [%]")
# plt.title("tri31 cantilever plate convergence")
# plt.grid(True, which="both", ls="--", alpha=0.4)
# plt.show()


# %% [markdown]
# ## What this unlocks
#
# * **Refinement sweep template.** Every later notebook that needs
#   a convergence acceptance test reuses this structure:
#   build inside a function, loop over ``lc``, print the error table.
# * **Awareness of element choice.** CST triangles lock in bending;
#   pick the element to suit the physics. Slot 18 (LTB of a shell
#   I-beam) uses ASDShellQ4 and doesn't have this problem.
