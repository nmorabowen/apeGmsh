# %% [markdown]
# # 17 — Modal Analysis of a Cantilever Beam
#
# **Curriculum slot:** Tier 5, slot 17.
# **Prerequisite:** 02 — 2D Cantilever Beam, 06 — Sections Catalog.
#
# ## Purpose
#
# Tier 5 covers the four major analysis types OpenSees exposes
# beyond static-linear: **modal**, **buckling**, **pushover**,
# **time-history**. This slot is the modal entry point.
#
# A linear-elastic cantilever of length $L$ with uniform mass per
# unit length $\bar{m} = \rho A$ has natural circular frequencies
#
# $$
# \omega_n \;=\; (\beta_n L)^{2} \sqrt{\dfrac{E I}{\bar{m}\,L^{4}}},
# $$
#
# where the non-dimensional eigenvalues $\beta_n L$ satisfy
# $\cos(\beta L)\cosh(\beta L) + 1 = 0$. The first three roots are
#
# | Mode | $\beta_n L$ | $\omega_n / \omega_1$ |
# |---|---|---|
# | 1 | 1.8751 | 1.000 |
# | 2 | 4.6941 | 6.267 |
# | 3 | 7.8548 | 17.55 |
#
# This notebook lumps mass onto beam nodes (consistent lumping is
# also an option but lumped is simpler and sufficient for the
# first few modes), runs an OpenSees eigenvalue analysis, and
# compares the first three frequencies to the analytical values.

# %% [markdown]
# ## 1. Imports and parameters

# %%
import numpy as np
import openseespy.opensees as ops

from apeGmsh import apeGmsh

# Geometry / section (same as slot 02)
L  = 3.0
E  = 2.1e11
nu = 0.3
G  = E / (2 * (1 + nu))
A  = 1.0e-3
Iy = 1.0e-5
Iz = 1.0e-5
J  = 2.0e-5
rho = 7850.0                       # steel density [kg/m^3]
m_bar = rho * A                    # mass per unit length [kg/m]

# Mesh density — higher for modal since the first 3 modes need
# enough segments to resolve the mode shapes (rule of thumb:
# ~10 elements per half-wavelength of the highest mode).
N_ELEM = 30
LC = L / N_ELEM


# %% [markdown]
# ## 2. Geometry + mesh (cantilever)

# %%
g_ctx = apeGmsh(model_name="17_modal", verbose=False)
g = g_ctx.__enter__()

p_base = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=LC)
p_tip  = g.model.geometry.add_point(L,   0.0, 0.0, lc=LC)
ln     = g.model.geometry.add_line(p_base, p_tip)
g.model.sync()

g.physical.add(0, [p_base], name="base")
g.physical.add(0, [p_tip],  name="tip")
g.physical.add(1, [ln],     name="beam")

g.mesh.structured.set_transfinite_curve(ln, N_ELEM + 1)
g.mesh.generation.generate(1)
fem = g.mesh.queries.get_fem_data()
print(f"mesh: {fem.info.n_nodes} nodes, {fem.info.n_elems} elements")


# %% [markdown]
# ## 3. OpenSees build

# %%
ops.wipe()
ops.model("basic", "-ndm", 3, "-ndf", 6)

# Nodes
for nid, xyz in fem.nodes.get():
    ops.node(int(nid), float(xyz[0]), float(xyz[1]), float(xyz[2]))

# Geometric transformation and beam elements
ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
for group in fem.elements.get(target="beam"):
    for eid, nodes in zip(group.ids, group.connectivity):
        ops.element("elasticBeamColumn", int(eid),
                    int(nodes[0]), int(nodes[1]),
                    A, E, G, J, Iy, Iz, 1)

# Fix the base fully
for n in fem.nodes.get(target="base").ids:
    ops.fix(int(n), 1, 1, 1, 1, 1, 1)

# Restrain all DOFs except uz (3) and ry (5) on non-base nodes.
# That isolates the in-plane bending modes that the analytical
# Euler-Bernoulli formula predicts. If we leave ux free the first
# axial mode (omega = (pi/2)*sqrt(E/(rho*L^2)) ~ 2700 rad/s) sneaks
# between the 2nd and 3rd bending modes and skews the mode ordering.
base_set = {int(n) for n in fem.nodes.get(target="base").ids}
for nid, _ in fem.nodes.get():
    if int(nid) in base_set:
        continue
    # free: uz, ry (DOFs 3, 5)     -> pure in-plane bending
    # fixed: ux, uy, uz=no, rx, ry=no, rz
    ops.fix(int(nid), 1, 1, 0, 1, 0, 1)


# %% [markdown]
# ## 4. Lumped mass assembly
#
# For a uniform beam of length $L$ meshed into $N$ equal segments,
# each internal node receives a tributary length $h = L/N$ and
# therefore a lumped mass $m_i = \bar{m} \, h$. End nodes get half
# that. The fixed base's mass is irrelevant (constrained out of the
# reduced problem) so we don't touch it.

# %%
h = L / N_ELEM
m_full = m_bar * h
m_half = m_full / 2.0

# Iterate all non-base mesh nodes; detect end (tip) vs. internal.
tip_node = int(next(iter(fem.nodes.get(target="tip").ids)))
tag_to_idx = {int(t): i for i, t in enumerate(fem.nodes.ids)}

for nid in fem.nodes.ids:
    nid = int(nid)
    if nid in base_set:
        continue
    m_here = m_half if nid == tip_node else m_full
    # ops.mass takes 6 values (mx, my, mz, Jx, Jy, Jz). Rotational
    # inertia is negligible for slender beams so we leave it as 0.
    ops.mass(nid, m_here, m_here, m_here, 0.0, 0.0, 0.0)


# %% [markdown]
# ## 5. Eigenvalue analysis

# %%
N_MODES = 3
eigen_values = ops.eigen(N_MODES)    # returns list of omega^2 for each mode
omegas = [float(np.sqrt(w2)) for w2 in eigen_values]
freqs  = [w / (2 * np.pi) for w in omegas]

print(f"{'mode':>6s} {'omega [rad/s]':>16s} {'freq [Hz]':>14s}")
for i, (w, f) in enumerate(zip(omegas, freqs), start=1):
    print(f"{i:>6d} {w:>16.6e} {f:>14.6e}")


# %% [markdown]
# ## 6. Verification against analytical $\omega_n$

# %%
# beta_n * L roots of  cos(x) cosh(x) + 1 = 0
beta_L = np.array([1.87510407, 4.69409113, 7.85475744])

# omega_n = (beta_n L)^2 * sqrt(EI / (m_bar * L^4))
omega_analytical = (beta_L**2) * np.sqrt(E * Iz / (m_bar * L**4))
freq_analytical  = omega_analytical / (2 * np.pi)

print(f"{'mode':>6s} {'FEM omega':>14s} {'analyt omega':>14s} {'err %':>10s}")
for i, (w_fem, w_an) in enumerate(zip(omegas, omega_analytical), start=1):
    err = abs(w_fem - w_an) / w_an * 100.0
    print(f"{i:>6d} {w_fem:>14.4e} {w_an:>14.4e} {err:>9.4f} %")


# %% [markdown]
# ## 7. (Optional) mode-shape extraction
#
# ``ops.nodeEigenvector(node_id, mode, dof)`` returns the mode's
# displacement at a given node/DOF. Uncomment the cell below in
# Jupyter to print the tip's modal amplitude.

# %%
# print(f"{'mode':>6s} {'tip u_z':>14s}")
# for m in range(1, N_MODES + 1):
#     ev = ops.nodeEigenvector(tip_node, m, 3)
#     print(f"{m:>6d} {ev:>14.6e}")


# %% [markdown]
# ## What this unlocks
#
# * **``ops.eigen(N)`` workflow.** Declare mass via ``ops.mass``;
#   call ``ops.eigen(N)`` with the desired mode count; receive a
#   list of $\omega^{2}$. Convert to $\omega$ or $f$ as needed.
# * **Lumped mass pattern.** For a uniform beam, end nodes get half
#   tributary, internal nodes get full. This is the minimal-viable
#   modal mass assembly.
# * **Out-of-plane restraint for 2D modal analyses in 3D models.**
#   Same pattern as slot 04 — fix $u_y$, $r_x$, $r_z$ on every
#   non-base node so only in-plane modes come out of the eigen
#   solve.

# %%
g_ctx.__exit__(None, None, None)
