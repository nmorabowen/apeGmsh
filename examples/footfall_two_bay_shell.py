"""Footfall vibration of a two-bay concrete flat slab on columns.

AISC Design Guide 11 (2nd ed.) Chapter 7 walking evaluation, ADR 0109:
two 6 m x 6 m bays of 200 mm slab (ShellMITC4) on six 400 x 400 columns
(elasticBeamColumn), evaluated at the centre of each bay.

Run with the opensees venv::

    python examples/footfall_two_bay_shell.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import gmsh  # noqa: E402

from apeGmsh import Results, apeGmsh  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402

# --- floor -----------------------------------------------------------------
LX, LY = 6.0, 6.0          # one bay, m
H_COL = 3.5                # storey height, m
T_SLAB = 0.20              # slab thickness, m
E_DYN = 1.35 * 30.0e9      # DG11 Section 3: dynamic modulus 1.35 Ec, Pa
NU = 0.2
RHO_SLAB = 2500.0          # kg/m3
SDL = 100.0                # superimposed dead + live in motion, kg/m2
RHO_EQ = RHO_SLAB + SDL / T_SLAB
B_COL = 0.40               # square column, m
A_COL, I_COL, J_COL = B_COL**2, B_COL**4 / 12.0, 0.141 * B_COL**4
BETA = 0.01 + 0.01 + 0.01  # Table 4-2: structure + ceiling/ductwork + paper office
Q, G = 747.0, 9.81         # 168 lb walker, SI

OUT = HERE / "_footfall_two_bay"
OUT.mkdir(exist_ok=True)

# --- geometry: two bays + six columns, fragmented so column tops are slab nodes
with apeGmsh(model_name="footfall-two-bay") as g:
    geo = g.model.geometry
    bays = [geo.add_rectangle(0.0, 0.0, 0.0, LX, LY),
            geo.add_rectangle(LX, 0.0, 0.0, LX, LY)]
    cols = []
    for x in (0.0, LX, 2 * LX):
        for y in (0.0, LY):
            top = geo.add_point(x, y, 0.0)
            base = geo.add_point(x, y, -H_COL)
            cols.append(geo.add_line(base, top))
    g.model.boolean.fragment(bays, cols, dim=2)
    g.model.geometry.remove_orphans()   # fragment consumed the column-top points
    g.model.sync()

    eps = 1e-6
    slab = [t for _, t in gmsh.model.getEntities(2)]
    lines = [t for _, t in gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -H_COL - eps, 2 * LX + eps, LY + eps, eps, 1)
        if abs(gmsh.model.getBoundingBox(1, t)[5] - gmsh.model.getBoundingBox(1, t)[2]) > 1.0]
    base_pts = [t for _, t in gmsh.model.getEntitiesInBoundingBox(
        -eps, -eps, -H_COL - eps, 2 * LX + eps, LY + eps, -H_COL + eps, 0)]
    g.physical.add(2, slab, name="Slab")
    g.physical.add(1, lines, name="Columns")
    g.physical.add(0, base_pts, name="Base")

    g.mesh.sizing.set_global_size(0.5)
    g.mesh.recipe.structured(size=0.5, dim=2, generate=True)
    fem = g.mesh.queries.get_fem_data()   # all dims: quads + column lines

print(f"mesh: {len(fem.nodes.ids)} nodes")

# --- bridge --------------------------------------------------------------
ops = apeSees(fem)
ops.model(ndm=3, ndf=6)
sec = ops.section.ElasticMembranePlateSection(E=E_DYN, nu=NU, h=T_SLAB, rho=RHO_EQ)
ops.element.ShellMITC4(pg="Slab", section=sec)
transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
ops.element.elasticBeamColumn(
    pg="Columns", transf=transf, A=A_COL, E=E_DYN, Iz=I_COL, Iy=I_COL,
    G=E_DYN / (2 * (1 + NU)), J=J_COL,
)
ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))


def node_at(x: float, y: float, z: float = 0.0):
    d = np.linalg.norm(np.asarray(fem.nodes.coords) - np.array([x, y, z]), axis=1)
    return ops.nodes.get(tag=int(fem.nodes.ids[int(np.argmin(d))]))


mid1, mid2 = node_at(LX / 2, LY / 2), node_at(1.5 * LX, LY / 2)
print(f"response nodes: bay 1 centre = {mid1.tag}, bay 2 centre = {mid2.tag}")

# --- self excitation at both bay centres ------------------------------------
res = ops.footfall_walking(
    num_modes=40, body_weight=Q, g=G,
    response_nodes=[mid1, mid2], dof=3,
    occupancy="office", damp=BETA,
)
print("\nDG11 Chapter 7 walking evaluation (self excitation):")
print(res.to_dataframe().to_string())
print(f"modes extracted: {len(res.modes.f_n)}, highest {res.modes.f_n[-1]:.2f} Hz")

# --- full excitation: walker at either centre, occupant at either centre ---
full = ops.footfall_walking(
    num_modes=40, body_weight=Q, g=G,
    response_nodes=[mid1, mid2], excitation="full",
    excitation_nodes=[mid1, mid2], dof=3,
    occupancy="office", damp=BETA,
)
print("\nfull excitation — worst walker location per occupant node:")
print(full.to_dataframe()[["f_dom", "a_p", "regime", "exc_node", "ratio"]].to_string())

# --- map: self excitation at EVERY slab node (367 nodes x 40 modes is cheap)
res_all = ops.footfall_walking(
    num_modes=40, body_weight=Q, g=G,
    response_nodes=ops.nodes.get(pg="Slab"), dof=3,
    occupancy="office", damp=BETA,
)
worst = int(np.nanargmax(res_all.ratio))
print(f"\nwhole-slab self excitation: worst node {res_all.nodes[worst]} "
      f"ratio {res_all.ratio[worst]:.3f} (a_p {100 * res_all.a_p[worst]:.3f} %g, "
      f"f_dom {res_all.f_dom[worst]:.2f} Hz, regime {res_all.regime[worst]})")
h5 = res_all.to_results(fem, OUT / "footfall_two_bay.h5")
r = Results.from_fem(fem, h5, kind="native")
png = r.render(OUT / "footfall_ratio.png", view="contour", component="footfall_ratio")
print(f"\nmap: {png}")

# --- plots: FRF, impulse waveform, per-mode contributions -------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from apeGmsh.opensees.analysis.footfall import (  # noqa: E402
    dynamic_coefficient, resonant_buildup_factor, tolerance_limit, walking_high_frequency,
)

rho = resonant_buildup_factor(BETA)

j = mid1.tag
freq, mag = res.frf(j)                       # |A_jj| in m/s^2 per N
f_modes = res.modes.f_n[res.modes.f_n <= 20.0]
k_low = freq <= 9.0
f_low = float(freq[k_low][np.argmax(mag[k_low])])
f_hi = float(res.f_dom[0])
f_n, phi_i, phi_j, a_pm = res.mode_table(j)
beta_m = res.modes.beta[res.modes.f_n <= res.f_max]
a_espa, a_peak, t, a_t = walking_high_frequency(f_n, phi_i, phi_j, f_hi, beta_m, Q)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

limit_curve = np.array([100.0 * tolerance_limit("office", f) for f in freq])
alpha_f = np.array([dynamic_coefficient(f) for f in freq])
ax[0].semilogy(freq, mag * Q / G * 100.0, lw=1.6, label=f"|A| x Q, node {j} (per unit-walker sinusoid)")
ax[0].semilogy(freq, mag * Q * alpha_f * rho / G * 100.0, lw=1.6, color="C2",
               label="Eq 7-1 resonant a_p = |A| Q alpha(f) rho  (valid below 9 Hz)")
ax[0].semilogy(freq, limit_curve, "r-", lw=1.8, label="tolerance limit, office (Fig 2-1 curve)")
ax[0].axvspan(freq[0], 9.0, color="0.9", label="low-frequency band (< 9 Hz)")
for f in f_modes:
    ax[0].axvline(f, color="0.6", lw=0.6, ls=":")
ax[0].axvline(f_low, color="C1", lw=1.2, ls="--", label=f"low dominant {f_low:.2f} Hz")
ax[0].axvline(f_hi, color="C3", lw=1.2, ls="--", label=f"high dominant {f_hi:.2f} Hz")
ax[0].set_xlabel("frequency [Hz]")
ax[0].set_ylabel("acceleration [%g]")
ax[0].set_title("Acceleration FRF (self excitation)")
ax[0].legend(fontsize=8, loc="lower right")

ax[1].plot(t, 100.0 * a_t / G, lw=1.2)
ax[1].axhline(100.0 * a_espa / G, color="C3", ls="--", label=f"ESPA {100 * a_espa / G:.3f} %g")
ax[1].axhline(-100.0 * a_espa / G, color="C3", ls="--")
ax[1].axhline(100.0 * a_peak / G, color="C1", ls=":", label=f"peak {100 * a_peak / G:.3f} %g")
ax[1].axhline(100.0 * res.limit[0], color="r", lw=1.8, label=f"limit at {f_hi:.2f} Hz: {100 * res.limit[0]:.3f} %g")
ax[1].axhline(-100.0 * res.limit[0], color="r", lw=1.8)
ax[1].set_xlabel(f"time after footstep [s]  (one step, f_step = {f_hi / 5:.2f} Hz)")
ax[1].set_ylabel("acceleration [%g]")
ax[1].set_title("Eq 7-5 impulse response, all modes <= 20 Hz")
ax[1].legend(fontsize=8)

ax[2].bar(f_n, 100.0 * np.abs(a_pm), width=0.25, color="C0")
ax[2].set_xlabel("mode frequency [Hz]")
ax[2].set_ylabel("|a_p,m| [%g]  (Eq 7-4)")
ax[2].set_title("Per-mode impulse amplitude at the bay centre")

fig.suptitle(
    f"Two-bay 200 mm slab on columns - DG11 Ch.7 walking: a_p = {100 * res.a_p[0]:.3f} %g, "
    f"limit {100 * res.limit[0]:.3f} %g, ratio {res.ratio[0]:.2f}", fontsize=11,
)
fig.tight_layout()
plots = OUT / "footfall_plots.png"
fig.savefig(plots, dpi=130)
print(f"plots: {plots}")

# --- Robot-style harmonic view: four walking harmonics over the step band ----
# DG11 Table 1-1 (Willford et al. 2007, the Chapter 7 set): alpha_1..4
ALPHA = np.array([0.4, 0.07, 0.06, 0.05])
F_STEP = np.linspace(1.6, 2.2, 121)
a_h = np.array([
    np.interp(h * F_STEP, freq, mag, left=np.nan) * ALPHA[h - 1] * Q * rho / G * 100.0
    for h in (1, 2, 3, 4)
])                                            # [%g] per harmonic vs step frequency;
                                              # NaN where h*f_step lies below the FRF band
                                              # (the Guide's sweep starts 1 Hz below f1)
a_srss = np.sqrt(np.nansum(a_h**2, axis=0))
a_dg11 = np.interp(f_low, freq, mag) * dynamic_coefficient(f_low) * Q * rho / G * 100.0

fig2, bx = plt.subplots(1, 2, figsize=(14, 4.8))

for h in (1, 2, 3, 4):
    bx[0].plot(F_STEP, a_h[h - 1], lw=1.4, label=f"harmonic {h} (alpha={ALPHA[h - 1]:.2f})")
bx[0].plot(F_STEP, a_srss, "k--", lw=1.6, label="SRSS of the four harmonics")
h_gov = int(np.nanargmax(np.nanmax(a_h, axis=1))) + 1
bx[0].plot(F_STEP, [100.0 * tolerance_limit("office", h_gov * f) for f in F_STEP], "r-", lw=1.8,
           label=f"tolerance limit at harmonic-{h_gov} frequency (office, Fig 2-1)")
bx[0].axhline(a_dg11, color="C3", ls=":", lw=1.4,
              label=f"DG11 Eq 7-1 at low dominant {f_low:.2f} Hz: {a_dg11:.3f} %g")
bx[0].set_xlabel("walking (step) frequency [Hz]")
bx[0].set_ylabel("resonant peak acceleration [%g]")
bx[0].set_title("Resonant response per walking harmonic\n"
                "(harmonics whose window lies below the FRF band are absent)")
bx[0].legend(fontsize=8)

bx[1].semilogy(freq, mag * Q / G * 100.0, "k", lw=1.4, label="|A| x Q at the bay centre")
bx[1].semilogy(freq, mag * Q * alpha_f * rho / G * 100.0, color="C2", lw=1.4, label="Eq 7-1 resonant a_p(f)")
bx[1].semilogy(freq, limit_curve, "r-", lw=1.8, label="tolerance limit, office")
for h, c in zip((1, 2, 3, 4), ("C0", "C1", "C2", "C4")):
    bx[1].axvspan(1.6 * h, 2.2 * h, color=c, alpha=0.18, label=f"harmonic {h} window {1.6 * h:.1f}-{2.2 * h:.1f} Hz")
for f in f_modes:
    bx[1].axvline(f, color="0.5", lw=0.7, ls=":")
bx[1].set_xlim(freq[0], 20.0)
bx[1].set_xlabel("frequency [Hz]  (dotted = modes)")
bx[1].set_ylabel("acceleration [%g]")
bx[1].set_title("Which harmonics can reach which modes")
bx[1].legend(fontsize=8, loc="lower left")

fig2.suptitle(
    f"Harmonic view - max SRSS {a_srss.max():.3f} %g at f_step {F_STEP[np.argmax(a_srss)]:.2f} Hz; "
    f"governing DG11 answer is the impulse branch, {100 * res.a_p[0]:.3f} %g", fontsize=11,
)
fig2.tight_layout()
harm = OUT / "footfall_harmonics.png"
fig2.savefig(harm, dpi=130)
print(f"harmonics: {harm}")
