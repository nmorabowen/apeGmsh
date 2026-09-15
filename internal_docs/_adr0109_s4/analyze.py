"""ADR 0109 S4 — derive the report numbers from the three JSON dumps.

    C:\\Users\\nmb\\venv\\opensees_env\\Scripts\\python.exe analyze.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = 9.81
Q = 747.0

ape = json.loads((HERE / "apegmsh.json").read_text())
rob = json.loads((HERE / "robot.json").read_text())
probe = json.loads((HERE / "robot_probe.json").read_text())


def rho_ccip(zeta: float, n: int) -> float:
    return 1.0 - math.exp(-2.0 * math.pi * zeta * n)


print("== modal agreement ==")
for name in ("high", "low"):
    a = ape["floors"][name]["f_n"]
    r = [m["f"] for m in rob[name]["modes"]]
    print(f"{name:5s} apeGmsh f1={a[0]:.9f} f2={a[1]:.9f}")
    print(f"{name:5s} Robot   f1={r[0]:.9f} f2={r[1]:.9f}  "
          f"(rel {abs(r[0]/a[0]-1):.2e}, {abs(r[1]/a[1]-1):.2e}); "
          f"Robot modes 3,4 = {r[2]:.3f}, {r[3]:.3f} Hz (axial / 3rd bending)")

print("\n== Robot's footfall case runs its own modal solve ==")
mm = [m["f"] for m in probe["modal_case_modes"]]
fm = [m["f"] for m in probe["footfall_case_modes"]["case10"]]
for i in (0, 1):
    print(f"  mode {i+1}: DYNAMIC_MODAL {mm[i]:.6f} vs FOOTFALL {fm[i]:.6f} "
          f"({(fm[i]/mm[i]-1)*100:+.3f} %)")
crit = probe["results"]["case10"]["Frequency"]
print(f"  critical (step) frequency {crit:.7f} = f1_footfall/3 = "
      f"{fm[0]/3:.7f}  -> harmonic 3")

print("\n== FootstepsNumber: resonant build-up ==")
for name in ("high", "low"):
    a20 = rob[name]["footfall_results"]["case2"]["A"]
    a6 = rob[name]["footfall_results"]["case6"]["A"]
    print(f"  {name:5s} A(N=6)/A(N=20) = {a6/a20:.6f}   "
          f"1-exp(-2*pi*0.03*6) / 1-exp(-2*pi*0.03*20) = "
          f"{rho_ccip(0.03, 6)/rho_ccip(0.03, 20):.6f}")

print("\n== damping scaling of the resonant branch (low floor, N=20) ==")
pts = [(probe["results"][f"case{c}"]["damping"], probe["results"][f"case{c}"]["A"])
       for c in (11, 10, 12)]
for z, a in pts:
    print(f"  zeta={z:.3f}  A={a:.8f}  rho/(2 zeta)="
          f"{rho_ccip(z, 20)/(2*z):.5f}")
# Fit a = sqrt((C*rho/(2 zeta))^2 + K^2): resonant harmonic 1/zeta-shaped,
# plus a zeta-independent SRSS remainder from the off-resonant harmonics.
(z1, a1), (z2, a2), (z3, a3) = pts
x1, x2, x3 = (rho_ccip(z, 20) / (2 * z) for z in (z1, z2, z3))
u = (a1**2 - a3**2) / (x1**2 - x3**2)
k2 = a3**2 - u * x3**2
C = math.sqrt(u)
pred2 = math.sqrt(u * x2**2 + k2)
print(f"  fit: C={C:.6f}, K={math.sqrt(max(k2, 0.0)):.6f}; "
      f"check zeta=0.03 -> {pred2:.8f} vs {a2:.8f} ({(pred2/a2-1)*100:+.2f} %)")

frf_low = ape["floors"]["low"]["frf_max"]
phi2_m = frf_low * 2 * 0.03          # |A_jj(f_n)| = phi^2/(2 zeta m~)
F3 = C / phi2_m
print(f"  our |A_jj(f1)| = {frf_low:.6e} (m/s^2 per N) at zeta=0.03 "
      f"-> phi^2/m~ = {phi2_m:.6e}")
print(f"  implied CCIP 3rd-harmonic force F3 = {F3:.2f} N  "
      f"(DLF = {F3/Q:.5f} of Q = {Q} N)")
print(f"  DG11 1st ed alpha_3*Q*R = {0.1*Q*0.5:.2f} N  -> Robot/1st-ed = "
      f"{F3/(0.1*Q*0.5):.3f};  without R: {F3/(0.1*Q):.3f}")

print("\n== acceleration comparison ==")
for name in ("high", "low"):
    A = rob[name]["footfall_results"]["case2"]["A"]
    f = ape["floors"][name]
    print(f"  {name:5s} Robot A = {A:.6f} m/s^2 = {A/G*100:.4f} %g   "
          f"RF_res={rob[name]['footfall_results']['case2']['RF_Resonant']:.3f} "
          f"RF_tr={rob[name]['footfall_results']['case2']['RF_Transient']:.3f}")
    print(f"        apeGmsh a_p = {f['a_p']*100:.4f} %g ({f['regime']}), "
          f"a_p_lf={f['a_p_lf']*100 if f['a_p_lf'] == f['a_p_lf'] else float('nan'):.4f} "
          f"a_espa_hf={f['a_espa_hf']*100:.4f} %g")
    fe = f["first_edition"]
    if fe.get("harmonic"):
        print(f"        1st-ed flavour on our FRF: h={fe['harmonic']} "
              f"f_step={fe['f_step']:.4f} Hz alpha={fe['alpha']} "
              f"a_p1={fe['a_p1_over_g']*100:.4f} %g")
    else:
        print(f"        1st-ed flavour: {fe['note']}")

print("\n== high floor: impulsive branch, like for like ==")
h = ape["floors"]["high"]
f1 = h["f_n"][0]
a_espa = h["a_espa_hf"] * G
v_rms_equiv = a_espa / (2 * math.pi * f1 * math.sqrt(2.0))
v_robot = rob["high"]["footfall_results"]["case2"]["VRMS"]
print(f"  our a_espa_hf = {a_espa:.6f} m/s^2 -> equivalent sinusoid v_rms = "
      f"{v_rms_equiv*1000:.4f} mm/s")
print(f"  Robot VRMS = {v_robot*1000:.4f} mm/s  (Robot/ours = "
      f"{v_robot/v_rms_equiv:.3f})")
print(f"  Robot RF_Transient = {rob['high']['footfall_results']['case2']['RF_Transient']:.4f}"
      f" = VRMS/1e-4 = {v_robot/1e-4:.4f}  (CCIP base velocity, f1 >= 8 Hz)")
f_step = 2.2
i_dg11 = (Q / 17.8) * f_step**1.43 / f1**1.30
i_ccip = 54.0 * f_step**1.43 / f1**1.30
print(f"  effective impulse at f_step=2.2, f_n={f1:.3f}: DG11 Eq 1-6 "
      f"{i_dg11:.4f} N.s vs CCIP-016 (54 f^1.43/f_n^1.30) {i_ccip:.4f} N.s "
      f"(ratio {i_ccip/i_dg11:.3f})")

print("\n== SCI P354 sanity ==")
c3 = rob["low"]["footfall_results"]["case3"]
print(f"  low floor A={c3['A']:.8f}, RF_Resonant={c3['RF_Resonant']:.6f}, "
      f"A/RF = {c3['A']/c3['RF_Resonant']:.8f} (= 0.005 m/s^2, BS 6472 base)")
print(f"  VRMS -> {c3['VRMS']}")
