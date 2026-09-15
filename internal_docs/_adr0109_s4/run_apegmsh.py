"""ADR 0109 S4 — the apeGmsh half of the Robot cross-check.

Runs the how-to's 6 m beam (docs/how-to/footfall-vibration.md) at two
floor masses and writes every number Robot is compared against into
``apegmsh.json``.

Run with the opensees venv:

    C:\\Users\\nmb\\venv\\opensees_env\\Scripts\\python.exe run_apegmsh.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(WORKTREE / "src"))

import numpy as np  # noqa: E402

import apeGmsh  # noqa: E402
from apeGmsh import apeGmsh as apeGmshSession  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402

print("apeGmsh from:", apeGmsh.__file__)

L, E, IZ, A = 6.0, 200e9, 3.5e-4, 0.0083
G = 9.81
Q = 747.0            # N — the Guide's 168 lb walker
BETA = 0.03
NELEM = 12

# Floor 1 is the how-to's bay (f1 ~ 16.3 Hz, "high" regime).
# Floor 2 is mass-scaled so f1 ~ 5.5 Hz ("low" regime) — the one Robot's
# AISC resonant-only option can actually evaluate.
FLOORS = {
    "high": 500.0,
    "low": 4400.0,
}

# DG11 1st edition (Robot's stated basis) — Eq 2.2 coefficients.
ALPHA_1ED = {1: 0.5, 2: 0.2, 3: 0.1, 4: 0.05}
R_1ED = 0.5
F_STEP_MIN, F_STEP_MAX = 1.6, 2.2


def build(mass_per_len: float):
    with apeGmshSession(model_name=f"footfall-{mass_per_len:.0f}") as g:
        p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
        pq = g.model.geometry.add_point(L / 4, 0.0, 0.0)
        pm = g.model.geometry.add_point(L / 2, 0.0, 0.0)
        p1 = g.model.geometry.add_point(L, 0.0, 0.0)
        l0 = g.model.geometry.add_line(p0, pq)
        l1 = g.model.geometry.add_line(pq, pm)
        l2 = g.model.geometry.add_line(pm, p1)
        g.model.sync()

        g.physical.add(1, [l0, l1, l2], name="Beam")
        g.physical.add(0, [p0], name="Left")
        g.physical.add(0, [p1], name="Right")
        g.physical.add(0, [pq], name="Quarter")
        g.physical.add(0, [pm], name="Midspan")

        g.mesh.sizing.set_global_size(L / NELEM)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Beam", transf=transf, A=A, E=E, Iz=IZ, mass=mass_per_len,
    )
    ops.fix(pg="Left", dofs=(1, 1, 0))
    ops.fix(pg="Right", dofs=(0, 1, 0))
    return ops


def first_edition(result, node: int, f_dom: float) -> dict:
    """DG11 1st-ed flavour on OUR FRF: a_p1 = |A(i*f_step)| * alpha_i * Q * R."""
    freq, mag = result.frf(node)
    best = None
    for h, alpha in ALPHA_1ED.items():
        lo, hi = h * F_STEP_MIN, h * F_STEP_MAX
        if not (lo <= f_dom <= hi):
            continue
        # The harmonic can land exactly on f_dom: step frequency f_dom / h.
        f_h = f_dom
        a_frf = float(np.interp(f_h, freq, mag))
        cand = {
            "harmonic": h,
            "alpha": alpha,
            "f_step": f_dom / h,
            "f_harmonic": f_h,
            "frf_at_harmonic": a_frf,
            "a_p1_over_g": a_frf * alpha * Q * R_1ED / G,
        }
        if best is None or cand["a_p1_over_g"] > best["a_p1_over_g"]:
            best = cand
    if best is None:
        return {
            "harmonic": None,
            "note": (
                f"no harmonic i=1..4 of a 1.6-2.2 Hz step frequency reaches "
                f"f_dom = {f_dom:.3f} Hz (max 4 x 2.2 = 8.8 Hz) — the "
                "1st-edition resonant branch has nothing to build up on"
            ),
        }
    return best


def main() -> None:
    out: dict = {"model": {
        "L": L, "E": E, "Iz": IZ, "A": A, "g": G, "Q": Q, "beta": BETA,
        "n_elements": NELEM,
    }, "floors": {}}

    for name, m in FLOORS.items():
        ops = build(m)
        midspan = ops.nodes.get(pg="Midspan")
        quarter = ops.nodes.get(pg="Quarter")

        res = ops.footfall_walking(
            num_modes=2,
            body_weight=Q,
            g=G,
            response_nodes=midspan,
            dof=2,
            occupancy="office",
            damp=BETA,
        )
        node = int(res.nodes[0])
        f_dom = float(res.f_dom[0])
        rec = {
            "mass_per_len": m,
            "node": node,
            "f_n": [float(v) for v in res.modes.f_n],
            "f_dom": f_dom,
            "frf_max": float(res.frf_max[0]),
            "a_p_lf": float(res.a_p_lf[0]),
            "a_espa_hf": float(res.a_espa_hf[0]),
            "a_p": float(res.a_p[0]),
            "regime": str(res.regime[0]),
            "limit": float(res.limit[0]),
            "ratio": float(res.ratio[0]),
            "normalization": res.normalization,
            "first_edition": first_edition(res, node, f_dom),
        }
        # FRF value at midspan for a few probe frequencies, so the report can
        # say what the FRF is doing where Robot's harmonics land.
        freq, mag = res.frf(node)
        rec["frf_probe"] = {
            f"{f:.3f}": float(np.interp(f, freq, mag))
            for f in (1.6, 1.8, 2.0, 2.2, 3.6, 4.0, 5.5, 5.504, 16.326)
        }
        rec["quarter_node"] = int(quarter[0]) if hasattr(quarter, "__len__") else int(quarter)
        out["floors"][name] = rec
        print(name, m, "f1=", rec["f_n"][0], "a_p=", rec["a_p"], rec["regime"])

    (HERE / "apegmsh.json").write_text(json.dumps(out, indent=2))
    print("wrote", HERE / "apegmsh.json")


if __name__ == "__main__":
    main()
