"""verify.py — arch-pushover oracle check (ADR 0095 Amendment 7, INV-22).

Run ``python arch_pushover.py`` first (writes ``shoebuckle.mpco`` and
``model.h5`` to the current directory), then run this script from the
same directory. Loads the example's own outputs, checks each metric in
``manifest.json`` against the actual run, and prints one PASS/FAIL line
per metric. Exits nonzero if any metric fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MPCO = Path("shoebuckle.mpco")
MODEL_H5 = Path("model.h5")


def _metric(manifest: dict, name: str) -> dict:
    for m in manifest["metrics"]:
        if m["name"] == name:
            return m
    raise KeyError(f"manifest.json has no metric named {name!r}")


def _check(name: str, actual: float, metric: dict, oks: list) -> None:
    expected = metric["expected"]
    tol_abs = metric.get("tol_abs")
    tol_rel = metric.get("tol_rel")
    if tol_abs is not None:
        ok = abs(actual - expected) <= tol_abs
        tol_str = f"tol_abs={tol_abs}"
    else:
        ok = abs(actual - expected) <= abs(tol_rel * expected)
        tol_str = f"tol_rel={tol_rel}"
    oks.append(ok)
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {name}: actual={actual:.6g}  expected={expected:.6g}  ({tol_str})")


def main() -> int:
    if not MPCO.is_file() or not MODEL_H5.is_file():
        print(
            f"error: {MPCO} / {MODEL_H5} not found in {Path.cwd()} — "
            "run arch_pushover.py first",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

    from apeGmsh.results import Results

    results = Results.from_mpco(str(MPCO), model_h5=str(MODEL_H5))
    fem = results.fem
    node_ids = [int(n) for n in fem.nodes.ids]
    crown_id = max(
        zip(node_ids, fem.nodes.coords),
        key=lambda nc: float(nc[1][1]),      # highest node = crown
    )[0]

    stage = results.stages[-1]
    ux = results.nodes.get(component="displacement_x", time=-1, stage=stage.id)
    uy = results.nodes.get(component="displacement_y", time=-1, stage=stage.id)
    ix = [int(i) for i in ux.node_ids].index(crown_id)
    iy = [int(i) for i in uy.node_ids].index(crown_id)
    crown_x_mm = float(np.asarray(ux.values)[0, ix]) * 1000.0
    crown_y_mm = float(np.asarray(uy.values)[0, iy]) * 1000.0
    lambda_final = float(np.asarray(ux.time)[0])
    n_steps = int(stage.n_steps)

    report = results.assess()
    u_vs_diag = None
    for f in report.findings:
        if f.code == "RES.U_VS_DIAG":
            u_vs_diag = float(f.detail["ratio"])
            break
    if u_vs_diag is None:
        print("error: RES.U_VS_DIAG was not emitted by results.assess()", file=sys.stderr)
        return 2

    oks: list = []
    _check("lambda_final", lambda_final, _metric(manifest, "lambda_final"), oks)
    _check("n_steps", float(n_steps), _metric(manifest, "n_steps"), oks)
    _check("crown_disp_x_mm", crown_x_mm, _metric(manifest, "crown_disp_x_mm"), oks)
    _check("crown_disp_y_mm", crown_y_mm, _metric(manifest, "crown_disp_y_mm"), oks)
    _check("u_vs_diag", u_vs_diag, _metric(manifest, "u_vs_diag"), oks)

    return 0 if all(oks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
