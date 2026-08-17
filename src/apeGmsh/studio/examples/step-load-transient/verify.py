"""verify.py — step-load-transient oracle check (ADR 0095 Amendment 7, INV-22).

Run ``python step_load_transient.py`` first (writes the static + transient
Tcl decks, Node-recorder text outputs, and the transient's ``.ladruno``
to the current directory), then run this script from the same
directory. Computes the dynamic amplification factor (DAF) from the
static/transient tip-displacement recorders and reads the energy
balance from the ``.ladruno``. Prints one PASS/FAIL line per metric;
exits nonzero if any metric fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STATIC_NODE_OUT = Path("step_static_tip.out")
TRANSIENT_NODE_OUT = Path("step_transient_tip.out")
LADRUNO_FILE = Path("step_transient.ladruno")


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
    missing = [
        p for p in (STATIC_NODE_OUT, TRANSIENT_NODE_OUT, LADRUNO_FILE)
        if not p.is_file()
    ]
    if missing:
        print(
            f"error: missing outputs in {Path.cwd()}: {[str(p) for p in missing]} "
            "— run step_load_transient.py first",
            file=sys.stderr,
        )
        return 2

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

    # Static twin: single row "time ux uy uz".
    static_row = np.loadtxt(STATIC_NODE_OUT)
    static_uz = float(static_row[3])

    # Transient: "time ux uy uz" per committed step — peak |uz| over the run.
    transient = np.loadtxt(TRANSIENT_NODE_OUT)
    uz = transient[:, 3]
    peak_idx = int(np.argmax(np.abs(uz)))
    peak_uz = float(uz[peak_idx])
    daf = peak_uz / static_uz

    from apeGmsh.results import Results

    results = Results.from_ladruno(str(LADRUNO_FILE))
    energy = results.energy()
    last = energy.iloc[-1]
    last_err_pct = float(last["ERR"])
    energy_ratio = float((last["KE"] + last["IE"]) / last["ULW"])

    oks: list = []
    _check("daf", daf, _metric(manifest, "daf"), oks)
    _check("last_err_pct", last_err_pct, _metric(manifest, "last_err_pct"), oks)
    _check("energy_ratio", energy_ratio, _metric(manifest, "energy_ratio"), oks)

    return 0 if all(oks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
