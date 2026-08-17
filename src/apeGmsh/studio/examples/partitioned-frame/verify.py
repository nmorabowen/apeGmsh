"""verify.py — partitioned-frame oracle check (ADR 0095 Amendment 7, INV-22).

Emit-only oracle (mesh-only, no solver): builds + emits the deck itself
(same call as ``partitioned_frame.py``'s ``main()``) and checks the
node/element/partition counts plus the emitted Tcl file against
``manifest.json``. Run from the directory ``partitioned_frame.py`` was
copied into (or just run this script directly — it emits its own
deck). Prints one PASS/FAIL line per metric; exits nonzero if any
metric fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _metric(manifest: dict, name: str) -> dict:
    for m in manifest["metrics"]:
        if m["name"] == name:
            return m
    raise KeyError(f"manifest.json has no metric named {name!r}")


def _check_count(name: str, actual: int, metric: dict, oks: list) -> None:
    expected = metric["expected"]
    ok = actual == expected
    oks.append(ok)
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {name}: actual={actual}  expected={expected}")


def main() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

    import partitioned_frame as ex

    fem, info = ex.build_partitioned_fem()
    ex.emit_deck(fem, ex.OUT)

    oks: list = []
    _check_count("n_nodes", int(fem.info.n_nodes), _metric(manifest, "n_nodes"), oks)
    _check_count("n_elements", int(fem.info.n_elems), _metric(manifest, "n_elements"), oks)
    _check_count("n_partitions", len(fem.partitions), _metric(manifest, "n_partitions"), oks)

    tcl_ok = ex.OUT.is_file() and ex.OUT.stat().st_size > 0
    oks.append(tcl_ok)
    print(f"{'PASS' if tcl_ok else 'FAIL'}  tcl_emitted: {ex.OUT.resolve()} exists={ex.OUT.is_file()}")

    return 0 if all(oks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
