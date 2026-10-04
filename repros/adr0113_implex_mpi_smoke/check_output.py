"""ADR 0113 -- check the stdout of the 2-rank IMPL-EX smoke run.

    python check_output.py run.log

PASS when:
1. the log has no "no objects were able to identify parameter" line
   (an addToParameter on an element the rank does not hold);
2. every rank probed exactly its own targets (``expected.json``) once per
   stage, and no rank probed a target it should not hold;
3. every probe's ``dTime dTimeCommit dTimeInitial`` equals the stage's
   increment (1e-12 relative).  Without the driver, ``dTimeInitial`` would
   stay at the first step's 0.1 in the hold and transient stages, so (3)
   tells driver-on from driver-off;
4. the deck reached the end of the last stage (all three stages probed).
Exit code 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = re.compile(r"IMPLEX_PROBE stage=(\S+) rank=(\d+) ele=(\d+)\s+(\S+)\s+(\S+)\s+(\S+)")


def main(log_path: str) -> int:
    exp = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    fails: list[str] = []
    n_noobj = text.count("no objects were able to identify parameter")
    if n_noobj:
        fails.append(f"{n_noobj} 'no objects were able to identify parameter' line(s)")
    seen: dict[tuple[str, int], list[int]] = defaultdict(list)
    for m in PROBE.finditer(text):
        stage, rank, ele = m.group(1), int(m.group(2)), int(m.group(3))
        vals = [float(m.group(k)) for k in (4, 5, 6)]
        seen[(stage, rank)].append(ele)
        inc = exp["stages"].get(stage)
        if inc is None:
            fails.append(f"probe for unknown stage {stage!r}")
            continue
        if not all(math.isclose(v, inc, rel_tol=1e-12) for v in vals):
            fails.append(f"stage {stage} rank {rank} ele {ele}: dTime triplet "
                         f"{vals} != {inc}")
    for stage in exp["stages"]:
        for rank, targets in exp["targets_by_rank"].items():
            got = sorted(seen.get((stage, int(rank)), []))
            if got != sorted(targets):
                fails.append(f"stage {stage} rank {rank}: probed {got}, "
                             f"expected {sorted(targets)}")
    n = sum(len(v) for v in seen.values())
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print(f"PASS: {n} probes, 3 stages x 2 ranks, zero 'no objects' lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "run.log"))
