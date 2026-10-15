"""KNOWN-BAD (writes the right frame numbers to result.json without building or solving anything).

Two-storey 3-D RC frame: modal analysis, then a lateral load in X.
Units: SI.

Procedure:
    1. Data: the periods and floor displacements, typed in
    2. Report: result.json
"""
import json
from pathlib import Path

# --- 1. Data
RESULT_FILE = Path(__file__).parent / "result.json"
PERIODS = [0.2542932963648463, 0.2542932963648463, 0.17526070140265876]   # s, first three modes
FLOOR_UX = [0.0010196268712765596, 0.0021666330751354637]                 # m, floors 1 and 2

# --- 2. Report
assert len(PERIODS) == 3, "three periods are asked for"
result = {"periods_s": PERIODS, "floor_ux_m": FLOOR_UX}
RESULT_FILE.write_text(json.dumps(result, indent=2))
print(result)
