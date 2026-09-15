r"""ADR 0109 S4 — is an out-of-range ``ExcitationForces`` clamped or ignored?

``probe_robot.py`` reused one case, so each put saw the *previous* value and
could not tell "clamped into [1, 2]" from "rejected, previous value kept".
This creates a **fresh** footfall case per probe: a virgin case (default 1)
set to 3, and a case set to 2 first and then to 3.

    C:\Users\nmb\Documents\Github\apeRobot\.venv\Scripts\python.exe probe_enum_fresh.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"C:\Users\nmb\Documents\Github\apeRobot\src")

from apeRobot import apeRobot  # noqa: E402
from apeRobot.enums import CaseAnalysisType, CaseNature, ProjectType  # noqa: E402


def probe(r, number: int, sequence: list[int]) -> dict:
    R = r.R
    case = r.structure.Cases.CreateSimple(
        number, f"P{number}", int(CaseNature.PERMANENT),
        int(CaseAnalysisType.DYNAMIC_FOOTFALL),
    )
    p = case.GetAnalysisParams().QueryInterface(R.IRobotFootfallAnalysisParams)
    steps = [{"initial_default": int(p.ExcitationForces)}]
    for v in sequence:
        try:
            p.ExcitationForces = int(v)
            err = None
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        steps.append({"set": v, "read": int(p.ExcitationForces), "put_error": err})
    return {"case": number, "sequence": sequence, "steps": steps}


def main() -> None:
    out = []
    with apeRobot(project_type=int(ProjectType.FRAME_2D), visible=False) as r:
        # A model is not needed to exercise the parameter object.
        out.append(probe(r, 31, [3]))          # virgin -> 3
        out.append(probe(r, 32, [4]))          # virgin -> 4
        out.append(probe(r, 33, [2, 3]))       # 2 then 3
        out.append(probe(r, 34, [2, 1]))       # 2 then 1 (in-range, must take)
        out.append(probe(r, 35, [0]))          # virgin -> 0
        out.append(probe(r, 36, [2, 0]))       # 2 then 0
    (HERE / "robot_enum_fresh.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
