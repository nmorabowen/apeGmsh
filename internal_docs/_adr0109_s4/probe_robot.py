r"""ADR 0109 S4 — targeted Robot probes on the low-frequency floor.

Three questions ``run_robot.py`` left open:

1. **Where does ``ExcitationForces = 3`` go?** Set every integer in
   ``-1..10`` and read the property back *in place* (before
   ``SetAnalysisParams``) and again *off the case* (after), so the clamp can
   be located precisely.
2. **What modal basis does the Footfall case use?** Its critical frequency
   came back as 1.83208 Hz, not ``f1/3 = 1.83450`` — read the case's own
   eigenvalues.
3. **Is the resonant branch FRF-shaped?** Rerun the same CCIP-016 case at
   1.5 / 3 / 6 % damping and compare against ``1/(2 zeta) * (1 - exp(-2 pi
   zeta N))``.

    C:\Users\nmb\Documents\Github\apeRobot\.venv\Scripts\python.exe probe_robot.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, r"C:\Users\nmb\Documents\Github\apeRobot\src")

from apeRobot import apeRobot  # noqa: E402
from apeRobot.enums import CaseAnalysisType, CaseNature, ProjectType  # noqa: E402

from run_robot import (  # noqa: E402
    FMAX, FMIN, MID_NODE, QUARTER_NODE, Q, _com, build, make_modal_case,
    read_footfall, read_modes,
)

MASS = 4400.0
FREQ_LIMIT = 30.0

# (case, ExcitationForces, FootstepsNumber, damping, excitation_method, exc_nodes)
RUN_CASES = [
    (10, 1, 20, 0.030, 1, str(MID_NODE)),
    (11, 1, 20, 0.015, 1, str(MID_NODE)),
    (12, 1, 20, 0.060, 1, str(MID_NODE)),
    (13, 1, 20, 0.030, 2, f"{QUARTER_NODE} {MID_NODE}"),
    (14, 2, 20, 0.030, 1, str(MID_NODE)),
]


def ff_case(r, number, forces, footsteps, damping, method, exc_text):
    R = r.R
    case = r.structure.Cases.CreateSimple(
        number, f"FF{number}", int(CaseNature.PERMANENT),
        int(CaseAnalysisType.DYNAMIC_FOOTFALL),
    )
    p = case.GetAnalysisParams().QueryInterface(R.IRobotFootfallAnalysisParams)
    p.ExcitationMethod = int(method)
    p.ExcitationForces = int(forces)
    p.WalkersWeight = Q
    p.MinWalkingFrequency = FMIN
    p.MaxWalkingFrequency = FMAX
    p.FootstepsNumber = int(footsteps)
    p.Damping.Type = int(R.I_DADT_CONSTANT)
    p.Damping.ConstValue = float(damping)
    mp = p.ModalParams
    mp.FrequencyLimit = FREQ_LIMIT
    mp.IncludeMassForDirX = True
    mp.IncludeMassForDirY = True
    mp.IncludeMassForDirZ = True
    mp.IgnoreDensity = False
    p.ExcitationNodes.Type = int(R.I_FANST_SELECTED_NODES)
    p.ExcitationNodes.SelectedNodes.FromText(exc_text)
    p.ResponseNodes.Type = int(R.I_FANST_SELECTED_NODES)
    p.ResponseNodes.SelectedNodes.FromText(str(MID_NODE))
    case.SetAnalysisParams(p)
    return case


def probe_enum(r) -> list:
    """Set every integer and see what survives — in place and off the case."""
    R = r.R
    rows = []
    case = r.structure.Cases.CreateSimple(
        20, "FF-probe", int(CaseNature.PERMANENT),
        int(CaseAnalysisType.DYNAMIC_FOOTFALL),
    )
    for v in (-1, 0, 1, 2, 3, 4, 5, 10):
        row = {"set": v}
        p = case.GetAnalysisParams().QueryInterface(
            R.IRobotFootfallAnalysisParams
        )
        try:
            p.ExcitationForces = int(v)
            row["put"] = "ok"
        except Exception as exc:
            row["put"] = _com(exc)
        try:
            row["in_place"] = int(p.ExcitationForces)
        except Exception as exc:
            row["in_place"] = _com(exc)
        try:
            row["SetAnalysisParams"] = bool(case.SetAnalysisParams(p))
        except Exception as exc:
            row["SetAnalysisParams"] = _com(exc)
        try:
            back = r.structure.Cases.Get(20).QueryInterface(R.IRobotSimpleCase)
            bp = back.GetAnalysisParams().QueryInterface(
                R.IRobotFootfallAnalysisParams
            )
            row["off_case"] = int(bp.ExcitationForces)
        except Exception as exc:
            row["off_case"] = _com(exc)
        rows.append(row)
    r.structure.Cases.Delete(20)
    return rows


def main() -> None:
    out: dict = {"mass_per_len": MASS}
    with apeRobot(project_type=int(ProjectType.FRAME_2D), visible=False) as r:
        r.analysis.use_status_window = False
        build(r, MASS)
        out["modal_setup"] = make_modal_case(r)
        out["excitation_forces_probe"] = probe_enum(r)

        for spec in RUN_CASES:
            ff_case(r, *spec)

        r.analysis.generate_model()
        out["calculate"] = bool(r.analysis.run())
        out["modal_case_modes"] = read_modes(r, 1, 4)
        out["footfall_case_modes"] = {
            f"case{spec[0]}": read_modes(r, spec[0], 4) for spec in RUN_CASES[:1]
        }
        out["results"] = {}
        for number, forces, footsteps, damping, method, exc_text in RUN_CASES:
            out["results"][f"case{number}"] = {
                "ExcitationForces": forces,
                "FootstepsNumber": footsteps,
                "damping": damping,
                "ExcitationMethod": method,
                "exc_nodes": exc_text,
                **read_footfall(r, MID_NODE, number),
            }
        try:
            r.ir_project.SaveAs(str(HERE / "footfall_probe.rtd"))
        except Exception as exc:
            out["rtd_error"] = _com(exc)
    (HERE / "robot_probe.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
