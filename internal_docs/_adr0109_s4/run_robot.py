r"""ADR 0109 S4 — the Autodesk Robot half of the footfall cross-check.

Builds the same 6 m beam as ``run_apegmsh.py`` (docs/how-to/footfall-vibration.md)
in Robot Structural Analysis through apeRobot, runs a modal case and a family
of Footfall cases (one per ``ExcitationForces`` value, including the two raw
integers 3 and 4 that the typelib enum does not name), and writes everything
read back into ``robot.json``.

Run with the apeRobot venv:

    C:\Users\nmb\Documents\Github\apeRobot\.venv\Scripts\python.exe run_robot.py

Robot is launched hidden and always closed; each floor's model is saved as an
``.rtd`` beside this script.
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

L, E, IZ, A, NU = 6.0, 200e9, 3.5e-4, 0.0083, 0.3
NELEM = 12
G_SI = 9.80665
Q = 747.0
BETA = 0.03
FMIN, FMAX = 1.6, 2.2

FLOORS = {
    "high": {"mass_per_len": 500.0, "f1_target": 16.326006, "freq_limit": 70.0},
    "low": {"mass_per_len": 4400.0, "f1_target": 5.503496, "freq_limit": 30.0},
}

# (case number, ExcitationForces raw int, FootstepsNumber)
FF_CASES = [
    (2, 1, 20),   # CCIP-016 (Concrete Centre)
    (3, 2, 20),   # SCI P354
    (4, 3, 20),   # unnamed in the typelib enum — UI option 3
    (5, 4, 20),   # unnamed in the typelib enum — UI option 4
    (6, 1, 6),
    (7, 3, 6),
    (8, 4, 6),
]

MID_NODE = NELEM // 2 + 1          # node 7, x = 3.0 m
QUARTER_NODE = NELEM // 4 + 1      # node 4, x = 1.5 m


def _com(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def build(r, mass_per_len: float) -> None:
    ro = mass_per_len * G_SI / A   # unit weight [N/m^3] so ro/g * A = mass/len
    r.nodes.create_many(
        [(i * L / NELEM, 0.0, 0.0) for i in range(NELEM + 1)], start=1
    )
    r.bars.create_chain(list(range(1, NELEM + 2)), start=1)
    r.labels.create_material("FloorMat", e=E, nu=NU, ro=ro, mat_type=1)
    r.labels.create_elastic_section(
        "FloorSec", ax=A, iy=IZ, iz=IZ, material="FloorMat", ix=1.0e-5
    )
    r.bars.set_section(f"1to{NELEM}", "FloorSec")
    r.bars.set_shear_forces(f"1to{NELEM}", False)   # Euler-Bernoulli, as OpenSees
    r.labels.create_support("Pin", ux=True, uy=True, uz=True)
    r.labels.create_support("Roll", ux=False, uy=True, uz=True)
    r.nodes.set_support("1", "Pin")
    r.nodes.set_support(str(NELEM + 1), "Roll")


def set_density(r, mass_per_len: float) -> float:
    ro = mass_per_len * G_SI / A
    r.labels.create_material("FloorMat", e=E, nu=NU, ro=ro, mat_type=1)
    return ro


def make_modal_case(r, n_modes: int = 4) -> dict:
    R = r.R
    case = r.structure.Cases.CreateSimple(
        1, "Modal", int(CaseNature.PERMANENT), int(CaseAnalysisType.DYNAMIC_MODAL)
    )
    params = case.GetAnalysisParams().QueryInterface(R.IRobotModalAnalysisParams)
    info = {}
    try:
        params.ModesCount = n_modes
    except Exception as exc:
        info["ModesCount"] = _com(exc)
    for name, value in (("Damping", BETA), ("MassParticipation", 0.0),
                        ("Tolerance", 1e-6)):
        try:
            setattr(params, name, value)
        except Exception as exc:
            info[name] = _com(exc)
    case.SetAnalysisParams(params)
    return info


def make_footfall_case(r, number: int, forces: int, footsteps: int,
                       freq_limit: float) -> dict:
    R = r.R
    rec: dict = {"case": number, "requested_forces": forces,
                 "requested_footsteps": footsteps}
    try:
        case = r.structure.Cases.CreateSimple(
            number, f"FF-EF{forces}-FS{footsteps}", int(CaseNature.PERMANENT),
            int(CaseAnalysisType.DYNAMIC_FOOTFALL),
        )
    except Exception as exc:
        rec["error"] = f"CreateSimple(DYNAMIC_FOOTFALL): {_com(exc)}"
        return rec
    try:
        params = case.GetAnalysisParams().QueryInterface(
            R.IRobotFootfallAnalysisParams
        )
    except Exception as exc:
        rec["error"] = f"GetAnalysisParams->IRobotFootfallAnalysisParams: {_com(exc)}"
        return rec

    for name, value in (
        ("ExcitationMethod", int(R.I_FAEM_SELF_EXCITATION)),
        ("WalkersWeight", Q),
        ("MinWalkingFrequency", FMIN),
        ("MaxWalkingFrequency", FMAX),
        ("FootstepsNumber", footsteps),
    ):
        try:
            setattr(params, name, value)
        except Exception as exc:
            rec[f"set_{name}"] = _com(exc)
    try:
        params.ExcitationForces = int(forces)
    except Exception as exc:
        rec["set_ExcitationForces"] = _com(exc)

    try:
        damp = params.Damping
        damp.Type = int(R.I_DADT_CONSTANT)
        damp.ConstValue = BETA
    except Exception as exc:
        rec["set_Damping"] = _com(exc)

    try:
        mp = params.ModalParams
        mp.FrequencyLimit = float(freq_limit)
        mp.IncludeMassForDirX = True
        mp.IncludeMassForDirY = True
        mp.IncludeMassForDirZ = True
        mp.IgnoreDensity = False
    except Exception as exc:
        rec["set_ModalParams"] = _com(exc)

    for attr in ("ExcitationNodes", "ResponseNodes"):
        try:
            sel = getattr(params, attr)
            sel.Type = int(R.I_FANST_SELECTED_NODES)
            sel.SelectedNodes.FromText(str(MID_NODE))
        except Exception as exc:
            rec[f"set_{attr}"] = _com(exc)

    try:
        ok = case.SetAnalysisParams(params)
        rec["SetAnalysisParams"] = bool(ok)
    except Exception as exc:
        rec["error"] = f"SetAnalysisParams: {_com(exc)}"
        return rec

    # Read the parameters back off the case — this is how we find out which
    # raw integer Robot actually kept.
    try:
        back = r.structure.Cases.Get(number).QueryInterface(R.IRobotSimpleCase)
        bp = back.GetAnalysisParams().QueryInterface(R.IRobotFootfallAnalysisParams)
        rec["readback"] = {
            "ExcitationForces": int(bp.ExcitationForces),
            "ExcitationMethod": int(bp.ExcitationMethod),
            "WalkersWeight": float(bp.WalkersWeight),
            "FootstepsNumber": int(bp.FootstepsNumber),
            "MinWalkingFrequency": float(bp.MinWalkingFrequency),
            "MaxWalkingFrequency": float(bp.MaxWalkingFrequency),
            "Damping.Type": int(bp.Damping.Type),
            "Damping.ConstValue": float(bp.Damping.ConstValue),
            "ModalParams.FrequencyLimit": float(bp.ModalParams.FrequencyLimit),
            "ExcitationNodes.Type": int(bp.ExcitationNodes.Type),
            "ResponseNodes.Type": int(bp.ResponseNodes.Type),
        }
    except Exception as exc:
        rec["readback_error"] = _com(exc)
    return rec


def read_footfall(r, node: int, case: int) -> dict:
    try:
        v = r.structure.Results.Advanced.FootfallValue(int(node), int(case))
    except Exception as exc:
        return {"error": f"IRobotAdvancedResultServer.FootfallValue: {_com(exc)}"}
    out = {}
    for name in ("Frequency", "A", "RF_Resonant", "RF_Transient", "RF_Overall",
                 "VRMS", "VRMQ", "ExcitationNode"):
        try:
            out[name] = float(getattr(v, name))
        except Exception as exc:
            out[name] = _com(exc)
    return out


def read_modes(r, case: int, n: int = 4) -> list:
    modes = []
    for m in range(1, n + 1):
        try:
            ev = r.structure.Results.Advanced.Eigenvalues.Value(int(case), m)
            modes.append({
                "mode": m,
                "f": float(ev.Frequence),
                "T": float(ev.Period),
                "damping": float(ev.Damping),
            })
        except Exception as exc:
            modes.append({"mode": m, "error": _com(exc)})
            break
    return modes


def run_floor(name: str, spec: dict, project_type: int) -> dict:
    out: dict = {"name": name, **spec, "project_type": int(project_type)}
    with apeRobot(project_type=project_type, visible=False, verbose=True) as r:
        out["robot_version"] = r.robot_version
        r.analysis.use_status_window = False
        build(r, spec["mass_per_len"])
        out["ro_used"] = spec["mass_per_len"] * G_SI / A
        out["modal_setup"] = make_modal_case(r)

        out["footfall_setup"] = [
            make_footfall_case(r, num, ef, fs, spec["freq_limit"])
            for num, ef, fs in FF_CASES
        ]

        r.analysis.generate_model()
        out["calculate"] = bool(r.analysis.run())
        out["results_available"] = r.results.available

        out["modes"] = read_modes(r, 1, 4)
        out["footfall_results"] = {
            f"case{num}": read_footfall(r, MID_NODE, num)
            for num, _, _ in FF_CASES
        }
        # Params as they stand after the run.
        R = r.R
        post = {}
        for num, _, _ in FF_CASES:
            try:
                c = r.structure.Cases.Get(num).QueryInterface(R.IRobotSimpleCase)
                p = c.GetAnalysisParams().QueryInterface(
                    R.IRobotFootfallAnalysisParams
                )
                post[f"case{num}"] = {
                    "ExcitationForces": int(p.ExcitationForces),
                    "FootstepsNumber": int(p.FootstepsNumber),
                }
            except Exception as exc:
                post[f"case{num}"] = _com(exc)
        out["post_run_params"] = post

        rtd = HERE / f"footfall_{name}.rtd"
        try:
            r.ir_project.SaveAs(str(rtd))
            out["rtd"] = str(rtd)
        except Exception as exc:
            out["rtd_error"] = _com(exc)
    return out


def main() -> None:
    project_type = int(ProjectType.FRAME_2D)
    if len(sys.argv) > 1 and sys.argv[1] == "3d":
        project_type = int(ProjectType.FRAME_3D)
    results = {}
    for name, spec in FLOORS.items():
        try:
            results[name] = run_floor(name, spec, project_type)
        except Exception:
            results[name] = {"fatal": traceback.format_exc()}
        print(json.dumps(results[name], indent=2, default=str)[:4000])
    (HERE / "robot.json").write_text(json.dumps(results, indent=2, default=str))
    print("wrote", HERE / "robot.json")


if __name__ == "__main__":
    main()
