"""The apeGmsh skill-eval graders pass what is right and fail what is wrong (#2003).

Every grader must pass its known-good script and fail each known-bad one, at the
stage and for the reason ``evals/apegmsh/tasks.toml`` names, so a grader that
passes everything fails here. Lanes: the mesh and assembly tasks and every
known-bad script that fails before a solve need only gmsh (suite lane); the
solve tasks' known-good scripts, and their oracle-level known-bads, run an
OpenSees analysis (``live``, the live-stock lane). The solve oracles are also
checked on recorded numbers here, so the suite lane covers them without a solve.

Never runs ``claude``: the eval itself is the maintainer's (``scripts/skill_eval.py``).
"""
from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evals" / "apegmsh"
with open(EVAL_DIR / "tasks.toml", "rb") as _fh:
    TASKS = {t["id"]: t for t in tomllib.load(_fh)["task"]}

_LIVE = pytest.mark.live


@pytest.fixture(scope="module")
def graders():
    """``evals/apegmsh/graders.py``, imported with sys.path and sys.modules restored after."""
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.get(name) for name in ("graders",)}
    sys.path.insert(0, str(EVAL_DIR))
    try:
        yield importlib.import_module("graders")
    finally:
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _lane(lane: str) -> list[pytest.MarkDecorator]:
    return [_LIVE] if lane == "live" else []


GOOD = [pytest.param(tid, marks=_lane(t["lane"]), id=tid) for tid, t in TASKS.items()]
BAD = [
    pytest.param(tid, bad, marks=_lane(bad["lane"]), id=f"{tid}-{Path(bad['script']).stem}")
    for tid, t in TASKS.items() for bad in t["known_bad"]
]


# --------------------------------------------------------------------------- the task file

def test_six_tasks_of_the_designed_kinds(graders) -> None:
    tasks = graders.load_tasks()
    kinds = sorted(t["kind"] for t in tasks.values())
    assert kinds == ["assembly", "mesh", "mesh", "readback", "solve", "solve"]
    assert set(tasks) == set(graders.ORACLES)


def test_every_task_has_both_kinds_of_fixture_on_disk() -> None:
    for tid, task in TASKS.items():
        scripts = [task["known_good"]] + [b["script"] for b in task["known_bad"]]
        assert task["known_bad"], f"{tid} has no known-bad script: a pass-everything grader would go unseen"
        for script in scripts:
            path = (EVAL_DIR / script).resolve()
            assert path.is_file(), f"{tid}: missing fixture {script}"
            assert path.is_relative_to(EVAL_DIR / "fixtures"), f"{tid}: {script} is not a frozen fixture"


def test_the_card_defects_are_known_bad() -> None:
    """The card names two defects the graders must fail: the skeleton without Base, and edges=."""
    scripts = {Path(b["script"]).name for t in TASKS.values() for b in t["known_bad"]}
    assert {"skeleton_without_base.py", "field_distance_edges.py"} <= scripts
    edges = (EVAL_DIR / "fixtures" / "known_bad" / "field_distance_edges.py").read_text(encoding="utf-8")
    assert "distance(edges=" in edges
    skeleton = (EVAL_DIR / "fixtures" / "known_bad" / "skeleton_without_base.py").read_text(encoding="utf-8")
    assert 'pg="Base"' in skeleton and '"Base"' not in skeleton.split("ops.fix")[0]


def test_each_prompt_names_its_outputs_and_result_keys() -> None:
    for tid, task in TASKS.items():
        for name in task["outputs"] + task["result_keys"]:
            assert f"`{name}`" in task["prompt"], f"{tid}: the prompt never names `{name}`"


def test_load_tasks_refuses_an_unknown_oracle(graders, tmp_path: Path) -> None:
    text = (EVAL_DIR / "tasks.toml").read_text(encoding="utf-8")
    bad = tmp_path / "tasks.toml"
    bad.write_text(text.replace('oracle = "pratt_truss"', 'oracle = "no_such_oracle"'), encoding="utf-8")
    with pytest.raises(ValueError, match="no oracle named"):
        graders.load_tasks(bad)


def test_load_tasks_refuses_result_keys_the_oracle_does_not_read(graders, tmp_path: Path) -> None:
    text = (EVAL_DIR / "tasks.toml").read_text(encoding="utf-8")
    bad = tmp_path / "tasks.toml"
    bad.write_text(text.replace('result_keys = ["periods_s", "floor_ux_m"]',
                                'result_keys = ["periods_s"]'), encoding="utf-8")
    with pytest.raises(ValueError, match="result_keys"):
        graders.load_tasks(bad)


# --------------------------------------------------------------------------- end to end

@pytest.mark.parametrize("task_id", GOOD)
def test_known_good_passes(graders, task_id: str) -> None:
    grade = graders.grade(task_id, EVAL_DIR / TASKS[task_id]["known_good"])
    assert grade.passed, json.dumps(grade.to_dict(), indent=2)


@pytest.mark.parametrize(("task_id", "bad"), BAD)
def test_known_bad_fails_at_its_stage_for_its_reason(graders, task_id: str, bad: dict) -> None:
    grade = graders.grade(task_id, EVAL_DIR / bad["script"])
    report = json.dumps(grade.to_dict(), indent=2)
    assert not grade.passed, report
    assert grade.failed_at == bad["fails_at"], report
    if bad["fails_at"] == "run":
        evidence = grade.output_tail
    elif bad["fails_at"] == "lint":
        evidence = "\n".join(grade.lint)
    else:
        evidence = "\n".join(f"{c.name}: {c.detail}" for c in grade.checks if not c.ok)
    assert bad["reason"] in evidence, report


def test_a_missing_script_fails_at_run(graders, tmp_path: Path) -> None:
    grade = graders.grade("plate_with_hole", tmp_path / "solution.py")
    assert not grade.passed and grade.failed_at == "run"


# --------------------------------------------------------------------------- solve oracles on numbers

# The known-good scripts' own result.json (stock openseespy, 6308b303 reference scripts).
PRATT = {"bottom_chord_max_axial_N": 400000.0, "end_post_axial_N": -353553.3905932742,
         "midspan_uy_m": -0.01701878306203469}
FRAME = {"periods_s": [0.2542932963648463, 0.2542932963648463, 0.17526070140265876],
         "floor_ux_m": [0.0010196268712765596, 0.0021666330751354637]}
FOOTING = {
    "base_reaction_geostatic_kN": 3599.9996, "base_reaction_final_kN": 4200.0027,
    "footing_settlement_m": [
        0.0010603, 0.0021206, 0.0031809, 0.0042412, 0.0053015, 0.0063618, 0.0074221, 0.0084824,
        0.0095427, 0.010603, 0.0116633, 0.0127236, 0.0137839, 0.0148442, 0.0159045, 0.0169648,
        0.0180251, 0.0190854, 0.0201465, 0.0213367, 0.0227985, 0.0245525, 0.026586, 0.0289785,
        0.0317743, 0.0350515, 0.038821, 0.0431407, 0.0480303, 0.0536233,
    ],
}


def _oracle(graders, task: str, data: object, tmp_path: Path) -> list:
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")
    return graders.ORACLES[task](tmp_path)


@pytest.mark.parametrize(("task", "data"), [("pratt_truss", PRATT), ("frame_modal", FRAME),
                                            ("staged_footing", FOOTING)])
def test_solve_oracle_passes_the_reference_numbers(graders, task: str, data: dict, tmp_path: Path) -> None:
    checks = _oracle(graders, task, data, tmp_path)
    assert checks and all(c.ok for c in checks), [c.__dict__ for c in checks if not c.ok]


@pytest.mark.parametrize(("task", "key", "factor", "check"), [
    ("pratt_truss", "bottom_chord_max_axial_N", 1.01, "bottom_chord_by_sections"),
    ("pratt_truss", "midspan_uy_m", 0.99, "midspan_uy_by_stiffness"),
    ("frame_modal", "floor_ux_m", 2.0, "rayleigh_vs_eigen_T1"),
    ("frame_modal", "periods_s", 1.05, "T1_vs_golden"),
    ("staged_footing", "base_reaction_final_kN", 0.99, "base_carries_footing_load"),
    ("staged_footing", "footing_settlement_m", 0.5, "elastic_slope_vs_flamant"),
])
def test_solve_oracle_fails_a_wrong_number(graders, task, key, factor, check, tmp_path: Path) -> None:
    data = json.loads(json.dumps({"pratt_truss": PRATT, "frame_modal": FRAME,
                                  "staged_footing": FOOTING}[task]))
    value = data[key]
    data[key] = [v * factor for v in value] if isinstance(value, list) else value * factor
    failed = {c.name for c in _oracle(graders, task, data, tmp_path) if not c.ok}
    assert check in failed, failed


def test_the_truss_stiffness_oracle_agrees_with_the_opensees_solve(graders) -> None:
    """Cross-engine: the grader's numpy solve (no apeGmsh, no OpenSees) against the known-good run."""
    assert math.isclose(graders.truss_midspan_uy(), PRATT["midspan_uy_m"], rel_tol=1e-9)


def test_an_elastic_footing_fails_the_plasticity_check(graders, tmp_path: Path) -> None:
    data = json.loads(json.dumps(FOOTING))
    first = data["footing_settlement_m"][0]
    data["footing_settlement_m"] = [first * k for k in range(1, 31)]
    failed = {c.name for c in _oracle(graders, "staged_footing", data, tmp_path) if not c.ok}
    assert failed == {"clay_yields"}


@pytest.mark.parametrize(("payload", "check"), [
    ({}, "result_keys"),
    ([1, 2, 3], "result_keys"),
    ({**PRATT, "midspan_uy_m": float("nan")}, "result_values"),
    ({**PRATT, "midspan_uy_m": "down"}, "result_values"),
])
def test_a_malformed_result_fails_closed(graders, payload, check, tmp_path: Path) -> None:
    checks = _oracle(graders, "pratt_truss", payload, tmp_path)
    assert any(c.name == check and not c.ok for c in checks), [c.__dict__ for c in checks]


def test_no_result_file_fails_closed(graders, tmp_path: Path) -> None:
    checks = graders.ORACLES["pratt_truss"](tmp_path)
    assert [c.name for c in checks] == ["result_file"] and not checks[0].ok


def test_no_model_file_fails_closed(graders, tmp_path: Path) -> None:
    checks = graders.ORACLES["plate_with_hole"](tmp_path)
    assert [c.name for c in checks] == ["model_file"] and not checks[0].ok


def test_an_element_the_graders_cannot_measure_raises(graders) -> None:
    import numpy as np

    class _Nodes:
        ids = np.array([1, 2], dtype=object)
        coords = np.zeros((2, 3))

    class _Fem:
        nodes = _Nodes()

    with pytest.raises(ValueError, match="no measure"):
        graders._Mesh(_Fem()).measures(2, 7, np.array([[1, 2, 1, 2, 1, 2, 1]]))


# --------------------------------------------------------------------------- the runner, without claude

def test_dry_run_prints_the_plan_and_the_recomputable_skill_hash() -> None:
    tree = subprocess.run(["git", "rev-parse", "HEAD:skills/apegmsh"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "skill_eval.py"), "--dry-run",
                          "--skill-ref", "HEAD", "--claude", "no-such-claude-binary"],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    assert f"skill hash     {tree}" in run.stdout
    assert "planned runs" in run.stdout and "claude CLI     not found" in run.stdout


@pytest.fixture
def skill_eval(graders, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("skill_eval")
    yield module
    sys.modules.pop("skill_eval", None)


def test_the_skill_arm_installs_the_canonical_tree_as_apegmsh(skill_eval, tmp_path: Path) -> None:
    _, tree = skill_eval.resolve_skill("HEAD")
    skill_eval.install_skill_tree(tree, tmp_path / "apegmsh")
    assert (tmp_path / "apegmsh" / "SKILL.md").is_file()
    assert (tmp_path / "apegmsh" / "references" / "model-scripts.md").is_file()


def test_the_mirror_is_refused_because_it_is_not_named_apegmsh(skill_eval, tmp_path: Path) -> None:
    mirror = subprocess.run(["git", "rev-parse", "HEAD:.claude/skills/apegmsh-helper"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    with pytest.raises(SystemExit, match="expected"):
        skill_eval.install_skill_tree(mirror, tmp_path / "apegmsh")


def test_stream_parse_records_which_skills_fired(skill_eval) -> None:
    events = [
        {"type": "system", "subtype": "init", "model": "m-1", "skills": ["apegmsh"]},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Skill", "input": {"skill": "gmsh-structural"}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Skill", "input": {"skill": "apegmsh"}}]}},
        {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 5}, "num_turns": 2,
         "total_cost_usd": 0.01, "is_error": False, "result": "done"},
    ]
    parsed = skill_eval.parse_stream([json.dumps(e) for e in events] + ["not json"])
    assert parsed["model"] == "m-1"
    assert parsed["fired_skills"] == ["gmsh-structural", "apegmsh"]
    assert parsed["usage"] == {"input_tokens": 10, "output_tokens": 5}
