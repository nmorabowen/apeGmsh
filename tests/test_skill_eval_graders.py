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


MODEL_SIZE = {"pratt_truss": (12, 21, False), "frame_modal": (29, 42, True), "staged_footing": (2433, 1100, False)}
GEOSTATIC_UY = -0.0004       # m, the footing centre's settlement at the end of stage 1 (any value)


def _probe(task: str, work: Path, *, nodes: int | None = None) -> list[dict]:
    """A probe log as a faithful run writes it: one bridge, one OpenSees model of its size."""
    fem_nodes, elements, eigen = MODEL_SIZE[task]
    model = {"opened_by_model": True, "node": fem_nodes if nodes is None else nodes,
             "element": elements, "eigen": int(eigen), "analyze": [0]}
    record = {"pid": 1, "cwd": str(work), "models": [model], "recorders": [],
              "apesees": [{"fem_nodes": fem_nodes}], "probe_errors": []}
    if task == "staged_footing":
        record["recorders"] = [["Node", "-file", "uy.out", "-node", "7", "-dof", "2", "disp"],
                               ["Node", "-file", "reaction.out", "-node", "1", "2", "-dof", "2", "reaction"]]
    return [record]


def _write_recorders(work: Path, data: dict) -> None:
    """The recorder files a staged-footing run would leave: 10 + 30 rows."""
    import numpy as np

    settlement = np.asarray(data["footing_settlement_m"])
    stage1 = np.linspace(GEOSTATIC_UY / 10, GEOSTATIC_UY, 10)
    np.savetxt(work / "uy.out", np.concatenate([stage1, GEOSTATIC_UY - settlement]))
    totals = np.concatenate([np.linspace(360.0, data["base_reaction_geostatic_kN"], 10),
                             np.linspace(3620.0, data["base_reaction_final_kN"], 30)])
    np.savetxt(work / "reaction.out", np.column_stack([0.4 * totals, 0.6 * totals]))


def _oracle(graders, task: str, data: object, tmp_path: Path, probe: list | None = None) -> list:
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")
    if task == "staged_footing" and isinstance(data, dict) and "footing_settlement_m" in data:
        _write_recorders(tmp_path, FOOTING)
    return graders.ORACLES[task](tmp_path, _probe(task, tmp_path) if probe is None else probe)


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


@pytest.mark.parametrize("task", ["pratt_truss", "frame_modal", "staged_footing"])
def test_right_numbers_without_a_run_fail(graders, task: str, tmp_path: Path) -> None:
    """No probe record: nothing built the model or solved it, so the numbers prove nothing."""
    data = {"pratt_truss": PRATT, "frame_modal": FRAME, "staged_footing": FOOTING}[task]
    failed = {c.name for c in _oracle(graders, task, data, tmp_path, probe=[]) if not c.ok}
    assert {"probe_clean", "built_with_apesees", "opensees_solved_the_apegmsh_model"} <= failed


def test_an_opensees_model_that_is_not_the_apegmsh_model_fails(graders, tmp_path: Path) -> None:
    """A bridge was built, but OpenSees solved a model of another size (a hand-built one)."""
    probe = _probe("pratt_truss", tmp_path, nodes=13)
    failed = {c.name for c in _oracle(graders, "pratt_truss", PRATT, tmp_path, probe=probe) if not c.ok}
    assert failed == {"opensees_solved_the_apegmsh_model"}


def test_a_failed_analyze_fails(graders, tmp_path: Path) -> None:
    probe = _probe("pratt_truss", tmp_path)
    probe[0]["models"][0]["analyze"] = [0, -3]
    failed = {c.name for c in _oracle(graders, "pratt_truss", PRATT, tmp_path, probe=probe) if not c.ok}
    assert failed == {"opensees_solved_the_apegmsh_model"}


def test_a_frame_without_an_eigen_solve_fails(graders, tmp_path: Path) -> None:
    probe = _probe("frame_modal", tmp_path)
    probe[0]["models"][0]["eigen"] = 0
    failed = {c.name for c in _oracle(graders, "frame_modal", FRAME, tmp_path, probe=probe) if not c.ok}
    assert failed == {"opensees_solved_the_apegmsh_model"}


def test_readback_numbers_not_in_the_recorder_files_fail(graders, tmp_path: Path) -> None:
    """The settlements reported differ from the recorded ones: not read back from disk."""
    data = json.loads(json.dumps(FOOTING))
    data["footing_settlement_m"][5] *= 1.001
    data["base_reaction_final_kN"] += 0.01
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")
    _write_recorders(tmp_path, FOOTING)
    checks = graders.ORACLES["staged_footing"](tmp_path, _probe("staged_footing", tmp_path))
    failed = {c.name for c in checks if not c.ok}
    assert {"settlement_from_recorder", "reactions_from_recorder"} <= failed


def test_readback_without_recorder_files_fails(graders, tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(json.dumps(FOOTING), encoding="utf-8")
    checks = graders.ORACLES["staged_footing"](tmp_path, _probe("staged_footing", tmp_path))
    assert {c.name for c in checks if not c.ok} == {"recorder_files"}


def test_the_truss_stiffness_oracle_agrees_with_the_opensees_solve(graders) -> None:
    """Cross-engine: the grader's numpy solve (no apeGmsh, no OpenSees) against the known-good run."""
    assert math.isclose(graders.truss_midspan_uy(), PRATT["midspan_uy_m"], rel_tol=1e-9)


def test_an_elastic_footing_fails_the_plasticity_check(graders, tmp_path: Path) -> None:
    data = json.loads(json.dumps(FOOTING))
    first = data["footing_settlement_m"][0]
    data["footing_settlement_m"] = [first * k for k in range(1, 31)]
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")
    _write_recorders(tmp_path, data)
    checks = graders.ORACLES["staged_footing"](tmp_path, _probe("staged_footing", tmp_path))
    assert {c.name for c in checks if not c.ok} == {"clay_yields"}


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
    checks = graders.ORACLES["pratt_truss"](tmp_path, _probe("pratt_truss", tmp_path))
    assert [c.name for c in checks] == ["result_file"] and not checks[0].ok


def test_no_model_file_fails_closed(graders, tmp_path: Path) -> None:
    checks = graders.ORACLES["plate_with_hole"](tmp_path, [])
    assert [c.name for c in checks] == ["model_file"] and not checks[0].ok


class _Nodes:
    def __init__(self, coords) -> None:
        import numpy as np

        self.coords = np.asarray(coords, dtype=float)
        self.ids = np.arange(1, len(self.coords) + 1).astype(object)


class _Fem:
    def __init__(self, coords) -> None:
        self.nodes = _Nodes(coords)


def test_a_five_node_pyramid_is_measured(graders) -> None:
    """A unit-square-based pyramid of height 1 has volume 1/3 (a 5-node element)."""
    import numpy as np

    fem = _Fem([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)])
    volume = graders._Mesh(fem).measures(3, 5, np.array([[1, 2, 3, 4, 5]]))
    assert math.isclose(float(volume[0]), 1 / 3, rel_tol=1e-12)


def test_an_element_the_graders_cannot_measure_is_a_failed_check_not_a_crash(graders, monkeypatch,
                                                                             tmp_path: Path) -> None:
    import numpy as np

    fem = _Fem([(0, 0, 0), (1, 0, 0)])
    with pytest.raises(ValueError, match="no measure"):
        graders._Mesh(fem).measures(2, 7, np.array([[1, 2, 1, 2, 1, 2, 1]]))
    monkeypatch.setattr(graders, "_load_fem", lambda work: (fem, []))
    oracle = graders._mesh_oracle(
        lambda f: [graders.Check("area", bool(graders._Mesh(f).measures(2, 7, np.ones((1, 7), int)).size))])
    checks = oracle(tmp_path, [])
    assert [(c.name, c.ok) for c in checks] == [("mesh_values", False)]
    assert "no measure" in checks[0].detail


def test_a_latin1_solution_is_graded_not_crashed(graders, tmp_path: Path) -> None:
    """A script saved in Latin-1 fails to run (Python reads UTF-8); the grade records it."""
    script = tmp_path / "latin1.py"
    script.write_bytes('# caf\xe9 au lait: written by an editor set to Latin-1\nprint("ok")\n'.encode("latin-1"))
    grade = graders.grade("pratt_truss", script)
    assert not grade.passed and grade.failed_at == "run"
    assert "utf-8" in grade.output_tail.lower()


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


def test_a_grader_exception_is_a_failed_run_and_the_plan_goes_on(skill_eval, monkeypatch,
                                                                 tmp_path: Path) -> None:
    def boom(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xe9", 0, 1, "invalid continuation byte")

    monkeypatch.setattr(skill_eval.graders, "grade", boom)
    row = skill_eval.grade_safely("pratt_truss", tmp_path / "solution.py", python=sys.executable, tasks={})
    assert row["passed"] is False and row["failed_at"] == "grader_error"
    assert "UnicodeDecodeError" in row["error"]


@pytest.mark.parametrize(("arm", "available", "ok"), [
    ("skill", ["apegmsh"], True),
    ("skill", [{"name": "apegmsh", "path": "x"}], True),
    ("skill", ["apegmsh-helper"], False),          # a substring is not the skill
    ("skill", None, False),                        # no list: cannot show what loaded
    ("previous", [], False),
    ("no_skill", ["apegmsh"], False),
    ("no_skill", ["anthropic-skills:apegmsh-helper"], True),
    ("no_skill", None, False),
])
def test_each_arm_must_load_exactly_what_it_claims(skill_eval, arm, available, ok) -> None:
    if ok:
        skill_eval.check_arm_skills(arm, available)
    else:
        with pytest.raises(SystemExit):
            skill_eval.check_arm_skills(arm, available)


def test_instruction_files_above_the_scratch_root_are_found(skill_eval, tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("leak", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert str(tmp_path / "CLAUDE.md") in skill_eval.instruction_files(nested)


def test_the_default_scratch_root_is_fixed_and_outside_home(skill_eval) -> None:
    root = skill_eval.default_scratch()
    assert root == skill_eval.default_scratch()
    assert not root.resolve().is_relative_to(Path.home().resolve())


def test_the_grader_environment(graders, tmp_path: Path) -> None:
    import os

    env = graders.child_env(sys.executable)
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(os.path.abspath(sys.executable))
    assert env["PYTHONPATH"] == str(graders.SRC)
    assert graders.PROBE_ENV not in env and graders.PROBE_KEY_ENV not in env
    probed = graders.child_env(sys.executable, tmp_path / "probe.jsonl", b"k" * 32, src=tmp_path)
    assert probed["PYTHONPATH"].split(os.pathsep) == [str(tmp_path), str(graders.PROBE_DIR)]
    with pytest.raises(ValueError, match="key"):
        graders.child_env(sys.executable, tmp_path / "probe.jsonl")


FAKE_OPENSEES = ("def model(*a): pass\ndef node(*a): pass\ndef element(*a): pass\n"
                 "def eigen(*a): return [1.0]\ndef analyze(*a): return 0\ndef recorder(*a): pass\n")


def _run_probed(graders, tmp_path: Path, body: str, key: bytes) -> Path:
    """Run ``body`` under the probe with a stand-in ``opensees`` (no openseespy needed)."""
    import os

    fake = tmp_path / "fake"
    fake.mkdir(exist_ok=True)
    (fake / "opensees.py").write_text(FAKE_OPENSEES, encoding="utf-8")
    # First on sys.path inside the child: a .pth (the fork's installer) can put a real
    # ``opensees`` ahead of PYTHONPATH.
    script = f"import sys\nsys.path.insert(0, {str(fake)!r})\n" + body
    log = tmp_path / "probe.jsonl"
    env = graders.child_env(sys.executable, log, key)
    env["PYTHONPATH"] = os.pathsep.join([str(fake), env["PYTHONPATH"]])
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True, timeout=120)
    return log


def test_the_probe_logs_each_opensees_model_per_process(graders, tmp_path: Path) -> None:
    key = b"\x01" * 32
    log = _run_probed(graders, tmp_path, (
        "import opensees as ops\n"
        "assert ops.__file__.endswith('opensees.py'), ops.__file__\n"
        "ops.model('basic'); ops.node(1); ops.node(2); ops.element('truss', 1)\n"
        "ops.recorder('Node', '-file', 'u.out'); ops.analyze(1)\n"
        "ops.model('basic'); ops.node(1); ops.eigen(1)\n"), key)
    (record,) = graders.read_probe(log, key)
    assert record["probe_errors"] == []
    assert [(m["node"], m["element"], m["eigen"], m["analyze"]) for m in record["models"]] == [
        (2, 1, 0, [0]), (1, 0, 1, [])]
    assert record["recorders"] == [["Node", "-file", "u.out"]]


def test_the_script_cannot_see_the_probe_log_or_key(graders, tmp_path: Path) -> None:
    key = b"\x02" * 32
    log = _run_probed(graders, tmp_path, (
        "import os, sitecustomize\n"
        "assert 'APEGMSH_EVAL_PROBE_LOG' not in os.environ\n"
        "assert 'APEGMSH_EVAL_PROBE_KEY' not in os.environ\n"
        "public = [v for k, v in vars(sitecustomize).items() if not k.startswith('__')]\n"
        "assert not any(isinstance(v, (bytes, dict)) for v in public), public\n"), key)
    assert len(graders.read_probe(log, key)) == 1


def test_a_child_python_process_is_probed_too(graders, tmp_path: Path) -> None:
    """A deck run's interpreter is a child of the script: it gets the key, its code does not."""
    key = b"\x03" * 32
    child = (f"import sys; sys.path.insert(0, {str(tmp_path / 'fake')!r}); "
             "import os; assert 'APEGMSH_EVAL_PROBE_KEY' not in os.environ; "
             "import opensees as ops; ops.model('basic'); ops.node(1); ops.analyze(1)")
    log = _run_probed(graders, tmp_path, (
        "import subprocess\n"
        f"subprocess.run([sys.executable, '-c', {child!r}], check=True)\n"), key)
    records = graders.read_probe(log, key)
    assert len(records) == 2 and all(r["probe_errors"] == [] for r in records)
    assert [m["node"] for r in records for m in r["models"]] == [1]


def test_a_forged_probe_record_fails_the_grade(graders, tmp_path: Path) -> None:
    """Worst case: the script found the log and appended a record claiming a solve."""
    key = b"\x04" * 32
    log = _run_probed(graders, tmp_path, "import opensees\n", key)
    forged = {"pid": 9, "cwd": ".", "recorders": [], "probe_errors": [],
              "apesees": [{"fem_nodes": 12}],
              "models": [{"opened_by_model": True, "node": 12, "element": 21, "eigen": 0, "analyze": [0]}]}
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(forged) + "\n")                                       # unsigned
        fh.write(json.dumps({"record": json.dumps(forged), "mac": "0" * 64}) + "\n")   # bad MAC
    probe = graders.read_probe(log, key)
    errors = [e for r in probe for e in r["probe_errors"]]
    assert len(errors) == 2 and all("fails its MAC" in e for e in errors)
    checks = {c.name: c.ok for c in graders.check_evidence(probe, min_nodes=12, min_elements=21)}
    assert checks == {"probe_clean": False, "built_with_apesees": False,
                      "opensees_solved_the_apegmsh_model": False}


def test_the_agent_environment_names_no_repository_path(skill_eval, tmp_path: Path, monkeypatch) -> None:
    import os

    root = str(skill_eval.REPO)
    monkeypatch.setenv("PWD", root)                                  # a shell started in the repo
    monkeypatch.setenv("PATH", os.pathsep.join([str(skill_eval.REPO / "scripts"), os.environ["PATH"]]))
    env = skill_eval.agent_env(sys.executable, tmp_path / "apegmsh-src", tmp_path / "config")
    leaks = {k: v for k, v in env.items() if root.lower() in v.lower()
             or skill_eval.REPO.as_posix().lower() in v.lower()}
    assert leaks == {}
    assert env["PYTHONPATH"] == str(tmp_path / "apegmsh-src")
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "config")
    with pytest.raises(SystemExit, match="outside the repository"):
        skill_eval.agent_env(sys.executable, skill_eval.REPO / "src", tmp_path / "config")


def test_src_is_extracted_at_the_ref(skill_eval, tmp_path: Path) -> None:
    tree = skill_eval.extract_src("HEAD", tmp_path / "src")
    assert (tmp_path / "src" / "apeGmsh" / "__init__.py").is_file()
    assert tree == subprocess.run(["git", "rev-parse", "HEAD:src"], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()


def test_the_config_is_private_and_per_user(skill_eval, tmp_path: Path) -> None:
    import os
    import stat

    config = skill_eval.default_config()
    assert config.resolve().is_relative_to(Path.home().resolve())   # under the user's profile
    target = tmp_path / "a" / "config"
    skill_eval.make_private_dir(target)
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_only_python_children_get_the_probe_key(graders, tmp_path: Path) -> None:
    """A non-Python child, or Python with site off, sees neither the key nor the log path."""
    key = b"\x05" * 32
    shell = "['cmd', '/c', 'set']" if sys.platform == "win32" else "['env']"
    log = _run_probed(graders, tmp_path, (
        "import subprocess\n"
        f"listing = subprocess.run({shell}, capture_output=True, text=True, check=True).stdout\n"
        "assert 'APEGMSH_EVAL_PROBE' not in listing, listing\n"
        "isolated = subprocess.run([sys.executable, '-I', '-c', "
        "'import os; print(sorted(k for k in os.environ if k.startswith(\"APEGMSH_EVAL\")))'], "
        "capture_output=True, text=True, check=True).stdout\n"
        "assert isolated.strip() == '[]', isolated\n"), key)
    assert len(graders.read_probe(log, key)) == 1          # only the script's own record


def test_the_agent_environment_is_an_allow_list(skill_eval, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OLDPWD", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "venv"))
    monkeypatch.setenv("SOME_TOOL_TOKEN", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("TEMP", str(tmp_path))
    env = skill_eval.agent_env(sys.executable, tmp_path / "apegmsh-src", tmp_path / "config")
    assert not {"OLDPWD", "VIRTUAL_ENV", "SOME_TOOL_TOKEN"} & set(env)
    assert env["ANTHROPIC_API_KEY"] == "sk-test-not-real" and env["TEMP"] == str(tmp_path)
    expected = set(skill_eval.AGENT_ENV_ALLOW) | {"PATH", "PYTHONPATH", "MPLBACKEND",
                                                  "LADRUNO_OPENSEES_QUIET", "CLAUDE_CONFIG_DIR"}
    assert set(env) <= expected


def test_a_run_that_touches_the_repository_is_contaminated(skill_eval, tmp_path: Path) -> None:
    checkout = Path("C:/Users/someone/Github/apeGmsh") if sys.platform == "win32" else Path("/home/someone/apeGmsh")
    roots = [skill_eval.REPO, checkout]
    repo_file = skill_eval.REPO / "evals" / "apegmsh" / "graders.py"
    gitbash = "/" + str(checkout).replace("\\", "/").replace(":", "", 1) if sys.platform == "win32" else str(checkout)
    calls = [
        {"name": "Read", "input": {"file_path": str(repo_file)}},
        {"name": "Bash", "input": {"command": f"grep -rn distance {gitbash}/skills"}},
        {"name": "Grep", "input": {"pattern": "edges", "path": str(checkout).lower()}},
        {"name": "Read", "input": {"file_path": str(checkout) + "-fork/notes.md"}},   # a sibling
        {"name": "Write", "input": {"file_path": str(tmp_path / "solution.py"), "content": "x = 1"}},
        {"name": "Glob", "input": {"pattern": "*.py", "path": str(tmp_path)}},
    ]
    hits = skill_eval.contamination(calls, roots)
    assert [h.split(":", 1)[0] for h in hits] == ["Read", "Bash", "Grep"], hits
    assert skill_eval.contamination(calls[3:], roots) == []


def test_the_transcript_keeps_every_tool_call(skill_eval) -> None:
    events = [{"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
        {"type": "text", "text": "hi"},
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}}]
    parsed = skill_eval.parse_stream([json.dumps(e) for e in events])
    assert [c["name"] for c in parsed["tool_calls"]] == ["Read", "Bash"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs; POSIX is covered by the 0700 test")
def test_the_windows_work_root_is_locked_to_its_owner(skill_eval, tmp_path: Path) -> None:
    target = tmp_path / "work-root"
    skill_eval.make_private_dir(target)
    aces = skill_eval.acl_principals(target)
    assert aces and not any(inherited for _, inherited in aces), aces
    owner = skill_eval._windows_owner().lower()
    others = [p for p, _ in aces if p.lower() != owner and not p.upper().endswith("\\SYSTEM")]
    assert others == [], aces
