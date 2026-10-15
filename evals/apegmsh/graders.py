"""Deterministic graders for the apeGmsh skill evals (SK1-S0, #2003).

A task passes when its script runs, lints clean, and meets the task's oracle.
The grader never trusts what the agent left behind: it copies the script alone
into a fresh directory as ``solution.py``, runs it there with this checkout's
``src`` on ``PYTHONPATH``, lints it with the skill's ``lint_model_script.py``
(this checkout's copy), and reads the outputs the run itself wrote.

Each oracle names the right answer independently of apeGmsh:

* ``plate_with_hole``, ``footing_mesh``, ``column_on_footing``: FEMData
  invariants read back from ``model.h5`` (exact areas and volumes, nodes on
  their named edges, the grading and the conformity the task asks for).
* ``pratt_truss``: statics by hand (method of sections and of joints) and the
  midspan deflection from an independent numpy stiffness solve.
* ``frame_modal``: Rayleigh's period from the reported floor displacements,
  against the reported eigen period, and that period against a golden.
* ``staged_footing``: base-reaction equilibrium, the Flamant elastic slope, and
  the plastic share of the final settlement.

Deck diffs are never used: deck text changes by design. ``tasks.toml`` holds
the prompts; the numbers below restate the prompts' data.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parents[1]
SRC = REPO / "src"
LINT = REPO / "skills" / "apegmsh" / "scripts" / "lint_model_script.py"
TASKS_FILE = EVAL_DIR / "tasks.toml"

SCRIPT_NAME = "solution.py"
RESULT_NAME = "result.json"
MODEL_NAME = "model.h5"
RUN_TIMEOUT = 900          # s, one script run; the reference-script test allows the same
OUTPUT_TAIL = 3000         # characters of stdout/stderr kept in a grade

TASK_KEYS = {"id", "kind", "oracle", "lane", "prompt", "outputs", "result_keys",
             "known_good", "known_bad"}
KNOWN_BAD_KEYS = {"script", "fails_at", "reason", "lane"}
KINDS = {"mesh", "solve", "readback", "assembly"}
LANES = {"suite", "live"}
FAIL_STAGES = {"run", "lint", "oracle"}


# --------------------------------------------------------------------------- records

@dataclass
class Check:
    """One oracle assertion: its name, whether it held, and the numbers behind it."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class Grade:
    """The grade of one script against one task."""

    task: str
    ran: bool
    returncode: int | None
    lint: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    output_tail: str = ""

    @property
    def passed(self) -> bool:
        return self.ran and not self.lint and bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def failed_at(self) -> str | None:
        """The first stage that failed: ``run``, ``lint`` or ``oracle``; ``None`` on a pass."""
        if not self.ran:
            return "run"
        if self.lint:
            return "lint"
        if not self.checks or not all(c.ok for c in self.checks):
            return "oracle"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "passed": self.passed,
            "failed_at": self.failed_at,
            "ran": self.ran,
            "returncode": self.returncode,
            "lint": self.lint,
            "checks": [c.__dict__ for c in self.checks],
            "output_tail": self.output_tail,
        }


# --------------------------------------------------------------------------- tasks

def load_tasks(path: Path = TASKS_FILE) -> dict[str, dict[str, Any]]:
    """Read ``tasks.toml`` and refuse anything the graders cannot honour."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    unknown_top = set(data) - {"common", "task"}
    if unknown_top:
        raise ValueError(f"{path.name}: unknown top-level tables {sorted(unknown_top)}")
    tasks: dict[str, dict[str, Any]] = {}
    for task in data.get("task", []):
        extra = set(task) - TASK_KEYS
        missing = TASK_KEYS - set(task)
        if extra or missing:
            raise ValueError(f"task {task.get('id')!r}: extra keys {sorted(extra)}, missing {sorted(missing)}")
        if task["kind"] not in KINDS:
            raise ValueError(f"task {task['id']!r}: unknown kind {task['kind']!r}")
        if task["lane"] not in LANES:
            raise ValueError(f"task {task['id']!r}: unknown lane {task['lane']!r}")
        if task["oracle"] not in ORACLES:
            raise ValueError(f"task {task['id']!r}: no oracle named {task['oracle']!r}")
        expected_keys = list(RESULT_KEYS.get(task["oracle"], ()))
        if task["result_keys"] != expected_keys:
            raise ValueError(f"task {task['id']!r}: result_keys {task['result_keys']} but the oracle "
                             f"reads {expected_keys}")
        for bad in task["known_bad"]:
            if (set(bad) != KNOWN_BAD_KEYS or bad["fails_at"] not in FAIL_STAGES
                    or bad["lane"] not in LANES):
                raise ValueError(f"task {task['id']!r}: malformed known_bad entry {bad!r}")
        if task["id"] in tasks:
            raise ValueError(f"duplicate task id {task['id']!r}")
        tasks[task["id"]] = task
    if not tasks:
        raise ValueError(f"{path.name} defines no tasks")
    return tasks


def load_common(path: Path = TASKS_FILE) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)["common"]


# --------------------------------------------------------------------------- run, lint, grade

def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env["MPLBACKEND"] = "Agg"                 # a stray plt.show() must not block the grade
    env["LADRUNO_OPENSEES_QUIET"] = "1"       # the fork build's banner is noise in the tail
    return env


def lint_script(script: Path, *, python: str = sys.executable) -> list[str]:
    """Run the skill's lint; return its findings (``path:line: RULE message`` lines)."""
    run = subprocess.run(
        [python, str(LINT), script.name], cwd=script.parent,
        capture_output=True, text=True, timeout=120,
    )
    findings = [ln for ln in run.stdout.splitlines() if ln.strip()]
    if run.returncode not in (0, 1):
        raise RuntimeError(f"lint crashed (exit {run.returncode}):\n{run.stderr[-OUTPUT_TAIL:]}")
    if run.returncode == 1 and not findings:
        raise RuntimeError("lint exited 1 without printing a finding")
    return findings


def grade(task_id: str, script: Path, *, python: str = sys.executable,
          workdir: Path | None = None, timeout: float = RUN_TIMEOUT,
          tasks: dict[str, dict[str, Any]] | None = None) -> Grade:
    """Grade ``script`` against task ``task_id`` in a fresh directory."""
    tasks = tasks if tasks is not None else load_tasks()
    if task_id not in tasks:
        raise KeyError(f"unknown task {task_id!r}; known: {sorted(tasks)}")
    oracle = ORACLES[tasks[task_id]["oracle"]]
    script = Path(script)
    if not script.is_file():
        return Grade(task_id, ran=False, returncode=None,
                     output_tail=f"no script at {script}")

    own_dir = workdir is None
    work = Path(tempfile.mkdtemp(prefix=f"skill_eval_{task_id}_")) if own_dir else Path(workdir)
    try:
        if not own_dir:
            if work.exists() and any(work.iterdir()):
                raise FileExistsError(f"grading directory {work} is not empty")
            work.mkdir(parents=True, exist_ok=True)
        target = work / SCRIPT_NAME
        shutil.copyfile(script, target)
        try:
            run = subprocess.run(
                [python, SCRIPT_NAME], cwd=work, env=_child_env(),
                capture_output=True, text=True, timeout=timeout,
            )
            returncode: int | None = run.returncode
            tail = (run.stdout + "\n" + run.stderr)[-OUTPUT_TAIL:]
        except subprocess.TimeoutExpired:
            returncode, tail = None, f"timed out after {timeout:.0f} s"
        result = Grade(task_id, ran=returncode == 0, returncode=returncode, output_tail=tail)
        result.lint = lint_script(target, python=python)
        if result.ran:
            result.checks = oracle(work)
        return result
    finally:
        if own_dir:
            shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- helpers

def _rel(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


def _close(name: str, actual: float, expected: float, tol: float) -> Check:
    err = _rel(actual, expected)
    return Check(name, err <= tol, f"got {actual:.6g}, expected {expected:.6g} (rel {err:.2e}, tol {tol:.0e})")


def _read_result(work: Path, keys: tuple[str, ...]) -> tuple[dict[str, Any] | None, list[Check]]:
    path = work / RESULT_NAME
    if not path.is_file():
        return None, [Check("result_file", False, f"the run wrote no {RESULT_NAME}")]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [Check("result_file", False, f"{RESULT_NAME} is not JSON: {exc}")]
    if not isinstance(data, dict):
        return None, [Check("result_keys", False, f"{RESULT_NAME} holds a {type(data).__name__}, not an object")]
    missing = [k for k in keys if k not in data]
    if missing:
        return None, [Check("result_keys", False, f"missing keys {missing}")]
    return data, [Check("result_keys", True, f"keys {list(keys)} present")]


def _number(data: dict[str, Any], key: str) -> float:
    value = float(data[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def _numbers(data: dict[str, Any], key: str) -> np.ndarray:
    values = np.asarray(data[key], dtype=float).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} holds a non-finite value")
    return values


def _load_fem(work: Path) -> tuple[Any, list[Check]]:
    path = work / MODEL_NAME
    if not path.is_file():
        return None, [Check("model_file", False, f"the run wrote no {MODEL_NAME}")]
    from apeGmsh import FEMData   # imported here so the json oracles need no apeGmsh
    try:
        fem = FEMData.from_h5(str(path))
    except Exception as exc:  # noqa: BLE001 - any read failure is a failed grade, named
        return None, [Check("model_file", False, f"FEMData.from_h5 failed: {type(exc).__name__}: {exc}")]
    return fem, [Check("model_file", True, f"{MODEL_NAME} read back")]


# Corner counts per (dim, nodes per element). The first corners of a gmsh
# element are its vertices in every order, so measures use those alone.
_CORNERS = {
    (2, 3): ("tri", 3), (2, 6): ("tri", 3),
    (2, 4): ("quad", 4), (2, 8): ("quad", 4), (2, 9): ("quad", 4),
    (3, 4): ("tet", 4), (3, 10): ("tet", 4),
    (3, 8): ("hex", 8), (3, 20): ("hex", 8), (3, 27): ("hex", 8),
    (3, 6): ("prism", 6), (3, 15): ("prism", 6), (3, 18): ("prism", 6),
}
# Hexahedron and prism split into tetrahedra (gmsh corner order).
_HEX_TETS = ((0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (3, 4, 6, 7), (1, 4, 5, 6))
_PRISM_TETS = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))


class _Mesh:
    """Node coordinates by id and per-group element blocks, read from a FEMData."""

    def __init__(self, fem: Any) -> None:
        self.fem = fem
        ids = np.asarray([int(i) for i in fem.nodes.ids])
        self.row = {int(n): r for r, n in enumerate(ids)}
        self.xyz = np.asarray(fem.nodes.coords, dtype=float)

    def nodes(self, pg: str) -> np.ndarray:
        """Coordinates of the nodes of physical group ``pg`` (raises KeyError if absent)."""
        return np.asarray(self.fem.nodes.select(pg=pg).coords, dtype=float)

    def node_ids(self, pg: str) -> set[int]:
        return {int(n) for n in self.fem.nodes.select(pg=pg).ids}

    def blocks(self, pg: str) -> list[tuple[int, int, np.ndarray]]:
        """``(dim, npe, connectivity)`` per element type in group ``pg``."""
        out = []
        for group in self.fem.elements.select(pg=pg).groups():
            rows = [np.asarray(conn, dtype=int) for _, conn in group]
            conn = np.vstack(rows) if rows else np.zeros((0, group.npe), dtype=int)
            out.append((int(group.dim), int(group.npe), conn))
        return out

    def measures(self, dim: int, npe: int, conn: np.ndarray) -> np.ndarray:
        """Area (dim 2) or volume (dim 3) of each element, from its corners."""
        if (dim, npe) not in _CORNERS:
            raise ValueError(f"no measure for a {dim}-D element with {npe} nodes")
        family, n_corners = _CORNERS[(dim, npe)]
        rows = np.vectorize(self.row.__getitem__)(conn[:, :n_corners]) if len(conn) else conn
        p = self.xyz[rows] if len(conn) else np.zeros((0, n_corners, 3))
        if family == "tri":
            return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        if family == "quad":
            a = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            b = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])
            return 0.5 * (np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1))
        splits = {"tet": ((0, 1, 2, 3),), "hex": _HEX_TETS, "prism": _PRISM_TETS}[family]
        total = np.zeros(len(p))
        for i, j, k, m in splits:
            total += np.abs(np.einsum(
                "ij,ij->i", p[:, j] - p[:, i], np.cross(p[:, k] - p[:, i], p[:, m] - p[:, i]),
            )) / 6.0
        return total

    def element_measures(self, pg: str, dim: int) -> tuple[np.ndarray, list[Check]]:
        """Measures of the ``dim``-D elements of ``pg``; a block of another dim fails."""
        values, checks = [], []
        for block_dim, npe, conn in self.blocks(pg):
            if block_dim != dim:
                checks.append(Check(f"{pg}_dim", False, f"holds {block_dim}-D elements ({npe} nodes)"))
                continue
            values.append(self.measures(block_dim, npe, conn))
        if not values:
            checks.append(Check(f"{pg}_dim", False, f"holds no {dim}-D elements"))
            return np.zeros(0), checks
        return np.concatenate(values), checks


def _group_checks(mesh: _Mesh, names: list[str]) -> list[Check]:
    checks = []
    for name in names:
        try:
            count = len(mesh.node_ids(name))
            checks.append(Check(f"group_{name}", count > 0, f"{count} nodes"))
        except KeyError as exc:
            checks.append(Check(f"group_{name}", False, str(exc)))
    return checks


def _on_line(name: str, coords: np.ndarray, axis: int, value: float,
             span_axis: int, span: tuple[float, float], tol: float = 1e-6) -> Check:
    """All nodes at ``coords[:, axis] == value`` and covering ``span`` along ``span_axis``."""
    if len(coords) == 0:
        return Check(name, False, "no nodes")
    off = float(np.max(np.abs(coords[:, axis] - value)))
    lo, hi = float(coords[:, span_axis].min()), float(coords[:, span_axis].max())
    ok = off <= tol and lo <= span[0] + tol and hi >= span[1] - tol
    return Check(name, ok, f"max offset {off:.2e}, span [{lo:.4g}, {hi:.4g}] vs {list(span)}")


# --------------------------------------------------------------------------- mesh oracles

# plate_with_hole: a 10 x 5 m plate, hole of radius 1 m at its centre, elements of 0.25 m.
PLATE_LENGTH, PLATE_HEIGHT, HOLE_RADIUS, PLATE_SIZE = 10.0, 5.0, 1.0, 0.25
# The hole is meshed as a polygon: the inscribed-polygon error at 0.25 m is ~0.1 % of
# the hole, ~0.01 % of the plate. A plate without its hole is 6.7 % off.
TOL_PLATE_AREA = 0.005
# A 0.25 m element is 0.0625 m2 as a quad, 0.027 m2 as a triangle; 1.5x covers a quad
# mesh, and gmsh's default size on this plate is several times coarser.
MAX_MEAN_PLATE_ELEMENT = 1.5 * PLATE_SIZE**2


def check_plate_with_hole(fem: Any) -> list[Check]:
    mesh = _Mesh(fem)
    checks = _group_checks(mesh, ["plate", "left_edge", "right_edge", "hole_edge"])
    if not all(c.ok for c in checks):
        return checks
    areas, dim_checks = mesh.element_measures("plate", dim=2)
    checks += dim_checks
    exact = PLATE_LENGTH * PLATE_HEIGHT - math.pi * HOLE_RADIUS**2
    checks.append(_close("plate_area", float(areas.sum()), exact, TOL_PLATE_AREA))
    mean = float(areas.mean()) if len(areas) else math.inf
    checks.append(Check("element_size", mean <= MAX_MEAN_PLATE_ELEMENT,
                        f"mean element area {mean:.4g} m2, at most {MAX_MEAN_PLATE_ELEMENT:.4g}"))
    checks.append(_on_line("left_edge_at_x0", mesh.nodes("left_edge"), 0, 0.0, 1, (0.0, PLATE_HEIGHT)))
    checks.append(_on_line("right_edge_at_xL", mesh.nodes("right_edge"), 0, PLATE_LENGTH, 1,
                           (0.0, PLATE_HEIGHT)))
    hole = mesh.nodes("hole_edge")
    radius = np.hypot(hole[:, 0] - PLATE_LENGTH / 2, hole[:, 1] - PLATE_HEIGHT / 2)
    off = float(np.max(np.abs(radius - HOLE_RADIUS)))
    min_nodes = int(0.65 * 2 * math.pi * HOLE_RADIUS / PLATE_SIZE)
    checks.append(Check("hole_edge_on_circle", off <= 1e-6 and len(hole) >= min_nodes,
                        f"{len(hole)} nodes (at least {min_nodes}), max radius offset {off:.2e}"))
    return checks


# footing_mesh: a 20 x 10 m block, a 2 m footing at the top centre, 6-node triangles
# graded from 0.25 m along the footing to 1.0 m by 8 m away.
SOIL_WIDTH, SOIL_DEPTH, FOOTING_WIDTH = 20.0, 10.0, 2.0
SIZE_FOOTING, SIZE_FAR = 0.25, 1.0
TOL_BLOCK_AREA = 1e-6        # straight edges: the corners tile the block exactly
# A triangle of side h has area 0.433 h^2. Along the footing (h = 0.25) that is 0.027 m2;
# 2.2x allows gmsh's size scatter. At the base (h = 1.0, 10 m from the footing) it is
# 0.43 m2; a uniform 0.25 m mesh would put 0.027 m2 there, so 0.2 m2 separates the two.
MAX_MEAN_AREA_AT_FOOTING = 2.2 * 0.433 * SIZE_FOOTING**2
MIN_MEAN_AREA_AT_BASE = 0.2
MAX_FOOTING_EDGE = 1.25 * SIZE_FOOTING


def check_footing_mesh(fem: Any) -> list[Check]:
    mesh = _Mesh(fem)
    checks = _group_checks(mesh, ["soil", "base", "sides", "footing", "footing_centre"])
    if not all(c.ok for c in checks):
        return checks
    blocks = mesh.blocks("soil")
    npes = sorted({npe for dim, npe, _ in blocks if dim == 2})
    checks.append(Check("six_node_triangles", npes == [6] and all(d == 2 for d, _, _ in blocks),
                        f"2-D element node counts {npes}"))
    if not checks[-1].ok:
        return checks
    conn = np.vstack([c for _, _, c in blocks])
    areas = mesh.measures(2, 6, conn)
    checks.append(_close("block_area", float(areas.sum()), SOIL_WIDTH * SOIL_DEPTH, TOL_BLOCK_AREA))

    half, depth = SOIL_WIDTH / 2, SOIL_DEPTH
    checks.append(_on_line("base_at_bottom", mesh.nodes("base"), 1, -depth, 0, (-half, half)))
    sides = mesh.nodes("sides")
    side_off = float(np.max(np.abs(np.abs(sides[:, 0]) - half)))
    both = bool(sides[:, 0].min() < 0 < sides[:, 0].max())
    checks.append(Check("sides_at_x_pm_half", side_off <= 1e-6 and both,
                        f"max offset {side_off:.2e}, both sides present: {both}"))
    checks.append(_on_line("footing_on_ground", mesh.nodes("footing"), 1, 0.0, 0,
                           (-FOOTING_WIDTH / 2, FOOTING_WIDTH / 2)))
    footing_x = mesh.nodes("footing")[:, 0]
    inside = bool(np.all(np.abs(footing_x) <= FOOTING_WIDTH / 2 + 1e-6))
    checks.append(Check("footing_within_strip", inside, f"x in [{footing_x.min():.4g}, {footing_x.max():.4g}]"))
    centre = mesh.nodes("footing_centre")
    checks.append(Check("footing_centre_node", len(centre) == 1 and float(np.abs(centre).max()) <= 1e-6,
                        f"{len(centre)} node(s) at {centre.tolist()}"))

    # Grading: element size where the task fixes it, read from the elements touching
    # the footing and the base.
    footing_ids, base_ids = mesh.node_ids("footing"), mesh.node_ids("base")
    at_footing = np.array([any(int(n) in footing_ids for n in row) for row in conn])
    at_base = np.array([any(int(n) in base_ids for n in row) for row in conn])
    fine = float(areas[at_footing].mean()) if at_footing.any() else math.inf
    coarse = float(areas[at_base].mean()) if at_base.any() else 0.0
    checks.append(Check("fine_at_footing", fine <= MAX_MEAN_AREA_AT_FOOTING,
                        f"mean area {fine:.4g} m2 at the footing, at most {MAX_MEAN_AREA_AT_FOOTING:.4g}"))
    checks.append(Check("coarse_at_base", coarse >= MIN_MEAN_AREA_AT_BASE,
                        f"mean area {coarse:.4g} m2 at the base, at least {MIN_MEAN_AREA_AT_BASE:.4g}"))
    footing_lines = [c for d, _, c in mesh.blocks("footing") if d == 1]
    if footing_lines:
        ends = np.vstack(footing_lines)[:, :2]
        a = mesh.xyz[np.vectorize(mesh.row.__getitem__)(ends[:, 0])]
        b = mesh.xyz[np.vectorize(mesh.row.__getitem__)(ends[:, 1])]
        longest = float(np.linalg.norm(b - a, axis=1).max())
    else:
        longest = math.inf
    checks.append(Check("footing_edge_size", longest <= MAX_FOOTING_EDGE,
                        f"longest footing edge {longest:.4g} m, at most {MAX_FOOTING_EDGE:.4g}"))
    return checks


# column_on_footing: a 0.5 x 0.5 x 3 m column on a 2 x 2 x 0.5 m footing (top at z = 0).
COLUMN_VOLUME = 0.5 * 0.5 * 3.0
FOOTING_VOLUME = 2.0 * 2.0 * 0.5
TOL_VOLUME = 1e-6            # boxes: the corners fill them exactly


def check_column_on_footing(fem: Any) -> list[Check]:
    mesh = _Mesh(fem)
    checks = _group_checks(mesh, ["column", "footing"])
    if not all(c.ok for c in checks):
        return checks
    for name, exact in (("column", COLUMN_VOLUME), ("footing", FOOTING_VOLUME)):
        volumes, dim_checks = mesh.element_measures(name, dim=3)
        checks += dim_checks
        checks.append(_close(f"{name}_volume", float(volumes.sum()), exact, TOL_VOLUME))

    # Conformal: every column node on the contact face is a footing node too. A
    # non-fragmented assembly duplicates them, so the column's face nodes are its own.
    column_ids, footing_ids = mesh.node_ids("column"), mesh.node_ids("footing")
    contact = {n for n in column_ids if abs(mesh.xyz[mesh.row[n], 2]) <= 1e-6}
    shared = contact & footing_ids
    checks.append(Check("contact_face_conformal", len(contact) >= 4 and shared == contact,
                        f"{len(shared)} of {len(contact)} column contact-face nodes are footing nodes"))
    rounded = np.round(mesh.xyz, 9)
    unique = len(np.unique(rounded, axis=0))
    checks.append(Check("no_duplicate_nodes", unique == len(rounded),
                        f"{len(rounded) - unique} nodes share a location with another"))
    return checks


# --------------------------------------------------------------------------- solve oracles

# pratt_truss: 6 panels of 4 m, depth 4 m, E = 200 GPa; chords and end posts 6000 mm2,
# verticals 3000 mm2, diagonals 4000 mm2; 100 kN down at B1..B5; pin B0, roller B6.
TRUSS_PANEL, TRUSS_DEPTH, TRUSS_E, TRUSS_LOAD = 4.0, 4.0, 200e9, 100e3
TRUSS_MEMBERS = {
    6.0e-3: ["B0-B1", "B1-B2", "B2-B3", "B3-B4", "B4-B5", "B5-B6",
             "T1-T2", "T2-T3", "T3-T4", "T4-T5", "B0-T1", "T5-B6"],
    3.0e-3: ["B1-T1", "B2-T2", "B3-T3", "B4-T4", "B5-T5"],
    4.0e-3: ["T1-B2", "T2-B3", "T4-B3", "T5-B4"],
}
TOL_TRUSS = 1e-4             # determinate and linear: only round-off and print precision


def truss_midspan_uy() -> float:
    """Midspan (B3) vertical displacement by an independent numpy stiffness solve, m."""
    xy = {f"B{i}": (i * TRUSS_PANEL, 0.0) for i in range(7)}
    xy.update({f"T{i}": (i * TRUSS_PANEL, TRUSS_DEPTH) for i in range(1, 6)})
    index = {name: i for i, name in enumerate(xy)}
    k_global = np.zeros((2 * len(xy), 2 * len(xy)))
    for area, bars in TRUSS_MEMBERS.items():
        for bar in bars:
            a, b = bar.split("-")
            d = np.subtract(xy[b], xy[a])
            length = float(np.hypot(*d))
            c = np.concatenate([-d, d]) / length
            dofs = [2 * index[a], 2 * index[a] + 1, 2 * index[b], 2 * index[b] + 1]
            k_global[np.ix_(dofs, dofs)] += TRUSS_E * area / length * np.outer(c, c)
    force = np.zeros(2 * len(xy))
    for joint in ("B1", "B2", "B3", "B4", "B5"):
        force[2 * index[joint] + 1] = -TRUSS_LOAD
    fixed = {2 * index["B0"], 2 * index["B0"] + 1, 2 * index["B6"] + 1}
    free = [i for i in range(len(force)) if i not in fixed]
    u = np.zeros_like(force)
    u[free] = np.linalg.solve(k_global[np.ix_(free, free)], force[free])
    return float(u[2 * index["B3"] + 1])


def check_pratt_truss(data: dict[str, Any]) -> list[Check]:
    reaction = 5 * TRUSS_LOAD / 2
    # Method of sections through panel B2-B3, moments about T2.
    bottom_chord = (reaction * 2 * TRUSS_PANEL - TRUSS_LOAD * TRUSS_PANEL) / TRUSS_DEPTH
    # Method of joints at B0: the end post balances the reaction.
    end_post = -reaction * math.hypot(TRUSS_PANEL, TRUSS_DEPTH) / TRUSS_DEPTH
    return [
        _close("bottom_chord_by_sections", _number(data, "bottom_chord_max_axial_N"), bottom_chord, TOL_TRUSS),
        _close("end_post_by_joints", _number(data, "end_post_axial_N"), end_post, TOL_TRUSS),
        _close("midspan_uy_by_stiffness", _number(data, "midspan_uy_m"), truss_midspan_uy(), TOL_TRUSS),
    ]


# frame_modal: 2 x 2 bays of 6 m, two storeys of 3.2 m, floor load 8 kPa at each
# floor master, lateral load 10 % of the seismic weight in X with F_k = V h_k / sum(h).
FRAME_PLAN, FRAME_STOREY, FRAME_PRESSURE, FRAME_G = 12.0, 3.2, 8e3, 9.81
FRAME_FLOOR_WEIGHT = FRAME_PRESSURE * FRAME_PLAN**2
FRAME_FLOOR_MASS = FRAME_FLOOR_WEIGHT / FRAME_G
FRAME_FORCES = [0.10 * 2 * FRAME_FLOOR_WEIGHT * k / 3 for k in (1, 2)]   # h_k / sum(h) = k / 3
# Golden: T1 of the frame_modal reference script at 6308b303 (0.25429 s). The task fixes
# every section and mass, so a faithful model lands within round-off; 1 % allows an
# agent's own torsion constants (they do not enter the X translation mode).
FRAME_T1_GOLDEN = 0.2542933
TOL_FRAME_T1 = 0.01
TOL_RAYLEIGH = 0.005         # as the reference script: Rayleigh vs eigen T1
TOL_SYMMETRY = 1e-3          # symmetric plan: T1 and T2 coincide


def check_frame_modal(data: dict[str, Any]) -> list[Check]:
    periods = _numbers(data, "periods_s")
    ux = _numbers(data, "floor_ux_m")
    checks = [Check("result_shapes", len(periods) >= 3 and len(ux) == 2,
                    f"{len(periods)} periods (at least 3), {len(ux)} floor displacements (2)")]
    if not checks[0].ok:
        return checks
    checks.append(Check("floors_sway_in_x", bool(0 < ux[0] < ux[1]), f"floor ux {ux.tolist()} m"))
    kinetic = sum(FRAME_FLOOR_MASS * u**2 for u in ux)
    work = sum(f * u for f, u in zip(FRAME_FORCES, ux))
    rayleigh = 2 * math.pi * math.sqrt(kinetic / work) if work > 0 else math.inf
    checks.append(_close("rayleigh_vs_eigen_T1", float(periods[0]), rayleigh, TOL_RAYLEIGH))
    checks.append(_close("T1_vs_golden", float(periods[0]), FRAME_T1_GOLDEN, TOL_FRAME_T1))
    checks.append(_close("T1_equals_T2_by_symmetry", float(periods[1]), float(periods[0]), TOL_SYMMETRY))
    return checks


# staged_footing: the footing_mesh block of J2 clay (E = 30 MPa, nu = 0.3,
# 18 kN/m3, su = 75 kPa) on a 1 m slice, self-weight then 300 kPa in 30 increments.
CLAY_E, CLAY_NU, CLAY_GAMMA, FOOTING_PRESSURE, FOOTING_STEPS = 30.0e3, 0.3, 18.0, 300.0, 30
TOL_EQUILIBRIUM = 1e-3       # as the reference script
TOL_FLAMANT = 0.15           # the hand estimate ignores the rigid base and the rollers
MIN_PLASTIC_SHARE = 0.10     # 300 kPa is 0.78 of the Prandtl limit: contained yield


def flamant_slope() -> float:
    """Centre settlement per kPa of a flexible strip on a layer (Poulos & Davis), m/kPa."""
    b, h = FOOTING_WIDTH / 2, SOIL_DEPTH
    int_sin = b * math.log(1 + (h / b) ** 2)
    int_alpha = 2 * h * math.atan(b / h) + int_sin
    strain = (1 - CLAY_NU**2) * (int_alpha + int_sin) - CLAY_NU * (1 + CLAY_NU) * (int_alpha - int_sin)
    return strain / (math.pi * CLAY_E)


def check_staged_footing(data: dict[str, Any]) -> list[Check]:
    geostatic = _number(data, "base_reaction_geostatic_kN")
    final = _number(data, "base_reaction_final_kN")
    settlement = _numbers(data, "footing_settlement_m")
    weight = CLAY_GAMMA * SOIL_WIDTH * SOIL_DEPTH
    footing_load = FOOTING_PRESSURE * FOOTING_WIDTH
    checks = [
        _close("base_carries_self_weight", geostatic, weight, TOL_EQUILIBRIUM),
        _close("base_carries_footing_load", final - geostatic, footing_load, TOL_EQUILIBRIUM),
        Check("settlement_history", len(settlement) == FOOTING_STEPS and bool(np.all(np.diff(settlement) > 0))
              and bool(settlement[0] > 0), f"{len(settlement)} values (expected {FOOTING_STEPS}), "
              "positive and increasing downwards"),
    ]
    if not checks[-1].ok:
        return checks
    step = FOOTING_PRESSURE / FOOTING_STEPS
    slope = float(settlement[0]) / step
    checks.append(_close("elastic_slope_vs_flamant", slope, flamant_slope(), TOL_FLAMANT))
    elastic = slope * FOOTING_PRESSURE
    share = (float(settlement[-1]) - elastic) / elastic
    checks.append(Check("clay_yields", share > MIN_PLASTIC_SHARE,
                        f"plastic share {share:.3f}, more than {MIN_PLASTIC_SHARE}"))
    return checks


# --------------------------------------------------------------------------- registry

def _json_oracle(check: Callable[[dict[str, Any]], list[Check]],
                 keys: tuple[str, ...]) -> Callable[[Path], list[Check]]:
    def oracle(work: Path) -> list[Check]:
        data, checks = _read_result(work, keys)
        if data is None:
            return checks
        try:
            return checks + check(data)
        except (TypeError, ValueError) as exc:
            return checks + [Check("result_values", False, f"{type(exc).__name__}: {exc}")]
    return oracle


def _mesh_oracle(check: Callable[[Any], list[Check]]) -> Callable[[Path], list[Check]]:
    def oracle(work: Path) -> list[Check]:
        fem, checks = _load_fem(work)
        if fem is None:
            return checks
        return checks + check(fem)
    return oracle


RESULT_KEYS = {
    "pratt_truss": ("bottom_chord_max_axial_N", "end_post_axial_N", "midspan_uy_m"),
    "frame_modal": ("periods_s", "floor_ux_m"),
    "staged_footing": ("base_reaction_geostatic_kN", "base_reaction_final_kN", "footing_settlement_m"),
}

ORACLES: dict[str, Callable[[Path], list[Check]]] = {
    "plate_with_hole": _mesh_oracle(check_plate_with_hole),
    "footing_mesh": _mesh_oracle(check_footing_mesh),
    "column_on_footing": _mesh_oracle(check_column_on_footing),
    "pratt_truss": _json_oracle(check_pratt_truss, RESULT_KEYS["pratt_truss"]),
    "frame_modal": _json_oracle(check_frame_modal, RESULT_KEYS["frame_modal"]),
    "staged_footing": _json_oracle(check_staged_footing, RESULT_KEYS["staged_footing"]),
}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python graders.py TASK SCRIPT")
    print(json.dumps(grade(sys.argv[1], Path(sys.argv[2])).to_dict(), indent=2))
