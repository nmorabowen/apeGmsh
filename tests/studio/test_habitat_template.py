"""ADR 0095 Amendment 6 S7a — habitat template package data + `studio init`.

Headless (no gmsh/Qt session needed): the init verb only copies files,
substitutes placeholders, and shells out to the target habitat's own
``scripts/check_template.py``. Subprocess calls use ``sys.executable``
with ``PYTHONPATH`` pinned to this worktree's ``src`` — see the module
docstring note in ``apeGmsh/studio/_init_habitat.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
TEMPLATE_DIR = SRC / "apeGmsh" / "studio" / "template"


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["APEGMSH_QUIET"] = "1"
    env["LADRUNO_OPENSEES_QUIET"] = "1"
    return env


def _run_studio(args: list, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "apeGmsh.studio", *args],
        cwd=str(cwd) if cwd is not None else None,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )


def _run_check_template(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(target / "scripts" / "check_template.py"), "--strict"],
        cwd=str(target),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )


def _iter_source_files(root: Path) -> Iterable[str]:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p.relative_to(root).as_posix()


def _snapshot(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _init(target: Path, *, name: str = "demo-habitat", model: str = "demo"):
    return _run_studio(["init", "--name", name, "--model", model, "--root", str(target)])


# ---------------------------------------------------------------------------
# init -> the habitat's own check_template.py --strict passes
# ---------------------------------------------------------------------------


def test_init_then_strict_check_passes(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    check = _run_check_template(target)
    assert check.returncode == 0, check.stdout + check.stderr
    assert "TEMPLATE OK" in check.stdout


def test_init_creates_model_lineage(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target, model="beam")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (target / "models" / "beam" / "src" / "README.md").is_file()
    assert (target / "models" / "beam" / "cases").is_dir()


# ---------------------------------------------------------------------------
# Re-init is refused, with no side effects
# ---------------------------------------------------------------------------


def test_reinit_refused_no_side_effects(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    first = _init(target)
    assert first.returncode == 0, first.stdout + first.stderr

    before = _snapshot(target)
    second = _init(target, name="other-habitat", model="other")
    assert second.returncode != 0
    assert "already initialized" in (second.stdout + second.stderr)

    after = _snapshot(target)
    assert before == after


def test_init_refuses_non_empty_conflicting_target(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    target.mkdir()
    (target / "ape.project.yaml").write_text("not a habitat\n", encoding="utf-8")

    proc = _init(target)
    assert proc.returncode != 0
    assert "error" in (proc.stdout + proc.stderr).lower()
    # No side effects: the conflicting file is untouched, nothing else copied.
    assert (target / "ape.project.yaml").read_text(encoding="utf-8") == "not a habitat\n"
    assert not (target / "APE").exists()


# ---------------------------------------------------------------------------
# Substitution completeness
# ---------------------------------------------------------------------------


def test_no_placeholders_remain_after_init(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    leftover = []
    for p in target.rglob("*"):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "__HABITAT_NAME__" in text or "__MODEL_ID__" in text:
            leftover.append(str(p.relative_to(target)))
    assert leftover == []


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


def test_status_cold_start_exits_2(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    status = _run_studio(["--status", "--root", str(target)])
    assert status.returncode == 2
    assert "no .apegmsh state" in status.stdout


# ---------------------------------------------------------------------------
# INV-20 grep gate over the PACKAGED template
# ---------------------------------------------------------------------------

# Specimen names from the steel-connection extraction the template was
# lifted from. Allowed only in TEMPLATE.md's provenance note and the
# explicit "canonical worked example" pointer lines in the root/APE
# READMEs and the two instruction files that reference it by name.
SPECIMEN_TERMS = ("shear", "girder", "filler", "steel_connection", "steel-connection")
SPECIMEN_ALLOWED_FILES = {
    "TEMPLATE.md",
    "README.md",
    "APE/README.md",
    "APE/instructions/analysis-profiling.md",
    "APE/instructions/how-we-work.md",
    "APE/instructions/socratic-geometry.md",
}

# The template author's own personal skill suites (INV-20): named only in
# APE/skills/README.md's "No skill catalog ships with this template"
# section (+ its own harvest.py classifier), and as non-gating profiling
# guidance in analysis-profiling.md / continuous-improvement.md /
# postmortem/templates/01_metrics.md ("feed hotspots into ..." — never a
# require/gate, just where a human might later look).
SKILL_SUITE_TERMS = ("abaqus-theory", "aci-concrete", "opensees-performance")
SKILL_SUITE_ALLOWED_FILES = {
    "APE/instructions/analysis-profiling.md",
    "APE/instructions/continuous-improvement.md",
    "APE/skills/harvest.py",
    "APE/skills/README.md",
    "postmortem/templates/01_metrics.md",
}


def _grep_packaged_template(terms: tuple) -> dict:
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    hits: dict = {}
    for rel in _iter_source_files(TEMPLATE_DIR):
        text = (TEMPLATE_DIR / rel).read_text(encoding="utf-8")
        matches = [line for line in text.splitlines() if pattern.search(line)]
        if matches:
            hits[rel] = matches
    return hits


def test_inv20_specimen_names_confined_to_provenance_and_pointers() -> None:
    hits = _grep_packaged_template(SPECIMEN_TERMS)
    stray = set(hits) - SPECIMEN_ALLOWED_FILES
    assert not stray, (
        "specimen content leaked outside TEMPLATE.md / worked-example "
        f"pointer lines: {sorted(stray)}"
    )


def test_inv20_skill_suite_names_confined_to_personal_list() -> None:
    hits = _grep_packaged_template(SKILL_SUITE_TERMS)
    stray = set(hits) - SKILL_SUITE_ALLOWED_FILES
    assert not stray, (
        "personal skill-suite names leaked outside the optional-use "
        f"README guidance: {sorted(stray)}"
    )


def test_inv20_skills_readme_carries_agent_guidance() -> None:
    text = (TEMPLATE_DIR / "APE" / "skills" / "README.md").read_text(encoding="utf-8")
    assert "not distributed" in text or "not required" in text
    assert "if any of these skills happen to be installed" in text.lower() or (
        "if not, ignore" in text.lower()
    )


# ---------------------------------------------------------------------------
# Packaged-data completeness (catches a package-data glob gap)
# ---------------------------------------------------------------------------


def test_packaged_template_reachable_via_importlib_resources() -> None:
    sys.path.insert(0, str(SRC))
    try:
        from apeGmsh.studio._init_habitat import _iter_template_files
    finally:
        sys.path.remove(str(SRC))

    disk_files = set(_iter_source_files(TEMPLATE_DIR))
    assert disk_files, "template package dir is unexpectedly empty"

    packaged_files = {rel for rel, _ in _iter_template_files()}
    assert packaged_files == disk_files


def test_dotfile_renames_ship_under_safe_names() -> None:
    assert (TEMPLATE_DIR / "dot.gitignore").is_file()
    assert (TEMPLATE_DIR / "cursor" / "mcp.json").is_file()
    # And the real dotfile names are NOT what's shipped (that's the hazard
    # this rename works around — a literal nested .gitignore would apply
    # its rules to this subtree of apeGmsh's own repo).
    assert not (TEMPLATE_DIR / ".gitignore").exists()
    assert not (TEMPLATE_DIR / ".cursor").exists()


def test_init_restores_real_dotfile_names(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (target / ".gitignore").is_file()
    assert (target / ".cursor" / "mcp.json").is_file()


# ---------------------------------------------------------------------------
# Visual-talk convention (apeCAD / apeSketch) — ADR: template requisites
# ---------------------------------------------------------------------------


def test_socratic_geometry_documents_ask_protocol() -> None:
    text = (TEMPLATE_DIR / "APE" / "instructions" / "socratic-geometry.md").read_text(
        encoding="utf-8"
    )
    assert "tools/apeCAD/open_interface.py" in text
    assert "tools/apeSketch/open_interface.py" in text
    assert "tools/apeSketch/human_sketches/" in text
    assert "agentic_sketches/" in text
    assert "Document.to_json()" in text
    assert "to_frame()" in text


def test_ape_project_yaml_documents_requisites() -> None:
    text = (TEMPLATE_DIR / "ape.project.yaml").read_text(encoding="utf-8")
    assert "requires:" in text
    assert "apeCAD.git" in text
    assert "apeSketch.git" in text


def test_init_preserves_requisites_and_still_passes_strict(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    proc = _init(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    manifest = (target / "ape.project.yaml").read_text(encoding="utf-8")
    assert "requires:" in manifest
    assert "apeCAD.git" in manifest
    assert "apeSketch.git" in manifest

    check = _run_check_template(target)
    assert check.returncode == 0, check.stdout + check.stderr
    assert "TEMPLATE OK" in check.stdout
