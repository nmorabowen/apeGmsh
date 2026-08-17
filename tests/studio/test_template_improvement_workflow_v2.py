"""Template improvement-workflow v2 — promotion lanes, proof gate, drift
tool (see `APE/instructions/continuous-improvement.md`'s "Promotion
lanes" section and `APE/instructions/session-postmortem.md` step 6).

Headless: `template_drift.py` and `start_session.py`'s backlog parser only
touch the filesystem, no gmsh/Qt/OpenSees needed. Subprocess calls use
``sys.executable`` with ``PYTHONPATH`` pinned to this worktree's ``src``
(same convention as ``test_habitat_template.py``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"


def _subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["APEGMSH_QUIET"] = "1"
    env["LADRUNO_OPENSEES_QUIET"] = "1"
    return env


def _init(target: Path, *, name: str = "demo-habitat", model: str = "demo"):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "apeGmsh.studio",
            "init",
            "--name",
            name,
            "--model",
            model,
            "--root",
            str(target),
        ],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )


def _run_drift(target: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(target / "scripts" / "template_drift.py")],
        cwd=str(target),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# template_drift.py
# ---------------------------------------------------------------------------


def test_fresh_habitat_has_no_modified_drift(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    drift = _run_drift(target)
    assert drift.returncode == 0, drift.stdout + drift.stderr
    out = drift.stdout

    # Fresh habitat == template, except brief.md carries the substituted
    # __HABITAT_NAME__/__MODEL_ID__ placeholders — expected memory drift,
    # not a Lane B candidate.
    assert "MODIFIED — habitat differs from template (Lane B candidates) (0)" in out
    assert "(memory — expected drift, not candidates)" in out
    assert "APE/memory/brief.md" in out


def test_modified_instruction_file_appears_under_modified(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    how_we_work = target / "APE" / "instructions" / "how-we-work.md"
    how_we_work.write_text(
        how_we_work.read_text(encoding="utf-8") + "\nHabitat-specific addition.\n",
        encoding="utf-8",
    )

    drift = _run_drift(target)
    assert drift.returncode == 0, drift.stdout + drift.stderr
    out = drift.stdout
    assert "MODIFIED — habitat differs from template (Lane B candidates) (1)" in out
    assert "APE/instructions/how-we-work.md" in out


# ---------------------------------------------------------------------------
# start_session.py — backlog burn-down parser
# ---------------------------------------------------------------------------


def _import_start_session(target: Path):
    """Import ``start_session`` scoped to ``target`` habitat (its module-
    level HABITAT/BACKLOG_FILE constants resolve from ``target/scripts``)."""
    scripts_dir = str(target / "scripts")
    for mod in ("start_session", "_habitat"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, scripts_dir)
    try:
        import start_session  # type: ignore
    finally:
        sys.path.remove(scripts_dir)
    return start_session


def test_backlog_summary_tolerates_missing_file(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    backlog = target / "postmortem" / "backlog" / "open.md"
    backlog.unlink()

    mod = _import_start_session(target)
    assert mod._print_backlog_summary() is None  # must not raise


def test_backlog_summary_finds_planted_p0(tmp_path: Path, capsys) -> None:
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    backlog = target / "postmortem" / "backlog" / "open.md"
    backlog.write_text(
        "# Open friction backlog (living)\n\n"
        "## Open\n\n"
        "| ID | Target | Proposal | P | Proof (next postmortem) | Evidence (session) | Notes |\n"
        "|----|--------|----------|---|-------------------------|--------------------|-------|\n"
        "| B1 | studio | MCP root drifts on reload | P0 | status.root stable across 3 starts | session-2026-01-01 | |\n"
        "| B2 | skills | slow door open | P2 | n/a | | |\n\n"
        "## Done\n\n"
        "| ID | Target | Resolution | Date |\n"
        "|----|--------|------------|------|\n"
        "| | | | |\n",
        encoding="utf-8",
    )

    mod = _import_start_session(target)
    items = mod._parse_backlog_open_items(backlog)
    assert ("P0", "MCP root drifts on reload") in items
    assert all(priority != "P2" for priority, _ in items)

    capsys.readouterr()
    mod._print_backlog_summary()
    out = capsys.readouterr().out
    assert "backlog: 1 open P0/P1" in out
    assert "MCP root drifts on reload" in out


# ---------------------------------------------------------------------------
# start_session.py — visual-talk requisite check (apeCAD / apeSketch)
# ---------------------------------------------------------------------------


def test_visual_talk_requisite_warns_when_import_fails(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    mod = _import_start_session(target)

    def _boom(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}")

    line = mod._check_visual_talk_requisite("apeCAD", "apeCAD", import_func=_boom)
    assert line.startswith("WARN:")
    assert "apeCAD" in line
    assert "pip install git+https://github.com/nmorabowen/apeCAD.git" in line


def test_visual_talk_requisite_ok_when_import_succeeds(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    mod = _import_start_session(target)

    def _present(name: str):
        return object()

    line = mod._check_visual_talk_requisite("apeSketch", "apeSketch", import_func=_present)
    assert line == "OK: apeSketch import OK"


# ---------------------------------------------------------------------------
# start_session.py — `studio init` availability (ADR 0095 Amendment 6, S7a)
# ---------------------------------------------------------------------------


def test_studio_init_guard_warns_when_module_is_absent(tmp_path: Path) -> None:
    """A checkout older than S7a has apeGmsh.studio but no _init_habitat.
    Say so, instead of leaving `studio init` to fail later with a bare
    ModuleNotFoundError that reads like a bug."""
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    mod = _import_start_session(target)

    def _boom(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}")

    line = mod._check_studio_init_available(import_func=_boom)
    assert line.startswith("WARN:")
    assert "studio init" in line
    assert "S7a" in line


def test_studio_init_guard_ok_on_a_current_checkout(tmp_path: Path) -> None:
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    mod = _import_start_session(target)
    # Real import against this worktree — _init_habitat is what ran `init`.
    assert mod._check_studio_init_available() == "OK: studio init available"


def test_print_visual_talk_requisites_covers_both_packages(tmp_path: Path, capsys) -> None:
    target = tmp_path / "habitat"
    init = _init(target)
    assert init.returncode == 0, init.stdout + init.stderr

    mod = _import_start_session(target)

    capsys.readouterr()
    mod._print_visual_talk_requisites()
    out = capsys.readouterr().out
    assert "apeCAD" in out
    assert "apeSketch" in out
