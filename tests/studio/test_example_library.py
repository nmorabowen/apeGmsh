"""ADR 0095 Amendment 7 S8a — the oracle-bearing example library.

``example list/show/copy`` is headless (no gmsh/Qt session needed for
the CLI itself — it only reads packaged JSON/Markdown and copies
files). The mesh-smoke lane at the bottom does replay each packaged
example's script through ``ReplayRunner`` up to ``phase="mesh"``
(geometry + ``generate()``, same cheap CI-safe gate as
``tests/examples/test_examples_replay_smoke.py``).

CI honesty (ADR 0095 Amendment 7): this lane's coverage stops at the
mesh gate. Full-solve ``verify.py`` needs openseespy (arch-pushover)
or the Ladruno classic exe (step-load-transient) and is **not** run
on CI — see the PR body for the three local verify transcripts
(INV-22).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
EXAMPLES_DIR = SRC / "apeGmsh" / "studio" / "examples"

EXAMPLE_NAMES = ["arch-pushover", "partitioned-frame", "step-load-transient"]
SCRIPT_BY_EXAMPLE = {
    "arch-pushover": "arch_pushover.py",
    "partitioned-frame": "partitioned_frame.py",
    "step-load-transient": "step_load_transient.py",
}

# INV-20's specimen-name list (same terms as tests/studio/test_habitat_template.py).
SPECIMEN_TERMS = ("shear", "girder", "filler", "steel_connection", "steel-connection")


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


def _iter_source_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_shows_all_three_with_tags() -> None:
    proc = _run_studio(["example", "list"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in EXAMPLE_NAMES:
        assert name in proc.stdout


def test_list_tag_filter() -> None:
    proc = _run_studio(["example", "list", "--tag", "mesh-only"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "partitioned-frame" in proc.stdout
    assert "arch-pushover" not in proc.stdout
    assert "step-load-transient" not in proc.stdout


def test_list_unknown_tag_yields_nothing() -> None:
    proc = _run_studio(["example", "list", "--tag", "not-a-real-tag"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == ""


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_prints_manifest_and_readme() -> None:
    proc = _run_studio(["example", "show", "arch-pushover"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "title:" in proc.stdout
    assert "metrics:" in proc.stdout
    assert "# arch-pushover" in proc.stdout  # README heading


def test_show_unknown_name_errors_nonzero() -> None:
    proc = _run_studio(["example", "show", "not-a-real-example"])
    assert proc.returncode != 0
    assert "unknown example" in (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_lands_runnable_directory(tmp_path: Path) -> None:
    dest = tmp_path / "landed"
    proc = _run_studio(["example", "copy", "arch-pushover", "--dest", str(dest)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / "arch_pushover.py").is_file()
    assert (dest / "manifest.json").is_file()
    assert (dest / "README.md").is_file()
    assert (dest / "verify.py").is_file()


def test_copy_default_dest_is_cwd_name(tmp_path: Path) -> None:
    proc = _run_studio(["example", "copy", "partitioned-frame"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "partitioned-frame" / "partitioned_frame.py").is_file()


def test_copy_refuses_existing_destination(tmp_path: Path) -> None:
    dest = tmp_path / "landed"
    dest.mkdir()
    marker = dest / "keepme.txt"
    marker.write_text("pre-existing\n", encoding="utf-8")

    proc = _run_studio(["example", "copy", "arch-pushover", "--dest", str(dest)])
    assert proc.returncode != 0
    assert "already exists" in (proc.stdout + proc.stderr)
    # No side effects: the pre-existing directory is untouched.
    assert marker.read_text(encoding="utf-8") == "pre-existing\n"
    assert not (dest / "arch_pushover.py").exists()


def test_copy_unknown_name_errors_nonzero(tmp_path: Path) -> None:
    proc = _run_studio(["example", "copy", "not-a-real-example"], cwd=tmp_path)
    assert proc.returncode != 0
    assert "unknown example" in (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# INV-20 grep gate over the PACKAGED examples
# ---------------------------------------------------------------------------


def test_no_specimen_names_in_packaged_examples() -> None:
    pattern = re.compile("|".join(re.escape(t) for t in SPECIMEN_TERMS), re.IGNORECASE)
    hits: dict = {}
    for p in _iter_source_files(EXAMPLES_DIR):
        text = p.read_text(encoding="utf-8")
        matches = [line for line in text.splitlines() if pattern.search(line)]
        if matches:
            hits[str(p.relative_to(EXAMPLES_DIR))] = matches
    assert not hits, f"specimen names leaked into packaged examples: {hits}"


# ---------------------------------------------------------------------------
# manifest schema sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_manifest_schema_sanity(name: str) -> None:
    manifest = json.loads(
        (EXAMPLES_DIR / name / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == 1
    assert manifest["name"] == name
    assert manifest["title"]
    assert manifest["tags"]
    assert manifest["teaches"]
    assert manifest["requires"]
    assert manifest["provenance"]
    assert manifest["metrics"]
    for m in manifest["metrics"]:
        assert m["name"]
        assert "expected" in m
        assert ("tol_abs" in m) or ("tol_rel" in m)


# ---------------------------------------------------------------------------
# Packaged-data completeness (catches a package-data glob gap)
# ---------------------------------------------------------------------------


def test_packaged_examples_reachable_via_importlib_resources() -> None:
    sys.path.insert(0, str(SRC))
    try:
        from apeGmsh.studio._examples import _example_dir, _iter_example_names
    finally:
        sys.path.remove(str(SRC))

    assert set(_iter_example_names()) == set(EXAMPLE_NAMES)

    for name in EXAMPLE_NAMES:
        disk_dir = EXAMPLES_DIR / name
        disk_files = {p.name for p in _iter_source_files(disk_dir)}
        node = _example_dir(name)
        packaged_files = {c.name for c in node.iterdir() if c.is_file()}
        assert packaged_files == disk_files, name
        for fname in disk_files:
            assert (node / fname).read_bytes() == (disk_dir / fname).read_bytes()


# ---------------------------------------------------------------------------
# Mesh-smoke lane (geometry + generate() only — see module docstring)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_example_replays_to_mesh_gate_via_packaged_copy(
    name: str, tmp_path: Path,
) -> None:
    sys.path.insert(0, str(SRC))
    try:
        from apeGmsh.studio._examples import run_copy
    finally:
        sys.path.remove(str(SRC))

    dest = tmp_path / name
    assert run_copy(name, str(dest)) == 0

    from apeGmsh.studio import ReplayRunner

    script = dest / SCRIPT_BY_EXAMPLE[name]
    runner = ReplayRunner(root=dest)
    result = runner.run_until(script, phase="mesh")

    assert result.ok, result.error
    # All three instantiate apeSees(fem) right after the mesh is built
    # (or, for step-load-transient, right after fixed/tip resolution),
    # so the mesh-phase gate always stops there — same convention as
    # tests/examples/test_examples_replay_smoke.py.
    assert result.stopped_at == "apeSees"
