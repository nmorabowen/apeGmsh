"""ADR 0095 S2 / INV-4 — failed replay keeps the last good frame."""
from __future__ import annotations

from pathlib import Path

from apeGmsh.studio import ReplayResult, ReplayRunner
from apeGmsh.studio._replay import _exec_hold_open


def test_failed_replay_keeps_last_good_session(tmp_path: Path) -> None:
    token = object()

    def fake_exec(path: Path, phase: str) -> ReplayResult:
        if path.name.startswith("bad"):
            raise RuntimeError("boom")
        return ReplayResult(
            ok=True,
            phase=phase,
            geometry_hash="abc",
            error=None,
            session=token,
        )

    runner = ReplayRunner(exec_fn=fake_exec)
    good = tmp_path / "good.py"
    good.write_text("pass\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    r1 = runner.run_until(good)
    assert r1.ok
    assert r1.session is token

    r2 = runner.run_until(bad)
    assert r2.ok is False
    assert r2.session is token
    assert r2.error is not None
    assert "boom" in r2.error
    assert runner.last_good is r1 or runner.last_good.session is token


def test_first_failure_has_no_session(tmp_path: Path) -> None:
    def fake_exec(path: Path, phase: str) -> ReplayResult:
        raise SyntaxError("nope")

    runner = ReplayRunner(exec_fn=fake_exec)
    script = tmp_path / "bad.py"
    script.write_text("!!!\n", encoding="utf-8")
    r = runner.run_until(script)
    assert r.ok is False
    assert r.session is None


def test_skip_is_per_phase(tmp_path: Path) -> None:
    calls: list[str] = []
    token = object()

    def fake_exec(path: Path, phase: str) -> ReplayResult:
        calls.append(phase)
        return ReplayResult(
            ok=True,
            phase=phase,
            geometry_hash=None,
            error=None,
            session=token,
        )

    runner = ReplayRunner(exec_fn=fake_exec)
    script = tmp_path / "box.py"
    script.write_text("print(1)\n", encoding="utf-8")
    r1 = runner.run_until(script, phase="model")
    r2 = runner.run_until(script, phase="mesh")
    r3 = runner.run_until(script, phase="mesh")
    assert r1.ok and r2.ok and r3.ok
    assert r2.skipped is False
    assert r3.skipped is True
    assert calls == ["model", "mesh"]


def test_unchanged_hash_skips_remesh(tmp_path: Path) -> None:
    calls = {"n": 0}
    token = object()

    def fake_exec(path: Path, phase: str) -> ReplayResult:
        calls["n"] += 1
        return ReplayResult(
            ok=True,
            phase=phase,
            geometry_hash=None,
            error=None,
            session=token,
        )

    runner = ReplayRunner(exec_fn=fake_exec)
    script = tmp_path / "box.py"
    script.write_text("print(1)\n", encoding="utf-8")
    r1 = runner.run_until(script)
    r2 = runner.run_until(script)
    assert r1.ok and r2.ok
    assert r2.skipped is True
    assert r2.session is token
    assert calls["n"] == 1
    assert r1.geometry_hash == r2.geometry_hash


def test_hold_open_box_script(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "box.py"
    script.write_text(
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='studio_box') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n"
        "    g.physical.from_label('body', name='Body')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner()
    result = runner.run_until(script)
    try:
        assert result.ok, result.error
        assert result.session is not None
        assert result.session.is_active
        assert "body" in result.session.labels.get_all()
    finally:
        if result.session is not None and result.session.is_active:
            from apeGmsh.studio import project
            from apeGmsh.studio._names import lookup_from_gmsh
            import gmsh

            vols = gmsh.model.getEntities(3)
            assert vols
            dim, tag = vols[0]
            rec = lookup_from_gmsh(dim, tag)
            assert "body" in rec.labels
            assert "Body" in rec.physical_groups
            env = project(
                [(dim, tag)],
                phase="model",
                names=lookup_from_gmsh,
            )
            assert env.labels == ("body",)
            assert env.physical_groups == ("Body",)
            result.session.end()


_SOLVE_SCRIPT = """\
from pathlib import Path
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees

root = Path(__file__).resolve().parent
with apeGmsh(model_name='studio_box') as g:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')
    g.physical.from_label('body', name='Body')
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    (root / 'after_generate').write_text('yes', encoding='utf-8')
    fem = g.mesh.queries.get_fem_data(dim=3)
ops = apeSees(fem)
(root / 'after_apesees').write_text('yes', encoding='utf-8')
"""


def test_model_phase_does_not_generate(tmp_path: Path, monkeypatch) -> None:
    import json

    import gmsh

    from apeGmsh.studio._host import session_has_mesh
    from apeGmsh.studio._paths import names_path

    script = tmp_path / "box.py"
    script.write_text(_SOLVE_SCRIPT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner()
    result = runner.run_until(script, phase="model")
    try:
        assert result.ok, result.error
        assert result.stopped_at == "g.mesh.generation.generate"
        assert not (tmp_path / "after_generate").exists()
        assert not (tmp_path / "after_apesees").exists()
        assert session_has_mesh() is False
        manifest = json.loads(names_path(tmp_path).read_text(encoding="utf-8"))
        assert manifest["phase"] == "model"
        assert "body" in manifest["labels"]
        assert "Body" in manifest["physical_groups"]
        assert manifest["counts"]["elements"]["3"] == 0
        vols = [e for e in manifest["entities"] if e["dim"] == 3]
        assert vols
        assert "body" in vols[0]["labels"]
        assert "Body" in vols[0]["physical_groups"]
    finally:
        if result.session is not None and result.session.is_active:
            result.session.end()
        try:
            gmsh.finalize()
        except Exception:
            pass


def test_mesh_phase_generates_but_skips_apesees(tmp_path: Path, monkeypatch) -> None:
    import json

    from apeGmsh.studio._host import session_has_mesh
    from apeGmsh.studio._paths import names_path

    script = tmp_path / "box.py"
    script.write_text(_SOLVE_SCRIPT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner()
    result = runner.run_until(script, phase="mesh")
    try:
        assert result.ok, result.error
        assert result.stopped_at == "apeSees"
        assert (tmp_path / "after_generate").read_text(encoding="utf-8") == "yes"
        assert not (tmp_path / "after_apesees").exists()
        assert session_has_mesh() is True
        manifest = json.loads(names_path(tmp_path).read_text(encoding="utf-8"))
        assert manifest["phase"] == "mesh"
        assert "body" in manifest["labels"]
        assert "Body" in manifest["physical_groups"]
        assert manifest["counts"]["elements"]["3"] > 0
    finally:
        if result.session is not None and result.session.is_active:
            result.session.end()



def test_replay_file_dunder_is_absolute(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "sub"
    nested.mkdir()
    marker = nested / "here.txt"
    script = nested / "s.py"
    script.write_text(
        "from pathlib import Path\n"
        f"Path(r'{marker}').write_text(str(Path(__file__)), encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = _exec_hold_open(Path("sub") / "s.py", "model")
    assert result.ok
    recorded = Path(marker.read_text(encoding="utf-8"))
    assert recorded.is_absolute()
    assert recorded.name == "s.py"
    assert recorded.parent == nested


def test_replay_runs_under_main_guard(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "s.py"
    marker = tmp_path / "ran.txt"
    script.write_text(
        "from pathlib import Path\n"
        f"if __name__ == '__main__':\n"
        f"    Path(r'{marker}').write_text('yes', encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = _exec_hold_open(script, "model")
    assert result.ok
    assert marker.read_text(encoding="utf-8") == "yes"


_POC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "box.py"


def test_poc_fixture_model_phase(tmp_path: Path, monkeypatch) -> None:
    from apeGmsh.studio import last_run, ledger_path

    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner()
    result = runner.run_until(_POC_FIXTURE, phase="model")
    try:
        assert result.ok, result.error
        assert result.stopped_at == "g.mesh.generation.generate"
        rec = last_run(ledger_path(tmp_path))
        assert rec is not None
        assert "body" in rec["labels"]
        assert "Body" in rec["physical_groups"]
        assert rec["counts"]["elements"]["3"] == 0
    finally:
        if result.session is not None and result.session.is_active:
            result.session.end()


def test_replay_imports_siblings_from_script_dir(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
    marker = nested / "out.txt"
    (nested / "s.py").write_text(
        "from pathlib import Path\n"
        "import helper\n"
        f"Path(r'{marker}').write_text(str(helper.VALUE), encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = _exec_hold_open(nested / "s.py", "model")
    assert result.ok, result.error
    assert marker.read_text(encoding="utf-8") == "7"
