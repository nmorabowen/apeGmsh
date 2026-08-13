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


def test_hold_open_box_script(tmp_path: Path) -> None:
    script = tmp_path / "box.py"
    script.write_text(
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='studio_box') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n"
        "    g.physical.from_label('body', name='Body')\n",
        encoding="utf-8",
    )
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
