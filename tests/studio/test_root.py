"""ADR 0095 S5a — explicit project root (INV-15)."""
from __future__ import annotations

import json
from pathlib import Path

from apeGmsh.studio._mcp import run_until, status
from apeGmsh.studio._mcp_log import append_mcp_call
from apeGmsh.studio._paths import (
    ROOT_ENV,
    ledger_path,
    mcp_calls_path,
    names_path,
    resolve_root,
)
from apeGmsh.studio._replay import ReplayRunner


def test_resolve_root_explicit_wins(tmp_path: Path, monkeypatch) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / ".apegmsh").mkdir()
    monkeypatch.setenv(ROOT_ENV, str(other))
    monkeypatch.chdir(tmp_path)
    assert resolve_root(tmp_path) == tmp_path.resolve()


def test_resolve_root_env_then_ancestor(tmp_path: Path, monkeypatch) -> None:
    proj = tmp_path / "proj"
    nested = proj / "src"
    nested.mkdir(parents=True)
    (proj / ".apegmsh").mkdir()
    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.chdir(nested)
    assert resolve_root() == proj.resolve()

    env_root = tmp_path / "env_proj"
    env_root.mkdir()
    monkeypatch.setenv(ROOT_ENV, str(env_root))
    assert resolve_root() == env_root.resolve()


def test_resolve_root_skips_home_habitat(tmp_path: Path, monkeypatch) -> None:
    """A stale ~/ .apegmsh must not capture an unrelated cwd."""
    # tmp_path lives under the real home on this machine; INV-15 must
    # skip Path.home()/.apegmsh when start is not home itself.
    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_root() == tmp_path.resolve()
    assert resolve_root() != Path.home().resolve()


def test_resolve_root_cwd_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_root() == tmp_path.resolve()


def test_resolve_root_blank_explicit_and_env_unset(
    tmp_path: Path, monkeypatch
) -> None:
    """Empty / whitespace root= and APEGMSH_ROOT fall through (S5f)."""
    proj = tmp_path / "proj"
    nested = proj / "src"
    nested.mkdir(parents=True)
    (proj / ".apegmsh").mkdir()
    monkeypatch.setenv(ROOT_ENV, "   ")
    monkeypatch.chdir(nested)
    assert resolve_root("") == proj.resolve()
    assert resolve_root("  ") == proj.resolve()
    monkeypatch.delenv(ROOT_ENV, raising=False)
    assert resolve_root("") == proj.resolve()


def test_cli_blank_root_falls_through(tmp_path: Path, monkeypatch, capsys) -> None:
    """``--root ''`` must not bind cwd via Path('') → '.' (Opus S5f)."""
    from apeGmsh.studio.__main__ import main

    proj = tmp_path / "proj"
    nested = proj / "src"
    nested.mkdir(parents=True)
    (proj / ".apegmsh").mkdir()
    (proj / ".apegmsh" / "names.json").write_text(
        '{"schema":1,"phase":"model","labels":["body"],'
        '"physical_groups":[],"counts":null,"entities":[]}\n',
        encoding="utf-8",
    )
    monkeypatch.delenv(ROOT_ENV, raising=False)
    monkeypatch.chdir(nested)
    assert main(["--status", "--root", "", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["root"]).resolve() == proj.resolve()
    assert payload["names"]["labels"] == ["body"]


def test_resolve_root_home_when_start_is_home(
    tmp_path: Path, monkeypatch
) -> None:
    """Only when start *is* home may ~/ .apegmsh win."""
    home = Path.home().resolve()
    monkeypatch.delenv(ROOT_ENV, raising=False)
    if not (home / ".apegmsh").is_dir():
        return
    assert resolve_root(start=home) == home


def test_two_roots_isolated(tmp_path: Path, monkeypatch) -> None:
    """One process, two roots: status(A) never sees B's ledger."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    script_a = a / "box.py"
    script_b = b / "box.py"
    src = (
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='studio_box') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n"
        "    g.physical.from_label('body', name='Body')\n"
    )
    script_a.write_text(src, encoding="utf-8")
    script_b.write_text(src, encoding="utf-8")
    # Process cwd is neither project — root= must carry the habitat.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ROOT_ENV, raising=False)

    ra = ReplayRunner(root=a).run_until(script_a, phase="model", root=a)
    rb = ReplayRunner(root=b).run_until(script_b, phase="model", root=b)
    try:
        assert ra.ok, ra.error
        assert rb.ok, rb.error
        assert names_path(a).is_file()
        assert names_path(b).is_file()
        assert not names_path(tmp_path).is_file()
        sa = status(root=a)
        sb = status(root=b)
        assert sa["empty"] is False
        assert sb["empty"] is False
        assert Path(sa["root"]).resolve() == a.resolve()
        assert Path(sb["root"]).resolve() == b.resolve()
        assert sa["n_runs"] == 1
        assert sb["n_runs"] == 1
        assert ledger_path(a) != ledger_path(b)
        rec_a = sa["last_run"]
        assert rec_a is not None
        assert Path(rec_a["root"]).resolve() == a.resolve()
        assert Path(rec_a["cwd"]).resolve() == a.resolve()
    finally:
        for r in (ra, rb):
            if r.session is not None and r.session.is_active:
                r.session.end()


def test_replay_habitat_not_exec_cwd(tmp_path: Path, monkeypatch) -> None:
    """names/ledger land under root even when script lives elsewhere."""
    root = tmp_path / "habitat"
    scripts = tmp_path / "scripts"
    root.mkdir()
    scripts.mkdir()
    script = scripts / "box.py"
    script.write_text(
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='studio_box') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ROOT_ENV, raising=False)
    result = ReplayRunner().run_until(script, phase="model", root=root)
    try:
        assert result.ok, result.error
        assert names_path(root).is_file()
        assert ledger_path(root).is_file()
        assert not (scripts / ".apegmsh").exists()
        assert not (tmp_path / ".apegmsh" / "names.json").exists()
        rec = json.loads(
            ledger_path(root).read_text(encoding="utf-8").strip().splitlines()[-1]
        )
        assert Path(rec["root"]).resolve() == root.resolve()
        assert Path(rec["cwd"]).resolve() == scripts.resolve()
    finally:
        if result.session is not None and result.session.is_active:
            result.session.end()


def test_mcp_calls_follow_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ROOT_ENV, raising=False)
    proj = tmp_path / "proj"
    proj.mkdir()
    path = append_mcp_call("status", {}, {"empty": True}, root=proj)
    assert path is not None
    assert path == mcp_calls_path(proj)
    assert path.is_file()
    assert path.parent == proj / ".apegmsh"


def test_run_until_passes_root_flag(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    recorded: dict = {}

    def fake_run(argv, **kwargs):
        recorded["argv"] = list(argv)
        recorded["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(returncode=0, stdout="replay ok\n", stderr="")

    monkeypatch.setattr("apeGmsh.studio._mcp.subprocess.run", fake_run)
    result = run_until(script, phase="mesh", root=tmp_path)
    assert result["ok"] is True
    assert result["root"] == str(tmp_path.resolve())
    argv = recorded["argv"]
    assert "--root" in argv
    assert str(tmp_path.resolve()) in argv
