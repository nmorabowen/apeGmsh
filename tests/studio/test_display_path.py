"""ADR 0095 S5i — root-relative display paths."""
from __future__ import annotations

from pathlib import Path

from apeGmsh.studio._ledger import make_record
from apeGmsh.studio._paths import display_path, resolve_under


def test_display_path_relative_under_root(tmp_path: Path) -> None:
    script = tmp_path / "src" / "box.py"
    script.parent.mkdir()
    script.write_text("pass\n", encoding="utf-8")
    assert display_path(script, tmp_path) == "src/box.py"
    assert resolve_under(tmp_path, "src/box.py") == script.resolve()


def test_display_path_absolute_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / "other" / "box.py"
    outside.parent.mkdir()
    habitat = tmp_path / "habitat"
    habitat.mkdir()
    assert display_path(outside, habitat) == str(outside.resolve())


def test_make_record_stores_relative_script(tmp_path: Path) -> None:
    script = tmp_path / "box.py"
    script.write_text("pass\n", encoding="utf-8")
    rec = make_record(
        script=script,
        phase="model",
        ok=True,
        root=tmp_path,
        cwd=tmp_path,
    )
    assert rec["script"] == "box.py"
    assert rec["cwd"] in (".", "")
    assert Path(rec["root"]).resolve() == tmp_path.resolve()
