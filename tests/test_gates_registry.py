"""internal_docs/program/gates.md must list every check_quirks.py rule (scripts/gates_registry.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("gates_registry", ROOT / "scripts" / "gates_registry.py")
gates_registry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gates_registry
_spec.loader.exec_module(gates_registry)


def test_gates_md_is_current(capsys):
    code = gates_registry.main(["--check"])
    assert code == 0, capsys.readouterr().out


def test_every_rule_has_a_row():
    text = (ROOT / gates_registry.GATES).read_text(encoding="utf-8")
    for rule in gates_registry._load_quirks().RULES:
        assert f"| quirk `{rule}` |" in text


def test_hand_deleted_quirk_row_fails_check(tmp_path, monkeypatch, capsys):
    text = (ROOT / gates_registry.GATES).read_text(encoding="utf-8")
    kept = [ln for ln in text.splitlines() if not ln.startswith("| quirk `vtk-import`")]
    assert len(kept) == len(text.splitlines()) - 1
    fake = tmp_path / "gates.md"
    fake.write_text("\n".join(kept) + "\n", encoding="utf-8")
    monkeypatch.setattr(gates_registry, "GATES", fake)
    assert gates_registry.main(["--check"]) == 1
    assert "missing row: quirk `vtk-import`" in capsys.readouterr().out


def test_edited_waiver_count_fails_check(tmp_path, monkeypatch):
    text = (ROOT / gates_registry.GATES).read_text(encoding="utf-8")
    lines = text.splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("| quirk `adr-number`"))
    cells = lines[i].split(" | ")
    cells[-2] = "9"
    lines[i] = " | ".join(cells)
    fake = tmp_path / "gates.md"
    fake.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(gates_registry, "GATES", fake)
    assert gates_registry.main(["--check"]) == 1
