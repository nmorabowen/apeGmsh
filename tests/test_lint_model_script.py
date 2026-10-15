"""Self-test for the advisory model-script lint (skills/apegmsh/scripts/lint_model_script.py).

Three layers, as the R4 card asks: known-bad control scripts from the readability
workshop must be flagged, the three reference scripts banked in ``model-scripts.md``
must be clean on S1/S3/V5/T1/T2, and one positive and one negative snippet per rule.
Stdlib only; nothing is imported from apeGmsh or run.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINT = ROOT / "skills" / "apegmsh" / "scripts" / "lint_model_script.py"
DOC = ROOT / "skills" / "apegmsh" / "references" / "model-scripts.md"
LOOPS = ROOT / "internal_docs" / "readability"

_spec = importlib.util.spec_from_file_location("lint_model_script", LINT)
lint = importlib.util.module_from_spec(_spec)
sys.modules["lint_model_script"] = lint
_spec.loader.exec_module(lint)

FENCE = re.compile(
    r"<!-- reference-script: (?P<name>[\w.]+) -->\n```python\n(?P<body>.*?)```",
    re.DOTALL,
)
REFERENCES = {m["name"]: m["body"] for m in FENCE.finditer(DOC.read_text(encoding="utf-8"))}


def rules_hit(text: str, rule: str) -> list[int]:
    """Lines where ``rule`` fires on a snippet."""
    return [f.line for f in lint.lint_text(textwrap.dedent(text), (rule,))]


def findings_in(path: Path, rule: str) -> list:
    return lint.lint_text(path.read_text(encoding="utf-8"), (rule,))


# ------------------------------------------------------------ known bad

def test_loop3_f_flags_absolute_path_unit_literal_and_fstring_arithmetic() -> None:
    path = LOOPS / "loop3" / "f.py"
    assert findings_in(path, "S3")
    assert findings_in(path, "V5")
    assert any("f-string" in f.message for f in findings_in(path, "T1"))


def test_loop1_e_flags_expected_results_block() -> None:
    claims = [f for f in findings_in(LOOPS / "loop1" / "e.py", "T4") if "Expected" in f.message]
    assert len(claims) >= 2          # the ~300 kN and ~200 kN lines


def test_loop1_b_flags_geometry_handles_and_ids_index() -> None:
    messages = [f.message for f in findings_in(LOOPS / "loop1" / "b.py", "V1")]
    assert any("handle" in m for m in messages)
    assert any(".ids[" in m for m in messages)


def test_loop1_b_flags_mid_script_import() -> None:
    assert 86 in [f.line for f in findings_in(LOOPS / "loop1" / "b.py", "S1")]


# ------------------------------------------------------------ known good

def test_reference_scripts_were_extracted() -> None:
    assert set(REFERENCES) == {"pratt_truss.py", "frame_modal.py", "staged_footing.py"}


@pytest.mark.parametrize("name", ["pratt_truss.py", "frame_modal.py", "staged_footing.py"])
def test_reference_scripts_are_clean(name: str) -> None:
    findings = lint.lint_text(REFERENCES[name], ("S1", "S3", "V5", "T1", "T2", "R1"))
    assert findings == [], [(f.line, f.rule, f.message) for f in findings]


def test_reference_scripts_use_no_raw_ops() -> None:
    # ADR 0115 R2: live read-back is `.read`, so no reference script needs raw openseespy.
    for name, body in REFERENCES.items():
        assert "raw-ops:" not in body and "ops_raw" not in body, name


# ------------------------------------------------------------ synthetic

def test_s1() -> None:
    assert rules_hit("import os\nx = 1\nimport sys\n", "S1") == [3]
    assert rules_hit('"""doc"""\nimport matplotlib\nmatplotlib.use("Agg")\nimport numpy\nx = 1\n', "S1") == []
    assert rules_hit("x = 1\nif x:\n    import os\n", "S1") == [3]


def test_s2() -> None:
    assert rules_hit("def helper():\n    pass\n", "S2") == [1]
    assert rules_hit("x = 1\n", "S2") == []


def test_s3() -> None:
    assert rules_hit('out = "C:\\\\runs\\\\out"\n', "S3") == [1]
    assert rules_hit('out = "/home/me/out"\n', "S3") == [1]
    assert rules_hit('out = Path(__file__).parent / "out"\nlabel = "a / b"\n', "S3") == []


def test_v1() -> None:
    assert rules_hit("p = geo.add_point(0, 0, 0)\n", "V1") == [1]
    assert rules_hit('geo.add_point(0, 0, 0, label="a")\n', "V1") == []
    assert rules_hit("n = fem.nodes.select(pg='a').ids[0]\n", "V1") == [1]
    assert rules_hit("ops.fix(nodes=[1, 2], dofs=(1, 1))\n", "V1") == [1]
    assert rules_hit("ops.fix(pg='base', dofs=(1, 1))\n", "V1") == []


def test_v3() -> None:
    # Maintainer ruling on #2006 (Q3, #1197): a box promoted to a group is the
    # second choice after a label, so in_box is legal; hand matching is not.
    assert rules_hit('g.model.select(None, dim=2).in_box(LO, HI).to_physical("Base")\n', "V3") == []
    assert rules_hit("n = fem.nodes.select(pg='Body').in_box(LO, HI).ids\n", "V3") == []
    assert rules_hit("m = np.isclose(y, 0.0)\n", "V3") == [1]
    assert rules_hit("for p in fem.nodes.coords:\n    pass\n", "V3") == [1]
    assert rules_hit("for p in fem.nodes.select(pg='a').ids:\n    pass\n", "V3") == []


def test_v4() -> None:
    assert rules_hit("ops.test.NormDispIncr(tol=1e-8, max_iter=5)\n", "V4") == [1]
    assert rules_hit("ops.test.NormDispIncr(tol=TOL, max_iter=MAX_ITER)\nops.model(ndm=2, ndf=2)\n", "V4") == []
    assert rules_hit("E_STEEL = 200e9\n", "V4") == [1]
    assert rules_hit("E_STEEL = 200e9  # Pa, Young's modulus\n", "V4") == []
    assert rules_hit("# Pa, Young's modulus\nE_STEEL = 200e9\n", "V4") == []
    assert rules_hit("a = 1; b = 2\n", "V4") == [1]
    assert rules_hit("ax.set_xlim(0.0, 12.5)\nprint(round(x, 3))\n", "V4") == []


def test_v5() -> None:
    assert rules_hit("mm = x * 1000\n", "V5") == [1]
    assert rules_hit("kn = x / 1e3\n", "V5") == [1]
    assert rules_hit("mm = x / MM\nKN = 1e3\n", "V5") == []


def test_t1() -> None:
    assert rules_hit('print(f"{x * 2:.1f}")\n', "T1") == [1]
    assert rules_hit('print(f"{x / KN:.1f} {y:.2f}")\n', "T1") == []
    assert rules_hit("r = [[a for a in row] for row in rows]\n", "T1") == [1]
    assert rules_hit("r = [a for a in row]\n", "T1") == []
    assert rules_hit("x = 1  # " + "y" * 120 + "\n", "T1") == [1]


def test_t2() -> None:
    assert rules_hit("import openseespy.opensees as ops\n", "T2") == [1]
    assert rules_hit("from openseespy import opensees\n", "T2") == [1]
    assert rules_hit("import openseespy.opensees as ops_raw\nops_raw.wipe()\n", "T2") == [2]
    ok = "import openseespy.opensees as ops_raw\nops_raw.wipe()  # raw-ops: flush the recorders\n"
    assert rules_hit(ok, "T2") == []
    above = "import openseespy.opensees as ops_raw\n# raw-ops: flush the recorders\nops_raw.wipe()\n"
    assert rules_hit(above, "T2") == []


def test_t4() -> None:
    assert rules_hit("ops.analyze(steps=1)\nassert True\n", "T4") == [1]
    assert rules_hit("status = ops.analyze(steps=1)\nassert True\n", "T4") == [1]
    assert rules_hit("status = ops.analyze(steps=1)\nassert status == 0\n", "T4") == []
    assert rules_hit("x = 1\n", "T4") == [1]                      # no assert at all
    assert rules_hit('"""Expected tension ~300 kN."""\nassert True\n', "T4") == [1]
    assert rules_hit('"""Hand check against the run."""\nassert True\n', "T4") == []


# ------------------------------------------------------------ CLI and mirror

def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(LINT), *args], capture_output=True, text=True)


def test_cli_exit_codes_output_format_and_rule_selection(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\nimport os\nmm = x * 1000\n", encoding="utf-8")
    result = run_cli(str(bad))
    assert result.returncode == 1
    assert f"{bad}:2: S1 " in result.stdout and f"{bad}:3: V5 " in result.stdout

    only_v5 = run_cli("--rules", "V5", str(bad))
    assert "S1" not in only_v5.stdout and "V5" in only_v5.stdout

    clean = tmp_path / "clean.py"
    clean.write_text("x = 1\n", encoding="utf-8")
    assert run_cli("--rules", "S1,V5", str(clean)).returncode == 0
    assert run_cli("--rules", "NOPE", str(clean)).returncode == 2


def test_skill_mirror_carries_the_lint() -> None:
    mirror = ROOT / ".claude" / "skills" / "apegmsh-helper" / "scripts" / "lint_model_script.py"
    assert mirror.read_bytes() == LINT.read_bytes()
    registry = mirror.with_name("removed_symbols.json")      # R1 reads it beside the lint
    assert registry.read_text(encoding="utf-8") == \
        LINT.with_name("removed_symbols.json").read_text(encoding="utf-8")


# ------------------------------------------------------------ R1: removed symbols

def test_r1_flags_removed_symbols_through_aliases_and_imports() -> None:
    script = """\
from apeGmsh import apeGmsh, FEMData, Assembly
from apeGmsh.assembly import Assembly as Asm
from apeGmsh.interop.etabs_import import emit_mass
from apeGmsh.results.session import PlotView
with apeGmsh(model_name="m") as session:
    session.opensees.ingest()
    snap = session.mesh.queries.get_fem_data(dim=3)
ids = snap.nodes.get_ids(pg="Base")
FEMData.compose([])
Asm("a").add("part.h5")
plot = PlotView("p1", kind="path")
"""
    assert rules_hit(script, "R1") == [1, 3, 6, 8, 9, 10, 11]


def test_r1_leaves_live_symbols_alone() -> None:
    script = """\
from apeGmsh import apeGmsh
from apeGmsh.assembly import Assembly
from apeGmsh.results.session import PlotView
with apeGmsh(model_name="m") as g:
    g.masses.volume("Body", density=DENSITY)
    fem = g.mesh.queries.get_fem_data(dim=3)
ids = fem.nodes.select(pg="Base").ids
ops = Assembly("a").instance("b", "b.h5").bridge(ndm=3, ndf=3)
plot = PlotView("p1", kind="xy")
path = "out"
opensees = ops.mass_from_model()
"""
    assert rules_hit(script, "R1") == []


@pytest.mark.parametrize("entry", lint.load_registry(), ids=lambda e: e["symbol"])
def test_r1_flags_every_registry_entry(entry: dict) -> None:
    if entry["kind"] == "member":
        script = f'x = {entry["call"]}("p", {entry["keyword"]}="{entry["symbol"]}")\n'
    elif "from" in entry:
        script = f"from {entry['from']} import {entry['symbol'].split('.')[0]}\nx = {entry['symbol']}\n"
    elif "import" in entry:
        script = f"import {entry['import']}\nx = {entry['symbol']}\n"
    else:
        script = f"x = {entry['symbol']}\n"
    assert rules_hit(script, "R1"), script
    for spelling in entry.get("also", []):
        assert rules_hit(f"x = {spelling}\n", "R1"), spelling


def test_r1_never_skips_without_the_registry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(lint, "REGISTRY_FILE", tmp_path / "removed_symbols.json")
    with pytest.raises(lint.RegistryError, match="R1 never skips"):
        lint.lint_text("x = g.opensees\n", ("R1",))
    lonely = tmp_path / "lint_model_script.py"           # the lint copied without its data
    lonely.write_bytes(LINT.read_bytes())
    script = tmp_path / "s.py"
    script.write_text("x = 1\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(lonely), "--rules", "R1", str(script)],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "R1 never skips" in result.stderr, result
    # Without R1 the rest of the lint still runs.
    only_s1 = subprocess.run([sys.executable, str(lonely), "--rules", "S1", str(script)],
                             capture_output=True, text=True)
    assert only_s1.returncode == 0, only_s1


def test_r1_refuses_a_malformed_registry(tmp_path: Path) -> None:
    for body in ('{"symbols": []}', '{"symbols": [{"kind": "attribute", "symbol": "g.x"}]}',
                 '{"symbols": [{"kind": "class", "symbol": "g.x", "use": "-", "why": "-"}]}',
                 '{"symbols": [{"kind": "attribute", "symbol": "g x", "use": "-", "why": "-"}]}',
                 "not json"):
        bad = tmp_path / "bad.json"
        bad.write_text(body, encoding="utf-8")
        lint._REGISTRY.pop(bad, None)
        with pytest.raises(lint.RegistryError):
            lint.load_registry(bad)
