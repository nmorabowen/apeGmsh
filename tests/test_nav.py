"""scripts/nav.py: each command gives the answer a reader of the fixture can check by eye.

The fixture is a two-module package written into ``tmp_path``; every line
number asserted below can be read off ``SHAPES`` / ``USE`` (line 1 is the
docstring). The tool itself is loaded from its file, as
``tests/test_check_quirks.py`` loads its script, and runs in-process with
``--root`` pointed at the fixture, so it never indexes this checkout.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "nav.py"


def _load():
    spec = importlib.util.spec_from_file_location("nav", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nav = _load()

SHAPES = '''\
"""Shapes: a tiny family for nav tests."""
from typing import Protocol

NODES_PATH = "/model/nodes"


# ---------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------
class Drawable(Protocol):
    def draw(self): ...
    def area(self): ...


class Shape:
    """Base of the family."""

    def area(self):
        return 0.0


class Circle(Shape):
    def draw(self):
        return "circle"

    def area(self):
        # Square is named in this comment only
        return 3.14


class Square(Shape):
    def area(self):
        return 1.0


class Triangle(Shape):
    def draw(self):
        return "tri"


def dispatch(s):
    """Square is named in this docstring too."""
    if isinstance(s, (Circle, Square)):
        return s.area()
    return None


ALL_SHAPES = (Circle, Square)


def write(f, s):
    f.create_dataset("/model/nodes", data=s)
    x = f["model/nodes"]
    if "model/nodes" in f:
        return f[NODES_PATH]
    return x
'''

USE = '''\
"""Uses shapes."""
from .shapes import Circle


def make():
    return Circle()


def late():
    from apeGmsh.geo import shapes
    return shapes.dispatch(make())
'''


@pytest.fixture
def root(tmp_path):
    pkg = tmp_path / "src" / "apeGmsh" / "geo"
    pkg.mkdir(parents=True)
    (tmp_path / "src" / "apeGmsh" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text('"""Geo package."""\n')
    (pkg / "shapes.py").write_text(SHAPES)
    (pkg / "use.py").write_text(USE)
    return tmp_path


def run(root, capsys, *argv):
    assert nav.main(["--root", str(root), *argv]) == 0
    return capsys.readouterr().out.splitlines()


# ------------------------------------------------------------ the contract
def test_stdlib_only_and_never_imports_apegmsh():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            tops.add(node.module.split(".")[0])
    assert tops - {"__future__"} <= sys.stdlib_module_names, tops - sys.stdlib_module_names


def test_a_run_imports_no_apegmsh_module(root, capsys):
    before = set(sys.modules)
    run(root, capsys, "where", "Circle")
    assert not [m for m in set(sys.modules) - before if m.startswith("apeGmsh")]


def test_gitignore_holds_the_cache_dir():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert nav.CACHE_DIR + "/" in lines


# ------------------------------------------------------------ the commands
def test_map_lists_banners_classes_and_methods_with_ranges(root, capsys):
    out = run(root, capsys, "map", "geo/shapes.py")
    assert out[0].startswith("src/apeGmsh/geo/shapes.py  (56 lines)")
    assert "  == Protocols  (L8)" in out
    assert "  class Circle 22-28" in out
    assert "    area 26-28" in out
    assert "  dispatch 41-45  -- Square is named in this docstring too." in out


def test_at_names_the_innermost_symbol_and_its_section(root, capsys):
    out = run(root, capsys, "at", "geo/shapes.py:27", "geo/shapes.py:4")
    assert out[0].startswith("geo/shapes.py:27  in Circle.area (26-28)  [section: Protocols]")
    assert out[1] == "geo/shapes.py:4  <module level>"


def test_at_rejects_a_location_without_a_line(root, capsys):
    with pytest.raises(SystemExit, match="expected FILE:LINE"):
        run(root, capsys, "at", "geo/shapes.py")


def test_where_filters_by_kind(root, capsys):
    out = run(root, capsys, "where", "area")
    assert [ln.split("  ")[0] for ln in out] == [
        "geo/shapes.py:12-12", "geo/shapes.py:18-19", "geo/shapes.py:26-28",
        "geo/shapes.py:32-33"]
    assert run(root, capsys, "where", "area", "--kind", "def") == ["no definition of 'area'"]


def test_refs_skip_comments_and_docstrings(root, capsys):
    out = run(root, capsys, "refs", "Square")
    assert out[0].startswith("Square: 2 refs in 1 files")
    assert "      dispatch  L43" in out and "      <module>  L48" in out
    assert run(root, capsys, "refs", "Circle", "--kind", "call") == [
        "Circle: 1 refs in 1 files, by enclosing scope (comments/docstrings excluded)",
        "  geo/use.py", "      make  L6"]
    assert run(root, capsys, "refs", "Square", "--file", "use.py") == [
        "no code references to 'Square'"]


def test_unknown_kind_is_an_error_not_an_empty_answer(root, capsys):
    with pytest.raises(SystemExit) as exc:
        run(root, capsys, "refs", "Square", "--kind", "cal")
    assert exc.value.code == 2
    assert "unknown kind cal" in capsys.readouterr().err


def test_h5_classifies_write_read_probe_and_constant_use(root, capsys):
    out = run(root, capsys, "h5", "model/nodes")
    assert out[1:] == ["  write/read/probe/via-const geo/shapes.py:52  write",
                       "  via-const/const  geo/shapes.py:4  <module>",
                       "  path constants: NODES_PATH (geo/shapes.py)"]
    out = run(root, capsys, "h5", "model/nodes", "--kind", "probe")
    assert out[1:] == ["  probe            geo/shapes.py:54  write",
                       "  path constants: NODES_PATH (geo/shapes.py)"]


def test_impl_marks_who_implements_the_method(root, capsys):
    out = run(root, capsys, "impl", "Drawable.draw")
    rows = {ln.split()[1]: ln.split()[0] for ln in out[1:]}
    assert rows == {"Shape": "MISSING", "Circle": "IMPL", "Square": "MISSING",
                    "Triangle": "IMPL"}


def test_family_reports_the_enumerations_and_their_gaps(root, capsys):
    out = run(root, capsys, "family", "Shape", "--nodocs")
    assert out[0] == "family: 3 public subclasses of Shape"
    # the isinstance class tuple is dispatch only, not also a table
    assert [ln for ln in out if "/3   " in ln] == [
        "    2/3   dispatch geo/shapes.py:43  dispatch  missing: Triangle",
        "    2/3   table    geo/shapes.py:48  ALL_SHAPES  missing: Triangle",
        "    2/3   geo/shapes.py  missing: Triangle"]


def test_family_takes_base_or_names(root, capsys):
    with pytest.raises(SystemExit):
        run(root, capsys, "family")


def test_pkg_up_and_deps_follow_the_imports(root, capsys):
    out = run(root, capsys, "pkg", "geo")
    assert out[0].startswith("geo: 3 modules, 68 lines.")
    assert any(ln.split()[:3] == ["shapes.py", "56", "in=1+0"] for ln in out)
    assert run(root, capsys, "up", "dispatch") == ["dispatch", "  <- late  geo/use.py:11"]
    assert run(root, capsys, "up", "nobody") == ["no call sites of 'nobody'"]
    out = run(root, capsys, "deps", "geo.shapes")
    # use.py imports shapes eagerly at L2 (Circle) and lazily inside late() (the module)
    assert out == ["apeGmsh.geo.shapes: imported by 1 files  geo:eager=1",
                   "  eager/lazy geo/use.py:2  *, Circle"]


# ----------------------------------------------------- bound and cache
def test_every_answer_is_bounded_and_the_rest_is_reachable(root, capsys):
    big = root / "src" / "apeGmsh" / "geo" / "big.py"
    big.write_text("".join(f"def f{i}():\n    pass\n" for i in range(80)))
    out = run(root, capsys, "map", "big.py")
    assert len(out) == nav.MAX_LINES
    assert out[-1] == "... 22 more lines, narrow with --depth 0 or --lines A-B"
    assert out[-2] == "  f57 115-116"
    rest = run(root, capsys, "map", "big.py", "--lines", "117-160")
    assert rest[1] == "  f58 117-118" and rest[-1] == "  f79 159-160"


def test_bounded_leaves_a_short_answer_alone():
    assert nav.bounded("a\nb\n", "x", limit=2) == "a\nb\n"
    assert nav.bounded("a\nb\nc\n", "x", limit=2) == "a\n... 2 more lines, narrow with x\n"


def test_cache_is_per_root_reused_and_invalidated_by_an_edit(root, capsys):
    run(root, capsys, "where", "Circle")
    assert (root / nav.CACHE_DIR / "index.marshal").is_file()
    nav.main(["--root", str(root), "-v", "where", "Circle"])
    assert "; 0 parsed" in capsys.readouterr().err
    (root / "src" / "apeGmsh" / "geo" / "use.py").write_text(USE + "\n\ndef later():\n    pass\n")
    nav.main(["--root", str(root), "-v", "where", "later"])
    cap = capsys.readouterr()
    assert "; 1 parsed" in cap.err
    assert cap.out.startswith("geo/use.py:14-15  def later")


def test_a_corrupt_cache_is_rebuilt(root, capsys):
    cache = root / nav.CACHE_DIR / "index.marshal"
    cache.parent.mkdir()
    cache.write_bytes(b"not marshal")
    assert run(root, capsys, "where", "Triangle")[0].startswith("geo/shapes.py:36-38  class")


def test_a_root_without_sources_fails_loudly(tmp_path, capsys):
    with pytest.raises(SystemExit, match="not an apeGmsh checkout"):
        run(tmp_path, capsys, "where", "x")


# ------------------------------------------------ review findings (#1234)
def test_impl_refuses_an_ambiguous_protocol_and_file_picks_one(root, capsys):
    (root / "src" / "apeGmsh" / "geo" / "other.py").write_text(
        "class Drawable:\n    def draw(self): ...\n")
    with pytest.raises(SystemExit, match=r"ambiguous: src/apeGmsh/geo/other.py:1, "
                                         r"src/apeGmsh/geo/shapes.py:10 \(narrow with --file\)"):
        run(root, capsys, "impl", "Drawable.draw")
    out = run(root, capsys, "impl", "Drawable.draw", "--file", "shapes.py")
    assert out[0].startswith("Drawable (geo/shapes.py:10) declares 2 public methods")


@pytest.mark.parametrize("line", ["0", "57"])
def test_at_refuses_a_line_outside_the_file(root, capsys, line):
    with pytest.raises(SystemExit, match=f"shapes.py has 56 lines; line {line} is out of range"):
        run(root, capsys, "at", f"geo/shapes.py:{line}")


def test_map_refuses_a_range_past_the_end(root, capsys):
    with pytest.raises(SystemExit, match="shapes.py has 56 lines; --lines 57-90 starts past"):
        run(root, capsys, "map", "geo/shapes.py", "--lines", "57-90")


def test_find_file_accepts_dot_and_absolute_paths(root, capsys, tmp_path_factory):
    want = run(root, capsys, "at", "src/apeGmsh/geo/shapes.py:27")
    assert run(root, capsys, "at", "./src/apeGmsh/geo/shapes.py:27") == want
    assert run(root, capsys, "at", f"{root / 'src/apeGmsh/geo/shapes.py'}:27") == want
    outside = tmp_path_factory.mktemp("elsewhere") / "shapes.py"
    with pytest.raises(SystemExit, match="is outside"):
        run(root, capsys, "at", f"{outside}:1")


def test_a_cut_refs_answer_names_the_flag_that_lists_the_files(root, capsys):
    geo = root / "src" / "apeGmsh" / "geo"
    for i in range(10):
        (geo / f"u{i}.py").write_text("".join(
            f"def g{j}():\n    return Square\n" for j in range(8)))
    out = run(root, capsys, "refs", "Square")
    assert len(out) == nav.MAX_LINES
    assert out[-1].endswith("narrow with --kind/--file (--limit 1 lists every file)")
    listed = [ln.strip() for ln in run(root, capsys, "refs", "Square", "--limit", "1")
              if ln.startswith("  geo/")]
    assert sorted(listed) == sorted([f"geo/u{i}.py" for i in range(10)] + ["geo/shapes.py"])
