"""Skill fence gate: every Python fence in the canonical skill resolves.

An audit of ``skills/apegmsh/`` on 2026-10-14 found 14 of 61 probed claims
stale or wrong, and 22 of 32 line citations pointing at the wrong line. The
cheatsheet taught ``g.mesh.field.distance(edges=...)`` and
``threshold(input_field=...)``, whose live keywords are ``curves=`` and
``distance_field=``; ``gotchas.md`` said ``gauss.select`` was not shipped
while ``GaussResultsComposite`` inherits it. Nothing failed.

This lane runs in the suite with no marker. It checks, for every markdown
file of the canonical skill except the ``DEFERRED`` ones another chain owns
(the ``.claude/skills/apegmsh-helper/`` mirror is byte-checked by
``scripts/sync_skill.py --check``):

* **fence-name / fence-keyword**: each Python fence resolves through the
  shared static resolver (``tests/_doc_resolver.py``): no missing module,
  attribute or keyword. Pages use ``g``, ``fem``, ``ops`` and ``results``
  without binding them, so those four are seeded as the session, the
  snapshot, the bridge and the results reader. A page that rebinds one
  takes its own binding. Whatever the resolver cannot type is skipped, not
  guessed;
* **fence-parse**: each Python fence parses. A pseudo-signature belongs in
  a ``text`` fence;
* **skill-freshness**: no freshness stamp. A stamp records when a reader
  last looked, not what is true; the gates are what keep the text true;
* **line-citation**: no ``<file>.py:<line>`` citation, with or without a
  ``src/`` prefix. Line numbers move under every edit;
* **link**: relative markdown links and backticked ``references/<x>.md``
  paths name files that exist;
* **machine-path / description-size / skill-size**: the SKILL.md
  description holds no ``C:\\Users`` path and is at most 1,024 characters
  (both the canonical and the mirror's derived description), and SKILL.md
  is at most 200 lines.

Today's known findings sit in ``tests/skill_fences_baseline.txt``, one line
per finding, keyed by file, rule and detail (never by line number, which
moves). The baseline only shrinks: a finding not in it fails, a line that
no longer matches a finding fails (delete it), and the line count is
pinned to ``SKILL_FENCES_BASELINE``. Raising the pin is a maintainer gate
(PROGRAM.md section 4). The slice that owns a skill file clears its lines.

The controls at the bottom are fixture strings, because the real defects
are being removed: each gate must flag the defect it exists for.
"""
from __future__ import annotations

import ast
import importlib
import re
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from tests._doc_resolver import _T, _Page

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / "skills" / "apegmsh"
MIRROR_SKILL_MD = REPO / ".claude" / "skills" / "apegmsh-helper" / "SKILL.md"
BASELINE_FILE = Path(__file__).with_name("skill_fences_baseline.txt")

#: The number of lines in ``skill_fences_baseline.txt``. It only goes down.
SKILL_FENCES_BASELINE = 97

#: Skill files another chain's open issue owns, left out of the gate and the
#: baseline so its edits cannot turn this test red. Name -> owning issue.
#: SK2 removes these entries when #1989 lands.
DEFERRED: dict[str, str] = {
    "opensees-bridge.md": "#1989",
    "ladruno.md": "#1989",
}

#: The receivers a page uses without binding them, seeded per page.
_SEEDS: dict[str, tuple[str, str]] = {
    "g": ("apeGmsh", "apeGmsh"),
    "fem": ("apeGmsh", "FEMData"),
    "ops": ("apeGmsh.opensees", "apeSees"),
    "results": ("apeGmsh", "Results"),
}

#: A file with a Python fence must resolve at least this many names, so a
#: page whose fences stop binding what they use cannot pass by checking nothing.
_MIN_CHECKED = 2

DESCRIPTION_MAX = 1024
SKILL_MD_MAX_LINES = 200

_STAMP = re.compile(r"<!--\s*skill-freshness:")
# A `<file>.py:<line>` citation, or a bare `:<line>` continuing one earlier on
# the same line (`_model_geometry.py:1147 (add_polyline), :3197 (add_void_sweep)`).
_CITATION = re.compile(r"(?P<file>[\w./]+\.py):(?P<line>\d+)|(?<![\w./:]):(?P<cont>\d+)")
_MACHINE_PATH = re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE)
_MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
_REF_PATH = re.compile(r"`(references/[\w-]+\.md)`")
_PROBLEM = re.compile(r"^\s*\S+:\d+: (?P<msg>.*)$")

Finding = str        # "<file> | <rule> | <detail>"


def _finding(rel: str, rule: str, detail: str) -> Finding:
    return f"{rel} | {rule} | {detail}"


def _seeded_page(rel: str) -> _Page:
    page = _Page(rel)
    page.seed({
        name: _T("inst", getattr(importlib.import_module(module), cls))
        for name, (module, cls) in _SEEDS.items()
    })
    return page


# ── Fences ──────────────────────────────────────────────────────────

def _uses_a_seed(body: str) -> bool:
    """``body`` parses and reads an attribute of ``g`` / ``fem`` / ``ops`` / ``results``."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and n.value.id in _SEEDS for n in ast.walk(tree))


def gated_fences(text: str) -> list[tuple[int, str]]:
    """``(first_content_lineno, dedented_body)`` of every fence the gate
    resolves, in page order: each ``python`` / ``py`` fence, and each untagged
    fence that is Python by evidence (it parses and uses a seeded receiver), so
    dropping the language tag cannot take a fence out of the gate."""
    fences: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        opener = lines[i].strip()
        if not opener.startswith("```"):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith("```"):
            j += 1
        body = textwrap.dedent("\n".join(lines[i + 1:j]))
        info = opener[3:].strip()
        if info in ("python", "py") or (info == "" and _uses_a_seed(body)):
            fences.append((i + 2, body))
        i = j + 1
    return fences


def fence_findings(text: str, rel: str) -> tuple[list[Finding], set[str]]:
    """The fence findings of one page, and the names its fences resolved."""
    page = _seeded_page(rel)
    found: list[Finding] = []
    for line, body in gated_fences(text):
        before = len(page.problems)
        page.fence(body, line)
        for problem in page.problems[before:]:
            m = _PROBLEM.match(problem)
            msg = m["msg"] if m else problem.strip()
            if "does not parse" in msg:
                first = next((s.strip() for s in body.splitlines() if s.strip()), "")
                found.append(_finding(rel, "fence-parse", f"{msg}; fence opens `{first[:60].rstrip()}`"))
            elif "takes no keyword" in msg:
                found.append(_finding(rel, "fence-keyword", msg))
            else:
                found.append(_finding(rel, "fence-name", msg))
    return found, page.checked


# ── Structure ───────────────────────────────────────────────────────

def description(text: str) -> str | None:
    """The ``description`` of a SKILL.md front matter, folded as YAML folds
    a ``>`` block or a multi-line plain scalar (lines joined by one space)."""
    m = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if m is None:
        return None
    lines = m.group(1).splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("description:"):
            continue
        head = line[len("description:"):].strip()
        parts = [] if head in (">", ">-", "|", "|-") else [head]
        for cont in lines[i + 1:]:
            if cont and not cont[0].isspace():
                break
            parts.append(cont.strip())
        return " ".join(p for p in parts if p)
    return None


def description_findings(text: str, rel: str) -> list[Finding]:
    desc = description(text)
    if desc is None:
        return [_finding(rel, "description-size", "no description in the front matter")]
    found = []
    if len(desc) > DESCRIPTION_MAX:
        found.append(_finding(rel, "description-size", f"description over {DESCRIPTION_MAX:,} characters"))
    found += [_finding(rel, "machine-path", f"description names `{m.group(0)}`")
              for m in _MACHINE_PATH.finditer(desc)]
    return found


def citations(text: str) -> list[str]:
    """Every line citation, a continuation spelled with its line's last cited file."""
    found: list[str] = []
    for line in text.splitlines():
        last = None
        for m in _CITATION.finditer(line):
            if m["file"]:
                last = m["file"]
                found.append(f"{last}:{m['line']}")
            elif last is not None:
                found.append(f"{last}:{m['cont']}")
    return found


def structure_findings(text: str, rel: str, base: Path) -> list[Finding]:
    """Stamps, line citations and dead links of one page; ``base`` is the
    directory its relative links resolve against."""
    found = [_finding(rel, "skill-freshness", "freshness stamp") for _ in _STAMP.finditer(text)]
    found += [_finding(rel, "line-citation", c) for c in citations(text)]
    for m in _MD_LINK.finditer(text):
        target = m.group(1).split("#", 1)[0]
        if target and not re.match(r"^[a-z]+:", target) and not (base / target).exists():
            found.append(_finding(rel, "link", f"`{m.group(1)}` does not resolve"))
    for m in _REF_PATH.finditer(text):
        if not (SKILL / m.group(1)).is_file():
            found.append(_finding(rel, "link", f"`{m.group(1)}` does not resolve"))
    if rel == "SKILL.md":
        found += description_findings(text, rel)
        if len(text.splitlines()) > SKILL_MD_MAX_LINES:
            found.append(_finding(rel, "skill-size", f"SKILL.md over {SKILL_MD_MAX_LINES} lines"))
    return found


# ── The corpus ──────────────────────────────────────────────────────

def _skill_files() -> list[Path]:
    return sorted(p for p in SKILL.rglob("*.md") if p.name not in DEFERRED)


@pytest.fixture(scope="module")
def corpus() -> tuple[Counter[Finding], dict[str, set[str]]]:
    found: Counter[Finding] = Counter()
    checked: dict[str, set[str]] = {}
    for path in _skill_files():
        rel = path.relative_to(SKILL).as_posix()
        text = path.read_text(encoding="utf-8")
        fences, names = fence_findings(text, rel)
        found.update(fences)
        found.update(structure_findings(text, rel, path.parent))
        if gated_fences(text):
            checked[rel] = names
    mirror = MIRROR_SKILL_MD.read_text(encoding="utf-8")
    found.update(description_findings(mirror, "mirror:apegmsh-helper/SKILL.md"))
    return found, checked


def _baseline() -> list[Finding]:
    lines = BASELINE_FILE.read_text(encoding="utf-8").splitlines()
    return [line.rstrip() for line in lines if line.strip() and not line.startswith("#")]


def ratchet_diff(found: Counter[Finding], baseline: Counter[Finding]) -> tuple[list[str], list[str]]:
    """``(new, stale)``: findings the baseline does not hold, and baseline
    lines no finding matches any more (both as multisets)."""
    return sorted((found - baseline).elements()), sorted((baseline - found).elements())


def test_skill_corpus_is_found() -> None:
    names = {p.relative_to(SKILL).as_posix() for p in _skill_files()}
    assert "SKILL.md" in names and "references/model-scripts.md" in names, sorted(names)


def test_deferred_files_exist_and_name_an_issue() -> None:
    for name, issue in DEFERRED.items():
        hits = list(SKILL.rglob(name))
        assert len(hits) == 1, f"DEFERRED names `{name}`, which matches {hits}: drop or fix the entry"
        assert re.fullmatch(r"#\d+", issue), f"DEFERRED[{name!r}] = {issue!r} names no issue"
    assert not [line for line in _baseline() if line.split(" | ")[0].rsplit("/", 1)[-1] in DEFERRED], (
        "a deferred file is out of the gate, so it has no baseline lines"
    )


def test_skill_findings_are_resolved_or_baselined(corpus) -> None:
    found, _ = corpus
    new, stale = ratchet_diff(found, Counter(_baseline()))
    assert not new, (
        "Skill drift: fix the SKILL TEXT (the code is the truth). A pseudo-signature "
        "goes in a ```text fence. New findings:\n  " + "\n  ".join(new)
    )
    assert not stale, (
        "These baseline lines match no finding any more. The baseline only shrinks: "
        f"delete them from {BASELINE_FILE.name} and lower SKILL_FENCES_BASELINE:\n  "
        + "\n  ".join(stale)
    )


def test_baseline_only_shrinks() -> None:
    n = len(_baseline())
    assert n <= SKILL_FENCES_BASELINE, (
        f"{BASELINE_FILE.name} has {n} lines, above SKILL_FENCES_BASELINE = "
        f"{SKILL_FENCES_BASELINE}. The baseline only shrinks: fix the skill text "
        "instead (raising the pin needs the maintainer)."
    )
    assert n == SKILL_FENCES_BASELINE, (
        f"{BASELINE_FILE.name} shrank to {n} lines: lower SKILL_FENCES_BASELINE to {n}, "
        "so the freed slack cannot be spent on a new finding."
    )


def test_every_fenced_file_resolves_names(corpus) -> None:
    _, checked = corpus
    thin = {rel: sorted(names) for rel, names in checked.items() if len(names) < _MIN_CHECKED}
    assert not thin, (
        f"these files resolve fewer than {_MIN_CHECKED} names: their fences no longer "
        f"bind what they use, so the gate guards (almost) nothing there: {thin}"
    )
    every = set().union(*checked.values())
    for name, (_, cls) in _SEEDS.items():
        assert any(c.startswith(f"{cls}.") for c in every), (
            f"the seeded `{name}` ({cls}) resolved nothing across the skill: the seed is dead"
        )


# ── Controls: each gate flags the defect it exists for ──────────────

def _fence(src: str) -> str:
    return f"```python\n{src}\n```\n"


def test_control_edges_keyword_is_flagged() -> None:
    # The cheatsheet fence before Q #2036: `g` is never bound, so the seed types it.
    found, checked = fence_findings(_fence(
        "g.mesh.field.distance(edges=[1], sampling=100)\n"
        "g.mesh.field.threshold(input_field=1, size_min=0.1, size_max=1.0)"
    ), "cheatsheet.md")
    assert _finding("cheatsheet.md", "fence-keyword",
                    "`FieldHelper.distance()` takes no keyword `edges=`") in found, found
    assert _finding("cheatsheet.md", "fence-keyword",
                    "`FieldHelper.threshold()` takes no keyword `input_field=`") in found, found
    assert {"apeGmsh.mesh", "Mesh.field", "FieldHelper.distance"} <= checked


def test_control_live_keywords_and_inherited_methods_resolve() -> None:
    # The gauss.select gotcha: `select` is inherited from a mixin, so a fence that
    # calls it must resolve, not read as missing (the false "not shipped" claim).
    found, checked = fence_findings(_fence(
        "g.mesh.field.distance(curves=[1], sampling=100)\n"
        "sel = results.elements.gauss.select(pg='Body')\n"
        "ops.fix(pg='Base', dofs=(1, 1, 1))"
    ), "live.md")
    assert found == [], found
    assert {"GaussResultsComposite.select", "ElementResultsComposite.gauss",
            "apeSees.fix"} <= checked, sorted(checked)


def test_control_missing_attribute_and_unparsable_fence() -> None:
    found, _ = fence_findings(
        _fence("ops.node(1, 0.0, 0.0)") + _fence("def add(self, x=..., *, y: float = …): ..."),
        "bad.md",
    )
    assert _finding("bad.md", "fence-name", "`apeSees` has no attribute `node`") in found
    assert any(f.startswith("bad.md | fence-parse | ") and "fence opens `def add(" in f
               for f in found), found


def test_control_attributes_set_from_an_annotated_parameter_resolve() -> None:
    # `FEMData.__init__` does `self.nodes = nodes` with `nodes: NodeComposite`.
    # Before the resolver typed that, every `fem.nodes.*` chain was skipped:
    # these two edits to fem-broker.md:108 and :111 passed.
    found, checked = fence_findings(_fence(
        'fem.nodes.get_ids(pg="Base")\n'
        'fem.nodes.select(pg="Body", partitionz=3)\n'
        'fem.nodes.select(pg="Body", partition=3)'
    ), "fem.md")
    assert _finding("fem.md", "fence-name", "`NodeComposite` has no attribute `get_ids`") in found, found
    assert _finding("fem.md", "fence-keyword",
                    "`NodeComposite.select()` takes no keyword `partitionz=`") in found, found
    assert len(found) == 2, found
    assert {"FEMData.nodes", "NodeComposite.select"} <= checked, sorted(checked)


def test_control_fem_broker_mutations_are_caught() -> None:
    # The same two edits applied to the real page must be new findings.
    text = (SKILL / "references" / "fem-broker.md").read_text(encoding="utf-8")
    for old, new, expected in (
        ('base = fem.nodes.select(pg="Base")', 'base = fem.nodes.get_ids(pg="Base")',
         "`NodeComposite` has no attribute `get_ids`"),
        ('fem.nodes.select(pg="Body", partition=3)', 'fem.nodes.select(pg="Body", partitionz=3)',
         "`NodeComposite.select()` takes no keyword `partitionz=`"),
    ):
        assert text.count(old) == 1, f"fem-broker.md no longer holds `{old}`: update this control"
        found, _ = fence_findings(text.replace(old, new), "references/fem-broker.md")
        assert any(f.endswith(f" | {expected}") for f in found), found


def test_control_a_seed_survives_only_a_plain_untyped_call() -> None:
    # fem-broker.md rebinds `fem = FEMData.from_msh(...)`, whose return is not
    # annotated: the seed holds. A loop target named `g` is not the session.
    found, _ = fence_findings(_fence(
        "fem = make_snapshot()\nfem.nodes.get_ids()\n"
        "for g in groups:\n    g.not_a_session_attribute()"
    ), "reseed.md")
    assert found == [_finding("reseed.md", "fence-name", "`NodeComposite` has no attribute `get_ids`")], found


def test_control_untagged_python_fences_are_gated() -> None:
    text = (
        "```\nfem.nodes.get_ids(pg='Base')\n```\n"      # Python by evidence: gated
        "```\nmodel.h5\n  /meta\n```\n"                  # not Python: left alone
        "```\nx = 1  # parses, but uses no seeded receiver\n```\n"
        "```text\nops.node(1, 0.0, 0.0)\n```\n"          # explicitly text: left alone
    )
    assert [body for _, body in gated_fences(text)] == ["fem.nodes.get_ids(pg='Base')"]
    found, _ = fence_findings(text, "bare.md")
    assert found == [_finding("bare.md", "fence-name", "`NodeComposite` has no attribute `get_ids`")], found


def test_control_a_page_binding_overrides_the_seed() -> None:
    # A page that binds `ops` to raw openseespy is not judged as the bridge.
    found, _ = fence_findings(_fence(
        "import openseespy.opensees as ops\nops.node(1, 0.0, 0.0)"
    ), "raw.md")
    assert found == [], found


def test_control_line_citations_are_flagged() -> None:
    text = (
        "See `src/apeGmsh/mesh/FEMData.py:1508` and `FEMData.py:1508`.\n"
        # The 2.19.0 row as fem-broker.md and opensees-bridge.md ship it: a
        # restated constant (live: 2.32.0) pinned to a line that moved.
        "| bridge `SCHEMA_VERSION` | **`\"2.19.0\"`** | `src/apeGmsh/opensees/emitter/h5.py:414` |\n"
        "- bridge `SCHEMA_VERSION` (`opensees/emitter/h5.py:414`) = **2.19.0**\n"
        "verified: tests/test_femdata_from_h5.py::test_session_save_then_from_h5\n"
        # Continuations, as api-cheatsheet.md:119 and :744 ship them.
        "`# src/apeGmsh/core/_model_geometry.py:1147 (add_polyline), :3197 (add_void_sweep)`\n"
        "`# ConstraintsComposite.py:624 (contact), :989 (plane); EmbedmentsComposite.py:129 (embed)`\n"
        "Run at 10:30, see note :5 and `{'a':1}`.\n"     # a bare :N with no citation before it
    )
    found = structure_findings(text, "cite.md", SKILL)
    cited = sorted(f.split(" | ")[2] for f in found if " | line-citation | " in f)
    assert cited == sorted([
        "src/apeGmsh/mesh/FEMData.py:1508", "FEMData.py:1508",
        "src/apeGmsh/opensees/emitter/h5.py:414", "opensees/emitter/h5.py:414",
        "src/apeGmsh/core/_model_geometry.py:1147", "src/apeGmsh/core/_model_geometry.py:3197",
        "ConstraintsComposite.py:624", "ConstraintsComposite.py:989", "EmbedmentsComposite.py:129",
    ]), cited


def test_control_freshness_stamp_and_dead_links_are_flagged() -> None:
    text = (
        "<!-- skill-freshness: verified against apeGmsh main@5c92ca92 (2026-08-15) -->\n"
        "[bridge](opensees-bridge.md) [gone](no-such-page.md#x) [site](https://x.y/z.md)\n"
        "Read `references/model-scripts.md`, then `references/no-such-card.md`.\n"
    )
    found = structure_findings(text, "refs.md", SKILL / "references")
    assert _finding("refs.md", "skill-freshness", "freshness stamp") in found
    assert _finding("refs.md", "link", "`no-such-page.md#x` does not resolve") in found
    assert _finding("refs.md", "link", "`references/no-such-card.md` does not resolve") in found
    assert len([f for f in found if " | link | " in f]) == 2, found


def _skill_md(desc: str, body_lines: int) -> str:
    head = f"---\nname: apegmsh\ndescription: >\n  {desc}\n---\n"
    return head + "line\n" * (body_lines - head.count("\n"))


def test_control_description_size_and_machine_path() -> None:
    # 1,815 characters: the description as it stood before Q #2036.
    long = _skill_md("x" * 1815, 50)
    assert len(description(long)) == 1815
    assert _finding("SKILL.md", "description-size", "description over 1,024 characters") in \
        structure_findings(long, "SKILL.md", SKILL)
    fits = _skill_md("x" * 1024, 50)
    assert structure_findings(fits, "SKILL.md", SKILL) == []
    path = _skill_md(r"the apeGmsh repository (C:\Users\nmora\Github\apeGmsh)", 50)
    assert _finding("SKILL.md", "machine-path", "description names `C:\\Users\\`") in \
        structure_findings(path, "SKILL.md", SKILL)


def test_control_description_folds_a_plain_multiline_scalar() -> None:
    text = "---\nname: x\ndescription: one two\n  three four\nother: y\n---\nbody\n"
    assert description(text) == "one two three four"


def test_control_skill_md_line_limit() -> None:
    assert _finding("SKILL.md", "skill-size", "SKILL.md over 200 lines") in \
        structure_findings(_skill_md("ok", 201), "SKILL.md", SKILL)
    assert structure_findings(_skill_md("ok", 200), "SKILL.md", SKILL) == []
    # The limit is SKILL.md's alone; a reference card may be longer.
    assert structure_findings(_skill_md("ok", 201), "references/x.md", SKILL) == []


def test_control_ratchet_rejects_new_and_stale_lines() -> None:
    a, b = _finding("x.md", "fence-name", "a"), _finding("x.md", "fence-name", "b")
    assert ratchet_diff(Counter([a, a]), Counter([a, a])) == ([], [])
    assert ratchet_diff(Counter([a, a, b]), Counter([a, a])) == ([b], [])
    # One of two identical findings cleared: one baseline line is stale.
    assert ratchet_diff(Counter([a]), Counter([a, a])) == ([], [a])
