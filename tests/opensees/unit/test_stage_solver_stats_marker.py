"""ADR 0106 S2 -- the ``APEGMSH_STAGE open|close <name>`` runtime marker.

Covers:

- INV-1: a staged deck with no ``stats=True`` anywhere carries no
  ``APEGMSH_STAGE`` line at all, in either emitter.
- A stats-declaring staged deck emits exactly one open/close pair per
  stage, name last on the line, matching the S1 parser's
  ``r"APEGMSH_STAGE (open|close) (.+)$"``.
- The marker survives a stage name with spaces and embedded quotes.
- A partitioned staged deck emitting ``Mumps(stats=True)`` carries the
  same marker pair per stage without disturbing a single byte of the
  ``if {[getPID] == K} { ... }`` rank guards -- proved by removing
  exactly the marker lines (and the ``-stats`` token they are gated
  on) from the stats deck and reproducing the no-stats deck line for
  line.
"""
from __future__ import annotations

import re

import pytest
from typing import cast

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)
from tests.opensees.integration.test_emit_partitioned_staged import (
    _make_4quad_2pg_2part_fem,
)

_MARKER_RE = re.compile(r"^APEGMSH_STAGE (open|close) (.+)$")


# ---------------------------------------------------------------------------
# Fixtures -- a plain two-stage, unpartitioned model
# ---------------------------------------------------------------------------


def _make_two_pg_fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0),
            ],
            node_pgs={},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock":   _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "cimbra": _ElementGroupView(
                    ids=(2,), connectivity=((2, 5, 6, 3),),
                ),
            },
        ),
    )


def _chain(ops: apeSees, *, stats: bool) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.Pardiso(stats=stats),
        "analysis":    ops.analysis.Static(),
    }


def _two_stage_ops(
    *, stats: bool, names: "tuple[str, str]" = ("gravity", "push"),
) -> apeSees:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    with ops.stage(name=names[0]) as s:
        s.analysis(**_chain(ops, stats=stats))
        s.run(n_increments=1)
    with ops.stage(name=names[1]) as s:
        s.analysis(**_chain(ops, stats=stats))
        s.run(n_increments=1)
    return ops


#: Pulls the ``(phase, name)`` pair out of the SOURCE line -- Tcl's
#: ``puts "APEGMSH_STAGE open <name>"`` or Python's
#: ``print("APEGMSH_STAGE open <name>", flush=True)``. The name never
#: contains a literal ``"`` (both emitters normalise it to ``'``), so
#: the first ``"`` after ``APEGMSH_STAGE`` always closes the marker
#: text -- what remains between that point and the source-level quote
#: is exactly what ``puts`` / ``print`` would write to the runtime
#: stream, i.e. the string the S1 parser's
#: ``r"APEGMSH_STAGE (open|close) (.+)$"`` matches.
_SOURCE_MARKER_RE = re.compile(r'APEGMSH_STAGE (open|close) ([^"]*)"')


def _markers(lines: "list[str]") -> "list[tuple[str, str]]":
    out = []
    for ln in lines:
        m = _SOURCE_MARKER_RE.search(ln)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


# ---------------------------------------------------------------------------
# INV-1 -- no stats anywhere, no marker anywhere
# ---------------------------------------------------------------------------


def test_no_stats_no_marker_tcl() -> None:
    ops = _two_stage_ops(stats=False)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    text = "\n".join(emitter.lines())
    assert "APEGMSH_STAGE" not in text


def test_no_stats_no_marker_py() -> None:
    ops = _two_stage_ops(stats=False)
    emitter = PyEmitter()
    ops.build().emit(emitter)
    text = "\n".join(emitter.lines())
    assert "APEGMSH_STAGE" not in text


# ---------------------------------------------------------------------------
# One open/close pair per stage, name last on the line
# ---------------------------------------------------------------------------


def test_stats_emits_one_open_close_pair_per_stage_tcl() -> None:
    ops = _two_stage_ops(stats=True, names=("gravity", "push"))
    emitter = TclEmitter()
    ops.build().emit(emitter)
    markers = _markers(emitter.lines())
    assert markers == [
        ("open", "gravity"), ("close", "gravity"),
        ("open", "push"), ("close", "push"),
    ]


def test_stats_emits_one_open_close_pair_per_stage_py() -> None:
    ops = _two_stage_ops(stats=True, names=("gravity", "push"))
    emitter = PyEmitter()
    ops.build().emit(emitter)
    markers = _markers(emitter.lines())
    assert markers == [
        ("open", "gravity"), ("close", "gravity"),
        ("open", "push"), ("close", "push"),
    ]


def test_tcl_marker_line_is_the_exact_runtime_text_the_s1_parser_matches() -> None:
    """Strip the ``puts "..."`` Tcl wrapper by hand (not via
    ``_SOURCE_MARKER_RE``) and check the S1 parser's own
    ``r"APEGMSH_STAGE (open|close) (.+)$"`` matches what is LEFT --
    i.e. exactly what ``puts`` writes to stdout at runtime."""
    ops = _two_stage_ops(stats=True, names=("gravity", "push"))
    emitter = TclEmitter()
    ops.build().emit(emitter)
    line = next(
        ln for ln in emitter.lines()
        if ln.startswith('puts "APEGMSH_STAGE open')
    )
    runtime_text = line.removeprefix('puts "').removesuffix('"')
    m = _MARKER_RE.fullmatch(runtime_text)
    assert m is not None
    assert m.group(1) == "open"
    assert m.group(2) == "gravity"


def test_py_marker_line_is_the_exact_runtime_text_the_s1_parser_matches() -> None:
    ops = _two_stage_ops(stats=True, names=("gravity", "push"))
    emitter = PyEmitter()
    ops.build().emit(emitter)
    line = next(
        ln for ln in emitter.lines()
        if ln.startswith('print("APEGMSH_STAGE open')
    )
    runtime_text = line.removeprefix('print("').removesuffix('", flush=True)')
    m = _MARKER_RE.fullmatch(runtime_text)
    assert m is not None
    assert m.group(1) == "open"
    assert m.group(2) == "gravity"


def test_open_marker_precedes_close_marker_within_stage_block_tcl() -> None:
    ops = _two_stage_ops(stats=True)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()
    open_i = lines.index('puts "APEGMSH_STAGE open gravity"')
    close_i = lines.index('puts "APEGMSH_STAGE close gravity"')
    assert open_i < close_i
    # The close marker sits before the stage's own ``loadConst`` reset.
    load_const_i = next(
        i for i, ln in enumerate(lines) if ln == "loadConst -time 0.0"
    )
    assert close_i < load_const_i


# ---------------------------------------------------------------------------
# A stage name with spaces AND quotes survives both emitters
# ---------------------------------------------------------------------------


def test_marker_survives_name_with_spaces_and_quotes_tcl() -> None:
    name = 'ground "water" table'
    ops = _two_stage_ops(stats=True, names=(name, "push"))
    emitter = TclEmitter()
    ops.build().emit(emitter)
    markers = _markers(emitter.lines())
    # Tcl quoting normalises ``"`` -> ``'`` (the existing strategy-name
    # precedent) so the marker's own enclosing ``puts "..."`` string
    # cannot be closed early by the name.
    assert markers[0] == ("open", "ground 'water' table")
    assert markers[1] == ("close", "ground 'water' table")


def test_marker_survives_name_with_spaces_and_quotes_py() -> None:
    name = 'ground "water" table'
    ops = _two_stage_ops(stats=True, names=(name, "push"))
    emitter = PyEmitter()
    ops.build().emit(emitter)
    markers = _markers(emitter.lines())
    assert markers[0] == ("open", "ground 'water' table")
    assert markers[1] == ("close", "ground 'water' table")


@pytest.mark.parametrize("emitter_cls", [TclEmitter, PyEmitter])
@pytest.mark.parametrize(("name", "expected"), [
    ("cost $rank", "cost _rank"),            # Tcl variable substitution
    ("a\\b", "a/b"),                        # backslash escape, both lanes
    ("push {2}", "push (2)"),                # Tcl braces
    ("line one\nline two", "line one line two"),  # a newline would split the marker
    ("  padded\t name ", "padded name"),
])
def test_marker_name_is_immune_to_substitution(emitter_cls, name, expected) -> None:
    """A stage name is unrestricted (only non-empty), so ``$``, ``\``,
    braces and newlines reach the deck; ``stage_marker_name`` normalises
    them once for both lanes so the runtime line the S1 parser sees is
    the same in tcl and py and never a Tcl substitution."""
    ops = _two_stage_ops(stats=True, names=(name, "push"))
    emitter = emitter_cls()
    ops.build().emit(emitter)
    markers = _markers(emitter.lines())
    assert markers[0] == ("open", expected)
    assert markers[1] == ("close", expected)
    for ln in emitter.lines():
        if "APEGMSH_STAGE" in ln:
            assert "$" not in ln and "\\" not in ln and "\n" not in ln


def test_marker_survives_name_with_brackets_tcl() -> None:
    """``[``/``]`` would trigger Tcl command substitution inside the
    double-quoted ``puts`` string if left unescaped -- normalised to
    parens the same way ``analyze`` normalises a strategy name."""
    name = "stage [1]"
    ops = _two_stage_ops(stats=True, names=(name, "push"))
    emitter = TclEmitter()
    ops.build().emit(emitter)
    markers = _markers(emitter.lines())
    assert markers[0] == ("open", "stage (1)")
    text = "\n".join(emitter.lines())
    assert "[1]" not in text.split("=== Stage:")[0]  # no stray literal bracket in a puts line
    assert 'puts "APEGMSH_STAGE open stage (1)"' in text


# ---------------------------------------------------------------------------
# Partitioned staged deck -- marker present, rank guards untouched
# ---------------------------------------------------------------------------


def _mumps_chain(ops: apeSees, *, stats: bool) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.Mumps(stats=stats),
        "analysis":    ops.analysis.Static(),
    }


def _partitioned_staged_ops(fem: FEMStub, *, stats: bool) -> apeSees:
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="rock_base", dofs=(1, 1))
    with ops.stage(name="rock_only") as s:
        s.analysis(**_mumps_chain(ops, stats=stats))
        s.run(n_increments=2)
    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_mumps_chain(ops, stats=stats))
        s.run(n_increments=3)
    return ops


def _drop_stage_marker_lines(lines: "list[str]") -> "list[str]":
    """Remove every ``puts "APEGMSH_STAGE ...\"`` line and its paired
    ``flush stdout`` line, in order -- the exact two lines
    ``_emit_stage_marker`` adds and nothing else."""
    out: "list[str]" = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('puts "APEGMSH_STAGE '):
            assert lines[i + 1] == "flush stdout", (
                f"expected 'flush stdout' immediately after {ln!r}"
            )
            i += 2
            continue
        out.append(ln)
        i += 1
    return out


def test_partitioned_staged_marker_is_the_only_diff_from_no_stats_tcl() -> None:
    fem = _make_4quad_2pg_2part_fem()
    lines_plain = TclEmitter()
    _partitioned_staged_ops(fem, stats=False).build().emit(lines_plain)
    fem2 = _make_4quad_2pg_2part_fem()
    lines_stats = TclEmitter()
    _partitioned_staged_ops(fem2, stats=True).build().emit(lines_stats)

    plain = lines_plain.lines()
    stats = lines_stats.lines()

    # Sanity: the stats deck really does carry the markers and the
    # no-stats deck carries none.
    assert _markers(stats) == [
        ("open", "rock_only"), ("close", "rock_only"),
        ("open", "install_cimbra"), ("close", "install_cimbra"),
    ]
    assert _markers(plain) == []

    stripped = _drop_stage_marker_lines(stats)
    assert len(stripped) == len(plain), (
        "removing exactly the marker lines should reproduce the "
        "no-stats deck's line count"
    )
    diffs = [
        (i, a, b) for i, (a, b) in enumerate(zip(stripped, plain)) if a != b
    ]
    # The ONLY remaining difference anywhere in the deck is the
    # ``-stats`` flag on the two ``system Mumps`` lines -- every rank
    # guard (``if {[getPID] == K} { ... }``) and everything inside it
    # is byte-identical.
    for i, a, b in diffs:
        assert a.lstrip().startswith("system Mumps"), (
            f"unexpected non-marker diff at line {i}: {a!r} vs {b!r}"
        )
        assert a == b + " -stats", (
            f"unexpected non-marker diff at line {i}: {a!r} vs {b!r}"
        )
    assert diffs, "expected the two 'system Mumps ... -stats' lines to differ"


def test_partitioned_staged_marker_not_inside_a_rank_guard_tcl() -> None:
    """The marker rides the same global scope as ``stage_open`` /
    ``analyze`` (Comment: 'Analysis chain -- global; each rank executes
    locally') -- it must NOT land inside an ``if {[getPID] == K} {`` /
    ``}`` bracket, where it would fire on only one rank's local domain
    instead of every rank's process."""
    fem = _make_4quad_2pg_2part_fem()
    ops = _partitioned_staged_ops(fem, stats=True)
    emitter = TclEmitter()
    ops.build().emit(emitter)
    lines = emitter.lines()

    depth = 0
    for ln in lines:
        stripped = ln.strip()
        if re.match(r"^if \{\[getPID\] == \d+\} \{$", stripped):
            depth += 1
            continue
        if stripped == "}" and depth > 0:
            depth -= 1
            continue
        if stripped.startswith("puts \"APEGMSH_STAGE"):
            assert depth == 0, f"marker line inside a rank guard: {ln!r}"
