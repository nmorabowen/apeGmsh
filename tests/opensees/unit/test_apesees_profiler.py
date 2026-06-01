"""Unit tests for the Ladruno-fork stack profiler surface (``ops.profiler.*``).

Covers P1 (emit surface) + P2 (live fork-gate) of
``internal_docs/plan_profiler_integration.md``:

* the ``_ProfilerNS`` verbs record ordered entries on the bridge;
* ``_split_profiler_records`` classifies verbs by bracket side
  (``start`` / ``reset`` before ``analyze``; ``stop`` / ``report`` /
  ``memory`` after);
* ``ops.tcl`` / ``ops.py`` flush the bracket around the ``analyze`` line;
* the live emitter re-raises a friendly "requires the Ladruno fork build"
  error when ``ops.profiler`` is absent (stock openseespy);
* ``ops.analyze(profile=...)`` brackets the live run.

No fork and no openseespy are needed: deck emit is pure text, the live
fork-gate is exercised against a hand-rolled ``_ops`` stub, and the live
``analyze(profile=)`` bracket runs through a recording fake.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _make_single_quad_fem() -> FEMStub:
    """One 4-node quad on a unit square in XY (mirrors the SSI-1 fixture)."""
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        node_pgs={"Left": [1, 4], "Bottom": [1, 2]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Rock": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),)),
        },
    )
    return FEMStub(nodes=nodes, elements=elements)


def _build_quad_with_chain() -> apeSees:
    """A complete, analyzable single-quad bridge (valid analysis chain)."""
    ops = apeSees(_make_single_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1.0e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(
        pg="Rock", thickness=1.0, material=mat, plane_type="PlaneStrain",
    )
    ops.fix(pg="Left", dofs=(1, 0))
    ops.fix(pg="Bottom", dofs=(0, 1))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1.0e-4, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()
    return ops


# ---------------------------------------------------------------------------
# Namespace recording + bracket-side split
# ---------------------------------------------------------------------------

def test_profiler_verbs_record_in_order() -> None:
    ops = apeSees(_make_single_quad_fem(), default_orientation=None)
    ops.profiler.start(deep=True, memory=True)
    ops.profiler.report("profile.h5", run="caseA")
    assert ops._profiler_records == [
        ("start", ("-deep", "-memory")),
        ("report", ("profile.h5", "-run", "caseA")),
    ]


def test_profiler_start_no_flags_records_empty_tail() -> None:
    ops = apeSees(_make_single_quad_fem(), default_orientation=None)
    ops.profiler.start()
    assert ops._profiler_records == [("start", ())]


def test_split_classifies_verbs_by_bracket_side() -> None:
    ops = apeSees(_make_single_quad_fem(), default_orientation=None)
    ops.profiler.start(deep=True)
    ops.profiler.reset()
    ops.profiler.stop()
    ops.profiler.report("p.h5")
    ops.profiler.memory()
    pre, post = ops._split_profiler_records()
    assert [v for v, _ in pre] == ["start", "reset"]
    assert [v for v, _ in post] == ["stop", "report", "memory"]


# ---------------------------------------------------------------------------
# Deck emit — the bracket lands around the analyze line
# ---------------------------------------------------------------------------

def test_tcl_deck_brackets_analyze_line(tmp_path) -> None:
    ops = _build_quad_with_chain()
    ops.profiler.start(deep=True)
    ops.profiler.report("profile.h5", run="caseA")
    path = str(tmp_path / "deck.tcl")
    ops.tcl(path, analyze_steps=5)
    lines = (tmp_path / "deck.tcl").read_text(encoding="utf-8").splitlines()

    i_start = lines.index("profiler start -deep")
    i_analyze = lines.index("analyze 5")
    i_report = lines.index("profiler report profile.h5 -run caseA")
    assert i_start < i_analyze < i_report


def test_py_deck_brackets_analyze_line(tmp_path) -> None:
    ops = _build_quad_with_chain()
    ops.profiler.start(deep=True)
    ops.profiler.report("profile.h5", run="caseA")
    path = str(tmp_path / "deck.py")
    ops.py(path, analyze_steps=5)
    lines = (tmp_path / "deck.py").read_text(encoding="utf-8").splitlines()

    i_start = lines.index("ops.profiler('start', '-deep')")
    i_analyze = lines.index("ops.analyze(5)")
    i_report = lines.index(
        "ops.profiler('report', 'profile.h5', '-run', 'caseA')"
    )
    assert i_start < i_analyze < i_report


def test_no_profiler_records_means_no_profiler_lines(tmp_path) -> None:
    ops = _build_quad_with_chain()
    path = str(tmp_path / "deck.tcl")
    ops.tcl(path, analyze_steps=5)
    text = (tmp_path / "deck.tcl").read_text(encoding="utf-8")
    assert "profiler" not in text


# ---------------------------------------------------------------------------
# Recording emitter captures the call
# ---------------------------------------------------------------------------

def test_recording_emitter_captures_profiler_call() -> None:
    e = RecordingEmitter()
    e.profiler("start", "-deep")
    assert ("profiler", ("start", "-deep"), {}) in e.calls


# ---------------------------------------------------------------------------
# Live fork-gate (P2)
# ---------------------------------------------------------------------------

def test_live_profiler_missing_command_raises_friendly() -> None:
    # Stock openseespy has no ``profiler`` attribute.
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = object()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="Ladruno fork"):
        le.profiler("start", "-deep")


def test_live_profiler_present_forwards_call() -> None:
    captured: list[tuple[object, ...]] = []

    class _Ops:
        def profiler(self, *args: object) -> None:
            captured.append(args)

    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = _Ops()  # type: ignore[attr-defined]
    le.profiler("report", "p.h5", "-run", "caseA")
    assert captured == [("report", "p.h5", "-run", "caseA")]


# ---------------------------------------------------------------------------
# Live analyze(profile=) brackets the run (no openseespy — recording fake)
# ---------------------------------------------------------------------------

def test_analyze_profile_brackets_live_run(monkeypatch) -> None:
    import apeGmsh.opensees.emitter.live as live_mod

    holder: dict[str, "_RecLive"] = {}

    class _RecLive(RecordingEmitter):
        supports_partitions = False

        def __init__(self, *, wipe: bool = True) -> None:
            super().__init__()
            holder["e"] = self

        def analyze(self, *, steps: int, dt: float | None = None) -> int:
            self.calls.append(("analyze", (), {"steps": steps, "dt": dt}))
            return 0

    monkeypatch.setattr(live_mod, "LiveOpsEmitter", _RecLive)

    ops = _build_quad_with_chain()
    ops.analyze(steps=5, profile="p.h5", profile_run="caseA", profile_deep=True)

    calls = holder["e"].calls
    i_start = next(
        i for i, c in enumerate(calls)
        if c[0] == "profiler" and c[1] == ("start", "-deep")
    )
    i_analyze = next(i for i, c in enumerate(calls) if c[0] == "analyze")
    i_report = next(
        i for i, c in enumerate(calls)
        if c[0] == "profiler" and c[1] == ("report", "p.h5", "-run", "caseA")
    )
    assert i_start < i_analyze < i_report


def test_analyze_without_profile_emits_no_profiler(monkeypatch) -> None:
    import apeGmsh.opensees.emitter.live as live_mod

    holder: dict[str, "_RecLive2"] = {}

    class _RecLive2(RecordingEmitter):
        supports_partitions = False

        def __init__(self, *, wipe: bool = True) -> None:
            super().__init__()
            holder["e"] = self

        def analyze(self, *, steps: int, dt: float | None = None) -> int:
            self.calls.append(("analyze", (), {"steps": steps, "dt": dt}))
            return 0

    monkeypatch.setattr(live_mod, "LiveOpsEmitter", _RecLive2)

    ops = _build_quad_with_chain()
    ops.analyze(steps=5)
    assert not any(c[0] == "profiler" for c in holder["e"].calls)
