"""The ``n_steps=`` cadence warning fires only on the emitting routes (#1980).

A record with ``n_steps=`` and no ``dt=`` cannot be honoured by a
classic recorder; ``spec/_emit.py::_warn_n_steps_unsupported`` says so
at every ``emit_logical`` call. The emitting routes (``to_tcl_commands``,
``to_python_commands``, ``LiveRecorders.begin_stage``) must warn once per
record. The calls that only translate a record (``Results.from_recorders``
reading the files back, ``LiveRecorders.__enter__`` pre-translating) go
through ``writers._cache.quiet_emit_logical`` and stay silent.

Run with ``-W error::UserWarning`` (warn-as-contract).
"""
from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.spec._emit import emit_logical
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.writers._cache import (
    _N_STEPS_CADENCE_WARNING_RE,
    quiet_emit_logical,
)

from tests.conftest import _open_model_from_h5, _stub_model_h5_path
from tests.test_results_live_recorders import (
    _CADENCE_COMPONENTS,
    FakeOps,
    _cadence_spec,
    _m,
    _make_spec,
)
from tests.test_results_transcoder import _MockFem, _write_synthetic_node_out

_SRC = Path(__file__).resolve().parents[1] / "src" / "apeGmsh" / "results"
# warnings.filterwarnings compiles ``message`` with re.I and uses .match.
_FILTER = re.compile(_N_STEPS_CADENCE_WARNING_RE, re.I)


def _cadence(caught: list) -> list[str]:
    return [str(w.message) for w in caught
            if issubclass(w.category, UserWarning)
            and "Emitting an every-step recorder" in str(w.message)]


def _two_record_spec() -> ResolvedRecorderSpec:
    return _make_spec(*(
        ResolvedRecorderRecord(
            category="nodes", name=name, components=("displacement_x",),
            dt=None, n_steps=5, node_ids=np.array([1, 2]),
        )
        for name in ("a", "b")
    ))


# =====================================================================
# Regex guard: the filter matches the emitter's real message
# =====================================================================

@pytest.mark.parametrize("category", list(_CADENCE_COMPONENTS))
def test_regex_matches_the_real_n_steps_warning(category: str) -> None:
    (rec,) = _cadence_spec(category, dt=None, n_steps=5).records
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        emit_logical(rec, output_dir="out")
    user = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user) == 1, [str(w.message) for w in caught]
    assert _FILTER.match(str(user[0].message)), str(user[0].message)


def test_regex_does_not_match_other_warnings() -> None:
    assert not _FILTER.match(
        "LiveRecorders: skipping record 'f' (category='fibers')")
    assert not _FILTER.match("Record 'r': unknown component 'x'")


@pytest.mark.parametrize("category", list(_CADENCE_COMPONENTS))
def test_quiet_emit_logical_is_emit_logical_without_the_warning(
    category: str,
) -> None:
    (rec,) = _cadence_spec(category, dt=None, n_steps=5).records
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet = quiet_emit_logical(rec, output_dir="out", stage_id="s")
    assert _cadence(caught) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loud = emit_logical(rec, output_dir="out", stage_id="s")
    assert quiet == loud and quiet


def test_quiet_emit_logical_keeps_other_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apeGmsh.results.spec._emit as emit_mod

    def warn_other(*_a, **_k):
        warnings.warn("unrelated", UserWarning, stacklevel=2)
        return []

    monkeypatch.setattr(emit_mod, "emit_logical", warn_other)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet_emit_logical(None)
    assert [str(w.message) for w in caught] == ["unrelated"]


# =====================================================================
# Quiet routes
# =====================================================================

def test_from_recorders_reads_an_n_steps_record_silently(
    tmp_path: Path,
) -> None:
    fem = _MockFem([1, 2])
    out = tmp_path / "out"
    out.mkdir()
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="r", components=("displacement_x",),
            dt=None, n_steps=5, node_ids=np.array([1, 2]),
        ),),
    )
    _write_synthetic_node_out(
        out / "r_disp.out", time=[0.0, 1.0], node_ids=[1, 2], dofs=[1],
        value_fn=lambda k, nid, dof: 0.1 * (nid + k),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Results.from_recorders(
            spec, out, fem=fem, cache_root=tmp_path / "cache",
            model=_open_model_from_h5(_stub_model_h5_path()),
        ) as r:
            slab = r.nodes.get(component="displacement_x")
    np.testing.assert_allclose(slab.values[0], [0.1, 0.2])
    assert _cadence(caught) == []


def test_live_enter_is_silent_and_begin_stage_warns_once_per_record(
    tmp_path: Path,
) -> None:
    spec = _two_record_spec()
    live = spec.emit_recorders(str(tmp_path), ops=FakeOps(), model=_m(spec))
    with warnings.catch_warnings(record=True) as entered:
        warnings.simplefilter("always")
        live.__enter__()
    try:
        assert _cadence(entered) == []
        with warnings.catch_warnings(record=True) as staged:
            warnings.simplefilter("always")
            live.begin_stage("s")
        live.end_stage()
    finally:
        live.__exit__(None, None, None)
    msgs = _cadence(staged)
    assert len(msgs) == 2, msgs
    assert "'a'" in msgs[0] and "'b'" in msgs[1]


# =====================================================================
# Emitting routes warn once per record
# =====================================================================

@pytest.mark.parametrize("route", ["to_tcl_commands", "to_python_commands"])
def test_deck_routes_warn_once_per_record(route: str) -> None:
    spec = _two_record_spec()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lines = getattr(spec, route)(output_dir="out", model=_m(spec))
    assert lines
    msgs = _cadence(caught)
    assert len(msgs) == 2, msgs
    assert "'a'" in msgs[0] and "'b'" in msgs[1]


# =====================================================================
# The read path calls quiet_emit_logical by name, never an alias
# =====================================================================

def _loud_calls(node: ast.AST) -> list[int]:
    return [n.lineno for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "emit_logical"]


def _no_alias(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert all(a.asname != "emit_logical" for a in node.names), (
                node.lineno)


def test_read_path_never_calls_emit_logical() -> None:
    tree = ast.parse(
        (_SRC / "transcoders/_recorder.py").read_text(encoding="utf-8"))
    _no_alias(tree)
    assert _loud_calls(tree) == []
    tree = ast.parse((_SRC / "writers/_cache.py").read_text(encoding="utf-8"))
    _no_alias(tree)
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "list_source_files"]
    assert _loud_calls(fn) == []


# =====================================================================
# A refusal at begin_stage leaves the stage unissued
# =====================================================================

def _ok_then_b_spec() -> ResolvedRecorderSpec:
    return _make_spec(*(
        ResolvedRecorderRecord(
            category="nodes", name=name, components=("displacement_x",),
            dt=None, n_steps=n_steps, node_ids=np.array([1, 2]),
        )
        for name, n_steps in (("ok", None), ("b", 5))
    ))


@pytest.mark.parametrize("refusal", ["cadence_warning_as_error", "to_deck_raises"])
def test_begin_stage_refusal_issues_no_recorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: str,
) -> None:
    import apeGmsh.results.live._recorders as live_mod

    spec = _ok_then_b_spec()
    fake = FakeOps()
    expected: type[Exception] = UserWarning
    if refusal == "to_deck_raises":
        real = live_mod.to_deck
        expected = RuntimeError

        def refuse_b_at_stage(logical, *a, **k):
            if logical.record_name == "b" and "s1__" in logical.file_path:
                raise RuntimeError("refused b")
            return real(logical, *a, **k)

        monkeypatch.setattr(live_mod, "to_deck", refuse_b_at_stage)
    with warnings.catch_warnings():
        if refusal == "cadence_warning_as_error":
            warnings.simplefilter("error", UserWarning)
        else:
            warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(expected):
            with spec.emit_recorders(
                str(tmp_path), ops=fake, model=_m(spec),
            ) as live:
                live.begin_stage("s1")
    assert fake.recorder_calls == []
    assert [s.name for s in live.stages] == []
    assert live.tags == ()
