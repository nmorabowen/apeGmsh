"""``spec.emit_mpco`` against a fake ops domain.

These tests use a stand-in ``FakeOps`` so the build-gate, args
construction, and lifecycle can be exercised without an MPCO-capable
openseespy build.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.fixtures.split_backend import split_backend


# =====================================================================
# Fake ops module
# =====================================================================

class FakeOps:
    """Minimal ops stand-in: records ``recorder`` and ``remove`` calls."""

    def __init__(self) -> None:
        self.recorder_calls: list[tuple] = []
        self.remove_calls: list[tuple] = []
        self._next_tag = 1

    def recorder(self, *args) -> int:
        self.recorder_calls.append(tuple(args))
        tag = self._next_tag
        self._next_tag += 1
        return tag

    def remove(self, *args) -> None:
        self.remove_calls.append(tuple(args))


class BuildGateOps:
    """ops stand-in that simulates a build without the MPCO recorder."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.recorder_calls: list[tuple] = []
        self.remove_calls: list[tuple] = []

    def recorder(self, *args):
        self.recorder_calls.append(tuple(args))
        # Simulate openseespy raising when 'mpco' isn't registered.
        raise self._exc

    def remove(self, *args) -> None:
        self.remove_calls.append(tuple(args))


def _make_spec(*records: ResolvedRecorderRecord) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id="testhash",
        records=tuple(records),
    )


# =====================================================================
# Happy path
# =====================================================================

def test_emit_mpco_issues_one_recorder(tmp_path: Path) -> None:
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="nodes", name="r",
            components=("displacement_x", "displacement_y"),
            dt=None, n_steps=None,
            node_ids=np.array([1, 2]),
        ),
        ResolvedRecorderRecord(
            category="gauss", name="g",
            components=("stress_xx",),
            dt=None, n_steps=None,
            element_ids=np.array([10]),
        ),
    )
    fake = FakeOps()
    target = tmp_path / "run.mpco"

    with spec.emit_mpco(target, ops=fake) as live:
        # Exactly one ops.recorder call — MPCO is a single-recorder path
        assert len(fake.recorder_calls) == 1
        args = fake.recorder_calls[0]

        assert args[0] == "mpco"
        # Path should round-trip; backslashes vs slashes are fine
        # because the helper joins with forward slash on every platform.
        assert Path(args[1]) == target
        # Both -N (nodal) and -E (element) tokens are present
        assert "-N" in args
        assert "-E" in args
        assert "displacement" in args
        assert "stresses" in args

        assert live.tag == 1

    # Removing the recorder on exit flushes the HDF5 file
    assert fake.remove_calls == [("recorder", 1)]


def test_emit_mpco_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.mpco"
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    assert not target.parent.exists()
    with spec.emit_mpco(target, ops=fake):
        pass
    assert target.parent.is_dir()


def test_emit_mpco_handles_modal_records(tmp_path: Path) -> None:
    """Unlike emit_recorders, emit_mpco SUPPORTS modal records — the
    MPCO recorder writes mode shapes via the modesOfVibration token."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="modal", name="modes",
        components=(),
        dt=None, n_steps=None,
        n_modes=10,
    ))
    fake = FakeOps()
    target = tmp_path / "modes.mpco"
    with spec.emit_mpco(target, ops=fake):
        args = fake.recorder_calls[0]
        # Modal contributes the modesOfVibration token on -N
        assert "modesOfVibration" in args


def test_emit_mpco_handles_fibers_and_layers(tmp_path: Path) -> None:
    """Fibers/layers ARE supported on the MPCO path (they're MPCO-only
    today; this is the natural channel for them)."""
    spec = _make_spec(
        ResolvedRecorderRecord(
            category="fibers", name="fib",
            components=("fiber_stress",),
            dt=None, n_steps=None,
            element_ids=np.array([1]),
        ),
        ResolvedRecorderRecord(
            category="layers", name="lay",
            components=("fiber_strain",),
            dt=None, n_steps=None,
            element_ids=np.array([2]),
        ),
    )
    fake = FakeOps()
    target = tmp_path / "fl.mpco"
    with spec.emit_mpco(target, ops=fake):
        args = fake.recorder_calls[0]
        assert "section.fiber.stress" in args
        assert "section.fiber.strain" in args


def test_emit_mpco_with_dt(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    target = tmp_path / "run.mpco"
    with spec.emit_mpco(target, ops=fake):
        args = fake.recorder_calls[0]
        assert "-T" in args
        t_idx = args.index("-T")
        assert args[t_idx + 1] == "dt"
        assert args[t_idx + 2] == 0.01


# =====================================================================
# Cadence: one -T per recorder; different cadences warn once (#1948)
# =====================================================================

def _cadence_record(name: str, dt=None, n_steps=None, category="nodes"):
    return ResolvedRecorderRecord(
        category=category, name=name, components=("displacement_x",),
        dt=dt, n_steps=n_steps, node_ids=np.array([1]),
    )


def _emit_and_catch(tmp_path: Path, *records) -> "tuple[tuple, list]":
    fake = FakeOps()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _make_spec(*records).emit_mpco(tmp_path / "c.mpco", ops=fake):
            pass
    hits = [w for w in caught if "takes one cadence" in str(w.message)]
    return fake.recorder_calls[0], hits


def test_two_dt_cadences_warn_once_and_name_the_one_used(tmp_path: Path) -> None:
    args, hits = _emit_and_catch(tmp_path, _cadence_record("a", dt=0.05),
                                 _cadence_record("b", dt=0.01))
    assert args[args.index("-T"):] == ("-T", "dt", 0.01)
    assert len(hits) == 1
    assert "-T dt 0.01" in str(hits[0].message)


@pytest.mark.parametrize("records, used", [
    ((("a", 0.01, None), ("b", None, 5)), "-T dt 0.01"),
    ((("a", None, 10), ("b", None, 5)), "-T nsteps 5"),
    ((("a", 0.01, None), ("b", None, None)), "-T dt 0.01"),
])
def test_any_cadence_difference_warns(tmp_path: Path, records, used) -> None:
    _, hits = _emit_and_catch(
        tmp_path, *(_cadence_record(n, dt=d, n_steps=s) for n, d, s in records))
    assert len(hits) == 1 and used in str(hits[0].message)


@pytest.mark.parametrize("records", [
    (("a", 0.01, None), ("b", 0.01, None)),
    (("a", None, 5), ("b", None, 5)),
    (("a", None, None), ("b", None, None)),
])
def test_one_cadence_is_silent(tmp_path: Path, records) -> None:
    """Run with ``-W error::UserWarning``: a warning here raises."""
    _, hits = _emit_and_catch(
        tmp_path, *(_cadence_record(n, dt=d, n_steps=s) for n, d, s in records))
    assert hits == []


def _modal(dt=None, n_steps=None) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(category="modal", name="m",
                                  components=(), dt=dt, n_steps=n_steps)


@pytest.mark.parametrize("nodes, modal, tail, used", [
    # The modal record's cadence wins -T, so it must be compared.
    ({"dt": 0.05}, {"dt": 0.01}, ("-T", "dt", 0.01), "-T dt 0.01"),
    ({}, {"n_steps": 5}, ("-T", "nsteps", 5), "-T nsteps 5"),
    # A modal record at every step beside a timed record is sampled too.
    ({"dt": 0.01}, {}, ("-T", "dt", 0.01), "-T dt 0.01"),
])
def test_a_modal_record_cadence_is_compared(
    tmp_path: Path, nodes, modal, tail, used,
) -> None:
    """The MPCO ``-T`` gate skips the whole ``record()`` call, modes
    included, so a modal record's cadence is a cadence like any other."""
    args, hits = _emit_and_catch(
        tmp_path, _cadence_record("a", **nodes), _modal(**modal))
    assert args[args.index("-T"):] == tail
    assert len(hits) == 1 and used in str(hits[0].message)


def test_a_modal_record_at_the_same_cadence_is_silent(tmp_path: Path) -> None:
    _, hits = _emit_and_catch(
        tmp_path, _cadence_record("a", dt=0.01), _modal(dt=0.01))
    assert hits == []


def test_the_warning_points_at_the_users_line(tmp_path: Path) -> None:
    """Every emit route names the caller's line, not an apeGmsh frame."""
    from apeGmsh.results.spec._emit import mpco_ops_args

    spec = _make_spec(_cadence_record("a", dt=0.05),
                      _cadence_record("b", dt=0.01))
    calls = {
        "emit_mpco": lambda: spec.emit_mpco(tmp_path / "c.mpco", ops=FakeOps()).__enter__(),
        "tcl": lambda: spec.to_mpco_tcl_command(),
        "python": lambda: spec.to_mpco_python_command(),
        "ops_args": lambda: mpco_ops_args(spec.records),
    }
    for route, call in calls.items():
        with pytest.warns(UserWarning, match="takes one cadence") as rec:
            call()
        assert [Path(w.filename) for w in rec] == [Path(__file__)], route
        assert rec[0].lineno == call.__code__.co_firstlineno, route


# =====================================================================
# Build-gate
# =====================================================================

def test_emit_mpco_build_gate_wraps_runtime_error(tmp_path: Path) -> None:
    """If ops.recorder raises (typical when MPCO isn't compiled in),
    we re-raise with a clear remediation pointer."""
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = BuildGateOps(RuntimeError("WARNING - recorder type mpco unknown"))
    target = tmp_path / "run.mpco"

    with pytest.raises(RuntimeError) as excinfo:
        with spec.emit_mpco(target, ops=fake):
            pass

    msg = str(excinfo.value)
    assert "MPCO recorder" in msg
    assert "STKO" in msg
    assert "emit_recorders" in msg
    # The original exception is chained
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "mpco unknown" in str(excinfo.value.__cause__)


def test_emit_mpco_build_gate_wraps_arbitrary_exception(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = BuildGateOps(ValueError("argument parse error"))
    target = tmp_path / "run.mpco"
    with pytest.raises(RuntimeError):
        with spec.emit_mpco(target, ops=fake):
            pass


# =====================================================================
# Robustness
# =====================================================================

def test_remove_failure_warns_does_not_raise(tmp_path: Path) -> None:
    class BrokenRemove(FakeOps):
        def remove(self, *args):
            raise RuntimeError("simulated remove failure")

    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = BrokenRemove()
    target = tmp_path / "run.mpco"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_mpco(target, ops=fake):
            pass
    assert any(
        ".mpco file may not be flushed" in str(w.message) for w in caught
    )


def test_single_use(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    fake = FakeOps()
    target = tmp_path / "run.mpco"
    live = spec.emit_mpco(target, ops=fake)
    with live:
        pass
    with pytest.raises(RuntimeError, match="single-use"):
        with live:
            pass


def test_path_property(tmp_path: Path) -> None:
    spec = _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    ))
    target = tmp_path / "run.mpco"
    fake = FakeOps()
    with spec.emit_mpco(target, ops=fake) as live:
        assert live.path == target


# =====================================================================
# Without ops=: the module the bridge drives, never openseespy by name
# =====================================================================

def _nodal_spec() -> ResolvedRecorderSpec:
    return _make_spec(ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2]),
    ))


def test_defaults_to_the_module_the_bridge_drives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, stray = FakeOps(), FakeOps()
    split_backend(monkeypatch, bound, stray)
    with _nodal_spec().emit_mpco(tmp_path / "run.mpco"):
        pass
    assert len(bound.recorder_calls) == 1
    assert bound.remove_calls == [("recorder", 1)]
    assert stray.recorder_calls == []


def test_an_explicit_ops_beats_the_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, stray, mine = FakeOps(), FakeOps(), FakeOps()
    split_backend(monkeypatch, bound, stray)
    with _nodal_spec().emit_mpco(tmp_path / "run.mpco", ops=mine):
        pass
    assert len(mine.recorder_calls) == 1
    assert bound.recorder_calls == [] and stray.recorder_calls == []
