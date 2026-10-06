"""Live-emitter gates keyed on what a stock build does wrong — fork-free.

The live measurements behind each gate are in
``tests/test_meshable_part_route.py`` (equation ties, stock vs fork against
the series closed form) and ``tests/opensees/live/test_tet10_volume_live.py``
(TenNodeTetrahedron against E·A/H). These tests lock the gate logic on fake
``ops`` modules, so the ``suite`` lane holds it without any backend:

* ``equationConstraint`` runs on any build that has the command, and a
  build without it (openseespy 3.7.1.2) gets a curated refusal;
* a stock process that took an ``equationConstraint`` row refuses the next
  model, because stock ``wipe()`` keeps EQ rows (upstream
  ``Domain::clearAll()`` omits them);
* ``TenNodeTetrahedron`` is refused on stock, and silent on a fork build with
  the ``ladrunoBuild`` stamp;
* ``constraints LadrunoProjection`` gets the curated fork message on stock.

Every gate reads ``BackendInfo`` (F2-b, #1498), whose one fork signal is
``ladrunoBuild()`` returning a sha. A fork build predating that stamp (it has
``criticalTimeStep`` but no ``ladrunoBuild``) is therefore gated as stock.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.emitter.live import LiveOpsEmitter


class _StockOps:
    """Stock-shaped fake (no ``ladrunoBuild``), recording calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def wipe(self) -> None:
        self.calls.append(("wipe",))

    def element(self, *args: object) -> None:
        self.calls.append(("element", *args))

    def constraints(self, *args: object) -> None:
        self.calls.append(("constraints", *args))

    def equationConstraint(self, *args: object) -> None:  # noqa: N802
        self.calls.append(("equationConstraint", *args))


class _StockOps371(_StockOps):
    """openseespy 3.7.1.2: predates upstream's equationConstraint."""

    equationConstraint = None  # type: ignore[assignment]

    def __getattribute__(self, name: str) -> object:
        if name == "equationConstraint":
            raise AttributeError(name)
        return super().__getattribute__(name)


class _UnstampedForkOps(_StockOps):
    """A fork build predating fork PR #718: fork-only commands, no stamp."""

    def criticalTimeStep(self) -> float:  # noqa: N802
        return 1.0


class _ForkOps(_UnstampedForkOps):
    """A fork build by ``BackendInfo``: ``ladrunoBuild()`` answers a sha."""

    def ladrunoBuild(self) -> str:  # noqa: N802
        return "0" * 40


@pytest.fixture(autouse=True)
def _clean_process_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live, "_STOCK_EQ_ROWS_LIVE", False)


def _emitter(ops: object) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[attr-defined]
    le._in_partition = False
    le._fork_verified_types = set()
    return le


def _new_model(monkeypatch: pytest.MonkeyPatch, ops: object) -> LiveOpsEmitter:
    """A real ``LiveOpsEmitter(wipe=True)`` over ``ops``."""
    monkeypatch.setattr(live, "_get_ops", lambda: ops)
    return LiveOpsEmitter(wipe=True)


# -- equationConstraint -----------------------------------------------------

def test_equation_constraint_runs_on_stock_with_the_command() -> None:
    ops = _StockOps()
    _emitter(ops).equationConstraint(4, 1, 1.0, [(1, 1, -0.5), (2, 1, -0.5)])
    assert ops.calls == [("equationConstraint", 4, 1, 1.0, 1, 1, -0.5, 2, 1, -0.5)]


def test_equation_constraint_refused_on_a_build_without_it() -> None:
    ops = _StockOps371()
    with pytest.raises(RuntimeError, match="openseespy >= 3.8.0"):
        _emitter(ops).equationConstraint(4, 1, 1.0, [(1, 1, -1.0)])
    assert ops.calls == []


def test_stock_rows_refuse_the_next_model(monkeypatch: pytest.MonkeyPatch) -> None:
    ops = _StockOps()
    first = _new_model(monkeypatch, ops)
    first.equationConstraint(4, 1, 1.0, [(1, 1, -1.0)])
    with pytest.raises(RuntimeError, match="stock wipe\\(\\) cannot clear"):
        _new_model(monkeypatch, ops)
    # the refusal comes before a wipe that would pretend to reset the domain
    assert ops.calls.count(("wipe",)) == 1


def test_fork_rows_do_not_refuse_the_next_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = _ForkOps()   # the fork clears EQ rows in clearAll() (fork PR #312)
    _new_model(monkeypatch, ops).equationConstraint(4, 1, 1.0, [(1, 1, -1.0)])
    _new_model(monkeypatch, ops)
    assert ops.calls.count(("wipe",)) == 2


def test_unstamped_fork_rows_refuse_the_next_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One signal: without ladrunoBuild the build reads as stock, so its rows
    # are marked and the next model is refused (fail closed), with the note.
    ops = _UnstampedForkOps()
    _new_model(monkeypatch, ops).equationConstraint(4, 1, 1.0, [(1, 1, -1.0)])
    with pytest.raises(RuntimeError, match="fork PR #718"):
        _new_model(monkeypatch, ops)


def test_a_stock_model_without_rows_leaves_the_process_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = _StockOps()
    _new_model(monkeypatch, ops).element("stdBrick", 1, 1, 2, 3, 4, 5, 6, 7, 8, 1)
    _new_model(monkeypatch, ops)


def test_partition_block_neither_gates_nor_marks() -> None:
    le = _emitter(_StockOps371())
    le._in_partition = True
    le._ops = live._NoOpOps()  # type: ignore[attr-defined]
    le.equationConstraint(4, 1, 1.0, [(1, 1, -1.0)])
    assert live._STOCK_EQ_ROWS_LIVE is False


# -- TenNodeTetrahedron -----------------------------------------------------

_TET10_ARGS = (7, *range(1, 11), 1)


def test_tet10_refused_on_stock() -> None:
    ops = _StockOps()
    with pytest.raises(RuntimeError, match="6x too small"):
        _emitter(ops).element("TenNodeTetrahedron", *_TET10_ARGS)
    assert ops.calls == []


def test_tet10_silent_on_a_stamped_fork() -> None:
    ops = _ForkOps()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _emitter(ops).element("TenNodeTetrahedron", *_TET10_ARGS)
    assert ops.calls == [("element", "TenNodeTetrahedron", *_TET10_ARGS)]


def test_tet10_refused_on_an_unstamped_fork() -> None:
    # Was a Tet10UnverifiedBuildWarning; with BackendInfo as the one signal
    # an unstamped fork is stock, so the element is refused before the call.
    ops = _UnstampedForkOps()
    with pytest.raises(RuntimeError, match="6x too small.*fork PR #718"):
        _emitter(ops).element("TenNodeTetrahedron", *_TET10_ARGS)
    assert ops.calls == []


@pytest.mark.parametrize("ele", ["FourNodeTetrahedron", "stdBrick"])
def test_other_solids_untouched_on_stock(ele: str) -> None:
    ops = _StockOps()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _emitter(ops).element(ele, 1, 2, 3, 4, 5, 1)
    assert ops.calls == [("element", ele, 1, 2, 3, 4, 5, 1)]


# -- LadrunoProjection ------------------------------------------------------

def test_ladruno_projection_refused_on_stock() -> None:
    ops = _StockOps()
    with pytest.raises(RuntimeError, match="LadrunoProjection is fork-only"):
        _emitter(ops).constraints("LadrunoProjection")
    assert ops.calls == []


def test_ladruno_projection_refused_on_an_unstamped_fork() -> None:
    ops = _UnstampedForkOps()
    with pytest.raises(RuntimeError, match="fork-only.*fork PR #718"):
        _emitter(ops).constraints("LadrunoProjection")
    assert ops.calls == []


def test_ladruno_projection_passes_on_the_fork() -> None:
    ops = _ForkOps()
    _emitter(ops).constraints("LadrunoProjection", "-verbose")
    assert ops.calls == [("constraints", "LadrunoProjection", "-verbose")]


# -- partition blocks read the real build ------------------------------------

class _RCForkOps(_ForkOps):
    def nDMaterial(self, *args: object) -> None:  # noqa: N802
        self.calls.append(("nDMaterial", *args))


@pytest.mark.parametrize("rank", [0, 1])
def test_rc_c2_gate_classifies_the_real_build_inside_a_partition(
    monkeypatch: pytest.MonkeyPatch, rank: int,
) -> None:
    # Inside a non-zero partition block ``_ops`` is the ``_NoOpOps``
    # stand-in; the gate must classify the bound module, not the stand-in,
    # so a stamped fork is not refused as stock there.
    from apeGmsh.opensees import _rc_c2_flags

    monkeypatch.setattr(_rc_c2_flags, "LADRUNO_RC_C2_MIN_BUILD", "c" * 40)
    ops = _RCForkOps()
    le = _new_model(monkeypatch, ops)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # the rank!=0 notice
        le.partition_open(rank)
    le.nDMaterial("LadrunoRCConcrete", 1, 30.0, "-betaC", 189.0)
    le.partition_close()
    expected = [("nDMaterial", "LadrunoRCConcrete", 1, 30.0, "-betaC", 189.0)]
    assert [c for c in ops.calls if c[0] == "nDMaterial"] == (
        expected if rank == 0 else []
    )


def test_tet10_gate_reads_the_cached_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The bound module is probed once (the resolver's cache), not per element.
    probes: list[int] = []

    class _CountingFork(_ForkOps):
        def ladrunoBuild(self) -> str:  # noqa: N802
            probes.append(1)
            return "0" * 40

    ops = _CountingFork()
    monkeypatch.setattr(live, "_OPS_CACHE", ops)
    monkeypatch.setattr(live, "_BACKEND_INFO", None)
    le = _emitter(ops)
    for tag in range(1, 51):
        le.element("TenNodeTetrahedron", tag, *range(1, 11), 1)
    assert len(probes) == 1
