"""``LadrunoMembrane`` ``diaphragm_nodes=`` and ``f11=`` / ``f22=`` / ``f12=``.

The fork flags of build ``c83ee812e`` (fork WP-204): ``-diaphragmNodes``
(in-plane stiffness and mass, no gravity share) and the in-plane modifiers
``-f11 -f22 -f12``, and the build check that holds them to fork WP-208
(``b6d3e3d2``), the first build where they give the exact in-plane
stiffness. Numeric oracles:
``tests/opensees/live/test_ladruno_membrane_diaphragm_live.py``.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.build.pg_expand import (
    resolve_ladruno_membrane_diaphragm_nodes,
)
from apeGmsh.opensees._internal.build.stage_emit import (
    FORK_BUILD_HAS_MEMBRANE_WP208,
    MEMBRANE_WP208_GUARD,
    MEMBRANE_WP208_MESSAGE,
    fork_build_has_membrane_wp208,
    fork_guard_lines,
)
from apeGmsh.opensees.element.shell import LadrunoMembrane
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.node import Node, NodeSet
from tests.opensees.unit.test_ladruno_membrane import _2x2_slab_fem

_STIFF = (25000.5, 0.2, 0.2)


def _node_set(*tags: int) -> NodeSet:
    bridge = object()
    return NodeSet(
        nodes=tuple(Node(tag=t, coords=(0.0, 0.0, 0.0), bridge=bridge) for t in tags),  # type: ignore[arg-type]
        bridge=bridge,  # type: ignore[arg-type]
    )


def _membrane_args(rec: RecordingEmitter) -> tuple[object, ...]:
    calls = [
        c for c in rec.calls
        if c[0] == "element" and c[1][0] == "LadrunoMembrane"
    ]
    assert len(calls) == 1
    return calls[0][1]


def _flag(args: tuple[object, ...], flag: str, n: int) -> tuple[object, ...]:
    i = args.index(flag)
    return args[i + 1:i + 1 + n]


class TestConstruction:
    @pytest.mark.parametrize("name", ["f11", "f22", "f12"])
    @pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
    def test_modifier_must_be_finite_and_positive(self, name: str, bad: float) -> None:
        with pytest.raises(ValueError, match=f"{name}= must be finite and > 0"):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=_STIFF, **{name: bad})

    @pytest.mark.parametrize(
        "kw", [{"f11": 0.5}, {"f12": 0.3}, {"diaphragm_nodes": (5,)}],
    )
    def test_stiffness_flags_need_a_stiffness(self, kw: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="act on the in-plane stiffness"):
            LadrunoMembrane(nodes=(1, 2, 3), pressure=-1.0, **kw)

    def test_diaphragm_nodes_forms_are_normalised(self) -> None:
        ele = LadrunoMembrane(
            nodes=(1, 2, 3), stiff=_STIFF,
            diaphragm_nodes=(7, "Edge", _node_set(8, 9)),
        )
        assert ele.diaphragm_nodes == (7, "Edge", 8, 9)
        assert LadrunoMembrane(
            nodes=(1, 2, 3), stiff=_STIFF, diaphragm_nodes=4,
        ).diaphragm_nodes == (4,)

    @pytest.mark.parametrize("bad", [(), (0,), (True,), (1.5,)])
    def test_bad_diaphragm_nodes_refused(self, bad: tuple[object, ...]) -> None:
        with pytest.raises((TypeError, ValueError)):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=_STIFF, diaphragm_nodes=bad)  # type: ignore[arg-type]


class TestEmit:
    def test_direct_emit_places_the_new_flags(self) -> None:
        ele = LadrunoMembrane(
            nodes=(1, 2, 3, 4), support_nodes=(1,), diaphragm_nodes=(5, 6),
            stiff=_STIFF, f11=0.7, f22=0.35, f12=0.25,
        )
        rec = RecordingEmitter()
        ele._emit(rec, tag=9)
        args = rec.calls[0][1]
        assert args == (
            "LadrunoMembrane", 9,
            "-nodes", 1, 2, 3, 4,
            "-supportNodes", 1,
            "-diaphragmNodes", 5, 6,
            "-method", "tributary",
            "-stiff", 25000.5, 0.2, 0.2,
            "-f11", 0.7, "-f22", 0.35, "-f12", 0.25,
        )

    def test_defaults_emit_neither_flag(self) -> None:
        rec = RecordingEmitter()
        LadrunoMembrane(nodes=(1, 2, 3), stiff=_STIFF)._emit(rec, tag=1)
        args = rec.calls[0][1]
        assert "-diaphragmNodes" not in args
        assert not {"-f11", "-f22", "-f12"} & set(args)

    def test_direct_emit_refuses_a_name(self) -> None:
        ele = LadrunoMembrane(nodes=(1, 2, 3), stiff=_STIFF, diaphragm_nodes="Edge")
        with pytest.raises(RuntimeError, match="requires emitting through"):
            ele._emit(RecordingEmitter(), tag=1)

    def test_bridge_resolves_pg_and_label_names(self) -> None:
        ops = apeSees(_2x2_slab_fem())
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            pg="Slab", support_nodes="Corners",
            diaphragm_nodes=("c1", "Corners", 2), stiff=_STIFF, f22=0.5,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
        rec = RecordingEmitter()
        ops.build().emit(rec)
        args = _membrane_args(rec)
        # Label c1 is node 1; first occurrence kept.
        assert _flag(args, "-diaphragmNodes", 5) == (1, 3, 7, 9, 2)
        assert _flag(args, "-f22", 1) == (0.5,)

    def test_unknown_name_is_refused(self) -> None:
        ops = apeSees(_2x2_slab_fem())
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            pg="Slab", support_nodes="Corners", diaphragm_nodes="Nope",
            stiff=_STIFF,
        )
        with pytest.raises(BridgeError):
            ops.build().emit(RecordingEmitter())

    def test_h5_replay_keeps_the_new_flags(self, tmp_path: Path) -> None:
        ops = apeSees(_2x2_slab_fem())
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            pg="Slab", support_nodes="Corners", diaphragm_nodes=(2, 8),
            stiff=_STIFF, f11=0.6, f12=0.4,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
        tcl = TclEmitter()
        ops.build().emit(tcl)
        h5 = tmp_path / "m.h5"
        ops.h5(h5)
        direct = [ln.strip() for ln in tcl.lines() if ln.strip().startswith("element LadrunoMembrane")]
        replay = [
            ln.strip() for ln in OpenSeesModel.from_h5(h5).build("tcl").splitlines()
            if ln.strip().startswith("element LadrunoMembrane")
        ]
        assert len(direct) == 1 and "-diaphragmNodes 2 8" in direct[0]
        assert "-f11 0.6" in direct[0] and "-f12 0.4" in direct[0]
        assert replay == direct


class TestInteriorDiaphragmNode:
    """A ``pg=`` panel's interior mesh node is skipped from the deck unless
    the membrane names it; ``diaphragm_nodes=`` names it, as
    ``support_nodes=`` and ``holes=`` do."""

    @staticmethod
    def _deck(tmp_path: "Path | None" = None, **kw: Any) -> list[str]:
        ops = apeSees(_2x2_slab_fem())
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            pg="Slab", support_nodes="Corners", stiff=_STIFF, **kw)
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
        if tmp_path is not None:
            h5 = tmp_path / "m.h5"
            ops.h5(h5)
            return OpenSeesModel.from_h5(h5).build("tcl").splitlines()
        tcl = TclEmitter()
        ops.build().emit(tcl)
        return tcl.lines()

    @staticmethod
    def _has_node_5(lines: list[str]) -> bool:
        return any(ln.strip().startswith("node 5 ") for ln in lines)

    def test_interior_node_is_skipped_by_default(self) -> None:
        assert not self._has_node_5(self._deck())

    def test_interior_diaphragm_node_is_emitted(self) -> None:
        lines = self._deck(diaphragm_nodes=(5,))
        assert self._has_node_5(lines)
        assert any("-diaphragmNodes 5" in ln for ln in lines)

    def test_interior_diaphragm_node_is_emitted_on_h5_replay(
        self, tmp_path: Path,
    ) -> None:
        lines = self._deck(tmp_path, diaphragm_nodes=(5,))
        assert self._has_node_5(lines)
        assert any("-diaphragmNodes 5" in ln for ln in lines)


def test_diaphragm_resolver_refuses_a_non_membrane() -> None:
    spec: Any = types.SimpleNamespace(diaphragm_nodes=(5,))
    with pytest.raises(BridgeError, match="is not a LadrunoMembrane"):
        resolve_ladruno_membrane_diaphragm_nodes(_2x2_slab_fem(), spec)


# ---------------------------------------------------------------------------
# The fork WP-208 build check
# ---------------------------------------------------------------------------

_PIN = "23d937bc4011d1b29230671511531e23f777d78c"
#: The maintainer's venv build, which predates c83ee812e.
_STALE_FORK = "79e0623670c61c768357a01dff1477a4a0ef7e06"
_WP208_ARGS: dict[str, tuple[int | float | str, ...]] = {
    "diaphragm": ("-nodes", 1, 2, 3, "-diaphragmNodes", 4,
                  "-stiff", 1.0, 0.2, 0.1),
    "f11": ("-nodes", 1, 2, 3, "-stiff", 1.0, 0.2, 0.1, "-f11", 0.5),
    "f22": ("-nodes", 1, 2, 3, "-stiff", 1.0, 0.2, 0.1, "-f22", 0.5),
    "f12": ("-nodes", 1, 2, 3, "-stiff", 1.0, 0.2, 0.1, "-f12", 0.5),
}
_PLAIN_ARGS: tuple[int | float | str, ...] = (
    "-nodes", 1, 2, 3, "-stiff", 1.0, 0.2, 0.1)


def _pin_sha() -> str:
    root = Path(__file__).resolve().parents[3]
    for line in (root / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


def test_the_wp208_table_lists_the_pin_and_not_the_stale_build() -> None:
    carries = {sha for sha, ok in FORK_BUILD_HAS_MEMBRANE_WP208.items() if ok}
    assert _pin_sha() in carries, "FORK_PIN carries b6d3e3d2; measure a bump"
    assert _STALE_FORK not in carries
    assert set(MEMBRANE_WP208_GUARD.builds) == carries
    assert MEMBRANE_WP208_GUARD.stock_runs is False
    assert not fork_build_has_membrane_wp208(None)
    assert not fork_build_has_membrane_wp208("c" * 40)


class _StockOps:
    """Stock-shaped fake: no ``ladrunoBuild``; records element calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def element(self, *args: Any) -> None:
        self.calls.append(args)


class _ForkOps(_StockOps):
    def __init__(self, build: str) -> None:
        super().__init__()
        self._build = build

    def ladrunoBuild(self) -> str:  # noqa: N802
        return self._build


def _live(ops: object) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[attr-defined]
    le._in_partition = False
    le._fork_verified_types = set()
    return le


@pytest.mark.parametrize("which", sorted(_WP208_ARGS))
@pytest.mark.parametrize(
    "ops", [_StockOps(), _ForkOps(_STALE_FORK), _ForkOps("c" * 40)],
    ids=["stock", "stale-fork", "unmeasured-fork"],
)
def test_live_route_refuses_below_wp208(ops: _StockOps, which: str) -> None:
    ops.calls.clear()
    with pytest.raises(RuntimeError) as err:
        _live(ops).element("LadrunoMembrane", 1, *_WP208_ARGS[which])
    assert str(err.value) == MEMBRANE_WP208_MESSAGE
    assert ops.calls == []


@pytest.mark.parametrize("which", sorted(_WP208_ARGS))
def test_live_route_passes_on_a_wp208_build(which: str) -> None:
    le = _live(_ForkOps(_PIN))
    # What follows the check (the fork-element probe) is recorded, not run.
    called: list[tuple[Any, ...]] = []
    le._element_fork_gated = (  # type: ignore[method-assign]
        lambda ele_type, tag, args, key: called.append((ele_type, tag, *args)))
    le.element("LadrunoMembrane", 1, *_WP208_ARGS[which])
    assert called == [("LadrunoMembrane", 1, *_WP208_ARGS[which])]


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
def test_a_wp208_deck_checks_the_build_once(emitter: Any) -> None:
    dialect = "tcl" if emitter is TclEmitter else "py"
    guard = fork_guard_lines(MEMBRANE_WP208_GUARD, dialect)
    em = emitter()
    em.element("LadrunoMembrane", 1, *_WP208_ARGS["diaphragm"])
    em.element("LadrunoMembrane", 2, *_WP208_ARGS["f11"])
    lines = em.lines()
    assert lines.count(guard[0]) == 1
    at = lines.index(guard[0])
    assert lines[at:at + len(guard)] == guard
    assert at < next(i for i, ln in enumerate(lines) if "LadrunoMembrane" in ln)


@pytest.mark.parametrize("emitter", [TclEmitter, PyEmitter])
def test_a_plain_membrane_deck_has_no_wp208_guard(emitter: Any) -> None:
    em = emitter()
    em.element("LadrunoMembrane", 1, *_PLAIN_ARGS)
    assert not any("ladrunoBuild" in ln for ln in em.lines())


def test_the_bridge_deck_carries_the_wp208_guard() -> None:
    ops = apeSees(_2x2_slab_fem())
    ops.model(ndm=3, ndf=6)
    ops.element.LadrunoMembrane(
        pg="Slab", support_nodes="Corners", stiff=_STIFF, f11=0.5)
    ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
    tcl = TclEmitter()
    ops.build().emit(tcl)
    guard = fork_guard_lines(MEMBRANE_WP208_GUARD, "tcl")
    assert tcl.lines().count(guard[0]) == 1


def test_the_tcl_wp208_guard_on_stub_builds() -> None:
    tkinter = pytest.importorskip("tkinter")
    tcl = tkinter.Tcl()
    guard = "\n".join(fork_guard_lines(MEMBRANE_WP208_GUARD, "tcl"))
    with pytest.raises(tkinter.TclError) as err:
        tcl.eval(guard)  # stock: no ladrunoBuild, stops
    assert str(err.value) == MEMBRANE_WP208_MESSAGE
    for answer, ok in ((_PIN, True), (f" {_PIN}\n", True),
                       (_STALE_FORK, False), ("", False), ("0" * 40, False)):
        tcl.eval(f"proc ladrunoBuild {{}} {{ return {{{answer}}} }}")
        if ok:
            tcl.eval(guard)
            continue
        with pytest.raises(tkinter.TclError) as err:
            tcl.eval(guard)
        assert str(err.value) == MEMBRANE_WP208_MESSAGE, answer


def test_the_py_wp208_guard_on_stub_builds() -> None:
    code = compile(
        "\n".join(fork_guard_lines(MEMBRANE_WP208_GUARD, "py")),
        "<guard>", "exec")
    with pytest.raises(SystemExit) as err:
        exec(code, {"ops": types.SimpleNamespace()})  # stock stops
    assert err.value.code == MEMBRANE_WP208_MESSAGE
    for build, ok in ((_PIN, True), (_STALE_FORK, False), ("", False)):
        engine = {"ops": types.SimpleNamespace(ladrunoBuild=lambda b=build: b)}
        if ok:
            exec(code, engine)
            continue
        with pytest.raises(SystemExit) as err:
            exec(code, engine)
        assert err.value.code == MEMBRANE_WP208_MESSAGE
