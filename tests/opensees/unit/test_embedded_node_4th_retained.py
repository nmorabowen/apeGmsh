"""The openseespy emitters carry the 4th retained node of a tet4 embedded
tie (#1621, program slice B4-a).

``ASDEmbeddedNodeElement``'s parser reads the tag, the constrained node
and three retained nodes with ``OPS_GetInt`` and the optional 4th
retained node with ``OPS_GetString`` + ``std::stoi`` inside a
``catch(...)``. Under openseespy an ``int`` argument is not a string
there, so ``PyEmitter`` and ``LiveOpsEmitter`` passing every node as an
int tied a tet4 host to three of its four corners. The 4th node now
goes as its decimal string; a 3-node (triangle) call is byte-for-byte
what it was. The Tcl line is the oracle for the node order; the live
proof that the string reaches the element is
``tests/opensees/live/test_embedded_tet4_live.py``.
"""
from __future__ import annotations

import types
from typing import Any

import pytest

from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.emitter.base import _embedded_retained_args
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter


def _py_line(*nodes: int, **kw: Any) -> str:
    e = PyEmitter()
    e.embeddedNode(7, 50, *nodes, **kw)
    (line,) = [ln for ln in e.lines() if "ASDEmbeddedNodeElement" in ln]
    return line


def _live_args(monkeypatch: pytest.MonkeyPatch, *nodes: int) -> tuple[Any, ...]:
    mod = types.ModuleType("openseespy.opensees")
    calls: list[tuple[Any, ...]] = []
    mod.wipe = lambda: None  # type: ignore[attr-defined]
    mod.element = lambda *a: calls.append(a)  # type: ignore[attr-defined]
    monkeypatch.setattr(live, "_get_ops", lambda: mod)
    LiveOpsEmitter(wipe=False).embeddedNode(7, 50, *nodes, stiffness=1e12)
    (args,) = calls
    return args


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_three_retained_nodes_stay_ints() -> None:
    assert _embedded_retained_args((1, 2, 3)) == [1, 2, 3]


def test_fourth_retained_node_is_its_decimal_string() -> None:
    got = _embedded_retained_args((1, 2, 3, 4))
    assert got == [1, 2, 3, "4"]
    assert [type(a) for a in got] == [int, int, int, str]


@pytest.mark.parametrize("nodes", [(), (1,), (1, 2), (1, 2, 3, 4, 5)])
def test_other_counts_pass_through_as_ints(nodes: tuple[int, ...]) -> None:
    """Only the tet4 call changes; every other count is what it was."""
    got = _embedded_retained_args(nodes)
    assert got == list(nodes)
    assert all(type(a) is int for a in got)


# ---------------------------------------------------------------------------
# The .py deck
# ---------------------------------------------------------------------------

def test_py_deck_tet4_line_carries_the_4th_node_as_a_string() -> None:
    assert _py_line(1, 2, 3, 4, stiffness=1e12) == (
        "ops.element('ASDEmbeddedNodeElement', 7, 50, 1, 2, 3, '4', "
        "'-K', 1000000000000.0)"
    )


def test_py_deck_triangle_line_is_unchanged() -> None:
    assert _py_line(1, 2, 3, stiffness=1e12) == (
        "ops.element('ASDEmbeddedNodeElement', 7, 50, 1, 2, 3, "
        "'-K', 1000000000000.0)"
    )


def test_py_deck_flags_follow_the_4th_node() -> None:
    """Parser order: the 4th node is the token at index 5 (after tag,
    cnode, r1..r3); every flag comes after it (``-rot`` / ``-K`` /
    ``-KP``)."""
    assert _py_line(1, 2, 3, 4, stiffness=1e12, stiffness_p=5e11,
                    rotational=True) == (
        "ops.element('ASDEmbeddedNodeElement', 7, 50, 1, 2, 3, '4', "
        "'-rot', '-K', 1000000000000.0, '-KP', 500000000000.0)"
    )


def test_py_deck_node_order_matches_the_tcl_line() -> None:
    """The Tcl form is the oracle: same tokens, same order, the only
    difference being the quotes openseespy needs around the 4th node."""
    t = TclEmitter()
    t.embeddedNode(7, 50, 1, 2, 3, 4, stiffness=1e12)
    (tcl,) = [ln for ln in t.lines() if "ASDEmbeddedNodeElement" in ln]
    assert tcl.split()[:7] == [
        "element", "ASDEmbeddedNodeElement", "7", "50", "1", "2", "3",
    ]
    assert tcl.split()[7] == "4"
    py = _py_line(1, 2, 3, 4, stiffness=1e12)
    assert "7, 50, 1, 2, 3, '4'," in py


# ---------------------------------------------------------------------------
# The live call
# ---------------------------------------------------------------------------

def test_live_call_passes_the_4th_node_as_a_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _live_args(monkeypatch, 1, 2, 3, 4)
    assert args == (
        "ASDEmbeddedNodeElement", 7, 50, 1, 2, 3, "4", "-K", 1e12,
    )
    assert type(args[6]) is str


def test_live_call_triangle_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _live_args(monkeypatch, 1, 2, 3)
    assert args == ("ASDEmbeddedNodeElement", 7, 50, 1, 2, 3, "-K", 1e12)
    assert all(type(a) is int for a in args[1:6])
