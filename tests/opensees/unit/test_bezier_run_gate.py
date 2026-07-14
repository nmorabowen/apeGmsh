"""B3 — fork-build run gate for the Bézier elements.

``LiveOpsEmitter.element`` raises a clear *"requires the Ladruno fork
build"* error when a deck containing ``BezierTri6`` / ``BezierTet10`` is
run in-process on a build that does not know the element — instead of the
cryptic openseespy error (or, worse, a silent no-op that fails much later).
Deck emission (``ops.tcl`` / ``ops.py``) is unaffected; only the in-process
run is gated.

These unit tests are fork-free: they drive ``LiveOpsEmitter.element``
against fake ``ops`` objects that model the two stock-build failure modes
(raise vs silently-drop) and the fork-build success path.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.emitter.live import _FORK_ONLY_ELEMENTS, LiveOpsEmitter

_FORK_ELEMS = sorted(_FORK_ONLY_ELEMENTS)


def _bare_emitter(ops: object) -> LiveOpsEmitter:
    """A ``LiveOpsEmitter`` with a fake ops, bypassing ``__init__`` (so no
    openseespy import / wipe). Only the attributes the gate reads are set."""
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[attr-defined]
    le._in_partition = False
    le._fork_verified_types = set()
    return le


class _RaiseOps:
    """Stock build that raises on an unknown element type."""

    def element(self, *_args: object) -> None:
        raise RuntimeError("unknown element type")


class _SilentOps:
    """Stock build that warns + returns without creating the element."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def element(self, *args: object) -> None:
        self.calls.append(args)

    def getEleTags(self) -> list[int]:
        return []  # element was never created


class _ForkOps:
    """Fork build: the element is created and shows up in getEleTags."""

    def __init__(self) -> None:
        self.tags: list[int] = []

    def element(self, _etype: str, tag: int, *_args: object) -> None:
        self.tags.append(tag)

    def getEleTags(self) -> list[int]:
        return list(self.tags)


@pytest.mark.parametrize("etype", _FORK_ELEMS)
def test_gate_raises_friendly_when_build_raises(etype: str) -> None:
    le = _bare_emitter(_RaiseOps())
    with pytest.raises(RuntimeError, match="requires the Ladruno fork build"):
        le.element(etype, 1, 1)


@pytest.mark.parametrize("etype", _FORK_ELEMS)
def test_gate_raises_friendly_when_build_silently_drops(etype: str) -> None:
    le = _bare_emitter(_SilentOps())
    with pytest.raises(RuntimeError, match="requires the Ladruno fork build"):
        le.element(etype, 7, 1)


@pytest.mark.parametrize("etype", _FORK_ELEMS)
def test_gate_passes_and_caches_when_built(etype: str) -> None:
    ops = _ForkOps()
    le = _bare_emitter(ops)
    le.element(etype, 1, 1)
    assert etype in le._fork_verified_types
    # Second element of the SAME fork type: cached, so getEleTags is NOT
    # probed again — prove it by making a re-probe blow up.
    ops.getEleTags = None  # type: ignore[assignment]
    le.element(etype, 2, 1)  # must not raise
    assert ops.tags == [1, 2]


def test_non_fork_element_is_not_gated() -> None:
    # A stock-shaped ops with empty getEleTags would FAIL the gate if it were
    # applied — but a non-fork element must pass straight through, unprobed.
    ops = _SilentOps()
    le = _bare_emitter(ops)
    le.element("quad", 1, 1, 2, 3, 4, 0.1, "PlaneStress", 1)
    assert ops.calls == [("quad", 1, 1, 2, 3, 4, 0.1, "PlaneStress", 1)]


def test_gate_skipped_inside_partition_block() -> None:
    # In a non-zero partition block the ops is a no-op stand-in with no real
    # domain — the gate must NOT probe it (would false-positive).
    ops = _SilentOps()
    le = _bare_emitter(ops)
    le._in_partition = True
    le.element("BezierTri6", 1, 1)  # no raise despite empty getEleTags
    assert ops.calls == [("BezierTri6", 1, 1)]
    assert le._fork_verified_types == set()
