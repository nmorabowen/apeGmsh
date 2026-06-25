"""Unit tests for ``partition_open`` / ``partition_close`` (ADR 0027, P4).

The two emission-scoping methods bracket a per-rank emission block.
Tcl wraps the block in ``if {[getPID] == K} { ... }``; Py wraps it in
``if getPID() == K: ...``; LiveOps treats itself as rank 0 and
suppresses non-zero rank emissions; Recording captures both events.

Tcl / Py also emit a one-shot runtime shim (defining a ``getPID``
fallback that returns 0) on the first ``partition_open`` call across
the emitter's lifetime, so single-process OpenSees still runs the
deck. The shim must emit **exactly once** regardless of how many
partition blocks are opened.

Per ADR 0027 INV-1: the per-rank emission text is byte-identical
across owning ranks. These tests assert the bracket *shape*; the
build-time fan-out invariants are exercised in the P5 integration
suite.
"""
from __future__ import annotations

import inspect
import sys
import types
import warnings
from typing import Any, get_type_hints
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter


# ---------------------------------------------------------------------------
# 1. Protocol contract
# ---------------------------------------------------------------------------


def test_emitter_protocol_has_partition_open_close() -> None:
    """``Emitter`` Protocol declares both methods with the right signatures."""
    # Both attributes exist on the Protocol.
    assert hasattr(Emitter, "partition_open"), (
        "Emitter Protocol must declare partition_open (ADR 0027, P4)."
    )
    assert hasattr(Emitter, "partition_close"), (
        "Emitter Protocol must declare partition_close (ADR 0027, P4)."
    )

    # partition_open(self, rank: int) -> None
    open_hints = get_type_hints(Emitter.partition_open)
    assert open_hints.get("rank") is int
    assert open_hints.get("return") is type(None)

    # partition_close(self) -> None
    close_hints = get_type_hints(Emitter.partition_close)
    assert close_hints.get("return") is type(None)

    # Positional parameter shape (rank is the sole non-self parameter).
    open_sig = inspect.signature(Emitter.partition_open)
    params = list(open_sig.parameters.values())
    assert [p.name for p in params] == ["self", "rank"]

    close_sig = inspect.signature(Emitter.partition_close)
    assert [p.name for p in close_sig.parameters.values()] == ["self"]


# ---------------------------------------------------------------------------
# 2. RecordingEmitter mirrors the Protocol
# ---------------------------------------------------------------------------


def test_recording_emitter_has_partition_open_close_methods() -> None:
    """RecordingEmitter records both bracket events with the
    ``(name, args, kwargs)`` shape used by every other Protocol method.
    """
    e = RecordingEmitter()
    assert callable(getattr(e, "partition_open", None))
    assert callable(getattr(e, "partition_close", None))

    e.partition_open(0)
    e.node(1, 0.0, 0.0, 0.0)
    e.partition_close()
    e.partition_open(2)
    e.partition_close()

    assert e.calls == [
        ("partition_open", (0,), {}),
        ("node", (1, 0.0, 0.0, 0.0), {}),
        ("partition_close", (), {}),
        ("partition_open", (2,), {}),
        ("partition_close", (), {}),
    ]


# ---------------------------------------------------------------------------
# 3-4. TclEmitter — round-trip, shim once
# ---------------------------------------------------------------------------


def test_tcl_emitter_partition_open_close_round_trip() -> None:
    """Two partition blocks bracket their content with
    ``if {[getPID] == K} { ... }``; the shim line precedes the first
    block exactly once.
    """
    e = TclEmitter()
    e.partition_open(0)
    e.node(1, 0.0, 0.0, 0.0)
    e.partition_close()
    e.partition_open(1)
    e.node(2, 1.0, 0.0, 0.0)
    e.partition_close()

    text = "\n".join(e.lines())
    # Shim present.
    assert "proc getPID {} { return 0 }" in text
    # Both rank guards present in order.
    rank0_idx = text.index("if {[getPID] == 0} {")
    rank1_idx = text.index("if {[getPID] == 1} {")
    assert rank0_idx < rank1_idx

    # Nodes appear inside their respective blocks, indented 4 spaces.
    assert "    node 1 0.0 0.0 0.0" in text
    assert "    node 2 1.0 0.0 0.0" in text

    # Each block has its own closing brace and a trailing blank line.
    lines = e.lines()
    # Two opener lines + two closer lines.
    openers = [
        i for i, ln in enumerate(lines)
        if ln.startswith("if {[getPID] == ")
    ]
    closers = [i for i, ln in enumerate(lines) if ln == "}"]
    assert len(openers) == 2
    # At least two non-section closing braces (closing each partition).
    assert len(closers) >= 2

    # rank-0 content is inside the rank-0 block.
    o0, c0 = openers[0], closers[0]
    assert o0 < c0
    assert "    node 1 0.0 0.0 0.0" in lines[o0 + 1 : c0]

    # rank-1 content is inside the rank-1 block.
    o1 = openers[1]
    c1 = next(i for i in closers if i > o1)
    assert "    node 2 1.0 0.0 0.0" in lines[o1 + 1 : c1]


def test_tcl_emitter_shim_emitted_once() -> None:
    """The ``proc getPID`` shim line appears **exactly once** even
    when partition_open is called many times."""
    e = TclEmitter()
    e.partition_open(0)
    e.partition_close()
    e.partition_open(1)
    e.partition_close()
    e.partition_open(2)
    e.partition_close()

    shim_count = sum(
        1 for ln in e.lines() if "proc getPID {} { return 0 }" in ln
    )
    assert shim_count == 1, (
        f"Expected exactly one ``proc getPID`` shim line, got "
        f"{shim_count}. Lines:\n" + "\n".join(e.lines())
    )


def test_tcl_emitter_shim_uses_info_commands_guard() -> None:
    """The shim must guard with ``info commands``, NOT ``info procs``:
    ``if {[info commands getPID] == ""} { proc getPID {} { return 0 } }``.

    OpenSeesMP registers ``getPID`` via ``Tcl_CreateCommand`` (a C
    command, invisible to ``info procs``). An ``info procs`` guard
    therefore overrides the native command with the rank-0 fallback and
    every MPI rank builds rank 0's submodel — run-verified under
    ``mpiexec -n 8 OpenSeesMP`` (8 identical part files) before the fix.
    ``info commands`` sees both the native command and a prior proc, so
    the shim stays a single-process-only fallback.
    """
    e = TclEmitter()
    e.partition_open(0)
    e.partition_close()

    # The shim line is a single Tcl statement that guards on
    # whether ``getPID`` is already defined.
    shim_line = next(
        ln for ln in e.lines()
        if "proc getPID" in ln
    )
    assert "info commands getPID" in shim_line
    assert "info procs" not in shim_line
    assert "proc getPID {} { return 0 }" in shim_line


def test_tcl_emitter_partition_block_is_byte_identical_across_ranks() -> None:
    """ADR 0027 INV-1: per-rank text is byte-identical bar the rank int."""
    e = TclEmitter()
    for rank in (0, 1, 2):
        e.partition_open(rank)
        e.node(99, 1.0, 2.0, 3.0)
        e.partition_close()

    lines = e.lines()
    # Extract each opener + content + closer.
    blocks: list[list[str]] = []
    cur: list[str] = []
    in_block = False
    for ln in lines:
        if ln.startswith("if {[getPID] == "):
            in_block = True
            cur = [ln]
            continue
        if in_block:
            cur.append(ln)
            if ln == "}":
                blocks.append(cur)
                cur = []
                in_block = False
    assert len(blocks) == 3
    # Strip the rank integer; the rest must be identical across blocks.
    import re
    stripped = [
        [re.sub(r"== \d+", "== K", ln) for ln in b]
        for b in blocks
    ]
    assert stripped[0] == stripped[1] == stripped[2]


# ---------------------------------------------------------------------------
# 5-6. PyEmitter — round-trip, shim once
# ---------------------------------------------------------------------------


def test_py_emitter_partition_open_close_round_trip() -> None:
    """Two partition blocks bracket their content with
    ``if getPID() == K:`` and indented body. The shim defines a
    ``getPID()`` fallback wrapped in a ``try / except ImportError``
    once before the first block.
    """
    e = PyEmitter()
    e.partition_open(0)
    e.node(1, 0.0, 0.0, 0.0)
    e.partition_close()
    e.partition_open(1)
    e.node(2, 1.0, 0.0, 0.0)
    e.partition_close()

    lines = e.lines()
    text = "\n".join(lines)

    # Shim block — must wrap the import in try/except ImportError.
    assert "try:" in text
    assert "from openseespy.opensees import getPID" in text
    assert "except ImportError:" in text
    assert "def getPID()" in text
    assert "return 0" in text

    # Both rank guards present in order.
    rank0_idx = text.index("if getPID() == 0:")
    rank1_idx = text.index("if getPID() == 1:")
    assert rank0_idx < rank1_idx

    # Nodes indented inside their blocks (4 spaces).
    assert "    ops.node(1, 0.0, 0.0, 0.0)" in text
    assert "    ops.node(2, 1.0, 0.0, 0.0)" in text

    # No Tcl-style closing braces (Python uses indentation).
    # Allow the brace to appear in shim text if any; assert no bare ``}`` line.
    assert not any(ln.strip() == "}" for ln in lines)


def test_py_emitter_shim_emitted_once() -> None:
    """The ``def getPID()`` shim definition appears **exactly once**."""
    e = PyEmitter()
    e.partition_open(0)
    e.partition_close()
    e.partition_open(1)
    e.partition_close()
    e.partition_open(2)
    e.partition_close()

    text = "\n".join(e.lines())
    # ``def getPID()`` appears in the fallback definition exactly once.
    assert text.count("def getPID()") == 1
    # ``from openseespy.opensees import getPID`` also exactly once.
    assert text.count("from openseespy.opensees import getPID") == 1
    # ``try:`` / ``except ImportError:`` pair exactly once.
    assert text.count("except ImportError:") == 1


def test_py_emitter_partition_close_does_not_emit_closing_brace() -> None:
    """Python uses indentation, not braces — ``partition_close`` must
    not emit any ``}`` token."""
    e = PyEmitter()
    e.partition_open(0)
    e.partition_close()
    for ln in e.lines():
        assert ln.strip() != "}"


# ---------------------------------------------------------------------------
# 7-9. LiveOpsEmitter — rank 0 passes through, non-zero suppresses,
# one warning per instance.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_ops(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``openseespy.opensees`` module that records calls.

    Each attribute access returns a closure that appends ``(name, args)``
    to ``module.call_log``. The module is an actual ``types.ModuleType``
    so ``isinstance(mod, ModuleType)`` in ``LiveOpsEmitter._get_ops``
    succeeds.
    """
    mod = types.ModuleType("openseespy.opensees")
    mod.call_log = []  # type: ignore[attr-defined]

    def _make_recorder(name: str) -> Any:
        def _fn(*args: Any, **kwargs: Any) -> int:
            mod.call_log.append((name, args, kwargs))  # type: ignore[attr-defined]
            return 0
        return _fn

    # Pre-populate every method LiveOpsEmitter calls in this file.
    for name in (
        "wipe", "model", "node", "fix", "mass", "equalDOF",
        "rigidLink", "rigidDiaphragm", "element", "region",
        "uniaxialMaterial", "nDMaterial", "section", "geomTransf",
        "patch", "fiber", "layer", "beamIntegration", "timeSeries",
        "pattern", "load", "eleLoad", "sp", "recorder", "constraints",
        "numberer", "system", "test", "algorithm", "integrator",
        "analysis", "analyze", "eigen",
    ):
        setattr(mod, name, _make_recorder(name))

    parent = types.ModuleType("openseespy")
    parent.opensees = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openseespy", parent)
    monkeypatch.setitem(sys.modules, "openseespy.opensees", mod)

    # Pin the live emitter's backend to THIS fake for the test's duration.
    # ``LiveOpsEmitter`` resolves its backend through the module-level
    # ``_get_ops`` cache (``live._OPS_CACHE``), which is populated once per
    # process and survives across tests — so a backend resolved by an earlier
    # test (the real fork via ``get_backend_name`` in conftest, or a previous
    # ``fake_ops``) would otherwise be reused here and this fixture's fake
    # ignored. Patching ``_get_ops`` directly bypasses both the cache and the
    # resolution order (which prefers a bare ``import opensees`` fork over
    # ``openseespy.opensees``), so every test gets its own isolated fake.
    # monkeypatch restores the original on teardown.
    from apeGmsh.opensees.emitter import live as live_mod
    monkeypatch.setattr(live_mod, "_get_ops", lambda: mod)
    return mod


def test_live_emitter_rank_0_passes_through(
    fake_ops: types.ModuleType,
) -> None:
    """``partition_open(0)`` passes through — ``ops.node(...)`` reaches
    the real (fake) openseespy module."""
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    e = LiveOpsEmitter(wipe=False)
    fake_ops.call_log.clear()  # type: ignore[attr-defined]

    e.partition_open(0)
    e.node(7, 1.0, 2.0, 3.0)
    e.partition_close()

    calls = fake_ops.call_log  # type: ignore[attr-defined]
    assert ("node", (7, 1.0, 2.0, 3.0), {}) in calls


def test_live_emitter_non_zero_rank_suppresses(
    fake_ops: types.ModuleType,
) -> None:
    """``partition_open(K!=0)`` suppresses emission until
    ``partition_close``; emits a UserWarning on the first non-zero call.
    """
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    e = LiveOpsEmitter(wipe=False)
    fake_ops.call_log.clear()  # type: ignore[attr-defined]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e.partition_open(5)
        e.node(7, 1.0, 2.0, 3.0)
        e.partition_close()

    # The node call did NOT reach the fake module.
    calls = fake_ops.call_log  # type: ignore[attr-defined]
    assert not any(c[0] == "node" for c in calls), (
        f"node should not have reached openseespy under rank=5; got: {calls}"
    )
    # Exactly one UserWarning fired.
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    msg = str(user_warnings[0].message)
    assert "single-process" in msg
    assert "rank=5" in msg


def test_live_emitter_warns_once_per_instance(
    fake_ops: types.ModuleType,
) -> None:
    """A single LiveOpsEmitter instance fires the non-zero-rank
    ``UserWarning`` **at most once** across multiple non-zero
    ``partition_open`` calls."""
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    e = LiveOpsEmitter(wipe=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e.partition_open(1)
        e.partition_close()
        e.partition_open(2)
        e.partition_close()
        e.partition_open(3)
        e.partition_close()

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1, (
        f"Expected exactly one UserWarning across three non-zero "
        f"partition_open calls, got {len(user_warnings)}: "
        f"{[str(w.message) for w in user_warnings]}"
    )


def test_live_emitter_restores_real_ops_after_close(
    fake_ops: types.ModuleType,
) -> None:
    """After ``partition_close`` of a non-zero block, subsequent
    emit calls reach the real ops module again (the suppression flag
    must reset)."""
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    e = LiveOpsEmitter(wipe=False)
    fake_ops.call_log.clear()  # type: ignore[attr-defined]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e.partition_open(1)
        e.node(1, 0.0, 0.0, 0.0)
        e.partition_close()

    # Now emit outside any partition block — must reach the fake module.
    e.node(2, 0.0, 0.0, 0.0)
    calls = fake_ops.call_log  # type: ignore[attr-defined]
    assert ("node", (2, 0.0, 0.0, 0.0), {}) in calls
    assert ("node", (1, 0.0, 0.0, 0.0), {}) not in calls


def test_live_emitter_rank_0_does_not_warn(
    fake_ops: types.ModuleType,
) -> None:
    """``partition_open(0)`` is the pass-through case — no warning."""
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    e = LiveOpsEmitter(wipe=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e.partition_open(0)
        e.partition_close()
        e.partition_open(0)
        e.partition_close()

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0
