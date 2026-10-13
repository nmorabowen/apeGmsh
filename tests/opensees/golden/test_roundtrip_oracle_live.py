"""R1b oracle, live net (K2c, #1816): a class B replay deck must run.

The static declare-before-use check in ``test_roundtrip_oracle.py`` sees
only the references the archive's ``decl_params`` name. It cannot see node
and element tags (connectivity, ``fix``, ``region -node/-ele``, loads,
recorders) or anything inside a ``proc`` body. This module is the second
net: for every ledger row that is class B, or class C for a
fatal reorder (#1814), it writes the bridge's openseespy deck
(``ops.py``) and the replayed one (``from_h5(h5()).build("py")``), and
runs each in its own subprocess.

A deck *runs* when the process exits 0 **and** prints the completion
marker appended as its last line: an OpenSees error can still exit 0, so
the marker is what proves the deck reached its end.

- A B row: the bridge deck runs, and so does the replay.
- A #1814 C row: the bridge deck runs, and the replay does not. The pin
  goes when #1814 is fixed and the row leaves class C.

A row whose bridge deck does not run on this backend must be one the
archive's solve stamp marks ``requires=("fork",)``; stock openseespy cannot
build it, so there the test asserts only that refusal, and the static check
is its only net. Any other bridge deck that does not run fails.
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel
from tests.opensees.golden.test_roundtrip_oracle import CORPUS, Model, _waivers

pytestmark = pytest.mark.live

_MARKER = "APEGMSH_K2C_DECK_COMPLETE"


def _rows(klass: str, cites: str = "") -> list[Model]:
    w = _waivers()
    return [m for m in CORPUS
            if m.id in w and w[m.id].klass == klass and cites in w[m.id].reason]


_B_ROWS = _rows("B")
_FATAL_ROWS = _rows("C", cites="#1814")


def _decks(model: Model, tmp: Path) -> tuple[Path, Path, tuple[str, ...]]:
    """The bridge's and the replay's openseespy decks of ``model``, and the
    capabilities its archive's solve stamp says a backend needs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = model.build()
        bridge = tmp / "bridge.py"
        ops.py(str(bridge), progress=False)
        archive = tmp / "model.h5"
        ops.h5(str(archive))
        replay = tmp / "replay.py"
        model_h5 = OpenSeesModel.from_h5(str(archive))
        model_h5.build("py", out=str(replay))
    stamp = model_h5.solve_stamp
    assert stamp is not None, f"{model.id}: the archive carries no solve stamp"
    return bridge, replay, tuple(stamp.requires)


def _run(deck: Path) -> tuple[bool, str]:
    """Run ``deck`` with a trailing completion marker in a subprocess."""
    text = deck.read_text(encoding="utf-8")
    deck.write_text(f"{text.rstrip()}\nprint({_MARKER!r})\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(deck)], capture_output=True, text=True,
        timeout=300, cwd=str(deck.parent),
    )
    out = proc.stdout.strip().splitlines()
    ran = proc.returncode == 0 and bool(out) and out[-1] == _MARKER
    return ran, (f"rc={proc.returncode}\nstdout:\n{proc.stdout[-1500:]}\n"
                 f"stderr:\n{proc.stderr[-1500:]}")


@pytest.fixture(autouse=True)
def _openseespy() -> None:
    pytest.importorskip("openseespy.opensees")


def _bridge_runs(model: Model, bridge: Path, requires: tuple[str, ...]) -> bool:
    """Whether the bridge deck runs; one that does not must need the fork."""
    ran, log = _run(bridge)
    assert ran or "fork" in requires, (
        f"{model.id}: the bridge deck does not run, and its solve stamp "
        f"requires {requires}:\n{log}")
    return ran


def test_the_net_runs_some_stock_row() -> None:
    assert _B_ROWS and _FATAL_ROWS, "the live net would run nothing"


@pytest.mark.parametrize("model", _B_ROWS, ids=[m.id for m in _B_ROWS])
def test_a_class_b_replay_runs_where_the_bridge_deck_runs(
        model: Model, tmp_path: Path) -> None:
    bridge, replay, requires = _decks(model, tmp_path)
    if not _bridge_runs(model, bridge, requires):
        return  # a fork-only deck on a stock backend: asserted refused above
    ran, log = _run(replay)
    assert ran, (
        f"{model.id} is class B but its replay deck does not run: a fatal "
        f"reorder the static check missed; make the row C with its bug:\n{log}")


@pytest.mark.parametrize("model", _FATAL_ROWS, ids=[m.id for m in _FATAL_ROWS])
def test_a_fatal_reorder_row_replay_does_not_run(
        model: Model, tmp_path: Path) -> None:
    bridge, replay, requires = _decks(model, tmp_path)
    assert "fork" not in requires, f"{model.id}: pin a stock-runnable row here"
    ran, log = _run(bridge)
    assert ran, f"{model.id}: the bridge deck does not run:\n{log}"
    ran, log = _run(replay)
    assert not ran, (
        f"{model.id}'s replay deck now runs: #1814 may be fixed; re-derive "
        "its ledger row")
