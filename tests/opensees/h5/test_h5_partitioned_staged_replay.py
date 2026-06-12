"""Flat replay of PARTITIONED staged archives (ADR 0055 Phase 5 / P5.2).

Post-P5.1 the stage buckets of a partitioned staged archive are
rank-agnostic (flat-equivalent), so the existing staged replay
(``compose._replay_staged_into``) accepts them as-is and re-emits a
FLAT single-process staged deck — the same degrade the non-staged
partitioned path has always had.  This module locks that acceptance:

1. ``build('tcl')`` / ``build('py')`` of a partitioned staged archive
   succeed and produce a FLAT deck — stage blocks present, no
   ``getPID`` rank brackets.
2. The replayed deck is CONTENT-equal (line multiset) to the
   unpartitioned archive's replay of the same model — the archive
   carries the flat logical program, so the two replays may differ
   only in line ORDER (rank-major vs target-major capture, all within
   the same ``domainChange`` barriers).
3. The P5.1 dedupe survives end-to-end: a cross-rank shared-node HOLD
   emits exactly one ``sp ... -const`` line; a PG load spanning ranks
   emits exactly one ``load`` line per node.
4. (live) The replayed py deck RUNS under single-process OpenSees —
   the INV-5 runtime conditional and the getPID shim make the flat
   deck portable by construction.
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from apeGmsh.opensees import OpenSeesModel

from tests.opensees.h5.test_h5_partitioned_staged_capture import (
    _flat_bridge,
    _partitioned_bridge,
)


_MP_AUTO_EMIT_FILTERS = (
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


def _replayed_tcl(tmp_path: Path, *, partitioned: bool) -> str:
    archive = tmp_path / ("part.h5" if partitioned else "flat.h5")
    bridge = _partitioned_bridge() if partitioned else _flat_bridge()
    bridge.h5(str(archive))
    out = OpenSeesModel.from_h5(str(archive)).build("tcl")
    assert isinstance(out, str)
    return out


def test_replay_tcl_is_flat_staged_deck(tmp_path: Path) -> None:
    tcl = _replayed_tcl(tmp_path, partitioned=True)
    assert "getPID" not in tcl, (
        "flat replay must not reproduce rank brackets (the partitioned "
        "re-emit is P5.4, demand-gated)"
    )
    # Both stages re-drive: their analyze loops and the between-stage
    # reset are present.
    assert "loadConst -time 0.0" in tcl
    assert "wipeAnalysis" in tcl
    assert tcl.count("analyze") >= 2


def test_replay_py_is_flat_staged_deck(tmp_path: Path) -> None:
    archive = tmp_path / "part.h5"
    _partitioned_bridge().h5(str(archive))
    py = OpenSeesModel.from_h5(str(archive)).build("py")
    assert isinstance(py, str)
    assert "getPID() ==" not in py, (
        "flat replay must not reproduce rank brackets"
    )
    assert "wipeAnalysis" in py


def test_replay_content_equals_unpartitioned_replay(tmp_path: Path) -> None:
    """Line MULTISET equality — the partitioned archive replays the
    same flat program the unpartitioned archive does; only intra-stage
    ordering may differ (rank-major capture)."""
    tcl_part = _replayed_tcl(tmp_path, partitioned=True)
    tcl_flat = _replayed_tcl(tmp_path, partitioned=False)

    def _lines(text: str) -> Counter:
        return Counter(
            ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )

    assert _lines(tcl_part) == _lines(tcl_flat)


def test_shared_node_hold_and_loads_emit_once(tmp_path: Path) -> None:
    """End-to-end P5.1 dedupe proof on the replayed deck: the
    cross-rank shared HOLD node (5) yields ONE ``sp 5 2 ... -const``
    line; each ``Fill`` node's load yields ONE ``load`` line."""
    tcl = _replayed_tcl(tmp_path, partitioned=True)

    hold_lines = re.findall(r"^\s*sp\s+(\d+)\s+\d+.*-const\s*$",
                            tcl, re.MULTILINE)
    assert sorted(hold_lines) == ["5", "6"], (
        f"expected one HOLD sp per FillTop node, got {hold_lines!r}"
    )
    load_nodes = re.findall(r"^\s*load\s+(\d+)\s", tcl, re.MULTILINE)
    assert sorted(load_nodes) == ["3", "4", "5", "6"], (
        f"expected one load per Fill node, got {load_nodes!r}"
    )


@pytest.mark.live
def test_replayed_py_deck_runs_single_process(tmp_path: Path) -> None:
    """The flat replay of a partitioned staged archive RUNS under
    single-process OpenSees (subprocess exec of the py deck)."""
    pytest.importorskip("openseespy.opensees")
    archive = tmp_path / "part.h5"
    _partitioned_bridge().h5(str(archive))
    deck = tmp_path / "deck.py"
    OpenSeesModel.from_h5(str(archive)).build("py", out=str(deck))

    proc = subprocess.run(
        [sys.executable, str(deck)],
        capture_output=True, text=True, timeout=300,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, (
        f"replayed deck failed (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
