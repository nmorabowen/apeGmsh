"""The set verdict of ``<stem>.workspace.h5`` (ADR 0132 D3), on the schema's worked case.

``architecture/h5-schema.md`` ("The verdict"): a frame whose roof nodes 31 to 33
sit at z = 9 and base nodes 1 to 3 at z = 0, a ground node 103 on node 3's
position, and zero-length springs 501 and 502 on the pair 3-103 at (12, 0, 0).
Each test re-writes ``<stem>.h5`` the way the case describes and reads the
verdict the case states.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results.user import SetDef, WorkspaceFile, _workspace, write_workspace

from tests.results.user._workspace_helpers import (
    FEM_HASH,
    MODEL_HASH,
    column_centroid,
    frame,
)

SETS = [
    SetDef("roof", "nodes", [31, 32, 33]),
    SetDef("base", "nodes", [1, 2, 3]),
    SetDef("column line", "nodes", [1, 11, 21, 31]),
    SetDef("springs", "elements", [501, 502]),
    SetDef("columns", "elements", [201, 202, 203]),
]


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    """The frame and a workspace holding the case's sets, saved against it."""
    frame(tmp_path / "frame.h5")
    return write_workspace(tmp_path / "frame.workspace.h5", sets=SETS)


def _verdict(ws: Path, name: str):
    return WorkspaceFile(ws).verdicts().by_name(name)


@pytest.fixture
def spy(monkeypatch):
    """Counts the per-member checks the verdict runs."""
    calls: list[int] = []
    real = _workspace.match_members

    def counted(items, ids, xyz):
        calls.append(len(ids))
        return real(items, ids, xyz)

    monkeypatch.setattr(_workspace, "match_members", counted)
    return calls


def test_the_saved_rows_carry_positions_and_the_identity_hashes(saved: Path) -> None:
    ws = WorkspaceFile(saved)
    roof, springs, cols = ws.set("roof"), ws.set("springs"), ws.set("columns")
    assert roof.ids.tolist() == [31, 32, 33]
    np.testing.assert_array_equal(roof.xyz, [[0, 0, 9], [6, 0, 9], [12, 0, 9]])
    assert (roof.fem_hash, roof.model_hash) == (FEM_HASH, ""), "a node set stores no model_hash"
    np.testing.assert_array_equal(springs.xyz, [[12, 0, 0], [12, 0, 0]])
    assert springs.model_hash == MODEL_HASH
    np.testing.assert_array_equal(cols.xyz, [column_centroid(k) for k in range(3)])


def test_an_unchanged_mesh_reads_current_without_the_per_member_check(saved: Path, spy) -> None:
    table = WorkspaceFile(saved).verdicts()
    assert {v.name: v.verdict for v in table.verdicts} == {s.name: "current" for s in SETS}
    assert not table.refused
    assert all(not v.checked for v in table.verdicts)
    assert spy == [], "the per-member check ran although every identity hash is equal"
    # Even springs, whose members are ambiguous, read current: nothing moved.
    assert _verdict(saved, "springs").members == (True, True)


def test_a_relabel_moves_fem_hash_and_the_roof_reads_moved(saved: Path, spy) -> None:
    frame(saved.with_name("frame.h5"), fem_hash="b" * 32)
    roof = _verdict(saved, "roof")
    assert (roof.verdict, roof.matched, roof.total, roof.checked) == ("moved", 3, 3, True)
    assert roof.max_distance == 0.0
    assert spy, "a differing hash must run the per-member check"
    # Node 3 shares (12, 0, 0) with the ground node 103: ambiguous, so unmatched.
    base = _verdict(saved, "base")
    assert (base.verdict, base.members) == ("partial", (True, True, False))


def test_base_node_3_moved_along_x_reads_partial_2_of_3(saved: Path) -> None:
    frame(saved.with_name("frame.h5"), fem_hash="b" * 32, nodes={3: (12.5, 0.0, 0.0)})
    base = _verdict(saved, "base")
    assert (base.verdict, base.matched, base.total) == ("partial", 2, 3)
    assert base.members == (True, True, False)


def test_a_column_line_moved_up_reads_lost(saved: Path) -> None:
    moved = {n: (0.0, 0.0, z + 0.5) for n, z in ((1, 0.0), (11, 3.0), (21, 6.0), (31, 9.0))}
    frame(saved.with_name("frame.h5"), fem_hash="b" * 32, nodes=moved)
    line = _verdict(saved, "column line")
    assert (line.verdict, line.matched, line.total) == ("lost", 0, 4)


def test_springs_501_502_read_lost_after_a_declaration_swap(saved: Path) -> None:
    """``model_hash`` moves, each tag still sits at (12, 0, 0), but each position
    fits two elements: both members are ambiguous, so ``lost``, never ``moved``."""
    frame(saved.with_name("frame.h5"), model_hash="c" * 32, springs=[502, 501])
    table = WorkspaceFile(saved).verdicts()
    springs = table.by_name("springs")
    assert (springs.verdict, springs.matched, springs.total) == ("lost", 0, 2)
    assert table.by_name("columns").verdict == "moved"
    # A node set does not read model_hash: the swap leaves it current.
    assert table.by_name("roof").verdict == "current"


def test_a_declaration_swap_that_shifts_tags_never_reads_current(saved: Path) -> None:
    """``fem_hash`` is equal and the mesh unchanged; only ``model_hash`` names the
    shift, so an element set judged by ``fem_hash`` alone would read current."""
    frame(saved.with_name("frame.h5"), model_hash="c" * 32,
          beam_tags=list(range(203, 212)), springs=[201, 202])
    columns = _verdict(saved, "columns")
    assert columns.verdict in ("partial", "lost")
    assert columns.members == (False, False, False)


def test_a_model_without_fem_hash_compares_as_different(saved: Path, spy) -> None:
    frame(saved.with_name("frame.h5"), fem_hash=None)
    assert _verdict(saved, "roof").verdict == "moved"
    assert spy


def test_a_model_without_model_hash_runs_the_check_on_element_sets(saved: Path) -> None:
    frame(saved.with_name("frame.h5"), model_hash=None)
    table = WorkspaceFile(saved).verdicts()
    assert table.by_name("columns").checked
    assert table.by_name("roof").verdict == "current"


def test_an_element_set_on_a_model_without_opensees_refuses_no_tag_map(saved: Path) -> None:
    frame(saved.with_name("frame.h5"), opensees=False)
    table = WorkspaceFile(saved).verdicts()
    refused = {r.name: r.reason for r in table.refused}
    assert refused == {"springs": "no-tag-map", "columns": "no-tag-map"}
    with pytest.raises(KeyError, match="no-tag-map"):
        table.by_name("springs")
    assert table.by_name("roof").verdict == "current"


def test_a_tag_now_naming_a_coincident_pair_member_is_not_matched(saved: Path) -> None:
    """Node 103 is placed off node 3: base node 3 is then unambiguous again."""
    frame(saved.with_name("frame.h5"), fem_hash="b" * 32, nodes={103: (12.0, -1.0, 0.0)})
    assert _verdict(saved, "base").verdict == "moved"


def test_the_tolerance_is_relative_to_the_bounding_box_diagonal(saved: Path) -> None:
    diagonal = float(np.linalg.norm([12.0, 0.0, 9.0]))
    inside, outside = 0.5e-9 * diagonal, 2e-9 * diagonal
    frame(saved.with_name("frame.h5"), fem_hash="b" * 32,
          nodes={31: (inside, 0.0, 9.0), 32: (6.0 + outside, 0.0, 9.0)})
    roof = _verdict(saved, "roof")
    assert roof.members == (True, False, True)
    assert roof.max_distance == pytest.approx(inside, rel=1e-6)


def test_verdicts_name_the_paired_model_or_refuse_unpaired(saved: Path, tmp_path: Path) -> None:
    other = frame(tmp_path / "other.h5", fem_hash="d" * 32)
    assert WorkspaceFile(saved).verdicts(other).by_name("roof").verdict == "moved"
    saved.with_name("frame.h5").unlink()
    with pytest.raises(_workspace.WorkspaceRefused) as err:
        WorkspaceFile(saved).verdicts()
    assert err.value.reason == "unpaired"
