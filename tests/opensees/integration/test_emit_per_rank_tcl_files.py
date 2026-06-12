"""Integration tests for ADR 0061 per-rank Tcl deck emission.

``apeSees.tcl(path, per_rank=True)`` writes a driver deck plus one
``ranks/rank<K>_<seq>.tcl`` fragment per ``if {[getPID] == K} { ... }``
block; the driver guards each fragment behind a one-line ``source``.

The load-bearing assertion is **reassembly parity**: re-inlining every
fragment back into its guard block (re-adding the 4-space indent the
way ``_LineBuf`` applied it) must reproduce the monolithic deck
line-for-line. That single check proves the split is layout-only —
no content moved across a rank boundary, no global line was captured
into a fragment, and ordering (including the staged skeleton) is
preserved.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    make_two_column_frame,
    make_two_column_frame_partitioned,
)
from tests.opensees.integration.test_emit_partitioned_staged import (
    _make_4quad_2pg_2part_fem,
    _setup_partitioned_staged_ops,
)


_SOURCE_GUARD = re.compile(
    r"^if \{\[getPID\] == (\d+)\} \{ source \[file join "
    r"\[file dirname \[info script\]\] ranks (rank\d+_\d+\.tcl)\] \}$"
)


def _make_partitioned_ops(fem: FEMStub) -> apeSees:
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    return ops


def _reassemble(driver_path: Path) -> list[str]:
    """Re-inline every sourced fragment into its guard block.

    Mirrors what ``_write_per_rank_tcl`` undid: guard-source line →
    ``if {[getPID] == K} {`` + body re-indented 4 spaces (every line,
    including blanks — matching ``_LineBuf.append``) + ``}``. The
    trailing blank after the guard line is kept as the block's
    trailing blank.
    """
    ranks_dir = driver_path.parent / "ranks"
    out: list[str] = []
    for line in driver_path.read_text(encoding="utf-8").splitlines():
        m = _SOURCE_GUARD.match(line)
        if m is None:
            out.append(line)
            continue
        rank, fname = m.group(1), m.group(2)
        frag = (ranks_dir / fname).read_text(encoding="utf-8").splitlines()
        assert frag[0].startswith("# apeGmsh per-rank fragment"), (
            f"{fname} must start with the fragment banner; got {frag[0]!r}"
        )
        out.append("if {[getPID] == " + rank + "} {")
        out.extend("    " + body_line for body_line in frag[1:])
        out.append("}")
    return out


def _assert_per_rank_parity(ops_factory, tmp_path: Path) -> None:
    """Emit monolithic + per-rank decks from identically-built models
    and assert reassembly parity plus the fragment-content invariants.

    ``ops_factory`` is called twice — ``tcl()`` consumes a build, so
    each emit gets its own apeSees instance over the same FEM stub.
    """
    mono_path = tmp_path / "mono" / "main.tcl"
    pr_path = tmp_path / "per_rank" / "main.tcl"
    mono_path.parent.mkdir()
    pr_path.parent.mkdir()

    ops_factory().tcl(str(mono_path))
    ops_factory().tcl(str(pr_path), per_rank=True)

    mono = mono_path.read_text(encoding="utf-8").splitlines()
    assert _reassemble(pr_path) == mono

    # Fragment-content invariants: rank-local files carry no guards
    # (the driver owns sequencing) and no global shim.
    for frag in sorted((pr_path.parent / "ranks").glob("rank*.tcl")):
        text = frag.read_text(encoding="utf-8")
        assert "getPID" not in text, (
            f"{frag.name} contains a getPID guard — per-rank fragments "
            "must be guard-free (the driver owns the guards)."
        )


def test_per_rank_reassembly_parity_base_partitioned(
    tmp_path: Path,
) -> None:
    """2-partition frame: driver + fragments reassemble to the
    monolithic deck line-for-line."""
    _assert_per_rank_parity(
        lambda: _make_partitioned_ops(make_two_column_frame_partitioned()),
        tmp_path,
    )


def test_per_rank_reassembly_parity_staged_partitioned(
    tmp_path: Path,
) -> None:
    """Staged + partitioned (SSI-2.C fixture: stage-activated cimbra +
    initial_stress): every per-rank block — base topology, stage
    topology, addToParameter fan-out — splits and reassembles cleanly,
    and the sequential stage skeleton stays in the driver."""
    _assert_per_rank_parity(
        lambda: _setup_partitioned_staged_ops(_make_4quad_2pg_2part_fem()),
        tmp_path,
    )

    # The staged fixture must actually exercise multiple blocks for a
    # rank (base + stage) — otherwise this test silently degrades to
    # the base case.
    ranks_dir = tmp_path / "per_rank" / "ranks"
    names = sorted(p.name for p in ranks_dir.glob("rank*.tcl"))
    assert any(n.startswith("rank1_1") for n in names), (
        f"expected rank 1 to have a second block (stage fragment); "
        f"got {names}"
    )


def test_per_rank_driver_keeps_globals_and_stage_skeleton(
    tmp_path: Path,
) -> None:
    """Globals (materials, shim, analysis chain) and the sequential
    stage skeleton (domainChange, loadConst, wipeAnalysis) live in the
    driver, never in fragments."""
    path = tmp_path / "main.tcl"
    _setup_partitioned_staged_ops(_make_4quad_2pg_2part_fem()).tcl(
        str(path), per_rank=True,
    )
    driver = path.read_text(encoding="utf-8")
    assert "nDMaterial ElasticIsotropic" in driver
    assert "proc getPID" in driver
    assert "domainChange" in driver
    fragments = sorted((tmp_path / "ranks").glob("rank*.tcl"))
    assert fragments, "per-rank emit must write at least one fragment"
    for frag in fragments:
        text = frag.read_text(encoding="utf-8")
        assert "nDMaterial" not in text
        assert "domainChange" not in text
        assert "numberer" not in text


def test_per_rank_requires_partitioned_model(tmp_path: Path) -> None:
    """Unpartitioned model + per_rank=True fails loud."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    with pytest.raises(ValueError, match="per_rank=True requires"):
        ops.tcl(str(tmp_path / "main.tcl"), per_rank=True)
    assert not (tmp_path / "main.tcl").exists()


def test_per_rank_and_split_mutually_exclusive(tmp_path: Path) -> None:
    """split=True + per_rank=True fails loud before building."""
    ops = _make_partitioned_ops(make_two_column_frame_partitioned())
    with pytest.raises(ValueError, match="mutually\\s+exclusive"):
        ops.tcl(str(tmp_path / "main.tcl"), split=True, per_rank=True)
