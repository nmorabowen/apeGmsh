"""H5 capture dedupe under partition brackets (ADR 0055 Phase 5 / P5.0a).

The global partitioned pass replicates records across owning ranks by
design (ADR 0027 INV-4 fan-out): a ``fix`` on a shared boundary node
emits inside BOTH ranks' ``partition_open`` brackets; a cross-rank MP
constraint replicates byte-identically on every owning rank (INV-1);
a ``Plain`` pattern re-opens the SAME tag once per owning rank with
rank-filtered lines.

The H5 emitter is a *capture* target, not a streaming deck: its file
shape is the flat logical model (``/opensees/bcs/fix`` rows,
``/opensees/constraints/*`` rows, one ``/opensees/patterns/<type>_<tag>``
group per pattern).  Before P5.0a the replicated emission leaked into
that capture verbatim:

* a shared-node ``fix`` / ``mass`` landed TWICE in ``/opensees/bcs/*``
  (flat replay then double-applies the SP — doubled penalty stiffness);
* a cross-rank ``equalDOF`` landed twice in
  ``/opensees/constraints/equalDOF``;
* a rank-spanning ``Plain`` pattern CRASHED the write outright — two
  ``_PatternRecord``s with the same tag collide on the
  ``patterns/Plain_<tag>`` group name.

These tests lock the partition-aware capture: while a partition
bracket is open, replicated global captures dedupe on full record
identity, and a pattern re-open of an already-captured tag RESUMES
that record (merging the per-rank line subsets).  Flat-build behavior
is unchanged (last test).
"""
from __future__ import annotations

import os
import tempfile
from typing import cast

import h5py
import pytest

from apeGmsh._kernel.records._constraints import NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.h5 import H5Emitter

from tests.opensees.fixtures.fem_stub import make_two_column_frame


# The partitioned builds below declare no MP-friendly chain — the
# ADR 0027 INV-5 auto-emit warnings are contracted behavior locked by
# other tests and would otherwise drown these assertions.
_MP_AUTO_EMIT_FILTERS = (
    "ignore:MP constraints are present in the model:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared numberer:UserWarning",
    "ignore:len.fem.partitions. > 1 with no user-declared system:UserWarning",
)
pytestmark = [pytest.mark.filterwarnings(f) for f in _MP_AUTO_EMIT_FILTERS]


def _shared_node_fem():
    """Two-column frame, 2 ranks, node 2 SHARED across both ranks.

    Rank 0 owns nodes {1, 2} + element 1; rank 1 owns nodes {2, 3, 4}
    + element 2.  Node 2 is the cross-rank boundary node — any fix /
    mass / load on it replicates into both rank brackets.
    """
    fem = make_two_column_frame()
    fem.set_partitions([
        (0, [1, 2], [1]),
        (1, [2, 3, 4], [2]),
    ])
    return fem


def _bridge(fem) -> apeSees:
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    return ops


def _emit_to_file(ops: apeSees, path: str) -> None:
    bm = ops.build()
    emitter = H5Emitter(model_name="dedupe_test", snapshot_id="")
    bm.emit(emitter)
    emitter.write(path)


def test_shared_node_fix_captured_once() -> None:
    """A fix on a cross-rank shared node emits in BOTH rank brackets
    but must land exactly ONCE in ``/opensees/bcs/fix``."""
    ops = _bridge(_shared_node_fem())
    ops.fix(nodes=[2], dofs=(1, 1, 1, 1, 1, 1))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.h5")
        _emit_to_file(ops, path)
        with h5py.File(path, "r") as f:
            rows = f["/opensees/bcs/fix"][:]
            targets = [r["target"].decode() for r in rows]
            assert targets.count("2") == 1, (
                "shared-node fix must capture once, got "
                f"{targets.count('2')} rows for node 2 ({targets!r})"
            )


def test_shared_node_mass_captured_once() -> None:
    """A mass on a cross-rank shared node lands exactly once in
    ``/opensees/bcs/mass``."""
    ops = _bridge(_shared_node_fem())
    ops.mass(nodes=[2], values=(5.0, 5.0, 5.0, 0.0, 0.0, 0.0))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.h5")
        _emit_to_file(ops, path)
        with h5py.File(path, "r") as f:
            rows = f["/opensees/bcs/mass"][:]
            targets = [r["target"].decode() for r in rows]
            assert targets.count("2") == 1, (
                "shared-node mass must capture once, got "
                f"{targets.count('2')} rows ({targets!r})"
            )


def test_cross_rank_equal_dof_captured_once() -> None:
    """A cross-rank equalDOF replicates on both owning ranks (ADR 0027
    INV-1) but must land exactly once in
    ``/opensees/constraints/equalDOF``."""
    fem = _shared_node_fem()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3],
            name="cross_equal_dof",
        ),
    ])
    ops = _bridge(fem)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.h5")
        _emit_to_file(ops, path)
        with h5py.File(path, "r") as f:
            rows = f["/opensees/constraints/equalDOF"][:]
            pairs = [(int(r["master"]), int(r["slave"])) for r in rows]
            assert pairs.count((2, 4)) == 1, (
                "cross-rank equalDOF must capture once, got "
                f"{pairs!r}"
            )


def test_rank_spanning_pattern_writes_single_merged_group() -> None:
    """A Plain pattern with loads on rank-0-only AND rank-1-only nodes
    re-opens the same tag once per rank.  Before P5.0a this CRASHED
    the write (``patterns/Plain_<tag>`` group-name collision); now the
    re-open resumes the captured record and the file carries ONE
    pattern group holding BOTH load rows."""
    ops = _bridge(_shared_node_fem())
    series = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=series) as p:
        p.load(node=1, forces=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))  # rank 0
        p.load(node=4, forces=(0.0, 2.0, 0.0, 0.0, 0.0, 0.0))  # rank 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.h5")
        _emit_to_file(ops, path)
        with h5py.File(path, "r") as f:
            patterns = f["/opensees/patterns"]
            plain_groups = [k for k in patterns if k.startswith("Plain_")]
            assert len(plain_groups) == 1, (
                "rank-spanning pattern must merge into one group, got "
                f"{plain_groups!r}"
            )
            loads = patterns[plain_groups[0]]["loads"][:]
            load_nodes = sorted(int(r["target"]) for r in loads)
            assert load_nodes == [1, 4], (
                f"merged pattern must carry both rank's loads, got "
                f"{load_nodes!r}"
            )


def test_shared_node_load_captured_once() -> None:
    """A load on the SHARED node emits inside both ranks' pattern
    blocks but must land exactly once in the merged pattern group."""
    ops = _bridge(_shared_node_fem())
    series = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=series) as p:
        p.load(node=2, forces=(0.0, 0.0, -9.0, 0.0, 0.0, 0.0))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.h5")
        _emit_to_file(ops, path)
        with h5py.File(path, "r") as f:
            patterns = f["/opensees/patterns"]
            plain_groups = [k for k in patterns if k.startswith("Plain_")]
            assert len(plain_groups) == 1
            loads = patterns[plain_groups[0]]["loads"][:]
            load_nodes = [int(r["target"]) for r in loads]
            assert load_nodes.count(2) == 1, (
                "shared-node load must capture once, got "
                f"{load_nodes!r}"
            )


def test_flat_build_duplicate_fixes_keep_both_rows() -> None:
    """OUTSIDE partition brackets nothing changes: a flat build that
    genuinely declares the same fix twice keeps both rows (today's
    behavior — the dedupe is scoped to partition-bracketed captures
    only)."""
    fem = make_two_column_frame()  # unpartitioned
    ops = _bridge(fem)
    ops.fix(nodes=[1], dofs=(1, 1, 1, 1, 1, 1))
    ops.fix(nodes=[1], dofs=(1, 1, 1, 1, 1, 1))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.h5")
        _emit_to_file(ops, path)
        with h5py.File(path, "r") as f:
            rows = f["/opensees/bcs/fix"][:]
            targets = [r["target"].decode() for r in rows]
            assert targets.count("1") == 2, (
                "flat-build duplicate fix capture must stay unchanged "
                f"(2 rows), got {targets!r}"
            )
