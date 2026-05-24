"""Pin the 0-based runtime-rank fix at the build-path seam.

Schema 2.11.0 flipped the rank seam from Gmsh's 1-based
``PartitionRecord.id`` to OpenSeesMP's 0-based ``getPID()`` convention.
This test pins the contract end-to-end against a real Gmsh-partitioned
session:

* The Tcl deck contains gates ``if {[getPID] == K}`` with
  ``K in {0, 1, 2, 3}`` (not ``{1, 2, 3, 4}`` — the old bug).
* The Py deck contains gates ``if getPID() == K:`` with the same set.
* The H5 archive's ``/opensees/partitions/`` group has sub-groups
  ``partition_00`` through ``partition_03`` (the loop-index naming
  has been 0-based since 2.10.0; the per-group ``rank`` attr was the
  buggy 1-based value before 2.11.0).
* The per-group ``rank`` attribute is the 0-based rank value
  (matching ``getPID()``).
* The parallel ``partition_ids`` column on every
  ``/opensees/element_meta/{type}/`` group carries values in
  ``{0..3}`` (matching the rank attrs).
* The broker's ``fem.partitions[i].id`` is **unchanged** — Gmsh
  still emits 1-based labels on the record itself; only the
  runtime-rank seam flipped.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

# Gmsh import is the long pole — skip the whole module cleanly if it
# is not on the box (mirrors the e2e test's posture).
gmsh = pytest.importorskip("gmsh")


def _build_partitioned_frame_session(n_parts: int):
    """Return a session with a partitioned 3-storey frame.

    Mirrors the geometry used by ``test_partition_pipeline_e2e.py``
    (corner-quad column / beam layout), tuned just large enough so
    METIS gets ``n_parts`` non-empty partitions.
    """
    from apeGmsh import apeGmsh

    sess = apeGmsh(model_name="zero_based_rank_test", verbose=False)
    sess.begin()
    try:
        plan = 5.0
        story_height = 3.0
        lc = 1.5
        n_stories = 3

        points: dict[tuple, int] = {}
        for c, (x, y) in enumerate(
            ((0, 0), (plan, 0), (plan, plan), (0, plan)), start=1,
        ):
            points[("base", c)] = gmsh.model.geo.addPoint(x, y, 0, lc)
        for s in range(1, n_stories + 1):
            for c, (x, y) in enumerate(
                ((0, 0), (plan, 0), (plan, plan), (0, plan)), start=1,
            ):
                points[(s, c)] = gmsh.model.geo.addPoint(
                    x, y, s * story_height, lc,
                )

        column_lines: list[int] = []
        for s in range(1, n_stories + 1):
            prev = "base" if s == 1 else s - 1
            for c in (1, 2, 3, 4):
                column_lines.append(
                    gmsh.model.geo.addLine(points[(prev, c)], points[(s, c)]),
                )
        beam_lines: list[int] = []
        for s in range(1, n_stories + 1):
            beam_lines.extend([
                gmsh.model.geo.addLine(points[(s, 1)], points[(s, 2)]),
                gmsh.model.geo.addLine(points[(s, 2)], points[(s, 3)]),
                gmsh.model.geo.addLine(points[(s, 3)], points[(s, 4)]),
                gmsh.model.geo.addLine(points[(s, 4)], points[(s, 1)]),
            ])

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, column_lines, name="Columns")
        gmsh.model.addPhysicalGroup(1, beam_lines, name="Beams")
        gmsh.model.addPhysicalGroup(
            0, [points[("base", c)] for c in (1, 2, 3, 4)], name="Base",
        )

        sess.mesh.sizing.set_global_size(lc)
        sess.mesh.generation.generate(1)
        sess.mesh.partitioning.renumber(dim=1, method="simple", base=1)
        sess.mesh.partitioning.partition(n_parts)
        return sess
    except Exception:
        sess.end()
        raise


def _wire_apesees(fem):
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    transf_col = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    transf_beam = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Columns", transf=transf_col,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.element.elasticBeamColumn(
        pg="Beams", transf=transf_beam,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    return ops


def test_tcl_gates_are_zero_based(tmp_path: Path) -> None:
    """A 4-rank emit produces gates ``[0, 1, 2, 3]``, NOT ``[1, 2, 3, 4]``.

    This is the load-bearing assertion for the 0-based-runtime-rank
    fix: under ``mpiexec -np 4 OpenSeesMP``, ``getPID()`` returns
    ``0..3``; the emitted gates must hit every one of those values.
    """
    n_parts = 4
    sess = _build_partitioned_frame_session(n_parts)
    try:
        fem = sess.mesh.queries.get_fem_data(dim=1)
        assert len(fem.partitions) == n_parts

        ops = _wire_apesees(fem)
        tcl_path = tmp_path / "model.tcl"
        ops.tcl(str(tcl_path))
        text = tcl_path.read_text(encoding="utf-8")

        # Pull out every ``if {[getPID] == K} {`` literal.
        gate_re = re.compile(r"if \{\[getPID\] == (\d+)\} \{")
        gates = sorted({int(m.group(1)) for m in gate_re.finditer(text)})

        assert gates == [0, 1, 2, 3], (
            f"Tcl gates must be 0-based [0, 1, 2, 3] (matching "
            f"OpenSeesMP::getPID()); got {gates}.  If the bug has "
            f"regressed, gates would be [1, 2, 3, 4] (Gmsh's 1-based "
            f"PartitionRecord.id leaking through partition_open)."
        )
    finally:
        sess.end()


def test_py_gates_are_zero_based(tmp_path: Path) -> None:
    """The Py emitter mirrors Tcl — ``if getPID() == K:`` for K in 0..3."""
    n_parts = 4
    sess = _build_partitioned_frame_session(n_parts)
    try:
        fem = sess.mesh.queries.get_fem_data(dim=1)
        ops = _wire_apesees(fem)
        py_path = tmp_path / "model.py"
        ops.py(str(py_path))
        text = py_path.read_text(encoding="utf-8")

        gate_re = re.compile(r"if getPID\(\) == (\d+):")
        gates = sorted({int(m.group(1)) for m in gate_re.finditer(text)})
        assert gates == [0, 1, 2, 3], (
            f"Py gates must be 0-based [0, 1, 2, 3]; got {gates}"
        )
    finally:
        sess.end()


def test_h5_partition_groups_and_rank_attrs_are_zero_based(
    tmp_path: Path,
) -> None:
    """H5 group names + rank attrs + partition_ids column are 0-based.

    Group naming (``partition_NN``) has been 0-based since schema
    2.10.0 (loop-index); the new invariant in 2.11.0 is the
    per-group ``rank`` attr and the parallel ``partition_ids`` row
    values — they were the buggy 1-based ``PartitionRecord.id`` in
    2.10.0, and become the 0-based runtime rank in 2.11.0.
    """
    n_parts = 4
    sess = _build_partitioned_frame_session(n_parts)
    try:
        fem = sess.mesh.queries.get_fem_data(dim=1)
        ops = _wire_apesees(fem)
        h5_path = tmp_path / "model.h5"
        ops.h5(str(h5_path))

        with h5py.File(str(h5_path), "r") as f:
            parts_grp = f["/opensees/partitions"]
            assert int(parts_grp.attrs["n_partitions"]) == n_parts

            # Group naming + per-group rank attr.  Both must agree on
            # the 0-based convention.
            for k in range(n_parts):
                gname = f"partition_{k:02d}"
                assert gname in parts_grp, (
                    f"missing /opensees/partitions/{gname}"
                )
                rank_attr = int(parts_grp[gname].attrs["rank"])
                assert rank_attr == k, (
                    f"{gname}/@rank must be {k} (0-based runtime "
                    f"rank); got {rank_attr}.  If this is k+1 the bug "
                    f"has regressed: ``part.id`` (1-based) is being "
                    f"forwarded instead of the enumerate index."
                )

            # The parallel partition_ids column carries the same set
            # of rank values (0..N-1, one per emitted element).
            elem_meta = f["/opensees/element_meta"]
            seen_ranks: set[int] = set()
            for ele_type_name in elem_meta:
                g_t = elem_meta[ele_type_name]
                if "partition_ids" not in g_t:
                    continue
                pids = np.asarray(g_t["partition_ids"])
                seen_ranks.update(int(p) for p in pids if int(p) >= 0)
            assert seen_ranks == set(range(n_parts)), (
                f"partition_ids column must cover ranks "
                f"{set(range(n_parts))}; got {seen_ranks}"
            )
    finally:
        sess.end()


def test_broker_partition_id_is_unchanged(tmp_path: Path) -> None:
    """The broker's ``PartitionRecord.id`` is **not** touched by the fix.

    Gmsh assigns 1-based partition labels and the broker preserves
    them verbatim for Gmsh-side traceability.  The fix is scoped to
    the runtime-rank seam in the bridge — the broker contract is
    untouched.
    """
    n_parts = 4
    sess = _build_partitioned_frame_session(n_parts)
    try:
        fem = sess.mesh.queries.get_fem_data(dim=1)
        broker_ids = sorted(int(p.id) for p in fem.partitions)
        # Gmsh emits 1..N (1-based).  If the broker ever flips to
        # 0-based the contract breaks downstream — this test pins
        # the broker convention so the fix stays surgical.
        assert broker_ids == [1, 2, 3, 4], (
            f"fem.partitions[i].id must remain Gmsh's 1-based label; "
            f"got {broker_ids}"
        )
    finally:
        sess.end()
