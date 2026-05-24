"""Integration tests for ADR 0027 INV-4 — MPCO recorder per-rank region.

Per ADR 0027 §"Regions interaction" / INV-4, applied to the MPCO
recorder's internal region resolution (the path triggered by
``ops.recorder.MPCO(nodes_pg=..., elements_pg=...)``):

* ``recorder mpco ... -R <tag>`` emits ONCE globally (the recorder
  declaration is global; recorders write to disk, not to the model
  topology — one declaration is sufficient even under OpenSeesMP).
* ``region <tag> -node ... -ele ...`` emits per-rank, with the
  rank-owned intersection of the resolved filter ids.
* Empty intersection on a rank ⇒ no ``region`` line for that
  recorder on that rank.
* The ``<tag>`` is the SAME scalar across every emitting rank and on
  the global recorder line — MPCO post-processing stitches by tag
  identity.

When ``len(fem.partitions) <= 1`` the emit is byte-identical to the
pre-INV-4-fix path; the un-partitioned flat path is untouched on
this branch (asserted by
``test_mpco_region_unpartitioned_byte_identical_to_pre_change``).
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_column_frame_partitioned,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _per_rank_calls(
    rec: RecordingEmitter,
) -> dict[int, list[tuple]]:
    """Bucket every call between ``partition_open(K)`` / ``partition_close``
    into a per-rank list.  Calls outside any partition bracket are
    discarded — callers needing them should inspect ``rec.calls`` directly.
    """
    out: dict[int, list[tuple]] = {}
    cur: int | None = None
    for name, args, kwargs in rec.calls:
        if name == "partition_open":
            cur = int(args[0])
            out.setdefault(cur, [])
        elif name == "partition_close":
            cur = None
        elif cur is not None:
            out[cur].append((name, args, kwargs))
    return out


def _global_calls(
    rec: RecordingEmitter,
) -> list[tuple]:
    """Return only the calls outside any ``partition_open`` / ``partition_close``
    bracket.  The MPCO recorder declaration lives in global scope; the
    per-rank ``region`` lines live inside the brackets.
    """
    out: list[tuple] = []
    in_partition = False
    for name, args, kwargs in rec.calls:
        if name == "partition_open":
            in_partition = True
            continue
        if name == "partition_close":
            in_partition = False
            continue
        if not in_partition:
            out.append((name, args, kwargs))
    return out


def _mpco_recorder_calls(calls: list[tuple]) -> list[tuple]:
    """Filter for ``recorder("mpco", ...)`` calls only — the MPCO recorder
    declaration is emitted via ``emitter.recorder("mpco", file, ..., -R, tag)``.
    """
    return [
        (n, a, k) for (n, a, k) in calls
        if n == "recorder" and a and a[0] == "mpco"
    ]


def _region_calls(calls: list[tuple]) -> list[tuple]:
    return [(n, a, k) for (n, a, k) in calls if n == "region"]


def _scan_dash_R(mpco_args: tuple) -> int | None:
    """Pull the ``-R <tag>`` value out of an MPCO arg tail, or None if absent."""
    args = list(mpco_args)
    if "-R" not in args:
        return None
    return int(args[args.index("-R") + 1])


# ---------------------------------------------------------------------------
# Test 1 — region emitted per-rank under partitioning, recorder global
# ---------------------------------------------------------------------------

def test_mpco_region_emitted_per_rank_under_partitioning() -> None:
    """Two-rank fixture with a region spanning both ranks' nodes.

    The MPCO recorder declaration must emit EXACTLY ONCE in global
    scope (outside any ``partition_open(K)`` block).  The
    ``region <tag> -node ... -ele ...`` command must emit inside EACH
    rank's ``partition_open`` block — one per rank, carrying only that
    rank's owned subset.  The ``<tag>`` value is identical across every
    region emission AND the recorder's ``-R <tag>`` reference (INV-4 —
    MPCO post-merge stitches by tag identity).
    """
    fem = make_two_column_frame_partitioned()
    # Partition 0 owns nodes {1, 2} + element 1; partition 1 owns
    # {3, 4} + element 2 (see make_two_column_frame_partitioned).
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    # MPCO recorder filtering both nodes_pg + elements_pg — exercises
    # the dual-side intersection path.  "Base" -> nodes {1, 3} (spans
    # both ranks); "Cols" -> elements {1, 2} (one per rank).
    ops.recorder.MPCO(
        file="out/run.mpco",
        nodal_responses=("displacement",),
        elem_responses=("section.force",),
        nodes_pg="Base",
        elements_pg="Cols",
    )

    rec = RecordingEmitter()
    ops.build().emit(rec)

    # -- Global scope: exactly one ``recorder mpco`` line. ---------------
    global_calls = _global_calls(rec)
    mpco_recorder_globals = _mpco_recorder_calls(global_calls)
    assert len(mpco_recorder_globals) == 1, (
        f"INV-4: ``recorder mpco`` must emit EXACTLY ONCE in global "
        f"scope. Got {len(mpco_recorder_globals)}: {mpco_recorder_globals!r}"
    )
    # And NO region line in global scope (regions live inside ranks).
    assert _region_calls(global_calls) == [], (
        "INV-4: per-rank regions must NOT leak into global scope.\n"
        f"Got globals: {_region_calls(global_calls)!r}"
    )

    # Pull the recorder's -R tag.
    mpco_args = mpco_recorder_globals[0][1]
    recorder_R_tag = _scan_dash_R(mpco_args)
    assert recorder_R_tag is not None, (
        "INV-4: the global ``recorder mpco`` line must carry ``-R <tag>``."
    )

    # -- Per-rank scope: each rank emits exactly one region line. --------
    per_rank = _per_rank_calls(rec)
    assert set(per_rank) == {0, 1}, (
        f"Expected ranks {{0, 1}}, got {set(per_rank)}"
    )

    region_calls_r0 = _region_calls(per_rank[0])
    region_calls_r1 = _region_calls(per_rank[1])
    assert len(region_calls_r0) == 1, (
        f"rank 0 must emit one MPCO region line; got {region_calls_r0!r}"
    )
    assert len(region_calls_r1) == 1, (
        f"rank 1 must emit one MPCO region line; got {region_calls_r1!r}"
    )

    # The region tag MUST match the recorder's -R tag and be identical
    # across ranks (INV-4 — same scalar, MPCO stitches by tag).
    tag0 = int(region_calls_r0[0][1][0])
    tag1 = int(region_calls_r1[0][1][0])
    assert tag0 == tag1 == recorder_R_tag, (
        f"INV-4: region tag must be the SAME scalar across ranks AND "
        f"the recorder's ``-R`` reference; got rank0={tag0}, "
        f"rank1={tag1}, recorder -R={recorder_R_tag}"
    )

    # Per-rank node + element contents.  "Base" -> nodes (1, 3): rank
    # 0 owns 1; rank 1 owns 3.  "Cols" -> elements (1, 2): rank 0 owns
    # element 1; rank 1 owns element 2.
    def _scan_flag(args: tuple, flag: str) -> list:
        if flag not in args:
            return []
        a = list(args)
        i = a.index(flag) + 1
        out = []
        while i < len(a) and not (isinstance(a[i], str) and a[i].startswith("-")):
            out.append(a[i])
            i += 1
        return out

    r0_nodes = sorted(int(x) for x in _scan_flag(region_calls_r0[0][1], "-node"))
    r1_nodes = sorted(int(x) for x in _scan_flag(region_calls_r1[0][1], "-node"))
    r0_eles = sorted(int(x) for x in _scan_flag(region_calls_r0[0][1], "-ele"))
    r1_eles = sorted(int(x) for x in _scan_flag(region_calls_r1[0][1], "-ele"))
    assert r0_nodes == [1], f"rank 0 expected -node [1], got {r0_nodes}"
    assert r1_nodes == [3], f"rank 1 expected -node [3], got {r1_nodes}"
    assert r0_eles == [1], f"rank 0 expected -ele [1], got {r0_eles}"
    assert r1_eles == [2], f"rank 1 expected -ele [2], got {r1_eles}"


# ---------------------------------------------------------------------------
# Test 2 — empty intersection on a rank → no region on that rank
# ---------------------------------------------------------------------------

def test_mpco_region_empty_intersection_skipped() -> None:
    """A region whose ids ALL belong to one rank emits NO region line on
    the other rank.  The recorder declaration's ``-R <tag>`` reference
    still resolves on the emitting rank; MPCO handles a missing region
    on a rank as a no-op (no output contributed from that rank).
    """
    fem = make_two_column_frame_partitioned()
    # rank 0 owns nodes {1, 2} + element 1; rank 1 owns {3, 4} + ele 2.
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    # Filter targets only rank-0 territory (nodes {1, 2} via explicit
    # tuple).  Add an explicit elements= so the asymmetric-filter
    # __post_init__ guard accepts the combo; both selectors live
    # entirely on rank 0.
    ops.recorder.MPCO(
        file="out/run.mpco",
        nodal_responses=("displacement",),
        elem_responses=("section.force",),
        nodes=(1, 2),
        elements=(1,),
    )

    rec = RecordingEmitter()
    ops.build().emit(rec)

    per_rank = _per_rank_calls(rec)
    region_r0 = _region_calls(per_rank[0])
    region_r1 = _region_calls(per_rank[1])
    assert len(region_r0) == 1, (
        f"rank 0 owns all filter members; must emit one region. "
        f"Got: {region_r0!r}"
    )
    assert len(region_r1) == 0, (
        f"INV-4: rank 1 has empty intersection with filter; must emit "
        f"NO region line. Got: {region_r1!r}"
    )

    # Recorder declaration still emits globally with -R tag.
    mpco_globals = _mpco_recorder_calls(_global_calls(rec))
    assert len(mpco_globals) == 1
    assert _scan_dash_R(mpco_globals[0][1]) == int(region_r0[0][1][0])


# ---------------------------------------------------------------------------
# Test 3 — un-partitioned model byte-identical to pre-INV-4 behaviour
# ---------------------------------------------------------------------------

def test_mpco_region_unpartitioned_byte_identical_to_pre_change() -> None:
    """The un-partitioned flat emit path is unaffected by the INV-4
    work — the region + recorder pair appears in GLOBAL scope (no
    ``partition_open`` / ``partition_close`` bracketing), exactly the
    same order as before the per-rank MPCO refactor.

    This pins the byte-identity invariant: tests in
    ``test_full_emit_recording.py::test_mpco_nodes_pg_emits_region_and_R_flag``
    and friends would catch a regression on a different vector; this
    test makes the INV-4 contract explicit (no partition events fire
    on an unpartitioned MPCO emit).
    """
    fem = make_two_column_frame()  # PG "Base" -> nodes (1, 3)
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    ops.recorder.MPCO(
        file="out/run.mpco",
        nodal_responses=("displacement",),
        nodes_pg="Base",
    )

    rec = RecordingEmitter()
    ops.build().emit(rec)

    method_names = [n for (n, _a, _k) in rec.calls]
    assert "partition_open" not in method_names, (
        "unpartitioned MPCO emit MUST NOT trigger partition_open"
    )
    assert "partition_close" not in method_names, (
        "unpartitioned MPCO emit MUST NOT trigger partition_close"
    )

    region_calls = _region_calls(rec.calls)
    mpco_recorder_calls = _mpco_recorder_calls(rec.calls)

    # Exactly one region + one MPCO recorder, region precedes recorder,
    # same tag as before the INV-4 refactor.  Matches the pre-change
    # behaviour asserted by
    # tests/opensees/integration/test_full_emit_recording.py::
    # test_mpco_nodes_pg_emits_region_and_R_flag.
    assert len(region_calls) == 1
    assert len(mpco_recorder_calls) == 1
    region_idx = rec.calls.index(region_calls[0])
    mpco_idx = rec.calls.index(mpco_recorder_calls[0])
    assert region_idx < mpco_idx

    region_tag = int(region_calls[0][1][0])
    recorder_R = _scan_dash_R(mpco_recorder_calls[0][1])
    assert recorder_R == region_tag


# ---------------------------------------------------------------------------
# Test 4 — H5 persistence under partitioning
# ---------------------------------------------------------------------------

def test_mpco_region_h5_persistence_under_partitioning() -> None:
    """When emitted via ``ops.h5(...)``, the partitioned MPCO recorder
    persists as ONE recorder group (not duplicated per-rank); the
    auto-emitted region(s) persist under ``/opensees/regions/``.

    The H5 emitter accumulates region calls flatly regardless of
    partition_open scope, so under N partitions we expect N region
    records (one per emitting rank), all sharing the same ``tag``
    attribute.  The recorder group's ``params`` carries the same tag
    via its ``-R`` arg tail.
    """
    h5py = pytest.importorskip("h5py")
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.recorder.MPCO(
        file="out/run.mpco",
        nodal_responses=("displacement",),
        elem_responses=("section.force",),
        nodes_pg="Base",
        elements_pg="Cols",
    )

    emitter = H5Emitter()
    ops.build().emit(emitter)

    import tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        emitter.write(tmp)
        with h5py.File(tmp, "r") as f:
            recorders = f["/opensees/recorders"]
            recorder_names = list(recorders.keys())
            # ONE recorder declaration (not duplicated per-rank).
            assert len(recorder_names) == 1, (
                f"expected ONE recorder group, got {recorder_names}"
            )
            rec_group = recorders[recorder_names[0]]
            assert rec_group.attrs["type"] == "mpco"
            # The recorder's params are stored as the (params, params_str)
            # NaN/empty-sentinel pair from ``_write_param_array``: float
            # slots carry the numeric value, string slots are reconstructed
            # from the parallel UTF-8 vlen array.
            import math
            nums = list(rec_group.attrs["params"])
            strs = list(rec_group.attrs["params_str"])
            params_recovered: list[object] = []
            for n, s in zip(nums, strs):
                if isinstance(s, bytes):
                    s = s.decode()
                if s:
                    params_recovered.append(s)
                else:
                    params_recovered.append(int(n) if float(n).is_integer() else float(n))
            assert "-R" in params_recovered, (
                f"recorder params must carry -R; got {params_recovered!r}"
            )
            r_idx = params_recovered.index("-R")
            recorder_tag = int(params_recovered[r_idx + 1])

            # Region groups: ONE per emitting rank (here 2), all
            # sharing the same tag attribute.
            assert "regions" in f["/opensees"], (
                "H5 emitter must persist /opensees/regions/ when MPCO "
                "auto-emits region(s)"
            )
            regions = f["/opensees/regions"]
            region_names = sorted(regions.keys())
            assert len(region_names) == 2, (
                f"expected ONE region per emitting rank (2 here); got "
                f"{region_names}"
            )
            for rn in region_names:
                grp = regions[rn]
                assert int(grp.attrs["tag"]) == recorder_tag, (
                    f"region {rn!r} tag {int(grp.attrs['tag'])} must "
                    f"match recorder -R tag {recorder_tag}"
                )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
