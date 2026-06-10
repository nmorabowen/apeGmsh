"""Partitioned absorbing-skin end-to-end (ADR 0054 deferred item).

Drives the REAL pipeline — ``add_plane_wave_box`` -> mesh ->
``g.mesh.partitioning.partition(2)`` -> ``get_fem_data`` -> bridge with
``absorbing_boundary`` (base series on bottom) + staged
``s.activate_absorbing`` — and pins the per-rank emit shape:

1. Every skin element is emitted exactly once, inside exactly one
   ``partition_open`` block (no duplicates, no global strays; union over
   ranks = the closed-form skin count).
2. The stage flip is per-rank: each rank's ``flip_element_stage`` carries
   exactly the absorbing tags that rank emitted (the STKO per-partition
   idiom), and no flip appears in the global scope.
3. The ``-fx`` base input rides every bottom (``B``-containing) cell on
   whichever rank owns it (count = the closed-form bottom tally).

AB-3 unit-tested the per-rank filtering against stubs; this test proves
the composition over a real partitioned mesh (METIS cuts the skin
arbitrarily across ranks).
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.time_series.time_series import Path

# Same closed-form grid as tests/parts/test_plane_wave_box.py.
BOX = dict(x=(40.0, 4), y=(50.0, 5), z=(30.0, 3))
NX, NY, NZ = 4, 5, 3
SKIN_HEX = (
    2 * NY * NZ + 2 * NX * NZ + NX * NY      # face panels L/R/F/K/B
    + 4 * NZ + 2 * NY + 2 * NX               # vertical + bottom edges
    + 4                                      # bottom corners
)                                            # = 108
BOTTOM_HEX = NX * NY + 2 * NY + 2 * NX + 4   # every B-containing cell = 42


def _bucket_calls_by_rank(
    rec: RecordingEmitter,
) -> dict["int | None", list[tuple[str, tuple, dict]]]:
    """Bucket recorded calls by partition rank (``None`` = global scope).

    ``partition_open`` / ``partition_close`` / ``stage_open`` /
    ``stage_close`` are structural markers and not stored.
    """
    buckets: dict["int | None", list[tuple[str, tuple, dict]]] = {}
    rank: "int | None" = None
    for name, args, kwargs in rec.calls:
        if name == "partition_open":
            rank = int(args[0])
            continue
        if name == "partition_close":
            rank = None
            continue
        if name in ("stage_open", "stage_close"):
            continue
        buckets.setdefault(rank, []).append((name, args, kwargs))
    return buckets


@pytest.fixture(scope="module")
def partitioned_emit() -> dict["int | None", list[tuple[str, tuple, dict]]]:
    """Build the partitioned plane-wave model once; return rank buckets."""
    g = apeGmsh(model_name="pwb_part_e2e", verbose=False)
    g.begin()
    try:
        res = g.parts.add_plane_wave_box(**BOX)
        g.mesh.generation.generate(dim=3)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data()
    finally:
        g.end()

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    soil = ops.register(ElasticIsotropic(E=3410.0, nu=0.262, rho=2.4e-9))
    ts = ops.register(Path(values=(0.0, 1.0, 0.0), dt=0.1))
    ops.element.stdBrick(pg=res.soil_pg, material=soil)
    ops.element.absorbing_boundary(
        skin=res, material=soil, base_series=ts, base_dirs=("x",),
    )
    with ops.stage(name="dyn") as s:
        s.activate_absorbing(pg=res.skin_all_pg)
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-6, max_iter=20),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.1),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1, dt=0.01)

    rec = RecordingEmitter()
    ops.build().emit(rec)
    return _bucket_calls_by_rank(rec)


def _absorbing_tags_by_rank(
    buckets: dict["int | None", list[tuple[str, tuple, dict]]],
) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for rank, calls in buckets.items():
        if rank is None:
            continue
        out[int(rank)] = {
            int(a[1]) for n, a, _k in calls
            if n == "element" and a[0] == "ASDAbsorbingBoundary3D"
        }
    return out


def test_skin_elements_partition_disjointly(partitioned_emit) -> None:
    """Each skin element emits exactly once, on exactly one rank."""
    by_rank = _absorbing_tags_by_rank(partitioned_emit)
    assert set(by_rank) == {0, 1}
    # Both ranks own a non-trivial share of the skin (METIS cuts it).
    assert all(tags for tags in by_rank.values())
    assert by_rank[0].isdisjoint(by_rank[1])
    assert len(by_rank[0] | by_rank[1]) == SKIN_HEX


def test_no_absorbing_elements_in_global_scope(partitioned_emit) -> None:
    """Absorbing elements live inside partition blocks, never global."""
    stray = [
        a for n, a, _k in partitioned_emit.get(None, [])
        if n == "element" and a[0] == "ASDAbsorbingBoundary3D"
    ]
    assert stray == []


def test_flip_is_per_rank_and_matches_owned_tags(partitioned_emit) -> None:
    """Each rank flips exactly the absorbing tags it emitted (the STKO
    per-partition ``parameter``/``addToParameter ... stage`` idiom)."""
    by_rank = _absorbing_tags_by_rank(partitioned_emit)
    for rank in (0, 1):
        flips = [
            a for n, a, _k in partitioned_emit[rank]
            if n == "flip_element_stage"
        ]
        assert len(flips) == 1, (
            f"rank {rank} should emit exactly one flip block; got {flips}"
        )
        _pid, tags = flips[0]
        assert set(int(t) for t in tags) == by_rank[rank], (
            f"rank {rank} flip set != its owned absorbing tags"
        )
    # Never in the global scope.
    assert not any(
        n == "flip_element_stage" for n, _a, _k in partitioned_emit.get(None, [])
    )


def test_base_series_rides_bottom_cells_on_both_ranks(partitioned_emit) -> None:
    """-fx attaches to every B-containing skin cell, wherever it lands."""
    n_fx = 0
    for rank, calls in partitioned_emit.items():
        if rank is None:
            continue
        for n, a, _k in calls:
            if n == "element" and a[0] == "ASDAbsorbingBoundary3D" and "-fx" in a:
                n_fx += 1
    assert n_fx == BOTTOM_HEX
