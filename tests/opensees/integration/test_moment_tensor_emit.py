"""MT-2 (ADR 0062) end-to-end — p.moment_tensor → nodal load lines.

Meshes a structured hex box, embeds a moment-tensor source, emits through
the bridge, and checks the emitted ``load`` lines satisfy the discrete
representation theorem: zero net force and a first moment that recovers
``M`` (``Σ_a x_a ⊗ F_a = M``).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.geometry._moment_tensor import unit_moment_tensor
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic


@pytest.fixture(scope="module")
def box_fem():
    """A 2×2×2 structured hex box (3³ nodes, 8 hexes); soil PG."""
    g = apeGmsh(model_name="mt_emit", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.structured.set_transfinite("soil", n=3)
        g.mesh.generation.generate(dim=3)
        yield g.mesh.queries.get_fem_data()
    finally:
        g.end()


@pytest.fixture(scope="module")
def two_region_fem():
    """Two stacked hex boxes: soil PG (z 0..2) + cap PG (z 2..4)."""
    g = apeGmsh(model_name="mt_region", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, label="soil")
        g.model.geometry.add_box(0.0, 0.0, 2.0, 2.0, 2.0, 2.0, label="cap")
        g.physical.add(3, "soil", name="soil")
        g.physical.add(3, "cap", name="cap")
        g.mesh.structured.set_transfinite("soil", n=3)
        g.mesh.structured.set_transfinite("cap", n=3)
        g.mesh.generation.generate(dim=3)
        yield g.mesh.queries.get_fem_data()
    finally:
        g.end()


def _emit_with_region(fem, *, position, region):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat)
    ops.element.stdBrick(pg="cap", material=mat)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.moment_tensor(
            position=position, frame="z-down", M0=1.0e6,
            mech=dict(strike=0, dip=90, rake=0), region=region,
        )
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return rec


def test_region_confines_source_to_pg(two_region_fem):
    """A source inside the cap but region='soil' fails loud (the
    absorbing-skin guard mechanism)."""
    with pytest.raises(BridgeError, match="outside the region 'soil'"):
        _emit_with_region(two_region_fem, position=(1.0, 1.0, 3.0),
                          region="soil")


def test_region_allows_source_inside_pg(two_region_fem):
    """The same source inside the soil region emits normally."""
    rec = _emit_with_region(two_region_fem, position=(1.0, 1.0, 1.0),
                           region="soil")
    loads = [c for c in rec.calls if c[0] == "load"]
    assert len(loads) == 8                       # one soil hex, 8 corner nodes


def _load_calls(rec: RecordingEmitter):
    return [c for c in rec.calls if c[0] == "load"]


def _emit_source(fem, *, method, position):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.moment_tensor(
            position=position,
            frame="z-down",                 # no flip → M = M0·m
            M0=1.0e6,
            mech=dict(strike=350, dip=40, rake=113),
            method=method,
        )
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return rec


def _first_moment_from_loads(fem, load_calls):
    coord_of = {int(t): np.asarray(c, float)
                for t, c in zip(fem.nodes.ids, fem.nodes.coords)}
    nodes = [int(c[1][0]) for c in load_calls]
    forces = np.array([[float(v) for v in c[1][1:4]] for c in load_calls])
    xs = np.array([coord_of[n] for n in nodes])
    xs = xs - xs.mean(axis=0)
    return forces, xs.T @ forces


def test_consistent_emit_recovers_moment_tensor(box_fem):
    M = 1.0e6 * unit_moment_tensor(strike=350, dip=40, rake=113)
    rec = _emit_source(box_fem, method="consistent", position=(0.5, 0.5, 0.5))
    loads = _load_calls(rec)
    assert len(loads) == 8                       # one hex, 8 corner nodes
    forces, fm = _first_moment_from_loads(box_fem, loads)
    scale = np.abs(forces).max()
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-6 * scale)
    assert np.allclose(fm, M, rtol=1e-6, atol=1.0)


def test_dipole_emit_recovers_moment_tensor(box_fem):
    M = 1.0e6 * unit_moment_tensor(strike=350, dip=40, rake=113)
    # the interior node of a 3³ grid — all 6 axis neighbours present
    rec = _emit_source(box_fem, method="dipole", position=(1.0, 1.0, 1.0))
    loads = _load_calls(rec)
    # dipole couples on the 6 axis neighbours of the nearest grid node
    assert len(loads) == 6
    forces, fm = _first_moment_from_loads(box_fem, loads)
    scale = np.abs(forces).max()
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-6 * scale)
    assert np.allclose(fm, M, rtol=1e-6, atol=1.0)


def test_partitioned_emit_union_recovers_moment_tensor():
    """Under OpenSeesMP the per-rank load lines must reproduce the flat deck
    (Σ_a x_a⊗F_a = M over the union) — the SSI/wave-prop workload runs
    partitioned, so a silently-dropped source would be a blocker."""
    g = apeGmsh(model_name="mt_part", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.structured.set_transfinite("soil", n=3)
        g.mesh.generation.generate(dim=3)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data()
    finally:
        g.end()

    assert len(fem.partitions) == 2
    M = 1.0e6 * unit_moment_tensor(strike=350, dip=40, rake=113)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.moment_tensor(
            position=(0.5, 0.5, 0.5), frame="z-down", M0=1.0e6,
            mech=dict(strike=350, dip=40, rake=113), method="consistent",
        )
    rec = RecordingEmitter()
    ops.build().emit(rec)

    # collect load lines and the rank bracket each sits in
    loads, rank, any_in_partition = [], None, False
    for name, args, _ in rec.calls:
        if name == "partition_open":
            rank = int(args[0])
        elif name == "partition_close":
            rank = None
        elif name == "load":
            loads.append(args)
            if rank is not None:
                any_in_partition = True

    assert loads, "moment-tensor source dropped under partitioned emit"
    assert any_in_partition, "load lines must sit inside partition blocks"
    forces, fm = _first_moment_from_loads(fem, [("load", a) for a in loads])
    scale = np.abs(forces).max()
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-6 * scale)
    assert np.allclose(fm, M, rtol=1e-6, atol=1.0)


def test_non_zero_t0_fails_loud(box_fem):
    ops = apeSees(box_fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.moment_tensor(
            position=(0.5, 0.5, 0.5), frame="z-down", M0=1.0,
            mech=dict(strike=0, dip=90, rake=0), t0=1.5,
        )
    with pytest.raises(NotImplementedError, match="onset"):
        ops.build().emit(RecordingEmitter())
