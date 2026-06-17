"""MT-4 (ADR 0062) end-to-end — ops.fault.from_shakermaker → one pattern/source.

A finite fault becomes N ``Plain`` patterns (one per point source), each a
``Yoffe`` moment function at the source's rupture onset carrying a single
``moment_tensor``. Verifies the per-pattern emit recovers each source's
moment tensor (Σ_a x_a⊗F_a = M_k), distinct time series, the radians→degrees
angle handling, and the skip / peak-clamp / window-truncation guards.

(``from_ffsp`` is deferred to MT-5 — its FFSP unit contract needs validation
against a real FFSP run; see fault.py module docstring.)
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.geometry._moment_tensor import unit_moment_tensor
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.ns.fault import (
    WarnFaultPeakClamped,
    WarnFaultSubfaultSkipped,
    WarnFaultSubfaultTruncated,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic

LSCALE = 1000.0                                    # km → deck metres


@pytest.fixture(scope="module")
def box_fem():
    g = apeGmsh(model_name="fault_emit", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.structured.set_transfinite("soil", n=3)
        g.mesh.generation.generate(dim=3)
        yield g.mesh.queries.get_fem_data()
    finally:
        g.end()


class _PS:
    """Minimal duck-typed ShakerMaker PointSource (x km, angles RADIANS, tt)."""

    def __init__(self, x, angles_deg, tt):
        self.x = np.asarray(x)
        self.angles = np.radians(angles_deg)
        self.tt = tt


def _model(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat)
    return ops


def _patterns_loads(rec):
    blocks, cur = [], None
    for name, args, _ in rec.calls:
        if name == "pattern_open":
            cur = []
        elif name == "pattern_close":
            if cur is not None:
                blocks.append(cur)
            cur = None
        elif name == "load" and cur is not None:
            cur.append(args)
    return blocks


def _first_moment(fem, load_args):
    coord_of = {int(t): np.asarray(c, float)
                for t, c in zip(fem.nodes.ids, fem.nodes.coords)}
    nodes = [int(a[0]) for a in load_args]
    forces = np.array([[float(v) for v in a[1:4]] for a in load_args])
    xs = np.array([coord_of[n] for n in nodes])
    xs = xs - xs.mean(axis=0)
    return forces, xs.T @ forces


def test_from_shakermaker_one_pattern_per_source(box_fem):
    fault = [
        _PS([0.0005, 0.0005, 0.001], [350.0, 40.0, 113.0], 0.0),
        _PS([0.0015, 0.0015, 0.001], [30.0, 45.0, 90.0], 0.5),
    ]
    ops = _model(box_fem)
    patterns = ops.fault.from_shakermaker(
        fault, frame="z-down", M0=[1.0e12, 2.0e12], rise_time=1.0,
        peak_time=0.2, dt=0.02, t_total=4.0, length_scale=LSCALE,
        method="consistent",
    )
    assert len(patterns) == 2

    rec = RecordingEmitter()
    ops.build().emit(rec)
    blocks = _patterns_loads(rec)
    assert len(blocks) == 2
    assert len([c for c in rec.calls if c[0] == "timeSeries"]) == 2  # distinct

    # angles recovered as degrees → each source's moment tensor
    M0 = [1.0e12, 2.0e12]
    mech = [(350.0, 40.0, 113.0), (30.0, 45.0, 90.0)]
    for blk, m0, (s, d, r) in zip(blocks, M0, mech):
        M = m0 * unit_moment_tensor(strike=s, dip=d, rake=r)  # z-down, no flip
        forces, fm = _first_moment(box_fem, blk)
        assert np.allclose(forces.sum(axis=0), 0.0,
                           atol=1e-6 * np.abs(forces).max())
        assert np.allclose(fm, M, rtol=1e-6, atol=1.0)


def test_per_source_zero_moment_skipped(box_fem):
    fault = [
        _PS([0.001, 0.001, 0.001], [0.0, 90.0, 0.0], 0.0),
        _PS([0.001, 0.001, 0.001], [0.0, 90.0, 0.0], 0.0),
    ]
    ops = _model(box_fem)
    with pytest.warns(WarnFaultSubfaultSkipped, match="skipped 1"):
        patterns = ops.fault.from_shakermaker(
            fault, frame="z-down", M0=[1.0e12, 0.0], rise_time=1.0,
            peak_time=0.2, dt=0.02, t_total=4.0, length_scale=LSCALE,
        )
    assert len(patterns) == 1


def test_per_source_accepts_0d_array(box_fem):
    """np.ndim==0 detection — a 0-d numpy array broadcasts as a scalar."""
    fault = [
        _PS([0.001, 0.001, 0.001], [0.0, 90.0, 0.0], 0.0),
        _PS([0.0015, 0.0015, 0.001], [0.0, 90.0, 0.0], 0.0),
    ]
    ops = _model(box_fem)
    patterns = ops.fault.from_shakermaker(
        fault, frame="z-down", M0=np.array(1.0e12), rise_time=1.0,
        peak_time=0.2, dt=0.02, t_total=4.0, length_scale=LSCALE,
    )
    assert len(patterns) == 2


def test_peak_clamp_warns_and_still_emits(box_fem):
    fault = [_PS([0.001, 0.001, 0.001], [0.0, 90.0, 0.0], 0.0)]
    ops = _model(box_fem)
    with pytest.warns(WarnFaultPeakClamped, match="clamped peak_time"):
        patterns = ops.fault.from_shakermaker(
            fault, frame="z-down", M0=1.0e12, rise_time=1.0,
            peak_time=0.8,                          # > 0.49 → clamped
            dt=0.02, t_total=4.0, length_scale=LSCALE,
        )
    assert len(patterns) == 1                       # still constructible


def test_onset_past_window_warns(box_fem):
    fault = [_PS([0.001, 0.001, 0.001], [0.0, 90.0, 0.0], 10.0)]  # tt > t_total
    ops = _model(box_fem)
    with pytest.warns(WarnFaultSubfaultTruncated, match="never released"):
        ops.fault.from_shakermaker(
            fault, frame="z-down", M0=1.0e12, rise_time=1.0, peak_time=0.2,
            dt=0.02, t_total=4.0, length_scale=LSCALE,
        )
