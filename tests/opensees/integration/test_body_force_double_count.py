"""ADR 0054 close-out — the self-weight double-count guard.

A continuum element's constructor ``body_force`` is applied every step with
no load pattern (verified against Brick.cpp / FourNodeQuad.cpp + a live
probe).  If a ``p.from_model(case)`` also drives a gravity load onto the same
nodes **along the same axis**, the region carries its weight twice.
``validate_body_force_double_count`` warns (fail-soft) on that overlap, and is
silent for the legitimate *lateral-load + self-weight* combo (orthogonal, so
not collinear).

The loads are injected directly onto ``fem.nodes.loads`` (the resolver output)
so the test controls the load direction exactly — the collinearity
discriminator is the whole point.
"""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.record_sets import NodalLoadSet
from apeGmsh._kernel.records._loads import NodalLoadRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import WarnBodyForceDoubleCount
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic


@pytest.fixture(scope="module")
def box_fem():
    """A small structured hex box with one soil PG."""
    g = apeGmsh(model_name="bf_double", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.structured.set_transfinite("soil", n=3)
        g.mesh.generation.generate(dim=3)
        yield g.mesh.queries.get_fem_data()
    finally:
        g.end()


def _inject_loads(fem, force_xyz, *, pattern="dead"):
    recs = [
        NodalLoadRecord(node_id=int(n), force_xyz=force_xyz, pattern=pattern)
        for n in fem.nodes.ids
    ]
    fem.nodes.loads = NodalLoadSet(recs)


def _author(fem, *, body_force, from_model_case=None):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=mat, body_force=body_force)
    if from_model_case is not None:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model(from_model_case)
    return ops


def test_collinear_gravity_overlap_warns(box_fem):
    """body_force (0,0,-b) + from_model gravity (0,0,-w) on the same nodes
    → collinear → double-count warning."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0))
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0),
                  from_model_case="dead")
    with pytest.warns(WarnBodyForceDoubleCount, match="double-counted"):
        ops.build().emit(RecordingEmitter())


def test_orthogonal_lateral_load_does_not_warn(box_fem):
    """body_force vertical + from_model LATERAL (x) load → orthogonal →
    no double-count (the self-weight + pushover combo must stay quiet)."""
    _inject_loads(box_fem, (100.0, 0.0, 0.0))
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0),
                  from_model_case="dead")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnBodyForceDoubleCount)
        ops.build().emit(RecordingEmitter())  # must not raise


def test_body_force_only_does_not_warn(box_fem):
    """No from_model import → nothing to double-count."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0))  # present but never imported
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnBodyForceDoubleCount)
        ops.build().emit(RecordingEmitter())


def test_from_model_gravity_only_does_not_warn(box_fem):
    """from_model gravity with NO element body_force → single gravity."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0))
    ops = _author(box_fem, body_force=None, from_model_case="dead")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnBodyForceDoubleCount)
        ops.build().emit(RecordingEmitter())


def test_warning_names_pg_and_case(box_fem):
    """The message is actionable: it names the colliding PG and case."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0))
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0),
                  from_model_case="dead")
    with pytest.warns(WarnBodyForceDoubleCount) as rec:
        ops.build().emit(RecordingEmitter())
    msg = str(rec[0].message)
    assert "soil" in msg and "dead" in msg


def test_ladruno_up_body_double_count_warns(box_fem):
    """ADR 0074: LadrunoUP names its always-on solid self-weight ``body``
    (an acceleration), not ``body_force`` — the guard must still catch a
    ``from_model`` gravity overlap on the same nodes (direction-only
    collinearity, so the acceleration-vs-force unit difference is moot)."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0))
    ops = apeSees(box_fem)
    # Pure-saturated envelope ndf = ndm+1 (no builder bracket needed); no
    # analysis chain is registered, so the D4 solver gate does not fire.
    ops.model(ndm=3, ndf=4)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25, rho=2000.0))
    ops.element.LadrunoUP(
        pg="soil", material=mat, Kf=2.2e6, poro=0.4, rhoF=1.0,
        perm=(1e-8, 1e-8, 1e-8), body=(0.0, 0.0, -9.81),
    )
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.from_model("dead")
    with pytest.warns(WarnBodyForceDoubleCount, match="double-counted"):
        ops.build().emit(RecordingEmitter())
