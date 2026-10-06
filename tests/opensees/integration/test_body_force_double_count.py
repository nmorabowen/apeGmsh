"""ADR 0054 close-out — the self-weight double-count guard.

A continuum element's constructor ``body_force`` is applied every step with
no load pattern (verified against Brick.cpp / FourNodeQuad.cpp + a live
probe).  If a ``p.from_model(case)`` also drives a gravity load onto the same
nodes **along the same axis**, the region carries its weight twice.
``validate_body_force_double_count`` warns (fail-soft) on that overlap, and is
silent for the legitimate *lateral-load + self-weight* combo (orthogonal, so
not collinear).

Collinearity alone cannot tell a reduced self-weight from a vertical footing
line load (#1338), so the guard also reads each record's ``source`` — the
definition kind the resolver stamps.  The first block injects records
directly onto ``fem.nodes.loads`` (the resolver output) so the test controls
direction and source exactly; the second block runs the issue's own model
through the real resolver, so the stamping and the guard are proven
together.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.record_sets import NodalLoadSet
from apeGmsh._kernel.records._loads import NodalLoadRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    WarnBodyForceDoubleCount,
)
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


def _inject_loads(fem, force_xyz, *, pattern="dead", source="gravity"):
    recs = [
        NodalLoadRecord(node_id=int(n), force_xyz=force_xyz, pattern=pattern,
                        source=source)
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


def _build_silently(ops):
    """Build with the guard promoted to an error: a warning fails the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnBodyForceDoubleCount)
        ops.build().emit(RecordingEmitter())


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
    _build_silently(ops)


def test_body_force_only_does_not_warn(box_fem):
    """No from_model import → nothing to double-count."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0))  # present but never imported
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0))
    _build_silently(ops)


def test_from_model_gravity_only_does_not_warn(box_fem):
    """from_model gravity with NO element body_force → single gravity."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0))
    ops = _author(box_fem, body_force=None, from_model_case="dead")
    _build_silently(ops)


def test_warning_names_pg_case_and_source(box_fem):
    """The message is actionable: it names the colliding PG, case and
    the source kind that made the overlap a self-weight."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0))
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0),
                  from_model_case="dead")
    with pytest.warns(WarnBodyForceDoubleCount) as rec:
        ops.build().emit(RecordingEmitter())
    msg = str(rec[0].message)
    assert "soil" in msg and "dead" in msg and "source gravity" in msg


@pytest.mark.parametrize("source", ["line", "surface", "point", "face_load",
                                    "point_closest"])
def test_collinear_boundary_load_does_not_warn(box_fem, source):
    """#1338: a boundary load (line / surface / point / face) collinear with
    the body force is NOT a second self-weight — silent."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0), source=source)
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0),
                  from_model_case="dead")
    _build_silently(ops)


def test_collinear_body_load_warns(box_fem):
    """``g.loads.volume`` (source ``body``) is a body force like gravity."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0), source="body")
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0),
                  from_model_case="dead")
    with pytest.warns(WarnBodyForceDoubleCount, match="source body"):
        ops.build().emit(RecordingEmitter())


def test_unknown_source_still_warns_and_says_so(box_fem):
    """A record with ``source=None`` (a model.h5 older than neutral 2.35.0,
    or a synthesized record) may be either: the guard keeps warning, as
    every reader before the column did, and names the gap."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0), source=None)
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0),
                  from_model_case="dead")
    with pytest.warns(WarnBodyForceDoubleCount, match="source unknown") as rec:
        ops.build().emit(RecordingEmitter())
    assert "2.35.0" in str(rec[0].message)


def test_source_outside_the_vocabulary_fails_closed(box_fem):
    """A source string the vocabulary does not know cannot be classified:
    the guard raises rather than guessing either way."""
    _inject_loads(box_fem, (0.0, 0.0, -100.0), source="thermal")
    ops = _author(box_fem, body_force=(0.0, 0.0, -3924.0),
                  from_model_case="dead")
    with pytest.raises(BridgeError, match="'thermal'"):
        ops.build().emit(RecordingEmitter())


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


# ---------------------------------------------------------------------------
# #1338 end to end: the issue's model through the real resolver
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def soil_2d_fem():
    """The #1338 reproduction: a 2-D soil block whose top edge carries a
    vertical footing line load in case ``footing``, and whose area carries
    gravity in case ``dead``.  One mesh, two cases, resolved for real."""
    W, D = 4.0, 2.0
    g = apeGmsh(model_name="bf_1338", verbose=False)
    g.begin()
    try:
        geo = g.model.geometry
        for lbl, (x, y) in {"bl": (0, -D), "br": (W, -D),
                            "tr": (W, 0), "tl": (0, 0)}.items():
            geo.add_point(x, y, 0, label=lbl)
        geo.add_line("bl", "br", label="base")
        geo.add_line("br", "tr", label="right")
        geo.add_line("tr", "tl", label="top")
        geo.add_line("tl", "bl", label="left")
        geo.add_curve_loop(["base", "right", "top", "left"], label="loop")
        geo.add_plane_surface("loop", label="soil")
        g.physical.add_surface(["soil"], name="Soil")
        g.physical.add_curve(["base"], name="Base")
        with g.loads.case("footing"):
            g.loads.line("top", magnitude=100.0, direction=(0.0, -1.0, 0.0))
        with g.loads.case("dead"):
            g.loads.gravity("soil", density=1.8, g=(0.0, -10.0, 0.0))
        g.mesh.generation.generate(dim=2)
        yield g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()


def _author_2d(fem, case):
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=3e4, nu=0.3)
    ops.element.Tri31(pg="Soil", material=mat, thickness=1.0,
                      body_force=(0.0, -18.0))
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.from_model(case)
    return ops


def test_resolver_stamps_the_definition_kind(soil_2d_fem):
    loads = soil_2d_fem.nodes.loads
    assert {r.source for r in loads.by_pattern("footing")} == {"line"}
    assert {r.source for r in loads.by_pattern("dead")} == {"gravity"}


def test_issue_1338_vertical_footing_line_load_is_silent(soil_2d_fem):
    """The reported false positive: the footing case holds no gravity, so
    nothing is counted twice, although its nodal loads are collinear with
    the element body force.  Fails with the fix reverted."""
    _build_silently(_author_2d(soil_2d_fem, "footing"))


def test_issue_1338_gravity_case_still_warns(soil_2d_fem):
    """The real hazard on the same mesh: ``g.loads.gravity`` imported onto
    the body_force region."""
    with pytest.warns(WarnBodyForceDoubleCount, match="'dead' \\(source gravity"):
        _author_2d(soil_2d_fem, "dead").build().emit(RecordingEmitter())
