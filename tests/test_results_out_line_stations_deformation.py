"""The ``.out`` emit refuses line-stations section deformations (#2052).

Oracle: the ``.out`` route records a line-stations record as
``section force`` (or ``localForce`` for an elastic beam), and the
transcoder decodes those columns as section forces. Before #2052 a
record of ``axial_strain`` / ``curvature_*`` / ``torsional_strain`` /
``shear_strain_*`` emitted the same ``section force`` recorder, so forces
read back labelled as deformations. The response catalog routes every one
of those canonicals to ``section.deformation``
(``gauss_keyword_for_canonical(c, topology="line_stations")``), a bucket
this route never records, so the emit now refuses by name, as
``collect_mpco_tokens`` does for MPCO since #2025.

Every ``.out`` route reaches the recorder through ``emit_logical``: the
Tcl and Python exports, the live ``emit_recorders`` context (whose
``__enter__`` translates every record before issuing any), and the file
listing. Each one is exercised here. Force-only records still emit the
exact pair they did before.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._vocabulary import LINE_DIAGRAMS, LINE_STATION_DEFORMATIONS
from apeGmsh.opensees._response_catalog import gauss_keyword_for_canonical
from apeGmsh.results.spec._emit import emit_logical, to_ops_args
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from apeGmsh.results.writers._cache import list_source_files
from tests.fixtures.recorder_models import identity_model

_HASH = "h2052"


def _line(components, *, name="beam", cls=None) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name=name, components=tuple(components),
        dt=None, n_steps=None, element_ids=np.array([1, 2]),
        element_class_name=cls,
    )


def _spec(rec: ResolvedRecorderRecord) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(fem_snapshot_id=_HASH, records=(rec,))


def _model():
    return identity_model([1, 2], _HASH)


class _FakeOps:
    def __init__(self) -> None:
        self.recorder_calls: list[tuple] = []

    def recorder(self, *args) -> int:
        self.recorder_calls.append(tuple(args))
        return len(self.recorder_calls)

    def remove(self, *args) -> None:
        pass


def test_deformation_vocabulary_routes_to_section_deformation() -> None:
    # Non-vacuous: all six names, each routed to the bucket .out never records.
    assert len(LINE_STATION_DEFORMATIONS) == 6
    for c in LINE_STATION_DEFORMATIONS:
        assert gauss_keyword_for_canonical(
            c, topology="line_stations") == "section.deformation"
    for c in LINE_DIAGRAMS:
        assert gauss_keyword_for_canonical(
            c, topology="line_stations") == "section.force"


def _route_tcl(rec, tmp_path):
    return _spec(rec).to_tcl_commands(model=_model())


def _route_python(rec, tmp_path):
    return _spec(rec).to_python_commands(model=_model())


def _route_live(rec, tmp_path):
    fake = _FakeOps()
    with _spec(rec).emit_recorders(str(tmp_path), ops=fake, model=_model()):
        pass
    return fake.recorder_calls


def _route_files(rec, tmp_path):
    return list_source_files(_spec(rec), tmp_path)


def _route_logical(rec, tmp_path):
    return emit_logical(rec)


_ROUTES = [_route_logical, _route_tcl, _route_python, _route_live,
           _route_files]


@pytest.mark.parametrize("canonical", LINE_STATION_DEFORMATIONS)
@pytest.mark.parametrize("route", _ROUTES)
def test_out_deformation_component_raises(route, canonical, tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match=rf"'kappa_beam'.*line_stations.*'{canonical}'.*section force",
    ):
        route(_line((canonical,), name="kappa_beam"), tmp_path)


@pytest.mark.parametrize("route", _ROUTES)
def test_out_mixed_record_refuses_whole_naming_deformations(
    route, tmp_path,
) -> None:
    """Force + deformation refuses as a whole and names each deformation."""
    rec = _line(LINE_DIAGRAMS + LINE_STATION_DEFORMATIONS, name="mixed")
    with pytest.raises(ValueError) as info:
        route(rec, tmp_path)
    msg = str(info.value)
    assert "'mixed'" in msg
    for c in LINE_STATION_DEFORMATIONS:
        assert repr(c) in msg
    for c in LINE_DIAGRAMS:
        assert repr(c) not in msg


def test_out_elastic_beam_deformation_raises() -> None:
    """The localForce route refuses too: it writes end forces only."""
    with pytest.raises(ValueError, match=r"'curvature_z'.*localForce"):
        emit_logical(_line(("axial_force", "curvature_z"),
                           cls="ElasticBeam3d"))


def test_out_force_only_record_emits_unchanged(tmp_path) -> None:
    rec = _line(LINE_DIAGRAMS)
    force, gpx = emit_logical(rec)
    assert to_ops_args(force) == (
        "Element", "-file", "beam_line_stations.out", "-time", "-ele", 1, 2,
        "-precision", 17, "section", "force")
    assert to_ops_args(gpx) == (
        "Element", "-file", "beam_line_stations_gpx.out", "-time", "-ele",
        1, 2, "-precision", 17, "integrationPoints")
    assert _route_tcl(rec, tmp_path) == [
        "recorder Element -file beam_line_stations.out -time -ele 1 2 "
        "-precision 17 section force  ;# beam line_stations",
        "recorder Element -file beam_line_stations_gpx.out -time -ele 1 2 "
        "-precision 17 integrationPoints  ;# beam line_stations gpx",
    ]
    calls = []
    fake = _FakeOps()
    with _spec(rec).emit_recorders(str(tmp_path), ops=fake,
                                   model=_model()) as live:
        live.begin_stage("s")
        calls = fake.recorder_calls
        live.end_stage()
    assert [c[-2:] for c in calls] == [
        ("section", "force"), (17, "integrationPoints")]


def test_out_elastic_force_only_emits_local_force() -> None:
    [lr] = emit_logical(_line(LINE_DIAGRAMS, cls="ElasticBeam3d"))
    assert lr.response_tokens == ("localForce",)
