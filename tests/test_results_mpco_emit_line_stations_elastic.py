"""MPCO emit of an elastic-beam line-stations record (#2051).

Elastic beams (``ElasticBeam2d/3d``, ``ElasticTimoshenkoBeam2d/3d``,
``ModElasticBeam2d``) answer ``localForce`` but not ``section.force``. Before
#2051 an MPCO line-stations record emitted ``-E section.force`` only, so an
elastic beam's stations read back empty with no warning. The MPCO reader
already builds two end stations from ``localForce`` (#2020, #2048) and takes
``section.force`` first, so the emit now records ``localForce`` too when the
record names an elastic class, and warns when it cannot tell.

The live round trip is in
``tests/opensees/integration_ladruno/test_mpco_elastic_line_stations_emit_live.py``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh._vocabulary import LINE_DIAGRAMS
from apeGmsh.results.spec._emit import (
    collect_mpco_tokens,
    emit_mpco_python,
    emit_mpco_tcl,
    line_stations_from_local_force,
    mpco_ops_args,
)
from apeGmsh.results.spec._resolved import ResolvedRecorderRecord

#: The elastic-beam family, written out: the oracle does not ask the
#: predicate it checks.
ELASTIC = ("ElasticBeam2d", "ElasticBeam3d", "ElasticTimoshenkoBeam2d",
           "ElasticTimoshenkoBeam3d", "ModElasticBeam2d")
FORCE_BEAMS = ("ForceBeamColumn3d", "DispBeamColumn2d",
               "LadrunoDispBeamColumn3d")


def _line(cls, name="beam", components=LINE_DIAGRAMS) -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name=name, components=tuple(components),
        dt=None, n_steps=None, element_ids=np.array([1, 2]),
        element_class_name=cls,
    )


def _silent(records):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return collect_mpco_tokens(records)


def test_the_elastic_family_is_the_local_force_family() -> None:
    """Not vacuous: the predicate the emit asks names exactly these classes."""
    for cls in ELASTIC:
        assert line_stations_from_local_force(_line(cls)), cls
    for cls in FORCE_BEAMS:
        assert not line_stations_from_local_force(_line(cls)), cls


@pytest.mark.parametrize("cls", ELASTIC)
def test_elastic_class_also_records_local_force(cls: str) -> None:
    assert _silent([_line(cls)]) == ([], ["section.force", "localForce"])


@pytest.mark.parametrize("cls", ELASTIC)
@pytest.mark.parametrize("emit", [mpco_ops_args, emit_mpco_tcl, emit_mpco_python])
def test_every_mpco_route_carries_local_force(emit, cls: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = emit([_line(cls)])
    if emit is mpco_ops_args:
        assert out == ("mpco", "run.mpco", "-E", "section.force", "localForce")
    elif emit is emit_mpco_tcl:
        assert out == "recorder mpco run.mpco -E section.force localForce"
    else:
        assert out == ("ops.recorder('mpco', 'run.mpco', '-E', "
                       "'section.force', 'localForce')")


@pytest.mark.parametrize("cls", FORCE_BEAMS)
def test_section_force_class_records_section_force_only(cls: str) -> None:
    assert _silent([_line(cls)]) == ([], ["section.force"])


def test_mixed_force_and_elastic_records_carry_both_once() -> None:
    recs = [_line("ForceBeamColumn3d", name="cols"),
            _line("ElasticBeam3d", name="girders"),
            _line("ElasticTimoshenkoBeam3d", name="braces")]
    assert _silent(recs) == ([], ["section.force", "localForce"])


def test_no_class_warns_and_keeps_section_force() -> None:
    with pytest.warns(UserWarning) as rec:
        tokens = collect_mpco_tokens([_line(None, name="girder")])
    assert tokens == ([], ["section.force"])
    [w] = [w for w in rec if "line-stations" in str(w.message)]
    msg = str(w.message)
    assert "'girder'" in msg and "names no element class" in msg
    for cls in ELASTIC:
        assert cls in msg
    assert w.filename == __file__         # points at the caller


def test_unknown_class_warns_naming_it() -> None:
    with pytest.warns(UserWarning, match=r"'beam'.*'ElasticBeam3D'"):
        tokens = collect_mpco_tokens([_line("ElasticBeam3D")])
    assert tokens == ([], ["section.force"])


@pytest.mark.parametrize("emit", [mpco_ops_args, emit_mpco_tcl, emit_mpco_python])
def test_warning_reaches_the_user_frame_on_every_route(emit) -> None:
    with pytest.warns(UserWarning, match="names no element class") as rec:
        emit([_line(None)])
    assert {w.filename for w in rec} == {__file__}


def test_elastic_record_with_deformation_still_refuses_whole() -> None:
    rec = _line("ElasticBeam3d", name="kappa",
                components=("bending_moment_z", "curvature_z"))
    with pytest.raises(ValueError, match=r"'kappa'.*'curvature_z'"):
        collect_mpco_tokens([rec])
