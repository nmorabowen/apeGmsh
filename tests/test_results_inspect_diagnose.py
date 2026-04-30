"""``Results.inspect.diagnose(component)`` — routing-side diagnostics.

When a viewer or downstream consumer asks for a component and the slab
comes back empty, this method explains *which topology levels were
checked, where the component was found (if anywhere), and what
canonicals are actually present at each level*. It's the answer to
"why isn't my diagram rendering?"
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.results import Results


_FRAME = Path("tests/fixtures/results/elasticFrame.mpco")
_SPRINGS = Path("tests/fixtures/results/zl_springs.mpco")


@pytest.fixture
def frame_stage():
    if not _FRAME.exists():
        pytest.skip(f"Missing fixture: {_FRAME}")
    r = Results.from_mpco(_FRAME)
    return r.stage(r.stages[0].name)


def test_diagnose_finds_known_line_station_component(frame_stage):
    report = frame_stage.inspect.diagnose("axial_force")
    assert "FOUND in: line_stations" in report


def test_diagnose_finds_known_nodal_component(frame_stage):
    report = frame_stage.inspect.diagnose("displacement_z")
    assert "FOUND in: nodes" in report


def test_diagnose_unknown_component_lists_each_level(frame_stage):
    report = frame_stage.inspect.diagnose("totally_imaginary_thing")
    assert "NOT FOUND" in report
    # Each topology should appear in the report.
    for topology in ("nodes", "line_stations", "gauss",
                     "fibers", "layers", "springs"):
        assert topology in report


def test_diagnose_lists_available_components_at_each_level(frame_stage):
    """Sanity: the report shows real canonicals from each topology.

    The preview is the first 6 alphabetical components per level. For
    elasticFrame.mpco the nodes list begins with acceleration_*, so we
    just verify *some* nodal canonical surfaces. The line_stations
    list is short enough that all 6 canonicals fit in the preview.
    """
    report = frame_stage.inspect.diagnose("axial_force")
    # Some nodal canonical from the alphabetically-first preview
    assert "acceleration_x" in report
    # line_stations canonicals are all in the preview
    assert "bending_moment_z" in report


def test_diagnose_includes_remediation_hints_when_not_found(frame_stage):
    report = frame_stage.inspect.diagnose("xyz_not_a_thing")
    assert "spelling" in report.lower()
    assert "recorder" in report.lower()


def test_diagnose_marks_empty_topologies(frame_stage):
    """Topologies with no buckets at all should be labeled."""
    report = frame_stage.inspect.diagnose("axial_force")
    # gauss / fibers / layers / springs have no buckets in this fixture.
    assert "(empty" in report


def test_diagnose_finds_spring_components_on_springs_fixture():
    if not _SPRINGS.exists():
        pytest.skip(f"Missing fixture: {_SPRINGS}")
    r = Results.from_mpco(_SPRINGS)
    s = r.stage(r.stages[0].name)
    report = s.inspect.diagnose("spring_force_0")
    assert "FOUND in: springs" in report


def test_diagnose_handles_explicit_stage_arg():
    """Explicit stage arg must work even when there are multiple stages."""
    if not _FRAME.exists():
        pytest.skip(f"Missing fixture: {_FRAME}")
    r = Results.from_mpco(_FRAME)
    if len(r.stages) < 2:
        pytest.skip("fixture has only one stage")
    name = r.stages[1].name
    # Top-level Results — multi-stage; pass stage explicitly.
    report = r.inspect.diagnose("displacement_z", stage=name)
    assert "FOUND in: nodes" in report
