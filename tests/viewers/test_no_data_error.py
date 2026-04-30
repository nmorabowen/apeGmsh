"""Diagrams raise ``NoDataError`` instead of silently rendering nothing.

The contract: when a diagram's ``attach()`` finds no data to render
(empty slab, selector resolved to zero entities, etc.), it MUST raise
``NoDataError`` so the Diagrams tab can surface the failure to the
user. Without this, the diagram added a blank actor and the user had
no signal that anything was wrong.
"""
from __future__ import annotations

from pathlib import Path

import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DiagramSpec,
    LineForceDiagram,
    LineForceStyle,
    NoDataError,
    SlabSelector,
    SpringForceDiagram,
    SpringForceStyle,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


_FRAME = Path("tests/fixtures/results/elasticFrame.mpco")
_SPRINGS = Path("tests/fixtures/results/zl_springs.mpco")


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


# =====================================================================
# LineForceDiagram on the wrong fixture
# =====================================================================

def test_line_force_raises_nodata_for_unknown_component(headless_plotter):
    if not _FRAME.exists():
        pytest.skip(f"Missing fixture: {_FRAME}")
    r = Results.from_mpco(_FRAME)
    s = r.stage(r.stages[0].name)
    scene = build_fem_scene(s.fem)

    spec = DiagramSpec(
        kind="line_force",
        selector=SlabSelector(component="totally_not_a_thing"),
        style=LineForceStyle(),
    )
    diagram = LineForceDiagram(spec, s)
    with pytest.raises(NoDataError, match="totally_not_a_thing"):
        diagram.attach(headless_plotter, s.fem, scene)


def test_line_force_raises_nodata_on_fixture_with_no_line_stations(
    headless_plotter,
):
    """zl_springs.mpco has no beam line-station data."""
    if not _SPRINGS.exists():
        pytest.skip(f"Missing fixture: {_SPRINGS}")
    r = Results.from_mpco(_SPRINGS)
    s = r.stage(r.stages[0].name)
    scene = build_fem_scene(s.fem)

    spec = DiagramSpec(
        kind="line_force",
        selector=SlabSelector(component="bending_moment_z"),
        style=LineForceStyle(),
    )
    diagram = LineForceDiagram(spec, s)
    with pytest.raises(NoDataError):
        diagram.attach(headless_plotter, s.fem, scene)


# =====================================================================
# ContourDiagram on a missing nodal component
# =====================================================================

def test_contour_raises_nodata_for_missing_nodal_component(headless_plotter):
    if not _FRAME.exists():
        pytest.skip(f"Missing fixture: {_FRAME}")
    r = Results.from_mpco(_FRAME)
    s = r.stage(r.stages[0].name)
    scene = build_fem_scene(s.fem)

    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="not_a_real_component"),
        style=ContourStyle(),
    )
    diagram = ContourDiagram(spec, s)
    with pytest.raises(NoDataError, match="not_a_real_component"):
        diagram.attach(headless_plotter, s.fem, scene)


# =====================================================================
# SpringForceDiagram on a fixture with no springs
# =====================================================================

def test_spring_force_raises_nodata_when_no_springs(headless_plotter):
    """elasticFrame.mpco has no zero-length spring elements."""
    if not _FRAME.exists():
        pytest.skip(f"Missing fixture: {_FRAME}")
    r = Results.from_mpco(_FRAME)
    s = r.stage(r.stages[0].name)
    scene = build_fem_scene(s.fem)

    spec = DiagramSpec(
        kind="spring_force",
        selector=SlabSelector(component="spring_force_0"),
        style=SpringForceStyle(),
    )
    diagram = SpringForceDiagram(spec, s)
    with pytest.raises(NoDataError, match="spring"):
        diagram.attach(headless_plotter, s.fem, scene)


# =====================================================================
# Registry rolls back on attach failure
# =====================================================================

def test_registry_rolls_back_on_attach_failure(headless_plotter):
    """If attach raises, the diagram must not stay in the registry."""
    if not _FRAME.exists():
        pytest.skip(f"Missing fixture: {_FRAME}")
    r = Results.from_mpco(_FRAME)
    s = r.stage(r.stages[0].name)
    scene = build_fem_scene(s.fem)

    from apeGmsh.viewers.diagrams._registry import DiagramRegistry

    reg = DiagramRegistry()
    reg.bind(headless_plotter, s.fem, scene)

    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="totally_not_real"),
        style=ContourStyle(),
    )
    diagram = ContourDiagram(spec, s)

    with pytest.raises(NoDataError):
        reg.add(diagram)

    # Roll-back: registry has zero diagrams.
    assert len(list(reg.diagrams())) == 0
