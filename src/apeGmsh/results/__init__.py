"""Results module — backend-agnostic FEM post-processing.

The user-facing entry point is the :class:`Results` class.

Quick start
-----------
::

    from apeGmsh import Results

    results = Results.from_native("run.h5")     # or from_mpco, from_recorders
    print(results.inspect.summary())

    # Auto-resolve when there's only one stage
    disp = results.nodes.get(component="displacement_z", pg="Top")

    # Multi-stage: pick one
    gravity = results.stage("gravity")
    sigma = gravity.elements.gauss.get(component="stress_xx", pg="Body")

    # Modes (kind="mode" stages)
    for mode in results.modes:
        print(mode.mode_index, mode.frequency_hz)
"""
from ._beam_diagrams import (
    BeamSpanDiagram,
    BeamSpanSegmentLoad,
    parse_ele_load_beam_uniform,
    parse_membrane_subsegment_loads,
    reconstruct_beam_element_diagram,
    reconstruct_beam_span_diagrams,
)
from ._plane_recovery import OutOfPlaneRecoveryWarning
from ._slabs import (
    ElementSlab,
    FiberSlab,
    GaussCensus,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    LocalAxes,
    NodeSlab,
)
from .readers import (
    EigenMode,
    NativeReader,
    ResultLevel,
    ResultsReader,
    StageInfo,
    TimeSlice,
)
from .Results import Results
from .readers._ladruno_from_mpco import MPCOConversionRefused, convert_mpco
from .demo import make_demo_results
from .monitor import MonitorData, read_monitor, tail_monitor

__all__ = [
    "Results",
    "make_demo_results",
    # Explicit one-way MPCO -> .ladruno conversion (ADR 0122 D5)
    "convert_mpco",
    "MPCOConversionRefused",
    # Exact piecewise-cubic beam span diagram reconstruction
    "BeamSpanSegmentLoad",
    "BeamSpanDiagram",
    "parse_membrane_subsegment_loads",
    "parse_ele_load_beam_uniform",
    "reconstruct_beam_element_diagram",
    "reconstruct_beam_span_diagrams",
    # Ladruno live-telemetry monitor sink (lightweight, not a Results)
    "read_monitor",
    "tail_monitor",
    "MonitorData",
    # Backend-protocol exports
    "ResultsReader",
    "NativeReader",
    "ResultLevel",
    "StageInfo",
    "EigenMode",
    "TimeSlice",
    # Slab dataclasses (returned from .get())
    "NodeSlab",
    "ElementSlab",
    "LineStationSlab",
    "GaussCensus",
    "GaussSlab",
    "FiberSlab",
    "LayerSlab",
    "LocalAxes",
    # 2-D out-of-plane recovery advisory (filterable)
    "OutOfPlaneRecoveryWarning",
]
