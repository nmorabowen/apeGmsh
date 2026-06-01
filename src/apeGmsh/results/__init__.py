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
from ._slabs import (
    ElementSlab,
    FiberSlab,
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
from .demo import make_demo_results

__all__ = [
    "Results",
    "make_demo_results",
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
    "GaussSlab",
    "FiberSlab",
    "LayerSlab",
    "LocalAxes",
]
