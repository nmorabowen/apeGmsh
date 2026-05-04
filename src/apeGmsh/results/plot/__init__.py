"""results.plot — static matplotlib renderer for Results.

Parallel to :meth:`Results.viewer` (interactive Qt+VTK), but produces
publication-ready matplotlib figures suitable for headless pipelines
and inclusion in papers.

Phase 1 covers ``contour``, ``deformed``, ``history``, and ``mesh``.
Line forces / reactions / loads / fibers come later.

Quick start::

    from apeGmsh import Results
    import matplotlib.pyplot as plt

    results = Results.from_native("run.h5")
    results.plot.contour("displacement_z", step=-1)
    plt.savefig("u_z.png")

    results.plot.deformed(step=-1, scale=50, component="stress_xx")
    results.plot.history(node=412, component="displacement_x")
"""
from ._plot import ResultsPlot

__all__ = ["ResultsPlot"]
