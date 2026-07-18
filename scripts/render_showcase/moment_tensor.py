"""Showcase render — double-couple seismic source radiating through a solid block.

Model (docs/examples/moment-tensor-source.md): a 40 m homogeneous elastic
box (stdBrick, 16^3 structured hexes) with a vertical strike-slip
double-couple embedded at its centre via ``p.moment_tensor`` — the moment
tensor becomes consistent nodal forces M_ij dN_a/dx_j S(t), a pure
right-hand-side load. A Newmark transient radiates the source; the nodal
velocity field is captured every step through ``ops.domain_capture``.

Physics on show: the textbook four-lobe double-couple radiation pattern —
|v| lobes in the horizontal source plane with nodal planes along x and y,
propagating outward at the P/S wave speeds.

Render: off-screen 960x540 plotter, ``build_fem_scene`` + ``ResultsDirector``
driving a nodal ContourDiagram of |v| (custom scalar ``mag(velocity)``) on
the lower half-block — its top face IS the source plane, so the lobes read
like a clip view — exported with ``export_animation`` to
docs/assets/anim/moment-tensor.mp4 (~8 s at 30 fps).

Run:  LADRUNO_OPENSEES_QUIET=1 QT_QPA_PLATFORM=offscreen python moment_tensor.py
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyvista as pv

from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.results.capture.spec import DomainCaptureSpec
from apeGmsh.viewers.animation import export_animation
from apeGmsh.viewers.backends import PyVistaQtBackend
from apeGmsh.viewers.diagrams import (
    ContourDiagram, ContourStyle, DiagramSpec, SlabSelector,
)
from apeGmsh.viewers.diagrams._director import ResultsDirector
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

L = 40.0                                   # box side [m]
CTR = (L / 2, L / 2, L / 2)
DT, T_TOTAL = 5e-4, 0.12                   # through the direct P arrival
OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "anim" / "moment-tensor.mp4"


def solve(run_h5: Path) -> None:
    """Build the box, embed the double-couple, run the transient, capture v."""
    g = apeGmsh(model_name="mt_showcase", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, L, L, L, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.structured.set_transfinite("soil", n=17)   # 16^3 hexes
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data()
    finally:
        g.end()

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    soil = ops.register(ElasticIsotropic(E=2.0e8, nu=0.25, rho=2000.0))
    ops.element.stdBrick(pg="soil", material=soil)

    # The pattern's time series IS the moment function S(t) (smooth 0->1).
    S = ops.timeSeries.MomentStep(
        half_duration=0.008, t_total=T_TOTAL, dt=DT, t0=0.02,
    )
    with ops.pattern.Plain(series=S) as p:
        p.moment_tensor(
            position=CTR, frame="z-down", M0=1.0e8,
            mech=dict(strike=0, dip=90, rake=0),     # vertical strike-slip
            method="consistent", region="soil",
        )

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.Newmark(gamma=0.5, beta=0.25)
    ops.analysis.Transient()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    o = emitter.ops

    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(components="velocity")
    n_steps = int(round(T_TOTAL / DT))
    t0 = time.perf_counter()
    with ops.domain_capture(spec, path=run_h5, ops=o) as cap:
        cap.begin_stage("radiation", kind="transient")
        for k in range(n_steps):
            rc = emitter.analyze(steps=1, dt=DT)
            if rc != 0:
                raise RuntimeError(f"analyze failed at step {k} (rc={rc})")
            cap.step(t=o.getTime())
        cap.end_stage()
    print(f"solve: {n_steps} steps in {time.perf_counter() - t0:.1f} s")


def render(run_h5: Path) -> None:
    """Contour |v| on the lower half-block and export the mp4."""
    results = Results.from_native(run_h5, model=OpenSeesModel.from_h5(run_h5))
    results.nodes.define(
        "velocity_magnitude", "mag(velocity)", label="|v|", units="m/s",
    )

    coords = np.asarray(results.fem.nodes.coords, dtype=float)
    ids = np.asarray(results.fem.nodes.ids)
    half = tuple(int(i) for i in ids[coords[:, 2] <= L / 2 + 1e-6])

    # Colour scale fixed over the whole transient (robust to the near-
    # source amplitude spike: clip at the 99th percentile).
    slab = results.nodes.get(ids=half, component="velocity_magnitude")
    vmax = float(np.percentile(slab.values, 99.0))

    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = (960, 540)
    plotter.set_background("white")
    scene = build_fem_scene(results.fem)
    plotter.add_mesh(scene.grid, style="wireframe", color="gray", opacity=0.15)

    director = ResultsDirector(results)
    director._render_callback = plotter.render  # noqa: SLF001

    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="velocity_magnitude", ids=half),
        style=ContourStyle(cmap="magma_r", clim=(0.0, vmax), show_edges=False),
    )
    diagram = ContourDiagram(spec, results)
    diagram.attach(PyVistaQtBackend(plotter), results.fem, scene)
    director.registry.add(diagram)

    plotter.add_text(
        "Double-couple seismic source - |v| radiation pattern",
        position="upper_left", font_size=10, color="black",
    )
    plotter.camera_position = [(100.0, -60.0, 78.0), (20.0, 20.0, 10.0), (0.0, 0.0, 1.0)]

    export_animation(
        plotter, director, OUT, fps=30, step_stride=1,
        progress=lambda done, total: (
            print(f"frame {done}/{total}") if done % 60 == 0 or done == total else None
        ),
    )
    plotter.close()
    results.close()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="mt_showcase_") as td:
        run_h5 = Path(td) / "run.h5"
        solve(run_h5)
        render(run_h5)
