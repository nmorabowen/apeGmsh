"""Modal sweep showcase — first three bending modes of a steel cantilever.

Builds the eigen model from ``docs/examples/modal-analysis.md``: a 4 m steel
cantilever (E = 200 GPa, rho = 7850 kg/m^3, 0.10 x 0.20 m rectangle) meshed
into ~20 ``elasticBeamColumn`` elements with consistent distributed mass, and
solves K.phi = omega^2 M.phi for the first three modes (Euler-Bernoulli says
10.19 / 63.87 / 178.85 Hz). Mode stages are one-step snapshots, so to make
them *move* the script reads each eigenvector back through ``results.modes``
and synthesizes one stitched transient stage with ``NativeWriter`` —
u(t) = phi * sin(2*pi*t/T), 80 steps per mode, amplitude scaled to 15 % of
the span. It then renders that stage off-screen (pure VTK, no display: the
substrate grid warped by displacement per step, driven by a
``ResultsDirector`` through ``export_animation``) to
``docs/assets/anim/modal-sweep.mp4`` — 8 s @ 30 fps, 960x540.

Run:  python scripts/render_showcase/modal_sweep.py
"""
from __future__ import annotations

import os

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv

from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees, OpenSeesModel
from apeGmsh.results.capture.spec import DomainCaptureSpec
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.animation import export_animation
from apeGmsh.viewers.diagrams._director import ResultsDirector
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets" / "anim" / "modal-sweep.mp4"

# --- Problem data (consistent SI: m, N, Pa, kg) ---
L, E = 4.0, 200e9
b, h, rho = 0.10, 0.20, 7850.0
A, Iz, mbar = b * h, b * h**3 / 12.0, rho * b * h

# --- Animation layout ---
FPS = 30
STEPS_PER_MODE = 80          # 3 x 80 / 30 fps = 8 s total
CYCLE = 20                   # frames per oscillation (4 swings per mode)
AMP = 0.15 * L               # peak modal deflection [m], visible against L
STAGE = "modal sweep"


def solve_modes(run_h5: str) -> None:
    """Cantilever eigen model from the modal-analysis example -> run.h5."""
    with apeGmsh(model_name="modal-sweep") as g:
        p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
        p1 = g.model.geometry.add_point(L, 0.0, 0.0)
        beam = g.model.geometry.add_line(p0, p1)
        g.model.sync()
        g.physical.add(1, [beam], name="Beam")
        g.physical.add(0, [p0], name="Fixed")
        g.mesh.sizing.set_global_size(L / 20.0)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Beam", transf=transf, A=A, E=E, Iz=Iz,
        mass=mbar, c_mass=True,
    )
    ops.fix(pg="Fixed", dofs=(1, 1, 1))

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    ops.analyze(steps=1)                     # assemble K + M

    spec = DomainCaptureSpec(opensees=ops)
    spec.modal(n_modes=3)
    with ops.domain_capture(spec, path=run_h5) as cap:
        cap.capture_modes()


def synthesize_sweep(run_h5: str, anim_h5: str) -> list[float]:
    """Stitch the three eigenvectors into one oscillating transient stage."""
    results = Results.from_native(run_h5, model=OpenSeesModel.from_h5(run_h5))
    fem = results.model.fem
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_total = 3 * STEPS_PER_MODE
    U = np.zeros((n_total, ids.size, 3))
    envelope = np.sin(2.0 * np.pi * np.arange(STEPS_PER_MODE) / CYCLE)
    for k, mode in enumerate(sorted(results.modes, key=lambda m: m.mode_index)):
        phi = np.zeros((ids.size, 3))
        for j, comp in enumerate(("displacement_x", "displacement_y")):
            sl = mode.nodes.get(ids=ids, component=comp)
            col = {int(n): v for n, v in zip(sl.node_ids, sl.values[0])}
            phi[:, j] = [col.get(int(n), 0.0) for n in ids]
        phi *= AMP / np.linalg.norm(phi, axis=1).max()   # unit tip -> AMP
        s = k * STEPS_PER_MODE
        U[s:s + STEPS_PER_MODE] = phi[None, :, :] * envelope[:, None, None]
    freqs = [m.frequency_hz for m in
             sorted(results.eigen_modes, key=lambda m: m.mode_index)]
    results.close()

    with NativeWriter(anim_h5) as w:
        w.open(fem=fem, model_h5_src=run_h5)     # Composed: carry the model
        sid = w.begin_stage(
            name=STAGE, kind="transient",
            time=np.arange(n_total, dtype=np.float64) / FPS,
        )
        w.write_nodes(
            sid, "partition_0", node_ids=ids,
            components={
                "displacement_x": U[:, :, 0],
                "displacement_y": U[:, :, 1],
                "displacement_z": U[:, :, 2],
            },
        )
        w.end_stage()
    return freqs


def render(anim_h5: str, freqs: list[float]) -> None:
    """Off-screen plotter + director; warp the grid by displacement."""
    results = Results.from_native(anim_h5, model=OpenSeesModel.from_h5(anim_h5))
    scene = build_fem_scene(results.fem)
    grid, ref = scene.grid, np.array(scene.reference_points, copy=True)

    # (n_steps, n_nodes, 3) displacement history, in grid point order.
    U = np.zeros((3 * STEPS_PER_MODE, ref.shape[0], 3))
    for j, comp in enumerate(("displacement_x", "displacement_y")):
        sl = results.stage(STAGE).nodes.get(ids=scene.node_ids, component=comp)
        pos = {int(n): i for i, n in enumerate(sl.node_ids)}
        U[:, :, j] = sl.values[:, [pos[int(n)] for n in scene.node_ids]]

    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = (960, 540)
    plotter.set_background("white")
    plotter.add_mesh(grid, color="#2b6cb0", line_width=7,
                     render_lines_as_tubes=True)
    plotter.add_points(ref[:1], color="#1a202c", point_size=14,
                       render_points_as_spheres=True)    # the clamped end
    plotter.camera_position = [(L / 2, 0.0, 6.0), (L / 2, 0.0, 0.0),
                               (0.0, 1.0, 0.0)]
    label = plotter.add_text("", position="upper_left", font_size=14,
                             color="#1a202c")

    director = ResultsDirector(results)
    director._render_callback = plotter.render  # noqa: SLF001

    def _on_step(step: int) -> None:
        grid.points = ref + U[step]
        k = min(step // STEPS_PER_MODE, 2)
        text = f"Mode {k + 1}   f = {freqs[k]:.2f} Hz"
        label.SetText(2, text) if hasattr(label, "SetText") \
            else label.SetInput(text)
    director.on_step_changed.append(_on_step)
    _on_step(0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_animation(plotter, director, OUT, fps=FPS)
    finally:
        plotter.close()
        results.close()


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        run_h5 = str(Path(tmp) / "run.h5")
        anim_h5 = str(Path(tmp) / "modal-sweep.h5")
        solve_modes(run_h5)
        freqs = synthesize_sweep(run_h5, anim_h5)
        print("modes [Hz]:", ", ".join(f"{f:.2f}" for f in freqs))
        render(anim_h5, freqs)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
