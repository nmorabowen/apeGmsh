"""Plane-wave SSI showcase — a shear pulse climbs the absorbing soil column.

Builds the plane-wave capstone from ``docs/examples/plane-wave-ssi.md``:
a 40 m elastic soil column (Vs = 200 m/s) wrapped in its one-element
``ASDAbsorbingBoundary3D`` skin by ``g.parts.add_plane_wave_box``, shaken
from below by a 4 Hz Ricker shear-velocity pulse, solved as the example's
staged transient through the emitted openseespy deck. A staged deck runs
in a subprocess (live capture can't see it), so the full nodal
displacement field rides the fork's ``recorder ladruno`` and is read back
with ``Results.from_ladruno``. The render is a pure off-screen PyVista
plotter: a |u| ``ContourDiagram`` on the amplified-warp column in 3/4
view — the pulse sweeps up, doubles off the free surface, radiates out
the base, and the column goes quiet — encoded to
``docs/assets/anim/plane-wave-ssi.mp4`` (~8 s @ 30 fps, 960x540).

Run:  python scripts/render_showcase/plane_wave_ssi.py
"""
from __future__ import annotations

import os

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.time_series.time_series import Ricker

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets" / "anim" / "plane-wave-ssi.mp4"

# --- Problem data (docs/examples/plane-wave-ssi.md, consistent SI) ---
rho, nu, Vs = 2000.0, 0.3, 200.0     # density, Poisson, shear-wave speed
G = rho * Vs**2                      # 8.0e7 Pa
E = 2.0 * G * (1.0 + nu)             # 2.08e8 Pa
H = 40.0                             # soil column depth [m]
f_n, t_center, dt, t_total = 4.0, 0.15, 0.002, 1.0
n_steps = round(t_total / dt)        # 500 transient steps

# --- Animation layout ---
FPS = 30
STEP_STRIDE = 2                      # 500 steps -> ~251 frames ~ 8.4 s


def solve(work: Path):
    """The example's staged model + a whole-model .ladruno field capture."""
    lad, surf = work / "field.ladruno", work / "surf_vel.out"

    with apeGmsh(model_name="pwbox_ssi") as g:
        res = g.parts.add_plane_wave_box(
            x=(20.0, 2), y=(20.0, 2), z=(H, 16),   # 64 bricks + 208 skin
        )
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data()

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    soil = ops.register(ElasticIsotropic(E=E, nu=nu, rho=rho))
    base = ops.register(Ricker(f_n=f_n, t_total=t_total, dt=dt,
                               t_center=t_center, kind="velocity",
                               factor=0.01))
    ops.element.stdBrick(pg=res.soil_pg, material=soil)          # the soil
    ops.element.absorbing_boundary(                              # the skin
        skin=res, material=soil, base_series=base, base_dirs=("x",),
    )

    with ops.stage(name="dynamic") as s:
        s.activate_absorbing(pg=res.skin_all_pg)                 # hold -> absorb
        s.recorder(ops.recorder.Ladruno(                         # full field
            file=str(lad), nodal_responses=("displacement",),
        ))
        s.recorder(ops.recorder.Node(                            # example probe
            file=str(surf), response="vel",
            pg=res.free_surface_pg, dofs=(1,), time_format="dt",
        ))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=20),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.Newmark(gamma=0.5, beta=0.25),
            constraints=ops.constraints.Transformation(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Transient(),
        )
        s.run(n_increments=n_steps, dt=dt)

    deck = work / "ssi.py"
    ops.py(str(deck))
    with open(deck, "a", encoding="utf-8") as f:
        f.write("ops.wipe()\n")      # flush the .ladruno before exit
    subprocess.run([sys.executable, str(deck)], check=True, cwd=work)

    # The example's two physics checks: arrival ~ H/Vs, then quiet.
    data = np.loadtxt(surf)
    t, v = data[:, 0], np.abs(data[:, 1:].mean(axis=1))
    peak = v.max()
    print(f"first arrival   = {t[int(np.argmax(v > 0.05 * peak))]:.3f} s "
          f"(H/Vs = {H / Vs:.3f} s)")
    print(f"late/peak       = {v[t >= 0.7].max() / peak:.2%} (radiated)")
    return lad, fem


def render(lad: Path, fem) -> None:
    """Off-screen plotter: |u| contour on the amplified-warp column."""
    import pyvista as pv
    from apeGmsh.viewers.animation import export_animation
    from apeGmsh.viewers.backends import PyVistaQtBackend
    from apeGmsh.viewers.diagrams import (
        ContourDiagram, ContourStyle, DiagramSpec, ResultsDirector,
        SlabSelector,
    )
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    with Results.from_ladruno(lad, fem=fem) as r:
        r.nodes.define("disp_mag", "mag(displacement)",
                       label="|u|", units="m")

        # Precompute the warped point history: u is ~3e-5 m on a 40 m
        # column, so amplify until the peak sway reads as ~5 % of H.
        scene = build_fem_scene(r.fem)
        ref = np.asarray(scene.grid.points, dtype=np.float64).copy()
        ux = r.nodes.get(ids=scene.node_ids, component="displacement_x")
        scale = 0.05 * H / float(np.abs(ux.values).max())
        warped = np.repeat(ref[None], len(ux.values), axis=0)
        rows = [scene.node_id_to_idx[int(n)] for n in ux.node_ids]
        warped[:, rows, 0] += scale * np.asarray(ux.values)
        umax = float(r.nodes.get(component="disp_mag").values.max())

        plotter = pv.Plotter(off_screen=True, window_size=(960, 540))
        plotter.add_mesh(scene.grid, color="lightgray", show_edges=True)

        director = ResultsDirector(r)
        director._render_callback = plotter.render  # noqa: SLF001
        spec = DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="disp_mag"),
            style=ContourStyle(cmap="turbo", clim=(0.0, umax),
                               show_edges=True, fmt="%.1e"),
        )
        contour = ContourDiagram(spec, r)
        contour.attach(PyVistaQtBackend(plotter), r.fem, scene)
        director.registry.add(contour)
        plotter.add_text(
            "Plane-wave SSI - 4 Hz Ricker pulse through an absorbing "
            f"soil column  (x{scale:.0f} deformation)",
            position="upper_left", font_size=11,
        )

        # Warp before each diagram update so geometry and colors land
        # on the same step (same seam as the interactive viewer).
        base_set_step = director.set_step

        def _warp_then_step(i: int) -> None:
            pts = warped[min(int(i), warped.shape[0] - 1)]
            scene.grid.points = pts
            contour.sync_substrate_points(pts, scene)
            base_set_step(i)

        director.set_step = _warp_then_step
        plotter.camera_position = "iso"          # 3/4 view of the column
        plotter.camera.zoom(1.25)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        export_animation(plotter, director, OUT, fps=FPS,
                         step_stride=STEP_STRIDE)
        plotter.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        lad, fem = solve(Path(tmp))
        print(f"solved {n_steps} transient steps in "
              f"{time.perf_counter() - t0:.1f} s")
        render(lad, fem)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
