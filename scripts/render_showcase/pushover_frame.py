"""Pushover showcase — a W14x90 fiber moment frame sways into its mechanism.

Builds the steel moment-frame pushover from
``docs/examples/pushover-steel-frame.md``: a single-bay portal (B = 6 m,
H = 3.5 m) whose columns carry an apeSteel ``W14X90`` fiber section
(A992, Fy = 345 MPa, Steel02 b = 0.005) through force-based
distributed-plasticity elements (5 Lobatto points), a near-rigid elastic
beam forcing the clean four-hinge column-sway mechanism (Vp = 4Mp/H ~
1013 kN), pushed under displacement control to 2.5 % roof drift. The
full nodal displacement field is captured every step, then rendered
with a pure off-screen PyVista plotter (no Qt window — the desktop
viewer cannot host VTK on Windows' offscreen platform): a displacement-
magnitude ``ContourDiagram`` on tube-styled frame lines over a gray
undeformed ghost, grid points warped x8 each step before the diagram
syncs, encoded to ``docs/assets/anim/pushover-frame.mp4`` (8 s @ 30
fps, 960x540).

Run:  python scripts/render_showcase/pushover_frame.py
"""
from __future__ import annotations

import os

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
from pathlib import Path

import numpy as np

import apeSteel
import openseespy.opensees as opspy
from apeGmsh import apeGmsh, Results
from apeGmsh.opensees import apeSees, OpenSeesModel
from apeGmsh.opensees.section.fiber import RectPatch
from apeGmsh.results.capture.spec import DomainCaptureSpec

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets" / "anim" / "pushover-frame.mp4"

# --- Problem data (N-mm-MPa, apeSteel native base) ---
H, B = 3500.0, 6000.0                # storey height / bay width [mm]
Fy, E, b_hard = 345.0, 200_000.0, 0.005

# --- Animation layout ---
FPS = 30
N_STEPS = 240                        # 240 frames / 30 fps = 8 s
DRIFT = 0.025 * H                    # push to 2.5 % drift = 87.5 mm
DEFORM_SCALE = 8.0                   # x8 warp so the sway reads on screen


def solve(run_h5: str) -> float:
    """E10 fiber-frame pushover -> run.h5 with the full displacement field."""
    props = apeSteel.AISCv16Catalog().get_section_properties("W14X90")
    d, bf = props.overall_depth_d, props.flange_width_bf
    tf, tw = props.flange_thickness_tf, props.web_thickness_tw
    Vp_hand = 4.0 * Fy * props.plastic_section_modulus_strong_axis_Zx / H

    with apeGmsh(model_name="pushover_frame") as g:
        bl = g.model.geometry.add_point(0.0, 0.0, 0.0)
        br = g.model.geometry.add_point(B, 0.0, 0.0)
        tl = g.model.geometry.add_point(0.0, H, 0.0)
        tr = g.model.geometry.add_point(B, H, 0.0)
        col_l = g.model.geometry.add_line(bl, tl)
        col_r = g.model.geometry.add_line(br, tr)
        beam = g.model.geometry.add_line(tl, tr)
        g.model.sync()
        g.physical.add(1, [col_l, col_r], name="Columns")
        g.physical.add(1, [beam], name="Beam")
        g.physical.add(0, [bl, br], name="Base")
        g.physical.add(0, [tl], name="Roof")
        g.mesh.sizing.set_global_size(H / 4.0)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)

    roof_id = int(fem.nodes.select(pg="Roof").ids[0])

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))

    # Columns: W14x90 fibers (flange / web / flange) -> force-based element.
    steel = ops.uniaxialMaterial.Steel02(fy=Fy, E=E, b=b_hard)
    col_sec = ops.section.Fiber(patches=(
        RectPatch(material=steel, ny=4, nz=1,
                  yI=d / 2 - tf, zI=-bf / 2, yJ=d / 2, zJ=bf / 2),
        RectPatch(material=steel, ny=24, nz=1,
                  yI=-(d / 2 - tf), zI=-tw / 2, yJ=d / 2 - tf, zJ=tw / 2),
        RectPatch(material=steel, ny=4, nz=1,
                  yI=-d / 2, zI=-bf / 2, yJ=-(d / 2 - tf), zJ=bf / 2),
    ))
    col_integ = ops.beamIntegration.Lobatto(section=col_sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Columns", transf=transf,
                                integration=col_integ)
    # Beam: ~1000x a column -> near-rigid -> clean column-sway mechanism.
    ops.element.elasticBeamColumn(
        pg="Beam", transf=transf, A=props.gross_area_Ag * 10.0, E=E,
        Iz=props.moment_of_inertia_strong_axis_Ix * 1000.0,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1))

    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        pat.load(pg="Roof", forces=(1.0, 0.0, 0.0))   # direction only

    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-6, max_iter=30)
    ops.algorithm.Newton()
    dU = DRIFT / N_STEPS
    ops.integrator.DisplacementControl(node=roof_id, dof=1, dU=dU)
    ops.analysis.Static()

    # Every node's displacement each step (the animation), plus base
    # reactions so the mechanism plateau can be sanity-checked below.
    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(components="displacement", ids=fem.nodes.ids)
    spec.nodes(pg="Base", components="reaction_force")

    ops.run(wipe=False)
    with ops.domain_capture(spec, path=run_h5) as cap:
        cap.begin_stage("pushover", kind="static")
        for k in range(N_STEPS):
            if opspy.analyze(1) != 0:
                raise RuntimeError(f"non-convergence at step {k}")
            cap.step(t=(k + 1) * dU)
        cap.end_stage()
    return Vp_hand


def render(run_h5: str, Vp_hand: float) -> None:
    """Pure off-screen plotter: |u| contour on the x8-warped swaying frame."""
    import pyvista as pv
    from apeGmsh.viewers.animation import export_animation
    from apeGmsh.viewers.backends import PyVistaQtBackend
    from apeGmsh.viewers.diagrams import (
        ContourDiagram, ContourStyle, DiagramSpec, ResultsDirector,
        SlabSelector,
    )
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    with Results.from_native(run_h5,
                             model=OpenSeesModel.from_h5(run_h5)) as r:
        r.nodes.define("disp_mag", "mag(displacement)",
                       label="|u|", units="mm")

        # Mechanism check against the closed form Vp = 4Mp/H.
        rx = r.nodes.get(pg="Base", component="reaction_force_x")
        V_end = -np.asarray(rx.values)[-1].sum()
        print(f"V(2.5% drift) = {V_end / 1e3:.0f} kN   "
              f"vs Vp = 4Mp/H = {Vp_hand / 1e3:.0f} kN "
              f"({(V_end - Vp_hand) / Vp_hand * 100:+.1f}%, "
              f"Steel02 hardening)")

        # Precompute the x8-amplified point history on substrate rows.
        scene = build_fem_scene(r.fem)
        ref = np.asarray(scene.grid.points, dtype=np.float64).copy()
        warped = None
        for axis, comp in ((0, "displacement_x"), (1, "displacement_y")):
            sl = r.nodes.get(ids=scene.node_ids, component=comp)
            if warped is None:
                warped = np.repeat(ref[None], len(sl.values), axis=0)
            rows = [scene.node_id_to_idx[int(n)] for n in sl.node_ids]
            warped[:, rows, axis] += DEFORM_SCALE * np.asarray(sl.values)

        plotter = pv.Plotter(off_screen=True, window_size=(960, 540))
        plotter.add_mesh(scene.grid.copy(), color="gray",
                         line_width=3, opacity=0.5)   # undeformed ghost

        director = ResultsDirector(r)
        director._render_callback = plotter.render  # noqa: SLF001
        spec = DiagramSpec(
            kind="contour",
            selector=SlabSelector(component="disp_mag"),
            style=ContourStyle(cmap="turbo", clim=(0.0, DRIFT), fmt="%.0f"),
        )
        contour = ContourDiagram(spec, r)
        contour.attach(PyVistaQtBackend(plotter), r.fem, scene)
        director.registry.add(contour)
        for actor in plotter.renderer.actors.values():
            try:                                  # thicken the frame lines
                prop = actor.GetProperty()
                prop.SetLineWidth(9)
                prop.SetRenderLinesAsTubes(True)
            except AttributeError:
                pass
        plotter.add_text(
            "W14x90 fiber moment frame - pushover to 2.5% drift"
            f"  (x{DEFORM_SCALE:.0f} deformation)",
            position="upper_left", font_size=11,
        )

        # Warp before each diagram/scalar update so contour geometry and
        # colors land on the same step, then frame the amplified sway.
        base_set_step = director.set_step

        def _warp_then_step(i: int) -> None:
            pts = warped[min(int(i), warped.shape[0] - 1)]
            scene.grid.points = pts
            contour.sync_substrate_points(pts, scene)
            base_set_step(i)

        director.set_step = _warp_then_step
        lo, hi = warped.min(axis=(0, 1)), warped.max(axis=(0, 1))
        pad = 0.06 * max(hi[0] - lo[0], hi[1] - lo[1])
        x0, x1 = lo[0] - pad, hi[0] + pad
        y0, y1 = lo[1] - 3.0 * pad, hi[1] + pad    # scalar-bar room below
        plotter.view_xy()
        plotter.enable_parallel_projection()
        cam = plotter.camera
        cam.focal_point = ((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
        cam.position = ((x0 + x1) / 2, (y0 + y1) / 2, 2.0 * (x1 - x0))
        cam.parallel_scale = max((y1 - y0) / 2,
                                 (x1 - x0) / 2 * 540.0 / 960.0)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        export_animation(plotter, director, OUT, fps=FPS, step_stride=1)
        plotter.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_h5 = str(Path(tmp) / "pushover.h5")
        t0 = time.perf_counter()
        Vp_hand = solve(run_h5)
        print(f"solved {N_STEPS} pushover steps in "
              f"{time.perf_counter() - t0:.1f} s")
        render(run_h5, Vp_hand)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
