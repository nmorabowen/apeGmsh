"""Docs showcase stills — rendered PNGs for pages that ship bare code.

Builds and solves four models straight from their docs pages, then renders
one off-screen contour still each (``build_fem_scene`` substrate + a
``ContourDiagram`` attached through ``PyVistaQtBackend``, driven by a
``ResultsDirector`` parked on the last step):

* ``first-model-contour.png``      <- docs/tutorials/first-model.md (T1 cantilever)
* ``save-reload-deflection.png``   <- docs/tutorials/save-reload-view.md (T3 SS beam, reloaded)
* ``results-strategies-portal.png``<- docs/examples/results-strategies.md (E1 portal)
* ``staged-ssi-settlement.png``    <- docs/examples/staged-gravity-ssi.md (soil column, gravity)

The staged-SSI page runs its deck as a two-stage subprocess; for the still we
only need the *gravity* state, so the soil box + absorbing skin (which starts
in its "hold" stage and pins the boundary) is solved live with the same
pattern-controlled gravity case — physically the docs' stage 3a.

Run:  python scripts/render_showcase/stills.py
"""
from __future__ import annotations

import os

os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from apeGmsh import apeGmsh, FEMData, Results
from apeGmsh.opensees import apeSees, OpenSeesModel
from apeGmsh.results.capture.spec import DomainCaptureSpec

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "assets" / "tut"
WINDOW = (1280, 720)


# =====================================================================
# Shared renderer — substrate + contour diagram -> screenshot
# =====================================================================

def render_still(
    results: Results,
    *,
    component: str,
    out_path: Path,
    title: str,
    camera: str = "xy",
    deform_fraction: float = 0.12,
    line_tubes: bool = False,
    show_edges: bool = False,
) -> float:
    """Render one deformed contour still of ``results`` to ``out_path``.

    The selector-less contour covers every node, so the diagram's own
    submesh IS the whole mesh — no grey substrate actor is added (it
    would z-fight with / occlude the coincident contour surface).

    Returns the deformation amplification factor used.
    """
    from apeGmsh.viewers.backends import PyVistaQtBackend
    from apeGmsh.viewers.diagrams import (
        ContourDiagram, ContourStyle, DiagramSpec, ResultsDirector,
        SlabSelector,
    )
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = WINDOW

    scene = build_fem_scene(results.fem)

    director = ResultsDirector(results)
    director._render_callback = plotter.render  # noqa: SLF001
    last = max(0, director.n_steps - 1)

    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component=component),
        style=ContourStyle(cmap="turbo", show_edges=show_edges),
    )
    diagram = ContourDiagram(spec, results)
    diagram.attach(PyVistaQtBackend(plotter), results.fem, scene)
    director.registry.add(diagram)
    director.set_step(last)
    diagram.update_to_step(last)          # director may already be parked at last

    # -- Warp the substrate (and the contour submesh) by the last-step
    #    displacement field, scaled to a visible fraction of the model.
    ids = np.asarray(results.fem.nodes.ids, dtype=np.int64)
    ref = np.asarray(scene.grid.points, dtype=np.float64).copy()
    disp = np.zeros_like(ref)
    for axis, comp in enumerate(
        ("displacement_x", "displacement_y", "displacement_z"),
    ):
        try:
            slab = results.nodes.get(ids=ids, component=comp, time=[last])
        except Exception:
            continue
        if slab.values.size == 0:
            continue
        vals = np.asarray(slab.values[0], dtype=np.float64)
        for nid, v in zip(np.asarray(slab.node_ids), vals):
            idx = scene.node_id_to_idx.get(int(nid))
            if idx is not None:
                disp[idx, axis] = v
    max_d = float(np.abs(disp).max())
    scale = 1.0
    if max_d > 0.0:
        scale = deform_fraction * float(scene.model_diagonal) / max_d
        deformed = ref + scale * disp
        scene.grid.points = deformed
        diagram.sync_substrate_points(deformed, scene)

    if line_tubes:                        # thicken 1-D (beam) substrates
        for actor in plotter.renderer.actors.values():
            try:
                prop = actor.GetProperty()
                prop.SetLineWidth(7)
                prop.SetRenderLinesAsTubes(True)
            except AttributeError:
                pass

    plotter.add_text(title, position="upper_edge", font_size=13)
    if camera == "xy":
        plotter.view_xy()
        plotter.camera.zoom(1.05)
    else:
        plotter.view_isometric()
        plotter.camera.zoom(0.9)          # headroom for the title bar

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_path))
    plotter.close()
    return scale


def _wipe() -> None:
    import openseespy.opensees as opspy
    opspy.wipe()


def _capture_all_displacements(ops: apeSees, path: str, *, steps: int = 1,
                               stage: str = "static") -> None:
    """One static run recording every node's displacement -> ``path``."""
    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(components=["displacement"])          # no selector -> all nodes
    with ops.domain_capture(spec, path=path) as cap:
        cap.begin_stage(stage, kind="static")
        ops.analyze(steps=steps)
        cap.step(t=1.0)
        cap.end_stage()


# =====================================================================
# 1. T1 — cantilever (docs/tutorials/first-model.md)
# =====================================================================

def still_first_model(work: Path) -> Path:
    L, E = 3.0, 200e9
    b, h = 0.10, 0.20
    A, Iz = b * h, b * h**3 / 12.0
    P = 10_000.0

    with apeGmsh(model_name="cantilever") as g:
        p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
        p1 = g.model.geometry.add_point(L, 0.0, 0.0)
        beam = g.model.geometry.add_line(p0, p1)
        g.model.sync()
        g.physical.add(1, [beam], name="Beam")
        g.physical.add(0, [p0], name="Fixed")
        g.physical.add(0, [p1], name="Tip")
        g.mesh.sizing.set_global_size(L / 10.0)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(pg="Beam", transf=transf, A=A, E=E, Iz=Iz)
    ops.fix(pg="Fixed", dofs=(1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.load(pg="Tip", forces=(0.0, -P, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    run = str(work / "t1_run.h5")
    _capture_all_displacements(ops, run, stage="tip_load")
    _wipe()

    out = OUT_DIR / "first-model-contour.png"
    with Results.from_native(
        run, model=OpenSeesModel.from_h5(run, fem_root="/model"),
    ) as results:
        tip = abs(float(results.nodes.get(
            pg="Tip", component="displacement_y",
        ).values[-1, 0]))
        render_still(
            results, component="displacement_y", out_path=out,
            title=(f"T1 cantilever - tip deflection {tip * 1e3:.2f} mm "
                   f"(PL3/3EI = 6.75 mm)"),
            camera="xy", line_tubes=True,
        )
    return out


# =====================================================================
# 2. T4 — save / reload the T3 SS beam (docs/tutorials/save-reload-view.md)
# =====================================================================

def still_save_reload(work: Path) -> Path:
    L, E = 4.0, 200e9
    b, h = 0.10, 0.20
    A, Iz = b * h, b * h**3 / 12.0
    P = 20_000.0

    model_h5 = str(work / "ssbeam.h5")
    with apeGmsh(model_name="ssbeam", save_to=model_h5, overwrite=True) as g:
        p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
        pm = g.model.geometry.add_point(L / 2.0, 0.0, 0.0)
        p1 = g.model.geometry.add_point(L, 0.0, 0.0)
        sl = g.model.geometry.add_line(p0, pm)
        sr = g.model.geometry.add_line(pm, p1)
        g.model.sync()
        g.physical.add(1, [sl, sr], name="Beam")
        g.physical.add(0, [p0], name="Pin")
        g.physical.add(0, [p1], name="Roller")
        g.physical.add(0, [pm], name="Mid")
        g.mesh.sizing.set_global_size(L / 16.0)
        g.mesh.generation.generate(1)
        g.mesh.queries.get_fem_data()
        # ssbeam.h5 is written as the block exits.

    fem2 = FEMData.from_h5(model_h5)          # integrity-checked reload

    ops = apeSees(fem2)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(pg="Beam", transf=transf, A=A, E=E, Iz=Iz)
    ops.fix(pg="Pin", dofs=(1, 1, 0))
    ops.fix(pg="Roller", dofs=(0, 1, 0))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.load(pg="Mid", forces=(0.0, -P, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    run = str(work / "ssbeam_run.h5")
    _capture_all_displacements(ops, run, stage="midspan_load")
    _wipe()

    out = OUT_DIR / "save-reload-deflection.png"
    with Results.from_native(
        run, model=OpenSeesModel.from_h5(run, fem_root="/model"),
    ) as results:
        mid = abs(float(results.nodes.get(
            pg="Mid", component="displacement_y",
        ).values[-1, 0]))
        render_still(
            results, component="displacement_y", out_path=out,
            title=(f"T4 reloaded SS beam - midspan deflection "
                   f"{mid * 1e3:.2f} mm (PL3/48EI = 2.00 mm)"),
            camera="xy", line_tubes=True,
        )
    return out


# =====================================================================
# 3. E8 — portal frame (docs/examples/results-strategies.md)
# =====================================================================

def still_portal(work: Path) -> Path:
    H, B, E = 5.0, 5.0, 200e9
    bc, hc = 0.22, 0.22
    Ac, Ic = bc * hc, bc * hc**3 / 12
    bb, hb = 0.20, 0.50
    Ab, Ib = bb * hb, bb * hb**3 / 12
    P, W = 60e3, 300e3

    with apeGmsh(model_name="portal") as g:
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
        g.physical.add(0, [tl], name="RoofL")
        g.physical.add(0, [tr], name="RoofR")
        g.mesh.sizing.set_global_size(H / 6.0)
        g.mesh.generation.generate(1)
        fem = g.mesh.queries.get_fem_data(dim=1)

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(pg="Columns", transf=transf, A=Ac, E=E, Iz=Ic)
    ops.element.elasticBeamColumn(pg="Beam", transf=transf, A=Ab, E=E, Iz=Ib)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        pat.load(pg="RoofL", forces=(P / 2, -W / 2, 0.0))
        pat.load(pg="RoofR", forces=(P / 2, -W / 2, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    run = str(work / "portal_run.h5")
    _capture_all_displacements(ops, run, stage="lateral")
    _wipe()

    out = OUT_DIR / "results-strategies-portal.png"
    with Results.from_native(
        run, model=OpenSeesModel.from_h5(run, fem_root="/model"),
    ) as results:
        dl = results.nodes.get(pg="RoofL", component="displacement_x")
        dr = results.nodes.get(pg="RoofR", component="displacement_x")
        drift = 0.5 * (float(dl.values[-1, 0]) + float(dr.values[-1, 0]))
        render_still(
            results, component="displacement_x", out_path=out,
            title=(f"E8 portal frame - roof drift {drift * 1e3:.4f} mm "
                   f"(same number from every results strategy)"),
            camera="xy", line_tubes=True,
        )
    return out


# =====================================================================
# 4. Staged SSI — soil column after the gravity stage
#    (docs/examples/staged-gravity-ssi.md)
# =====================================================================

def still_staged_ssi(work: Path) -> Path:
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    rho, nu, Vs = 2000.0, 0.3, 200.0
    G = rho * Vs**2
    E = 2.0 * G * (1.0 + nu)
    H, g_acc = 40.0, 9.81

    with apeGmsh(model_name="grav_ssi") as g:
        res = g.parts.add_plane_wave_box(x=(20.0, 2), y=(20.0, 2), z=(H, 16))
        with g.loads.case("dead"):
            g.loads.gravity(res.soil_pg, density=rho, g=(0.0, 0.0, -g_acc))
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data()

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    soil = ops.register(ElasticIsotropic(E=E, nu=nu, rho=rho))
    ops.element.stdBrick(pg=res.soil_pg, material=soil)
    # The absorbing skin boots in its HOLD stage — a penalty support that
    # pins the truncation boundary while self-weight settles the column
    # (the docs page's stage 3a; the flip to absorbing is the dynamic
    # stage, which this still doesn't need).
    ops.element.absorbing_boundary(skin=res, material=soil)

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        pat.from_model("dead")                      # ramped nodal gravity
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=30)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()

    run = str(work / "grav_ssi_run.h5")
    _capture_all_displacements(ops, run, steps=10, stage="gravity")
    _wipe()

    out = OUT_DIR / "staged-ssi-settlement.png"
    with Results.from_native(
        run, model=OpenSeesModel.from_h5(run, fem_root="/model"),
    ) as results:
        surf = results.nodes.get(
            pg=res.free_surface_pg, component="displacement_z",
        )
        settle = float(np.mean(surf.values[-1, :]))
        render_still(
            results, component="displacement_z", out_path=out,
            title=(f"Staged SSI - gravity stage: surface settlement "
                   f"{settle * 100:.1f} cm (skin holding)"),
            camera="iso", deform_fraction=0.06, show_edges=True,
        )
    return out


# =====================================================================
# Driver
# =====================================================================

def main() -> None:
    jobs = [
        ("first-model", still_first_model),
        ("save-reload", still_save_reload),
        ("results-strategies", still_portal),
        ("staged-ssi", still_staged_ssi),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, fn in jobs:
            t0 = time.perf_counter()
            out = fn(work)
            dt = time.perf_counter() - t0
            kb = out.stat().st_size / 1024.0
            print(f"[{name:>18}] {out.name}  {kb:7.1f} KB  {dt:6.1f} s")


if __name__ == "__main__":
    main()
