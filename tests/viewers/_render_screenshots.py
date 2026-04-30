"""Off-screen renderer — produces visual smoke PNGs for each fixture.

Sister script to ``manual_check.py``: where that one opens a real Qt
window for human eyes, this one drives the same diagram catalogue
through a headless ``pyvista.Plotter(off_screen=True)`` and writes PNGs
to ``tests/viewers/_screenshots/``. The whole thing runs without Qt,
so it works in CI and on a headless dev machine.

Each fixture renders whatever subset of the catalogue its data carries:

* ``substrate``    — gray mesh, no field data (sanity for the ``(dim,
  npe)`` fallback).
* ``contour``      — nodal displacement scalar.
* ``deformed``     — warped mesh + undeformed reference.
* ``line_force``   — beam moment / shear / axial (skipped when the
  fixture has no line-station data).
* ``spring_force`` — zero-length spring arrows (skipped when the
  fixture has no spring data).

Usage::

    PYTHONPATH=src python tests/viewers/_render_screenshots.py

The exit code is 0 if every requested image was produced; non-zero if
any renderer raised. Missing component data is *not* an error — a
fixture without line-station results just skips that image.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import pyvista as pv

from apeGmsh.results import Results
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DeformedShapeDiagram,
    DeformedShapeStyle,
    DiagramSpec,
    LineForceDiagram,
    LineForceStyle,
    SlabSelector,
    SpringForceDiagram,
    SpringForceStyle,
)
from apeGmsh.viewers.scene.fem_scene import FEMSceneData, build_fem_scene


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "results"
_OUT_DIR = _REPO_ROOT / "tests" / "viewers" / "_screenshots"

_FIXTURES = (
    ("frame", _FIXTURE_DIR / "elasticFrame.mpco"),
    ("springs", _FIXTURE_DIR / "zl_springs.mpco"),
)

_WINDOW = (1280, 800)


# --------------------------------------------------------------------- #
# Plotter helpers
# --------------------------------------------------------------------- #

def _make_plotter() -> pv.Plotter:
    """Off-screen plotter with a fixed canvas size."""
    return pv.Plotter(off_screen=True, window_size=list(_WINDOW))


def _save(plotter: pv.Plotter, out_path: Path) -> None:
    """Render and save, then close. Camera reset happens at the call site."""
    plotter.show(screenshot=str(out_path), auto_close=False)
    plotter.close()


def _add_substrate(plotter: pv.Plotter, scene: FEMSceneData) -> Any:
    """Add the substrate mesh styled as the gray skeleton everyone draws on."""
    return plotter.add_mesh(
        scene.grid,
        color="#B0B0B0",
        line_width=2.0,
        show_edges=True,
        edge_color="#404040",
        opacity=0.95,
        name="substrate",
    )


# --------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------- #

def render_substrate(scene: FEMSceneData, out_path: Path) -> None:
    """Bare substrate — proves the (dim, npe) fallback rendered cells."""
    plotter = _make_plotter()
    _add_substrate(plotter, scene)
    plotter.add_axes()
    plotter.view_isometric()
    plotter.reset_camera()
    _save(plotter, out_path)


def render_contour(
    results: "Results",
    fem: Any,
    scene: FEMSceneData,
    component: str,
    step: int,
    out_path: Path,
) -> None:
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component=component),
        style=ContourStyle(cmap="viridis"),
    )
    diagram = ContourDiagram(spec, results)

    plotter = _make_plotter()
    _add_substrate(plotter, scene)
    diagram.attach(plotter, fem, scene)
    diagram.update_to_step(step)
    plotter.add_text(
        f"contour: {component} @ step {step}",
        position="upper_left", font_size=10,
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.reset_camera()
    try:
        _save(plotter, out_path)
    finally:
        diagram.detach()


def render_deformed(
    results: "Results",
    fem: Any,
    scene: FEMSceneData,
    components: tuple[str, ...],
    scale: float,
    step: int,
    out_path: Path,
) -> None:
    spec = DiagramSpec(
        kind="deformed_shape",
        selector=SlabSelector(component=components[0]),
        style=DeformedShapeStyle(components=components, scale=scale),
    )
    diagram = DeformedShapeDiagram(spec, results)

    plotter = _make_plotter()
    _add_substrate(plotter, scene)
    diagram.attach(plotter, fem, scene)
    diagram.update_to_step(step)
    plotter.add_text(
        f"deformed (×{scale:g}) @ step {step}",
        position="upper_left", font_size=10,
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.reset_camera()
    try:
        _save(plotter, out_path)
    finally:
        diagram.detach()


def render_line_force(
    results: "Results",
    fem: Any,
    scene: FEMSceneData,
    component: str,
    step: int,
    out_path: Path,
) -> None:
    spec = DiagramSpec(
        kind="line_force",
        selector=SlabSelector(component=component),
        style=LineForceStyle(),
    )
    diagram = LineForceDiagram(spec, results)

    plotter = _make_plotter()
    _add_substrate(plotter, scene)
    diagram.attach(plotter, fem, scene)
    diagram.update_to_step(step)
    plotter.add_text(
        f"line force: {component} @ step {step}",
        position="upper_left", font_size=10,
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.reset_camera()
    try:
        _save(plotter, out_path)
    finally:
        diagram.detach()


def render_spring_force(
    results: "Results",
    fem: Any,
    scene: FEMSceneData,
    component: str,
    step: int,
    out_path: Path,
) -> None:
    spec = DiagramSpec(
        kind="spring_force",
        selector=SlabSelector(component=component),
        style=SpringForceStyle(scale=None),    # auto-scale via fraction
    )
    diagram = SpringForceDiagram(spec, results)

    plotter = _make_plotter()
    _add_substrate(plotter, scene)
    diagram.attach(plotter, fem, scene)
    diagram.update_to_step(step)
    plotter.add_text(
        f"spring force: {component} @ step {step}",
        position="upper_left", font_size=10,
    )
    plotter.add_axes()
    plotter.view_isometric()
    plotter.reset_camera()
    try:
        _save(plotter, out_path)
    finally:
        diagram.detach()


# --------------------------------------------------------------------- #
# Per-fixture driver
# --------------------------------------------------------------------- #

def _scope_to_last_stage(results: "Results") -> tuple["Results", int]:
    """Pick the last stage and an end-of-stage step (deformation visible)."""
    stage = results.stages[-1]
    scoped = results.stage(stage.name)
    last_step = max(0, stage.n_steps - 1)
    return scoped, last_step


def _safe_render(label: str, fn, *args, **kwargs) -> tuple[str, Optional[str]]:
    """Run a renderer; return (status, error message)."""
    try:
        fn(*args, **kwargs)
        return ("ok", None)
    except Exception:
        return ("FAIL", traceback.format_exc())


def render_fixture(name: str, path: Path) -> list[tuple[str, str, Optional[str]]]:
    """Render every applicable image for one fixture.

    Returns a list of ``(label, status, error_or_none)`` per image
    attempted (skipped renderers don't appear).
    """
    print(f"\n=== {name}: {path.name} ===")
    if not path.exists():
        print(f"  fixture missing — skipping")
        return [(f"{name}_substrate", "SKIP", "fixture missing")]

    results = Results.from_mpco(path)
    fem = results.fem
    scene = build_fem_scene(fem)
    print(
        f"  scene: nodes={scene.grid.n_points} cells={scene.grid.n_cells}"
        f" skipped_types={scene.skipped_types}"
    )
    if scene.grid.n_cells == 0:
        # Bug we're fixing — bail loudly so it can never silently regress.
        return [(f"{name}_substrate", "FAIL",
                 "build_fem_scene produced 0 cells")]

    scoped, step = _scope_to_last_stage(results)

    out_paths = {
        "substrate":   _OUT_DIR / f"{name}_substrate.png",
        "contour":     _OUT_DIR / f"{name}_contour.png",
        "deformed":    _OUT_DIR / f"{name}_deformed.png",
        "line_force":  _OUT_DIR / f"{name}_line_force.png",
        "spring":      _OUT_DIR / f"{name}_spring_force.png",
    }
    log: list[tuple[str, str, Optional[str]]] = []

    # 1. Substrate (always).
    log.append(("substrate",) + _safe_render(
        "substrate", render_substrate, scene, out_paths["substrate"],
    ))

    # 2. Contour on displacement_z (every fixture has nodal disp).
    node_components = set(scoped.nodes.available_components())
    if "displacement_z" in node_components:
        log.append(("contour",) + _safe_render(
            "contour", render_contour,
            scoped, fem, scene,
            "displacement_z", step, out_paths["contour"],
        ))
    else:
        log.append(("contour", "SKIP", "no displacement_z in nodes"))

    # 3. Deformed on (displacement_x, _y, _z) where available.
    disp_axes = [c for c in ("displacement_x", "displacement_y",
                              "displacement_z") if c in node_components]
    if disp_axes:
        components = tuple(disp_axes)
        # Pick a scale that makes the warp visible without self-clipping —
        # 5% of the model diagonal divided by the largest displacement.
        scale = _auto_deformed_scale(scoped, fem, scene, components, step)
        log.append(("deformed",) + _safe_render(
            "deformed", render_deformed,
            scoped, fem, scene,
            components, scale, step, out_paths["deformed"],
        ))
    else:
        log.append(("deformed", "SKIP", "no displacement components"))

    # 4. Line force — only when the fixture carries line-station data.
    line_components = set(scoped.elements.line_stations.available_components())
    moment_components = [c for c in line_components
                         if c.startswith(("bending_moment", "moment"))]
    if moment_components:
        log.append(("line_force",) + _safe_render(
            "line_force", render_line_force,
            scoped, fem, scene,
            moment_components[0], step, out_paths["line_force"],
        ))
    else:
        log.append(("line_force", "SKIP", "no line-station components"))

    # 5. Spring force — only when the fixture has zero-length spring data.
    spring_components = set(scoped.elements.springs.available_components())
    if "spring_force_0" in spring_components:
        log.append(("spring",) + _safe_render(
            "spring", render_spring_force,
            scoped, fem, scene,
            "spring_force_0", step, out_paths["spring"],
        ))
    else:
        log.append(("spring", "SKIP", "no spring components"))

    return log


def _auto_deformed_scale(
    results: "Results", fem: Any, scene: FEMSceneData,
    components: tuple[str, ...], step: int,
) -> float:
    """Pick a scale factor that makes the deformed shape visible.

    Targets a peak-displacement amplification of ~5% of the model
    diagonal. Falls back to 1.0 when the displacement read fails.
    """
    try:
        peak = 0.0
        for c in components:
            slab = results.nodes.get(component=c, time=[step])
            if slab.values.size == 0:
                continue
            peak = max(peak, float(abs(slab.values).max()))
        if peak <= 0.0:
            return 1.0
        return float(0.05 * scene.model_diagonal / peak)
    except Exception:
        return 1.0


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #

def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {_OUT_DIR}")

    all_logs: list[tuple[str, str, str, Optional[str]]] = []
    for fname, fpath in _FIXTURES:
        for label, status, err in render_fixture(fname, fpath):
            all_logs.append((fname, label, status, err))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    n_ok = n_skip = n_fail = 0
    for fname, label, status, err in all_logs:
        if status == "ok":
            n_ok += 1
            print(f"  ok    {fname:8} {label}")
        elif status == "SKIP":
            n_skip += 1
            print(f"  skip  {fname:8} {label}  ({err})")
        else:
            n_fail += 1
            print(f"  FAIL  {fname:8} {label}")
            if err:
                # Indent the traceback for readability.
                for line in err.rstrip().splitlines():
                    print(f"           {line}")
    print(f"\nok={n_ok} skip={n_skip} fail={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
