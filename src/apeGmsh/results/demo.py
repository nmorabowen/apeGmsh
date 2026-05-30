"""Zero-setup demo results — a ready-to-view :class:`Results` in one call.

``make_demo_results()`` (also :meth:`Results.demo`) builds a small
cantilever-column model and a synthetic pushover so the viewer
(:meth:`Results.show_web`, :meth:`Results.viewer`) and the docs/tests
have *something* to render without the caller supplying an ``.mpco`` +
``model.h5`` pair.

It is a **real** model, not a stub: the column mesh is built through the
normal ``apeGmsh`` geometry/mesh pipeline and a genuine OpenSees model is
emitted via ``apeSees(fem).h5(...)`` — satisfying the Phase-8 ``model=``
contract honestly (no fake/empty model surface). The *displacements* are
synthetic, though: an analytic tip-loaded cantilever shape ramped over
``n_steps`` so scrubbing the step slider visibly bends the column. No
OpenSees analysis is run — the model is only emitted, and the deflection
is written directly — so this stays fast, deterministic, and free of an
``openseespy`` solve.

The results are written to a temporary directory that persists for the
process (so the returned :class:`Results` keeps a valid backing file);
pass ``path=`` to write somewhere stable instead.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .Results import Results

# Hold temp-dir references for the process lifetime so the backing files
# behind demo Results objects are not swept while still open.
_DEMO_DIRS: list[str] = []


def make_demo_results(
    *,
    length: float = 10.0,
    n_elements: int = 8,
    n_steps: int = 6,
    tip_drift: float = 2.0,
    path: "Optional[str | Path]" = None,
) -> "Results":
    """Build a ready-to-view demo :class:`Results` (a cantilever pushover).

    Parameters
    ----------
    length
        Column height (world units). The column runs up the ``+z`` axis.
    n_elements
        Number of line elements along the column (more = smoother bend).
    n_steps
        Number of pushover steps; the step slider in
        :meth:`Results.show_web` spans ``0 .. n_steps - 1``.
    tip_drift
        Final lateral tip displacement (along ``+x``) at the last step.
    path
        Directory to write ``demo_model.h5`` / ``demo_results.h5`` into.
        Defaults to a fresh temp directory kept alive for the process.

    Returns
    -------
    Results
        Backed by a native composed file; ``.show_web()`` /
        ``.viewer()`` render it immediately. Add a deformed-shape layer
        to see the bend animate across steps::

            from apeGmsh.viewers.diagrams import (
                DiagramSpec, DeformedShapeDiagram, DeformedShapeStyle,
                SlabSelector,
            )
            wv = make_demo_results().show_web(show=False)
            spec = DiagramSpec(
                kind="deformed_shape",
                selector=SlabSelector(component="displacement_x"),
                style=DeformedShapeStyle(scale=1.0),
            )
            wv.director.registry.add(DeformedShapeDiagram(spec, wv.director.results))
            wv.show()
    """
    if n_elements < 1:
        raise ValueError("n_elements must be >= 1")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    # Lazy imports — keep this module importable without dragging the
    # mesh / bridge / writer stack in at ``apeGmsh.results`` import time,
    # and avoid an import cycle through the apeGmsh package facade.
    from apeGmsh import apeGmsh
    from apeGmsh.opensees import OpenSeesModel, apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    if path is None:
        out_dir = Path(tempfile.mkdtemp(prefix="apegmsh_demo_"))
        _DEMO_DIRS.append(str(out_dir))
    else:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Mesh: a vertical cantilever column, one PG "Cols" ──────────────
    lc = length / n_elements
    g = apeGmsh(model_name="apeGmsh_demo", verbose=False)
    g.begin()
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0, lc=lc)
    p1 = g.model.geometry.add_point(0.0, 0.0, length, lc=lc)
    line = g.model.geometry.add_line(p0, p1)
    g.physical.add(1, [line], name="Cols")
    g.mesh.generation.set_order(1)
    g.mesh.generation.generate(1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    # ── Real OpenSees model emitted to model.h5 (no analysis run) ──────
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)

    model_path = out_dir / "demo_model.h5"
    ops.h5(str(model_path))
    model = OpenSeesModel.from_h5(str(model_path))

    # ── Synthetic pushover displacement: analytic cantilever shape ─────
    # u_x(z) = (z^2)(3L - z) / (2 L^3) is the unit tip-load deflection
    # shape (1.0 at the tip), ramped 0 .. tip_drift over the steps.
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    z = np.asarray(fem.nodes.coords, dtype=np.float64)[:, 2]
    shape = (z ** 2) * (3.0 * length - z) / (2.0 * length ** 3)
    amp = np.linspace(0.0, float(tip_drift), int(n_steps))
    ux = np.outer(amp, shape)                       # (n_steps, n_nodes)
    zeros = np.zeros_like(ux)

    results_path = out_dir / "demo_results.h5"
    with NativeWriter(results_path) as w:
        w.open(fem=fem, model_h5_src=model_path)
        sid = w.begin_stage(
            name="pushover", kind="static", time=amp.astype(np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={
                "displacement_x": ux,
                "displacement_y": zeros,
                "displacement_z": zeros,
            },
        )
        w.end_stage()

    # Pass ``model_path`` so the non-blocking subprocess viewer
    # (``Results.viewer(blocking=False)``) forwards ``--model-h5
    # demo_model.h5`` to the child — the composed ``demo_results.h5`` does
    # not carry an independently-readable ``/model`` zone, so the child
    # must re-read the model from the sibling archive, not from the
    # results file.
    return Results.from_native(results_path, model=model, model_path=model_path)


__all__ = ["make_demo_results"]
