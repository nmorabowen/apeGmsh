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

The deflection is written as a ``.ladruno`` (ADR 0131 D10), stamped
against the emitted ``demo_model.h5`` (ADR 0122 D3), and opened with
:meth:`Results.from_ladruno`. It holds nodal displacements only.

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
        Directory to write ``demo_model.h5`` / ``demo_results.ladruno``
        into. Defaults to a fresh temp directory kept alive for the
        process.

    Returns
    -------
    Results
        :meth:`Results.from_ladruno` of ``demo_results.ladruno``, bound to
        ``demo_model.h5`` (``model_h5=``); ``.show_web()`` /
        ``.viewer()`` render it immediately. Turn deformation on for the
        boot geometry (Geometry settings → Deformation, or
        ``geometries.set_deformation(geom_id, enabled=True,
        field="displacement", scale=1.0)``) to see the bend animate
        across steps; add a reference ghost for an undeformed overlay.
    """
    if n_elements < 1:
        raise ValueError("n_elements must be >= 1")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    # Lazy imports — keep this module importable without dragging the
    # mesh / bridge / writer stack in at ``apeGmsh.results`` import time,
    # and avoid an import cycle through the apeGmsh package facade.
    from apeGmsh import apeGmsh
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint
    from apeGmsh.results import Results
    from apeGmsh.results.readers._ladruno_writer import write_nodal_history

    if path is None:
        out_dir = Path(tempfile.mkdtemp(prefix="apegmsh_demo_"))
        _DEMO_DIRS.append(str(out_dir))
    else:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── Mesh: a vertical cantilever column, one PG "Cols" ──────────────
    lc = length / n_elements
    # library-internal: the demo writes its own files under ``path``;
    # no ADR 0112 D1 artifacts beside the caller
    g = apeGmsh(model_name="apeGmsh_demo", verbose=False, _artifacts=False)
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

    # ── Synthetic pushover displacement: analytic cantilever shape ─────
    # u_x(z) = (z^2)(3L - z) / (2 L^3) is the unit tip-load deflection
    # shape (1.0 at the tip), ramped 0 .. tip_drift over the steps.
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    z = coords[:, 2]
    shape = (z ** 2) * (3.0 * length - z) / (2.0 * length ** 3)
    amp = np.linspace(0.0, float(tip_drift), int(n_steps))
    ux = np.outer(amp, shape)                       # (n_steps, n_nodes)
    zeros = np.zeros_like(ux)

    # One static stage of nodal displacements, stamped against the bridge
    # archive (ADR 0122 D3). Nodal rows only, and no element bucket (ADR
    # 0131 D6: the writer refuses ``elements`` in PROVENANCE@unsupported);
    # the elements bind from ``demo_model.h5``. No engine solved the
    # values, so the stamp says ``opensees_backend="unknown"``.
    results_path = out_dir / "demo_results.ladruno"
    write_nodal_history(
        results_path, model_h5=model_path, node_ids=node_ids,
        coords=coords, ndm=3, kind="static", time=amp,
        results={"DISPLACEMENT": np.stack([ux, zeros, zeros], axis=-1)},
        source="demo", opensees_backend="unknown",
    )
    # ``model_h5=`` binds the archive's FEMData (physical groups, labels)
    # and model; the subprocess viewer forwards it as ``--model-h5``.
    return Results.from_ladruno(results_path, model_h5=model_path)


__all__ = ["make_demo_results"]
