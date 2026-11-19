"""Live oracle for the mass guard of ``spectrum_after_gravity`` (chain A,
link A1; gap 6).

V1's one-storey rigid diaphragm (``studies/rsa-gaps/v1``): four fixed-base
columns, their tops tied by ``rigidDiaphragm 3`` to a centre node, and the
400 t floor mass on the four column tops (the constrained nodes), under
``constraints Transformation``. Stock OpenSees and Ladruno forks older
than ``ladruno-v20261027`` (WP-211) count that mass as fixed in
``modalProperties``: free mass 0, Γ 0, a zero response with no error. The
driver refuses there. A build with WP-211 counts it, and the driver runs:
the X base shears of the modes then sum to ``M Sa``.

Which of the two this build is, is read from ``modalProperties`` itself on
a plain-openseespy copy of the model, so the test asserts the matching
outcome on stock (the ``live-stock`` lane) and on any fork build.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import NodeGroupRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.analysis.spectrum import DesignSpectrum, NodeDisp

openseespy = pytest.importorskip("openseespy.opensees")

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

A, B, H = 12.0, 8.0, 3.5
E, G = 25.0e9, 25.0e9 / 2.4
SIDE = 0.5
I_COL = SIDE**4 / 12.0
M_TOT = 400.0e3
SA = 5.0
CORNERS = [(-A / 2, -B / 2), (A / 2, -B / 2), (A / 2, B / 2), (-A / 2, B / 2)]
BASE = [1, 2, 3, 4]
TOPS = [5, 6, 7, 8]
MASTER = 9


def _diaphragm(handler: str = "Transformation") -> apeSees:
    coords = ([(x, y, 0.0) for x, y in CORNERS]
              + [(x, y, H) for x, y in CORNERS] + [(0.0, 0.0, H)])
    nodes = _NodesStub(ids=list(range(1, 10)), coords=coords,
                       node_pgs={"Base": BASE, "Tops": TOPS,
                                 "Centre": [MASTER]})
    groups = {"Cols": _ElementGroupView(ids=(1, 2, 3, 4), connectivity=tuple(
        zip(BASE, TOPS)))}
    fem = FEMStub(nodes=nodes, elements=_ElementsStub(elem_pgs=groups))
    fem.add_node_constraints([NodeGroupRecord(
        kind="rigid_diaphragm", master_node=MASTER, slave_nodes=TOPS,
        plane_normal=np.array([0.0, 0.0, 1.0]))])
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    tc = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(pg="Cols", transf=tc, A=SIDE**2, E=E,
                                  G=G, J=2.0 * I_COL, Iy=I_COL, Iz=I_COL)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    # As in V1, the diaphragm's nodes move in plane only. With the column
    # tops free out of plane, an old build still counts their mass: the
    # loss needs every DOF of the node fixed or MP-retained.
    ops.fix(pg="Tops", dofs=(0, 0, 1, 1, 1, 0))
    ops.fix(pg="Centre", dofs=(0, 0, 1, 1, 1, 0))
    m = M_TOT / len(TOPS)
    ops.mass(pg="Tops", values=(m, m, 0.0, 0.0, 0.0, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Tops", forces=(0.0, 0.0, -m * 9.81, 0.0, 0.0, 0.0))
    if handler == "Penalty":
        ops.constraints.Penalty(alpha_sp=1e14, alpha_mp=1e14)
    else:
        ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()
    return ops


def _build_counts_slave_mass() -> bool:
    """Whether this build's ``modalProperties`` counts mass on
    MP-retained DOFs under Transformation (WP-211), on a plain copy of the
    model."""
    from apeGmsh.opensees.emitter.live import get_ops

    o: Any = get_ops()
    o.wipe()
    o.model("basic", "-ndm", 3, "-ndf", 6)
    for i, (x, y) in enumerate(CORNERS):
        o.node(BASE[i], x, y, 0.0)
        o.fix(BASE[i], 1, 1, 1, 1, 1, 1)
        o.node(TOPS[i], x, y, H)
        o.fix(TOPS[i], 0, 0, 1, 1, 1, 0)
        o.mass(TOPS[i], M_TOT / 4, M_TOT / 4, 0.0, 0.0, 0.0, 0.0)
    o.node(MASTER, 0.0, 0.0, H)
    o.fix(MASTER, 0, 0, 1, 1, 1, 0)
    o.rigidDiaphragm(3, MASTER, *TOPS)
    o.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
    for i in range(4):
        o.element("elasticBeamColumn", i + 1, BASE[i], TOPS[i], SIDE**2, E,
                  G, 2.0 * I_COL, I_COL, I_COL, 1)
    o.constraints("Transformation")
    o.numberer("RCM")
    o.system("UmfPack")
    o.eigen("-fullGenLapack", 3)
    free = float(dict(o.modalProperties("-return"))["totalFreeMass"][0])
    o.wipe()
    return free > 0.5 * M_TOT


def _run(handler: str = "Transformation") -> Any:
    spec = DesignSpectrum((0.0, 5.0), (SA, SA), direction=1)
    return _diaphragm(handler).spectrum_after_gravity(
        3, spectra=[spec],
        extractors={"centre": NodeDisp(nodes=[MASTER], dofs=(1, 2, 6))},
        # Transformation leaves 3 DOFs, too few for ARPACK; under Penalty
        # the full solver meets the massless DOFs' infinite eigenvalues.
        solver="-genBandArpack" if handler == "Penalty" else "-fullGenLapack")


@pytest.mark.live
def test_mass_on_constrained_nodes_refused_or_counted() -> None:
    if not _build_counts_slave_mass():
        # Stock OpenSees, or a fork older than WP-211: without the guard
        # the driver ran and returned a zero response.
        with pytest.raises(BridgeError, match="free mass of 0") as err:
            _run()
        assert "Transformation" in str(err.value)
        assert "WP-211" in str(err.value)
        return
    _assert_sway_base_shear(_run())


@pytest.mark.live
def test_mass_on_constrained_nodes_under_penalty_runs() -> None:
    """Penalty numbers the constrained DOFs, so every build counts their
    mass: the guard lets the same model through."""
    _assert_sway_base_shear(_run("Penalty"))


def _assert_sway_base_shear(res: Any) -> None:
    """Each mode's X base shear is ``-M_eff,k Sa`` with the flat spectrum,
    and the effective masses sum to the whole mass (the X and Y sway modes
    are degenerate, so the solver may split it between them)."""
    base = res.modes("base_shear", 0)[:, 0]
    np.testing.assert_allclose(-float(np.sum(base)), M_TOT * SA, rtol=1e-6)
