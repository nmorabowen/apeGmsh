"""Live emit→run→read round-trip for ``Results.from_ladruno`` (recorder L2b-2).

Codifies the L2a manual verification on the *public* factory: a real
OpenSees run writes a ``.ladruno`` via ``recorder ladruno``, then
``Results.from_ladruno`` reads it back and the element value channels are
asserted against the live ``ops`` responses to machine precision.

Requires the **Ladruno fork build** of openseespy (stock builds don't
know the ``ladruno`` recorder). The module skips cleanly when the running
build doesn't produce a ``.ladruno``. Run it with the worktree on
``PYTHONPATH`` so the import resolves to this branch, not the editable
main install::

    PYTHONPATH=src C:/Users/nmora/venv/opensees_venv/Scripts/python.exe \
        -m pytest tests/results/test_ladruno_live_roundtrip.py -m live
"""
from __future__ import annotations

import numpy as np
import pytest

ops = pytest.importorskip("openseespy.opensees")
pytestmark = pytest.mark.live


def _ladruno_supported(tmp_path) -> bool:
    """Probe whether this build's recorder writes a ``.ladruno``."""
    path = str(tmp_path / "probe.ladruno")
    try:
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        ops.node(1, 0.0, 0.0)
        ops.node(2, 1.0, 0.0)
        ops.fix(1, 1, 1)
        ops.fix(2, 0, 1)
        ops.uniaxialMaterial("Elastic", 1, 1000.0)
        ops.element("truss", 1, 1, 2, 1.0, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 10.0, 0.0)
        ops.recorder("ladruno", path, "-N", "displacement", "-E", "basicForce")
        ops.system("BandSPD")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        ops.analyze(1)
        ops.wipe()
    except Exception:
        return False
    import os
    return os.path.exists(path)


@pytest.fixture(scope="module")
def _require_fork(tmp_path_factory):
    if not _ladruno_supported(tmp_path_factory.mktemp("probe")):
        pytest.skip("running build has no Ladruno '.ladruno' recorder")


def test_truss_roundtrip_nodes_and_line(_require_fork, tmp_path):
    from apeGmsh.results import Results

    path = str(tmp_path / "truss.ladruno")
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for i, x in enumerate([0.0, 1.0, 2.0], start=1):
        ops.node(i, x, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.fix(3, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, 1000.0)
    ops.element("truss", 1, 1, 2, 1.0, 1)
    ops.element("truss", 2, 2, 3, 1.0, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 7.0, 0.0)
    ops.recorder("ladruno", path, "-N", "displacement", "-E", "basicForce")
    ops.system("BandSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    live_disp = ops.nodeDisp(3, 1)
    live_axial = ops.basicForce(1)[0]
    ops.wipe()

    r = Results.from_ladruno(path)
    nslab = r.nodes.get(component="displacement_x")
    i3 = nslab.node_ids.tolist().index(3)
    np.testing.assert_allclose(nslab.values[-1, i3], live_disp, atol=1e-12)

    lslab = r.elements.line_stations.get(component="axial_force")
    # element 1, last step — matches ops.basicForce to machine precision.
    e1 = np.where(lslab.element_index == 1)[0]
    np.testing.assert_allclose(lslab.values[-1, e1[0]], live_axial, atol=1e-12)


def test_node_envelope_roundtrip(_require_fork, tmp_path):
    from apeGmsh.results import Results

    path = str(tmp_path / "env.ladruno")
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for i, x in enumerate([0.0, 1.0, 2.0], start=1):
        ops.node(i, x, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.fix(3, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, 1000.0)
    ops.element("truss", 1, 1, 2, 1.0, 1)
    ops.element("truss", 2, 2, 3, 1.0, 1)
    ops.recorder(
        "ladruno", path, "-N", "displacement", "-T", "dt", 1e-6, "-envelope",
    )
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 10.0, 0.0)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGen")
    ops.test("NormDispIncr", 1e-10, 10)
    ops.algorithm("Newton")
    ops.analysis("Static")
    path_u3 = []
    for di in (0.25, 0.25, 0.25, 0.25,
               -0.5, -0.5, -0.5, -0.5, -0.5,
               0.4, 0.4, 0.4, 0.4, 0.4):
        ops.integrator("LoadControl", di)
        ops.analyze(1)
        path_u3.append(ops.nodeDisp(3, 1))
    ops.remove("recorders")
    ops.wipe()

    u3 = np.asarray(path_u3)
    df = Results.from_ladruno(path).node_envelope("displacement_x")
    row = df.loc[3]
    # The file's reduced extremes match the python-tracked path extremes.
    np.testing.assert_allclose(row["min"], u3.min(), atol=1e-12)
    np.testing.assert_allclose(row["max"], u3.max(), atol=1e-12)
    np.testing.assert_allclose(row["absmax"], np.abs(u3).max(), atol=1e-12)
    # arg_step is the recorder's session commitTag (not a fresh 0-based index
    # — ops.wipe() doesn't reset it), so its exact value is a recorder
    # convention covered by the fixture test; here just assert it's a valid
    # non-negative step index.
    assert int(row["arg_step"]) >= 0


def test_quad_roundtrip_gauss_stress(_require_fork, tmp_path):
    from apeGmsh.results import Results

    path = str(tmp_path / "quad.ladruno")
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.node(3, 1.0, 1.0)
    ops.node(4, 0.0, 1.0)
    ops.fix(1, 1, 1)
    ops.fix(2, 0, 1)
    ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    ops.element("quad", 1, 1, 2, 3, 4, 1.0, "PlaneStress", 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 5.0, 0.0)
    ops.load(4, 5.0, 0.0)
    ops.recorder("ladruno", path, "-E", "stress")
    ops.system("BandGen")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    # ops.eleResponse stress: 4 GP × 3 comps (σxx, σyy, σxy), GP-major.
    live = np.asarray(ops.eleResponse(1, "stress"), dtype=float).reshape(4, 3)
    ops.wipe()

    r = Results.from_ladruno(path)
    slab = r.elements.gauss.get(component="stress_xx")
    # GaussSlab rows are GP-major for the single element → σxx per GP.
    np.testing.assert_allclose(slab.values[-1], live[:, 0], atol=1e-10)


def test_fiber_beam_roundtrip_section_and_fibers(_require_fork, tmp_path):
    from apeGmsh.results import Results

    path = str(tmp_path / "fiberbeam.ladruno")
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.uniaxialMaterial("Elastic", 1, 200000.0)
    ops.section("Fiber", 1)
    ops.patch("rect", 1, 2, 2, -0.05, -0.05, 0.05, 0.05)
    ops.geomTransf("Linear", 1)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 3.0, 2.0, 0.0)
    ops.recorder(
        "ladruno", path, "-E", "section.force", "section.fiber.stress",
    )
    ops.system("BandGen")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)
    # Live per-station section axial force P (dof 1) over the 3 Lobatto GPs.
    live_P = [ops.sectionForce(1, gp, 1) for gp in (1, 2, 3)]
    ops.wipe()

    r = Results.from_ladruno(path)

    # section.force → axial_force line stations match ops.sectionForce.
    axial = r.elements.line_stations.get(component="axial_force")
    e1 = np.where(axial.element_index == 1)[0]
    np.testing.assert_allclose(
        axial.values[-1, e1], live_P, atol=1e-10,
    )

    # Fiber stresses integrate (Σ σ·A) back to the section axial force at
    # each Gauss point — ties the fiber read to an independent live value.
    fib = r.elements.fibers.get(component="fiber_stress")
    for gp_i, P in enumerate(live_P):
        cols = np.where(fib.gp_index == gp_i)[0]
        axial_from_fibers = float(
            np.sum(fib.values[-1, cols] * fib.area[cols])
        )
        np.testing.assert_allclose(axial_from_fibers, P, atol=1e-8)
