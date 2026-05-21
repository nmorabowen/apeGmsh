"""``H5Model.element_local_axes_vecxz`` — the transforms ↔ element_meta
↔ vocabulary join that feeds the viewer's local-axis overlay and the
LineForceDiagram fill orientation.

Two layers:

* :func:`test_transf_arg_tail_index_vocabulary` — the pure
  position-resolution helper against the real element registry.
* :func:`test_element_local_axes_vecxz_end_to_end` — a real
  ``apeSees(fem).h5`` build (bridge fan-out → real ``fem_eids``),
  asserting the join recovers ``{fem_eid: vecxz}``.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import _ELEM_REGISTRY
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5_reader import (
    _FORCE_DISP_BEAMS,
    _transf_arg_tail_index,
)
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.opensees.fixtures.fem_stub import make_two_node_beam


def test_transf_arg_tail_index_vocabulary() -> None:
    R = _ELEM_REGISTRY

    # force/disp beam-column: transfTag is the first positional arg
    # after connectivity → args-tail index 0 (not in the registry).
    assert "forceBeamColumn" in _FORCE_DISP_BEAMS
    assert "dispBeamColumn" in _FORCE_DISP_BEAMS
    assert _transf_arg_tail_index("forceBeamColumn", 3, R) == 0
    assert _transf_arg_tail_index("dispBeamColumn", 2, R) == 0

    # elasticBeamColumn: slots_3d drops the leading "nodes" prefix, so
    # the tail index is slots.index("transfTag") - 1.
    #   3d = (nodes, A, E, G, Jx, Iy, Iz, transfTag) → 6
    #   2d = (nodes, A, E, Iz, transfTag)            → 3
    assert _transf_arg_tail_index("elasticBeamColumn", 3, R) == 6
    assert _transf_arg_tail_index("elasticBeamColumn", 2, R) == 3

    # ElasticTimoshenkoBeam:
    #   3d = (nodes, E, G, A, Jx, Iy, Iz, Avy, Avz, transfTag) → 8
    #   2d = (nodes, E, G, A, Iz, Avy, transfTag)              → 5
    assert _transf_arg_tail_index("ElasticTimoshenkoBeam", 3, R) == 8
    assert _transf_arg_tail_index("ElasticTimoshenkoBeam", 2, R) == 5

    # Non-transf element types and unknown tokens → None (skipped).
    assert _transf_arg_tail_index("Truss", 3, R) is None
    assert _transf_arg_tail_index("FourNodeTetrahedron", 3, R) is None
    assert _transf_arg_tail_index("totally_unknown_type", 3, R) is None


def test_element_local_axes_vecxz_end_to_end(tmp_path: Path) -> None:
    """Real bridge build: one vertical force-beam with an explicit
    ``vecxz=(1,0,0)``.  The join must key the vector by the FEM
    element id (= 1 here; see ``test_h5_end_to_end``)."""
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    out = tmp_path / "axes.h5"
    ops.h5(str(out))

    with h5_reader.open(str(out)) as model:
        # Sanity: the bridge fan-out recorded the FEM element id.
        em = model.element_meta_arrays("forceBeamColumn")
        assert list(em["fem_eids"]) == [1]

        vecxz = model.element_local_axes_vecxz()
        assert set(vecxz) == {1}
        np.testing.assert_allclose(vecxz[1], [1.0, 0.0, 0.0])

        # transform_arrays exposes the datasets transforms() omits.
        # Phase 8 / ADR 0019 — typed accessor returns records; the
        # transform-group name on disk is ``{type_token}_{tag}``.
        records = model.transforms()
        assert records, "no /opensees/transforms group emitted"
        first = records[0]
        arr = model.transform_arrays(f"{first.type_token}_{first.tag}")
        np.testing.assert_allclose(
            np.asarray(arr["per_element_vecxz"]).reshape(-1, 3)[0],
            [1.0, 0.0, 0.0],
        )
        assert int(
            np.asarray(arr["per_element_emitted_tag"]).reshape(-1)[0]
        ) >= 1

    # Mesh-only / no-transforms archives → empty dict, never raises.
    fem2 = make_two_node_beam()
    ops2 = apeSees(cast("object", fem2))
    ops2.model(ndm=3, ndf=6)
    out2 = tmp_path / "no_transf.h5"
    ops2.h5(str(out2))
    with h5_reader.open(str(out2)) as model2:
        assert model2.element_local_axes_vecxz() == {}
