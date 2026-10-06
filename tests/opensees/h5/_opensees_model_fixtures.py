"""Fixture builders for the ``OpenSeesModel`` test suite (Phase 3 / ADR 0019).

Each builder returns a ``model.h5`` file on disk plus the
:class:`FEMData` it was written from.  The Phase-3 tests need a real
:class:`FEMData` (not a ``FEMStub``) so the broker neutral zone is
written — :meth:`OpenSeesModel.from_h5` reloads via
:meth:`FEMData.from_h5` and that fails on a stub-only file.

Builders here are functions (not pytest fixtures) so each test can
write to its own ``tmp_path`` without forcing a session-scope cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.opensees.emitter.h5_reader import META_NDM_IS_SPATIAL_FROM

#: The last neutral minor before #1291's ``/meta/ndm`` fix: one below the
#: salvage shim's own ``META_NDM_IS_SPATIAL_FROM``. The ndm salvage tests
#: restamp a current file with it; ``NEUTRAL_PRIOR_MINOR`` cannot serve,
#: because it became 2.34.0 itself at the 2.35.0 bump (#1338) and moves
#: again at every later one.
PRE_NDM_FIX_STAMP = "{}.{}.0".format(
    META_NDM_IS_SPATIAL_FROM[0], META_NDM_IS_SPATIAL_FROM[1] - 1,
)
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


__all__ = [
    "build_simple_frame_fem",
    "build_simple_frame_h5",
    "build_frame_with_orientation_fan_out_h5",
]


def build_simple_frame_fem(*, ndm: int = 3) -> FEMData:
    """One-column FEMData with the ``"Cols"`` element PG populated.

    The column runs along z, or along y with ``ndm=2`` so a 2-D model
    lies in the z = 0 plane (#1337).
    """
    if ndm not in (2, 3):
        raise ValueError(f"fixture ndm must be 2 or 3, got {ndm!r}")
    top = [0.0, 1.0, 0.0] if ndm == 2 else [0.0, 0.0, 1.0]
    node_ids = np.array([1, 2], dtype=np.int64)
    node_coords = np.array(
        [[0.0, 0.0, 0.0], top], dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=1,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([1], dtype=np.int64),
        connectivity=np.array([[1, 2]], dtype=np.int64),
    )
    pg = {(1, 100): {
        "name": "Cols",
        "node_ids": node_ids,
        "node_coords": node_coords,
        "element_ids": np.array([1], dtype=np.int64),
    }}
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=2, n_elems=1, bandwidth=1, types=[line_info],
    )
    # Named like a session snapshot (#1307): a bridge on it writes its
    # model.h5 to the test's artifact directory instead of warning.
    return FEMData(nodes=nodes, elements=elements, info=info,
                   model_name="simple_frame")


def build_simple_frame_h5(tmp_path: Path) -> "tuple[Path, FEMData]":
    """Build a simple-frame ``model.h5`` and return the path + the FEM.

    The file carries materials (Steel02), one Fiber section, one
    Linear geomTransf, one Lobatto integration, one
    ``forceBeamColumn`` element, and a fix at the base node.  No
    patterns / recorders — the lean shape lets us focus the tests on
    the typed-record round-trip without spurious complexity.
    """
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    fem = build_simple_frame_fem()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )

    out = tmp_path / "simple_frame.h5"
    ops.h5(str(out))
    return out, fem


def build_frame_with_orientation_fan_out_h5(
    tmp_path: Path,
) -> "tuple[Path, FEMData]":
    """Build a frame with multiple ``geomTransf`` calls (the
    one-record-per-call schema deviation).

    Two PGs, two transforms with distinct explicit vecxz — exercises
    the per-emitted-call transform grouping the
    :class:`TransformRecord` docstring warns about.
    """
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    # Two parallel columns + one beam.
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ], dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2,
    )
    col_a = np.array([1], dtype=np.int64)
    col_b = np.array([2], dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array([[1, 2], [3, 4]], dtype=np.int64),
    )
    pg = {
        (1, 100): {
            "name": "Col_A",
            "node_ids": np.array([1, 2], dtype=np.int64),
            "node_coords": node_coords[:2],
            "element_ids": col_a,
        },
        (1, 101): {
            "name": "Col_B",
            "node_ids": np.array([3, 4], dtype=np.int64),
            "node_coords": node_coords[2:],
            "element_ids": col_b,
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=4, n_elems=2, bandwidth=2, types=[line_info],
    )
    fem = FEMData(nodes=nodes, elements=elements, info=info,
                  model_name="orientation_fan_out")

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    # Two transforms with explicit vecxz — distinct per column.
    transf_a = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    transf_b = ops.geomTransf.Linear(vecxz=(0.0, 1.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Col_A", transf=transf_a, integration=integ,
    )
    ops.element.forceBeamColumn(
        pg="Col_B", transf=transf_b, integration=integ,
    )

    out = tmp_path / "fan_out_frame.h5"
    ops.h5(str(out))
    return out, fem
