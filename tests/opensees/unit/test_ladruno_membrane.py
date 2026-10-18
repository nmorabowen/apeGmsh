"""Unit tests for Slice 2 — ``LadrunoMembrane`` (``ELE_TAG = 33022``)."""
from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.element.shell import LadrunoMembrane
from apeGmsh.opensees.emitter import live
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.section.plate import ElasticMembranePlateSection
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _2x2_slab_fem() -> FEMData:
    """2x2 quad mesh (9 nodes, 4 quad cells) with perimeter beams and corner supports.

    Node layout:
      7 --- 8 --- 9
      |  C  |  D  |
      4 --- 5 --- 6
      |  A  |  B  |
      1 --- 2 --- 3
    Node 5 is strictly interior; the outer boundary loop is (1, 2, 3, 6, 9, 8, 7, 4).
    """
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 1
            [2.0, 0.0, 0.0],  # 2
            [4.0, 0.0, 0.0],  # 3
            [0.0, 2.0, 0.0],  # 4
            [2.0, 2.0, 0.0],  # 5 (interior)
            [4.0, 2.0, 0.0],  # 6
            [0.0, 4.0, 0.0],  # 7
            [2.0, 4.0, 0.0],  # 8
            [4.0, 4.0, 0.0],  # 9
        ],
        dtype=np.float64,
    )
    node_ids = np.arange(1, 10, dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=4,
    )
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrilateral 4", dim=2, order=1, npe=4, count=4,
    )
    beam_eids = np.array([101, 102, 103, 104], dtype=np.int64)
    beam_conn = np.array([[1, 3], [3, 9], [9, 7], [7, 1]], dtype=np.int64)
    quad_eids = np.array([201, 202, 203, 204], dtype=np.int64)
    quad_conn = np.array(
        [
            [1, 2, 5, 4],
            [2, 3, 6, 5],
            [4, 5, 8, 7],
            [5, 6, 9, 8],
        ],
        dtype=np.int64,
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet({
            (0, 1): {
                "name": "Corners",
                "node_ids": np.array([1, 3, 7, 9], dtype=np.int64),
                "node_coords": coords[[0, 2, 6, 8]],
            },
        }),
        labels=LabelSet({
            (0, 1): {
                "name": "c1",
                "node_ids": np.array([1], dtype=np.int64),
                "node_coords": coords[[0]],
            },
        }),
    )
    elements = ElementComposite(
        groups={
            1: ElementGroup(element_type=line_info, ids=beam_eids, connectivity=beam_conn),
            3: ElementGroup(element_type=quad_info, ids=quad_eids, connectivity=quad_conn),
        },
        physical=PhysicalGroupSet({
            (1, 1): {
                "name": "Beams",
                "node_ids": np.array([1, 3, 7, 9], dtype=np.int64),
                "node_coords": coords[[0, 2, 6, 8]],
                "element_types": [1],
                "element_ids": beam_eids,
                "connectivity": beam_conn,
            },
            (2, 2): {
                "name": "Slab",
                "node_ids": node_ids,
                "node_coords": coords,
                "element_types": [3],
                "element_ids": quad_eids,
                "connectivity": quad_conn,
            },
        }),
        labels=LabelSet({
            (1, 1): {
                "name": "south_beam",
                "node_ids": np.array([1, 3], dtype=np.int64),
                "node_coords": coords[[0, 2]],
                "element_types": [1],
                "element_ids": np.array([101], dtype=np.int64),
                "connectivity": np.array([[1, 3]], dtype=np.int64),
            },
        }),
    )
    info = MeshInfo(n_nodes=9, n_elems=8, bandwidth=8, types=[line_info, quad_info])
    return FEMData(nodes=nodes, elements=elements, info=info)


class TestLadrunoMembraneConstructionAndValidation:
    def test_ele_tag_constant(self) -> None:
        assert LadrunoMembrane.ELE_TAG == 33022

    def test_requires_exactly_one_of_pg_or_nodes(self) -> None:
        with pytest.raises(ValueError, match="exactly one of pg= or nodes="):
            LadrunoMembrane()
        with pytest.raises(ValueError, match="exactly one of pg= or nodes="):
            LadrunoMembrane(pg="Slab", nodes=(1, 2, 3))

    def test_nodes_requires_at_least_three(self) -> None:
        with pytest.raises(ValueError, match="at least 3 boundary nodes"):
            LadrunoMembrane(nodes=(1, 2))

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            LadrunoMembrane(nodes=(1, 2, 3), method="invalid_method")

    @pytest.mark.parametrize("sa", [(1.0,), (1.0, 2.0, 3.0, 4.0), (0.0, 0.0), (0.0, 0.0, 0.0)])
    def test_invalid_span_axis_rejected(self, sa: tuple[float, ...]) -> None:
        with pytest.raises(ValueError, match="span_axis"):
            LadrunoMembrane(nodes=(1, 2, 3), span_axis=sa)

    @pytest.mark.parametrize("eta", [0.0, -1.0, 1.0001, 1.5, float("nan"), float("inf")])
    def test_invalid_ortho_ratio_rejected(self, eta: float) -> None:
        with pytest.raises(ValueError, match="ortho_ratio"):
            LadrunoMembrane(nodes=(1, 2, 3), ortho_ratio=eta)

    def test_ortho_ratio_upper_bound_one_accepted(self) -> None:
        m = LadrunoMembrane(nodes=(1, 2, 3), ortho_ratio=1.0)
        assert m.ortho_ratio == 1.0

    def test_stiff_thick_mat_section_and_E_nu_thickness_exclusivity(self) -> None:
        mat = ElasticIsotropic(E=30e9, nu=0.2)
        sec = ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)
        with pytest.raises(ValueError, match="mutually exclusive"):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=(30e9, 0.2, 0.2), thick=0.2, mat=mat)
        with pytest.raises(ValueError, match="mutually exclusive"):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=(30e9, 0.2, 0.2), section=sec)
        with pytest.raises(ValueError, match="mutually exclusive"):
            LadrunoMembrane(nodes=(1, 2, 3), section=sec, thickness=0.2, mat=mat)
        with pytest.raises(ValueError, match="mutually exclusive"):
            LadrunoMembrane(nodes=(1, 2, 3), E=30e9, nu=0.2, thickness=0.2, section=sec)
        with pytest.raises(ValueError, match="mutually exclusive"):
            LadrunoMembrane(nodes=(1, 2, 3), E=30e9, nu=0.2, thickness=0.2, mat=mat)
        with pytest.raises(ValueError, match="requires either mat= or both E= and nu="):
            LadrunoMembrane(nodes=(1, 2, 3), thickness=0.2)
        with pytest.raises(ValueError, match="supplied together"):
            LadrunoMembrane(nodes=(1, 2, 3), mat=mat)
        with pytest.raises(ValueError, match="must all be provided together"):
            LadrunoMembrane(nodes=(1, 2, 3), E=30e9, thickness=0.2)
        with pytest.raises(ValueError, match="must all be provided together"):
            LadrunoMembrane(nodes=(1, 2, 3), E=30e9, nu=0.2)

    def test_E_nu_thickness_and_mat_thickness_fields(self) -> None:
        m_iso = LadrunoMembrane(nodes=(1, 2, 3), E=30e9, nu=0.2, thickness=0.25)
        assert m_iso.E == 30e9
        assert m_iso.nu == 0.2
        assert m_iso.thickness == 0.25
        assert m_iso.stiff == (30e9, 0.2, 0.25)

        mat = ElasticIsotropic(E=30e9, nu=0.2)
        m_mat = LadrunoMembrane(nodes=(1, 2, 3), mat=mat, thickness=0.18)
        assert m_mat.thickness == 0.18
        assert m_mat.thick == 0.18
        assert m_mat.mat is mat

    def test_invalid_stiff_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="3-tuple"):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=(30e9, 0.2))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="E must be > 0"):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=(0.0, 0.2, 0.2))
        with pytest.raises(ValueError, match="nu must be in"):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=(30e9, 0.5, 0.2))
        with pytest.raises(ValueError, match="t must be > 0"):
            LadrunoMembrane(nodes=(1, 2, 3), stiff=(30e9, 0.2, -0.1))

    def test_invalid_hole_loop_and_duplicate_nodes_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 3 nodes"):
            LadrunoMembrane(nodes=(1, 2, 3, 4), holes=[(10, 11)])
        with pytest.raises(ValueError, match="at least 3 nodes"):
            LadrunoMembrane(nodes=(1, 2, 3, 4), holes=(10, 11))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="distinct"):
            LadrunoMembrane(nodes=(1, 1, 2))
        with pytest.raises(ValueError, match="distinct"):
            LadrunoMembrane(nodes=(1, 2, 3, 4), holes=[(5, 5, 6)])

    def test_invalid_rho_and_mesh_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            LadrunoMembrane(nodes=(1, 2, 3), rho=-1.0)
        with pytest.raises(ValueError, match="mesh_size"):
            LadrunoMembrane(nodes=(1, 2, 3), mesh_size=0.0)


class TestLadrunoMembraneEmitAndBridge:
    @pytest.mark.parametrize("method", ["tributary", "oneway", "fea_poisson", "fea_plate"])
    def test_direct_emit_all_methods_and_options(self, method: str) -> None:
        mat = ElasticIsotropic(E=30e9, nu=0.2)
        ele = LadrunoMembrane(
            nodes=(1, 2, 3, 4),
            frames=(10, 11),
            support_nodes=(1, 3),
            holes=[(21, 22, 23)],
            method=method,
            span_axis=(1.0, 0.0, 0.0),
            ortho_ratio=0.5,
            pressure=-5.0,
            thickness=0.2,
            mat=mat,
            rho=2400.0,
            lumped=True,
            mesh_size=0.5,
        )
        assert ele.dependencies() == (mat,)
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 77 if p is mat else 0)
        ele._emit(rec, tag=99)
        assert rec.calls == [
            (
                "element",
                (
                    "LadrunoMembrane",
                    99,
                    "-nodes", 1, 2, 3, 4,
                    "-frames", 10, 11,
                    "-supportNodes", 1, 3,
                    "-hole", 21, 22, 23,
                    "-method", method,
                    "-spanAxis", 1.0, 0.0, 0.0,
                    "-orthoRatio", 0.5,
                    "-pressure", -5.0,
                    "-thick", 0.2, "-mat", 77,
                    "-rho", 2400.0,
                    "-lumped",
                    "-meshSize", 0.5,
                ),
                {},
            )
        ]

    def test_pg_extracts_outer_boundary_and_resolves_frames_and_support_nodes(self) -> None:
        fem = _2x2_slab_fem()
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        beams = ops.element.elasticBeamColumn(
            pg="Beams", transf=transf, A=0.1, E=30e9, Iz=1e-3, Iy=1e-3, G=12e9, J=1e-4,
        )
        ops.element.LadrunoMembrane(
            pg="Slab",
            frames=beams,
            support_nodes="Corners",
            method="fea_plate",
            pressure=-8.0,
            E=30e9,
            nu=0.2,
            thickness=0.2,
            rho=2500.0,
            lumped=True,
            mesh_size=1.0,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))

        rec = RecordingEmitter()
        ops.build().emit(rec)
        mem_calls = [c for c in rec.calls if c[0] == "element" and c[1][0] == "LadrunoMembrane"]
        assert len(mem_calls) == 1
        args = mem_calls[0][1]
        # Tag 7 (after 2 element-spec seeds and 4 beam elements 3..6)
        assert args[:2] == ("LadrunoMembrane", 7)
        # Outer perimeter loop of 8 nodes (interior node 5 excluded!)
        assert args[2:11] == ("-nodes", 1, 2, 3, 6, 9, 8, 7, 4)
        assert args[11:16] == ("-frames", 3, 4, 5, 6)
        assert args[16:21] == ("-supportNodes", 1, 3, 7, 9)
        assert "-method" in args and args[args.index("-method") + 1] == "fea_plate"
        assert args[args.index("-stiff"):args.index("-stiff") + 4] == (
            "-stiff", 30e9, 0.2, 0.2,
        )

    def test_nodes_with_point_labels_and_node_handles(self, tmp_path: Path) -> None:
        fem = _2x2_slab_fem()
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        n3 = ops.nodes.get(tag=3)
        n9 = ops.nodes.get(tag=9)
        corners_set = ops.nodes.get(pg="Corners")
        ops.element.LadrunoMembrane(
            nodes=("c1", n3, n9, 7),
            support_nodes=(corners_set,),
            holes=[("c1", 2, 5)],
            pressure=-4.0,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))

        rec = RecordingEmitter()
        ops.build().emit(rec)
        mem_calls = [c for c in rec.calls if c[0] == "element" and c[1][0] == "LadrunoMembrane"]
        assert len(mem_calls) == 1
        args = mem_calls[0][1]
        assert args[2:7] == ("-nodes", 1, 3, 9, 7)
        assert args[7:12] == ("-supportNodes", 1, 3, 7, 9)
        assert args[12:16] == ("-hole", 1, 2, 5)

        # Ensure Node and NodeSet in nodes=/support_nodes= serialize cleanly to H5
        h5_path = tmp_path / "node_handles.h5"
        ops.h5(h5_path)
        assert h5_path.is_file()

    def test_frames_by_pg_name_and_label_and_element_tags_fem(self) -> None:
        fem = _2x2_slab_fem()
        ops = apeSees(fem, element_tags="fem")
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops.element.elasticBeamColumn(
            pg="Beams", transf=transf, A=0.1, E=30e9, Iz=1e-3, Iy=1e-3, G=12e9, J=1e-4,
        )
        sec = ops.section.ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2, rho=2400.0)
        n3 = ops.nodes.get(tag=3)
        corners_set = ops.nodes.get(pg="Corners")
        ops.element.LadrunoMembrane(
            pg="Slab",
            frames=("Beams", "south_beam"),
            support_nodes=("c1", n3, corners_set),
            section=sec,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))

        rec = RecordingEmitter()
        ops.build().emit(rec)
        mem_calls = [c for c in rec.calls if c[0] == "element" and c[1][0] == "LadrunoMembrane"]
        assert len(mem_calls) == 1
        args = mem_calls[0][1]
        # Under element_tags="fem", max FEM eid is 204, so synthetic macro-element gets 205
        assert args[1] == 205
        assert args[args.index("-frames"):args.index("-frames") + 5] == (
            "-frames", 101, 102, 103, 104,
        )
        assert args[args.index("-supportNodes"):args.index("-supportNodes") + 5] == (
            "-supportNodes", 1, 3, 7, 9,
        )
        assert args[args.index("-stiff"):args.index("-stiff") + 4] == (
            "-stiff", 30e9, 0.2, 0.2,
        )
        assert args[args.index("-rho"):args.index("-rho") + 2] == ("-rho", 2400.0)

    def test_pg_with_interior_hole_and_explicit_holes(self) -> None:
        """A 3x3 quad grid with the center quad missing automatically extracts both
        the 12-node outer boundary and the 4-node interior hole loop."""
        # 4x4 grid of nodes 1..16
        coords = np.array(
            [[float(j), float(i), 0.0] for i in range(4) for j in range(4)],
            dtype=np.float64,
        )
        node_ids = np.arange(1, 17, dtype=np.int64)
        # 8 quads around the perimeter (skipping the center quad (1,1))
        cells = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                n00 = i * 4 + j + 1
                n10 = n00 + 1
                n11 = n00 + 5
                n01 = n00 + 4
                cells.append((n00, n10, n11, n01))
        nodes = _NodesStub(
            ids=node_ids,
            coords=coords,
            node_pgs={"Corners": (1, 4, 13, 16)},
        )
        elements = _ElementsStub(
            elem_pgs={
                "DonutSlab": _ElementGroupView(
                    ids=tuple(range(1, len(cells) + 1)),
                    connectivity=tuple(cells),
                ),
            },
        )
        fem = FEMStub(nodes=nodes, elements=elements)
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(
            pg="DonutSlab",
            support_nodes="Corners",
            pressure=-5.0,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))

        rec = RecordingEmitter()
        ops.build().emit(rec)
        mem_calls = [c for c in rec.calls if c[0] == "element" and c[1][0] == "LadrunoMembrane"]
        assert len(mem_calls) == 1
        args = mem_calls[0][1]
        hole_idx = args.index("-hole")
        method_idx = args.index("-method")
        hole_nodes = set(args[hole_idx + 1:method_idx])
        assert hole_nodes == {6, 7, 10, 11}

    def test_pg_non_manifold_pinch_vertex_raises_bridge_error(self) -> None:
        """Two triangles sharing only vertex 3 have boundary degree 4 at vertex 3."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        nodes = _NodesStub(ids=np.arange(1, 6, dtype=np.int64), coords=coords, node_pgs={})
        elements = _ElementsStub(
            elem_pgs={
                "BowTie": _ElementGroupView(
                    ids=(1, 2),
                    connectivity=((1, 2, 3), (3, 4, 5)),
                ),
            },
        )
        fem = FEMStub(nodes=nodes, elements=elements)
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(pg="BowTie", support_nodes=(1, 2, 4))
        ops.fix(nodes=[1, 2, 4], dofs=(1, 1, 1, 1, 1, 1))
        with pytest.raises(BridgeError, match="boundary degree 4"):
            ops.build().emit(RecordingEmitter())

    def test_pg_disconnected_islands_raises_bridge_error(self) -> None:
        """Two disjoint quads in the same PG must raise BridgeError rather than
        treating the second island as an interior hole."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [6.0, 1.0, 0.0],
                [5.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        nodes = _NodesStub(ids=np.arange(1, 9, dtype=np.int64), coords=coords, node_pgs={})
        elements = _ElementsStub(
            elem_pgs={
                "TwoIslands": _ElementGroupView(
                    ids=(1, 2),
                    connectivity=((1, 2, 3, 4), (5, 6, 7, 8)),
                ),
            },
        )
        fem = FEMStub(nodes=nodes, elements=elements)
        ops = apeSees(cast("object", fem))
        ops.model(ndm=3, ndf=6)
        ops.element.LadrunoMembrane(pg="TwoIslands", support_nodes=(1, 2, 3))
        ops.fix(nodes=[1, 2, 3], dofs=(1, 1, 1, 1, 1, 1))
        with pytest.raises(BridgeError, match="disconnected polygon boundary loops"):
            ops.build().emit(RecordingEmitter())

    def test_pattern_surface_load_and_self_weight_on_membrane(self) -> None:
        fem = _2x2_slab_fem()
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        beams = ops.element.elasticBeamColumn(
            pg="Beams", transf=transf, A=0.1, E=30e9, Iz=1e-3, Iy=1e-3, G=12e9, J=1e-4,
        )
        slab = ops.element.LadrunoMembrane(
            pg="Slab",
            frames=beams,
            pressure=-5.0,
            E=30e9,
            nu=0.2,
            thickness=0.2,
            rho=2500.0,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as p:
            p.surface_load(ele=slab)
            p.surface_load(pg="Slab")
            p.self_weight(ele=slab, wz=-9.81)
            p.self_weight(pg="Slab", wz=-9.81)

        rec = RecordingEmitter()
        ops.build().emit(rec)
        ele_load_calls = [c[1] for c in rec.calls if c[0] == "eleLoad"]
        assert ele_load_calls == [
            ("-ele", 7, "-type", "-surfaceLoad"),
            ("-ele", 7, "-type", "-surfaceLoad"),
            ("-ele", 7, "-type", "-selfWeight", 0.0, 0.0, -9.81),
            ("-ele", 7, "-type", "-selfWeight", 0.0, 0.0, -9.81),
        ]

    def test_readback_preserves_rotational_dofs_on_shared_beam_membrane_nodes(self) -> None:
        """Perimeter beam nodes shared with a LadrunoMembrane must keep rotational
        DOFs unblocked in _readback.rotation_roles and rotation_roles_from_tokens."""
        from apeGmsh.opensees._internal.build.pg_expand import (
            ElementPlanRows,
            expand_spec_to_elements,
        )
        from apeGmsh.opensees._readback import rotation_roles, rotation_roles_from_tokens

        fem = _2x2_slab_fem()
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        beams = ops.element.elasticBeamColumn(
            pg="Beams", transf=transf, A=0.1, E=30e9, Iz=1e-3, Iy=1e-3, G=12e9, J=1e-4,
        )
        slab = ops.element.LadrunoMembrane(
            pg="Slab",
            frames=beams,
            pressure=-5.0,
        )
        beam_fan = expand_spec_to_elements(fem, beams)
        slab_fan = expand_spec_to_elements(fem, slab)
        plan = [
            (beams, ElementPlanRows(beam_fan.eids, beam_fan.conn, 3)),
            (slab, ElementPlanRows(slab_fan.eids, slab_fan.conn, 7)),
        ]

        roles = rotation_roles(plan)
        # Perimeter nodes 1..4, 6..9 are shared by elasticBeamColumn and LadrunoMembrane:
        # they MUST remain in roles.rotational and NOT in roles.blocked!
        assert roles.rotational == frozenset({1, 2, 3, 4, 6, 7, 8, 9})
        assert roles.blocked == {}
        assert roles.model_blocker is None

        tok_roles = rotation_roles_from_tokens([
            ("elasticBeamColumn", (1, 2, 3, 6, 9, 8, 7, 4)),
            ("LadrunoMembrane", (1, 2, 3, 6, 9, 8, 7, 4)),
        ])
        assert tok_roles.rotational == frozenset({1, 2, 3, 4, 6, 7, 8, 9})
        assert tok_roles.blocked == {}
        assert tok_roles.model_blocker is None

    def test_h5_round_trip_multiple_membranes_byte_identical_tcl_and_py(
        self, tmp_path: Path,
    ) -> None:
        from apeGmsh.opensees.emitter.py import PyEmitter
        from apeGmsh.opensees.emitter.tcl import TclEmitter

        fem = _2x2_slab_fem()
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        beams = ops.element.elasticBeamColumn(
            pg="Beams", transf=transf, A=0.1, E=30e9, Iz=1e-3, Iy=1e-3, G=12e9, J=1e-4,
        )
        mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
        # First membrane: long argument list (pg, frames, support_nodes, holes, mat+thickness, ...)
        ops.element.LadrunoMembrane(
            pg="Slab",
            frames=beams,
            support_nodes="Corners",
            holes=[(2, 6, 8)],
            method="oneway",
            span_axis=(0.6, 0.8, 0.1),
            ortho_ratio=0.25,
            pressure=-12.5,
            thickness=0.2,
            mat=mat,
            rho=2400.5,
            lumped=True,
            mesh_size=0.75,
        )
        # Second membrane: shorter argument list (explicit nodes=, E/nu/thickness)
        ops.element.LadrunoMembrane(
            nodes=(1, 3, 9, 7),
            support_nodes=(1, 3, 7, 9),
            method="tributary",
            pressure=-4.25,
            E=25000.5,
            nu=0.15,
            thickness=0.15,
        )
        ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))

        built = ops.build()
        tcl_em = TclEmitter()
        py_em = PyEmitter()
        built.emit(tcl_em)
        built.emit(py_em)

        h5_path = tmp_path / "membrane_model.h5"
        ops.h5(h5_path)
        reloaded = OpenSeesModel.from_h5(h5_path)

        direct_tcl = [
            line.strip() for line in tcl_em.lines()
            if line.strip().startswith("element LadrunoMembrane")
        ]
        reloaded_tcl = [
            line.strip() for line in reloaded.build("tcl").splitlines()
            if line.strip().startswith("element LadrunoMembrane")
        ]
        assert len(direct_tcl) == 2
        assert reloaded_tcl == direct_tcl

        direct_py = [
            line.strip() for line in py_em.lines()
            if "LadrunoMembrane" in line
        ]
        reloaded_py = [
            line.strip() for line in reloaded.build("py").splitlines()
            if "LadrunoMembrane" in line
        ]
        assert len(direct_py) == 2
        assert reloaded_py == direct_py

    def test_live_emitter_refuses_on_stock_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_ops = MagicMock(name="stock_ops")
        mock_ops.getEleTags.return_value = []
        monkeypatch.setattr(live, "_get_ops", lambda: mock_ops)

        emitter = live.LiveOpsEmitter(wipe=True)
        with pytest.raises(RuntimeError, match=r"(?s)LadrunoMembrane.*Ladruno fork"):
            emitter.element("LadrunoMembrane", 1, "-nodes", 1, 2, 3, "-method", "tributary")
