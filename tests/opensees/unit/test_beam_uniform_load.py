"""Unit tests for Slice 1 — Element-Form Loads (`from_model` + `Plain.beam_uniform`
/ `surface_load` / `self_weight`), Geometry-to-Local Beam Projection, and Stock
Refusal Gate.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh._kernel.defs.loads import LineLoadDef
from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh._kernel.record_sets import ComposeSet, ElementLoadSet
from apeGmsh._kernel.records._loads import ElementLoadRecord
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver
from apeGmsh.core.LoadsComposite import LoadsComposite
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
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ===========================================================================
# Helpers
# ===========================================================================


class _DummyGeometry:
    def __init__(self) -> None:
        self._load_defs: list[Any] = []


def _resolver(coords_by_tag: dict[int, tuple[float, float, float]]) -> LoadResolver:
    tags = np.array(sorted(coords_by_tag), dtype=np.int64)
    coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
    return LoadResolver(tags, coords)


def _beam_fem_data(
    *,
    ndm: int = 2,
    loads: list[ElementLoadRecord] | None = None,
) -> FEMData:
    """Synthetic ``FEMData`` with 2 beam elements (eids 10, 20) in PG ``"Beams"``."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([10, 20], dtype=np.int64),
        connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet({
            (0, 1): {
                "name": "Base",
                "node_ids": np.array([1, 3], dtype=np.int64),
                "node_coords": coords[[0, 2]],
            },
        }),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({
            (1, 1): {
                "name": "Beams",
                "node_ids": node_ids,
                "node_coords": coords,
                "element_types": [1],
                "element_ids": np.array([10, 20], dtype=np.int64),
                "connectivity": np.array([[1, 2], [2, 3]], dtype=np.int64),
            },
        }),
        labels=LabelSet({
            (1, 1): {
                "name": "span1",
                "node_ids": np.array([1, 2], dtype=np.int64),
                "node_coords": coords[:2],
                "element_types": [1],
                "element_ids": np.array([10], dtype=np.int64),
                "connectivity": np.array([[1, 2]], dtype=np.int64),
            },
        }),
    )
    if loads is not None:
        elements.loads = ElementLoadSet(loads)
    info = MeshInfo(n_nodes=3, n_elems=2, bandwidth=1, types=[line_info])
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test": ops.test.NormDispIncr(tol=1e-6, max_iter=10),
        "algorithm": ops.algorithm.Linear(),
        "integrator": ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.Plain(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def _emitted_elem_tags(rec: RecordingEmitter) -> list[int]:
    return [int(c[1][1]) for c in rec.calls if c[0] == "element"]


# ===========================================================================
# 1. LoadsComposite.line + LoadResolver.resolve_line_element
# ===========================================================================


class TestGeometryLineElementLoad:
    def test_full_span_uniform_params(self) -> None:
        g = _DummyGeometry()
        lc = LoadsComposite(g)
        with lc.case("DL"):
            d = lc.line(
                "beam",
                magnitude=12.0,
                direction=(0.0, -1.0, 0.0),
                target_form="element",
            )
        assert isinstance(d, LineLoadDef)
        assert d.span == (0.0, 1.0)
        assert d.pattern == "DL"
        r = _resolver({1: (0.0, 0.0, 0.0), 2: (4.0, 0.0, 0.0)})
        recs = r.resolve_line_element(d, [101])
        assert len(recs) == 1
        assert recs[0] == ElementLoadRecord(
            element_id=101,
            load_type="beamUniform",
            params={"wx": 0.0, "wy": -12.0, "wz": 0.0},
            pattern="DL",
        )

    def test_partial_span_uniform_params(self) -> None:
        g = _DummyGeometry()
        lc = LoadsComposite(g)
        d = lc.line(
            "beam",
            q_xyz=(0.0, -10.0, 0.0),
            span=(0.25, 0.75),
            target_form="element",
        )
        r = _resolver({1: (0.0, 0.0, 0.0), 2: (4.0, 0.0, 0.0)})
        recs = r.resolve_line_element(d, [5])
        assert len(recs) == 1
        assert recs[0].params == {
            "wx": 0.0,
            "wy": -10.0,
            "wz": 0.0,
            "wx_end": 0.0,
            "wy_end": -10.0,
            "wz_end": 0.0,
            "aL": 0.25,
            "bL": 0.75,
        }

    def test_trapezoidal_magnitude_and_q_xyz(self) -> None:
        g = _DummyGeometry()
        lc = LoadsComposite(g)
        d1 = lc.line(
            "beam",
            magnitude=(10.0, 20.0),
            direction=(0.0, 0.0, -1.0),
            span=(0.1, 0.9),
            target_form="element",
        )
        r = _resolver({1: (0.0, 0.0, 0.0), 2: (5.0, 0.0, 0.0)})
        recs1 = r.resolve_line_element(d1, [7])
        assert recs1[0].params == {
            "wx": 0.0,
            "wy": 0.0,
            "wz": -10.0,
            "wx_end": 0.0,
            "wy_end": 0.0,
            "wz_end": -20.0,
            "aL": 0.1,
            "bL": 0.9,
        }

        d2 = lc.line(
            "beam",
            q_xyz=((1.0, -2.0, -3.0), (4.0, -5.0, -6.0)),
            target_form="element",
        )
        recs2 = r.resolve_line_element(d2, [8])
        assert recs2[0].params == {
            "wx": 1.0,
            "wy": -2.0,
            "wz": -3.0,
            "wx_end": 4.0,
            "wy_end": -5.0,
            "wz_end": -6.0,
            "aL": 0.0,
            "bL": 1.0,
        }

    def test_varying_element_load_with_partial_span(self) -> None:
        g = _DummyGeometry()
        lc = LoadsComposite(g)
        d = lc.line(
            "beam",
            magnitude=lambda x, y, z: 10.0 + 2.0 * x,
            direction=(0.0, -1.0, 0.0),
            span=(0.25, 0.75),
            target_form="element",
        )
        r = _resolver({1: (0.0, 0.0, 0.0), 2: (4.0, 0.0, 0.0)})
        recs = r.resolve_line_element_varying(d, [(11, (0.0, -14.0, 0.0))])
        assert recs[0].params == {
            "wx": 0.0,
            "wy": -14.0,
            "wz": 0.0,
            "wx_end": 0.0,
            "wy_end": -14.0,
            "wz_end": 0.0,
            "aL": 0.25,
            "bL": 0.75,
        }

    @pytest.mark.parametrize(
        "bad_span",
        [(-0.1, 0.5), (0.0, 1.1), (0.5, 0.5), (0.8, 0.2)],
    )
    def test_invalid_span_rejected(self, bad_span: tuple[float, float]) -> None:
        g = _DummyGeometry()
        lc = LoadsComposite(g)
        with pytest.raises(ValueError, match="span"):
            lc.line(
                "beam",
                magnitude=10.0,
                direction=(0.0, -1.0, 0.0),
                span=bad_span,
                target_form="element",
            )

    def test_nodal_rejects_partial_span_and_trapezoidal_tuple(self) -> None:
        g = _DummyGeometry()
        lc = LoadsComposite(g)
        with pytest.raises(ValueError, match="target_form='element'"):
            lc.line(
                "beam",
                magnitude=10.0,
                direction=(0.0, -1.0, 0.0),
                span=(0.2, 0.8),
                target_form="nodal",
            )
        with pytest.raises(ValueError, match="target_form='element'"):
            lc.line(
                "beam",
                magnitude=(10.0, 20.0),
                direction=(0.0, -1.0, 0.0),
                target_form="nodal",
            )
        with pytest.raises(ValueError, match="target_form='element'"):
            lc.line(
                "beam",
                q_xyz=((0.0, -10.0, 0.0), (0.0, -20.0, 0.0)),
                target_form="nodal",
            )


# ===========================================================================
# 2. Direct Plain.beam_uniform / surface_load / self_weight
# ===========================================================================


class TestPlainDirectElementLoads:
    def test_2d_and_3d_beam_uniform_and_self_weight_and_surface_load(self) -> None:
        fem2d = _beam_fem_data(ndm=2)
        ops2d = apeSees(fem2d)
        ops2d.model(ndm=2, ndf=3)
        t2 = ops2d.geomTransf.Linear()
        ops2d.element.elasticBeamColumn(
            pg="Beams", transf=t2, A=0.01, E=200e9, Iz=1e-4,
        )
        ops2d.fix(pg="Base", dofs=(1, 1, 1))
        with ops2d.pattern.Plain(series=ops2d.timeSeries.Linear()) as p:
            p.beam_uniform(pg="Beams", wy=-10.0, wx=2.0)
            p.beam_uniform(
                label="span1",
                wy=-5.0,
                wx=1.0,
                span=(0.2, 0.8),
            )
            p.beam_uniform(
                label="span1",
                wy=(-5.0, -15.0),
                wx=(1.0, 3.0),
                span=(0.2, 0.8),
            )
            p.beam_uniform(ele=20, wy=-8.0)
            p.surface_load(ele=[10, 20])
            p.self_weight(pg="Beams", wx=0.0, wy=-1.0)

        rec = RecordingEmitter()
        ops2d.build().emit(rec)
        t_e1, t_e2 = _emitted_elem_tags(rec)
        ele_loads = [c[1] for c in rec.calls if c[0] == "eleLoad"]
        assert ele_loads == [
            ("-ele", t_e1, t_e2, "-type", "-beamUniform", -10.0, 2.0),
            ("-ele", t_e1, "-type", "-beamUniform", -5.0, 1.0, 0.2, 0.8),
            ("-ele", t_e1, "-type", "-beamUniform", -5.0, 1.0, 0.2, 0.8, -15.0, 3.0),
            ("-ele", t_e2, "-type", "-beamUniform", -8.0, 0.0),
            ("-ele", t_e1, t_e2, "-type", "-surfaceLoad"),
            ("-ele", t_e1, t_e2, "-type", "-selfWeight", 0.0, -1.0),
        ]

        # Verify 3D argument ordering:
        # Full-span: (wy_a, wz_a, wx_a)
        # Partial uniform: (wy_a, wz_a, wx_a, aL, bL)
        # Trapezoidal: (wy_a, wz_a, wx_a, aL, bL, wy_b, wz_b, wx_b)
        fem3d = _beam_fem_data(ndm=3)
        ops3d = apeSees(fem3d)
        ops3d.model(ndm=3, ndf=6)
        t3 = ops3d.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops3d.element.elasticBeamColumn(
            pg="Beams", transf=t3, A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
        )
        ops3d.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
        with ops3d.pattern.Plain(series=ops3d.timeSeries.Linear()) as p3:
            p3.beam_uniform(pg="Beams", wy=-10.0, wz=-4.0, wx=1.5)
            p3.beam_uniform(
                ele=10,
                wy=-10.0,
                wz=-4.0,
                wx=1.0,
                span=(0.1, 0.9),
            )
            p3.beam_uniform(
                ele=10,
                wy=(-10.0, -20.0),
                wz=(-4.0, -8.0),
                wx=(1.0, 2.0),
                span=(0.1, 0.9),
            )
            p3.self_weight(pg="Beams", wx=0.0, wy=0.0, wz=-1.0)

        rec3 = RecordingEmitter()
        ops3d.build().emit(rec3)
        t3_e1, t3_e2 = _emitted_elem_tags(rec3)
        ele_loads_3d = [c[1] for c in rec3.calls if c[0] == "eleLoad"]
        assert ele_loads_3d == [
            ("-ele", t3_e1, t3_e2, "-type", "-beamUniform", -10.0, -4.0, 1.5),
            ("-ele", t3_e1, "-type", "-beamUniform", -10.0, -4.0, 1.0, 0.1, 0.9),
            ("-ele", t3_e1, "-type", "-beamUniform", -10.0, -4.0, 1.0, 0.1, 0.9, -20.0, -8.0, 2.0),
            ("-ele", t3_e1, t3_e2, "-type", "-selfWeight", 0.0, 0.0, -1.0),
        ]

    def test_2d_rejects_nonzero_wz(self) -> None:
        fem2d = _beam_fem_data(ndm=2)
        ops2d = apeSees(fem2d)
        ops2d.model(ndm=2, ndf=3)
        t2 = ops2d.geomTransf.Linear()
        ops2d.element.elasticBeamColumn(
            pg="Beams", transf=t2, A=0.01, E=200e9, Iz=1e-4,
        )
        ops2d.fix(pg="Base", dofs=(1, 1, 1))
        with ops2d.pattern.Plain(series=ops2d.timeSeries.Linear()) as p:
            p.beam_uniform(pg="Beams", wy=-10.0, wz=-5.0)
        with pytest.raises((BridgeError, ValueError), match="wz is not supported in 2D"):
            ops2d.build().emit(RecordingEmitter())

        ops2d_sw = apeSees(fem2d)
        ops2d_sw.model(ndm=2, ndf=3)
        t2b = ops2d_sw.geomTransf.Linear()
        ops2d_sw.element.elasticBeamColumn(
            pg="Beams", transf=t2b, A=0.01, E=200e9, Iz=1e-4,
        )
        ops2d_sw.fix(pg="Base", dofs=(1, 1, 1))
        with ops2d_sw.pattern.Plain(series=ops2d_sw.timeSeries.Linear()) as p2:
            p2.self_weight(pg="Beams", wz=-1.0)
        with pytest.raises((BridgeError, ValueError), match="wz is not supported in 2D"):
            ops2d_sw.build().emit(RecordingEmitter())

    def test_target_xor_and_span_validation(self) -> None:
        fem2d = _beam_fem_data(ndm=2)
        ops = apeSees(fem2d)
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            with pytest.raises(ValueError, match="exactly one of pg=, label=, or ele="):
                p.beam_uniform(wy=-10.0)
            with pytest.raises(ValueError, match="exactly one of pg=, label=, or ele="):
                p.beam_uniform(pg="Beams", ele=1, wy=-10.0)
            with pytest.raises(ValueError, match="0.0 <= aL < bL <= 1.0"):
                p.beam_uniform(pg="Beams", wy=-10.0, span=(0.8, 0.2))


# ===========================================================================
# 3. from_model(case) element-load consumption & global-to-local projection
# ===========================================================================


class TestFromModelElementLoads:
    def test_2d_global_to_local_projection_horizontal_vertical_inclined(self) -> None:
        """2D frame with horizontal beam (1->2), vertical column (1->3), and
        45-degree inclined member (2->3) under global downward load (0, -10, 0).
        """
        L = 4.0
        nodes = _NodesStub(
            ids=[1, 2, 3],
            coords=[(0.0, 0.0, 0.0), (L, 0.0, 0.0), (0.0, L, 0.0)],
            node_pgs={"Base": [1, 3]},
        )
        elements = _ElementsStub(
            elem_pgs={
                "Frame": _ElementGroupView(
                    ids=(10, 20, 30),
                    connectivity=((1, 2), (1, 3), (2, 3)),
                ),
            },
        )
        fem = FEMStub(nodes=nodes, elements=elements)
        # Global load (0, -10, 0) on all 3 elements
        fem.elements.loads = ElementLoadSet([
            ElementLoadRecord(
                element_id=eid,
                load_type="beamUniform",
                params={"wx": 0.0, "wy": -10.0, "wz": 0.0},
                pattern="grav",
            )
            for eid in (10, 20, 30)
        ])

        ops = apeSees(cast("object", fem))
        ops.model(ndm=2, ndf=3)
        t = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(
            pg="Frame", transf=t, A=0.01, E=200e9, Iz=1e-4,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("grav")

        rec = RecordingEmitter()
        ops.build().emit(rec)
        t_e1, t_e2, t_e3 = _emitted_elem_tags(rec)
        ele_loads = [c[1] for c in rec.calls if c[0] == "eleLoad"]
        assert len(ele_loads) == 3

        # Element 1 (eid 10, horizontal (0,0)->(L,0)): x_local=(1,0), y_local=(0,1)
        # -> Wy = -10, Wx = 0
        assert ele_loads[0][:5] == ("-ele", t_e1, "-type", "-beamUniform", -10.0)
        assert math.isclose(float(ele_loads[0][5]), 0.0, abs_tol=1e-12)

        # Element 2 (eid 20, vertical (0,0)->(0,L)): x_local=(0,1), y_local=(-1,0)
        # -> Wy = 0, Wx = -10
        assert ele_loads[1][:4] == ("-ele", t_e2, "-type", "-beamUniform")
        assert math.isclose(float(ele_loads[1][4]), 0.0, abs_tol=1e-12)
        assert math.isclose(float(ele_loads[1][5]), -10.0, abs_tol=1e-12)

        # Element 3 (eid 30, inclined (L,0)->(0,L)): x_local=(-1/sqrt2, 1/sqrt2),
        # y_local=(-1/sqrt2, -1/sqrt2)
        # Global (0, -10) -> Wx = -10/sqrt2, Wy = +10/sqrt2
        c45 = 10.0 / math.sqrt(2.0)
        assert ele_loads[2][:4] == ("-ele", t_e3, "-type", "-beamUniform")
        assert math.isclose(float(ele_loads[2][4]), c45, rel_tol=1e-12)
        assert math.isclose(float(ele_loads[2][5]), -c45, rel_tol=1e-12)

    def test_3d_beam_projection_with_vecxz(self) -> None:
        """3D beams along X with vecxz=(0, 0, 1):
        x_local = (1, 0, 0)
        y_local = unit(vecxz x x_local) = (0, 1, 0)
        z_local = x_local x y_local = (0, 0, 1)
        Global load (1, -2, -3) -> Wx=1, Wy=-2, Wz=-3 -> (-beamUniform, -2, -3, 1).
        """
        fem = _beam_fem_data(
            ndm=3,
            loads=[
                ElementLoadRecord(
                    element_id=10,
                    load_type="beamUniform",
                    params={"wx": 1.0, "wy": -2.0, "wz": -3.0},
                    pattern="q3d",
                ),
                ElementLoadRecord(
                    element_id=20,
                    load_type="beamUniform",
                    params={"wx": 1.0, "wy": -2.0, "wz": -3.0},
                    pattern="q3d",
                ),
            ],
        )
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("q3d")

        rec = RecordingEmitter()
        ops.build().emit(rec)
        t_e1, t_e2 = _emitted_elem_tags(rec)
        ele_loads = [c[1] for c in rec.calls if c[0] == "eleLoad"]
        assert ele_loads == [
            ("-ele", t_e1, "-type", "-beamUniform", -2.0, -3.0, 1.0),
            ("-ele", t_e2, "-type", "-beamUniform", -2.0, -3.0, 1.0),
        ]

    def test_from_model_trapezoidal_and_surface_and_body_force(self) -> None:
        fem = _beam_fem_data(
            ndm=2,
            loads=[
                ElementLoadRecord(
                    element_id=10,
                    load_type="beamUniform",
                    params={
                        "wx": 0.0,
                        "wy": -10.0,
                        "wz": 0.0,
                        "wx_end": 0.0,
                        "wy_end": -20.0,
                        "wz_end": 0.0,
                        "aL": 0.25,
                        "bL": 0.75,
                    },
                    pattern="combo",
                ),
                ElementLoadRecord(
                    element_id=20,
                    load_type="surfacePressure",
                    params={"p": -50.0, "normal": True, "direction": (0.0, -1.0, 0.0)},
                    pattern="combo",
                ),
                ElementLoadRecord(
                    element_id=10,
                    load_type="bodyForce",
                    params={"g": (0.0, -9.81, 0.0)},
                    pattern="combo",
                ),
            ],
        )
        ops = apeSees(fem)
        ops.model(ndm=2, ndf=3)
        t = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.01, E=200e9, Iz=1e-4,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("combo")

        rec = RecordingEmitter()
        ops.build().emit(rec)
        t_e1, t_e2 = _emitted_elem_tags(rec)
        ele_loads = [c[1] for c in rec.calls if c[0] == "eleLoad"]
        assert ele_loads == [
            ("-ele", t_e1, "-type", "-beamUniform", -10.0, 0.0, 0.25, 0.75, -20.0, 0.0),
            ("-ele", t_e2, "-type", "-surfaceLoad"),
            ("-ele", t_e1, "-type", "-selfWeight", 0.0, -9.81),
        ]

    def test_from_model_non_beam_line_load_raises(self) -> None:
        fem = _beam_fem_data(
            ndm=2,
            loads=[
                ElementLoadRecord(
                    element_id=10,
                    load_type="beamUniform",
                    params={"wx": 0.0, "wy": -10.0, "wz": 0.0},
                    pattern="bad",
                ),
            ],
        )
        ops = apeSees(fem)
        ops.model(ndm=2, ndf=2)
        mat = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
        ops.element.Truss(pg="Beams", A=0.01, material=mat)
        ops.fix(pg="Base", dofs=(1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("bad")

        with pytest.raises(BridgeError, match="not a beam-column element"):
            ops.build().emit(RecordingEmitter())


# ===========================================================================
# 4. Staged, Partitioned, and model.h5 Round-Trip
# ===========================================================================


class TestStagedPartitionedAndH5RoundTrip:
    def test_staged_and_partitioned_ele_load_emission(self) -> None:
        nodes = _NodesStub(
            ids=[1, 2, 3, 4],
            coords=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (2.0, 2.0, 0.0)],
            node_pgs={"Base": [1, 3]},
        )
        elements = _ElementsStub(
            elem_pgs={
                "Beams": _ElementGroupView(
                    ids=(10, 20),
                    connectivity=((1, 2), (3, 4)),
                ),
            },
        )
        fem = FEMStub(nodes=nodes, elements=elements)
        fem.set_partitions([
            (0, [1, 2], [10]),
            (1, [3, 4], [20]),
        ])
        fem.elements.loads = ElementLoadSet([
            ElementLoadRecord(
                element_id=20,
                load_type="beamUniform",
                params={"wx": 0.0, "wy": -15.0, "wz": 0.0},
                pattern="stage_dl",
            ),
        ])

        ops = apeSees(cast("object", fem))
        ops.model(ndm=2, ndf=3)
        t = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.01, E=200e9, Iz=1e-4,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1))
        with ops.stage(name="s1") as s:
            with s.pattern(series=ops.timeSeries.Linear()) as p:
                p.from_model("stage_dl")
            s.analysis(**_full_chain(ops))
            s.run(n_increments=1, dt=1.0)

        rec = RecordingEmitter()
        ops.build().emit(rec)
        _t_e1, t_e2 = _emitted_elem_tags(rec)

        # Element 20 (ops tag t_e2) is owned by rank 1 only; rank 0 has no pattern content
        rank = None
        ele_loads_by_rank: list[tuple[int | None, tuple[Any, ...]]] = []
        for name, args, _kw in rec.calls:
            if name == "partition_open":
                rank = int(args[0])
            elif name == "partition_close":
                rank = None
            elif name == "eleLoad":
                ele_loads_by_rank.append((rank, args))

        assert ele_loads_by_rank == [
            (1, ("-ele", t_e2, "-type", "-beamUniform", -15.0, 0.0)),
        ]

    def test_h5_round_trip_preserves_ele_loads(self, tmp_path: Path) -> None:
        fem = _beam_fem_data(
            ndm=2,
            loads=[
                ElementLoadRecord(
                    element_id=10,
                    load_type="beamUniform",
                    params={
                        "wx": 0.0,
                        "wy": -12.0,
                        "wz": 0.0,
                        "wx_end": 0.0,
                        "wy_end": -24.0,
                        "wz_end": 0.0,
                        "aL": 0.2,
                        "bL": 0.8,
                    },
                    pattern="dl",
                ),
            ],
        )
        ops = apeSees(fem)
        ops.model(ndm=2, ndf=3)
        ts = ops.timeSeries.Linear()
        t = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.01, E=200000.5, Iz=1e-4,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1))
        with ops.pattern.Plain(series=ts) as p:
            p.from_model("dl")
            p.beam_uniform(ele=20, wy=-5.0, wx=1.0)
            p.surface_load(ele=20)
            p.self_weight(pg="Beams", wy=-1.0)
        h5_path = tmp_path / "beam_loads.h5"
        tcl_path = tmp_path / "direct.tcl"
        py_path = tmp_path / "direct.py"
        ops.h5(str(h5_path))
        ops.tcl(str(tcl_path), progress=False, flat=True)
        ops.py(str(py_path), progress=False)

        reloaded = OpenSeesModel.from_h5(str(h5_path))
        tcl_reloaded = [
            line.strip()
            for line in reloaded.build("tcl").splitlines()
            if "eleLoad" in line
        ]
        tcl_direct = [
            line.strip()
            for line in tcl_path.read_text(encoding="utf-8").splitlines()
            if "eleLoad" in line
        ]
        assert tcl_reloaded == tcl_direct == [
            "eleLoad -ele 3 -type -beamUniform -5.0 1.0",
            "eleLoad -ele 3 -type -surfaceLoad",
            "eleLoad -ele 2 3 -type -selfWeight 0.0 -1.0",
            "eleLoad -ele 2 -type -beamUniform -12.0 0.0 0.2 0.8 -24.0 0.0",
        ]

        py_reloaded = [
            line.strip()
            for line in reloaded.build("py").splitlines()
            if "eleLoad" in line
        ]
        py_direct = [
            line.strip()
            for line in py_path.read_text(encoding="utf-8").splitlines()
            if "eleLoad" in line
        ]
        assert py_reloaded == py_direct == [
            "ops.eleLoad('-ele', 3, '-type', '-beamUniform', -5.0, 1.0)",
            "ops.eleLoad('-ele', 3, '-type', '-surfaceLoad')",
            "ops.eleLoad('-ele', 2, 3, '-type', '-selfWeight', 0.0, -1.0)",
            "ops.eleLoad('-ele', 2, '-type', '-beamUniform', -12.0, 0.0, 0.2, 0.8, -24.0, 0.0)",
        ]


# ===========================================================================
# 5. Stock OpenSees Refusal Gate (LiveOpsEmitter.eleLoad)
# ===========================================================================


class _StockOps:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def eleLoad(self, *args: object) -> None:  # noqa: N802
        self.calls.append(("eleLoad", *args))


class _CurrentForkOps(_StockOps):
    def criticalTimeStep(self) -> float:  # noqa: N802
        return 1.0

    def ladrunoBuild(self) -> str:  # noqa: N802
        return "a" * 40


def _live_emitter(ops: object, *, ndm: int = 2) -> LiveOpsEmitter:
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[assignment]
    le._in_partition = False
    le._model_ndm = ndm
    return le


class TestLiveOpsEleLoadGate:
    def test_stock_allows_full_span_beam_uniform_and_self_weight(self) -> None:
        ops2 = _StockOps()
        le2 = _live_emitter(ops2, ndm=2)
        le2.eleLoad("-ele", 1, "-type", "-beamUniform", -10.0, 0.0)
        le2.eleLoad("-ele", 1, "-type", "-selfWeight", 0.0, -1.0)
        assert len(ops2.calls) == 2

        ops3 = _StockOps()
        le3 = _live_emitter(ops3, ndm=3)
        le3.eleLoad("-ele", 1, "-type", "-beamUniform", -10.0, -5.0, 0.0)
        assert len(ops3.calls) == 1

    def test_stock_refuses_partial_trapezoidal_beam_uniform(self) -> None:
        ops2 = _StockOps()
        le2 = _live_emitter(ops2, ndm=2)
        with pytest.raises(RuntimeError, match="requires the Ladruno fork"):
            le2.eleLoad(
                "-ele", 1, "-type", "-beamUniform", -10.0, 0.0, 0.2, 0.8, -20.0, 0.0,
            )
        assert ops2.calls == []

        ops3 = _StockOps()
        le3 = _live_emitter(ops3, ndm=3)
        with pytest.raises(RuntimeError, match="requires the Ladruno fork"):
            le3.eleLoad(
                "-ele", 1, "-type", "-beamUniform",
                -10.0, -5.0, 0.0, 0.2, 0.8, -20.0, -5.0, 0.0,
            )
        assert ops3.calls == []

    def test_stock_refuses_surface_load(self) -> None:
        ops = _StockOps()
        le = _live_emitter(ops, ndm=2)
        with pytest.raises(RuntimeError, match="requires the Ladruno fork"):
            le.eleLoad("-ele", 1, "-type", "-surfaceLoad")
        assert ops.calls == []

    def test_fork_allows_partial_beam_uniform_and_surface_load(self) -> None:
        ops = _CurrentForkOps()
        le = _live_emitter(ops, ndm=2)
        le.eleLoad(
            "-ele", 1, "-type", "-beamUniform", -10.0, 0.0, 0.2, 0.8, -20.0, 0.0,
        )
        le.eleLoad("-ele", 1, "-type", "-surfaceLoad")
        assert len(ops.calls) == 2


# ===========================================================================
# 6. Reviewer Regression Tests (1-based FEM IDs, Element Handles, Compose)
# ===========================================================================


class TestReviewerRegressions:
    def test_1_based_fem_eids_do_not_collide_with_shifted_ops_tags(self) -> None:
        """When FEM eids are [1, 2] and OpenSees tags are [2, 3], passing
        ele=[1, 2] (or a numpy int64 array, or an Element handle) must emit
        `-ele 2 3`, never `-ele 2 2`.
        """
        nodes = _NodesStub(
            ids=[1, 2, 3],
            coords=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)],
            node_pgs={"Base": [1, 3]},
        )
        elements = _ElementsStub(
            elem_pgs={
                "Beams": _ElementGroupView(
                    ids=(1, 2),
                    connectivity=((1, 2), (2, 3)),
                ),
            },
        )
        fem = FEMStub(nodes=nodes, elements=elements)
        ops = apeSees(cast("object", fem))
        ops.model(ndm=2, ndf=3)
        t = ops.geomTransf.Linear()
        beam_el = ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.01, E=200e9, Iz=1e-4,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.beam_uniform(ele=[1, 2], wy=-10.0)
            p.beam_uniform(ele=np.array([1, 2], dtype=np.int64), wy=-20.0)
            p.beam_uniform(ele=np.int64(2), wy=-30.0)
            p.beam_uniform(ele=beam_el, wy=-40.0)

        rec = RecordingEmitter()
        ops.build().emit(rec)
        t_e1, t_e2 = _emitted_elem_tags(rec)
        assert (t_e1, t_e2) == (2, 3)
        ele_loads = [c[1] for c in rec.calls if c[0] == "eleLoad"]
        assert ele_loads == [
            ("-ele", 2, 3, "-type", "-beamUniform", -10.0, 0.0),
            ("-ele", 2, 3, "-type", "-beamUniform", -20.0, 0.0),
            ("-ele", 3, "-type", "-beamUniform", -30.0, 0.0),
            ("-ele", 2, 3, "-type", "-beamUniform", -40.0, 0.0),
        ]

    def test_unregistered_ele_id_and_element_spec_raise_bridge_error(self) -> None:
        fem = _beam_fem_data(ndm=2)
        ops = apeSees(fem)
        ops.model(ndm=2, ndf=3)
        t = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(
            pg="Beams", transf=t, A=0.01, E=200e9, Iz=1e-4,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.beam_uniform(ele=999, wy=-10.0)
        with pytest.raises(BridgeError, match="FEM element id 999"):
            ops.build().emit(RecordingEmitter())

    def test_malformed_q_xyz_rejected(self) -> None:
        g = _DummyGeometry()
        lc = LoadsComposite(g)
        with pytest.raises(ValueError, match="q_xyz"):
            lc.line("beam", q_xyz=(1.0, 2.0), target_form="element")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="q_xyz"):
            lc.line(
                "beam",
                q_xyz=((1.0, 2.0), (3.0, 4.0)),  # type: ignore[arg-type]
                target_form="element",
            )

    def test_compose_rotates_trapezoidal_beam_uniform_params(self) -> None:
        from apeGmsh.mesh._compose import _place_load_record

        rec = ElementLoadRecord(
            element_id=10,
            load_type="beamUniform",
            params={
                "wx": 1.0,
                "wy": 2.0,
                "wz": 3.0,
                "wx_end": 4.0,
                "wy_end": 5.0,
                "wz_end": 6.0,
                "aL": 0.2,
                "bL": 0.8,
            },
            pattern="dl",
        )
        # 90 deg rotation about X: (x, y, z) -> (x, -z, y)
        rotated = _place_load_record(
            rec, rotate=(1.0, 0.0, 0.0, math.pi / 2.0),
        )
        assert rotated.params["aL"] == 0.2
        assert rotated.params["bL"] == 0.8
        assert (
            rotated.params["wx"],
            rotated.params["wy"],
            rotated.params["wz"],
        ) == pytest.approx((1.0, -3.0, 2.0), abs=1e-12)
        assert (
            rotated.params["wx_end"],
            rotated.params["wy_end"],
            rotated.params["wz_end"],
        ) == pytest.approx((4.0, -6.0, 5.0), abs=1e-12)

    @pytest.mark.parametrize("beam_cls", ["LadrunoDispBeamColumn", "LadrunoIMKBeam"])
    def test_from_model_3d_line_load_projects_on_ladruno_beams(
        self, beam_cls: str,
    ) -> None:
        loads = [
            ElementLoadRecord(
                element_id=10,
                load_type="beamUniform",
                params={"wx": 0.0, "wy": 0.0, "wz": -15.0, "coord_sys": "global"},
                pattern="dl",
            ),
        ]
        fem = _beam_fem_data(ndm=3, loads=loads)
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        if beam_cls == "LadrunoDispBeamColumn":
            sec = ops.section.Elastic(E=200e9, A=0.01, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-5)
            bi = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
            ops.element.LadrunoDispBeamColumn(pg="Beams", transf=t, integration=bi)
        else:
            mat = ops.uniaxialMaterial.Steel01(fy=250e6, E=200e9, b=0.01)
            ops.element.LadrunoIMKBeam(
                pg="Beams", transf=t, A=0.01, E=200e9, Iz=1e-4, G=80e9, Jx=1e-5, Iy=1e-4,
                mat_z=mat, mat_y=mat,
            )
        ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model("dl")

        rec = RecordingEmitter()
        ops.build().emit(rec)
        ele_loads = [c[1] for c in rec.calls if c[0] == "eleLoad"]
        assert len(ele_loads) == 1
        # With element along +X and vecxz=(0, 0, 1), local z = +Z, so global Fz=-15 -> wz=-15
        assert ele_loads[0][:5] == ("-ele", 2, "-type", "-beamUniform", pytest.approx(0.0))
        assert ele_loads[0][5] == pytest.approx(-15.0)
        assert ele_loads[0][6] == pytest.approx(0.0)

