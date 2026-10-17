"""Tests for Results unit awareness and unit-system translation (Slice U2)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from numpy import ndarray

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.results import Results
from apeGmsh.results._slabs import (
    ElementSlab,
    FiberSlab,
    GaussSlab,
    LayerSlab,
    LineStationSlab,
    NodeSlab,
    SpringSlab,
)
from apeGmsh.results.readers._ladruno import NodeEnvelope
from apeGmsh.results.readers._protocol import ResultLevel, StageInfo, TimeSlice
from apeGmsh.results.writers import stamp_provenance

LADRUNO_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno"


def _tiny_fem(units: str | None = None) -> FEMData:
    node_ids = np.array([1, 2, 3], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)
    group = ElementGroup(
        element_type=info,
        ids=np.array([2, 3], dtype=np.int64),
        connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64),
    )
    pg = {
        (1, 1): {
            "name": "Bars",
            "node_ids": node_ids,
            "node_coords": coords,
            "element_ids": group.ids,
        },
        (0, 2): {
            "name": "Base",
            "node_ids": node_ids[:1],
            "node_coords": coords[:1],
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet(pg),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: group},
        physical=PhysicalGroupSet(pg),
        labels=LabelSet({}),
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=MeshInfo(n_nodes=3, n_elems=2, bandwidth=2, types=[info]),
        units=units,
    )


def _make_model(fem: FEMData, tmp_path: Path, name: str = "m.h5") -> OpenSeesModel:
    ops = apeSees(fem, default_orientation=None, _artifacts=False)
    ops.model(ndm=2, ndf=3)
    p = tmp_path / name
    ops.h5(str(p))
    return OpenSeesModel.from_h5(p)


class _SyntheticMultiSlabReader:
    """In-memory ResultsReader covering all slab types, modes, energy, and envelopes."""

    def __init__(self, fem: FEMData, *, units: str | None = None) -> None:
        self._fem = fem
        self.units = units
        self._time = np.array([0.0, 1.0], dtype=np.float64)

    def stages(self) -> list[StageInfo]:
        return [
            StageInfo(id="stage_0", name="push", kind="static", n_steps=2),
            StageInfo(
                id="mode_1",
                name="mode_1",
                kind="mode",
                n_steps=1,
                eigenvalue=4.0 * np.pi**2,
                frequency_hz=1.0,
                period_s=1.0,
                mode_index=1,
            ),
        ]

    def time_vector(self, stage_id: str) -> ndarray:
        if stage_id == "mode_1":
            return np.array([0.0], dtype=np.float64)
        return self._time

    def partitions(self, stage_id: str) -> list[str]:
        return ["0"]

    def fem(self) -> Optional[FEMData]:
        return self._fem

    def opensees_model(self) -> None:
        return None

    def available_components(
        self, stage_id: str, level: ResultLevel,
    ) -> list[str]:
        if stage_id == "mode_1":
            return ["displacement_x"] if level == ResultLevel.NODES else []
        mapping = {
            ResultLevel.NODES: [
                "displacement_x",
                "rotation_z",
                "velocity_x",
                "acceleration_x",
                "reaction_x",
                "reaction_mz",
            ],
            ResultLevel.ELEMENTS: ["force_x", "moment_z"],
            ResultLevel.LINE_STATIONS: ["N", "Mz", "kappa_z"],
            ResultLevel.GAUSS: ["stress_xx", "strain_xx", "F11", "M11"],
            ResultLevel.FIBERS: ["stress", "strain"],
            ResultLevel.LAYERS: ["stress_xx", "strain_xx"],
            ResultLevel.SPRINGS: ["spring_force_0", "spring_deformation_0"],
        }
        return mapping.get(level, [])

    def read_nodes(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> NodeSlab:
        ids = np.array([1, 2, 3], dtype=np.int64)
        if stage_id == "mode_1":
            vals = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
            t = np.array([0.0], dtype=np.float64)
        else:
            vals = np.array([[0.0, 0.01, 0.02], [0.0, 0.02, 0.04]], dtype=np.float64)
            t = self._time
        return NodeSlab(component=component, values=vals, node_ids=ids, time=t)

    def read_elements(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> ElementSlab:
        eids = np.array([2, 3], dtype=np.int64)
        vals = np.ones((2, 2, 2), dtype=np.float64) * 10.0
        return ElementSlab(component=component, values=vals, element_ids=eids, time=self._time)

    def read_line_stations(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LineStationSlab:
        eidx = np.array([2, 2, 3, 3], dtype=np.int64)
        xi = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
        vals = np.ones((2, 4), dtype=np.float64) * 5.0
        return LineStationSlab(
            component=component,
            values=vals,
            element_index=eidx,
            station_natural_coord=xi,
            time=self._time,
        )

    def read_gauss(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> GaussSlab:
        eidx = np.array([2, 3], dtype=np.int64)
        nat = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        vals = np.ones((2, 2), dtype=np.float64) * 2000.0
        return GaussSlab(
            component=component,
            values=vals,
            element_index=eidx,
            natural_coords=nat,
            local_axes_quaternion=None,
            time=self._time,
        )

    def read_fibers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> FiberSlab:
        return FiberSlab(
            component=component,
            values=np.ones((2, 2), dtype=np.float64) * 250000.0,
            element_index=np.array([2, 2], dtype=np.int64),
            gp_index=np.array([0, 0], dtype=np.int64),
            y=np.array([-0.2, 0.2], dtype=np.float64),
            z=np.array([0.0, 0.0], dtype=np.float64),
            area=np.array([0.001, 0.001], dtype=np.float64),
            material_tag=np.array([1, 1], dtype=np.int64),
            time=self._time,
            station_natural_coord=np.array([-0.5, -0.5], dtype=np.float64),
        )

    def read_layers(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        gp_indices: Optional[ndarray] = None,
        layer_indices: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> LayerSlab:
        return LayerSlab(
            component=component,
            values=np.ones((2, 2), dtype=np.float64) * 15000.0,
            element_index=np.array([2, 3], dtype=np.int64),
            gp_index=np.array([0, 0], dtype=np.int64),
            layer_index=np.array([0, 1], dtype=np.int64),
            sub_gp_index=np.array([0, 0], dtype=np.int64),
            thickness=np.array([0.05, 0.05], dtype=np.float64),
            local_axes_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
            time=self._time,
        )

    def read_springs(
        self,
        stage_id: str,
        component: str,
        *,
        element_ids: Optional[ndarray] = None,
        time_slice: TimeSlice = None,
    ) -> SpringSlab:
        return SpringSlab(
            component=component,
            values=np.ones((2, 2), dtype=np.float64) * 3.0,
            element_index=np.array([2, 3], dtype=np.int64),
            time=self._time,
        )

    def read_energy(
        self,
        stage_id: str,
        *,
        region: Optional[int] = None,
        time_slice: TimeSlice = None,
    ) -> tuple[list[str], ndarray, ndarray]:
        cols = ["KE", "IE", "DW", "ULW", "RES", "ERR"]
        vals = np.array(
            [[10.0, 20.0, 5.0, 35.0, 0.01, 0.25], [12.0, 24.0, 6.0, 42.0, 0.02, 0.5]],
            dtype=np.float64,
        )
        return cols, vals, self._time

    def read_node_envelope(
        self,
        stage_id: str,
        component: str,
        *,
        node_ids: Optional[ndarray] = None,
    ) -> NodeEnvelope:
        return NodeEnvelope(
            node_ids=np.array([1, 2, 3], dtype=np.int64),
            min=np.array([0.0, -0.01, -0.02], dtype=np.float64),
            max=np.array([0.0, 0.02, 0.04], dtype=np.float64),
            absmax=np.array([0.0, 0.02, 0.04], dtype=np.float64),
            arg_step=np.array([0, 1, 1], dtype=np.int64),
        )


def test_declared_results_translates_without_source_and_scales_all_slabs(tmp_path: Path) -> None:
    fem = _tiny_fem(units="kN_m_s")
    model = _make_model(fem, tmp_path)
    reader = _SyntheticMultiSlabReader(fem, units="kN_m_s")
    res = Results(reader, fem=fem, model=model).stage("push")

    assert res.units == "kN_m_s"
    assert res.source_units == "kN_m_s"
    assert res.unit_symbol("displacement_x") == "m"
    assert res.unit_symbol("stress_xx", level="gauss") == "kPa"
    assert res.unit_symbol("strain_xx", level="gauss") == ""
    assert pytest.approx(res.unit_factor("displacement_x", target="N_mm_s")) == 1000.0
    assert pytest.approx(res.unit_factor("stress_xx", level="gauss", target="N_mm_s")) == 1e-3

    # Translate kN_m_s -> N_mm_s without needing source=
    res_mm = res.to_units("N_mm_s")
    assert res_mm.units == "N_mm_s"
    assert res_mm.source_units == "kN_m_s"
    assert res_mm.unit_symbol("displacement_x") == "mm"
    assert res_mm.unit_symbol("stress_xx", level="gauss") == "MPa"
    assert res_mm.unit_symbol("Mz", level="line") == "N·mm"

    # Bound FEM coordinates scaled m -> mm (x 1000)
    assert res_mm.fem is not None
    assert res_mm.fem.units == "N_mm_s"
    np.testing.assert_allclose(res_mm.fem.nodes.coords[:, 0], [0.0, 1000.0, 2000.0])

    # Node slabs: displacement (m -> mm: x1000), rotation (rad: x1), reaction_mz (kN*m -> N*mm: x1e6)
    u_mm = res_mm.nodes.get(component="displacement_x")
    np.testing.assert_allclose(u_mm.values, [[0.0, 10.0, 20.0], [0.0, 20.0, 40.0]])
    rot = res_mm.nodes.get(component="rotation_z")
    np.testing.assert_allclose(rot.values, [[0.0, 0.01, 0.02], [0.0, 0.02, 0.04]])
    rmz = res_mm.nodes.get(component="reaction_mz")
    np.testing.assert_allclose(rmz.values, np.array([[0.0, 0.01, 0.02], [0.0, 0.02, 0.04]]) * 1e6)

    # Element slabs: force_x (kN -> N: x1000), moment_z (kN*m -> N*mm: x1e6)
    ef = res_mm._reader.read_elements("stage_0", "force_x")
    np.testing.assert_allclose(ef.values, 10.0 * 1000.0)
    em = res_mm._reader.read_elements("stage_0", "moment_z")
    np.testing.assert_allclose(em.values, 10.0 * 1e6)

    # Line stations: N (x1000), Mz (x1e6), kappa_z (1/m -> 1/mm: x1e-3), xi unscaled
    ln_n = res_mm._reader.read_line_stations("stage_0", "N")
    np.testing.assert_allclose(ln_n.values, 5.0 * 1000.0)
    np.testing.assert_allclose(ln_n.station_natural_coord, [-1.0, 1.0, -1.0, 1.0])
    ln_k = res_mm._reader.read_line_stations("stage_0", "kappa_z")
    np.testing.assert_allclose(ln_k.values, 5.0 * 1e-3)

    # Gauss slabs: stress_xx (kPa -> MPa: x1e-3), strain_xx (x1), F11 (kN/m -> N/mm: x1), M11 (kN*m/m -> N*mm/mm: x1000)
    g_sig = res_mm._reader.read_gauss("stage_0", "stress_xx")
    np.testing.assert_allclose(g_sig.values, 2.0)
    g_eps = res_mm._reader.read_gauss("stage_0", "strain_xx")
    np.testing.assert_allclose(g_eps.values, 2000.0)
    g_f11 = res_mm._reader.read_gauss("stage_0", "F11")
    np.testing.assert_allclose(g_f11.values, 2000.0 * 1.0)
    g_m11 = res_mm._reader.read_gauss("stage_0", "M11")
    np.testing.assert_allclose(g_m11.values, 2000.0 * 1000.0)

    # Fibers: stress (kPa -> MPa: x1e-3), y/z (m -> mm: x1000), area (m^2 -> mm^2: x1e6)
    fib = res_mm._reader.read_fibers("stage_0", "stress")
    np.testing.assert_allclose(fib.values, 250.0)
    np.testing.assert_allclose(fib.y, [-200.0, 200.0])
    np.testing.assert_allclose(fib.area, [1000.0, 1000.0])

    # Layers: stress_xx (x1e-3), thickness (m -> mm: x1000)
    lay = res_mm._reader.read_layers("stage_0", "stress_xx")
    np.testing.assert_allclose(lay.values, 15.0)
    np.testing.assert_allclose(lay.thickness, [50.0, 50.0])

    # Springs: spring_force_0 (x1000), spring_deformation_0 (x1000)
    sp_f = res_mm._reader.read_springs("stage_0", "spring_force_0")
    np.testing.assert_allclose(sp_f.values, 3000.0)
    sp_u = res_mm._reader.read_springs("stage_0", "spring_deformation_0")
    np.testing.assert_allclose(sp_u.values, 3000.0)

    # Energy balance: KE/IE/DW/ULW/RES scaled by 1e6 (kN*m -> N*mm), ERR (%) unscaled
    df_e = res_mm.energy()
    np.testing.assert_allclose(df_e["KE"].to_numpy(), [10.0 * 1e6, 12.0 * 1e6])
    np.testing.assert_allclose(df_e["ERR"].to_numpy(), [0.25, 0.5])

    # Node envelope: min/max/absmax scaled by 1000, arg_step unscaled
    df_env = res_mm.node_envelope("displacement_x")
    np.testing.assert_allclose(df_env["max"].to_numpy(), [0.0, 20.0, 40.0])
    np.testing.assert_array_equal(df_env["arg_step"].to_numpy(), [0, 1, 1])

    # Mode stage: mode shape is dimensionless and unscaled
    top_mm = Results(reader, fem=fem, model=model).to_units("N_mm_s")
    mode = top_mm.modes[0]
    assert mode.frequency_hz == 1.0
    m_shape = mode.nodes.get(component="displacement_x")
    np.testing.assert_allclose(m_shape.values, [[0.0, 0.5, 1.0]])

    # Chaining to_units: res_mm.to_units("kN_m_s") returns back to original values
    res_back = res_mm.to_units("kN_m_s")
    assert res_back.units == "kN_m_s"
    assert res_back.source_units == "kN_m_s"
    np.testing.assert_allclose(
        res_back.nodes.get(component="displacement_x").values,
        [[0.0, 0.01, 0.02], [0.0, 0.02, 0.04]],
    )

    # Conflicting explicit source= on declared Results raises ValueError
    with pytest.raises(ValueError, match="conflicts with declared unit system"):
        res.to_units("N_mm_s", source="kip_in_s")


def test_undeclared_results_requires_explicit_source_for_translation(tmp_path: Path) -> None:
    fem = _tiny_fem(units=None)
    model = _make_model(fem, tmp_path, name="no_units.h5")
    reader = _SyntheticMultiSlabReader(fem, units=None)
    res = Results(reader, fem=fem, model=model).stage("push")

    assert res.units is None
    assert res.source_units is None
    assert res.unit_symbol("displacement_x") == ""
    assert res.unit_factor("displacement_x") == 1.0

    with pytest.raises(ValueError, match="no unit system was declared"):
        res.unit_factor("displacement_x", target="N_mm_s")

    with pytest.raises(ValueError, match="Pass source=<unit_system> explicitly"):
        res.to_units("N_mm_s")

    # Supplying source="kN_m_s" enables translation and records source_units
    res_mm = res.to_units("N_mm_s", source="kN_m_s")
    assert res_mm.units == "N_mm_s"
    assert res_mm.source_units == "kN_m_s"
    np.testing.assert_allclose(
        res_mm.nodes.get(component="displacement_x").values,
        [[0.0, 10.0, 20.0], [0.0, 20.0, 40.0]],
    )

    # Subsequent translation on res_mm does not need source= repeated
    res_si = res_mm.to_units("N_m_s")
    assert res_si.units == "N_m_s"
    assert res_si.source_units == "kN_m_s"
    np.testing.assert_allclose(
        res_si.nodes.get(component="reaction_x").values,
        np.array([[0.0, 0.01, 0.02], [0.0, 0.02, 0.04]]) * 1000.0,
    )


def test_from_ladruno_propagates_provenance_units_and_translates(tmp_path: Path) -> None:
    model_h5 = tmp_path / "truss_kn.h5"
    ops = apeSees(_tiny_fem("kN_m_s"), default_orientation=None, _artifacts=False)
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=2.0e8)
    ops.element.Truss(pg="Bars", A=1e-4, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="load") as s:
        with s.pattern(series=ops.timeSeries.Linear()) as p:
            p.load(pg="Bars", forces=(1.0, 0.0))
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=10),
            algorithm=ops.algorithm.Linear(),
            integrator=ops.integrator.LoadControl(dlam=0.25),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.BandGeneral(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=4)
    ops.h5(str(model_h5))

    ladruno_path = tmp_path / "truss2d.ladruno"
    shutil.copy(LADRUNO_FIXTURES / "truss2d.ladruno", ladruno_path)
    stamp_provenance(ladruno_path, model_h5)

    # Self-sufficient open (no model_h5=): reads units from INFO/PROVENANCE@units
    with Results.from_ladruno(ladruno_path) as r:
        assert r.units == "kN_m_s"
        assert r.source_units == "kN_m_s"
        assert not r._element_tag_pairing_missing()
        u_m = r.nodes.get(component="displacement_x").values
        r_mm = r.to_units("N_mm_s")
        assert not r_mm._element_tag_pairing_missing()
        assert r_mm.fem is not None
        assert r_mm.fem.elements._apegmsh_nodes_ref is r_mm.fem.nodes
        u_mm = r_mm.nodes.get(component="displacement_x").values
        np.testing.assert_allclose(u_mm, u_m * 1000.0)


def test_to_units_preserves_non_ladruno_typeerror_and_refuses_unit_conflicts(
    tmp_path: Path,
) -> None:
    from apeGmsh.results.writers import NativeWriter

    fem = _tiny_fem("kN_m_s")
    model = _make_model(fem, tmp_path, name="model_kn.h5")
    model_path = tmp_path / "model_kn.h5"
    res_path = tmp_path / "res_kn.h5"
    with NativeWriter(res_path) as w:
        w.open(fem=fem, source_type="test", model_h5_src=model_path)
        sid = w.begin_stage(name="push", kind="static", time=np.array([0.0, 1.0]))
        w.write_nodes(
            sid,
            "partition_0",
            node_ids=np.array([1, 2, 3], dtype=np.int64),
            components={
                "displacement_x": np.array([[0.0, 0.01, 0.02], [0.0, 0.02, 0.04]]),
            },
        )
        w.end_stage()

    with Results.from_native(res_path, model=model) as res:
        res_mm = res.to_units("N_mm_s")
        assert res_mm.fem is not None
        assert res_mm.fem.elements._apegmsh_nodes_ref is res_mm.fem.nodes
        assert res_mm._results_records_program_decl() == res._results_records_program_decl()
        with pytest.raises(TypeError, match="Ladruno-recorder feature"):
            res_mm.energy()
        with pytest.raises(TypeError, match="Ladruno-recorder feature"):
            res_mm.node_envelope("displacement_x")
        with pytest.raises(ValueError, match="Conflicting unit declarations"):
            res.bind(_tiny_fem("N_mm_s"))
        with pytest.raises(ValueError, match="Conflicting unit declarations"):
            res_mm.bind(_tiny_fem("kip_in_s"))
    with pytest.raises(ValueError, match="Conflicting unit declarations"):
        with Results.from_native(res_path, model=model, fem=_tiny_fem("N_mm_s")):
            pass



