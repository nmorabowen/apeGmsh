"""Consistent unit-system declaration and propagation (Slice U1).

Covers:
* ``resolve_unit_system`` for canonical IDs, ``baseUnits.BASE`` strings,
  ``baseUnits.systems`` modules, and ``baseUnits.make_system`` namespaces;
* ``UnitNotDeclaredWarning`` on ``apeGmsh(...)`` when ``units=`` is omitted,
  and silence when ``units=`` is given or ``_artifacts=False``;
* propagation through ``apeGmsh`` -> ``FEMData`` -> ``model.h5`` (``/meta/@units``)
  -> ``OpenSeesModel`` -> ``stamp_provenance`` (``INFO/PROVENANCE@units``)
  -> ``LadrunoReader``;
* hash invariance (``snapshot_id``, ``fem_hash``, ``model_hash`` do not depend
  on ``units``);
* ``compose`` unit-mismatch guard and ``apeSees`` unit-conflict guard.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apeGmsh import (
     FEMData,
    UNIT_SYSTEMS,
    UnitNotDeclaredWarning,
    apeGmsh,
    resolve_unit_system,
)
from apeGmsh.mesh._compose import ComposeError
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import ElementComposite, MeshInfo, NodeComposite
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.writers import stamp_provenance
from apeGmsh.units import (
    UNIT_SYSTEM_IDS,
    classify_field_dimension,
    conversion_factor,
    resolve_field_units,
    unit_symbol_for,
)

LADRUNO_FIXTURES = Path(__file__).parent / "fixtures" / "ladruno"


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


def test_resolve_unit_system_canonical_and_aliases() -> None:
    assert resolve_unit_system(None) is None
    for uid in UNIT_SYSTEM_IDS:
        assert resolve_unit_system(uid) == uid
        spec = UNIT_SYSTEMS[uid]
        assert resolve_unit_system(spec) == uid
        assert resolve_unit_system(spec.base_label) == uid

    assert resolve_unit_system("SI") == "N_m_s"
    assert resolve_unit_system("kip-in-s") == "kip_in_s"
    assert resolve_unit_system("kip-inches-s") == "kip_in_s"
    assert resolve_unit_system("dyne-cm-gram-s") == "dyne_cm_s"


def test_resolve_unit_system_from_baseunits_modules_and_namespaces() -> None:
    import importlib

    pytest.importorskip("baseUnits")
    from baseUnits._make_system import make_system as make_sys
    expected_ids = (
        "N_m_s",
        "kN_m_s",
        "tf_m_s",
        "kgf_m_s",
        "N_mm_s",
        "kip_in_s",
        "dyne_cm_s",
    )
    found = 0
    for uid in expected_ids:
        try:
            mod = importlib.import_module(f"baseUnits.systems.{uid}")
        except ModuleNotFoundError:
            continue
        assert resolve_unit_system(mod) == uid
        found += 1
    assert found >= 5

    assert resolve_unit_system(make_sys(force="kN", length="m", time="s")) == "kN_m_s"
    assert resolve_unit_system(make_sys(force="N", length="mm", time="s")) == "N_mm_s"
    assert resolve_unit_system(make_sys(force="kip", length="inches", time="s")) == "kip_in_s"


def test_resolve_unit_system_rejects_invalid_and_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown unit system"):
        resolve_unit_system(" furlong_firkin_fortnight ")

    with pytest.raises(TypeError, match="expected a canonical unit system"):
        resolve_unit_system(12345)  # type: ignore[arg-type]

    inconsistent = SimpleNamespace(BASE="kN-m-tonne-s", m=1.0, s=1.0, N=1e-3, kg=999.0)
    with pytest.raises(ValueError, match="Inconsistent unit system"):
        resolve_unit_system(inconsistent)


def test_unit_not_declared_warning_contract(tmp_path: Path) -> None:
    with pytest.warns(UnitNotDeclaredWarning, match="no consistent unit system declared"):
        g = apeGmsh(model_name="warn_model", save_to=tmp_path / "w.h5")
    assert g.units is None

    # Silent when units= is declared (verified under -W error::UnitNotDeclaredWarning)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnitNotDeclaredWarning)
        g_ok = apeGmsh(model_name="ok_model", units="kN_m_s", save_to=tmp_path / "ok.h5")
        assert g_ok.units == "kN_m_s"
        g_internal = apeGmsh(model_name="internal", _artifacts=False)
        assert g_internal.units is None


def test_session_to_femdata_to_h5_and_back(tmp_path: Path) -> None:
    h5_path = tmp_path / "beam.h5"
    with apeGmsh(model_name="beam", units="kN_m_s", save_to=h5_path, verbose=False) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 2.0, 0.5, 0.5, label="box")
        g.physical.add_volume("box", name="Body")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        assert fem.units == "kN_m_s"

    with h5py.File(h5_path, "r") as f:
        assert f["meta"].attrs["units"] == "kN_m_s"

    reloaded = FEMData.from_h5(str(h5_path))
    assert reloaded.units == "kN_m_s"

    chain_session = apeGmsh.from_h5(h5_path)
    assert chain_session.units == "kN_m_s"
    assert chain_session.mesh.queries.get_fem_data().units == "kN_m_s"


def test_undeclared_units_writes_no_meta_attr_and_corrupt_attr_raises(tmp_path: Path) -> None:
    fem = _tiny_fem(units=None)
    p = tmp_path / "no_units.h5"
    fem.to_h5(str(p))
    with h5py.File(p, "r") as f:
        assert "units" not in f["meta"].attrs
    assert FEMData.from_h5(str(p)).units is None

    bad = tmp_path / "bad_units.h5"
    shutil.copy(p, bad)
    with h5py.File(bad, "r+") as f:
        f["meta"].attrs["units"] = "not_a_unit_system"
    with pytest.raises(MalformedH5Error, match="/meta/units is invalid"):
        FEMData.from_h5(str(bad))


def test_units_do_not_affect_snapshot_id_or_lineage_hashes(tmp_path: Path) -> None:
    fem_none = _tiny_fem(units=None)
    fem_kn = fem_none.with_units("kN_m_s")
    fem_mm = fem_none.with_units("N_mm_s")

    assert fem_none.snapshot_id == fem_kn.snapshot_id == fem_mm.snapshot_id

    p_none = tmp_path / "none.h5"
    p_kn = tmp_path / "kn.h5"
    for fem, out in ((fem_none, p_none), (fem_kn, p_kn)):
        ops = apeSees(fem, default_orientation=None, _artifacts=False)
        ops.model(ndm=2, ndf=2)
        mat = ops.uniaxialMaterial.ElasticMaterial(E=200e6)
        ops.element.Truss(pg="Bars", A=0.01, material=mat)
        ops.fix(pg="Base", dofs=(1, 1))
        ops.h5(str(out))

    with h5py.File(p_none, "r") as f0, h5py.File(p_kn, "r") as f1:
        assert "units" not in f0["meta"].attrs
        assert f1["meta"].attrs["units"] == "kN_m_s"
        for k in ("fem_hash", "model_hash"):
            assert f0["meta/lineage"].attrs[k] == f1["meta/lineage"].attrs[k]

    model_kn = OpenSeesModel.from_h5(p_kn)
    assert model_kn.units == "kN_m_s"
    assert model_kn.fem.units == "kN_m_s"


def test_apesees_units_inheritance_override_and_conflict(tmp_path: Path) -> None:
    fem_kn = _tiny_fem(units="kN_m_s")
    ops = apeSees(fem_kn, default_orientation=None, _artifacts=False)
    assert ops.units == "kN_m_s"

    # Matching units= is allowed
    ops_same = apeSees(fem_kn, units="kN_m_s", default_orientation=None, _artifacts=False)
    assert ops_same.units == "kN_m_s"

    # Conflicting units= raises ValueError
    with pytest.raises(ValueError, match="conflict with fem.units"):
        apeSees(fem_kn, units="N_mm_s", default_orientation=None, _artifacts=False)

    # Supplying units= when fem.units is None stamps it onto ops.fem and model.h5
    fem_none = _tiny_fem(units=None)
    ops_override = apeSees(fem_none, units="N_mm_s", default_orientation=None, _artifacts=False)
    assert ops_override.units == "N_mm_s"
    assert ops_override.fem.units == "N_mm_s"
    out = tmp_path / "override.h5"
    ops_override.model(ndm=2, ndf=2)
    ops_override.h5(str(out))
    assert OpenSeesModel.from_h5(out).units == "N_mm_s"


def test_compose_preserves_units_and_refuses_mismatch(tmp_path: Path) -> None:
    from apeGmsh.assembly import Assembly

    p_kn1 = tmp_path / "m1.h5"
    p_kn2 = tmp_path / "m2.h5"
    p_mm = tmp_path / "mm.h5"
    for fem, path in (
        (_tiny_fem("kN_m_s"), p_kn1),
        (_tiny_fem("kN_m_s"), p_kn2),
        (_tiny_fem("N_mm_s"), p_mm),
    ):
        ops = apeSees(fem, default_orientation=None, _artifacts=False)
        ops.model(ndm=2, ndf=3)
        tr = ops.geomTransf.Linear()
        ops.element.elasticBeamColumn(pg="Bars", A=0.01, E=200e6, Iz=1e-4, transf=tr)
        ops.h5(str(path))

    asm = Assembly("ok")
    asm.instance("a", p_kn1)
    asm.instance("b", p_kn2, translate=(5.0, 0.0, 0.0))
    merged_fem = asm.fem()
    assert merged_fem.units == "kN_m_s"
    built_ops = asm.bridge(ndm=2, ndf=3)
    assert built_ops.fem.units == "kN_m_s"
    assert built_ops.units == "kN_m_s"

    asm_bad = Assembly("bad")
    asm_bad.instance("a", p_kn1)
    asm_bad.instance("b", p_mm, translate=(5.0, 0.0, 0.0))
    with pytest.raises(ComposeError, match="unit system mismatch"):
        asm_bad.fem()


def test_stamp_provenance_writes_units_to_ladruno_and_reader_reads_it(tmp_path: Path) -> None:
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

    with h5py.File(ladruno_path, "r") as f:
        assert f["INFO/PROVENANCE"].attrs["units"] == "kN_m_s"

    with LadrunoReader(ladruno_path) as reader:
        assert reader.units == "kN_m_s"
        fem = reader.fem()
        assert fem is not None and fem.units == "kN_m_s"


def test_dimension_classification_and_conversion_factors() -> None:
    assert classify_field_dimension("displacement_x").kind == "length"
    assert classify_field_dimension("rotation_z").kind == "angle"
    assert classify_field_dimension("velocity_y").kind == "velocity"
    assert classify_field_dimension("acceleration_z").kind == "acceleration"
    assert classify_field_dimension("reaction_x").kind == "force"
    assert classify_field_dimension("reaction_mz").kind == "moment"
    assert classify_field_dimension("stress_xx").kind == "pressure"
    assert classify_field_dimension("von_mises").kind == "pressure"
    assert classify_field_dimension("strain_xx").kind == "dimensionless"
    assert classify_field_dimension("N", level="line").kind == "force"
    assert classify_field_dimension("Mz", level="line").kind == "moment"
    assert classify_field_dimension("kappa_z", level="line").kind == "curvature"
    assert classify_field_dimension("F11", level="gauss").kind == "shell_force_per_length"
    assert classify_field_dimension("M11", level="gauss").kind == "shell_moment_per_length"
    assert classify_field_dimension("kinetic", level="energy").kind == "energy"

    assert pytest.approx(conversion_factor("N_m_s", "N_mm_s", "length")) == 1000.0
    assert pytest.approx(conversion_factor("kN_m_s", "N_m_s", "force")) == 1000.0
    assert pytest.approx(conversion_factor("N_m_s", "N_mm_s", "pressure")) == 1e-6
    assert pytest.approx(conversion_factor("N_m_s", "N_mm_s", "curvature")) == 1e-3
    assert unit_symbol_for("pressure", "kN_m_s") == "kPa"
    res = resolve_field_units("stress_xx", "element", "kN_m_s", "N_mm_s")
    assert pytest.approx(res.factor) == 1e-3
    assert res.unit == "MPa"

    spec = UNIT_SYSTEMS["kN_m_s"]
    assert spec.BASE == "kN-m-tonne-s"
    assert pytest.approx(spec.N) == 1e-3
    assert pytest.approx(spec.m) == 1.0
    assert pytest.approx(spec.s) == 1.0
    assert pytest.approx(spec.kg) == 1e-3


def test_femdata_units_setter_and_from_h5_conflict(tmp_path: Path) -> None:
    fem = _tiny_fem(None)
    fem.units = "kN-m-tonne-s"
    assert fem.units == "kN_m_s"
    with pytest.raises(ValueError, match="Unknown unit system"):
        fem.units = "bogus_units"

    h5_path = tmp_path / "m_kn.h5"
    fem.to_h5(str(h5_path))
    with pytest.raises(ValueError, match="conflicts with"):
        apeGmsh.from_h5(h5_path, units="N_mm_s")

