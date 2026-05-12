"""Unit tests for :class:`H5Emitter`.

Tests focus on the buffered-state per Protocol method. The
write-to-disk side is covered separately by ``test_h5_schema_v1`` and
the fixture suite in :mod:`tests.opensees.h5.fixtures`.
"""
from __future__ import annotations

from apeGmsh.opensees.emitter.h5 import H5Emitter, SCHEMA_VERSION
from apeGmsh.opensees._internal.tag_resolution import set_element_nodes


def test_h5emitter_protocol_conformance() -> None:
    """H5Emitter must satisfy the frozen :class:`Emitter` Protocol."""
    from apeGmsh.opensees.emitter.base import Emitter
    e: Emitter = H5Emitter()
    assert e is not None


def test_h5emitter_initial_state() -> None:
    e = H5Emitter()
    meta = e._meta_attrs()
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["ndm"] == 0
    assert meta["ndf"] == 0
    assert meta["model_name"] == "model"


def test_h5emitter_constructor_overrides() -> None:
    e = H5Emitter(
        schema_version="1.2.3",
        model_name="cantilever",
        apegmsh_version="0.99.0-test",
        snapshot_id="abc123",
    )
    meta = e._meta_attrs()
    assert meta["schema_version"] == "1.2.3"
    assert meta["model_name"] == "cantilever"
    assert meta["apeGmsh_version"] == "0.99.0-test"
    assert meta["snapshot_id"] == "abc123"


def test_h5emitter_model_sets_dimensionality() -> None:
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    meta = e._meta_attrs()
    assert meta["ndm"] == 3
    assert meta["ndf"] == 6


def test_h5emitter_node_records_coords() -> None:
    e = H5Emitter()
    e.node(1, 0.0, 0.0, 0.0)
    e.node(2, 1.0, 2.0, 3.0)
    assert e._node_tags == [1, 2]
    assert e._node_coords == [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]


def test_h5emitter_node_2d_coords_pad_z_zero() -> None:
    e = H5Emitter()
    e.node(1, 0.0, 0.0)
    assert e._node_coords == [(0.0, 0.0, 0.0)]


def test_h5emitter_fix_and_mass() -> None:
    e = H5Emitter()
    e.fix(1, 1, 1, 1, 0, 0, 0)
    e.mass(2, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert len(e._fixes) == 1
    assert e._fixes[0].tag == 1
    assert e._fixes[0].dofs == (1, 1, 1, 0, 0, 0)
    assert e._masses[0].values == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def test_h5emitter_uniaxial_material() -> None:
    e = H5Emitter()
    e.uniaxialMaterial("Steel02", 1, 420.0e6, 200.0e9, 0.01)
    assert e._uniaxial[0].type_token == "Steel02"
    assert e._uniaxial[0].tag == 1
    assert e._uniaxial[0].params == (420.0e6, 200.0e9, 0.01)


def test_h5emitter_section_simple() -> None:
    e = H5Emitter()
    e.section("ElasticMembranePlateSection", 7, 30.0e9, 0.2, 0.20, 2400.0)
    assert e._sections_simple[0].tag == 7


def test_h5emitter_section_open_close_with_patch_fiber() -> None:
    e = H5Emitter()
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.patch("rect", 2, 8, 8, -0.2, -0.2, 0.2, 0.2)
    e.fiber(0.1, 0.0, 0.001, 1)
    e.section_close()
    assert e._open_section is None
    assert len(e._sections_complex) == 1
    sec = e._sections_complex[0]
    assert sec.type_token == "Fiber"
    assert sec.tag == 1
    assert sec.params == ("-GJ", 1.0e9)
    assert len(sec.patches) == 1
    assert sec.patches[0].kind == "rect"
    assert sec.patches[0].args == (2, 8, 8, -0.2, -0.2, 0.2, 0.2)
    assert len(sec.fibers) == 1
    assert sec.fibers[0].mat_tag == 1


def test_h5emitter_geomtransf_records_each_call() -> None:
    e = H5Emitter()
    e.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    e.geomTransf("Linear", 2, 1.0, 0.0, 0.0)
    assert len(e._transforms) == 2
    assert e._transforms[0].tag == 1
    assert e._transforms[1].vec == (1.0, 0.0, 0.0)


def test_h5emitter_beam_integration_records_call() -> None:
    e = H5Emitter()
    e.beamIntegration("Lobatto", 1, 5, 5)
    assert e._beam_integrations[0].type_token == "Lobatto"
    assert e._beam_integrations[0].args == (5, 5)


def test_h5emitter_element_captures_connectivity_from_sidechannel() -> None:
    e = H5Emitter()
    set_element_nodes(e, (10, 11))
    e.element("forceBeamColumn", 5, 10, 11, 1, 1)
    assert e._elements[0].connectivity == (10, 11)
    assert e._elements[0].args == (10, 11, 1, 1)


def test_h5emitter_element_without_connectivity_context_uses_empty_tuple() -> None:
    e = H5Emitter()
    e.element("FourNodeTetrahedron", 1, 1, 2, 3, 4)
    assert e._elements[0].connectivity == ()


def test_h5emitter_time_series() -> None:
    e = H5Emitter()
    e.timeSeries("Linear", 1, "-factor", 1.0)
    assert e._time_series[0].type_token == "Linear"


def test_h5emitter_pattern_open_close_collects_loads() -> None:
    e = H5Emitter()
    e.pattern_open("Plain", 1, 1)
    e.load(10, 1.0, 0.0, 0.0)
    e.sp(20, 1, 0.0)
    e.eleLoad("-ele", 5, "-type", "-beamUniform", 0.0, -10.0)
    e.pattern_close()
    assert len(e._patterns_complete) == 1
    pat = e._patterns_complete[0]
    assert pat.type_token == "Plain"
    assert len(pat.loads) == 1
    assert pat.loads[0].forces == (1.0, 0.0, 0.0)
    assert len(pat.sps) == 1
    assert pat.sps[0].dof == 1
    assert len(pat.ele_loads) == 1


def test_h5emitter_pattern_singleline_uniformexcitation() -> None:
    e = H5Emitter()
    e.pattern_open("UniformExcitation", 2, 1, "-accel", 1)
    # The bridge does NOT call pattern_close after a single-line pattern;
    # H5Emitter must still capture it. We close explicitly to flush.
    e.pattern_close()
    assert len(e._patterns_complete) == 1
    pat = e._patterns_complete[0]
    assert pat.type_token == "UniformExcitation"


def test_h5emitter_recorder_records_call() -> None:
    e = H5Emitter()
    e.recorder("Node", "-file", "disp.out", "-node", 1, "-dof", 1, "disp")
    assert e._recorders[0].kind == "Node"


def test_h5emitter_analysis_chain() -> None:
    e = H5Emitter()
    e.constraints("Transformation")
    e.numberer("RCM")
    e.system("BandGeneral")
    e.test("NormDispIncr", 1.0e-6, 10)
    e.algorithm("Newton")
    e.integrator("LoadControl", 0.05)
    e.analysis("Static")
    rc = e.analyze(steps=20)
    assert rc == 0
    assert e._analysis_attrs["handler"] == "Transformation"
    assert e._analysis_attrs["numberer"] == "RCM"
    assert e._analysis_attrs["system"] == "BandGeneral"
    assert e._analysis_attrs["test"] == "NormDispIncr"
    assert e._analysis_attrs["algorithm"] == "Newton"
    assert e._analysis_attrs["integrator"] == "LoadControl"
    assert e._analysis_attrs["analysis"] == "Static"
    assert e._analyze_call == (20, None)


def test_h5emitter_write_meta_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Step 2: /meta is written with the schema attrs."""
    import h5py
    e = H5Emitter(model_name="cantilever", apegmsh_version="0.99.0")
    e.model(ndm=3, ndf=6)
    out = tmp_path / "model.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        assert "meta" in f
        meta = f["meta"]
        assert meta.attrs["schema_version"] == SCHEMA_VERSION
        assert meta.attrs["model_name"] == "cantilever"
        assert int(meta.attrs["ndm"]) == 3
        assert int(meta.attrs["ndf"]) == 6
        assert meta.attrs["apeGmsh_version"] == "0.99.0"


def test_h5emitter_write_meta_with_no_model_call(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An emitter with no model() call still writes /meta with ndm=0/ndf=0."""
    import h5py
    e = H5Emitter()
    out = tmp_path / "model.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        assert int(f["meta"].attrs["ndm"]) == 0
        assert int(f["meta"].attrs["ndf"]) == 0


def _s(v: object) -> str:
    """Decode an h5py compound-field string (bytes or str) to str."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def test_h5emitter_write_bcs_fix_compound_dataset(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """/opensees/bcs/fix is a compound dataset with target_kind / target / dofs."""
    import h5py
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.fix(1, 1, 1, 1, 1, 1, 1)
    e.fix(2, 1, 1, 0, 0, 0, 0)
    out = tmp_path / "bcs.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        ds = f["opensees/bcs/fix"]
        assert ds.shape == (2,)
        rows = ds[:]
        assert _s(rows[0]["target_kind"]) == "node"
        assert _s(rows[0]["target"]) == "1"
        assert tuple(rows[0]["dofs"]) == (1, 1, 1, 1, 1, 1)
        assert tuple(rows[1]["dofs"]) == (1, 1, 0, 0, 0, 0)


def test_h5emitter_write_bcs_mass_compound_dataset(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.mass(5, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0)
    out = tmp_path / "mass.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        ds = f["opensees/bcs/mass"]
        rows = ds[:]
        assert _s(rows[0]["target"]) == "5"
        assert tuple(rows[0]["values"]) == (100.0, 100.0, 100.0, 0.0, 0.0, 0.0)


def test_h5emitter_no_bcs_no_group(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """If no fix / mass calls, /opensees/bcs is not created at all.

    With no other bridge content the parent /opensees group is also
    skipped (lazy creation in H5Emitter._ops_group)."""
    import h5py
    e = H5Emitter()
    e.model(ndm=2, ndf=3)
    out = tmp_path / "no_bcs.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        assert "opensees" not in f


def test_h5emitter_write_uniaxial_material(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    import numpy as np
    e = H5Emitter()
    e.model(ndm=3, ndf=6)
    e.uniaxialMaterial("Steel02", 1, 420.0e6, 200.0e9, 0.01)
    out = tmp_path / "mat.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/materials/uniaxial/Steel02_1"]
        assert g.attrs["type"] == "Steel02"
        assert int(g.attrs["tag"]) == 1
        np.testing.assert_array_equal(
            g.attrs["params"], np.array([420.0e6, 200.0e9, 0.01]),
        )


def test_h5emitter_write_nd_material(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.nDMaterial("ElasticIsotropic", 5, 30.0e9, 0.2, 2400.0)
    out = tmp_path / "ndmat.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/materials/nd/ElasticIsotropic_5"]
        assert g.attrs["type"] == "ElasticIsotropic"


def test_h5emitter_write_simple_section(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.section(
        "ElasticMembranePlateSection", 2, 30.0e9, 0.2, 0.20, 2400.0,
    )
    out = tmp_path / "sec.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/sections/ElasticMembranePlateSection_2"]
        assert g.attrs["type"] == "ElasticMembranePlateSection"


def test_h5emitter_write_fiber_section_with_patches_fibers_layers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.uniaxialMaterial("Concrete02", 1, -30.0e6, -0.002, -25.0e6, -0.006, 0.1, 2.5e6, 200.0e6)
    e.uniaxialMaterial("Steel02", 2, 420.0e6, 200.0e9, 0.01, 20.0, 0.925, 0.15)
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.patch("rect", 1, 8, 8, -0.2, -0.2, 0.2, 0.2)
    e.fiber(0.1, 0.0, 0.001, 2)
    e.layer("straight", 2, 4, 0.001, -0.18, 0.0, 0.18, 0.0)
    e.section_close()
    out = tmp_path / "fiber.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/sections/Fiber_1"]
        # patches
        patches = g["patches"][:]
        assert len(patches) == 1
        assert _s(patches[0]["kind"]) == "rect"
        assert _s(patches[0]["material_ref"]) == "/opensees/materials/uniaxial/Concrete02_1"
        assert int(patches[0]["ny"]) == 8
        assert int(patches[0]["nz"]) == 8
        # fibers
        fibers = g["fibers"][:]
        assert len(fibers) == 1
        assert _s(fibers[0]["material_ref"]) == "/opensees/materials/uniaxial/Steel02_2"
        assert float(fibers[0]["area"]) == 0.001
        # layers
        layers = g["layers"][:]
        assert len(layers) == 1
        assert _s(layers[0]["kind"]) == "straight"
        assert int(layers[0]["n_bars"]) == 4


def test_h5emitter_write_geomtransf_per_call_groups(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    import numpy as np
    e = H5Emitter()
    e.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    e.geomTransf("PDelta", 2, 1.0, 0.0, 0.0)
    out = tmp_path / "tr.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g1 = f["opensees/transforms/Linear_1"]
        np.testing.assert_array_equal(
            g1["per_element_vecxz"][:], np.array([[0.0, 0.0, 1.0]]),
        )
        np.testing.assert_array_equal(g1["per_element_emitted_tag"][:], [1])
        g2 = f["opensees/transforms/PDelta_2"]
        np.testing.assert_array_equal(
            g2["per_element_vecxz"][:], np.array([[1.0, 0.0, 0.0]]),
        )


def test_h5emitter_write_beam_integration_group(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    import numpy as np
    e = H5Emitter()
    e.beamIntegration("Lobatto", 1, 5, 5)
    out = tmp_path / "bi.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/beam_integration/Lobatto_1"]
        assert g.attrs["type"] == "Lobatto"
        np.testing.assert_array_equal(
            g.attrs["params"], np.array([5.0, 5.0]),
        )


def test_h5emitter_write_elements_grouped_by_type(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    import numpy as np
    e = H5Emitter()
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 10, 1, 2, 1, 1)
    set_element_nodes(e, (2, 3))
    e.element("forceBeamColumn", 11, 2, 3, 1, 1)
    set_element_nodes(e, (3, 4, 5, 6))
    e.element("FourNodeTetrahedron", 12, 3, 4, 5, 6, 99)  # noqa
    out = tmp_path / "ele.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g_fbc = f["elements/forceBeamColumn"]
        np.testing.assert_array_equal(g_fbc["ids"][:], [10, 11])
        np.testing.assert_array_equal(
            g_fbc["connectivity"][:], np.array([[1, 2], [2, 3]]),
        )
        g_tet = f["elements/FourNodeTetrahedron"]
        np.testing.assert_array_equal(g_tet["ids"][:], [12])
        assert g_tet["connectivity"].shape == (1, 4)


def test_h5emitter_write_time_series(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.timeSeries("Linear", 1, "-factor", 1.0)
    out = tmp_path / "ts.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/time_series/Linear_1"]
        assert g.attrs["type"] == "Linear"
        assert int(g.attrs["tag"]) == 1


def test_h5emitter_write_plain_pattern_with_loads_and_series_ref(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.timeSeries("Linear", 1, "-factor", 1.0)
    e.pattern_open("Plain", 1, 1)
    e.load(10, 1.0, 0.0, 0.0)
    e.load(11, 0.0, 1.0, 0.0)
    e.sp(20, 1, 0.001)
    e.pattern_close()
    out = tmp_path / "pat.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/patterns/Plain_1"]
        assert g.attrs["type"] == "Plain"
        assert g.attrs["series_ref"] == "/opensees/time_series/Linear_1"
        loads = g["loads"][:]
        assert len(loads) == 2
        assert _s(loads[0]["target"]) == "10"
        assert tuple(loads[0]["forces"]) == (1.0, 0.0, 0.0)
        sps = g["sps"][:]
        assert len(sps) == 1
        assert int(sps[0]["dof"]) == 1
        assert float(sps[0]["value"]) == 0.001


def test_h5emitter_write_uniform_excitation_pattern(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.timeSeries("Path", 5, "-dt", 0.01, "-filePath", "elcentro.txt")
    e.pattern_open("UniformExcitation", 2, 1, "-accel", 5)
    e.pattern_close()
    out = tmp_path / "uniex.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/patterns/UniformExcitation_2"]
        assert g.attrs["type"] == "UniformExcitation"
        assert int(g.attrs["direction"]) == 1
        assert g.attrs["series_ref"] == "/opensees/time_series/Path_5"
        # No /loads sub-dataset for single-line patterns.
        assert "loads" not in g


def test_h5emitter_write_node_recorder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.recorder("Node", "-file", "disp.out", "-node", 1, 2, "-dof", 1, 2, 3, "disp")
    out = tmp_path / "rec.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        g = f["opensees/recorders/Node_0"]
        assert g.attrs["type"] == "Node"
        assert g.attrs["file"] == "disp.out"


def test_h5emitter_write_analysis_chain(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import h5py
    e = H5Emitter()
    e.constraints("Transformation")
    e.numberer("RCM")
    e.system("BandGeneral")
    e.test("NormDispIncr", 1.0e-6, 10)
    e.algorithm("Newton")
    e.integrator("LoadControl", 0.05)
    e.analysis("Static")
    e.analyze(steps=20, dt=0.1)
    out = tmp_path / "ana.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        a = f["opensees/analysis"]
        assert a.attrs["handler"] == "Transformation"
        assert a.attrs["numberer"] == "RCM"
        assert a.attrs["system"] == "BandGeneral"
        assert a.attrs["algorithm"] == "Newton"
        assert int(a.attrs["analyze_steps"]) == 20
        assert float(a.attrs["analyze_dt"]) == 0.1


def test_h5emitter_no_analysis_no_group(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """No analysis chain → /opensees/analysis absent; with no other
    bridge content the parent /opensees group is also absent."""
    import h5py
    e = H5Emitter()
    out = tmp_path / "no_ana.h5"
    e.write(str(out))
    with h5py.File(out, "r") as f:
        assert "opensees" not in f


def test_h5emitter_round_trip_minimal_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end smoke: build a minimal column model and verify every
    expected group is present in the file."""
    import h5py
    e = H5Emitter(model_name="cantilever")
    e.model(ndm=3, ndf=6)
    e.node(1, 0.0, 0.0, 0.0)
    e.node(2, 0.0, 0.0, 1.0)
    e.fix(1, 1, 1, 1, 1, 1, 1)
    e.uniaxialMaterial("Steel02", 1, 420.0e6, 200.0e9, 0.01, 20.0, 0.925, 0.15)
    e.uniaxialMaterial("Concrete02", 2, -30.0e6, -0.002, -25.0e6, -0.006, 0.1, 2.5e6, 200.0e6)
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.patch("rect", 2, 8, 8, -0.2, -0.2, 0.2, 0.2)
    e.section_close()
    e.geomTransf("PDelta", 1, 1.0, 0.0, 0.0)
    e.beamIntegration("Lobatto", 1, 1, 5)
    set_element_nodes(e, (1, 2))
    e.element("forceBeamColumn", 1, 1, 2, 1, 1)
    e.timeSeries("Path", 1, "-dt", 0.01, "-filePath", "elcentro.txt")
    e.pattern_open("UniformExcitation", 1, 1, "-accel", 1)
    e.pattern_close()
    out = tmp_path / "column.h5"
    e.write(str(out))

    with h5py.File(out, "r") as f:
        # Required: /meta
        assert int(f["meta"].attrs["ndm"]) == 3
        # Constitutive
        assert "opensees/materials/uniaxial/Steel02_1" in f
        assert "opensees/materials/uniaxial/Concrete02_2" in f
        # Section
        assert "opensees/sections/Fiber_1" in f
        assert f["opensees/sections/Fiber_1"]["patches"].shape == (1,)
        # Transform + beamIntegration
        assert "opensees/transforms/PDelta_1" in f
        assert "opensees/beam_integration/Lobatto_1" in f
        # Element (stays at root — neutral zone)
        assert "elements/forceBeamColumn" in f
        # BCs
        assert "opensees/bcs/fix" in f
        # Time series + pattern
        assert "opensees/time_series/Path_1" in f
        assert "opensees/patterns/UniformExcitation_1" in f
