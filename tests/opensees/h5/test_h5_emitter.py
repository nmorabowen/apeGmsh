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


def test_h5emitter_write_raises_until_step_2() -> None:
    """Step 1 scaffold: write() raises NotImplementedError."""
    import pytest
    e = H5Emitter()
    with pytest.raises(NotImplementedError):
        e.write("/tmp/should_not_exist.h5")
