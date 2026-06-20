"""Unit tests for :class:`PyEmitter`."""
from __future__ import annotations

from apeGmsh.opensees.emitter.py import PyEmitter


def _payload(emitter: PyEmitter) -> list[str]:
    """Return non-comment, non-import lines."""
    skip_starts = ("#", "import ", "ops.wipe(")
    return [
        ln for ln in emitter.lines()
        if not any(ln.startswith(s) for s in skip_starts)
    ]


def test_lines_start_with_banner_import_wipe() -> None:
    e = PyEmitter()
    lines = e.lines()
    assert lines[0].startswith("# auto-generated")
    assert "import openseespy.opensees as ops" in lines
    assert "ops.wipe()" in lines


def test_model_emits_basic_with_flags() -> None:
    e = PyEmitter()
    e.model(ndm=3, ndf=6)
    assert _payload(e) == ["ops.model('basic', '-ndm', 3, '-ndf', 6)"]


def test_node_emits_call() -> None:
    e = PyEmitter()
    e.node(1, 0.0, 0.0, 0.0)
    assert _payload(e) == ["ops.node(1, 0.0, 0.0, 0.0)"]


def test_fix_emits_dofs() -> None:
    e = PyEmitter()
    e.fix(2, 1, 1, 1, 0, 0, 0)
    assert _payload(e) == ["ops.fix(2, 1, 1, 1, 0, 0, 0)"]


def test_mass_emits_values() -> None:
    e = PyEmitter()
    e.mass(2, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0)
    assert _payload(e) == ["ops.mass(2, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0)"]


def test_eigen_default_solver_emits_ops_call() -> None:
    e = PyEmitter()
    rc = e.eigen(5)
    assert rc == []
    assert _payload(e) == ["ops.eigen('-genBandArpack', 5)"]


def test_eigen_custom_solver_passed_through() -> None:
    e = PyEmitter()
    e.eigen(3, solver="-fullGenLapack")
    assert _payload(e) == ["ops.eigen('-fullGenLapack', 3)"]


def test_profiler_start_emits_ops_call() -> None:
    e = PyEmitter()
    e.profiler("start", "-deep")
    assert _payload(e) == ["ops.profiler('start', '-deep')"]


def test_profiler_report_emits_ops_call() -> None:
    e = PyEmitter()
    e.profiler("report", "profile.h5", "-run", "caseA")
    assert _payload(e) == [
        "ops.profiler('report', 'profile.h5', '-run', 'caseA')",
    ]


def test_uniaxial_material_quotes_type_token() -> None:
    e = PyEmitter()
    e.uniaxialMaterial("Steel02", 1, 420e6, 200e9, 0.01, 20.0, 0.925, 0.15)
    out = _payload(e)[0]
    assert out.startswith("ops.uniaxialMaterial('Steel02', 1, ")


def test_nd_material_quotes_strings() -> None:
    e = PyEmitter()
    e.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3, 7800.0)
    assert _payload(e) == [
        "ops.nDMaterial('ElasticIsotropic', 1, 200000000000.0, 0.3, 7800.0)",
    ]


def test_geomtransf_emits_call() -> None:
    e = PyEmitter()
    e.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    assert _payload(e) == ["ops.geomTransf('Linear', 1, 0.0, 0.0, 1.0)"]


def test_section_open_emits_section_call_close_is_noop() -> None:
    e = PyEmitter()
    n_before = len(e.lines())
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.fiber(0.0, 0.0, 0.01, 1)
    e.section_close()
    out = _payload(e)
    assert out == [
        "ops.section('Fiber', 1, '-GJ', 1000000000.0)",
        "ops.fiber(0.0, 0.0, 0.01, 1)",
    ]
    # section_close did not add a line.
    n_after = len(e.lines())
    assert n_after == n_before + 2


def test_patch_layer_fiber_inside_section() -> None:
    e = PyEmitter()
    e.section_open("Fiber", 1)
    e.patch("rect", 1, 4, 2, -0.1, -0.05, 0.1, 0.05)
    e.layer("straight", 1, 4, 1e-4, -0.05, -0.025, 0.05, 0.025)
    e.fiber(0.0, 0.06, 1e-4, 1)
    e.section_close()
    payload = _payload(e)
    assert payload[0].startswith("ops.section(")
    assert payload[1].startswith("ops.patch(")
    assert payload[2].startswith("ops.layer(")
    assert payload[3].startswith("ops.fiber(")


def test_element_emits_call() -> None:
    e = PyEmitter()
    e.element("forceBeamColumn", 1, 1, 2, 1, "-section", 1, 5)
    assert _payload(e) == [
        "ops.element('forceBeamColumn', 1, 1, 2, 1, '-section', 1, 5)",
    ]


def test_timeseries_emits_call() -> None:
    e = PyEmitter()
    e.timeSeries("Linear", 1)
    assert _payload(e) == ["ops.timeSeries('Linear', 1)"]


def test_pattern_open_emits_pattern_call_close_is_noop() -> None:
    e = PyEmitter()
    e.pattern_open("Plain", 1, 1)
    e.load(2, 100.0, 0.0, 0.0)
    e.pattern_close()
    assert _payload(e) == [
        "ops.pattern('Plain', 1, 1)",
        "ops.load(2, 100.0, 0.0, 0.0)",
    ]


def test_pattern_uniformexcitation_with_no_body() -> None:
    e = PyEmitter()
    e.pattern_open("UniformExcitation", 1, 1, "-accel", 1)
    e.pattern_close()
    assert _payload(e) == [
        "ops.pattern('UniformExcitation', 1, 1, '-accel', 1)",
    ]


def test_pattern_h5drm_single_line_identity() -> None:
    # Drive the real H5DRM primitive through the py emitter (ADR 0066):
    # km->m crd_scale, identity transform, zero x0. File path is quoted.
    from apeGmsh.opensees.pattern.pattern import H5DRM

    e = PyEmitter()
    H5DRM(h5drm="motions.h5drm")._emit(e, tag=1)
    assert _payload(e) == [
        "ops.pattern('H5DRM', 1, 'motions.h5drm', 1.0, 1000.0, 1.0, 1, "
        "1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)",
    ]


def test_sp_emits_call() -> None:
    e = PyEmitter()
    e.sp(2, 1, 0.005)
    assert _payload(e) == ["ops.sp(2, 1, 0.005)"]


def test_eleload_emits_call() -> None:
    e = PyEmitter()
    e.eleLoad("-ele", 1, "-type", "-beamUniform", -2400.0, 0.0)
    assert _payload(e) == [
        "ops.eleLoad('-ele', 1, '-type', '-beamUniform', -2400.0, 0.0)",
    ]


def test_recorder_emits_call() -> None:
    e = PyEmitter()
    e.recorder(
        "Node", "-file", "disp.out", "-node", 1, 2, "-dof", 1, 2, 3, "disp",
    )
    assert _payload(e) == [
        "ops.recorder('Node', '-file', 'disp.out', '-node', 1, 2, '-dof', "
        "1, 2, 3, 'disp')",
    ]


def test_constraints_with_args() -> None:
    e = PyEmitter()
    e.constraints("Penalty", 1e10, 1e10)
    assert _payload(e) == ["ops.constraints('Penalty', 10000000000.0, 10000000000.0)"]


def test_numberer_emits_call() -> None:
    e = PyEmitter()
    e.numberer("RCM")
    assert _payload(e) == ["ops.numberer('RCM')"]


def test_system_emits_call() -> None:
    e = PyEmitter()
    e.system("BandGeneral")
    assert _payload(e) == ["ops.system('BandGeneral')"]


def test_test_emits_call() -> None:
    e = PyEmitter()
    e.test("NormDispIncr", 1e-6, 10)
    assert _payload(e) == ["ops.test('NormDispIncr', 1e-06, 10)"]


def test_algorithm_emits_call() -> None:
    e = PyEmitter()
    e.algorithm("Newton")
    assert _payload(e) == ["ops.algorithm('Newton')"]


def test_integrator_emits_call() -> None:
    e = PyEmitter()
    e.integrator("LoadControl", 0.05)
    assert _payload(e) == ["ops.integrator('LoadControl', 0.05)"]


def test_analysis_emits_call() -> None:
    e = PyEmitter()
    e.analysis("Static")
    assert _payload(e) == ["ops.analysis('Static')"]


def test_analyze_steps_only() -> None:
    # Fail-loud contract: every analyze emits a per-increment loop whose
    # rc check aborts the deck on the first failed increment (a batched
    # ``ops.analyze(20)`` short-circuits internally and the deck would
    # silently run on with the stage partial — or not applied at all).
    e = PyEmitter()
    ret = e.analyze(steps=20)
    assert ret == 0
    lines = _payload(e)
    assert lines[0] == "for _apesees_i in range(20):"
    assert lines[1] == "    if ops.analyze(1) != 0:"
    assert lines[2].lstrip().startswith("raise SystemExit(")
    assert "FAILED at increment" in lines[2]
    assert "ops.getTime()" in lines[2]


def test_analyze_with_dt() -> None:
    e = PyEmitter()
    e.analyze(steps=100, dt=0.01)
    lines = _payload(e)
    assert lines[0] == "for _apesees_i in range(100):"
    assert lines[1] == "    if ops.analyze(1, 0.01) != 0:"


def test_analyze_label_names_the_stage_in_the_banner() -> None:
    e = PyEmitter()
    e.analyze(steps=5, dt=0.1, label="Gravity")
    text = "\n".join(_payload(e))
    assert "of stage 'Gravity'" in text


def test_string_with_quote_is_escaped() -> None:
    e = PyEmitter()
    e.recorder("Node", "-file", "disp's.out")
    out = _payload(e)[0]
    # The single quote inside the filename is escaped.
    assert "disp\\'s.out" in out
