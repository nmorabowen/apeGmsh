"""Unit tests for :class:`TclEmitter`.

Each test exercises one Protocol method and asserts the rendered Tcl
text. The output must round-trip through the OpenSees Tcl parser; the
subprocess test layer (``tests/opensees/subprocess/``) handles real
parser verification.
"""
from __future__ import annotations

from apeGmsh.opensees.emitter.tcl import TclEmitter


def _stripped(emitter: TclEmitter) -> list[str]:
    """Lines with the auto-generated banner removed."""
    return [ln for ln in emitter.lines() if not ln.startswith("# ")]


def test_lines_start_with_auto_generated_banner() -> None:
    e = TclEmitter()
    lines = e.lines()
    assert lines[0].startswith("# auto-generated")


def test_model_emits_basicbuilder() -> None:
    e = TclEmitter()
    e.model(ndm=3, ndf=6)
    assert _stripped(e) == ["model BasicBuilder -ndm 3 -ndf 6"]


def test_node_emits_one_line() -> None:
    e = TclEmitter()
    e.node(1, 0.0, 0.0, 0.0)
    assert _stripped(e) == ["node 1 0.0 0.0 0.0"]


def test_fix_emits_dofs() -> None:
    e = TclEmitter()
    e.fix(1, 1, 1, 1, 0, 0, 0)
    assert _stripped(e) == ["fix 1 1 1 1 0 0 0"]


def test_mass_emits_values() -> None:
    e = TclEmitter()
    e.mass(2, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0)
    assert _stripped(e) == ["mass 2 50.0 50.0 50.0 0.0 0.0 0.0"]


def test_eigen_default_solver_emits_genbandarpack() -> None:
    e = TclEmitter()
    rc = e.eigen(5)
    assert rc == []
    assert _stripped(e) == ["eigen -genBandArpack 5"]


def test_eigen_custom_solver_passed_through() -> None:
    e = TclEmitter()
    e.eigen(3, solver="-fullGenLapack")
    assert _stripped(e) == ["eigen -fullGenLapack 3"]


def test_modal_properties_bare_emits_command_only() -> None:
    e = TclEmitter()
    rc = e.modal_properties()
    assert rc == {}
    assert _stripped(e) == ["modalProperties"]


def test_modal_properties_unorm_and_file() -> None:
    e = TclEmitter()
    e.modal_properties(unorm=True, out="props.txt")
    assert _stripped(e) == ["modalProperties -unorm -file props.txt"]


def test_profiler_start_emits_subcommand_and_flags() -> None:
    e = TclEmitter()
    e.profiler("start", "-deep", "-memory")
    assert _stripped(e) == ["profiler start -deep -memory"]


def test_profiler_report_emits_file_and_run() -> None:
    e = TclEmitter()
    e.profiler("report", "profile.h5", "-run", "caseA")
    assert _stripped(e) == ["profiler report profile.h5 -run caseA"]


def test_uniaxial_material_emits_type_token_and_params() -> None:
    e = TclEmitter()
    e.uniaxialMaterial("Steel02", 1, 420e6, 200e9, 0.01, 20.0, 0.925, 0.15)
    out = _stripped(e)[0]
    assert out.startswith("uniaxialMaterial Steel02 1 ")
    # All numeric params present.
    assert "420000000.0" in out
    assert "0.01" in out


def test_nd_material_renders_string_params() -> None:
    e = TclEmitter()
    e.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3, 7800.0)
    assert _stripped(e) == [
        "nDMaterial ElasticIsotropic 1 200000000000.0 0.3 7800.0",
    ]


def test_geomtransf_renders_vecxz() -> None:
    e = TclEmitter()
    e.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    assert _stripped(e) == ["geomTransf Linear 1 0.0 0.0 1.0"]


def test_section_block_brackets_with_brace() -> None:
    e = TclEmitter()
    e.section_open("Fiber", 1, "-GJ", 1.0e9)
    e.fiber(0.0, 0.0, 0.01, 1)
    e.section_close()
    out = _stripped(e)
    assert out[0].endswith("{")
    assert out[0].startswith("section Fiber 1 -GJ 1000000000.0")
    assert out[1].strip().startswith("fiber")
    assert out[-1] == "}"


def test_patch_layer_fiber_indented_inside_section() -> None:
    e = TclEmitter()
    e.section_open("Fiber", 1)
    e.patch("rect", 1, 4, 2, -0.1, -0.05, 0.1, 0.05)
    e.layer("straight", 1, 4, 1e-4, -0.05, -0.025, 0.05, 0.025)
    e.fiber(0.0, 0.06, 1e-4, 1)
    e.section_close()
    out = _stripped(e)
    # Each interior line is indented (4 spaces).
    for ln in out[1:-1]:
        assert ln.startswith("    ")


def test_element_emits_tag_and_args() -> None:
    e = TclEmitter()
    e.element("forceBeamColumn", 1, 1, 2, 1, "-section", 1, 5)
    assert _stripped(e) == [
        "element forceBeamColumn 1 1 2 1 -section 1 5",
    ]


def test_timeseries_emits_type_and_tag() -> None:
    e = TclEmitter()
    e.timeSeries("Linear", 1)
    assert _stripped(e) == ["timeSeries Linear 1"]


def test_pattern_plain_opens_block_and_closes() -> None:
    e = TclEmitter()
    e.pattern_open("Plain", 1, 1)
    e.load(2, 100.0, 0.0, 0.0)
    e.pattern_close()
    out = _stripped(e)
    assert out[0].endswith("{")
    assert out[0].startswith("pattern Plain 1 1")
    assert out[1].strip().startswith("load")
    assert out[2] == "}"


def test_pattern_uniformexcitation_is_single_line() -> None:
    e = TclEmitter()
    e.pattern_open("UniformExcitation", 1, 1, "-accel", 1)
    e.pattern_close()
    out = _stripped(e)
    assert len(out) == 1
    assert out[0] == "pattern UniformExcitation 1 1 -accel 1"


def test_pattern_h5drm_single_line_identity() -> None:
    # Drive the real H5DRM primitive through the Tcl emitter (ADR 0066):
    # single line (not a block), km->m crd_scale, identity transform.
    from apeGmsh.opensees.pattern.pattern import H5DRM

    e = TclEmitter()
    H5DRM(h5drm="motions.h5drm")._emit(e, tag=1)
    out = _stripped(e)
    assert out == [
        "pattern H5DRM 1 motions.h5drm 1.0 1000.0 1.0 1 "
        "1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0"
    ]


def test_sp_inside_pattern_indented() -> None:
    e = TclEmitter()
    e.pattern_open("Plain", 1, 1)
    e.sp(2, 1, 0.005)
    e.pattern_close()
    out = _stripped(e)
    assert out[1].startswith("    sp ")


def test_eleload_inside_pattern_indented() -> None:
    e = TclEmitter()
    e.pattern_open("Plain", 1, 1)
    e.eleLoad("-ele", 1, "-type", "-beamUniform", -2400.0, 0.0)
    e.pattern_close()
    out = _stripped(e)
    assert out[1].startswith("    eleLoad ")


def test_recorder_emits_kind_token() -> None:
    e = TclEmitter()
    e.recorder("Node", "-file", "disp.out", "-node", 1, 2, "-dof", 1, 2, 3, "disp")
    assert _stripped(e) == [
        "recorder Node -file disp.out -node 1 2 -dof 1 2 3 disp",
    ]


def test_constraints_with_args() -> None:
    e = TclEmitter()
    e.constraints("Penalty", 1e10, 1e10)
    assert _stripped(e) == ["constraints Penalty 10000000000.0 10000000000.0"]


def test_constraints_without_args() -> None:
    e = TclEmitter()
    e.constraints("Transformation")
    assert _stripped(e) == ["constraints Transformation"]


def test_numberer_emits_one_token() -> None:
    e = TclEmitter()
    e.numberer("RCM")
    assert _stripped(e) == ["numberer RCM"]


def test_system_emits_args() -> None:
    e = TclEmitter()
    e.system("BandGeneral")
    assert _stripped(e) == ["system BandGeneral"]


def test_test_emits_args() -> None:
    e = TclEmitter()
    e.test("NormDispIncr", 1e-6, 10)
    assert _stripped(e) == ["test NormDispIncr 1e-06 10"]


def test_algorithm_emits_args() -> None:
    e = TclEmitter()
    e.algorithm("Newton")
    assert _stripped(e) == ["algorithm Newton"]


def test_integrator_emits_args() -> None:
    e = TclEmitter()
    e.integrator("LoadControl", 0.05)
    assert _stripped(e) == ["integrator LoadControl 0.05"]


def test_analysis_emits_one_token() -> None:
    e = TclEmitter()
    e.analysis("Static")
    assert _stripped(e) == ["analysis Static"]


def test_analyze_with_steps_only() -> None:
    # Fail-loud contract: every analyze emits a per-increment loop whose
    # rc check aborts the deck (Tcl ``error``) on the first failed
    # increment — a batched ``analyze 20`` short-circuits internally and
    # the deck would silently run on with the stage partial.
    e = TclEmitter()
    ret = e.analyze(steps=20)
    assert ret == 0
    lines = _stripped(e)
    assert lines[0].startswith("for {set _apesees_i 0} {$_apesees_i < 20}")
    assert lines[1].strip() == "if {[analyze 1] != 0} {"
    assert lines[2].strip().startswith(
        'error "apeGmsh: analyze FAILED at increment'
    )
    assert "[getTime]" in lines[2]


def test_analyze_with_steps_and_dt() -> None:
    e = TclEmitter()
    e.analyze(steps=100, dt=0.01)
    lines = _stripped(e)
    assert lines[0].startswith("for {set _apesees_i 0} {$_apesees_i < 100}")
    assert lines[1].strip() == "if {[analyze 1 0.01] != 0} {"


def test_analyze_label_names_the_stage_in_the_banner() -> None:
    e = TclEmitter()
    e.analyze(steps=5, dt=0.1, label="Gravity")
    text = "\n".join(_stripped(e))
    assert "of stage 'Gravity'" in text


def test_preamble_inserts_comment_at_top() -> None:
    e = TclEmitter()
    e.preamble("custom note")
    assert e.lines()[0] == "# custom note"
