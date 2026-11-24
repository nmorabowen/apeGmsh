"""Slice #3120: ``mpm.Stage`` (ADR 0134 Amendment 4).

The fork's ``mpm stage <matTag> <s>`` (``OPS_LadrunoMPM.cpp``, ``sub ==
"stage"``; ``LadrunoMPMSoil.cpp::stageWalk`` at ``ladruno-v20261123b``,
3eb195bbc) stages every particle whose material has the tag, through the
material's own ``materialState`` parameter, and answers the count. Offline
oracles here:

* every damage case of the slice card is refused by name, at construction or
  at the emit, before the fork sees the deck; a refused call registers
  nothing and a refused emit writes nothing;
* the owner's rulings: DruckerPrager stages 0/1/2 and LadrunoSANISAND 0/1 are
  admitted and nothing else is; a stage after the last analyze is refused;
* the deck writes each stage row where it was declared among the analyze
  rows, ``model.h5`` archives it there, and both replays (flat and composed)
  keep it there;
* the emitted tag is the ``MATERIAL_TAG`` of that body's particles;
* the fork's mode-F SANISAND refusal off a B-spline grid (WP-232k rule K0)
  is refused by apeGmsh too, and a tripwire fails when ``Grid`` admits a
  B-spline basis.

The fork-side behaviour is ``tests/opensees/integration_ladruno/
test_mpm_stage_live.py``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees import mpm as mpm_mod
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.material.nd import (
    DruckerPrager,
    ElasticIsotropic,
    LadrunoSANISAND,
    ManzariDafalias,
)

from ._meshes import grid_nodes, make_fem, quads
from .test_mpm_bridge import _composed_fem, _deck_lines

H = 0.1
DP_KW: dict[str, float] = dict(
    K=1.0e4, G=5.0e3, sigmaY=10.0, rho=0.1, rhoBar=0.1, Kinf=0.0, Ko=0.0,
    delta1=0.0, delta2=0.0, H=0.0, theta=1.0)
SAND_KW: dict[str, float] = dict(
    G0=264.32, nu=0.3129, e_init=0.6944, Mc=1.3309, c=0.71, lambda_c=0.027,
    e0=0.83, ksi=0.45, P_atm=101.0, m=0.005, h0=1.3, Ch=0.968, nb=3.5,
    A0=0.05, nd=5.75, z_max=12.5, cz=1100.0, rho=2.0)


def _dp() -> DruckerPrager:
    return DruckerPrager(**DP_KW)


def _sand(**kw: Any) -> LadrunoSANISAND:
    return LadrunoSANISAND(**{**SAND_KW, **kw})


def _bridge(fem: Any = None) -> apeSees:
    """A DruckerPrager body ``Soil`` on a 20 x 10 grid; ``fill`` (elastic,
    nDMaterial tag 1) is registered first, so ``weak`` is tag 2."""
    if fem is None:
        nodes, ids = grid_nodes((4, 2), (0.5, 0.0), (1.5, 0.4))
        fem = make_fem(nodes, {"Soil": quads(ids)})
    ops = apeSees(fem, units="kN_m_s", _artifacts=False)
    ops.model(ndm=2, ndf=2)
    ops.nDMaterial.ElasticIsotropic(E=1.0e4, nu=0.3, name="fill")
    ops.nDMaterial.DruckerPrager(**DP_KW, name="weak")
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    ops.mpm.body("Soil", material="weak", density=1.8, ppc=2)
    ops.fix(pg="Soil", dofs=[1, 1])
    return ops


def _staged(ops: apeSees) -> None:
    ops.mpm.gravity(0.0, -9.81)
    ops.mpm.stage("weak", 0)
    ops.mpm.analyze(10, 1.0e-4)
    ops.mpm.stage("weak", 2)
    ops.mpm.analyze(20, 1.0e-4)


def _no_files(root: Path) -> None:
    assert sorted(p.name for p in root.rglob("*") if p.is_file()) == []


# -- the arguments, refused at construction ----------------------------------

@pytest.mark.parametrize("cls,stages", [
    (DruckerPrager, (0, 1, 2)), (LadrunoSANISAND, (0, 1))])
def test_the_owner_ruling_admits_these_stages(cls: type, stages: tuple) -> None:
    """Owner ruling 1 (#3120): DruckerPrager 0/1/2, LadrunoSANISAND 0/1."""
    assert mpm_mod.STAGES == {DruckerPrager: (0, 1, 2), LadrunoSANISAND: (0, 1)}
    mat = _dp() if cls is DruckerPrager else _sand()
    for s in stages:
        assert mpm_mod.Stage(material=mat, stage=s).args(7) == ("stage", 7, s)


@pytest.mark.parametrize("make,match", [
    # the material is not an nDMaterial, or its class has no stage
    (lambda: mpm_mod.Stage(material="weak", stage=0),  # type: ignore[arg-type]
     r"Stage\.material must be an nDMaterial, got str"),
    (lambda: mpm_mod.Stage(material=ElasticIsotropic(E=1.0, nu=0.3), stage=0),
     r"ElasticIsotropic has no stage the fork's mpm stage walk reaches; it "
     r"stages DruckerPrager 0/1/2, LadrunoSANISAND 0/1"),
    # ManzariDafalias answers materialState too, but the ruling names
    # LadrunoSANISAND only: refused by name, never assumed.
    (lambda: mpm_mod.Stage(material=ManzariDafalias(**SAND_KW), stage=0),
     r"ManzariDafalias has no stage"),
    # stage: a bool, a float, a string, outside the class's set
    (lambda: mpm_mod.Stage(material=_dp(), stage=True),
     r"Stage\.stage must be an integer, got True"),
    (lambda: mpm_mod.Stage(material=_dp(), stage=1.0),  # type: ignore[arg-type]
     r"Stage\.stage must be an integer, got 1\.0"),
    (lambda: mpm_mod.Stage(material=_dp(), stage="2"),  # type: ignore[arg-type]
     r"Stage\.stage must be an integer, got '2'"),
    (lambda: mpm_mod.Stage(material=_dp(), stage=3),
     r"DruckerPrager reads stages \(0, 1, 2\), got 3"),
    (lambda: mpm_mod.Stage(material=_dp(), stage=-1),
     r"DruckerPrager reads stages \(0, 1, 2\), got -1"),
    (lambda: mpm_mod.Stage(material=_sand(), stage=2),
     r"LadrunoSANISAND reads stages \(0, 1\), got 2"),
])
def test_stage_arguments_refuse_by_name(make: Callable[[], Any], match: str) -> None:
    with pytest.raises(mpm_mod.MPMError, match=match):
        make()


def test_a_subclass_of_a_staged_class_is_refused() -> None:
    """Keyed by the class itself: a subclass is not assumed to stage."""
    class Sub(DruckerPrager):
        pass

    with pytest.raises(mpm_mod.MPMError, match="Sub has no stage"):
        mpm_mod.Stage(material=Sub(**DP_KW), stage=0)


def test_a_refused_call_registers_nothing() -> None:
    ops = _bridge()
    n = len(ops._primitives)
    with pytest.raises(mpm_mod.MPMError, match="has no stage"):
        ops.mpm.stage("fill", 0)
    with pytest.raises(mpm_mod.MPMError, match="got 5"):
        ops.mpm.stage("weak", 5)
    with pytest.raises(KeyError):
        ops.mpm.stage("nope", 0)
    assert len(ops._primitives) == n


def test_the_call_takes_a_handle_or_a_name() -> None:
    ops = _bridge()
    weak = ops._names["weak"]
    a = ops.mpm.stage("weak", 0)
    b = ops.mpm.stage(weak, 2)
    assert a.material is b.material is weak
    assert (a.stage, b.stage) == (0, 2)


# -- the model-wide refusals, at the emit -------------------------------------

def _emit_refuses(tmp_path: Path, ops: apeSees, match: str) -> None:
    with pytest.raises((BridgeError, mpm_mod.MPMError), match=match):
        ops.tcl(str(tmp_path / "case1.tcl"))
    _no_files(tmp_path)


def test_a_stage_after_the_last_analyze_refuses(tmp_path: Path) -> None:
    """Owner ruling 2 (#3120): refused, not warned."""
    ops = _bridge()
    _staged(ops)
    ops.mpm.stage("weak", 0)
    _emit_refuses(tmp_path, ops, r"mpm stage \(DruckerPrager -> 0\) is declared "
                                 r"after the last ops\.mpm\.analyze")


def test_a_stage_with_no_analyze_at_all_refuses(tmp_path: Path) -> None:
    ops = _bridge()
    ops.mpm.stage("weak", 0)
    _emit_refuses(tmp_path, ops, r"after the last ops\.mpm\.analyze")


def test_a_stage_whose_material_no_body_uses_refuses(tmp_path: Path) -> None:
    ops = _bridge()
    ops.nDMaterial.DruckerPrager(**DP_KW, name="other")
    ops.mpm.stage("other", 0)
    ops.mpm.analyze(10, 1.0e-4)
    _emit_refuses(tmp_path, ops, r"mpm stage DruckerPrager -> 0: no "
                                 r"ops\.mpm\.body uses that material.*no "
                                 r"particle material has tag")


def test_two_stages_of_one_material_without_an_analyze_between_refuse(
    tmp_path: Path,
) -> None:
    ops = _bridge()
    ops.mpm.stage("weak", 0)
    ops.mpm.gravity(0.0, -9.81)            # not an analyze
    ops.mpm.stage("weak", 2)
    ops.mpm.analyze(10, 1.0e-4)
    _emit_refuses(tmp_path, ops, r"two mpm stages of one DruckerPrager material "
                                 r"\(0, then 2\) with no ops\.mpm\.analyze "
                                 r"between them")


def test_the_same_material_staged_again_after_an_analyze_emits(tmp_path: Path) -> None:
    ops = _bridge()
    _staged(ops)
    ops.tcl(str(tmp_path / "case1.tcl"))
    assert (tmp_path / "case1.particles.h5").is_file()


# The two SANISAND refusals are read off ``mpm_block``: a SANISAND body is
# refused anyway on every grid apeGmsh admits (rule K0, below), and the stage
# refusals run first, so the message a user meets names the stage problem.

def _sand_block(*stages_and_analyzes: Any) -> tuple:
    a, b = _sand(), _sand(nu=0.25)
    grid = mpm_mod.Grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    bodies = [mpm_mod.Body(name="a", pg="A", material=a, density=1.0, ppc=2),
              mpm_mod.Body(name="b", pg="B", material=b, density=1.0, ppc=2)]
    rest = [m(a, b) for m in stages_and_analyzes]
    return grid, bodies, rest


def _st(which: str, s: int) -> Callable[[Any, Any], mpm_mod.Stage]:
    return lambda a, b: mpm_mod.Stage(material=a if which == "a" else b, stage=s)


def _an() -> Callable[[Any, Any], mpm_mod.Analyze]:
    return lambda a, b: mpm_mod.Analyze(steps=1, dt=1.0e-4)


def test_two_sanisand_tags_staged_to_different_values_at_one_point_refuse() -> None:
    grid, bodies, rest = _sand_block(_st("a", 0), _st("b", 1), _an())
    with pytest.raises(mpm_mod.MPMError,
                       match=r"two LadrunoSANISAND materials staged to different "
                             r"values \(0, then 1\) with no ops\.mpm\.analyze "
                             r"between them: the fork's SANISAND stage flag is "
                             r"class-wide"):
        mpm_mod.mpm_block([grid, *bodies, *rest], ndm=2)


@pytest.mark.parametrize("rest", [
    (_st("a", 1), _st("b", 1), _an()),               # same value: redundant
    (_st("a", 0), _an(), _st("b", 1), _an()),        # an analyze between
], ids=["same-value", "analyze-between"])
def test_sanisand_stages_the_class_wide_flag_admits_pass_to_rule_k0(
    rest: tuple,
) -> None:
    """Admitted by the stage checks; then the body meets rule K0."""
    grid, bodies, members = _sand_block(*rest)
    with pytest.raises(mpm_mod.MPMError, match=mpm_mod.FORK_SANISAND_BASIS_REFUSAL):
        mpm_mod.mpm_block([grid, *bodies, *members], ndm=2)


# -- rule K0: SANISAND particles in mode F need a B-spline grid ---------------

@pytest.mark.parametrize("basis", [None, "gimp", "linear"])
def test_a_sanisand_body_on_a_gimp_or_linear_grid_refuses(
    tmp_path: Path, basis: str | None,
) -> None:
    """The fork refuses SANISAND particles in explicit mode F on gimp /
    linear at ``mpm analyze`` (``LadrunoMPMPattern::analyze`` at 3eb195bbc:
    "... at particles in explicit mode F needs a B-spline basis (bspline2 /
    bspline3)"). apeGmsh refuses the deck first, naming that error."""
    nodes, ids = grid_nodes((4, 2), (0.5, 0.0), (1.5, 0.4))
    ops = apeSees(make_fem(nodes, {"Soil": quads(ids)}), units="kN_m_s",
                  _artifacts=False)
    ops.model(ndm=2, ndf=2)
    ops.nDMaterial.LadrunoSANISAND(**SAND_KW, name="sand")
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10), basis=basis)
    ops.mpm.body("Soil", material="sand", density=1.8, ppc=2)
    ops.mpm.stage("sand", 0)
    ops.mpm.analyze(10, 1.0e-4)
    ops.fix(pg="Soil", dofs=[1, 1])
    _emit_refuses(tmp_path, ops,
                  r"mpm body 'Soil' is LadrunoSANISAND.*'at particles in explicit "
                  r"mode F needs a B-spline basis'.*WP-232k rule K0")


def test_rule_k0_is_every_admitted_basis_until_a_bspline_one_is_admitted() -> None:
    """The tripwire (Amendment 3's B10 pattern): ``mpm_block`` refuses every
    SANISAND body because :class:`Grid` admits gimp and linear only. When a
    B-spline basis is admitted, this test fails, and the refusal is narrowed
    to the gimp and linear grids in the same PR."""
    assert mpm_mod.BASES == ("gimp", "linear")
    for basis in ("bspline2", "bspline3"):
        with pytest.raises(mpm_mod.MPMError, match="Grid.basis"):
            mpm_mod.Grid(origin=(0.0, 0.0), h=H, cells=(4, 4), basis=basis)
    assert "rule K0" in (mpm_mod.mpm_block.__doc__ or "")


# -- ops.stage blocks are still refused, unchanged ----------------------------

def test_a_model_with_ops_stage_blocks_still_refuses() -> None:
    """``BuiltModel._mpm_block``'s guard, unchanged, with ``mpm.Stage`` rows
    in the block (checked on the guard itself, as ``test_mpm_bridge.py``
    does: an ``ops.stage`` block needs a whole chain to build)."""
    from dataclasses import replace

    from apeGmsh.opensees.emitter.tcl import TclEmitter

    ops = _bridge()
    _staged(ops)
    bm = ops.build()
    assert any(isinstance(m, mpm_mod.Stage) for m in bm.mpm_members())
    staged = replace(bm, stage_records=(object(),))
    with pytest.raises(BridgeError, match=r"ops\.stage block\(s\); MPM in this "
                                          r"version is deck-only and unstaged"):
        staged._mpm_block(TclEmitter(), staged.mpm_members())


# -- the deck, the archive and the replays ------------------------------------

def _mpm_words(text: str, prefix: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith(prefix)]


def test_the_tcl_deck_writes_each_stage_where_it_was_declared(tmp_path: Path) -> None:
    ops = _bridge()
    _staged(ops)
    ops.tcl(str(tmp_path / "case1.tcl"))
    lines = _mpm_words((tmp_path / "case1.tcl").read_text(encoding="utf-8"), "mpm ")
    assert [ln.split()[1] for ln in lines] == [
        "grid", "particles", "gravity", "stage", "analyze", "stage", "analyze"]
    assert lines[3:] == ["mpm stage 2 0", "mpm analyze 10 0.0001",
                         "mpm stage 2 2", "mpm analyze 20 0.0001"]


def test_the_py_deck_writes_each_stage_where_it_was_declared(tmp_path: Path) -> None:
    ops = _bridge()
    _staged(ops)
    ops.py(str(tmp_path / "case1.py"))
    lines = _mpm_words((tmp_path / "case1.py").read_text(encoding="utf-8"),
                       "ops.mpm(")
    assert lines[-4:] == ["ops.mpm('stage', 2, 0)", "ops.mpm('analyze', 10, 0.0001)",
                          "ops.mpm('stage', 2, 2)", "ops.mpm('analyze', 20, 0.0001)"]


def test_the_stage_tag_is_the_material_tag_of_the_body_particles(
    tmp_path: Path,
) -> None:
    """The shared namespace: the tag ``mpm stage`` names is the
    ``MATERIAL_TAG`` the particle set gives that body's particles."""
    ops = _bridge()
    _staged(ops)
    ops.tcl(str(tmp_path / "case1.tcl"))
    tags = {int(ln.split()[2]) for ln in _mpm_words(
        (tmp_path / "case1.tcl").read_text(encoding="utf-8"), "mpm stage ")}
    with h5py.File(tmp_path / "case1.particles.h5", "r") as f:
        set_tags = set(f["PARTICLES"]["MATERIAL_TAG"][...].tolist())
    assert tags == set_tags == {ops.tag_for(ops._names["weak"])} == {2}


def test_the_archive_stores_the_stage_rows_among_the_analyze_rows(
    tmp_path: Path,
) -> None:
    ops = _bridge()
    _staged(ops)
    ops.h5(str(tmp_path / "model.h5"))
    rows = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))._mpm_rows()
    assert [r[0] for r in rows] == ["grid", "body", "gravity", "stage",
                                    "analyze", "stage", "analyze"]
    assert (rows[3], rows[5]) == (("stage", 2, 0), ("stage", 2, 2))


@pytest.mark.parametrize("dialect", ["tcl", "py"])
def test_the_replay_keeps_each_stage_in_its_place(tmp_path: Path, dialect: str) -> None:
    ops = _bridge()
    _staged(ops)
    getattr(ops, dialect)(str(tmp_path / f"orig.{dialect}"))
    ops.h5(str(tmp_path / "model.h5"))
    m = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))
    text = m.build(dialect, out=str(tmp_path / f"again.{dialect}"))
    orig = (tmp_path / f"orig.{dialect}").read_text(encoding="utf-8")
    prefix = "mpm " if dialect == "tcl" else "ops.mpm("

    def keep(t: str) -> list[str]:
        return [ln for ln in _mpm_words(t, prefix) if "particles" not in ln]

    assert keep(text) == keep(orig)
    assert [ln for ln in keep(text) if "stage" in ln or "analyze" in ln] == (
        ["mpm stage 2 0", "mpm analyze 10 0.0001", "mpm stage 2 2",
         "mpm analyze 20 0.0001"] if dialect == "tcl" else
        ["ops.mpm('stage', 2, 0)", "ops.mpm('analyze', 10, 0.0001)",
         "ops.mpm('stage', 2, 2)", "ops.mpm('analyze', 20, 0.0001)"])


def test_the_composed_replay_keeps_each_stage_in_its_place(tmp_path: Path) -> None:
    """#3001 found a composed-replay order bug; the oracle is the bridge's own
    flat deck of the same composed model, line for line."""
    ops = _bridge(fem=_composed_fem())
    _staged(ops)
    bridge = tmp_path / "bridge.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(bridge), flat=True)
        ops.h5(str(tmp_path / "model.h5"))
    m = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))
    assert any(r.method == "partition_open" for r in m.program())
    again = tmp_path / "again.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.build("tcl", out=str(again), flat=True)
    want = _deck_lines(bridge, "bridge")
    got = _deck_lines(again, "again")
    assert got == want
    assert [ln for ln in got if ln.startswith(("mpm stage", "mpm analyze"))] == [
        "mpm stage 2 0", "mpm analyze 10 0.0001", "mpm stage 2 2",
        "mpm analyze 20 0.0001"]


def test_stage_is_a_member_of_the_family() -> None:
    assert mpm_mod.Stage in mpm_mod.ALL_MPM
    assert "Stage" in mpm_mod.__all__
    assert ":class:`Stage`            ``mpm stage <matTag> <s>``" in (mpm_mod.__doc__ or "")
