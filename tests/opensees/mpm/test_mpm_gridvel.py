"""Slice #3064: ``mpm.Gridvel`` and ``mpm.GridvelAffine`` (ADR 0134 Amendment 3).

The fork's ``mpm gridvel`` (``OPS_LadrunoMPM.cpp::cmdGridVel`` and
``LadrunoMPMImplicit.cpp::addGridVel`` / ``addGridVelAffine`` at
``ladruno-v20261118b``, cfe478254) prescribes grid velocities; mode F applies
them since fork #1109 (D18). Offline oracles here:

* every damage case of the slice card is refused by name, at the call or at
  the emit, before the fork sees the deck, and a refused call registers
  nothing;
* a deck carries the gridvel lines after the grid and before ``analyze``,
  in declaration order, and ``model.h5`` archives them as ``mpm`` rows;
* the fork's B-spline ghost-layer refusal (WP-232i B10) is unreachable from
  an apeGmsh deck because :class:`~apeGmsh.opensees.mpm.Grid` admits no
  B-spline basis; the tripwire below fails when that changes.

The fork-side behaviour (the particles move with the prescribed velocity) is
``tests/opensees/integration_ladruno/test_mpm_gridvel_live.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees import mpm as mpm_mod
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.material.nd import ElasticIsotropic

from ._meshes import grid_nodes, make_fem, quads

H = 0.1
BOX2 = ((0.4, -0.1), (1.6, 0.5))


def _bridge(ndm: int = 2) -> apeSees:
    nodes, ids = grid_nodes((4, 2), (0.5, 0.0), (1.5, 0.4))
    ops = apeSees(make_fem(nodes, {"Soil": quads(ids)}), units="kN_m_s",
                  _artifacts=False)
    ops.model(ndm=ndm, ndf=ndm)
    ops.nDMaterial.ElasticIsotropic(E=2.0e4, nu=0.25, name="soil")
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    ops.mpm.body("Soil", material="soil", density=1.8, ppc=2)
    ops.fix(pg="Soil", dofs=[1, 1])
    return ops


def _no_files(root: Path) -> None:
    assert sorted(p.name for p in root.rglob("*") if p.is_file()) == []


# -- the arguments, refused at construction ----------------------------------

def _gv(**kw: Any) -> mpm_mod.Gridvel:
    args: dict[str, Any] = dict(name="push", box=BOX2, v=(0.1, 0.0))
    args.update(kw)
    return mpm_mod.Gridvel(**args)


def _aff(**kw: Any) -> mpm_mod.GridvelAffine:
    args: dict[str, Any] = dict(name="shear", box=BOX2,
                                L=((0.0, 0.5), (0.0, 0.0)))
    args.update(kw)
    return mpm_mod.GridvelAffine(**args)


@pytest.mark.parametrize("make,match", [
    # box: wrong length, lo > hi, not a pair
    (lambda: _gv(box=((0.0,), (1.0,))), r"Gridvel\.box corners need 2 or 3"),
    (lambda: _gv(box=((0.0, 0.0), (1.0, 1.0, 1.0))), r"Gridvel\.box corners"),
    (lambda: _gv(box=((0.0, 0.6), (1.0, 0.5))), r"lo > hi on axis y"),
    (lambda: _gv(box=((0.0, 0.0, 2.0), (1.0, 1.0, 1.0)), v=(0, 0, 0)),
     r"lo > hi on axis z"),
    (lambda: _gv(box=(0.0, 1.0)), r"Gridvel\.box must be a pair"),
    (lambda: _gv(box=((0.0, float("nan")), (1.0, 1.0))), r"must be finite"),
    # v: wrong count, all free, a bad token
    (lambda: _gv(v=(0.1,)), r"Gridvel\.v needs 2 components"),
    (lambda: _gv(v=(0.1, 0.0, 0.0)), r"Gridvel\.v needs 2 components"),
    (lambda: _gv(v=("free", "free")), r"'free' on every component"),
    (lambda: _gv(v=(0.1, "fixed")), r"numbers or 'free', got 'fixed'"),
    (lambda: _gv(v=0.1), r"Gridvel\.v must be a sequence"),
    # profile with start; a bad trapezoid; a malformed profile
    (lambda: _gv(start=0.0, profile=(0.0, 0.1, 1.0)), r"profile= replaces start="),
    (lambda: _gv(profile=(0.0, -0.1, 1.0)), r"needs Tr >= 0 and t1 >= t0 \+ 2 Tr"),
    (lambda: _gv(profile=(0.0, 0.5, 0.9)), r"needs Tr >= 0 and t1 >= t0 \+ 2 Tr"),
    (lambda: _gv(profile=(0.0, 0.5)), r"Gridvel\.profile must be \(t0, Tr, t1\)"),
    # translate: an axis the box lacks, two axes, a free component
    (lambda: _gv(translate="z"), r"translate must be one of \('x', 'y'\)"),
    (lambda: _gv(translate="xy"), r"translate must be one of"),
    (lambda: _gv(v=("free", 0.0), translate="x"), r"the box would never move"),
    # the name is written bare into the deck line
    (lambda: _gv(name="two words"), r"Gridvel\.name .*one word"),
])
def test_gridvel_arguments_refuse_by_name(make: Callable[[], Any], match: str) -> None:
    with pytest.raises(mpm_mod.MPMError, match=match):
        make()


@pytest.mark.parametrize("make,match", [
    (lambda: _aff(L=((1.0, 0.0),)), r"GridvelAffine\.L must be 2x2"),
    (lambda: _aff(L=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))), r"GridvelAffine\.L must be 2x2"),
    (lambda: _aff(L=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))), r"GridvelAffine\.L must be 2x2"),
    (lambda: _aff(L=(1.0, 0.0, 0.0, 1.0)), r"GridvelAffine\.L must be 2 rows"),
    (lambda: _aff(ref=(0.0,)), r"GridvelAffine\.ref needs 2 coordinates"),
    (lambda: _aff(ref=(0.0, 0.0, 0.0)), r"GridvelAffine\.ref needs 2 coordinates"),
    (lambda: _aff(box=((1.0, 0.0), (0.0, 1.0))), r"lo > hi on axis x"),
    (lambda: _aff(name="a{b}"), r"brace"),
])
def test_gridvel_affine_arguments_refuse_by_name(
    make: Callable[[], Any], match: str,
) -> None:
    with pytest.raises(mpm_mod.MPMError, match=match):
        make()


def test_admitted_forms_write_the_fork_wire_format() -> None:
    """The card's wire format, token by token (``cmdGridVel`` at the pin)."""
    assert _gv(v=(0.1, "free"), start=0.25).args() == (
        "gridvel", "push", "box", 0.4, -0.1, 1.6, 0.5, "v", 0.1, "free",
        "start", 0.25)
    assert _gv(translate="x", profile=(0.0, 0.1, 1.0)).args() == (
        "gridvel", "push", "box", 0.4, -0.1, 1.6, 0.5, "v", 0.1, 0.0,
        "translate", "x", "profile", 0.0, 0.1, 1.0)
    # A degenerate box (a plane of nodes) and a zero-ramp profile are admitted.
    assert _gv(box=((0.5, 0.0), (0.5, 0.4)), profile=(0.0, 0.0, 0.0)).box == (
        (0.5, 0.0), (0.5, 0.4))
    assert _aff(ref=(0.0, 0.2)).args() == (
        "gridvel", "shear", "box", 0.4, -0.1, 1.6, 0.5, "affine",
        0.0, 0.5, 0.0, 0.0, "ref", 0.0, 0.2)
    three = mpm_mod.GridvelAffine(
        name="a3", box=((0, 0, 0), (1, 1, 1)),
        L=((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    assert three.args()[-9:] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)


def test_track_and_set_affine_are_not_offered() -> None:
    """The owner's ruling (#3064): no ``track`` (fork #1112 may change it),
    no ``setAffine`` (a deck declares one ``L``)."""
    with pytest.raises(TypeError, match="track"):
        mpm_mod.Gridvel(name="g", box=BOX2, v=(0.1, 0.0), track="x")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="track"):
        _bridge().mpm.gridvel("g", BOX2, (0.1, 0.0), track="x")  # type: ignore[call-arg]
    assert not hasattr(_bridge().mpm, "setAffine")


# -- the model-wide refusals, at the call or the emit ------------------------

def test_a_second_gridvel_of_one_name_refuses_at_the_call() -> None:
    ops = _bridge()
    ops.mpm.gridvel("g", BOX2, (0.1, 0.0))
    n = len(ops._primitives)
    with pytest.raises(mpm_mod.MPMError, match="a gridvel named 'g' is already"):
        ops.mpm.gridvel("g", BOX2, (0.2, 0.0))
    with pytest.raises(mpm_mod.MPMError, match="a gridvel named 'g' is already"):
        ops.mpm.gridvelAffine("g", BOX2, ((0.0, 1.0), (0.0, 0.0)))
    assert len(ops._primitives) == n


def test_a_second_gridvel_of_one_name_refuses_in_the_block() -> None:
    """Members registered without the namespace (the archive's rows, a
    hand-built list) meet the same refusal at the emit."""
    grid = mpm_mod.Grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    body = mpm_mod.Body(name="Soil", pg="Soil",
                        material=ElasticIsotropic(E=1.0e4, nu=0.3),
                        density=1.0, ppc=2)
    with pytest.raises(mpm_mod.MPMError, match=r"repeated: \['g'\]"):
        mpm_mod.mpm_block([grid, body, _gv(name="g"), _aff(name="g")], ndm=2)
    assert mpm_mod.mpm_block([grid, body, _gv(name="g"), _aff(name="h")],
                             ndm=2)[2:] == (_gv(name="g"), _aff(name="h"))


@pytest.mark.parametrize("which", ["gridvel", "gridvelAffine"])
def test_a_box_of_another_dimension_than_the_grid_refuses_at_emit(
    tmp_path: Path, which: str,
) -> None:
    ops = _bridge()
    box3 = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    if which == "gridvel":
        ops.mpm.gridvel("g3", box3, (0.1, 0.0, 0.0))
    else:
        ops.mpm.gridvelAffine("g3", box3, ((0, 0, 0),) * 3)
    with pytest.raises((BridgeError, mpm_mod.MPMError),
                       match=r"gridvel 'g3': its box .* is 3-D and the grid "
                             r"is 2-D"):
        ops.tcl(str(tmp_path / "case1.tcl"))
    _no_files(tmp_path)


def test_a_refused_call_registers_nothing() -> None:
    ops = _bridge()
    n = len(ops._primitives)
    with pytest.raises(mpm_mod.MPMError):
        ops.mpm.gridvel("g", BOX2, ("free", "free"))
    with pytest.raises(mpm_mod.MPMError):
        ops.mpm.gridvelAffine("g", BOX2, ((1.0,),))
    assert len(ops._primitives) == n


# -- the B-spline ghost-layer refusal (fork WP-232i B10) ---------------------

#: The fork's own message (``LadrunoMPMSoil.cpp::gridvelGhostConflict`` at
#: cfe478254) for a gridvel box that reaches the ghost layer behind a ``fix``
#: face on a B-spline grid, refused at ``mpm analyze``.
FORK_GHOST_REFUSAL = "reaches the ghost layer behind a `fix` face"


def test_the_bspline_ghost_case_is_unreachable_from_an_apegmsh_deck() -> None:
    """The fork refuses a gridvel box past a ``fix`` face on a ``bspline2`` /
    ``bspline3`` grid (``FORK_GHOST_REFUSAL``). apeGmsh cannot emit one: the
    grid admits gimp and linear only. When a B-spline basis is admitted, this
    test fails, and ``mpm_block`` must gain the static check (the box beyond
    ``origin`` / ``origin + cells * h`` on a fixed face) in the same PR."""
    assert mpm_mod.BASES == ("gimp", "linear")
    for basis in ("bspline2", "bspline3"):
        with pytest.raises(mpm_mod.MPMError, match="Grid.basis"):
            mpm_mod.Grid(origin=(0.0, 0.0), h=H, cells=(4, 4), basis=basis)
    assert "ghost layer behind a ``fix`` face" in (mpm_mod.mpm_block.__doc__ or "")


def test_the_docstrings_say_mode_f_only() -> None:
    for cls in (mpm_mod.Gridvel, mpm_mod.GridvelAffine):
        assert "**Mode F only" in (cls.__doc__ or "")


# -- the deck and the archive -------------------------------------------------

def _declare(ops: apeSees) -> None:
    ops.mpm.gravity(0.0, -9.81)
    ops.mpm.gridvel("push", BOX2, (0.1, "free"), translate="x",
                    profile=(0.0, 0.001, 0.01))
    ops.mpm.gridvelAffine("shear", BOX2, ((0.0, 0.5), (0.0, 0.0)), ref=(0.0, 0.2))
    ops.mpm.analyze(10, 1.0e-4)


def test_the_tcl_deck_writes_the_gridvel_lines_before_analyze(tmp_path: Path) -> None:
    ops = _bridge()
    _declare(ops)
    ops.tcl(str(tmp_path / "case1.tcl"))
    lines = [ln for ln in (tmp_path / "case1.tcl").read_text(encoding="utf-8")
             .splitlines() if ln.startswith("mpm ")]
    assert [ln.split()[1] for ln in lines] == [
        "grid", "particles", "gravity", "gridvel", "gridvel", "analyze"]
    assert lines[3] == ("mpm gridvel push box 0.4 -0.1 1.6 0.5 v 0.1 free "
                        "translate x profile 0.0 0.001 0.01")
    assert lines[4] == ("mpm gridvel shear box 0.4 -0.1 1.6 0.5 affine "
                        "0.0 0.5 0.0 0.0 ref 0.0 0.2")


def test_the_py_deck_writes_the_gridvel_calls(tmp_path: Path) -> None:
    ops = _bridge()
    _declare(ops)
    ops.py(str(tmp_path / "case1.py"))
    text = (tmp_path / "case1.py").read_text(encoding="utf-8")
    assert ("ops.mpm('gridvel', 'push', 'box', 0.4, -0.1, 1.6, 0.5, 'v', 0.1, "
            "'free', 'translate', 'x', 'profile', 0.0, 0.001, 0.01)") in text
    assert ("ops.mpm('gridvel', 'shear', 'box', 0.4, -0.1, 1.6, 0.5, 'affine', "
            "0.0, 0.5, 0.0, 0.0, 'ref', 0.0, 0.2)") in text


def test_the_archive_stores_the_gridvel_rows(tmp_path: Path) -> None:
    ops = _bridge()
    _declare(ops)
    ops.h5(str(tmp_path / "model.h5"))
    rows = OpenSeesModel.from_h5(str(tmp_path / "model.h5"))._mpm_rows()
    assert [r[0] for r in rows] == ["grid", "body", "gravity", "gridvel",
                                    "gridvel", "analyze"]
    assert rows[3] == _gv(v=(0.1, "free"), translate="x",
                          profile=(0.0, 0.001, 0.01)).args()
    assert rows[4] == _aff(ref=(0.0, 0.2)).args()
