"""The SANISAND build-time gates, reached through ``ops.tcl``.

The unit tests in ``unit/test_manzari_tangent_solver_gate.py`` and
``unit/test_sanisand_deck_gates.py`` call the validators directly.  These
reach them the way a user does — a real Gmsh mesh,
``ops.nDMaterial.LadrunoSANISAND(...)``, and a deck written to disk — so a
gate wired to nothing would fail here even with every unit test green.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

pytest.importorskip("gmsh")

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402
from apeGmsh.opensees._internal.build import (  # noqa: E402
    BridgeError,
    ManzariConvergenceTestWarning,
    ManzariTangentSolverWarning,
)


_LS_KWARGS = {
    "G0": 264.32, "nu": 0.3129, "e_init": 0.6944, "Mc": 1.33090, "c": 0.71,
    "lambda_c": 0.027, "e0": 0.83, "ksi": 0.45, "P_atm": 101.0, "m": 0.005,
    "h0": 1.3, "Ch": 0.968, "nb": 3.5, "A0": 0.05, "nd": 5.75,
    "z_max": 12.5, "cz": 1100.0, "rho": 2.0,
}


def _sand_column(g: apeGmsh):
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 4.0, label="soil")
    g.physical.add_surface("soil", name="Soil")
    g.mesh.structured.set_recombine("soil", dim=2)
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(2)
    g.mesh.structured.recombine()
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data(dim=2)


def _bridge(
    fem, *, tan_type: int | None = None, max_substeps: int = 0,
) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    kw = {} if tan_type is None else {"tan_type": tan_type}
    if max_substeps:
        kw["max_substeps"] = max_substeps
    mat = ops.nDMaterial.LadrunoSANISAND(**_LS_KWARGS, **kw)
    ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=mat)
    ops.analysis.Static()
    return ops


def test_default_sanisand_on_profilespd_warns(tmp_path: Path) -> None:
    with apeGmsh(model_name="sand_spd") as g:
        ops = _bridge(_sand_column(g))
        ops.system.ProfileSPD()
        with pytest.warns(ManzariTangentSolverWarning, match="UNSYMMETRIC"):
            ops.tcl(str(tmp_path / "deck.tcl"))


def test_default_sanisand_on_umfpack_is_silent(tmp_path: Path) -> None:
    with apeGmsh(model_name="sand_umf") as g:
        ops = _bridge(_sand_column(g))
        ops.system.UmfPack()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ManzariTangentSolverWarning)
            ops.tcl(str(tmp_path / "deck.tcl"))


def test_no_system_declared_warns(tmp_path: Path) -> None:
    """The OpenSees no-``system`` default is ProfileSPD."""
    with apeGmsh(model_name="sand_nosys") as g:
        ops = _bridge(_sand_column(g))
        with pytest.warns(ManzariTangentSolverWarning, match="ProfileSPD"):
            ops.tcl(str(tmp_path / "deck.tcl"))


def test_elastic_tangent_on_profilespd_is_silent(tmp_path: Path) -> None:
    with apeGmsh(model_name="sand_elastic") as g:
        ops = _bridge(_sand_column(g), tan_type=0)
        ops.system.ProfileSPD()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ManzariTangentSolverWarning)
            ops.tcl(str(tmp_path / "deck.tcl"))


def test_emitted_deck_carries_the_explicit_tail(tmp_path: Path) -> None:
    """The tangent is a fact of the deck, not of the parser version."""
    out = tmp_path / "deck.tcl"
    with apeGmsh(model_name="sand_tail") as g:
        ops = _bridge(_sand_column(g))
        ops.system.UmfPack()
        ops.tcl(str(out))
    line = next(
        ln for ln in out.read_text(encoding="utf-8").splitlines()
        if "LadrunoSANISAND" in ln
    )
    # 18 positionals, then the tail (IntScheme TanType JacoType TolF TolR).
    assert " 1 2 1 1e-07 1e-07 " in line or " 1 2 1 1e-07 1e-07" in line


def test_normdispincr_on_a_sanisand_deck_warns(tmp_path: Path) -> None:
    with apeGmsh(model_name="sand_dispincr") as g:
        ops = _bridge(_sand_column(g))
        ops.system.UmfPack()
        ops.test.NormDispIncr(tol=1e-8, max_iter=200)
        with pytest.warns(
            ManzariConvergenceTestWarning, match="UNREACHABLE"
        ):
            ops.tcl(str(tmp_path / "deck.tcl"))


def test_normunbalance_on_a_sanisand_deck_is_silent(tmp_path: Path) -> None:
    with apeGmsh(model_name="sand_unbalance") as g:
        ops = _bridge(_sand_column(g))
        ops.system.UmfPack()
        ops.test.NormUnbalance(tol=1e-2, max_iter=200)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ManzariConvergenceTestWarning)
            ops.tcl(str(tmp_path / "deck.tcl"))


def test_uncapped_sanisand_emits_no_maxsubsteps(tmp_path: Path) -> None:
    """Default 0 must leave the deck line byte-identical."""
    out = tmp_path / "deck.tcl"
    with apeGmsh(model_name="sand_uncapped") as g:
        ops = _bridge(_sand_column(g))
        ops.system.UmfPack()
        ops.tcl(str(out))
    line = next(
        ln for ln in out.read_text(encoding="utf-8").splitlines()
        if "LadrunoSANISAND" in ln
    )
    assert "-maxSubsteps" not in line


def test_capped_sanisand_is_refused_under_a_quad(tmp_path: Path) -> None:
    """FourNodeQuad does not propagate a material refusal."""
    with apeGmsh(model_name="sand_capped_quad") as g:
        ops = _bridge(_sand_column(g), max_substeps=80)
        ops.system.UmfPack()
        with pytest.raises(
            BridgeError, match="propagate a material refusal"
        ):
            ops.tcl(str(tmp_path / "deck.tcl"))
