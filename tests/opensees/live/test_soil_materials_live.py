"""Live oracles for the site-response soil nDMaterials (#2386, #2387).

A 1 x 10 m plane-strain column of ten quads under its own weight, base
fixed. In its elastic stage 0 a ``PressureIndependMultiYield`` with
moduli ``(G, K)`` is linear elastic, so the column must settle exactly
as an ``ElasticIsotropic`` one with ``E = 9KG / (3K + G)`` and
``nu = (3K - 2G) / (2 (3K + G))``. A staged deck flips the same material
0 -> 1 between two gravity stages, sides confined, and must run to
completion in openseespy.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.apesees import apeSees

pytestmark = pytest.mark.live

G = 6.0e4      # kPa
K = 3.0e5      # kPa
RHO = 1.8      # t/m^3
GAMMA = 18.0   # kN/m^3


@pytest.fixture(autouse=True)
def _backend() -> None:
    pytest.importorskip("openseespy.opensees")


@pytest.fixture(scope="module")
def column() -> Any:
    g = apeGmsh(model_name="soil_live", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 10.0, label="soil")
        g.physical.add_surface("soil", name="Soil")
        g.model.select(None, dim=1).on_plane(
            (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), tol=1e-6).to_physical("Base")
        g.model.select(None, dim=1).on_plane(
            (0.0, 10.0, 0.0), (0.0, 1.0, 0.0), tol=1e-6).to_physical("Top")
        for x, name in ((0.0, "Left"), (1.0, "Right")):
            g.model.select(None, dim=1).on_plane(
                (x, 0.0, 0.0), (1.0, 0.0, 0.0), tol=1e-6).to_physical(name)
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.structured.set_transfinite_automatic()
        g.mesh.generation.generate(dim=2)
        return g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()


def _chain(ops: apeSees, *, staged: bool = False) -> dict[str, object]:
    # The plastic stage's multi-yield return leaves a ~0.1 kN unbalance
    # floor, so the staged deck converges on the displacement increment.
    test = (ops.test.NormDispIncr(tol=1e-6, max_iter=30) if staged
            else ops.test.NormUnbalance(tol=1e-8, max_iter=20))
    return {
        "test": test,
        "algorithm": ops.algorithm.Newton(),
        "integrator": ops.integrator.LoadControl(dlam=1.0),
        "constraints": ops.constraints.Plain(),
        "numberer": ops.numberer.RCM(),
        "system": ops.system.UmfPack(),
        "analysis": ops.analysis.Static(),
    }


def _top_uy(fem: Any, mat_kind: str, **kw: float) -> float:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = getattr(ops.nDMaterial, mat_kind)(**kw)
    ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=mat,
                             body_force=(0.0, -GAMMA))
    ops.fix(pg="Base", dofs=(1, 1))
    _chain(ops)  # the chain primitives register as they are made
    assert ops.analyze(steps=1) == 0
    osp = get_ops()
    top = [int(n) for n in fem.nodes.select(pg="Top").ids]
    return sum(osp.nodeDisp(n, 2) for n in top) / len(top)


def test_elastic_stage_pimy_settles_like_elastic_isotropic(column) -> None:
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    ref = _top_uy(column, "ElasticIsotropic", E=E, nu=nu, rho=RHO)
    got = _top_uy(column, "PressureIndependMultiYield", nd=2, rho=RHO,
                  ref_shear_modulus=G, ref_bulk_modulus=K, cohesion=40.0)
    assert ref < 0.0
    assert got == pytest.approx(ref, rel=1e-6)


def test_staged_flip_deck_runs(column, tmp_path: Path) -> None:
    ops = apeSees(column)
    ops.model(ndm=2, ndf=2)
    clay = ops.nDMaterial.PressureIndependMultiYield(
        nd=2, rho=RHO, ref_shear_modulus=G, ref_bulk_modulus=K,
        cohesion=40.0)
    ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=clay,
                             body_force=(0.0, -GAMMA))
    ops.fix(pg="Base", dofs=(1, 1))
    # Oedometric sides, as in a site column: the clay stays elastic
    # under its own weight once plastic.
    ops.fix(pg="Left", dofs=(1, 0))
    ops.fix(pg="Right", dofs=(1, 0))
    with ops.stage(name="elastic") as s:
        s.update_material_stage(materials=[clay], stage=0)
        s.analysis(**_chain(ops, staged=True))
        s.run(n_increments=1)
    with ops.stage(name="plastic") as s:
        s.update_material_stage(materials=[clay], stage=1)
        s.analysis(**_chain(ops, staged=True))
        s.run(n_increments=1)
    deck = tmp_path / "soil_staged.py"
    report = ops.py(str(deck), run=True, python=sys.executable)
    assert report is not None and report.returncode == 0
    text = deck.read_text(encoding="utf-8")
    assert text.count("updateMaterialStage") == 2
