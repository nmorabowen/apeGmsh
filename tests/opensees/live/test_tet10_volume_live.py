"""TenNodeTetrahedron against the closed form, per build — the stock defect.

One block, nu=0, base fixed, top pushed down by delta: K = E*A/H exactly,
for any mesh that represents uniform strain. Upstream
``TenNodeTetrahedron::shp3d`` applies the tetrahedral 1/6 twice
(``xsj = Jdet``, where ``Jdet`` is already the element volume), so stock
reads exactly 1/6 of it — stiffness, mass, body force and reactions alike.
The fork fixed it in PR #520 (``xsj = 6.0*Jdet``). This is why
``LiveOpsEmitter`` refuses the element on stock.

If a stock wheel ever ships the fix, the stock test here fails: lift the
refusal in ``emitter/live.py`` (``_tet10_volume_fixed``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.emitter import live

pytestmark = pytest.mark.live

E, NU, SIDE, H, DELTA = 200_000.0, 0.0, 10.0, 10.0, 0.01
K_EXACT = E * SIDE * SIDE / H


def _ops():
    try:
        return live._get_ops()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")


def _block_stiffness(tmp_path: Path, order: int, element: str) -> float:
    from apeGmsh.opensees import apeSees

    with apeGmsh(model_name="blk", save_to=str(tmp_path / "b.h5"),
                 overwrite=True) as g:
        g.model.geometry.add_box(0.0, 0.0, 0.0, SIDE, SIDE, H, label="v")
        g.physical.add_volume("v", name="V")
        for z, name in ((0.0, "Bot"), (H, "Top")):
            faces = (g.model.select(None, dim=2)
                     .in_box((-1, -1, z - 0.01), (SIDE + 1, SIDE + 1, z + 0.01))
                     .result().tags())
            g.physical.add_surface(faces, name=name)
        g.mesh.recipe.unstructured(max_size=6.0)
        if order == 2:
            g.mesh.generation.set_order(2, bubble=False)
        fem = g.mesh.queries.get_fem_data(dim=None)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU)
    getattr(ops.element, element)(pg="V", material=mat)
    ops.fix(pg="Bot", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        pat.sp(pg="Top", dof=3, value=-DELTA)
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = live.LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    lv = emitter.ops
    lv.reactions()
    r_z = sum(lv.nodeReaction(int(t), 3) for t in fem.nodes.select(pg="Bot").ids)
    lv.wipe()
    return abs(r_z) / DELTA


def test_tet4_block_matches_closed_form(tmp_path: Path) -> None:
    """Control: FourNodeTetrahedron is right on every build."""
    _ops()
    k = _block_stiffness(tmp_path, 1, "FourNodeTetrahedron")
    assert k == pytest.approx(K_EXACT, rel=1e-9)


def test_tet10_block_is_one_sixth_on_stock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stock is WRONG: exactly 1/6 of E*A/H (the engine behind the refusal)."""
    if live._tet10_volume_fixed(_ops()) is not False:
        pytest.skip("stock-only: this build is not the defective upstream one")
    monkeypatch.setattr(live, "_tet10_volume_fixed", lambda ops: True)
    k = _block_stiffness(tmp_path, 2, "TenNodeTetrahedron")
    assert k / K_EXACT == pytest.approx(1 / 6, rel=1e-9)


@pytest.mark.ladruno_fork
def test_tet10_block_matches_closed_form_on_a_fixed_fork(tmp_path: Path) -> None:
    """The fork (PR #520) is RIGHT: E*A/H."""
    if live._tet10_volume_fixed(_ops()) is not True:
        pytest.skip("fork build predates the ladrunoBuild stamp: cannot "
                    "confirm the TenNodeTetrahedron fix (fork PR #520)")
    k = _block_stiffness(tmp_path, 2, "TenNodeTetrahedron")
    assert k == pytest.approx(K_EXACT, rel=1e-9)


def test_live_emitter_refuses_tet10_on_stock(tmp_path: Path) -> None:
    if live._tet10_volume_fixed(_ops()) is not False:
        pytest.skip("stock-only")
    with pytest.raises(RuntimeError, match="6x too small"):
        _block_stiffness(tmp_path, 2, "TenNodeTetrahedron")
    live.LiveOpsEmitter(wipe=True)   # leave a clean domain behind
