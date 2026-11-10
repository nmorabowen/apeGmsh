"""Stock ``SSPquadUP`` end to end over a real Gmsh 1 x 10 quad column.

Mesh-only lanes (gmsh, no OpenSees run):

* the Tcl and openseespy element lines carry ``OPS_SSPquadUP``'s
  positional order with the body pair;
* per-node ndf inference: under an ``ndf=3`` envelope no node carries a
  ``-ndf`` token; under ``ndf=2`` every soil node gets ``-ndf 3``;
* ``s.update_parameter("hPerm"/"vPerm", pg=...)`` emits an
  ``addToParameter ... element <tag> hPerm`` per element.

Live lanes (``live``; stock openseespy is enough):

* gravity on a drained-top column reaches hydrostatic pore pressure
  ``p = gamma_w * depth``, read as ``nodeVel(node, 3)`` (the element's
  integral-p contract: ``nodeDisp(node, 3)`` holds the time integral);
* the staged permeability update takes effect: an undrained first stage
  builds ``p`` near the saturated overburden, and only the stage that
  raises ``hPerm`` / ``vPerm`` drains it back to hydrostatic.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("gmsh")

from apeGmsh import apeGmsh  # noqa: E402
from apeGmsh.opensees import apeSees  # noqa: E402

H = 10.0          # column height (m); 10 unit quads
GAMMA_W = 9.81    # fluid density 1.0 t/m^3 times g (kPa / m)
E, NU = 1.0e5, 0.3
K_SKEL = E / (3.0 * (1.0 - 2.0 * NU))
G_SKEL = E / (2.0 * (1.0 + NU))
ALPHA = 0.25 * 1.0**2 / (K_SKEL + 4.0 * G_SKEL / 3.0)   # McGann et al.


def _column_fem(g: apeGmsh):
    """A transfinite 1 x H column of unit quads: pgs Soil, Base, Top."""
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, H, label="soil")
    g.physical.add_surface("soil", name="Soil")
    g.model.select(dim=1).on_plane(
        (0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Base")
    g.model.select(dim=1).on_plane(
        (0, H, 0), (0, 1, 0), tol=1e-6).to_physical("Top")
    g.mesh.structured.set_transfinite("soil", size=1.0)
    g.mesh.generation.generate(2)
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data(dim=2)


def _column(fem, *, ndf: int = 3, perm: float = 1e-3) -> apeSees:
    """The u-p column: laterally confined, base fixed, top drained."""
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=ndf)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, rho=2.0)
    ops.element.SSPquadUP(
        pg="Soil", material=mat, thickness=1.0, fluid_bulk=2.2e6,
        fluid_density=1.0, perm=(perm, perm), void_ratio=0.7,
        alpha=ALPHA, body=(0.0, -9.81),
    )
    ops.fix(pg="Soil", dofs=(1, 0, 0))
    ops.fix(pg="Base", dofs=(1, 1, 0))
    ops.fix(pg="Top", dofs=(1, 0, 1))      # DOF 3 fixed: p = 0, drained
    return ops


def _chain(ops: apeSees) -> dict:
    return dict(
        test=ops.test.NormDispIncr(tol=1e-10, max_iter=20),
        algorithm=ops.algorithm.Newton(),
        integrator=ops.integrator.Newmark(gamma=0.6, beta=0.3025),
        constraints=ops.constraints.Transformation(),
        numberer=ops.numberer.RCM(),
        system=ops.system.UmfPack(),
        analysis=ops.analysis.Transient(),
    )


def _declare_chain(ops: apeSees) -> None:
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=20)
    ops.algorithm.Newton()
    ops.integrator.Newmark(gamma=0.6, beta=0.3025)
    ops.analysis.Transient()


def _staged(fem, *, raise_perm: bool) -> apeSees:
    ops = _column(fem, perm=1e-9)
    with ops.stage(name="undrained") as s:
        s.analysis(**_chain(ops))
        s.run(n_increments=20, dt=0.1)
    with ops.stage(name="drain") as s:
        if raise_perm:
            s.update_parameter("hPerm", 1e-3, pg="Soil")
            s.update_parameter("vPerm", 1e-3, pg="Soil")
        s.analysis(**_chain(ops))
        s.run(n_increments=50, dt=0.1)
    return ops


def _lines(text: str, head: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines()
            if ln.strip().startswith(head)]


# ---------------------------------------------------------------------------
# Mesh-only lanes
# ---------------------------------------------------------------------------

class TestDeck:
    def test_tcl_and_py_element_lines(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="sspup_deck") as g:
            ops = _column(_column_fem(g))
            _declare_chain(ops)
            ops.tcl(str(tmp_path / "d.tcl"))
            ops.py(str(tmp_path / "d.py"))
        tcl = (tmp_path / "d.tcl").read_text(encoding="utf-8")
        py = (tmp_path / "d.py").read_text(encoding="utf-8")
        tail = f"1.0 2200000.0 1.0 0.001 0.001 0.7 {ALPHA!r} 0.0 -9.81"
        ele = _lines(tcl, "element SSPquadUP")
        assert len(ele) == 10
        for ln in ele:
            tok = ln.split()
            # element SSPquadUP tag n1..n4 matTag (8 tokens) + the 7
            # parser doubles + b1 b2 = 17: the parser's 15-argument form.
            assert len(tok) == 17, ln
            assert ln.endswith(tail), ln
        pyl = _lines(py, "ops.element('SSPquadUP'")
        assert len(pyl) == 10
        assert pyl[0].endswith(
            f"1.0, 2200000.0, 1.0, 0.001, 0.001, 0.7, {ALPHA!r}, 0.0, -9.81)"
        ), pyl[0]

    def test_h5_replay_reemits_the_same_element_lines(
        self, tmp_path: Path,
    ) -> None:
        from apeGmsh.opensees import OpenSeesModel

        with apeGmsh(model_name="sspup_h5") as g:
            ops = _column(_column_fem(g))
            _declare_chain(ops)
            ops.tcl(str(tmp_path / "fwd.tcl"))
            ops.h5(str(tmp_path / "model.h5"))
        fwd = (tmp_path / "fwd.tcl").read_text(encoding="utf-8")
        replay = OpenSeesModel.from_h5(str(tmp_path / "model.h5")).build("tcl")
        assert isinstance(replay, str)
        assert _lines(replay, "element SSPquadUP") == _lines(
            fwd, "element SSPquadUP")
        assert len(_lines(replay, "element SSPquadUP")) == 10

    def test_ndf_three_inferred_per_node(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="sspup_ndf") as g:
            fem = _column_fem(g)
            ops3 = _column(fem, ndf=3)
            ops3.tcl(str(tmp_path / "n3.tcl"))
            ops2 = _column(fem, ndf=2)
            ops2.tcl(str(tmp_path / "n2.tcl"))
        n3 = _lines((tmp_path / "n3.tcl").read_text(encoding="utf-8"), "node ")
        n2 = _lines((tmp_path / "n2.tcl").read_text(encoding="utf-8"), "node ")
        assert len(n3) == 22 and all("-ndf" not in ln for ln in n3)
        assert len(n2) == 22 and all(ln.endswith("-ndf 3") for ln in n2)

    def test_update_parameter_hperm_vperm_emitted(
        self, tmp_path: Path,
    ) -> None:
        with apeGmsh(model_name="sspup_param") as g:
            ops = _staged(_column_fem(g), raise_perm=True)
            ops.tcl(str(tmp_path / "s.tcl"))
        tcl = (tmp_path / "s.tcl").read_text(encoding="utf-8")
        add = _lines(tcl, "addToParameter")
        assert sum(ln.endswith(" hPerm") for ln in add) == 10, add
        assert sum(ln.endswith(" vPerm") for ln in add) == 10, add
        assert all(" element " in ln for ln in add)
        upd = _lines(tcl, "updateParameter")
        assert len(upd) == 2 and all(ln.endswith(" 0.001") for ln in upd)


# ---------------------------------------------------------------------------
# Live lanes (stock openseespy)
# ---------------------------------------------------------------------------

def _base_and_depths(live, node_ids) -> list[tuple[int, float]]:
    """(node, depth) for the x = 0 nodes of the column."""
    out = []
    for n in node_ids:
        x, y = live.nodeCoord(int(n))
        if abs(x) < 1e-9:
            out.append((int(n), H - y))
    return sorted(out, key=lambda t: t[1])


@pytest.mark.live
def test_column_reaches_hydrostatic_p_read_from_vel() -> None:
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import get_ops

    with apeGmsh(model_name="sspup_live") as g:
        fem = _column_fem(g)
        ops = _column(fem)
        _declare_chain(ops)
        assert ops.analyze(steps=50, dt=0.1) == 0
        live = get_ops()
        pts = _base_and_depths(live, fem.nodes.ids)

    assert len(pts) == 11
    for n, depth in pts:
        p = float(live.nodeVel(n, 3))
        assert p == pytest.approx(GAMMA_W * depth, rel=1e-4, abs=1e-6), (
            n, depth, p)
    # The displacement slot of DOF 3 is the time integral of p, not p.
    n_base, d_base = pts[-1]
    assert float(live.nodeDisp(n_base, 3)) > 2.0 * GAMMA_W * d_base


@pytest.mark.live
@pytest.mark.parametrize("raise_perm", [True, False])
def test_staged_perm_update_takes_effect(
    tmp_path: Path, raise_perm: bool,
) -> None:
    pytest.importorskip("openseespy.opensees")
    with apeGmsh(model_name="sspup_staged") as g:
        fem = _column_fem(g)
        ops = _staged(fem, raise_perm=raise_perm)
        deck = tmp_path / "staged.py"
        ops.py(str(deck))
        base = min(
            (int(n) for n in fem.nodes.select(pg="Base").ids),
            key=lambda t: float(fem.nodes.coords[
                list(fem.nodes.ids).index(t)][0]),
        )
    glb = runpy.run_path(str(deck), run_name="__main__")
    live = glb["ops"]
    p_base = float(live.nodeVel(base, 3))
    hydro = GAMMA_W * H
    if raise_perm:
        assert p_base == pytest.approx(hydro, rel=1e-3)
    else:
        # k = 1e-9 stays undrained: p near the saturated overburden.
        assert p_base > 1.5 * hydro


# ---------------------------------------------------------------------------
# alpha_per_area: per-element stabilization on a graded column
# ---------------------------------------------------------------------------

M_SKEL = K_SKEL + 4.0 * G_SKEL / 3.0
ALPHA_PER_AREA = 0.25 / M_SKEL           # McGann: alpha_e = 0.25 A_e / M


def _graded_column_fem(g: apeGmsh):
    """A 1 x H column of 10 cells graded by 1.3 along its sides: areas differ."""
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, H, label="soil")
    g.physical.add_surface("soil", name="Soil")
    g.model.select(dim=1).on_plane(
        (0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Base")
    g.model.select(dim=1).on_plane(
        (0, H, 0), (0, 1, 0), tol=1e-6).to_physical("Top")
    g.model.select(dim=1).on_plane(
        (0, 0, 0), (1, 0, 0), tol=1e-6).to_physical("Left")
    g.model.select(dim=1).on_plane(
        (1, 0, 0), (1, 0, 0), tol=1e-6).to_physical("Right")
    g.mesh.structured.set_transfinite_curve(["Base", "Top"], 2)
    g.mesh.structured.set_transfinite_curve(["Left", "Right"], 11, coef=1.3)
    g.mesh.structured.set_transfinite_surface("soil")
    g.mesh.structured.set_recombine("soil")
    g.mesh.generation.generate(2)
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data(dim=2)


def _graded_column(fem) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, rho=2.0)
    ops.element.SSPquadUP(
        pg="Soil", material=mat, thickness=1.0, fluid_bulk=2.2e6,
        fluid_density=1.0, perm=(1e-3, 1e-3), void_ratio=0.7,
        alpha_per_area=ALPHA_PER_AREA, body=(0.0, -9.81),
    )
    ops.fix(pg="Soil", dofs=(1, 0, 0))
    ops.fix(pg="Base", dofs=(1, 1, 0))
    ops.fix(pg="Top", dofs=(1, 0, 1))
    _declare_chain(ops)
    return ops


def _tcl_alphas_vs_areas(tcl: str) -> tuple[list[float], list[float]]:
    """(emitted alpha, shoelace area) per ``element SSPquadUP`` line."""
    xy: dict[int, tuple[float, float]] = {}
    for ln in _lines(tcl, "node "):
        tok = ln.split()
        xy[int(tok[1])] = (float(tok[2]), float(tok[3]))
    alphas, areas = [], []
    for ln in _lines(tcl, "element SSPquadUP"):
        tok = ln.split()
        quad = [xy[int(t)] for t in tok[3:7]]
        area = 0.5 * abs(sum(
            quad[i][0] * quad[(i + 1) % 4][1]
            - quad[(i + 1) % 4][0] * quad[i][1] for i in range(4)))
        alphas.append(float(tok[14]))
        areas.append(area)
    return alphas, areas


class TestAlphaPerArea:
    def test_each_line_carries_its_own_area(self, tmp_path: Path) -> None:
        with apeGmsh(model_name="sspup_alpha_area") as g:
            ops = _graded_column(_graded_column_fem(g))
            ops.tcl(str(tmp_path / "a.tcl"))
            ops.h5(str(tmp_path / "a.h5"))
        tcl = (tmp_path / "a.tcl").read_text(encoding="utf-8")
        alphas, areas = _tcl_alphas_vs_areas(tcl)
        assert len(alphas) == 10
        # graded: the cells really differ (the two sides grade in opposite
        # directions, so the cells are trapezoids, about 1.8x apart)
        assert max(areas) / min(areas) > 1.5
        for alpha, area in zip(alphas, areas):
            assert alpha == pytest.approx(ALPHA_PER_AREA * area, rel=1e-9)

        from apeGmsh.opensees import OpenSeesModel
        replay = OpenSeesModel.from_h5(str(tmp_path / "a.h5")).build("tcl")
        assert isinstance(replay, str)
        assert _lines(replay, "element SSPquadUP") == _lines(
            tcl, "element SSPquadUP")


@pytest.mark.live
def test_graded_column_alpha_per_area_reaches_hydrostatic() -> None:
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import get_ops

    with apeGmsh(model_name="sspup_alpha_live") as g:
        fem = _graded_column_fem(g)
        ops = _graded_column(fem)
        assert ops.analyze(steps=50, dt=0.1) == 0
        live = get_ops()
        pts = _base_and_depths(live, fem.nodes.ids)
    assert len(pts) == 11
    for n, depth in pts:
        assert float(live.nodeVel(n, 3)) == pytest.approx(
            GAMMA_W * depth, rel=1e-4, abs=1e-6)
