"""RC wall cages — `g.rebar.wall()` (vertical + horizontal bars in one or two
curtains, through-thickness cross-ties). The 4th standardized member."""
from __future__ import annotations

import pytest

from apeGmsh import apeGmsh


def _count(a: float, b: float, spacing: float) -> int:
    """Mirror RebarComposite._positions_by_spacing's position count."""
    n = max(1, round((b - a) / spacing))
    return n + 1


def test_wall_double_curtain_grid():
    with apeGmsh(model_name="wall_dc") as g:
        cage = g.rebar.wall(
            length=4.0, thickness=0.25, height=3.0, cover=0.04,
            vertical_db=0.016, vertical_spacing=0.3,
            horizontal_db=0.012, horizontal_spacing=0.3, curtains=2)
    verts = [b for b in cage.bars if b.role == "vertical"]
    horis = [b for b in cage.bars if b.role == "horizontal"]
    nv = _count(0.04, 3.96, 0.3)
    nh = _count(0.04, 2.96, 0.3)
    assert len(verts) == 2 * nv                       # two curtains
    assert len(horis) == 2 * nh
    # two distinct curtain y-planes near each face
    near, far = 0.04 + 0.016 / 2.0, 0.25 - 0.04 - 0.016 / 2.0
    ys = {round(b.path.points[0][1], 9) for b in verts}
    assert ys == {round(near, 9), round(far, 9)}
    for b in verts:                                   # vertical: constant x,y; spans z
        p0, p1 = b.path.points
        assert p0[0] == p1[0] and p0[1] == p1[1]
        assert p0[2] == pytest.approx(0.04) and p1[2] == pytest.approx(2.96)
    for b in horis:                                   # horizontal: constant y,z; spans x
        p0, p1 = b.path.points
        assert p0[1] == p1[1] and p0[2] == p1[2]
        assert p0[0] == pytest.approx(0.04) and p1[0] == pytest.approx(3.96)


def test_wall_crossties_span_thickness():
    with apeGmsh(model_name="wall_ct") as g:
        cage = g.rebar.wall(
            length=4.0, thickness=0.25, height=3.0, cover=0.04,
            vertical_db=0.016, vertical_spacing=0.3,
            horizontal_db=0.012, horizontal_spacing=0.3, curtains=2,
            crosstie_spacing=0.6)
    cts = [b for b in cage.bars if b.role == "crosstie"]
    near, far = 0.04 + 0.016 / 2.0, 0.25 - 0.04 - 0.016 / 2.0
    assert len(cts) == _count(0.04, 3.96, 0.6) * _count(0.04, 2.96, 0.6)
    for b in cts:
        p0, p1 = b.path.points
        assert p0[0] == p1[0] and p0[2] == p1[2]      # constant x, z
        assert {round(p0[1], 9), round(p1[1], 9)} == {round(near, 9), round(far, 9)}
        assert {b.start_hook.angle, b.end_hook.angle} == {90.0, 135.0}


def test_wall_single_curtain_mid_thickness():
    with apeGmsh(model_name="wall_sc") as g:
        cage = g.rebar.wall(
            length=3.0, thickness=0.2, height=3.0, cover=0.04,
            vertical_db=0.012, vertical_spacing=0.25,
            horizontal_db=0.012, horizontal_spacing=0.25, curtains=1,
            crossties=False)
    assert not [b for b in cage.bars if b.role == "crosstie"]
    ys = {round(b.path.points[0][1], 9) for b in cage.bars}
    assert ys == {round(0.2 / 2.0, 9)}                # one mid-thickness curtain


def test_wall_single_curtain_crosstie_warns():
    with apeGmsh(model_name="wall_sc_ct") as g:
        with pytest.warns(UserWarning, match="nothing to tie"):
            cage = g.rebar.wall(
                length=3.0, thickness=0.2, height=3.0, cover=0.04,
                vertical_db=0.012, vertical_spacing=0.25,
                horizontal_db=0.012, horizontal_spacing=0.25, curtains=1,
                crossties=True)
        assert not [b for b in cage.bars if b.role == "crosstie"]


def test_wall_places_embedded_as_independent_members():
    with apeGmsh(model_name="wall_place") as g:
        vol = g.model.geometry.add_box(0, 0, 0, 4.0, 0.25, 3.0)
        g.physical.add_volume([vol], name="Wall")
        cage = g.rebar.wall(
            length=4.0, thickness=0.25, height=3.0, cover=0.04,
            vertical_db=0.016, vertical_spacing=0.5,
            horizontal_db=0.012, horizontal_spacing=0.5, curtains=2,
            crosstie_spacing=1.0)
        g.rebar.place(cage, into="Wall", coupling="embedded", perfect=1.0e8)
        assert len(g.reinforce.reinforce_defs) == len(cage.bars)


def test_wall_validation():
    with apeGmsh(model_name="wall_bad") as g:
        with pytest.raises(ValueError):                       # curtains 3
            g.rebar.wall(length=3.0, thickness=0.25, height=3.0, cover=0.04,
                         vertical_db=0.012, vertical_spacing=0.3,
                         horizontal_db=0.012, horizontal_spacing=0.3, curtains=3)
        with pytest.raises(ValueError):                       # thickness too thin
            g.rebar.wall(length=3.0, thickness=0.05, height=3.0, cover=0.04,
                         vertical_db=0.016, vertical_spacing=0.3,
                         horizontal_db=0.012, horizontal_spacing=0.3, curtains=2)
        with pytest.raises(ValueError):                       # spacing ≤ 0
            g.rebar.wall(length=3.0, thickness=0.25, height=3.0, cover=0.04,
                         vertical_db=0.012, vertical_spacing=0.0,
                         horizontal_db=0.012, horizontal_spacing=0.3)


def test_wall_conformal_meshes_end_to_end():
    with apeGmsh(model_name="wall_conf") as g:
        g.model.geometry.add_box(0, 0, 0, 3.0, 0.25, 3.0, label="Wall")
        cage = g.rebar.wall(
            length=3.0, thickness=0.25, height=3.0, cover=0.05,
            vertical_db=0.016, vertical_spacing=1.0,
            horizontal_db=0.012, horizontal_spacing=1.0, curtains=2,
            crossties=False)
        g.rebar.place(cage, into="Wall", coupling="conformal")
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data()
        assert fem.info.n_nodes > 0
