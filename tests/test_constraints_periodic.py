"""``g.constraints.periodic`` — tied opposite boundaries (#2388).

``equal_dof`` pairs co-located nodes only, so a 1-D site-response column
had to tie its left and right sides with one ``equation_constraint`` per
elevation and DOF. ``periodic`` pairs the nodes of two sides whose
coordinates off the separation axis match, ties them with ``equalDOF``,
and refuses any side that does not pair node for node.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees


def _column(*, ny_left: int = 10, ny_right: int = 10, periodic=True,
            **kw):
    """A 1 x 10 quad column (Left x=0, Right x=1) with ``periodic`` ties."""
    g = apeGmsh(model_name="periodic_col", verbose=False)
    g.begin()
    G = g.model.geometry
    p = [G.add_point(0, 0, 0), G.add_point(1, 0, 0),
         G.add_point(1, 10, 0), G.add_point(0, 10, 0)]
    base, right = G.add_line(p[0], p[1]), G.add_line(p[1], p[2])
    top, left = G.add_line(p[2], p[3]), G.add_line(p[3], p[0])
    surf = G.add_plane_surface(G.add_curve_loop([base, right, top, left]))
    S = g.mesh.structured
    S.set_transfinite_curve(base, 2)
    S.set_transfinite_curve(top, 2)
    S.set_transfinite_curve(left, ny_left + 1)
    S.set_transfinite_curve(right, ny_right + 1)
    if ny_left == ny_right:
        S.set_transfinite_surface(surf)
        S.set_recombine(surf)
    g.physical.add_surface([surf], name="Soil")
    g.physical.add(1, [left], name="Left")
    g.physical.add(1, [right], name="Right")
    g.physical.add(1, [base], name="Base")
    if periodic:
        g.constraints.periodic("Left", "Right", axis="x", **kw)
    g.mesh.generation.generate(dim=2)
    return g


def test_column_sides_pair_at_every_elevation() -> None:
    g = _column(dofs=[1, 2], name="sides")
    try:
        fem = g.mesh.queries.get_fem_data(dim=2)
        recs = [c for c in fem.nodes.constraints if c.kind == "equal_dof"]
        xyz = {int(n): c for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
        assert len(recs) == 11
        for r in recs:
            m, s = xyz[r.master_node], xyz[r.slave_node]
            assert m[0] == pytest.approx(0.0) and s[0] == pytest.approx(1.0)
            assert m[1] == pytest.approx(s[1], abs=1e-9)
            assert list(r.dofs) == [1, 2] and r.name == "sides"
    finally:
        g.end()


def test_column_ties_emit_equal_dof(tmp_path: Path) -> None:
    g = _column(dofs=[1, 2])
    try:
        fem = g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3)
    ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(0, 1))
    out = tmp_path / "m.tcl"
    ops.tcl(str(out))
    lines = [ln.split() for ln in out.read_text().splitlines()
             if ln.strip().startswith("equalDOF ")]
    assert len(lines) == 11 and all(ln[3:] == ["1", "2"] for ln in lines)


@pytest.mark.filterwarnings("ignore:autosave")
def test_sides_meshed_differently_raise() -> None:
    g = _column(ny_left=10, ny_right=7)
    try:
        with pytest.raises(ValueError, match="do not pair node for node"):
            g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()


@pytest.mark.filterwarnings("ignore:autosave")
def test_a_node_on_both_sides_raises() -> None:
    g = apeGmsh(model_name="periodic_overlap", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="sq")
        g.physical.add_surface("sq", name="Sq")
        g.model.select(None, dim=1).on_plane(
            (0, 0, 0), (1, 0, 0), tol=1e-6).to_physical("Left")
        g.model.select(None, dim=1).on_plane(
            (0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Bottom")
        g.constraints.periodic("Left", "Bottom", axis="x")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=2)
        with pytest.raises(ValueError, match="on both sides"):
            g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()


def test_box_faces_pair_in_3d() -> None:
    g = apeGmsh(model_name="periodic_box", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="box")
        g.physical.add_volume("box", name="Box")
        for x, name in ((0.0, "X0"), (2.0, "X1")):
            g.model.select(None, dim=2).on_plane(
                (x, 0, 0), (1, 0, 0), tol=1e-6).to_physical(name)
        g.constraints.periodic("X0", "X1", axis="x", dofs=[1, 2, 3])
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.structured.set_transfinite_automatic()
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
        n_face = len(fem.nodes.select(pg="X0").ids)
        recs = [c for c in fem.nodes.constraints if c.kind == "equal_dof"]
        assert n_face > 4 and len(recs) == n_face
    finally:
        g.end()


@pytest.mark.parametrize("kw, match", [
    ({"axis": "w"}, "axis must be"),
    ({"axis": "x", "tolerance": 0.0}, "tolerance must be > 0"),
])
def test_bad_arguments_raise_at_the_call(kw, match) -> None:
    g = apeGmsh(model_name="periodic_bad", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="sq")
        with pytest.raises(ValueError, match=match):
            g.constraints.periodic("sq", "sq", **kw)
    finally:
        g.end()
