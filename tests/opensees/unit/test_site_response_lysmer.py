"""``lysmer_base`` — the compliant viscous base helper (#2391).

What the helper registers on the bridge (dashpots, ground fixity, the
vertical fixity it skips under a tie) and the input forces it adds. The
wave-propagation oracle is ``tests/opensees/live/test_lysmer_base_live.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.site_response import (
    _tied_slave_dofs,
    base_tributary_area,
    lysmer_base,
)

W = 2.0


@pytest.fixture(scope="module")
def strip() -> Any:
    """A 2 x 1 quad strip, base meshed with two line elements."""
    g = apeGmsh(model_name="lysmer_unit", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, W, 1.0, label="soil")
        g.physical.add_surface("soil", name="Soil")
        g.model.select(None, dim=1).on_plane(
            (0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Base")
        gnd = g.decouple_node_set("Base", label="base_ground")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.structured.set_transfinite_automatic()
        g.mesh.generation.generate(dim=2)
        return g.mesh.queries.get_fem_data(dim=2), gnd
    finally:
        g.end()


def _ops(fem: Any) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3, rho=2.0)
    ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=mat)
    return ops


def test_tributary_lengths_share_each_segment(strip) -> None:
    fem, _ = strip
    area = base_tributary_area(fem, "Base", ndm=2, thickness=0.5)
    assert sum(area.values()) == pytest.approx(W * 0.5)
    # Uniform segments: each end node carries half an interior node's.
    vals = sorted(area.values())
    assert vals[0] == pytest.approx(vals[1]) == pytest.approx(0.5 * vals[2])
    assert vals[2] == pytest.approx(vals[-1])


def test_dashpots_ground_fix_and_vertical_fix(strip, tmp_path: Path) -> None:
    fem, gnd = strip
    ops = _ops(fem)
    base = lysmer_base(ops, "Base", gnd, rho=2.0, vs=300.0)
    assert sum(base.c_s) == pytest.approx(2.0 * 300.0 * W)
    n = len(fem.nodes.select(pg="Base").ids)
    assert base.c_p is None and len(base.elements) == len(base.nodes) == n
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    text = deck.read_text()
    assert text.count("element zeroLength ") == n
    assert text.count("uniaxialMaterial Viscous ") == 2   # end, interior
    for g_tag in base.ground:
        assert f"fix {g_tag} 1 1" in text
    for n in base.nodes:
        assert f"fix {n} 0 1" in text


def test_vp_puts_a_dashpot_on_the_vertical_instead(strip, tmp_path) -> None:
    fem, gnd = strip
    ops = _ops(fem)
    base = lysmer_base(ops, "Base", gnd, rho=2.0, vs=300.0, vp=600.0)
    assert base.c_p is not None
    assert sum(base.c_p) == pytest.approx(2.0 * 600.0 * W)
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    assert not any(f"fix {n} 0 1" in deck.read_text() for n in base.nodes)


def test_add_input_loads_c_or_2c(strip, tmp_path: Path) -> None:
    fem, gnd = strip
    for motion, scale in (("outcrop", 1.0), ("incident", 2.0)):
        ops = _ops(fem)
        base = lysmer_base(ops, "Base", gnd, rho=2.0, vs=300.0)
        series = ops.timeSeries.Path(dt=0.01, values=(0.0, 1.0, 0.0))
        with ops.pattern.Plain(series=series) as p:
            base.add_input(p, motion=motion)
        deck = tmp_path / f"{motion}.tcl"
        ops.tcl(str(deck))
        loads = [ln.split() for ln in deck.read_text().splitlines()
                 if ln.strip().startswith("load ")]
        got = {int(ln[1]): float(ln[2]) for ln in loads}
        assert got == pytest.approx(
            {n: scale * c for n, c in zip(base.nodes, base.c_s)})


def test_tied_slave_dofs_are_read_from_equal_dof_records() -> None:
    recs = [SimpleNamespace(kind="equal_dof", slave_node=7, dofs=[1, 2]),
            SimpleNamespace(kind="rigid_beam", slave_node=8, dofs=[1])]
    fem = SimpleNamespace(nodes=SimpleNamespace(constraints=recs))
    assert _tied_slave_dofs(fem) == {(7, 1), (7, 2)}


@pytest.mark.parametrize("kw, err, match", [
    ({"vs": 0.0}, ValueError, "vs must be > 0"),
    ({"vp": 100.0}, ValueError, "vp must exceed vs"),
])
def test_bad_values_raise(strip, kw, err, match) -> None:
    fem, gnd = strip
    with pytest.raises(err, match=match):
        lysmer_base(_ops(fem), "Base", gnd, **{"rho": 2.0, "vs": 300.0, **kw})


def test_ground_must_be_a_node_set_over_the_base(strip) -> None:
    fem, gnd = strip
    with pytest.raises(TypeError, match="decouple_node_set"):
        lysmer_base(_ops(fem), "Base", 7, rho=2.0, vs=300.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is not a node label"):
        lysmer_base(_ops(fem), "Base", "nowhere", rho=2.0, vs=300.0)
    with pytest.raises(ValueError, match="not straight"):  # a surface
        lysmer_base(_ops(fem), "Soil", gnd, rho=2.0, vs=300.0)


def test_3d_base_face_gets_two_horizontal_dashpots(tmp_path: Path) -> None:
    g = apeGmsh(model_name="lysmer_3d", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="box")
        g.physical.add_volume("box", name="Box")
        g.model.select(None, dim=2).on_plane(
            (0, 0, 0), (0, 0, 1), tol=1e-6).to_physical("Base")
        gnd = g.decouple_node_set("Base", label="base_ground")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.structured.set_transfinite_automatic()
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3, rho=2.0)
    ops.element.stdBrick(pg="Box", material=mat)
    base = lysmer_base(ops, "Base", gnd, rho=2.0, vs=300.0, vp=600.0)
    assert base.horizontal == (1, 2) and base.vertical == 3
    assert sum(base.area) == pytest.approx(2.0)
    deck = tmp_path / "m.tcl"
    ops.tcl(str(deck))
    zl = [ln for ln in deck.read_text().splitlines()
          if ln.strip().startswith("element zeroLength ")]
    assert len(zl) == len(base.nodes)
    assert all("-dir 1 2 3" in ln for ln in zl)


def test_ground_label_pairs_by_position_after_a_reload(strip, tmp_path) -> None:
    """A model reloaded from model.h5 has the set's label, not its handle.
    The label is a node label of the snapshot once decoupled-node labels
    reach it (#2389); without that the label is unknown and this skips."""
    from apeGmsh.mesh.FEMData import FEMData

    fem, gnd = strip
    path = tmp_path / "model.h5"
    fem.to_h5(str(path))
    back = FEMData.from_h5(str(path))
    if "base_ground" not in back.nodes.labels:
        pytest.skip("decoupled-node labels are not on the snapshot (#2389)")
    by_label = lysmer_base(_ops(back), "Base", "base_ground", rho=2.0,
                           vs=300.0)
    by_handle = lysmer_base(_ops(fem), "Base", gnd, rho=2.0, vs=300.0)
    assert by_label.ground == by_handle.ground
    assert by_label.c_s == pytest.approx(by_handle.c_s)


def test_partly_fixed_tie_warns() -> None:
    from apeGmsh.opensees.site_response import _warn_partly_fixed_ties, PartlyFixedTieWarning

    recs = [SimpleNamespace(kind="equal_dof", master_node=1, slave_node=2,
                            dofs=[1, 2])]
    fem = SimpleNamespace(nodes=SimpleNamespace(constraints=recs))
    with pytest.warns(PartlyFixedTieWarning, match="#2396"):
        _warn_partly_fixed_ties(fem, {1}, 2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_partly_fixed_ties(fem, {3}, 2)   # master not fixed: silent
        recs[0].dofs = [2]
        _warn_partly_fixed_ties(fem, {1}, 2)   # only the fixed DOF: fine


def test_quadratic_base_is_refused(tmp_path) -> None:
    g = apeGmsh(model_name="lysmer_q", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0.0, 0.0, 0.0, W, 1.0, label="soil")
        g.physical.add_surface("soil", name="Soil")
        g.model.select(None, dim=1).on_plane(
            (0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Base")
        gnd = g.decouple_node_set("Base", label="base_ground")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.generation.generate(dim=2)
        g.mesh.generation.set_order(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()
    with pytest.raises(ValueError, match="quadratic base"):
        lysmer_base(_ops(fem), "Base", gnd, rho=2.0, vs=300.0)


def test_ndf_override_for_soil_under_a_wider_model(strip, tmp_path) -> None:
    fem, gnd = strip
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3, rho=2.0)
    ops.element.FourNodeQuad(pg="Soil", thickness=1.0, material=mat)
    base = lysmer_base(ops, "Base", gnd, rho=2.0, vs=300.0, ndf=2)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        base.add_input(p)
    ops.tcl(str(tmp_path / "m.tcl"))
    assert f"fix {base.ground[0]} 1 1" in (tmp_path / "m.tcl").read_text()


def test_equation_rows_count_as_ties(strip, tmp_path) -> None:
    fem, gnd = strip
    ops = _ops(fem)
    nodes = sorted(base_tributary_area(fem, "Base", ndm=2))
    ops.equation_constraint(constrained=(nodes[1], 2),
                            retained=[(nodes[0], 2, -1.0)])
    base = lysmer_base(ops, "Base", gnd, rho=2.0, vs=300.0)
    ops.tcl(str(tmp_path / "m.tcl"))
    text = (tmp_path / "m.tcl").read_text()
    assert f"fix {nodes[1]} 0 1" not in text
    assert f"fix {nodes[0]} 0 1" in text and len(base.nodes) == len(nodes)
