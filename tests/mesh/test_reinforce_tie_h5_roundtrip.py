"""Neutral-zone H5 round-trip for embedded-reinforcement ties (ADR 0067 P5.1).

`fem.elements.reinforce_ties` (g.reinforce `LadrunoEmbeddedRebar` couplings)
now persist through `FEMData.to_h5` / `from_h5` into a dedicated
`/reinforce_ties` group (neutral schema 2.15.0). Previously dropped with a
deferral warning. Built on a real non-matching mesh (no fork build needed)."""
from __future__ import annotations

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh._femdata_h5_io import NEUTRAL_SCHEMA_VERSION


def _build_rebar_in_tet(g, *, x0=0.5, y0=0.5, z_lo=0.2, z_hi=0.8, size=0.4):
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    p0 = gmsh.model.occ.addPoint(x0, y0, z_lo)
    p1 = gmsh.model.occ.addPoint(x0, y0, z_hi)
    ln = gmsh.model.occ.addLine(p0, p1)
    g.model.sync()
    g.mesh.sizing.set_global_size(size)
    g.mesh.generation.generate(3)
    g.physical.add(3, [box], name="concrete")
    g.physical.add(1, [ln], name="rebar")


def _reinforced_fem(**reinforce_kw):
    with apeGmsh(model_name="p5_h5", verbose=False) as g:
        _build_rebar_in_tet(g)
        g.reinforce(host="concrete", bars="rebar", **reinforce_kw)
        return g.mesh.queries.get_fem_data(dim=3)


def _plain_fem():
    with apeGmsh(model_name="p5_h5_plain", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="concrete")
        return g.mesh.queries.get_fem_data(dim=3)


def _eq(a, b):
    assert a.rebar_node == b.rebar_node
    assert list(a.host_nodes) == list(b.host_nodes)
    assert a.name == b.name and a.bond == b.bond
    assert a.enforce == b.enforce and a.bipenalty == b.bipenalty
    assert a.in_bounds == b.in_bounds
    for f in ("bond_scale", "perfect", "kt", "kt_alpha", "dtcr", "excess"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            assert x == pytest.approx(y), f
    for f in ("weights", "direction"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            assert np.allclose(np.asarray(x), np.asarray(y)), f


def _roundtrip(fem, tmp_path):
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    p = str(tmp_path / "m.h5")
    fem.to_h5(p)
    return read_fem_h5(p), p


def test_perfect_bond_ties_roundtrip(tmp_path):
    fem = _reinforced_fem(perfect=1.0e12, bar_diameter=0.025)
    src = sorted(fem.elements.reinforce_ties, key=lambda t: t.rebar_node)
    assert len(src) >= 2
    back, _ = _roundtrip(fem, tmp_path)
    got = sorted(back.elements.reinforce_ties, key=lambda t: t.rebar_node)
    assert len(got) == len(src)
    for a, b in zip(got, src):
        _eq(a, b)
        assert a.perfect == pytest.approx(1.0e12) and a.bond is None


def test_bond_by_name_ties_roundtrip(tmp_path):
    fem = _reinforced_fem(bond="bond1", bar_diameter=0.02,
                          kt=1.0e7, kt_alpha=0.5)
    src = sorted(fem.elements.reinforce_ties, key=lambda t: t.rebar_node)
    assert src and all(t.bond == "bond1" for t in src)
    back, _ = _roundtrip(fem, tmp_path)
    got = sorted(back.elements.reinforce_ties, key=lambda t: t.rebar_node)
    assert len(got) == len(src)
    for a, b in zip(got, src):
        _eq(a, b)
        assert a.bond == "bond1" and a.perfect is None
        assert a.bond_scale is not None and a.kt == pytest.approx(1.0e7)


def test_reinforced_snapshot_id_stable_on_roundtrip(tmp_path):
    # snapshot_id excludes the tie overlay (consistent with constraints), so a
    # reinforced model round-trips with an identical id even though the ties
    # are now read back into the broker.
    fem = _reinforced_fem(perfect=1.0e12, bar_diameter=0.025)
    back, _ = _roundtrip(fem, tmp_path)
    assert back.snapshot_id == fem.snapshot_id
    assert len(back.elements.reinforce_ties) == len(fem.elements.reinforce_ties)
    assert back.elements.reinforce_ties                       # really present


def test_tie_free_model_omits_group_and_keeps_snapshot(tmp_path):
    import h5py
    fem = _plain_fem()
    assert not fem.elements.reinforce_ties
    back, p = _roundtrip(fem, tmp_path)
    with h5py.File(p, "r") as f:
        assert "reinforce_ties" not in f                      # group omitted
    assert back.snapshot_id == fem.snapshot_id


def test_to_h5_no_deferral_warning(tmp_path, recwarn):
    fem = _reinforced_fem(perfect=1.0e12, bar_diameter=0.025)
    fem.to_h5(str(tmp_path / "m.h5"))
    assert not [w for w in recwarn.list
                if "not persisted" in str(w.message)
                or "deferred" in str(w.message)]


def test_writer_stamps_2_15_0():
    assert NEUTRAL_SCHEMA_VERSION == "2.15.0"
