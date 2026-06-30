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
    assert a.corot == b.corot
    for f in ("bond_scale", "perfect", "kt", "kt_alpha", "dtcr", "excess"):
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            assert x == pytest.approx(y), f
    for f in ("weights", "direction", "shape_b"):
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


def test_corot_ties_roundtrip(tmp_path):
    # corot=True ⇒ each tie carries shape_b (point-B weights parallel to
    # host_nodes); the whole tie + corot columns survive the round-trip.
    fem = _reinforced_fem(perfect=1.0e12, bar_diameter=0.025, corot=True)
    src = sorted(fem.elements.reinforce_ties, key=lambda t: t.rebar_node)
    assert src and all(t.corot for t in src)
    assert all(t.shape_b is not None and len(t.shape_b) == len(t.host_nodes)
               for t in src)
    back, _ = _roundtrip(fem, tmp_path)
    got = sorted(back.elements.reinforce_ties, key=lambda t: t.rebar_node)
    assert len(got) == len(src)
    for a, b in zip(got, src):
        _eq(a, b)
        assert a.corot is True and a.shape_b is not None


def test_non_corot_ties_have_no_shape_b(tmp_path):
    # The frozen-axis default ⇒ corot=False, shape_b=None (round-trips so).
    fem = _reinforced_fem(perfect=1.0e12, bar_diameter=0.025)
    back, _ = _roundtrip(fem, tmp_path)
    got = back.elements.reinforce_ties
    assert got and all(t.corot is False and t.shape_b is None for t in got)


def test_decode_presence_probes_corot_columns():
    # A row whose payload predates the 2.26.0 corot columns (a genuine 2.25.x
    # file) must still decode → corot off. Exercise the presence-probe by
    # dropping the corot/shape_b fields from a freshly encoded payload.
    from numpy.lib import recfunctions as rfn

    from apeGmsh._kernel.records._constraints import ReinforceTieRecord
    from apeGmsh.mesh._femdata_h5_io import (
        _decode_reinforce_tie,
        _encode_reinforce_tie,
    )
    from apeGmsh.mesh._record_h5 import reinforce_tie_payload_dtype

    rec = ReinforceTieRecord(
        kind="reinforce", rebar_node=9, host_nodes=[1, 2, 3, 4],
        weights=np.full(4, 0.25), direction=np.array([0.0, 0.0, 1.0]),
        perfect=1.0e12)
    full = np.zeros((1,), dtype=reinforce_tie_payload_dtype())
    full[0] = _encode_reinforce_tie(rec)
    drop = [n for n in full.dtype.names
            if n in ("corot", "shape_b", "has_shape_b")]
    trimmed = rfn.drop_fields(full, drop)
    assert "corot" not in trimmed.dtype.names
    row = np.zeros((1,), dtype=[("payload", trimmed.dtype)])
    row["payload"] = trimmed
    got = _decode_reinforce_tie(row[0], ReinforceTieRecord)
    assert got.corot is False and got.shape_b is None
    assert got.rebar_node == 9 and got.perfect == pytest.approx(1.0e12)


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


def test_writer_stamps_current_neutral_version():
    from tests.fixtures.schema import NEUTRAL_CURRENT
    assert NEUTRAL_SCHEMA_VERSION == NEUTRAL_CURRENT


# ── adversarial-review hardening (C0/C1/C2 + C5) ─────────────────────

def _bad_tie(**over):
    from apeGmsh._kernel.records._constraints import ReinforceTieRecord
    base = dict(kind="reinforce", rebar_node=9, host_nodes=[1, 2, 3, 4],
                weights=np.full(4, 0.25), direction=np.array([0.0, 0.0, 1.0]),
                perfect=1.0e12)
    base.update(over)
    return ReinforceTieRecord(**base)


def test_encode_rejects_empty_host_nodes():
    from apeGmsh.mesh._femdata_h5_io import _encode_reinforce_tie
    with pytest.raises(ValueError, match="host_nodes is empty"):
        _encode_reinforce_tie(_bad_tie(host_nodes=[], weights=None))


def test_encode_rejects_mismatched_weights():
    from apeGmsh.mesh._femdata_h5_io import _encode_reinforce_tie
    with pytest.raises(ValueError, match="weights length"):
        _encode_reinforce_tie(_bad_tie(host_nodes=[1, 2, 3, 4],
                                       weights=np.full(3, 1.0 / 3)))


def test_encode_rejects_empty_weights_array():
    from apeGmsh.mesh._femdata_h5_io import _encode_reinforce_tie
    with pytest.raises(ValueError, match="empty array"):
        _encode_reinforce_tie(_bad_tie(weights=np.empty(0)))


def test_reads_prior_minor_file_without_ties_group_within_window(tmp_path):
    # An in-window prior-minor file with the /reinforce_ties group stripped
    # must still read → empty ties. (Versions older than the reader's
    # two-version window are rejected; see tests.fixtures.schema.)
    import h5py

    from tests.fixtures.schema import NEUTRAL_PRIOR_MINOR
    fem = _reinforced_fem(perfect=1.0e12, bar_diameter=0.025)
    p = str(tmp_path / "old.h5")
    fem.to_h5(p)
    with h5py.File(p, "r+") as f:
        f["meta"].attrs["schema_version"] = NEUTRAL_PRIOR_MINOR
        f["meta"].attrs["neutral_schema_version"] = NEUTRAL_PRIOR_MINOR
        if "reinforce_ties" in f:
            del f["reinforce_ties"]
    from apeGmsh.mesh._femdata_h5_io import read_fem_h5
    back = read_fem_h5(p)                               # within window → no raise
    assert back.elements.reinforce_ties == []          # absent group → no ties
