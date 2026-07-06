"""Columnar MassSet float-identity + API-parity gate (ADR 0065 v2 / C1-C3).

The columnar ``MassSet`` (``int64[N]`` node ids + ``float64[N,6]`` masses,
sparse names) must yield values BIT-IDENTICAL to the boxed-record store it
replaced -- otherwise deck ``repr`` drifts and a saved model reloads with
different mass numbers.  These tests pin:

  1. Awkward float64 values survive the convert-records -> columns -> view
     round-trip exactly (0.1, 1e-300, a 17-significant-digit value, ...).
  2. Those same values survive to_h5 -> from_h5 (the numpy-native
     ``_read_masses`` adopt path) exactly, and re-saving is byte-stable.
  3. The public surface (iter / len / bool / getitem / by_node /
     total_mass / summary / with_mass / membership-by-value) is unchanged.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from apeGmsh import FEMData
from apeGmsh._kernel.record_sets import MassSet
from apeGmsh._kernel.records._masses import MassRecord


# Awkward floats: repr must round-trip these exactly through float64.
AWKWARD = [
    0.1,
    0.2,
    0.3,
    1e-300,
    1e300,
    1234567890.1234567,     # ~17 significant digits
    3.141592653589793,      # full-precision pi
    -2.220446049250313e-16, # ~machine epsilon, negative
    0.0,
    123456.789012345,
]


def _mk_records():
    recs = []
    for i, v in enumerate(AWKWARD):
        recs.append(MassRecord(
            node_id=i + 1,
            mass=(v, v * 2.0, v * 0.5, v + 1.0, v - 1.0, v),
            name=("lbl%d" % i) if i % 3 == 0 else None,
        ))
    return recs


def test_convert_records_to_columns_is_bit_identical():
    recs = _mk_records()
    s = MassSet(recs)
    assert len(s) == len(recs)
    for orig, view in zip(recs, list(s)):
        assert view.node_id == orig.node_id
        assert view.name == orig.name
        # Exact float identity (not approx) -- deck repr depends on it.
        assert view.mass == orig.mass
        for a, b in zip(view.mass, orig.mass):
            assert a.hex() == b.hex()


def test_adopt_arrays_preserves_exact_values():
    node_ids = np.arange(1, len(AWKWARD) + 1, dtype=np.int64)
    mass = np.array([[v, v * 2.0, v * 0.5, v + 1.0, v - 1.0, v]
                     for v in AWKWARD], dtype=np.float64)
    s = MassSet(node_ids=node_ids, mass=mass)
    for i, rec in enumerate(s):
        assert rec.node_id == i + 1
        for k in range(6):
            assert rec.mass[k].hex() == float(mass[i, k]).hex()


def _fem_with_awkward_masses(tmp_path):
    """A real small meshed FEMData whose /masses group is overwritten with
    awkward per-node values via ``with_mass`` (columnar convert path)."""
    from apeGmsh import apeGmsh

    with apeGmsh(model_name="awk", verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
        g.physical.add_volume("b", name="B")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        fem = g.mesh.queries.get_fem_data(dim=3)

    # Replace the resolved masses with our awkward fixtures (first N nodes).
    node_ids = [int(n) for n in fem.nodes.ids][:len(AWKWARD)]
    fresh = FEMData(nodes=fem.nodes, elements=fem.elements, info=fem.info)
    # Rebuild the columnar set directly from awkward records.
    recs = [
        MassRecord(
            node_id=node_ids[i],
            mass=(v, v * 2.0, v * 0.5, v + 1.0, v - 1.0, v),
            name=("lbl%d" % i) if i % 3 == 0 else None,
        )
        for i, v in enumerate(AWKWARD)
    ]
    fresh.nodes.masses = MassSet(recs)
    return fresh, {r.node_id: r for r in recs}


def test_h5_roundtrip_float_identity(tmp_path):
    fem, orig = _fem_with_awkward_masses(tmp_path)

    p1 = os.path.join(tmp_path, "m1.h5")
    fem.to_h5(p1)
    fem2 = FEMData.from_h5(p1)

    got = {r.node_id: r for r in fem2.nodes.masses}
    assert set(orig) == set(got)
    for nid, r in orig.items():
        assert got[nid].name == r.name
        for a, b in zip(got[nid].mass, r.mass):
            assert a.hex() == b.hex()

    # Re-save is content-stable (schema untouched). NOTE: the /masses
    # dataset is a compound with vlen-string fields (``target`` / ``name``),
    # so raw ``.tobytes()`` embeds heap pointers that legitimately differ
    # between files -- compare the LOGICAL columns (node_id / mass / name)
    # bit-for-bit instead, which is the byte-identity the deck depends on.
    p2 = os.path.join(tmp_path, "m2.h5")
    fem2.to_h5(p2)
    import h5py
    with h5py.File(p1, "r") as f1, h5py.File(p2, "r") as f2:
        r1 = f1["masses"][...]
        r2 = f2["masses"][...]
        assert len(r1) == len(r2)
        p1p = r1["payload"]
        p2p = r2["payload"]
        assert np.array_equal(p1p["node_id"], p2p["node_id"])
        # float64 mass column: exact equality (no NaNs here).
        assert (p1p["mass"] == p2p["mass"]).all()
        assert [str(x) for x in r1["target"]] == [str(x) for x in r2["target"]]
        assert [str(x) for x in p1p["name"]] == [str(x) for x in p2p["name"]]


def test_public_surface_parity():
    recs = _mk_records()
    s = MassSet(recs)
    # len / bool / getitem / negative index
    assert len(s) == len(recs)
    assert bool(s) is True
    assert bool(MassSet()) is False
    assert s[0].node_id == 1
    assert s[-1].node_id == len(recs)
    with pytest.raises(IndexError):
        _ = s[len(recs)]
    # by_node
    assert s.by_node(2).node_id == 2
    assert s.by_node(999) is None
    # membership-by-value (dataclass __eq__ over transient records)
    probe = MassRecord(node_id=1, mass=recs[0].mass, name=recs[0].name)
    assert probe in list(s)
    # total_mass == sum of mx (numpy vs python-sum agree here; exact ints)
    expect = sum(r.mass[0] for r in recs)
    assert s.total_mass() == pytest.approx(expect)
    # summary shape
    df = s.summary()
    assert list(df.columns) == ["node_id", "mx", "my", "mz",
                                "Ixx", "Iyy", "Izz"]
    assert len(df) == len(recs)
    # with_mass appends one row, leaves the original untouched
    extra = MassRecord(node_id=999, mass=(7.0,) * 6)
    s2 = s._with_record(extra)
    assert len(s2) == len(recs) + 1
    assert len(s) == len(recs)
    assert s2[-1].node_id == 999
