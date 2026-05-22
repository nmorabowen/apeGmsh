"""ADR 0018 C4 — ``ModelData.from_h5`` round-trip + enrich.

Acceptance gate from
``opensees/architecture/modeldata-enrichment-scope.md`` §3 C4:

* Round-trip — load → rewrite is fixed-point on the orientation pair
  (INV-9: ``fem_eids ↔ per_element_emitted_tag ↔ args`` row order
  preserved).
* Opaque ``snapshot_id`` carry-through (INV-8 — never recomputed).
* Enrich-in-place — ``from_h5`` then ``oriented_elements`` then
  ``write`` preserves prior records and appends the new ones.
* Optional-child probe uses ``in`` / H5Lexists, not ``Group.get()``
  (INV-15 — the ``project_h5py_optional_child_get_hazard`` pattern).

Uses ``FEMStub`` with ``FEMData.from_h5`` monkey-patched to bypass the
neutral-zone re-load (FEMStub does not produce a broker zone on
disk).  The orientation rehydrate logic is what C4 verifies; the
broker re-load is covered by ``tests/test_femdata_from_h5.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import ModelData
from apeGmsh.opensees.emitter import h5_reader

from tests.fixtures.schema import OPENSEES_CURRENT
from tests.opensees.fixtures.fem_stub import make_two_node_beam


@pytest.fixture()
def stub_fem_h5_loader(monkeypatch):
    """Monkeypatch ``FEMData.from_h5`` to return a fresh FEMStub.

    Sidesteps the neutral-zone re-load that FEMStub cannot satisfy
    (no ``snapshot_id`` attribute → broker writer is skipped on
    ModelData.write → file has no ``/nodes``/``/elements``).
    """
    from apeGmsh.mesh.FEMData import FEMData as _FEMData

    def _loader(cls, path):  # noqa: ANN001
        return make_two_node_beam()
    monkeypatch.setattr(_FEMData, "from_h5", classmethod(_loader))
    return _loader


# ---------------------------------------------------------------------------
# Round-trip fixed-point on the orientation pair (INV-9)
# ---------------------------------------------------------------------------

def test_round_trip_orientation_join_fixed_point(
    tmp_path: Path, stub_fem_h5_loader,
) -> None:
    fem = make_two_node_beam()
    md1 = ModelData(cast("object", fem), ndm=3, ndf=6, model_name="rt")
    md1.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    p1 = tmp_path / "before.h5"
    md1.write(str(p1))

    with h5_reader.open(str(p1)) as m:
        join1 = m.element_local_axes_vecxz()

    # Reload and rewrite without changes.
    md2 = ModelData.from_h5(str(p1))
    p2 = tmp_path / "after.h5"
    md2.write(str(p2))

    with h5_reader.open(str(p2)) as m:
        join2 = m.element_local_axes_vecxz()

    assert set(join1.keys()) == set(join2.keys())
    for k in join1:
        np.testing.assert_array_equal(join1[k], join2[k])


def test_round_trip_preserves_two_transforms_row_order(
    tmp_path: Path, stub_fem_h5_loader,
) -> None:
    """INV-9 specifically: appending records in disk-read order keeps
    ``fem_eids`` / ``args`` rows aligned with their transf tags."""
    from tests.opensees.fixtures.fem_stub import (
        _ElementGroupView, _ElementsStub, _NodesStub, FEMStub,
    )
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
        node_pgs={},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Col_A": _ElementGroupView(ids=(1,), connectivity=((1, 2),)),
            "Col_B": _ElementGroupView(ids=(2,), connectivity=((3, 4),)),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)

    # Override the monkeypatched loader for this test to return THIS fem.
    from apeGmsh.mesh.FEMData import FEMData as _FEMData
    _FEMData.from_h5 = classmethod(lambda cls, p: fem)  # type: ignore[attr-defined]

    md1 = ModelData(cast("object", fem), ndm=3, ndf=6)
    md1.oriented_elements(
        pg="Col_A", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    md1.oriented_elements(
        pg="Col_B", ele_type="forceBeamColumn", vecxz=(0.0, 1.0, 0.0),
    )
    p1 = tmp_path / "two_before.h5"
    md1.write(str(p1))

    md2 = ModelData.from_h5(str(p1))
    p2 = tmp_path / "two_after.h5"
    md2.write(str(p2))

    with h5_reader.open(str(p1)) as m1, h5_reader.open(str(p2)) as m2:
        j1 = m1.element_local_axes_vecxz()
        j2 = m2.element_local_axes_vecxz()

    assert j1.keys() == j2.keys() == {1, 2}
    np.testing.assert_array_equal(j1[1], j2[1])
    np.testing.assert_array_equal(j1[2], j2[2])
    # And the orientations themselves are preserved.
    np.testing.assert_array_equal(j2[1], (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(j2[2], (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# INV-8 — opaque snapshot_id carry-through
# ---------------------------------------------------------------------------

def test_snapshot_id_opaque_carry_through(
    tmp_path: Path, stub_fem_h5_loader,
) -> None:
    fem = make_two_node_beam()
    md1 = ModelData(cast("object", fem), ndm=3, ndf=6)
    md1.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    p1 = tmp_path / "snap.h5"
    md1.write(str(p1))

    # Stamp a known snapshot_id directly on the file.  This simulates
    # a file written elsewhere (e.g. ``apeSees(fem).h5()`` with a real
    # FEMData) whose hash must NOT be recomputed by ``ModelData``.
    UNIQUE = "TEST-SNAPSHOT-DEADBEEF-12345"
    with h5py.File(str(p1), "r+") as f:
        f["meta"].attrs["snapshot_id"] = UNIQUE

    md2 = ModelData.from_h5(str(p1))
    assert md2._loaded_snapshot_id == UNIQUE

    p2 = tmp_path / "snap_after.h5"
    md2.write(str(p2))

    with h5py.File(str(p2), "r") as f:
        sid = f["meta"].attrs["snapshot_id"]
        if isinstance(sid, bytes):
            sid = sid.decode("utf-8")
        assert sid == UNIQUE, (
            "INV-8: snapshot_id must be carried opaque through the "
            "round-trip, never recomputed."
        )


# ---------------------------------------------------------------------------
# Enrich-in-place after from_h5
# ---------------------------------------------------------------------------

def test_enrich_after_from_h5_overrides_for_same_eid(
    tmp_path: Path, stub_fem_h5_loader,
) -> None:
    """Reload + add a new oriented_elements for the same fem_eid: the
    new vecxz wins in the join (last-write-wins per fem_eid, since
    the reader iterates element_meta rows in disk order)."""
    fem = make_two_node_beam()
    md1 = ModelData(cast("object", fem), ndm=3, ndf=6)
    md1.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(1.0, 0.0, 0.0),
    )
    p1 = tmp_path / "v1.h5"
    md1.write(str(p1))

    md2 = ModelData.from_h5(str(p1))
    md2.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=(0.0, 1.0, 0.0),
    )
    p2 = tmp_path / "v2.h5"
    md2.write(str(p2))

    with h5_reader.open(str(p2)) as m:
        join = m.element_local_axes_vecxz()
    assert set(join.keys()) == {1}
    np.testing.assert_array_equal(join[1], (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------------
# INV-15 — optional-child probe behaviour
# ---------------------------------------------------------------------------

def test_from_h5_tolerates_missing_opensees_zone(
    tmp_path: Path, stub_fem_h5_loader,
) -> None:
    """A bridge-only file with no ``/opensees`` zone at all must load
    cleanly (returns a ModelData with no orientation records)."""
    # Write a file that has /meta but no /opensees children at all.
    p = tmp_path / "meta_only.h5"
    with h5py.File(str(p), "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = OPENSEES_CURRENT
        meta.attrs["ndm"] = 3
        meta.attrs["ndf"] = 6
        meta.attrs["model_name"] = "meta_only"
        meta.attrs["snapshot_id"] = ""

    md = ModelData.from_h5(str(p))
    assert md._em._transforms == []
    assert md._em._elements == []
