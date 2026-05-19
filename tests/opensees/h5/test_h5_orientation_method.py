"""ADR 0018 C2 — ``H5Emitter.add_oriented_elements`` round-trip.

Verifies that records appended by the schema-owning declarative writer
method are recovered by ``h5_reader.element_local_axes_vecxz()`` with
the correct ``{fem_eid: vecxz}`` mapping, and that the fail-loud guards
(ADR 0018 INV-6/7/9) raise at inject time.

This is the C2 acceptance test from
``opensees/architecture/modeldata-enrichment-scope.md`` §3 C2.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import H5Emitter


def _write_minimal(emitter: H5Emitter, path) -> None:
    """Drive bridge-only ``/meta`` + ``/opensees/...`` into ``path``.

    No broker neutral zone is needed for the orientation join — the
    reader's ``element_local_axes_vecxz`` reads only the two
    ``/opensees/...`` groups (see h5_reader.py:309-363).
    """
    with h5py.File(path, "w") as f:
        emitter._write_meta(f)
        emitter.write_opensees_into(f)


# ---------------------------------------------------------------------------
# Round-trip: writer → reader recovers {fem_eid: vecxz}
# ---------------------------------------------------------------------------

def test_round_trip_force_beam_column(tmp_path) -> None:
    em = H5Emitter(model_name="t", snapshot_id="snap")
    em.model(ndm=3, ndf=6)
    em.add_oriented_elements(
        type_token="forceBeamColumn",
        vecxz=(0.0, 1.0, 0.0),
        elements=[(101, (1, 2)), (102, (2, 3))],
        ndm=3,
    )
    p = tmp_path / "m.h5"
    _write_minimal(em, p)

    with h5_reader.open(str(p)) as mdl:
        join = mdl.element_local_axes_vecxz()

    assert set(join.keys()) == {101, 102}
    for eid in (101, 102):
        np.testing.assert_array_equal(join[eid], (0.0, 1.0, 0.0))


def test_round_trip_two_transforms_distinct_vecxz(tmp_path) -> None:
    em = H5Emitter(model_name="t", snapshot_id="snap")
    em.model(ndm=3, ndf=6)
    em.add_oriented_elements(
        type_token="forceBeamColumn",
        vecxz=(0.0, 0.0, 1.0),
        elements=[(10, (1, 2))],
        ndm=3,
    )
    em.add_oriented_elements(
        type_token="forceBeamColumn",
        vecxz=(0.0, 1.0, 0.0),
        elements=[(20, (3, 4))],
        ndm=3,
    )
    p = tmp_path / "m.h5"
    _write_minimal(em, p)

    with h5_reader.open(str(p)) as mdl:
        join = mdl.element_local_axes_vecxz()

    assert set(join.keys()) == {10, 20}
    np.testing.assert_array_equal(join[10], (0.0, 0.0, 1.0))
    np.testing.assert_array_equal(join[20], (0.0, 1.0, 0.0))


def test_round_trip_elastic_beam_column_3d(tmp_path) -> None:
    """elasticBeamColumn lives in `_ELEM_REGISTRY` — exercises the
    slot-via-registry path, not the `_FORCE_DISP_BEAMS` short-circuit."""
    em = H5Emitter(model_name="t", snapshot_id="snap")
    em.model(ndm=3, ndf=6)
    em.add_oriented_elements(
        type_token="elasticBeamColumn",
        vecxz=(0.0, 0.0, 1.0),
        elements=[(7, (1, 2))],
        ndm=3,
    )
    p = tmp_path / "m.h5"
    _write_minimal(em, p)

    with h5_reader.open(str(p)) as mdl:
        join = mdl.element_local_axes_vecxz()

    assert set(join.keys()) == {7}
    np.testing.assert_array_equal(join[7], (0.0, 0.0, 1.0))


def test_round_trip_elastic_beam_column_2d(tmp_path) -> None:
    """ndm=2 selects ``slots_2d``, which has a different transf slot
    index than the 3D variant — verifies the writer honors ``ndm``."""
    em = H5Emitter(model_name="t", snapshot_id="snap")
    em.model(ndm=2, ndf=3)
    em.add_oriented_elements(
        type_token="elasticBeamColumn",
        vecxz=(0.0, 0.0, 1.0),
        elements=[(99, (1, 2))],
        ndm=2,
    )
    p = tmp_path / "m.h5"
    _write_minimal(em, p)

    with h5_reader.open(str(p)) as mdl:
        join = mdl.element_local_axes_vecxz()

    assert set(join.keys()) == {99}
    np.testing.assert_array_equal(join[99], (0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Fail-loud guards (ADR 0018 INV-6 / INV-7 / INV-9)
# ---------------------------------------------------------------------------

def test_raises_on_unknown_type_token() -> None:
    em = H5Emitter(model_name="t", snapshot_id="")
    em.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="no transf slot"):
        em.add_oriented_elements(
            type_token="not_a_real_element",
            vecxz=(0.0, 0.0, 1.0),
            elements=[(1, (1, 2))],
            ndm=3,
        )


def test_raises_on_non_beam_type_token() -> None:
    """A registry entry with ``needs_transf=False`` must be rejected."""
    em = H5Emitter(model_name="t", snapshot_id="")
    em.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="no transf slot"):
        em.add_oriented_elements(
            type_token="FourNodeTetrahedron",
            vecxz=(0.0, 0.0, 1.0),
            elements=[(1, (1, 2, 3, 4))],
            ndm=3,
        )


def test_raises_on_bad_ndm() -> None:
    em = H5Emitter(model_name="t", snapshot_id="")
    em.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="ndm must be 2 or 3"):
        em.add_oriented_elements(
            type_token="forceBeamColumn",
            vecxz=(0.0, 0.0, 1.0),
            elements=[(1, (1, 2))],
            ndm=4,
        )


def test_raises_on_bad_vecxz_length() -> None:
    em = H5Emitter(model_name="t", snapshot_id="")
    em.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="3 components"):
        em.add_oriented_elements(
            type_token="forceBeamColumn",
            vecxz=(0.0, 0.0),  # type: ignore[arg-type]
            elements=[(1, (1, 2))],
            ndm=3,
        )


@pytest.mark.parametrize("bad_eid", [0, -1, -42])
def test_raises_on_non_positive_fem_eid(bad_eid: int) -> None:
    em = H5Emitter(model_name="t", snapshot_id="")
    em.model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="fem_eid must be > 0"):
        em.add_oriented_elements(
            type_token="forceBeamColumn",
            vecxz=(0.0, 0.0, 1.0),
            elements=[(bad_eid, (1, 2))],
            ndm=3,
        )
