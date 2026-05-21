"""Phase-3 unit tests for :class:`apeGmsh.cuts.FemToOpsTagMap`.

Tests build minimal ``model.h5`` files inline with h5py — just enough
for :func:`apeGmsh.opensees.emitter.h5_reader.open` to accept them and
for the tag map to walk ``/opensees/element_meta/{type_token}/``. We
deliberately don't drive the full ``H5Emitter`` here; the writer's
contract is covered by the existing tests under
``tests/opensees/h5/``. These tests own the *reader-side* mapping
logic only.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.cuts import FemToOpsTagMap


# --------------------------------------------------------------------- #
# Inline fixture builder — minimal model.h5 with element_meta + fem_eids
# --------------------------------------------------------------------- #
def _write_minimal_h5(
    path: Path,
    *,
    groups: dict[str, dict[str, np.ndarray]],
    schema_version: str = "2.6.0",
) -> None:
    """Write the smallest h5 that the reference reader accepts.

    ``groups`` is a mapping ``type_token -> {"ids": ..., "fem_eids": ...}``.
    Both arrays are written as int64. Only ``/meta/schema_version`` is
    required by the reader; we include the other ``/meta`` attrs the
    ``validate()`` path expects so tests can validate too if desired.
    """
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = schema_version
        meta.attrs["apeGmsh_version"] = "0.0.0-test"
        meta.attrs["created_iso"] = "2026-01-01T00:00:00Z"
        meta.attrs["ndm"] = 3
        meta.attrs["ndf"] = 6
        meta.attrs["snapshot_id"] = "test"
        meta.attrs["model_name"] = "test"

        em = f.create_group("opensees/element_meta")
        for type_token, arrays in groups.items():
            g = em.create_group(type_token)
            g.attrs["type"] = type_token
            g.create_dataset("ids", data=arrays["ids"].astype(np.int64))
            g.create_dataset("fem_eids", data=arrays["fem_eids"].astype(np.int64))


# --------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------- #
def test_round_trip_single_type(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11, 12]),
            "fem_eids": np.array([101, 102, 103]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)

    assert len(m) == 3
    assert m.ops_tag(101) == 10
    assert m.ops_tag(102) == 11
    assert m.ops_tag(103) == 12
    assert m.type_token_for(101) == "forceBeamColumn"
    assert m.type_tokens == ("forceBeamColumn",)
    assert m.n_sentinel == 0


def test_round_trip_multi_type(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11]),
            "fem_eids": np.array([101, 102]),
        },
        "FourNodeTetrahedron": {
            "ids": np.array([20, 21, 22]),
            "fem_eids": np.array([201, 202, 203]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)

    assert len(m) == 5
    assert m.ops_tag(101) == 10
    assert m.ops_tag(202) == 21
    assert m.type_token_for(102) == "forceBeamColumn"
    assert m.type_token_for(203) == "FourNodeTetrahedron"
    assert set(m.type_tokens) == {"forceBeamColumn", "FourNodeTetrahedron"}


# --------------------------------------------------------------------- #
# Sentinel handling
# --------------------------------------------------------------------- #
def test_sentinel_rows_dropped(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11, 12]),
            "fem_eids": np.array([101, -1, 103]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)

    assert len(m) == 2
    assert m.n_sentinel == 1
    assert m.ops_tag(101) == 10
    assert m.ops_tag(103) == 12
    assert 11 not in m       # ops_tag 11 was the sentinel-paired one
    assert -1 not in m


# --------------------------------------------------------------------- #
# Batched lookup + errors
# --------------------------------------------------------------------- #
def test_ops_tags_for_fem_eids_returns_tuple_in_order(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11, 12, 13]),
            "fem_eids": np.array([101, 102, 103, 104]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)
    tags = m.ops_tags_for_fem_eids([103, 101, 104])
    assert tags == (12, 10, 13)
    assert isinstance(tags, tuple)


def test_ops_tags_for_fem_eids_accepts_ndarray(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11]),
            "fem_eids": np.array([101, 102]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)
    tags = m.ops_tags_for_fem_eids(np.array([101, 102]))
    assert tags == (10, 11)


def test_missing_fem_eid_raises_keyerror(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10]),
            "fem_eids": np.array([101]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)
    with pytest.raises(KeyError, match="not in tag map"):
        m.ops_tags_for_fem_eids([101, 999])


def test_missing_fem_eid_single_lookup_raises_keyerror(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10]),
            "fem_eids": np.array([101]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)
    with pytest.raises(KeyError, match="not found in tag map"):
        m.ops_tag(999)


# --------------------------------------------------------------------- #
# Collision detection
# --------------------------------------------------------------------- #
def test_collision_across_types_raises(tmp_path: Path) -> None:
    """Same FEM eid under two type tokens — should never happen, but if
    it does, we catch it explicitly rather than silently last-wins."""
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10]),
            "fem_eids": np.array([101]),
        },
        "FourNodeTetrahedron": {
            "ids": np.array([20]),
            "fem_eids": np.array([101]),  # collision
        },
    })

    with pytest.raises(ValueError, match="bridge bug"):
        FemToOpsTagMap.from_h5(h5)


# --------------------------------------------------------------------- #
# Pre-Phase-8.6 file
# --------------------------------------------------------------------- #
def test_missing_fem_eids_dataset_raises_helpful(tmp_path: Path) -> None:
    """Files lacking the fem_eids dataset surface a clear
    "re-emit with newer apeGmsh" message rather than KeyError on an
    h5 dataset.  Originally this exercised the pre-Phase-8.6
    (schema 2.1.x) shape; per ADR 0023 the fixture must be inside
    the two-version reader window, so we use 2.6.0 with the dataset
    deliberately omitted."""
    h5 = tmp_path / "model.h5"
    with h5py.File(h5, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "2.6.0"
        em = f.create_group("opensees/element_meta")
        g = em.create_group("forceBeamColumn")
        g.attrs["type"] = "forceBeamColumn"
        g.create_dataset("ids", data=np.array([10], dtype=np.int64))
        # NO fem_eids dataset.

    with pytest.raises(ValueError, match="Phase 8.6"):
        FemToOpsTagMap.from_h5(h5)


# --------------------------------------------------------------------- #
# Reader-error propagation
# --------------------------------------------------------------------- #
def test_wrong_schema_major_propagates(tmp_path: Path) -> None:
    """Reader's SchemaVersionError should propagate cleanly."""
    from apeGmsh.opensees.emitter.h5_reader import SchemaVersionError

    h5 = tmp_path / "model.h5"
    _write_minimal_h5(
        h5,
        groups={
            "forceBeamColumn": {
                "ids": np.array([10]),
                "fem_eids": np.array([101]),
            },
        },
        schema_version="3.0.0",     # wrong major
    )

    with pytest.raises(SchemaVersionError):
        FemToOpsTagMap.from_h5(h5)


# --------------------------------------------------------------------- #
# Inverse lookup (OpenSees tags → FEM eids) — viewer integration uses
# this to walk SectionCutDef.element_ids (OpenSees tags) back into FEM
# eids for filter-highlight / connectivity queries.
# --------------------------------------------------------------------- #
def test_fem_eids_for_ops_tags_returns_tuple_in_order(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11, 12, 13]),
            "fem_eids": np.array([101, 102, 103, 104]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)
    fem_eids = m.fem_eids_for_ops_tags([12, 10, 13])
    assert fem_eids == (103, 101, 104)
    assert isinstance(fem_eids, tuple)


def test_fem_eids_for_ops_tags_accepts_ndarray(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11]),
            "fem_eids": np.array([101, 102]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)
    fem_eids = m.fem_eids_for_ops_tags(np.array([10, 11]))
    assert fem_eids == (101, 102)


def test_fem_eids_for_ops_tags_missing_raises(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10]),
            "fem_eids": np.array([101]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)
    with pytest.raises(KeyError, match="not in tag map"):
        m.fem_eids_for_ops_tags([10, 999])


def test_fem_eids_for_ops_tags_multi_type(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11]),
            "fem_eids": np.array([101, 102]),
        },
        "FourNodeTetrahedron": {
            "ids": np.array([20]),
            "fem_eids": np.array([201]),
        },
    })

    m = FemToOpsTagMap.from_h5(h5)
    assert m.fem_eids_for_ops_tags([20, 11]) == (201, 102)


# --------------------------------------------------------------------- #
# Repr (smoke)
# --------------------------------------------------------------------- #
def test_repr_smoke(tmp_path: Path) -> None:
    h5 = tmp_path / "model.h5"
    _write_minimal_h5(h5, groups={
        "forceBeamColumn": {
            "ids": np.array([10, 11]),
            "fem_eids": np.array([101, -1]),
        },
    })
    m = FemToOpsTagMap.from_h5(h5)
    s = repr(m)
    assert "FemToOpsTagMap" in s
    assert "n=1" in s
    assert "forceBeamColumn" in s
    assert "sentinel_dropped=1" in s
