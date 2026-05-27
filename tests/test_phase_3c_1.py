"""Phase 3C.1 — ``compose_hash()`` lineage wrapper tests.

Locks the compose-aware ``fem_hash`` wrapper introduced per
ADR 0038 §"Lineage chain extension" / design-point D4:

* uncomposed FEMData hashes byte-identically to today's
  :attr:`FEMData.snapshot_id` (backward compat — pre-2.9.0
  pin tests / bind-contract round-trips stay green);
* composed FEMData hashes are independent of the
  ``compose(A) → compose(B)`` vs ``compose(B) → compose(A)``
  call order (the wrapper folds in sorted-by-``module_label``
  contributions);
* adding a module, changing a module's label, or removing a
  module all perturb the hash deterministically;
* the wrapper round-trips through H5 — a saved-and-reloaded
  composed FEMData has the same ``compose_hash`` as the
  in-memory original.

These tests run entirely against the FEMData broker; no live
Gmsh session and no openseespy import is required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._femdata_hash import compute_snapshot_id
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees._internal.lineage import compose_hash


# ---------------------------------------------------------------------------
# Fixture builders — minimal FEMData with one Line2 element type
# ---------------------------------------------------------------------------


def _make_module_fem(
    *,
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
    composed_from=None,
) -> FEMData:
    """Small FEMData broker — same shape as test_compose_end_to_end."""
    if node_ids is None:
        node_ids = np.array([1, 2, 3], dtype=np.int64)
    if elem_ids is None:
        elem_ids = np.array([10, 11], dtype=np.int64)

    n = node_ids.size
    node_coords = np.array(
        [[float(i), 0.0, 0.0] for i in range(n)],
        dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    conn_rows = []
    for i in range(elem_ids.size):
        a = int(node_ids[i % n])
        b = int(node_ids[(i + 1) % n])
        conn_rows.append([a, b])
    conn = np.array(conn_rows, dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )

    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=n, n_elems=elem_ids.size, bandwidth=1,
        types=[line_info],
    )
    return FEMData(
        nodes=nodes, elements=elements, info=info,
        composed_from=composed_from,
    )


def _save(fem: FEMData, path: Path) -> Path:
    fem.to_h5(str(path))
    return path


@pytest.fixture
def host_h5(tmp_path: Path) -> Path:
    return _save(_make_module_fem(), tmp_path / "host.h5")


@pytest.fixture
def module_a_h5(tmp_path: Path) -> Path:
    return _save(
        _make_module_fem(
            node_ids=np.array([1, 2, 3], dtype=np.int64),
            elem_ids=np.array([10, 11], dtype=np.int64),
        ),
        tmp_path / "module_a.h5",
    )


@pytest.fixture
def module_b_h5(tmp_path: Path) -> Path:
    return _save(
        _make_module_fem(
            node_ids=np.array([1, 2, 3, 4], dtype=np.int64),
            elem_ids=np.array([20, 21, 22], dtype=np.int64),
        ),
        tmp_path / "module_b.h5",
    )


# ---------------------------------------------------------------------------
# Backward compatibility — uncomposed FEMData
# ---------------------------------------------------------------------------


def test_uncomposed_byte_identical_to_snapshot_id() -> None:
    """``compose_hash(fem) == fem.snapshot_id`` on uncomposed FEMData.

    The load-bearing backward-compat invariant per ADR 0038
    §"Lineage chain extension":
    "compose_hash() is byte-equivalent to today's fem_hash on
    uncomposed (fem.composed_from == ()) input."

    Pre-2.9.0 pin tests + bind-contract round-trips depend on this
    equality.
    """
    fem = _make_module_fem()
    assert not fem.composed_from  # invariant — uncomposed by construction
    assert compose_hash(fem) == fem.snapshot_id


def test_uncomposed_byte_identical_to_compute_snapshot_id() -> None:
    """``compose_hash(fem)`` agrees with the freshly-computed snapshot.

    Belt-and-braces on the previous test — bypasses the
    ``_snapshot_id_cache`` so we exercise the underlying
    :func:`compute_snapshot_id` path directly.
    """
    fem = _make_module_fem()
    assert compose_hash(fem) == compute_snapshot_id(fem)


# ---------------------------------------------------------------------------
# Stability / idempotence
# ---------------------------------------------------------------------------


def test_compose_hash_is_idempotent() -> None:
    """``compose_hash(fem) == compose_hash(fem)`` — no state leakage."""
    fem = _make_module_fem()
    assert compose_hash(fem) == compose_hash(fem)


def test_compose_hash_stable_across_distinct_instances() -> None:
    """Two identically-built FEMData instances hash to the same value."""
    a = _make_module_fem()
    b = _make_module_fem()
    # Distinct objects with the same canonical content.
    assert a is not b
    assert compose_hash(a) == compose_hash(b)


# ---------------------------------------------------------------------------
# Compose changes the hash (single-module sensitivity)
# ---------------------------------------------------------------------------


def test_adding_a_module_changes_the_hash(
    host_h5: Path, module_a_h5: Path,
) -> None:
    """``compose_hash`` distinguishes uncomposed vs one-module composed."""
    uncomposed = apeGmsh.from_h5(host_h5)
    h_before = compose_hash(uncomposed._fem)

    composed = apeGmsh.from_h5(host_h5)
    composed.compose(module_a_h5, label="A")
    h_after = compose_hash(composed._fem)

    assert h_before != h_after


def test_module_label_affects_the_hash(
    host_h5: Path, module_a_h5: Path,
) -> None:
    """The same source composed under two different labels hashes apart.

    ``compose_hash`` folds the label into the digest (via
    ``_hash_composed_from``'s label-update step) so the namespace
    identity is part of the fem-hash.
    """
    g1 = apeGmsh.from_h5(host_h5)
    g1.compose(module_a_h5, label="A")
    g2 = apeGmsh.from_h5(host_h5)
    g2.compose(module_a_h5, label="B")

    assert compose_hash(g1._fem) != compose_hash(g2._fem)


# ---------------------------------------------------------------------------
# composed_from iteration order independence
# ---------------------------------------------------------------------------


def test_compose_hash_sorts_composed_from_by_label(
    host_h5: Path, module_a_h5: Path, module_b_h5: Path,
) -> None:
    """``compose_hash`` folds ``composed_from`` in sorted-by-label order.

    Two FEMData instances with identical broker content but
    ``composed_from`` records appended in different orders must hash
    the same.  This is the structural compose-order-independence
    contract that ADR 0038 §"Lineage chain extension" assigns to the
    wrapper — the sort lives in
    :func:`apeGmsh.mesh._femdata_hash._hash_composed_from` today.

    Note: this test does NOT exercise full ``g.compose(A) then
    g.compose(B)`` vs ``g.compose(B) then g.compose(A)`` equivalence,
    because the tag-offset reservation scheme assigns windows in
    insertion order (so A-first vs B-first lands A's nodes at
    different absolute tags).  Compose-order-stable reservation is
    out of scope for 3C.1; the wrapper's invariance covers only the
    provenance-iteration axis it owns.
    """
    # Build two FEMData instances with identical node/element content
    # but ``composed_from`` records appended in different orders.  The
    # broker payload is byte-identical; only the iteration order over
    # the provenance list differs.
    from apeGmsh._kernel.records._compose import ComposeRecord
    from apeGmsh._kernel.record_sets import ComposeSet

    rec_a = ComposeRecord(
        label="A", source_path="a.h5", source_fem_hash="h_a",
        source_neutral_schema_version="2.9.0",
        translate=(0.0, 0.0, 0.0),
        composed_at="2026-05-26T00:00:00",
    )
    rec_b = ComposeRecord(
        label="B", source_path="b.h5", source_fem_hash="h_b",
        source_neutral_schema_version="2.9.0",
        translate=(0.0, 0.0, 0.0),
        composed_at="2026-05-26T00:00:00",
    )

    fem_ab = _make_module_fem(composed_from=(rec_a, rec_b))
    fem_ba = _make_module_fem(composed_from=(rec_b, rec_a))

    assert compose_hash(fem_ab) == compose_hash(fem_ba)


# ---------------------------------------------------------------------------
# Cross-session round-trip via from_h5
# ---------------------------------------------------------------------------


def test_compose_hash_round_trips_via_from_h5(
    host_h5: Path, module_a_h5: Path, tmp_path: Path,
) -> None:
    """Save a composed FEMData; reload via from_h5; ``compose_hash`` matches.

    Closes the cross-session continuity loop:
    ``compose_hash(in_memory) == compose_hash(saved_then_reloaded)``.
    """
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A")
    in_memory = g._fem
    expected = compose_hash(in_memory)

    out = tmp_path / "composed.h5"
    g.save(out)

    reloaded = FEMData.from_h5(str(out))
    assert compose_hash(reloaded) == expected


def test_compose_hash_round_trips_multi_module(
    host_h5: Path, module_a_h5: Path, module_b_h5: Path, tmp_path: Path,
) -> None:
    """Multi-module round-trip — both records contribute deterministically."""
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A")
    g.compose(module_b_h5, label="B")
    expected = compose_hash(g._fem)

    out = tmp_path / "multi.h5"
    g.save(out)

    reloaded = FEMData.from_h5(str(out))
    assert compose_hash(reloaded) == expected
