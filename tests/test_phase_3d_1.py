"""Phase 3D.1 — H5ModelReader Protocol widening for compose-awareness.

Locks the three new methods on
:class:`apeGmsh.opensees.emitter.h5_reader.H5Model` so the viewer,
``cuts``, and future foreign-format adapters can query the composition
graph of a loaded model without coupling to the writer-side
:class:`apeGmsh.mesh.FEMData.FEMData` API:

* ``iter_composed_from()`` — yields each :class:`ComposeRecord` in
  module-label storage order.
* ``composed_for_node(node_id)`` — module label that owns the node, or
  ``None`` for host-owned / not-present.
* ``composed_for_element(element_id)`` — same shape for elements.

Backward-compat: schema 2.8.x files (no ``/composed_from/`` group, no
``module_label`` parallel datasets) decode as "uncomposed" — iter
yields nothing and ``composed_for_*`` return ``None`` for every id.

These tests do not exercise OpenSeesMP (pure H5 + neutral-zone work).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees.emitter import h5_reader


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_module_fem(
    *,
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
) -> FEMData:
    """Tiny single-line-element FEMData (no compose state)."""
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
    return FEMData(nodes=nodes, elements=elements, info=info)


@pytest.fixture
def composed_h5(tmp_path: Path) -> Path:
    """Build host + 2 composed modules → save as composed model.h5."""
    host = _make_module_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    )
    host_path = tmp_path / "host.h5"
    host.to_h5(str(host_path))

    module_a = _make_module_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    )
    module_a_path = tmp_path / "module_a.h5"
    module_a.to_h5(str(module_a_path))

    module_b = _make_module_fem(
        node_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        elem_ids=np.array([20, 21, 22], dtype=np.int64),
    )
    module_b_path = tmp_path / "module_b.h5"
    module_b.to_h5(str(module_b_path))

    g = apeGmsh.from_h5(host_path)
    g.compose(module_a_path, label="A", translate=(10.0, 0.0, 0.0))
    g.compose(module_b_path, label="B", translate=(100.0, 0.0, 0.0))

    out = tmp_path / "composed.h5"
    g.save(out)
    return out


@pytest.fixture
def uncomposed_h5(tmp_path: Path) -> Path:
    """Bare FEMData round-trip — no compose state."""
    fem = _make_module_fem()
    out = tmp_path / "uncomposed.h5"
    fem.to_h5(str(out))
    return out


@pytest.fixture
def legacy_2_8_h5(tmp_path: Path) -> Path:
    """Synthesise a 2.8.x-shaped file: same neutral zone, but
    ``module_label`` datasets stripped and ``/composed_from/`` absent.

    A genuine 2.8.0 writer does not exist on the current branch, but
    the 2.9.0 → 2.8.x backward-compat rule is "absence of the new
    fields decodes as uncomposed" — which is the file shape this
    fixture produces.  The schema-version stamps stay at 2.9.0 because
    :func:`h5_reader.open` validates against the *opensees* zone, and
    the carve-out at ``h5_reader.py:200-204`` skips that check on
    neutral-only files entirely.
    """
    fem = _make_module_fem()
    out = tmp_path / "legacy_2_8.h5"
    fem.to_h5(str(out))

    # Strip the schema-2.9.0-only additions from the saved file.
    with h5py.File(str(out), "a") as f:
        if "composed_from" in f:
            del f["composed_from"]
        # /nodes/module_label
        if "nodes" in f and "module_label" in f["nodes"]:
            del f["nodes"]["module_label"]
        # /elements/{type}/module_label
        if "elements" in f:
            for type_name in f["elements"]:
                sub = f["elements"][type_name]
                if "module_label" in sub:
                    del sub["module_label"]
    return out


# ---------------------------------------------------------------------------
# iter_composed_from
# ---------------------------------------------------------------------------


def test_iter_composed_from_yields_records_in_label_order(
    composed_h5: Path,
) -> None:
    """Two compose modules → two records in storage (label) order."""
    with h5_reader.open(str(composed_h5)) as model:
        records = list(model.iter_composed_from())

    assert len(records) == 2
    assert all(isinstance(r, ComposeRecord) for r in records)
    labels = [r.label for r in records]
    assert labels == ["A", "B"]


def test_iter_composed_from_round_trips_provenance(
    composed_h5: Path,
) -> None:
    """Every ComposeRecord field round-trips through the H5 reader."""
    with h5_reader.open(str(composed_h5)) as model:
        records = list(model.iter_composed_from())

    a = next(r for r in records if r.label == "A")
    b = next(r for r in records if r.label == "B")

    # translate carried through from the compose() calls in the fixture.
    assert a.translate == (10.0, 0.0, 0.0)
    assert b.translate == (100.0, 0.0, 0.0)
    # No rotate / partition_rank supplied on this fixture.
    assert a.rotate is None
    assert b.rotate is None
    assert a.partition_rank is None
    assert b.partition_rank is None
    # source_path / source_neutral_schema_version are populated by
    # the writer; we don't pin exact values (the writer's choice may
    # drift) but they should be non-empty strings for compose entries
    # built from real H5 sources.
    assert a.source_path
    assert a.source_neutral_schema_version
    assert b.source_path
    assert b.source_neutral_schema_version


def test_iter_composed_from_yields_nothing_for_uncomposed(
    uncomposed_h5: Path,
) -> None:
    """A FEMData with empty ``composed_from`` writes no
    ``/composed_from/`` group; iter yields nothing."""
    with h5_reader.open(str(uncomposed_h5)) as model:
        records = list(model.iter_composed_from())
    assert records == []


def test_iter_composed_from_yields_nothing_for_legacy_2_8(
    legacy_2_8_h5: Path,
) -> None:
    """Schema 2.8.x files (no ``/composed_from/`` group) decode as
    uncomposed without raising."""
    with h5_reader.open(str(legacy_2_8_h5)) as model:
        records = list(model.iter_composed_from())
    assert records == []


# ---------------------------------------------------------------------------
# composed_for_node
# ---------------------------------------------------------------------------


def test_composed_for_node_returns_label_for_module_node(
    composed_h5: Path,
) -> None:
    """Find a node that came from module A; check its label.

    The merge engine assigns module rows new ids in the reserved tag
    span — we don't know exact ids ahead of time, so we walk the
    ``/nodes/module_label`` array to pick the first non-empty row,
    confirm its label matches one of the composed modules, and then
    re-query through the public API.
    """
    with h5_reader.open(str(composed_h5)) as model:
        nodes_data = model.nodes()
        ids = nodes_data["ids"]
        with h5py.File(str(composed_h5), "r") as f:
            labels = f["nodes"]["module_label"][:]

        # Find any module-owned node (non-empty label).
        for row, raw_label in enumerate(labels):
            if isinstance(raw_label, bytes):
                label_str = raw_label.decode("utf-8")
            else:
                label_str = str(raw_label)
            if label_str:
                node_id = int(ids[row])
                expected = label_str
                assert model.composed_for_node(node_id) == expected
                return

        pytest.fail("no module-owned node found in composed fixture")


def test_composed_for_node_returns_none_for_host_node(
    composed_h5: Path,
) -> None:
    """A node with empty-string ``module_label`` is host-owned → None."""
    with h5_reader.open(str(composed_h5)) as model:
        nodes_data = model.nodes()
        ids = nodes_data["ids"]
        with h5py.File(str(composed_h5), "r") as f:
            labels = f["nodes"]["module_label"][:]

        for row, raw_label in enumerate(labels):
            if isinstance(raw_label, bytes):
                label_str = raw_label.decode("utf-8")
            else:
                label_str = str(raw_label)
            if not label_str:  # host row
                node_id = int(ids[row])
                assert model.composed_for_node(node_id) is None
                return

        pytest.fail("no host node found in composed fixture")


def test_composed_for_node_returns_none_for_unknown_id(
    composed_h5: Path,
) -> None:
    """A node id that doesn't appear in ``/nodes/ids`` → None."""
    with h5_reader.open(str(composed_h5)) as model:
        assert model.composed_for_node(999_999_999) is None


def test_composed_for_node_returns_none_for_uncomposed(
    uncomposed_h5: Path,
) -> None:
    """Uncomposed files write an empty-string ``module_label`` for every
    row; every node maps to None."""
    with h5_reader.open(str(uncomposed_h5)) as model:
        for nid in (1, 2, 3):
            assert model.composed_for_node(nid) is None


def test_composed_for_node_returns_none_for_legacy_2_8(
    legacy_2_8_h5: Path,
) -> None:
    """Schema 2.8.x file (no ``module_label`` dataset on /nodes) → None
    for every id without raising."""
    with h5_reader.open(str(legacy_2_8_h5)) as model:
        for nid in (1, 2, 3):
            assert model.composed_for_node(nid) is None


# ---------------------------------------------------------------------------
# composed_for_element
# ---------------------------------------------------------------------------


def test_composed_for_element_returns_label_for_module_element(
    composed_h5: Path,
) -> None:
    """Mirror of the node test for elements — walk per-type
    ``module_label`` arrays."""
    with h5_reader.open(str(composed_h5)) as model:
        with h5py.File(str(composed_h5), "r") as f:
            for type_name in f["elements"]:
                sub = f["elements"][type_name]
                ids = sub["ids"][:]
                labels = sub["module_label"][:]
                for row, raw_label in enumerate(labels):
                    if isinstance(raw_label, bytes):
                        label_str = raw_label.decode("utf-8")
                    else:
                        label_str = str(raw_label)
                    if label_str:
                        eid = int(ids[row])
                        assert model.composed_for_element(eid) == label_str
                        return

        pytest.fail("no module-owned element found in composed fixture")


def test_composed_for_element_returns_none_for_host_element(
    composed_h5: Path,
) -> None:
    """Host-owned elements (empty-string ``module_label``) → None."""
    with h5_reader.open(str(composed_h5)) as model:
        with h5py.File(str(composed_h5), "r") as f:
            for type_name in f["elements"]:
                sub = f["elements"][type_name]
                ids = sub["ids"][:]
                labels = sub["module_label"][:]
                for row, raw_label in enumerate(labels):
                    if isinstance(raw_label, bytes):
                        label_str = raw_label.decode("utf-8")
                    else:
                        label_str = str(raw_label)
                    if not label_str:
                        eid = int(ids[row])
                        assert model.composed_for_element(eid) is None
                        return

        pytest.fail("no host element found in composed fixture")


def test_composed_for_element_returns_none_for_unknown_id(
    composed_h5: Path,
) -> None:
    """Unknown element id → None (no scan of any type matches)."""
    with h5_reader.open(str(composed_h5)) as model:
        assert model.composed_for_element(999_999_999) is None


def test_composed_for_element_returns_none_for_uncomposed(
    uncomposed_h5: Path,
) -> None:
    """Empty-string labels for every element row → None for every id."""
    with h5_reader.open(str(uncomposed_h5)) as model:
        for eid in (10, 11):
            assert model.composed_for_element(eid) is None


def test_composed_for_element_returns_none_for_legacy_2_8(
    legacy_2_8_h5: Path,
) -> None:
    """Schema 2.8.x file (no per-type ``module_label`` dataset) → None
    for every id without raising."""
    with h5_reader.open(str(legacy_2_8_h5)) as model:
        for eid in (10, 11):
            assert model.composed_for_element(eid) is None


# ---------------------------------------------------------------------------
# Protocol surface — methods exist and are callable
# ---------------------------------------------------------------------------


def test_h5_model_protocol_surface_callable(uncomposed_h5: Path) -> None:
    """The three new methods are present on :class:`H5Model` and
    callable on an open reader — the duck-typed "implements the
    Protocol" check.

    No :class:`typing.Protocol` class lands in source code as of
    Phase 3D.1 (ADR 0026 PR7-c reserved the Protocol module but
    never implemented it — see ``project_compose_v1_phase3_kickoff``).
    Foreign-format adapters conform structurally; this test pins the
    methods exist on the reference implementer.
    """
    with h5_reader.open(str(uncomposed_h5)) as model:
        # Existence.
        assert callable(model.iter_composed_from)
        assert callable(model.composed_for_node)
        assert callable(model.composed_for_element)
        # Invocability with valid args.
        assert list(model.iter_composed_from()) == []
        assert model.composed_for_node(1) is None
        assert model.composed_for_element(10) is None


def test_iter_composed_from_is_iterator(uncomposed_h5: Path) -> None:
    """``iter_composed_from()`` returns an iterator (not a list) — the
    Protocol contract.  Consumers can ``for r in iter_composed_from()``
    or ``list(...)`` without copying through a tuple."""
    import collections.abc

    with h5_reader.open(str(uncomposed_h5)) as model:
        it = model.iter_composed_from()
        assert isinstance(it, collections.abc.Iterator)
