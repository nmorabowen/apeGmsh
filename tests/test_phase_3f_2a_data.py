"""Phase 3F.2a — viewer data layer compose-module mappings.

Locks the per-entity compose-module-label mappings the viewer's
:class:`ViewerData` exposes alongside the existing partition mapping
(schema 2.9.0 / ADR 0038):

* :attr:`ViewerElements._module_by_eid` + :meth:`module_for` +
  :attr:`has_modules`
* :attr:`ViewerNodes._module_by_nid` + :meth:`module_for` +
  :attr:`has_modules`
* The two H5Model bulk accessors
  :meth:`H5Model.bulk_module_labels_for_nodes` /
  ``bulk_module_labels_for_elements`` (Phase 3F.2a addition)
* End-to-end ``ViewerData.from_fem`` / ``ViewerData.from_reader``
  populates the mappings from the broker / H5 sources.

DATA LAYER ONLY — no ColorMode enum, no controller / UI / palette
work.  Three slices follow this one (3F.2b enum+callback, 3F.2c UI,
3F.2d tests).

These tests do not exercise OpenSeesMP (pure H5 + neutral-zone work).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.viewers.data import ViewerData
from apeGmsh.viewers.data._viewer_data import (
    _decode_module_by_eid,
    _decode_module_by_nid,
)


# ---------------------------------------------------------------------------
# Fixture builders — mirror tests/test_phase_3d_1.py for consistency
# ---------------------------------------------------------------------------


def _make_module_fem(
    *,
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
) -> FEMData:
    """Tiny single-line-element FEMData with no compose state."""
    if node_ids is None:
        node_ids = np.array([1, 2, 3], dtype=np.int64)
    if elem_ids is None:
        elem_ids = np.array([10, 11], dtype=np.int64)

    n = node_ids.size
    if n > 0:
        node_coords = np.array(
            [[float(i), 0.0, 0.0] for i in range(n)],
            dtype=np.float64,
        )
    else:
        node_coords = np.zeros((0, 3), dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    if elem_ids.size > 0:
        conn_rows = []
        for i in range(elem_ids.size):
            a = int(node_ids[i % n])
            b = int(node_ids[(i + 1) % n])
            conn_rows.append([a, b])
        conn = np.array(conn_rows, dtype=np.int64)
    else:
        conn = np.zeros((0, 2), dtype=np.int64)
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
def uncomposed_fem() -> FEMData:
    """Bare FEMData — no compose state at all."""
    return _make_module_fem()


@pytest.fixture
def uncomposed_h5(tmp_path: Path) -> Path:
    """Bare FEMData round-trip — no compose state."""
    fem = _make_module_fem()
    out = tmp_path / "uncomposed.h5"
    fem.to_h5(str(out))
    return out


@pytest.fixture
def composed_h5(tmp_path: Path) -> Path:
    """Host + 2 composed modules → saved composed model.h5."""
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
def nested_composed_h5(tmp_path: Path) -> Path:
    """Depth-2 nested compose: outer composes a depth-1 file under
    label ``bayP``.

    The inner file already carries a depth-1 compose (label ``frameA``)
    so the result's rows owned by the inner module carry the joined
    label ``"bayP/frameA"`` per :func:`_join_module_label`.
    """
    empty = _make_module_fem(
        node_ids=np.array([], dtype=np.int64),
        elem_ids=np.array([], dtype=np.int64),
    )
    empty_path = tmp_path / "empty.h5"
    empty.to_h5(str(empty_path))

    leaf = _make_module_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    )
    leaf_path = tmp_path / "leaf.h5"
    leaf.to_h5(str(leaf_path))

    # depth-1: empty host composes leaf under label "frameA".
    g_inner = apeGmsh.from_h5(empty_path)
    g_inner.compose(leaf_path, label="frameA")
    depth_1 = tmp_path / "depth_1.h5"
    g_inner.save(depth_1)

    # depth-2: empty host composes depth-1 under label "bayP".
    g_outer = apeGmsh.from_h5(empty_path)
    g_outer.compose(depth_1, label="bayP")
    out = tmp_path / "depth_2.h5"
    g_outer.save(out)
    return out


@pytest.fixture
def legacy_2_8_h5(tmp_path: Path) -> Path:
    """Synthesise a 2.8.x-shaped file — strip the schema 2.9.0
    additions from a freshly-written uncomposed file.

    Reuses the same construction tests/test_phase_3d_1.py uses; the
    schema-version stamp stays at 2.9.0 because the carve-out at
    ``h5_reader.open`` skips the opensees-zone check for neutral-only
    files.
    """
    fem = _make_module_fem()
    out = tmp_path / "legacy_2_8.h5"
    fem.to_h5(str(out))

    with h5py.File(str(out), "a") as f:
        if "composed_from" in f:
            del f["composed_from"]
        if "nodes" in f and "module_label" in f["nodes"]:
            del f["nodes"]["module_label"]
        if "elements" in f:
            for type_name in f["elements"]:
                sub = f["elements"][type_name]
                if "module_label" in sub:
                    del sub["module_label"]
    return out


# =====================================================================
# ViewerData.from_fem — uncomposed broker
# =====================================================================


class TestViewerFromFEMUncomposed:
    """Uncomposed FEMData → empty mappings, ``has_modules == False``."""

    def test_elements_has_no_modules(
        self, uncomposed_fem: FEMData,
    ) -> None:
        vd = ViewerData.from_fem(uncomposed_fem)
        assert vd.elements.has_modules is False
        # Any id query returns None — no module-label metadata at all.
        assert vd.elements.module_for(10) is None
        assert vd.elements.module_for(11) is None
        assert vd.elements.module_for(999_999) is None

    def test_nodes_has_no_modules(
        self, uncomposed_fem: FEMData,
    ) -> None:
        vd = ViewerData.from_fem(uncomposed_fem)
        assert vd.nodes.has_modules is False
        assert vd.nodes.module_for(1) is None
        assert vd.nodes.module_for(2) is None
        assert vd.nodes.module_for(999_999) is None


# =====================================================================
# ViewerData.from_h5 — uncomposed file (schema 2.9.0, all-empty labels)
# =====================================================================


class TestViewerFromH5Uncomposed:
    """Uncomposed model.h5 → empty mappings; broker's H5 reader strips
    the all-empty ``module_label`` dataset on read so ``has_modules ==
    False`` on the round-trip too."""

    def test_elements_has_no_modules(self, uncomposed_h5: Path) -> None:
        vd = ViewerData.from_h5(str(uncomposed_h5))
        assert vd.elements.has_modules is False
        assert vd.elements.module_for(10) is None

    def test_nodes_has_no_modules(self, uncomposed_h5: Path) -> None:
        vd = ViewerData.from_h5(str(uncomposed_h5))
        assert vd.nodes.has_modules is False
        assert vd.nodes.module_for(1) is None


# =====================================================================
# ViewerData.from_h5 — composed file
# =====================================================================


class TestViewerFromH5Composed:
    """Composed model.h5 → at least one module-owned row, host rows
    map to None."""

    def test_elements_has_modules_and_labels_match_h5(
        self, composed_h5: Path,
    ) -> None:
        vd = ViewerData.from_h5(str(composed_h5))
        assert vd.elements.has_modules is True

        # Round-trip every per-type row in the file against
        # ``module_for`` — host rows (empty-string label) should map
        # to None; module rows should map to the file's stored label.
        with h5py.File(str(composed_h5), "r") as f:
            for type_name in f["elements"]:
                sub = f["elements"][type_name]
                ids = sub["ids"][:]
                labels = sub["module_label"][:]
                for row in range(min(len(ids), len(labels))):
                    eid = int(ids[row])
                    raw = labels[row]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    label = str(raw)
                    if label == "":
                        assert vd.elements.module_for(eid) is None
                    else:
                        assert vd.elements.module_for(eid) == label

    def test_nodes_has_modules_and_labels_match_h5(
        self, composed_h5: Path,
    ) -> None:
        vd = ViewerData.from_h5(str(composed_h5))
        assert vd.nodes.has_modules is True

        with h5py.File(str(composed_h5), "r") as f:
            ids = f["nodes"]["ids"][:]
            labels = f["nodes"]["module_label"][:]
            for row in range(min(len(ids), len(labels))):
                nid = int(ids[row])
                raw = labels[row]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                label = str(raw)
                if label == "":
                    assert vd.nodes.module_for(nid) is None
                else:
                    assert vd.nodes.module_for(nid) == label

    def test_unknown_id_returns_none(self, composed_h5: Path) -> None:
        """A query for an id not in the file → None (dict miss)."""
        vd = ViewerData.from_h5(str(composed_h5))
        assert vd.elements.module_for(999_999_999) is None
        assert vd.nodes.module_for(999_999_999) is None


# =====================================================================
# ViewerData — nested (depth-2) compose carries joined labels
# =====================================================================


class TestNestedComposeJoinedLabels:
    """Phase 3E.1 — nested compose results stamp joined labels on rows.

    The viewer's mapping must return the FULL joined label
    (e.g. ``"bayP/frameA"``), not the inner-only fragment.
    """

    def test_nodes_carry_joined_label(
        self, nested_composed_h5: Path,
    ) -> None:
        vd = ViewerData.from_h5(str(nested_composed_h5))
        assert vd.nodes.has_modules is True
        # At least one node has a depth-2 joined label.
        joined = {
            v for v in vd.nodes._module_by_nid.values()
            if "/" in v or "." in v
        }
        assert joined, (
            "expected at least one joined label like 'bayP/frameA' "
            "in nested compose result; got "
            f"{set(vd.nodes._module_by_nid.values())}"
        )

    def test_elements_carry_joined_label(
        self, nested_composed_h5: Path,
    ) -> None:
        vd = ViewerData.from_h5(str(nested_composed_h5))
        assert vd.elements.has_modules is True
        joined = {
            v for v in vd.elements._module_by_eid.values()
            if "/" in v or "." in v
        }
        assert joined, (
            "expected at least one joined label in nested compose "
            f"result; got {set(vd.elements._module_by_eid.values())}"
        )


# =====================================================================
# H5Model bulk accessors
# =====================================================================


class TestH5ModelBulkAccessors:
    """Round-trip the bulk accessors against the per-id
    :meth:`composed_for_node` / ``composed_for_element`` from Phase
    3D.1.  Both surfaces must agree on every id."""

    def test_bulk_nodes_matches_per_id(
        self, composed_h5: Path,
    ) -> None:
        with h5_reader.open(str(composed_h5)) as model:
            bulk = model.bulk_module_labels_for_nodes()
            ids = model.nodes()["ids"]
            for nid in ids:
                nid_int = int(nid)
                per_id = model.composed_for_node(nid_int)
                if per_id is None:
                    # Host or missing — must not appear in bulk.
                    assert nid_int not in bulk
                else:
                    assert bulk[nid_int] == per_id

    def test_bulk_elements_matches_per_id(
        self, composed_h5: Path,
    ) -> None:
        with h5_reader.open(str(composed_h5)) as model:
            bulk = model.bulk_module_labels_for_elements()
            # Collect every element id across every type group.
            all_eids: list[int] = []
            for alias in model.elements():
                arrays = model.element_arrays(alias)
                all_eids.extend(int(i) for i in arrays["ids"])
            for eid in all_eids:
                per_id = model.composed_for_element(eid)
                if per_id is None:
                    assert eid not in bulk
                else:
                    assert bulk[eid] == per_id

    def test_bulk_nodes_empty_on_uncomposed(
        self, uncomposed_h5: Path,
    ) -> None:
        """Uncomposed file: every ``module_label`` row is empty
        string; the bulk accessor returns ``{}``.  The broker's H5
        reader actually strips the dataset entirely on round-trip in
        this case (writer convention), but the bulk accessor handles
        both shapes via its ``in`` probe."""
        with h5_reader.open(str(uncomposed_h5)) as model:
            assert model.bulk_module_labels_for_nodes() == {}

    def test_bulk_elements_empty_on_uncomposed(
        self, uncomposed_h5: Path,
    ) -> None:
        with h5_reader.open(str(uncomposed_h5)) as model:
            assert model.bulk_module_labels_for_elements() == {}

    def test_bulk_nodes_empty_on_legacy_2_8(
        self, legacy_2_8_h5: Path,
    ) -> None:
        """Pre-2.9.0 file (no ``module_label`` dataset at all): bulk
        accessor returns ``{}`` without raising."""
        with h5_reader.open(str(legacy_2_8_h5)) as model:
            assert model.bulk_module_labels_for_nodes() == {}

    def test_bulk_elements_empty_on_legacy_2_8(
        self, legacy_2_8_h5: Path,
    ) -> None:
        with h5_reader.open(str(legacy_2_8_h5)) as model:
            assert model.bulk_module_labels_for_elements() == {}


# =====================================================================
# ViewerData.from_reader — degrades cleanly for foreign-format adapters
# =====================================================================


class TestForeignAdapterDegradesEmpty:
    """Foreign-format adapters (LS-DYNA d3plot, xDMF) that don't
    implement the bulk accessor degrade to an empty mapping via
    ``getattr`` — same convention as ``_decode_boundary_node_ids``.
    """

    def test_decode_module_by_nid_no_method_yields_empty(self) -> None:
        class _NoBulkReader:
            pass

        assert _decode_module_by_nid(_NoBulkReader()) == {}

    def test_decode_module_by_eid_no_method_yields_empty(self) -> None:
        class _NoBulkReader:
            pass

        assert _decode_module_by_eid(_NoBulkReader()) == {}


# =====================================================================
# Direct ctor wiring — mirror tests/viewers/data/test_viewer_data_partitions.py
# =====================================================================


class TestViewerEntitySlotWiring:
    """``ViewerElements`` / ``ViewerNodes`` accept the new
    ``module_by_*`` kwargs and surface them through ``module_for`` /
    ``has_modules`` — the slot-wiring counterpart of the existing
    ``partition_by_eid`` / ``boundary_node_ids`` tests at
    tests/viewers/data/test_viewer_data_partitions.py."""

    def test_viewer_elements_module_slot_defaults_empty(self) -> None:
        from apeGmsh.viewers.data._elements import (
            ElementLoadView,
            SurfaceConstraintView,
            ViewerElements,
        )
        from apeGmsh.viewers.data._nodes import _NamedNodeSelection

        empty_sel = _NamedNodeSelection(
            {}, raise_on_missing=True, label="x",
        )
        ve = ViewerElements(
            groups=[],
            physical=empty_sel, labels=empty_sel, selection=empty_sel,
            loads=ElementLoadView([]),
            constraints=SurfaceConstraintView([]),
        )
        assert ve.has_modules is False
        assert ve.module_for(42) is None

    def test_viewer_elements_module_slot_populated(self) -> None:
        from apeGmsh.viewers.data._elements import (
            ElementLoadView,
            SurfaceConstraintView,
            ViewerElements,
        )
        from apeGmsh.viewers.data._nodes import _NamedNodeSelection

        empty_sel = _NamedNodeSelection(
            {}, raise_on_missing=True, label="x",
        )
        ve = ViewerElements(
            groups=[],
            physical=empty_sel, labels=empty_sel, selection=empty_sel,
            loads=ElementLoadView([]),
            constraints=SurfaceConstraintView([]),
            module_by_eid={10: "A", 20: "B", 30: ""},
        )
        assert ve.has_modules is True
        assert ve.module_for(10) == "A"
        assert ve.module_for(20) == "B"
        # Empty-string label is filtered at ctor time → host-owned.
        assert ve.module_for(30) is None
        assert ve.module_for(99) is None

    def test_viewer_nodes_module_slot_defaults_empty(self) -> None:
        from apeGmsh.viewers.data._nodes import (
            MassView,
            NodalLoadView,
            NodeConstraintView,
            SPView,
            ViewerNodes,
            _NamedNodeSelection,
        )

        empty_sel = _NamedNodeSelection(
            {}, raise_on_missing=True, label="x",
        )
        vn = ViewerNodes(
            ids=np.array([], dtype=np.int64),
            coords=np.zeros((0, 3), dtype=np.float64),
            physical=empty_sel, labels=empty_sel, selection=empty_sel,
            loads=NodalLoadView([]), sp=SPView([]),
            masses=MassView([]), constraints=NodeConstraintView([]),
        )
        assert vn.has_modules is False
        assert vn.module_for(42) is None

    def test_viewer_nodes_module_slot_populated(self) -> None:
        from apeGmsh.viewers.data._nodes import (
            MassView,
            NodalLoadView,
            NodeConstraintView,
            SPView,
            ViewerNodes,
            _NamedNodeSelection,
        )

        empty_sel = _NamedNodeSelection(
            {}, raise_on_missing=True, label="x",
        )
        vn = ViewerNodes(
            ids=np.array([], dtype=np.int64),
            coords=np.zeros((0, 3), dtype=np.float64),
            physical=empty_sel, labels=empty_sel, selection=empty_sel,
            loads=NodalLoadView([]), sp=SPView([]),
            masses=MassView([]), constraints=NodeConstraintView([]),
            module_by_nid={1: "A", 2: "bayP/frameA", 3: ""},
        )
        assert vn.has_modules is True
        assert vn.module_for(1) == "A"
        # Joined labels are stored verbatim.
        assert vn.module_for(2) == "bayP/frameA"
        # Empty-string label is filtered → host-owned.
        assert vn.module_for(3) is None
        assert vn.module_for(99) is None
