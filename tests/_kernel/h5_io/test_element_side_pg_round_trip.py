"""B2 — element-side-only PG / label round-trip without phantom node-side.

Neutral schema 2.10.0 splits ``/physical_groups`` and ``/labels`` into
``node_side/`` + ``element_side/`` sub-trees so the reader no longer
has to guess which side an entry originated on.  The pre-2.10 read
heuristic ("if ``element_ids`` is present the entry is element-side")
silently promoted element-side-only PGs into a phantom node-side
entry, which then folded into the snapshot id and caused the round-
trip ``MalformedH5Error`` documented in the B2 design notes.

This test pins the fixed behaviour: a hand-built FEMData with one
element-side PG and one element-side label round-trips through H5
with their element_ids + node_ids preserved AND no phantom node-side
entry on the rebuilt composite.

No gmsh, no openseespy — pure broker construction (same pattern as
``tests/_kernel/resolvers/test_source_host_subelements.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------


def _make_element_side_only_fem(
    *,
    pg_name: str = "soil_block",
    pg_key: tuple[int, int] = (3, 99),
    label_name: str = "embed_host",
    label_key: tuple[int, int] = (3, 77),
) -> FEMData:
    """Build a 1-hex FEMData with an element-side PG + label and NO
    matching node-side entries.

    The element-side ``PhysicalGroupSet`` carries ``element_ids``,
    ``node_ids``, ``node_coords`` on ``pg_key``; the node-side
    ``PhysicalGroupSet`` is empty.  Same shape for the label set on
    ``label_key``.  This is the exact configuration the pre-2.10
    reader heuristic mishandled.
    """
    node_ids = np.arange(1, 9, dtype=np.int64)
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    hex_info = make_type_info(
        code=5, gmsh_name="Hexahedron 8", dim=3, order=1, npe=8, count=1,
    )
    hex_conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    hex_ids = np.array([100], dtype=np.int64)
    hex_group = ElementGroup(
        element_type=hex_info,
        ids=hex_ids,
        connectivity=hex_conn,
    )

    elem_pgs = {
        pg_key: {
            "name": pg_name,
            "element_ids": hex_ids.copy(),
            "node_ids": node_ids.copy(),
            "node_coords": coords.copy(),
        },
    }
    elem_labels = {
        label_key: {
            "name": label_name,
            "element_ids": hex_ids.copy(),
            "node_ids": node_ids.copy(),
            "node_coords": coords.copy(),
        },
    }
    elements = ElementComposite(
        groups={5: hex_group},
        physical=PhysicalGroupSet(elem_pgs),
        labels=LabelSet(elem_labels),
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=coords,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size,
        n_elems=hex_ids.size,
        bandwidth=1,
        types=[hex_info],
    )
    return FEMData(
        nodes=nodes,
        elements=elements,
        info=info,
        composed_from=ComposeSet(()),
    )


# ---------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------


class TestElementSideOnlyRoundTrip:
    def test_round_trip_preserves_element_side_pg_and_no_phantom(
        self, tmp_path: Path,
    ) -> None:
        fem = _make_element_side_only_fem()
        out = tmp_path / "elem_side_only.h5"

        # Round-trip — the snapshot_id verifier in from_h5 raises
        # ``MalformedH5Error`` if the rebuilt FEM hashes differently
        # than the stored id, which is the original drift symptom.
        fem.to_h5(str(out))
        rebuilt = FEMData.from_h5(str(out))

        # Element-side PG intact.
        assert (3, 99) in rebuilt.elements.physical.get_all()
        eids = rebuilt.elements.physical.element_ids((3, 99))
        assert eids.size == 1
        assert int(eids[0]) == 100
        nids = rebuilt.elements.physical.node_ids((3, 99))
        assert nids.size == 8
        np.testing.assert_array_equal(np.sort(nids), np.arange(1, 9))
        ncoords = rebuilt.elements.physical.node_coords((3, 99))
        assert ncoords.shape == (8, 3)

        # No phantom node-side PG on the same key.
        assert (3, 99) not in rebuilt.nodes.physical.get_all()

        # Element-side label intact, no phantom node-side label.
        assert (3, 77) in rebuilt.elements.labels.get_all()
        np.testing.assert_array_equal(
            np.sort(rebuilt.elements.labels.element_ids((3, 77))),
            np.array([100], dtype=np.int64),
        )
        assert (3, 77) not in rebuilt.nodes.labels.get_all()

    def test_snapshot_id_round_trips(self, tmp_path: Path) -> None:
        fem = _make_element_side_only_fem()
        out = tmp_path / "snap.h5"

        sid_before = fem.snapshot_id
        fem.to_h5(str(out))
        rebuilt = FEMData.from_h5(str(out))
        assert rebuilt.snapshot_id == sid_before

    def test_snapshot_id_includes_element_side_pg(
        self, tmp_path: Path,
    ) -> None:
        """B2 hash widening: adding an element-side PG MUST change
        the snapshot_id (closes the pre-2.10 drift hole)."""
        fem_a = _make_element_side_only_fem()
        fem_b = _make_element_side_only_fem(pg_name="different_label")
        assert fem_a.snapshot_id != fem_b.snapshot_id

    def test_snapshot_id_includes_element_side_label(
        self, tmp_path: Path,
    ) -> None:
        """B2 hash widening: adding an element-side label MUST change
        the snapshot_id."""
        fem_a = _make_element_side_only_fem(label_name="host_a")
        fem_b = _make_element_side_only_fem(label_name="host_b")
        assert fem_a.snapshot_id != fem_b.snapshot_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
