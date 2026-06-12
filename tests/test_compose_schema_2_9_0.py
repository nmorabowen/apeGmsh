"""H5 round-trip coverage for the 2.8.0 → 2.9.0 schema bump (Phase 3A.1).

Locks ADR 0038 §"Schema":

* ``neutral_schema_version`` stamp is ``2.9.0`` on fresh files.
* ``meta/@tag_span_max`` is present on save.
* ``module_label`` parallel datasets exist on ``/nodes/`` and each
  ``/elements/{type}/`` with empty strings for every row (Phase 3A.1
  doesn't populate them; Phase 3B's merge engine will).
* ``/composed_from/`` is OMITTED when ``fem.composed_from`` is empty.
* ``/composed_from/`` round-trips field-for-field when populated.
* 2.8.0 files load cleanly without warnings (the two-version window).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._femdata_h5_io import (
    COMPOSED_FROM_SCHEMA_VERSION,
    NEUTRAL_SCHEMA_VERSION,
    read_fem_h5,
)
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_simple_fem(
    composed_from: "ComposeSet | tuple[ComposeRecord, ...]" = (),
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
) -> FEMData:
    """Tiny FEMData fixture — one Line2 element, ``ndf`` undeclared."""
    if node_ids is None:
        node_ids = np.array([1, 2, 3], dtype=np.int64)
    if elem_ids is None:
        elem_ids = np.array([10, 20], dtype=np.int64)
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)[:node_ids.size]

    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    # Two trivial line segments stitching the nodes.
    conn = np.array(
        [[int(node_ids[i]), int(node_ids[min(i + 1, node_ids.size - 1)])]
         for i in range(elem_ids.size)],
        dtype=np.int64,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=elem_ids.size, bandwidth=1,
        types=[line_info],
    )
    return FEMData(
        nodes=nodes, elements=elements, info=info,
        composed_from=composed_from,
    )


# ---------------------------------------------------------------------------
# Schema-bump locks
# ---------------------------------------------------------------------------


def test_schema_version_is_2_12_0(tmp_path: Path) -> None:
    """Fresh save stamps the current neutral schema version (coupling-knob cpl_* bump 2.11 → 2.12)."""
    fem = _make_simple_fem()
    out = tmp_path / "model.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert NEUTRAL_SCHEMA_VERSION == "2.12.0"
        assert f["meta"].attrs["neutral_schema_version"] == "2.12.0"
        assert f["meta"].attrs["schema_version"] == "2.12.0"


def test_tag_span_max_written_on_save(tmp_path: Path) -> None:
    """``meta/@tag_span_max`` covers nodes + elements combined.

    Implementation note (deliberate): the writer computes
    ``max(max_node_tag, max_elem_tag) - min(min_node_tag, min_elem_tag) + 1``.
    Phase 3B's ``_compute_source_span`` reads this attr.
    """
    node_ids = np.array([10, 5010], dtype=np.int64)
    elem_ids = np.array([1000, 6000], dtype=np.int64)
    fem = _make_simple_fem(node_ids=node_ids, elem_ids=elem_ids)
    out = tmp_path / "span.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "tag_span_max" in f["meta"].attrs
        # max(5010, 6000) − min(10, 1000) + 1 = 6000 − 10 + 1 = 5991
        assert int(f["meta"].attrs["tag_span_max"]) == 5991


def test_module_label_dataset_written_with_empty_strings(tmp_path: Path) -> None:
    """``module_label`` parallel datasets exist on nodes + each elem type."""
    fem = _make_simple_fem()
    out = tmp_path / "ml.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        nodes_ds = f["nodes/module_label"]
        assert nodes_ds.shape == (3,)
        values = [
            v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            for v in nodes_ds[...]
        ]
        assert values == ["", "", ""]

        # One element type alias is "line2" per the GMSH name table.
        assert "elements/line2/module_label" in f
        elem_ds = f["elements/line2/module_label"]
        assert elem_ds.shape == (2,)
        elem_vals = [
            v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            for v in elem_ds[...]
        ]
        assert elem_vals == ["", ""]


# ---------------------------------------------------------------------------
# composed_from group
# ---------------------------------------------------------------------------


def test_composed_from_group_absent_when_empty(tmp_path: Path) -> None:
    """Uncomposed FEMData omits ``/composed_from/`` entirely."""
    fem = _make_simple_fem()
    out = tmp_path / "uncomposed.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert "composed_from" not in f


def test_composed_from_group_round_trip_when_non_empty(tmp_path: Path) -> None:
    """``/composed_from/`` round-trips field-for-field, including
    properties + the ``composed_from_schema_version`` attr."""
    rec_full = ComposeRecord(
        label="full",
        source_path="src_full.h5",
        source_fem_hash="abc123",
        source_neutral_schema_version="2.9.0",
        translate=(1.0, 2.0, 3.0),
        rotate=(0.1, 0.2, 0.3, 0.9),
        partition_rank=4,
        composed_at="2026-05-26T12:34:56Z",
        properties={"author": "alice", "build_id": 7},
    )
    rec_min = ComposeRecord(
        label="minimal",
        source_path="src_min.h5",
        source_fem_hash="def456",
        source_neutral_schema_version="2.8.0",
        translate=(0.0, 0.0, 0.0),
    )
    original = ComposeSet((rec_full, rec_min))

    fem = _make_simple_fem(composed_from=original)
    out = tmp_path / "composed.h5"
    fem.to_h5(str(out))

    with h5py.File(out, "r") as f:
        assert "composed_from" in f
        attrs = f["composed_from"].attrs
        assert attrs["composed_from_schema_version"] == COMPOSED_FROM_SCHEMA_VERSION

    rebuilt = read_fem_h5(str(out))
    reloaded = rebuilt.composed_from
    assert isinstance(reloaded, ComposeSet)
    assert reloaded == original
    # Field-by-field spot checks on the round-tripped records.
    full = reloaded["full"]
    assert full.translate == (1.0, 2.0, 3.0)
    assert full.rotate == (0.1, 0.2, 0.3, 0.9)
    assert full.partition_rank == 4
    assert full.composed_at == "2026-05-26T12:34:56Z"
    assert full.properties == {"author": "alice", "build_id": 7}
    minimal = reloaded["minimal"]
    assert minimal.rotate is None
    assert minimal.partition_rank is None


# ---------------------------------------------------------------------------
# h5py optional-child .get() hazard
# ---------------------------------------------------------------------------


def test_h5py_optional_child_uses_lexists_not_get() -> None:
    """The loader probes optional children with ``in``, not ``.get``.

    Grep-style guard: the previous PR #261 bug used ``Group.get()``
    which returns phantom truthy proxies on some optional-child
    layouts.  The compose-aware reader must use ``"composed_from" in
    parent`` and the per-feature parallel datasets must use
    ``"module_label" in sub``.
    """
    from importlib import resources
    import re

    text = (
        Path(__file__).resolve().parents[1]
        / "src" / "apeGmsh" / "mesh" / "_femdata_h5_io.py"
    ).read_text(encoding="utf-8")
    # Forbid both literal forms of the hazard for compose-aware probes.
    forbidden = [
        r'\.get\(\s*[\'"]composed_from[\'"]',
        r'\.get\(\s*[\'"]module_label[\'"]',
    ]
    for pattern in forbidden:
        assert re.search(pattern, text) is None, (
            f"_femdata_h5_io.py contains forbidden Group.get() probe "
            f"matching {pattern!r}; use ``name in group`` instead per "
            f"project_h5py_optional_child_get_hazard."
        )
