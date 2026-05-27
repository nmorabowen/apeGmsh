"""Phase 3B.2a — compose rewrite engine tests (ADR 0038).

Covers the rewrite-only slice of the compose merge pipeline:

* ``_compute_source_span`` — reads ``/meta/@tag_span_max`` with 2.8.x
  dataset-scan fallback (one-shot UserWarning).
* ``_compute_reservation`` — auto-sizing per-module windows with
  granularity-aligned bases.
* ``_apply_geometric_transform`` — rotate-then-translate on Nx3 coords.
* ``_rewrite_source_for_compose`` — end-to-end rewrite producing a
  :class:`_RewrittenBundle` ready for Phase 3B.2b's host merge.
* ``tag_rewrite_spec`` coverage — every IMPORT-verdict record kind
  declares a spec; DISCARD/DEFER kinds opt out with the ``None``
  sentinel.

3B.2a deliberately ships ONLY the producer side.  ``g.compose()``
itself still raises ``NotImplementedError``; FILTER warnings are
silent at this layer (3B.2b owns ``ComposeFilterWarning`` emission).
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh._kernel.records._loads import (
    ElementLoadRecord,
    NodalLoadRecord,
    SPRecord,
)
from apeGmsh._kernel.records._masses import MassRecord
from apeGmsh._kernel.records._partitions import PartitionRecord
from apeGmsh.mesh._compose import (
    Compose,
    _apply_geometric_transform,
    _compute_reservation,
    _compute_source_span,
    _RewrittenBundle,
    _rewrite_source_for_compose,
)
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
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


def _make_fem(
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
    constraints: "list | None" = None,
    elem_constraints: "list | None" = None,
    nodal_loads: "list | None" = None,
    masses: "list | None" = None,
    sp: "list | None" = None,
    node_pgs: "dict | None" = None,
    node_labels: "dict | None" = None,
    composed_from=(),
) -> FEMData:
    """Build a tiny FEMData with explicit tag ranges + optional records."""
    if node_ids is None:
        node_ids = np.array([1, 2, 3], dtype=np.int64)
    if elem_ids is None:
        elem_ids = np.array([10, 20], dtype=np.int64)

    n = node_ids.size
    node_coords = np.array(
        [[float(i), 0.0, 0.0] for i in range(n)],
        dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    # Build a connectivity that references the actual node tags.
    conn_rows = []
    for i in range(elem_ids.size):
        a = int(node_ids[i % n])
        b = int(node_ids[(i + 1) % n])
        conn_rows.append([a, b])
    conn = np.array(conn_rows, dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )

    pgs_dict = node_pgs or {}
    labels_dict = node_labels or {}

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pgs_dict),
        labels=LabelSet(labels_dict),
        constraints=constraints,
        loads=nodal_loads,
        sp=sp,
        masses=masses,
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
        constraints=elem_constraints,
    )
    info = MeshInfo(
        n_nodes=n, n_elems=elem_ids.size, bandwidth=1,
        types=[line_info],
    )
    return FEMData(
        nodes=nodes, elements=elements, info=info,
        composed_from=composed_from,
    )


def _write_fem_h5(fem: FEMData, path: Path) -> Path:
    """Save a FEMData to an H5 file and return the path."""
    fem.to_h5(str(path))
    return path


def _downgrade_to_2_8(path: Path) -> None:
    """Mutate an in-place H5 to look like a 2.8.0 source.

    Drops ``meta/@tag_span_max`` and ``module_label`` parallel datasets
    so the reader exercises the fallback path.  ``snapshot_id`` is
    recomputed against the new shape — leave it alone; the reader's
    B4 check tolerates the modified file because the rebuild stays
    compose-empty.
    """
    with h5py.File(path, "a") as f:
        f["meta"].attrs["neutral_schema_version"] = "2.8.0"
        f["meta"].attrs["schema_version"] = "2.8.0"
        if "tag_span_max" in f["meta"].attrs:
            del f["meta"].attrs["tag_span_max"]
        if "module_label" in f["nodes"]:
            del f["nodes/module_label"]
        if "elements" in f:
            for tname in list(f["elements"].keys()):
                sub = f["elements"][tname]
                if "module_label" in sub:
                    del sub["module_label"]


# ---------------------------------------------------------------------------
# _compute_source_span
# ---------------------------------------------------------------------------


def test_compute_source_span_2_9_0(tmp_path: Path) -> None:
    """2.9.0 source: ``span`` reads from ``/meta/@tag_span_max``."""
    fem = _make_fem(
        node_ids=np.array([5, 50, 500], dtype=np.int64),
        elem_ids=np.array([1000, 2000], dtype=np.int64),
    )
    src = _write_fem_h5(fem, tmp_path / "src_29.h5")
    span, min_tag, max_tag = _compute_source_span(src)
    # max(500, 2000) − min(5, 1000) + 1 = 2000 − 5 + 1 = 1996.
    assert span == 1996
    assert min_tag == 5
    assert max_tag == 2000


def test_compute_source_span_2_8_0_fallback(tmp_path: Path) -> None:
    """2.8.0 source: span computed by dataset scan + one UserWarning."""
    fem = _make_fem(
        node_ids=np.array([7, 70, 700], dtype=np.int64),
        elem_ids=np.array([3000, 4000], dtype=np.int64),
    )
    src = _write_fem_h5(fem, tmp_path / "src_28.h5")
    _downgrade_to_2_8(src)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        span, min_tag, max_tag = _compute_source_span(src)

    relevant = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "pre-2.9.0" in str(w.message)
    ]
    assert len(relevant) == 1, (
        f"expected one pre-2.9.0 UserWarning; got "
        f"{[(w.category.__name__, str(w.message)) for w in caught]}"
    )
    # max(700, 4000) − min(7, 3000) + 1 = 3994.
    assert span == 3994
    assert min_tag == 7
    assert max_tag == 4000


# ---------------------------------------------------------------------------
# _compute_reservation
# ---------------------------------------------------------------------------


def test_compute_reservation_first_compose() -> None:
    """First compose: base rounds ``host_max_tag`` up; size rounds span up."""
    base, size = _compute_reservation(
        source_span=8_000,
        host_max_tag=12_345,
        previous_reservations=(),
    )
    # ceil(12345 / 1M) → next granularity boundary above is 1_000_000.
    assert base == 1_000_000
    # ceil(8000 / 1M) * 1M = 1M (the round-up unit dominates a small span).
    assert size == 1_000_000


def test_compute_reservation_subsequent() -> None:
    """Subsequent compose: base = previous_base + previous_size."""
    base, size = _compute_reservation(
        source_span=8_000,
        host_max_tag=12_345,
        previous_reservations=((1_000_000, 1_000_000),),
    )
    assert base == 2_000_000
    assert size == 1_000_000


def test_compute_reservation_compose_size_override() -> None:
    """``compose_size_per_module`` overrides the auto-computed size."""
    base, size = _compute_reservation(
        source_span=8_000,
        host_max_tag=12_345,
        compose_size_per_module=50_000_000,
    )
    assert size == 50_000_000


# ---------------------------------------------------------------------------
# _apply_geometric_transform
# ---------------------------------------------------------------------------


def test_apply_geometric_transform_identity() -> None:
    """Identity translate + no rotate: input returned unchanged."""
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    out = _apply_geometric_transform(
        xyz, translate=(0.0, 0.0, 0.0), rotate=None,
    )
    # Fast path: identical object.
    assert out is xyz


def test_apply_geometric_transform_pure_translate() -> None:
    """Pure translate shifts every row by the translate vector."""
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    out = _apply_geometric_transform(
        xyz, translate=(10.0, 20.0, 30.0), rotate=None,
    )
    expected = xyz + np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(out, expected)


def test_apply_geometric_transform_rotate_about_z() -> None:
    """Axis-angle (0,0,1, π/2) rotates (1,0,0) → (0,1,0)."""
    xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    out = _apply_geometric_transform(
        xyz, translate=(0.0, 0.0, 0.0),
        rotate=(0.0, 0.0, 1.0, math.pi / 2.0),
    )
    np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0], atol=1e-12)


def test_apply_geometric_transform_rotate_then_translate() -> None:
    """Rotation runs first, then translation.

    A point at (1,0,0) rotated π/2 about z gives (0,1,0); translate by
    (10,0,0) gives (10,1,0).  A point at origin rotated by anything
    still gives origin, then translates straight to (10,0,0).
    """
    xyz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    out = _apply_geometric_transform(
        xyz,
        translate=(10.0, 0.0, 0.0),
        rotate=(0.0, 0.0, 1.0, math.pi / 2.0),
    )
    np.testing.assert_allclose(out[0], [10.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[1], [10.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# _rewrite_source_for_compose — offsets + namespacing
# ---------------------------------------------------------------------------


def _rewrite(src_path, label="conn_a", base=1_000_000, **kw):
    """Convenience: call _rewrite_source_for_compose with reasonable defaults."""
    span, min_tag, _ = _compute_source_span(src_path)
    return _rewrite_source_for_compose(
        source_path=src_path,
        label=label,
        translate=kw.pop("translate", (0.0, 0.0, 0.0)),
        rotate=kw.pop("rotate", None),
        partition_rank=kw.pop("partition_rank", None),
        properties=kw.pop("properties", {}),
        base=base,
        size=kw.pop("size", 1_000_000),
        source_span=span,
        source_min_tag=min_tag,
    )


def test_rewrite_offsets_node_tags(tmp_path: Path) -> None:
    """Node ids get offset = base - source_min_tag."""
    fem = _make_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    )
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    bundle = _rewrite(src, base=1_000_000)
    # source_min_tag = 1 → offset = 999_999.
    expected = np.array([1_000_000, 1_000_001, 1_000_002], dtype=np.int64)
    np.testing.assert_array_equal(
        np.asarray(bundle.node_ids, dtype=np.int64), expected,
    )


def test_rewrite_offsets_element_tags_and_connectivity(tmp_path: Path) -> None:
    """Element tags AND their connectivity entries are offset."""
    fem = _make_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    )
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    bundle = _rewrite(src, base=1_000_000)
    # offset = 999_999.
    group = bundle.element_groups[1]
    np.testing.assert_array_equal(
        np.asarray(group.ids, dtype=np.int64),
        np.array([1_000_009, 1_000_010], dtype=np.int64),
    )
    # Connectivity (originally [[1,2],[2,3]]) also shifts.
    np.testing.assert_array_equal(
        np.asarray(group.connectivity, dtype=np.int64),
        np.array([
            [1_000_000, 1_000_001],
            [1_000_001, 1_000_002],
        ], dtype=np.int64),
    )


def test_rewrite_offsets_pg_node_ids_and_element_ids(tmp_path: Path) -> None:
    """PG node_ids / element_ids arrays get offset."""
    # Construct a PG keyed on (dim, tag) with node + element members.
    node_pgs = {
        (0, 1): {
            "name": "top",
            "node_ids": np.array([1, 2], dtype=np.int64),
            "node_coords": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64,
            ),
        },
    }
    fem = _make_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
        node_pgs=node_pgs,
    )
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    bundle = _rewrite(src, base=1_000_000, label="conn_a")
    # node-side PGs should carry offset arrays.
    pg_entries = list(bundle.node_physical.values())
    assert pg_entries, "expected at least one node PG in bundle"
    pg = pg_entries[0]
    np.testing.assert_array_equal(
        np.asarray(pg["node_ids"], dtype=np.int64),
        np.array([1_000_000, 1_000_001], dtype=np.int64),
    )


def test_rewrite_namespaces_pg_names(tmp_path: Path) -> None:
    """PG names are prefixed with ``{label}.``."""
    node_pgs = {
        (0, 1): {
            "name": "top_flange",
            "node_ids": np.array([1], dtype=np.int64),
            "node_coords": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        },
    }
    fem = _make_fem(node_pgs=node_pgs)
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    bundle = _rewrite(src, label="conn_a")
    pg = next(iter(bundle.node_physical.values()))
    assert pg["name"] == "conn_a.top_flange"


def test_rewrite_namespaces_label_names(tmp_path: Path) -> None:
    """Label names are prefixed with ``{label}.``."""
    node_labels = {
        (0, 1): {
            "name": "weld_zone",
            "node_ids": np.array([1], dtype=np.int64),
            "node_coords": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        },
    }
    fem = _make_fem(node_labels=node_labels)
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    bundle = _rewrite(src, label="conn_a")
    label_rec = next(iter(bundle.node_labels.values()))
    assert label_rec["name"] == "conn_a.weld_zone"


def test_rewrite_offsets_constraint_records(tmp_path: Path) -> None:
    """NodePairRecord master_node / slave_node fields get offset."""
    constraints = [
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=10,
            slave_node=11,
            dofs=[1, 2, 3],
        ),
    ]
    # The constraint's master/slave nodes must exist in the FEM for the
    # H5 round-trip; bump node_ids range to include them.
    fem = _make_fem(
        node_ids=np.array([10, 11, 12], dtype=np.int64),
        elem_ids=np.array([100, 101], dtype=np.int64),
        constraints=constraints,
    )
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    bundle = _rewrite(src, base=1_000_000)
    # source_min_tag = 10 → offset = 999_990.
    # master 10 → 1_000_000; slave 11 → 1_000_001.
    rewrites = [
        r for r in bundle.node_constraints
        if isinstance(r, NodePairRecord)
    ]
    assert rewrites, "expected at least one NodePairRecord"
    rec = rewrites[0]
    assert rec.master_node == 1_000_000
    assert rec.slave_node == 1_000_001


def test_rewrite_drops_discard_kinds_silently(tmp_path: Path) -> None:
    """PartitionRecord (DISCARD verdict) is dropped without a warning.

    Partitions live on ``fem.partitions`` (a separate composite); the
    rewriter doesn't carry them into the bundle and ``read_fem_h5``
    isn't re-derived for them either.  The bundle has no partition
    attribute at all — the DISCARD verdict is structural.
    """
    fem = _make_fem()
    src = _write_fem_h5(fem, tmp_path / "src.h5")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = _rewrite(src)

    # No UserWarnings about DISCARD partitions.
    relevant = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "partition" in str(w.message).lower()
    ]
    assert relevant == [], (
        f"DISCARD verdict must be silent; got: "
        f"{[(w.category.__name__, str(w.message)) for w in relevant]}"
    )
    # And the bundle exposes no ``partitions`` attribute.
    assert not hasattr(bundle, "partitions")


def test_rewrite_drops_filter_kinds_silently(tmp_path: Path) -> None:
    """FILTER verdicts (stages / recorders / etc.) drop silently in 3B.2a.

    3B.2a does not read any FILTER-verdict content (it lives in the
    OpenSees zone, which ``read_fem_h5`` skips by design).  We assert
    that ``_rewrite_source_for_compose`` emits no
    :class:`UserWarning` over a vanilla neutral-zone source — 3B.2b
    owns the :class:`ComposeFilterWarning` emission.
    """
    fem = _make_fem()
    src = _write_fem_h5(fem, tmp_path / "src.h5")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _rewrite(src)

    # No UserWarnings (the 2.9.0 fast path doesn't trigger the
    # pre-2.9.0 scan warning either).
    relevant = [
        w for w in caught if issubclass(w.category, UserWarning)
    ]
    assert relevant == [], (
        f"3B.2a must not emit warnings; 3B.2b owns ComposeFilterWarning. "
        f"Got: {[(w.category.__name__, str(w.message)) for w in relevant]}"
    )


def test_rewrite_preserves_record_count_invariant(tmp_path: Path) -> None:
    """Round-trip count: every IMPORT record in source surfaces in bundle.

    Sanity check that the rewriter doesn't accidentally drop IMPORT
    records.  Counts are taken over the broker shape; DISCARD/FILTER
    kinds are structurally absent from the neutral zone so they're not
    part of this count.
    """
    constraints = [
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=1, slave_node=2, dofs=[1, 2, 3],
        ),
    ]
    nodal_loads = [
        NodalLoadRecord(
            node_id=3,
            force_xyz=(1.0, 0.0, 0.0),
        ),
    ]
    masses = [MassRecord(node_id=2, mass=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))]
    fem = _make_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
        constraints=constraints,
        nodal_loads=nodal_loads,
        masses=masses,
    )
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    bundle = _rewrite(src)

    assert len(bundle.node_constraints) == 1
    assert len(bundle.nodal_loads) == 1
    assert len(bundle.mass_records) == 1
    assert bundle.node_ids.size == 3
    # Elements: one group with 2 records.
    assert bundle.element_groups[1].ids.size == 2


def test_rewrite_applies_geometric_transform_to_nodes(tmp_path: Path) -> None:
    """Translate shifts the node coords on the rewritten bundle."""
    fem = _make_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    )
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    bundle = _rewrite(src, translate=(10.0, 0.0, 0.0))
    # Original coords were (i, 0, 0) for i=0..2; translate shifts x by 10.
    expected = np.array(
        [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(bundle.node_coords, expected)


# ---------------------------------------------------------------------------
# tag_rewrite_spec coverage — cover-set drift sentinel
# ---------------------------------------------------------------------------


def test_tag_rewrite_spec_coverage() -> None:
    """Every neutral-zone record dataclass declares a ``tag_rewrite_spec``.

    IMPORT-verdict kinds carry a populated spec dict;
    DISCARD/DEFER-verdict kinds opt out via the ``None`` sentinel.  A
    future record dataclass added without a declaration would fail
    here.

    The neutral-zone record kinds enumerated below are the IMPORT
    cover-set referenced by ADR 0038 §"Tag-reference rewrite
    checklist" line 308-340.  Materials / sections / integration rules
    are NOT in this list — they live in the OpenSees zone (separate
    from the neutral zone) and will be added in a future phase per the
    spec interpretation in the 3B.2a report.
    """
    import_kinds = (
        NodePairRecord,
        NodeGroupRecord,
        InterpolationRecord,
        SurfaceCouplingRecord,
        NodeToSurfaceRecord,
        NodalLoadRecord,
        ElementLoadRecord,
        SPRecord,
        MassRecord,
    )
    for cls in import_kinds:
        spec = getattr(cls, "tag_rewrite_spec", "<unset>")
        assert spec is not None and spec != "<unset>", (
            f"{cls.__name__} missing tag_rewrite_spec — cover-set drift "
            f"per ADR 0038 §'Tag-reference rewrite checklist'"
        )
        assert isinstance(spec, dict), (
            f"{cls.__name__}.tag_rewrite_spec must be a dict; "
            f"got {type(spec).__name__}"
        )
        for key in ("tag_fields_scalar", "tag_fields_array", "name_fields"):
            assert key in spec, (
                f"{cls.__name__}.tag_rewrite_spec missing key {key!r}"
            )

    # DISCARD / DEFER kinds opt out with the ``None`` sentinel.
    for cls in (PartitionRecord, ComposeRecord):
        assert cls.tag_rewrite_spec is None, (
            f"{cls.__name__} should opt out of rewrite with "
            f"``tag_rewrite_spec = None`` (DISCARD / DEFER verdict)"
        )


# ---------------------------------------------------------------------------
# Schema-version gating
# ---------------------------------------------------------------------------


def test_pre_2_8_0_schema_rejected(tmp_path: Path) -> None:
    """Pre-2.8.0 source raises :class:`SchemaVersionError`.

    The reader's two-version window accepts 2.8.x and 2.9.x; the
    2.7.0 marker is outside the supported range.  Surfacing
    SchemaVersionError matches the contract from
    ``apeGmsh.opensees._internal.schema_version``.
    """
    from apeGmsh.opensees._internal.schema_version import SchemaVersionError

    fem = _make_fem()
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    # Force the schema marker to 2.7.0 — outside the reader window.
    with h5py.File(src, "a") as f:
        f["meta"].attrs["neutral_schema_version"] = "2.7.0"
        f["meta"].attrs["schema_version"] = "2.7.0"

    with pytest.raises(SchemaVersionError):
        _compute_source_span(src)


# ---------------------------------------------------------------------------
# Contract pin: g.compose() returns a ComposedModule handle in 3B.2c
# ---------------------------------------------------------------------------


def test_compose_compose_returns_handle(tmp_path: Path) -> None:
    """``g.compose()`` is wired end-to-end in Phase 3B.2c.

    Locks the contract that the public entry point is no longer a
    stub.  A chain-phase session (from_h5) keeps the test self-
    contained — no live gmsh state required.
    """
    fem = _make_fem()
    src = _write_fem_h5(fem, tmp_path / "src.h5")
    g = apeGmsh.from_h5(src)
    handle = g.compose(src, label="m")
    assert handle.label == "m"
