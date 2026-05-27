"""End-to-end compose tests — Phase 3B.2c / ADR 0038.

Locks the wired ``g.compose(...)`` API: cross-session composition via
``apeGmsh.from_h5``, anchor resolution, FILTER-warning emission, the
host pattern-field rewrite reversal, and ``get_fem_data()`` /
``compose_inspect`` / ``compose_list`` interactions.

These tests run entirely against the FEMData broker — no live Gmsh
session is required (every fixture uses :meth:`apeGmsh.from_h5` to
load a pre-saved H5 directly into chain phase).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh._kernel.records._loads import NodalLoadRecord
from apeGmsh._kernel.records._masses import MassRecord
from apeGmsh.mesh._compose import (
    ComposeAnchorError,
    ComposeFilterWarning,
    ComposedModule,
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
# Fixture builders
# ---------------------------------------------------------------------------


def _make_module_fem(
    *,
    node_ids: "np.ndarray | None" = None,
    elem_ids: "np.ndarray | None" = None,
    node_pgs: "dict | None" = None,
    nodal_loads: "list | None" = None,
    masses: "list | None" = None,
) -> FEMData:
    """Small FEMData with a single Line2 element type."""
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
        physical=PhysicalGroupSet(node_pgs or {}),
        labels=LabelSet({}),
        loads=nodal_loads,
        masses=masses,
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


def _save_module(fem: FEMData, path: Path) -> Path:
    fem.to_h5(str(path))
    return path


@pytest.fixture
def host_h5(tmp_path: Path) -> Path:
    """Saved host module with node ids 1..3, elem ids 10..11."""
    fem = _make_module_fem()
    return _save_module(fem, tmp_path / "host.h5")


@pytest.fixture
def module_a_h5(tmp_path: Path) -> Path:
    """Saved module A — same shape; will be tag-shifted on compose."""
    fem = _make_module_fem(
        node_ids=np.array([1, 2, 3], dtype=np.int64),
        elem_ids=np.array([10, 11], dtype=np.int64),
    )
    return _save_module(fem, tmp_path / "module_a.h5")


@pytest.fixture
def module_b_h5(tmp_path: Path) -> Path:
    """Saved module B with a slightly different tag range."""
    fem = _make_module_fem(
        node_ids=np.array([1, 2, 3, 4], dtype=np.int64),
        elem_ids=np.array([20, 21, 22], dtype=np.int64),
    )
    return _save_module(fem, tmp_path / "module_b.h5")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_module_compose_round_trip(
    host_h5: Path, module_a_h5: Path, tmp_path: Path,
) -> None:
    """Build host, compose module, save composed model, reload + check."""
    g = apeGmsh.from_h5(host_h5)
    handle = g.compose(module_a_h5, label="A")
    assert isinstance(handle, ComposedModule)
    assert handle.label == "A"

    out = tmp_path / "composed.h5"
    g.save(out)

    reloaded = FEMData.from_h5(str(out))
    # One compose record present.
    assert "A" in reloaded.composed_from
    assert len(reloaded.composed_from) == 1
    rec = reloaded.composed_from["A"]
    assert rec.label == "A"
    # Node count = 3 (host) + 3 (module) = 6.
    assert reloaded.info.n_nodes == 6
    # Element count = 2 + 2 = 4.
    assert reloaded.info.n_elems == 4
    # module_label populated for the bundle rows.
    ml = getattr(reloaded.nodes, "_module_label", None)
    assert ml is not None
    labels_in_order = list(ml)
    assert labels_in_order.count("A") == 3
    assert labels_in_order.count("") == 3


def test_multi_module_compose(
    host_h5: Path, module_a_h5: Path, module_b_h5: Path, tmp_path: Path,
) -> None:
    """Two modules with different translates → both records present, no
    tag overlap."""
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A", translate=(0.0, 0.0, 0.0))
    g.compose(module_b_h5, label="B", translate=(100.0, 0.0, 0.0))

    out = tmp_path / "multi.h5"
    g.save(out)

    reloaded = FEMData.from_h5(str(out))
    assert "A" in reloaded.composed_from
    assert "B" in reloaded.composed_from
    # 3 host + 3 A + 4 B = 10 nodes.
    assert reloaded.info.n_nodes == 10
    # 2 host + 2 A + 3 B = 7 elements.
    assert reloaded.info.n_elems == 7
    # Tag ranges disjoint — every node id appears exactly once.
    node_ids = list(reloaded.nodes.ids)
    assert len(set(int(x) for x in node_ids)) == len(node_ids)


def test_cross_session_compose_via_from_h5(
    host_h5: Path, module_a_h5: Path, tmp_path: Path,
) -> None:
    """Day 1: build + save.  Day 2: from_h5 + compose + save."""
    # "Day 2" — load a previously-saved host, compose a module, save.
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A")
    final = tmp_path / "final.h5"
    g.save(final)

    reloaded = FEMData.from_h5(str(final))
    assert "A" in reloaded.composed_from
    assert reloaded.info.n_nodes == 6
    assert reloaded.info.n_elems == 4


def test_compose_with_anchor_resolution(
    tmp_path: Path, module_a_h5: Path,
) -> None:
    """Anchor resolves to a PG centroid translate."""
    # Host with a PG named "anchor_pt" at the origin offset (5, 0, 0).
    node_ids = np.array([10, 11, 12], dtype=np.int64)
    node_coords = np.array(
        [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2,
    )
    conn = np.array([[10, 11], [11, 12]], dtype=np.int64)
    elem_ids = np.array([100, 101], dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )
    pgs = {
        (0, 1): {
            "name": "anchor_pt",
            "node_ids": np.array([10], dtype=np.int64),
            "node_coords": np.array(
                [[5.0, 0.0, 0.0]], dtype=np.float64,
            ),
        },
    }
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords,
        physical=PhysicalGroupSet(pgs),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=3, n_elems=2, bandwidth=1, types=[line_info])
    host_fem = FEMData(nodes=nodes, elements=elements, info=info)
    host_path = tmp_path / "host_with_pg.h5"
    host_fem.to_h5(str(host_path))

    g = apeGmsh.from_h5(host_path)
    g.compose(module_a_h5, label="A", anchor="anchor_pt")

    handle_rec = g._fem.composed_from["A"]
    # Anchor PG has one node at (5, 0, 0); centroid is (5, 0, 0).
    assert handle_rec.translate == (5.0, 0.0, 0.0)


def test_compose_with_anchor_conflict_raises(
    host_h5: Path, module_a_h5: Path,
) -> None:
    """``anchor=`` + non-zero ``translate=`` raises ComposeAnchorError."""
    g = apeGmsh.from_h5(host_h5)
    with pytest.raises(ComposeAnchorError):
        g.compose(
            module_a_h5,
            label="A",
            anchor="some_pg",
            translate=(1.0, 0.0, 0.0),
        )


def test_compose_with_unknown_anchor_raises(
    host_h5: Path, module_a_h5: Path,
) -> None:
    """Anchor PG that doesn't exist on the host raises
    :class:`ComposeAnchorError`."""
    g = apeGmsh.from_h5(host_h5)
    with pytest.raises(ComposeAnchorError):
        g.compose(module_a_h5, label="A", anchor="does_not_exist")


def test_compose_filter_warning_for_stages(
    host_h5: Path, tmp_path: Path,
) -> None:
    """Source H5 carrying ``/opensees/stages/...`` emits one
    :class:`ComposeFilterWarning` per kind (stages = 1).

    Recorders / analysis-settings stay silent.
    """
    # Build a module with stage content in /opensees/.
    src_fem = _make_module_fem()
    src_path = tmp_path / "module_with_stages.h5"
    src_fem.to_h5(str(src_path))
    # Hand-inject a /opensees/stages/ sub-group so the filter probe
    # picks it up.  Recorders also added → must stay silent.
    with h5py.File(str(src_path), "a") as f:
        ops = f.create_group("opensees")
        stages = ops.create_group("stages")
        stages.create_group("stage_0")
        recorders = ops.create_group("recorders")
        recorders.create_group("rec_0")

    g = apeGmsh.from_h5(host_h5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g.compose(src_path, label="A")

    relevant = [
        w for w in caught if issubclass(w.category, ComposeFilterWarning)
    ]
    # Exactly one stages warning, no recorder warning.
    assert len(relevant) == 1, [str(w.message) for w in relevant]
    assert "stages" in str(relevant[0].message).lower()


def test_compose_pattern_field_not_namespaced(
    host_h5: Path, tmp_path: Path,
) -> None:
    """Regression for 3B.2a's pattern namespacing — patterns are
    FILTER-verdict, the host owns the pattern name."""
    # Module with a nodal load on a known pattern name.
    fem = _make_module_fem(
        nodal_loads=[
            NodalLoadRecord(
                node_id=2,
                force_xyz=(1.0, 0.0, 0.0),
                pattern="dead",
                name=None,
            ),
        ],
    )
    src = _save_module(fem, tmp_path / "module_with_pattern.h5")

    g = apeGmsh.from_h5(host_h5)
    g.compose(src, label="A")

    loads = list(g._fem.nodes.loads)
    assert len(loads) == 1
    # Pattern field must remain "dead" — NOT "A.dead".
    assert loads[0].pattern == "dead"


def test_compose_get_fem_data_returns_composed(
    host_h5: Path, module_a_h5: Path,
) -> None:
    """After compose, ``get_fem_data()`` returns the merged FEMData."""
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A")
    fem = g.mesh.queries.get_fem_data()
    assert "A" in fem.composed_from
    assert fem.info.n_nodes == 6
    assert fem.info.n_elems == 4


def test_compose_inspect_after_compose(
    host_h5: Path, module_a_h5: Path,
) -> None:
    """``compose_inspect`` works post-compose (metadata-only read)."""
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A")
    info = g.compose_inspect(module_a_h5)
    assert "neutral_schema_version" in info
    # The source is uncomposed (it's the module file).
    assert info["composed_from"] == ()


def test_compose_list_returns_modules(
    host_h5: Path, module_a_h5: Path, module_b_h5: Path,
) -> None:
    """``g.compose_list()`` returns the composed modules in label order."""
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A")
    g.compose(module_b_h5, label="B", translate=(100.0, 0.0, 0.0))

    modules = g.compose_list()
    assert len(modules) == 2
    assert [m.label for m in modules] == ["A", "B"]


def test_compose_broker_mutation_preserves_compose_state(
    host_h5: Path, module_a_h5: Path,
) -> None:
    """A broker mutation after compose must not drop the composed module.

    Chain-phase sessions short-circuit ``get_fem_data()`` to the cached
    ``_fem`` chain head — so even after a counter bump (simulating any
    broker mutation) the composed module survives.
    """
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A")
    # Simulate a mutation that bumps the counter (e.g. what
    # ``g.constraints.bc.fix(...)`` would do post-extraction on a
    # begun session).
    g._bump_fem_counter()
    fem = g.mesh.queries.get_fem_data()
    # Compose state survives the invalidation.
    assert "A" in fem.composed_from
    assert fem.info.n_nodes == 6


def test_from_h5_session_compose_workflow(
    host_h5: Path, module_a_h5: Path, tmp_path: Path,
) -> None:
    """The full chain-phase workflow runs cleanly."""
    g = apeGmsh.from_h5(host_h5)
    g.compose(module_a_h5, label="A")
    out = tmp_path / "final.h5"
    g.save(out)
    # File exists + reloads correctly.
    assert out.exists()
    reloaded = FEMData.from_h5(str(out))
    assert "A" in reloaded.composed_from
