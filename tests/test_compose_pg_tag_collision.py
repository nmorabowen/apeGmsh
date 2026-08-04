"""Compose PG / label ``(dim, tag)`` key-collision regression tests.

Gmsh auto-assigns physical-group and label tags ``1, 2, 3, ...`` in
EVERY source file, so every composed module re-uses the host's ``(dim,
tag)`` pairs.  Historically the merge shifted a colliding bundle key to
``(dim, tag + 1_000_000_000)`` — a fixed single shift, so the SECOND
colliding module landed on the FIRST module's shifted key and silently
overwrote it.  With 3+ modules every compose destroyed the previous
module's PGs, labels, and mesh selections; a downstream
``g.constraints.tie(master_label=...)`` against a destroyed PG resolves
to a silent no-op and the part is left unattached.

Locked here:

* the rewriter offsets group KEYS into the module's reserved window
  (disjoint by construction, mirroring node / element tag offsets);
* 3+ module composes keep every host + module PG / label / selection;
* the merge never overwrites — a forced collision probes a FREE key
  and warns (:class:`ComposeTagCollisionWarning`);
* the lossless count invariant raises :class:`ComposeTagCollisionError`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh.mesh import (
    ComposeTagCollisionError,
    ComposeTagCollisionWarning,
)
from apeGmsh.mesh._compose import (
    _merged_mesh_selection,
    _merged_named_groups,
    _next_free_group_key,
)
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.mesh.MeshSelectionSet import MeshSelectionStore


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


_COORDS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _pg(name: str, nids: list[int]) -> dict:
    arr = np.asarray(nids, dtype=np.int64)
    return {
        "name": name,
        "node_ids": arr,
        "node_coords": _COORDS[arr - 1],
    }


def _make_module_fem(prefix: str) -> FEMData:
    """4-node / 2-line module whose PGs + labels use gmsh-style auto
    tags ``(dim, 1), (dim, 2), (dim, 3)`` — identical in EVERY module,
    exactly the collision the merge must survive."""
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    elem_ids = np.array([10, 11], dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2,
    )
    group = ElementGroup(
        element_type=line_info,
        ids=elem_ids,
        connectivity=np.array([[1, 2], [2, 3]], dtype=np.int64),
    )
    node_pgs = {
        (2, 1): _pg(f"{prefix}_top", [1, 2]),
        (2, 2): _pg(f"{prefix}_bot", [3, 4]),
        (3, 3): _pg(f"{prefix}_vol", [1, 2, 3, 4]),
    }
    elem_pgs = {
        (1, 1): {
            **_pg(f"{prefix}_lines", [1, 2, 3]),
            "element_ids": np.array([10, 11], dtype=np.int64),
        },
    }
    node_labels = {
        (2, 1): _pg(f"{prefix}_lbl", [1, 2]),
    }
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=_COORDS,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet(node_labels),
    )
    elements = ElementComposite(
        groups={1: group},
        physical=PhysicalGroupSet(elem_pgs),
        labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=4, n_elems=2, bandwidth=1, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info)


MODULE_LABELS = ("m1", "m2", "m3", "m4")


@pytest.fixture
def composed_session(tmp_path: Path):
    """Host + 4 composed modules, all with colliding gmsh auto tags."""
    host_p = tmp_path / "host.h5"
    _make_module_fem("host").to_h5(str(host_p))
    g = apeGmsh.from_h5(host_p)
    for label in MODULE_LABELS:
        p = tmp_path / f"{label}.h5"
        _make_module_fem(label).to_h5(str(p))
        g.compose(p, label=label)
    return g


# ---------------------------------------------------------------------------
# End-to-end: 3+ modules, every PG / label survives
# ---------------------------------------------------------------------------


def test_multi_module_compose_keeps_every_pg_and_label(
    composed_session,
) -> None:
    """Host + 4 modules -> 5x every group, correctly namespaced."""
    fem = composed_session.mesh.queries.get_fem_data(dim=None)

    expected_node_pgs = sorted(
        ["host_top", "host_bot", "host_vol"]
        + [f"{m}.{m}_{s}" for m in MODULE_LABELS
           for s in ("top", "bot", "vol")]
    )
    expected_elem_pgs = sorted(
        ["host_lines"] + [f"{m}.{m}_lines" for m in MODULE_LABELS]
    )
    expected_labels = sorted(
        ["host_lbl"] + [f"{m}.{m}_lbl" for m in MODULE_LABELS]
    )

    assert sorted(fem.nodes.physical.names()) == expected_node_pgs
    assert sorted(fem.elements.physical.names()) == expected_elem_pgs
    assert sorted(fem.nodes.labels.names()) == expected_labels

    # Count invariant: nothing merged away.
    assert len(fem.nodes.physical) == 3 * (1 + len(MODULE_LABELS))
    assert len(fem.elements.physical) == 1 * (1 + len(MODULE_LABELS))
    assert len(fem.nodes.labels) == 1 * (1 + len(MODULE_LABELS))


def test_multi_module_compose_pg_membership_is_module_local(
    composed_session,
) -> None:
    """Each surviving PG resolves to ITS module's offset node ids —
    the exact lookup ``g.constraints.tie(master_label=...)`` performs."""
    fem = composed_session.mesh.queries.get_fem_data(dim=None)
    node_ml = np.asarray(fem.nodes._module_label, dtype=object)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    for m in MODULE_LABELS:
        owned = set(
            int(t) for t, lbl in zip(node_ids, node_ml) if str(lbl) == m
        )
        assert owned, f"module {m} owns no nodes"
        pg_nodes = fem.nodes.physical.node_ids(f"{m}.{m}_vol")
        got = {int(x) for x in pg_nodes}
        assert got, f"{m}.{m}_vol resolved empty"
        assert got <= owned, (
            f"{m}.{m}_vol nodes {sorted(got)[:5]}... escaped module "
            f"{m}'s rows"
        )


def test_multi_module_compose_keys_are_disjoint_per_module(
    composed_session,
) -> None:
    """Bundle group keys were offset into each module's reserved
    window: 5 distinct (dim, tag) keys carry the 5 '*_vol' names."""
    fem = composed_session.mesh.queries.get_fem_data(dim=None)
    groups = fem.nodes.physical._groups
    vol_keys = {
        key: info["name"]
        for key, info in groups.items()
        if str(info.get("name", "")).endswith("_vol")
    }
    assert len(vol_keys) == 1 + len(MODULE_LABELS)
    # Host keeps its small gmsh tag; every module's key sits above its
    # window base (>= 1_000_000 granularity).
    tags_by_name = {v: k[1] for k, v in vol_keys.items()}
    assert tags_by_name["host_vol"] == 3
    module_tags = [
        tags_by_name[f"{m}.{m}_vol"] for m in MODULE_LABELS
    ]
    assert all(t >= 1_000_000 for t in module_tags)
    assert len(set(module_tags)) == len(module_tags)


def test_multi_module_compose_survives_h5_round_trip(
    composed_session, tmp_path: Path,
) -> None:
    """Save the composed model and reload — nothing lost on disk."""
    out = tmp_path / "composed.h5"
    composed_session.save(out)
    reloaded = FEMData.from_h5(str(out))
    assert len(reloaded.nodes.physical) == 3 * (1 + len(MODULE_LABELS))
    assert len(reloaded.elements.physical) == 1 * (1 + len(MODULE_LABELS))
    assert len(reloaded.nodes.labels) == 1 * (1 + len(MODULE_LABELS))
    for m in MODULE_LABELS:
        assert f"{m}.{m}_vol" in reloaded.nodes.physical
        assert f"{m}.{m}_lines" in reloaded.elements.physical
        assert f"{m}.{m}_lbl" in reloaded.nodes.labels


# ---------------------------------------------------------------------------
# Unit: merge helpers never overwrite
# ---------------------------------------------------------------------------


def _entry(name: str) -> dict:
    return {
        "name": name,
        "node_ids": np.array([1], dtype=np.int64),
        "node_coords": np.zeros((1, 3), dtype=np.float64),
    }


def test_merged_named_groups_collision_probes_free_key() -> None:
    """A forced key collision re-keys (with a warning) — never drops."""
    host = {(2, 1): _entry("host_face")}
    bundle = {(2, 1): _entry("m.face")}
    with pytest.warns(ComposeTagCollisionWarning):
        out = _merged_named_groups(host, bundle)
    assert len(out) == 2
    names = sorted(info["name"] for info in out.values())
    assert names == ["host_face", "m.face"]
    # Host entry untouched at its original key.
    assert out[(2, 1)]["name"] == "host_face"


def test_merged_named_groups_three_way_collision_keeps_all() -> None:
    """The historical bug: module 2's shifted key == module 1's
    shifted key.  The probe must keep all three entries."""
    merged = {(2, 1): _entry("host_face")}
    with pytest.warns(ComposeTagCollisionWarning):
        merged = _merged_named_groups(merged, {(2, 1): _entry("m1.face")})
    with pytest.warns(ComposeTagCollisionWarning):
        merged = _merged_named_groups(merged, {(2, 1): _entry("m2.face")})
    with pytest.warns(ComposeTagCollisionWarning):
        merged = _merged_named_groups(merged, {(2, 1): _entry("m3.face")})
    assert len(merged) == 4
    names = sorted(info["name"] for info in merged.values())
    assert names == ["host_face", "m1.face", "m2.face", "m3.face"]


def test_merged_named_groups_disjoint_keys_pass_through() -> None:
    """No collision -> no warning, plain union."""
    host = {(2, 1): _entry("a")}
    bundle = {(2, 1_000_001): _entry("m.b")}
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", ComposeTagCollisionWarning)
        out = _merged_named_groups(host, bundle)
    assert set(out) == {(2, 1), (2, 1_000_001)}


def test_next_free_group_key_skips_occupied_tags() -> None:
    merged = {(2, 1): {}, (2, 2): {}, (2, 7): {}, (3, 9): {}}
    d, t = _next_free_group_key(merged, 2, 2)
    assert (d, t) not in merged
    assert d == 2 and t == 8  # max tag for dim 2 is 7 -> 8


def test_merged_mesh_selection_collision_probes_free_key() -> None:
    host = MeshSelectionStore({(2, 1): _entry("host_sel")})
    bundle = MeshSelectionStore({(2, 1): _entry("m.sel")})
    with pytest.warns(ComposeTagCollisionWarning):
        out = _merged_mesh_selection(host, bundle)
    assert len(out._sets) == 2
    names = sorted(info["name"] for info in out._sets.values())
    assert names == ["host_sel", "m.sel"]


def test_compose_tag_collision_error_is_compose_error() -> None:
    from apeGmsh.mesh import ComposeError

    assert issubclass(ComposeTagCollisionError, ComposeError)
    assert issubclass(ComposeTagCollisionError, ValueError)
