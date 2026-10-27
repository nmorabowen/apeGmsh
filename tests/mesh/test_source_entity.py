"""#2551 S1 — the neutral per-element source-entity stream (neutral 2.36.0).

``/elements/{alias}/entity`` is ``(E,) int32``, row-parallel to ``ids``:
the tag of the Gmsh model entity of the group's dimension each element
is classified on, partition entities resolved to their parent, ``-1``
unknown. Oracles, each independent of the code under test:

* Gmsh's own global ``getElements(dim, -1)`` for ids, connectivity, type
  order and the flat tag list (the per-entity capture must reproduce it
  byte for byte, so S1 moves no id, deck or golden);
* Gmsh's per-element ``getElement(tag)`` entity, mapped through
  ``getParent``, for every element's source entity (I1), including a
  curve drawn against its natural direction;
* the unpartitioned extraction of the same mesh, by element id (I2);
* the stream written, and the stream read back (I4);
* the committed 2.35 corpus file (no stream, I6);
* hand-edited files and composite arguments the contract refuses (I5).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh import FEMData, LabelSet, PhysicalGroupSet
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._fem_extract import extract_raw
from apeGmsh.mesh.FEMData import ElementComposite
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

from tests.fixtures import schema

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "schema_corpus"


# ── helpers ─────────────────────────────────────────────────────────

def _frame_and_blocks(g) -> dict[str, int]:
    """Two fragment-conformal boxes plus two free lines, one reversed.

    Returns the tags the tests name: the reversed line, whose element
    nodes run against the order the curve was drawn in.
    """
    del g  # the session owns the current gmsh model; drive it directly
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.addBox(1, 0, 0, 1, 1, 1)
    gmsh.model.occ.fragment([(3, 1)], [(3, 2)])
    p1 = gmsh.model.occ.addPoint(3, 0, 0)
    p2 = gmsh.model.occ.addPoint(4, 0, 0)
    p3 = gmsh.model.occ.addPoint(4, 1, 0)
    fwd = gmsh.model.occ.addLine(p1, p2)
    rev = gmsh.model.occ.addLine(p3, p2)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.34)
    gmsh.model.mesh.generate(3)
    return {"fwd": int(fwd), "rev": int(rev)}


def _oracle_entity(eid: int) -> int | None:
    """Source entity from Gmsh's per-element lookup, or ``None``.

    ``getElement`` does not know the interface elements a partition
    creates; those return ``None`` and are checked separately.
    """
    try:
        _, _, edim, etag = gmsh.model.mesh.getElement(int(eid))
    except Exception:
        return None
    p_dim, p_tag = gmsh.model.getParent(edim, etag)
    if p_dim < 0:
        return int(etag)
    return int(p_tag) if p_dim == edim else -1


def _check_against_gmsh(dim: int | None) -> dict:
    """Assert extract_raw matches Gmsh's global call and per-element oracle."""
    raw = extract_raw(dim=dim)
    types, tags, nodes = gmsh.model.mesh.getElements(
        -1 if dim is None else dim, -1)
    assert list(raw["groups"]) == [int(t) for t in types]
    for k, code in enumerate(int(t) for t in types):
        grp = raw["groups"][code]
        assert grp["ids"].dtype == np.int64
        assert np.array_equal(grp["ids"], np.asarray(tags[k], dtype=np.int64))
        assert np.array_equal(
            grp["conn"].ravel(), np.asarray(nodes[k], dtype=np.int64))
        assert grp["entity"].dtype == np.int32
        assert grp["entity"].shape == grp["ids"].shape
        for eid, ent in zip(grp["ids"], grp["entity"]):
            want = _oracle_entity(int(eid))
            if want is None:
                # A partition-interface element: its parent entity has a
                # higher dimension, so it has no source of its own dim.
                assert ent == -1, eid
            else:
                assert ent == want, (code, int(eid), int(ent), want)
    assert raw["elem_tags"] == [int(t) for block in tags for t in block]
    return raw


def _by_id(raw: dict) -> dict[int, int]:
    return {int(e): int(s)
            for grp in raw["groups"].values()
            for e, s in zip(grp["ids"], grp["entity"])}


def _group(code: int, ids, dim: int = 1) -> ElementGroup:
    ids = np.asarray(ids, dtype=np.int64)
    info = make_type_info(code=code, gmsh_name=f"t{code}", dim=dim,
                          order=1, npe=2, count=ids.size)
    conn = np.stack([ids, ids + 1], axis=1)
    return ElementGroup(element_type=info, ids=ids, connectivity=conn)


def _composite(groups, **kw) -> ElementComposite:
    return ElementComposite(groups=groups, physical=PhysicalGroupSet({}),
                            labels=LabelSet({}), **kw)


@pytest.fixture(scope="module")
def frame_h5(tmp_path_factory) -> Path:
    """The frame-and-blocks model, extracted at every dim and saved."""
    path = tmp_path_factory.mktemp("sle") / "frame.h5"
    with apeGmsh(model_name="sle_frame", verbose=False) as g:
        _frame_and_blocks(g)
        fem = g.mesh.queries.get_fem_data(dim=None)
        fem.to_h5(str(path))
    return path


# ── I1 / I3: capture ────────────────────────────────────────────────

def test_capture_reproduces_gmsh_and_names_each_elements_entity():
    with apeGmsh(model_name="sle_capture", verbose=False) as g:
        tags = _frame_and_blocks(g)
        for dim in (None, 0, 1, 2, 3):
            _check_against_gmsh(dim)
        raw = extract_raw(dim=1)
        line = raw["groups"][1]
        # The reversed curve's elements carry its tag like any other.
        _, rev_ids, _ = gmsh.model.mesh.getElements(1, tags["rev"])
        rows = np.isin(line["ids"], np.asarray(rev_ids[0], dtype=np.int64))
        assert rows.any()
        assert set(line["entity"][rows].tolist()) == {tags["rev"]}


def test_partitioned_entities_resolve_to_their_parent():
    """I2: partitioning leaves ``entity`` unchanged by element id, and
    the interface elements it adds are ``-1`` (no member is cut at a
    rank boundary)."""
    with apeGmsh(model_name="sle_part", verbose=False) as g:
        _frame_and_blocks(g)
        before = _by_id(extract_raw(dim=None))
        g.mesh.partitioning.partition(2)
        assert gmsh.model.getNumberOfPartitions() == 2
        # The partition moved every element onto partition entities.
        moved = [t for d, t in gmsh.model.getEntities(1)
                 if gmsh.model.getParent(1, t)[0] == 1
                 and len(gmsh.model.mesh.getElements(1, t)[1])]
        assert moved
        gmsh.model.mesh.rebuildElementCache()
        after = _by_id(_check_against_gmsh(None))
    common = before.keys() & after.keys()
    assert common == before.keys()
    assert all(after[e] == before[e] for e in common)
    added = after.keys() - before.keys()
    assert added  # PartitionCreateTopology adds interface elements
    assert {after[e] for e in added} == {-1}


_MSH2_TAG0 = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
1
1 3 2 0 0 1 2 3 4
$EndElements
"""


def test_msh2_elementary_tag_zero_reads_as_unknown(tmp_path):
    """A MSH 2.2 element on elementary tag 0 (legal in that format, and
    read on main) is classified on entity 0, which is no source tag."""
    path = tmp_path / "tag0.msh"
    path.write_text(_MSH2_TAG0)
    fem = FEMData.from_msh(str(path), dim=2)
    (grp,) = list(fem.elements)
    assert grp.ids.tolist() == [1]
    assert fem.elements.entity[grp.type_code].tolist() == [-1]


def test_ghost_cells_carry_no_source_entity():
    """Ghost entities (``PartitionCreateGhostCells``) have no parent and
    hold copies of another partition's elements: their rows are ``-1``,
    and every element still has its real surface on its own row."""
    with apeGmsh(model_name="sle_ghost", verbose=False):
        surf = gmsh.model.occ.addRectangle(0, 0, 0, 2, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.5)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 1)
        try:
            gmsh.model.mesh.partition(2)
            ghosts = [t for d, t in gmsh.model.getEntities(2)
                      if gmsh.model.getType(d, t).startswith("Ghost")]
            raw = extract_raw(dim=2)
        finally:
            gmsh.option.setNumber("Mesh.PartitionCreateGhostCells", 0)
    assert ghosts
    ent = np.concatenate([g["entity"] for g in raw["groups"].values()])
    ids = np.concatenate([g["ids"] for g in raw["groups"].values()])
    assert set(ent.tolist()) == {surf, -1}
    assert not set(ghosts) & set(ent.tolist())
    assert set(ids[ent == surf].tolist()) == set(ids.tolist())


def test_factory_carries_the_stream_onto_the_snapshot(frame_h5):
    fem = FEMData.from_h5(str(frame_h5))
    assert fem.elements.entity is not None
    assert set(fem.elements.entity) == {g.type_code for g in fem.elements}
    for grp in fem.elements:
        ent = fem.elements.entity[grp.type_code]
        assert ent.dtype == np.int32 and ent.shape == grp.ids.shape
        assert (ent > 0).all()  # unpartitioned: every element has a source


# ── I4: round trip ──────────────────────────────────────────────────

def test_round_trip_is_byte_identical(frame_h5, tmp_path):
    fem = FEMData.from_h5(str(frame_h5))
    again = tmp_path / "again.h5"
    fem.to_h5(str(again))
    with h5py.File(frame_h5, "r") as a, h5py.File(again, "r") as b:
        assert sorted(a["elements"]) == sorted(b["elements"])
        for alias in a["elements"]:
            da, db = a["elements"][alias]["entity"], b["elements"][alias]["entity"]
            assert da.dtype == np.int32 == db.dtype
            assert da[...].tobytes() == db[...].tobytes()
    back = FEMData.from_h5(str(again))
    for code, arr in fem.elements.entity.items():
        assert np.array_equal(back.elements.entity[code], arr)


def test_the_stream_stays_out_of_snapshot_id(frame_h5, tmp_path):
    stripped = tmp_path / "stripped.h5"
    shutil.copy(frame_h5, stripped)
    with h5py.File(stripped, "r+") as f:
        for alias in list(f["elements"]):
            del f["elements"][alias]["entity"]
    with_stream = FEMData.from_h5(str(frame_h5))
    without = FEMData.from_h5(str(stripped))
    assert without.elements.entity is None
    assert with_stream.snapshot_id == without.snapshot_id


def test_a_streamless_snapshot_writes_no_dataset(frame_h5, tmp_path):
    stripped = tmp_path / "stripped.h5"
    shutil.copy(frame_h5, stripped)
    with h5py.File(stripped, "r+") as f:
        for alias in list(f["elements"]):
            del f["elements"][alias]["entity"]
    fem = FEMData.from_h5(str(stripped))
    assert fem.elements.entity is None
    out = tmp_path / "out.h5"
    fem.to_h5(str(out))
    with h5py.File(out, "r") as f:
        assert not any("entity" in f["elements"][a] for a in f["elements"])
    grp = next(iter(fem.elements))
    _, ent = fem.elements.source_entity(grp.ids[:2], dim=grp.element_type.dim)
    assert ent.tolist() == [-1, -1]


# ── I4: compose carries the stream row for row ─────────────────────

def _compose(host, source: Path, label: str, dx: float) -> FEMData:
    from apeGmsh.mesh._compose import _compose_module
    return _compose_module(host, str(source), label=label,
                           translate=(dx, 0.0, 0.0), rotate=None,
                           partition_rank=None)


def test_compose_carries_entity_unrewritten_and_keys_stay_distinct(frame_h5):
    from apeGmsh.assembly._assembly import _base_fem

    src = FEMData.from_h5(str(frame_h5))
    fem = _compose(_compose(_base_fem(), frame_h5, "a", 0.0),
                   frame_h5, "b", 10.0)
    assert fem.elements.entity is not None
    for grp in fem.elements:
        code = grp.type_code
        ent = fem.elements.entity[code]
        lbl = fem.elements.module_label[code]
        src_ent = src.elements.entity[code]
        assert ent.shape == grp.ids.shape
        # Rows are the two modules' rows in order, values never offset.
        assert np.array_equal(ent, np.concatenate([src_ent, src_ent]))
        assert lbl.tolist() == ["a"] * src_ent.size + ["b"] * src_ent.size
        keys_a = set(zip(lbl[: src_ent.size].tolist(),
                         ent[: src_ent.size].tolist()))
        keys_b = set(zip(lbl[src_ent.size:].tolist(),
                         ent[src_ent.size:].tolist()))
        assert keys_a and keys_a.isdisjoint(keys_b)
        # The member key the planner (S2) asks for, by element id.
        labels, by_id = fem.elements.source_entity(grp.ids, dim=grp.element_type.dim)
        assert np.array_equal(by_id, ent) and labels.tolist() == lbl.tolist()


def test_compose_fills_minus_one_for_a_streamless_side(frame_h5, tmp_path):
    from apeGmsh.assembly._assembly import _base_fem

    stripped = tmp_path / "old.h5"
    shutil.copy(frame_h5, stripped)
    with h5py.File(stripped, "r+") as f:
        for alias in list(f["elements"]):
            del f["elements"][alias]["entity"]
    src = FEMData.from_h5(str(frame_h5))
    host = _compose(_base_fem(), stripped, "old", 0.0)
    assert host.elements.entity is None  # neither side carries it
    fem = _compose(host, frame_h5, "new", 10.0)
    for grp in fem.elements:
        code = grp.type_code
        n_old = src.elements.entity[code].size
        ent = fem.elements.entity[code]
        assert ent[:n_old].tolist() == [-1] * n_old
        assert np.array_equal(ent[n_old:], src.elements.entity[code])


def test_composed_stream_round_trips(frame_h5, tmp_path):
    from apeGmsh.assembly._assembly import _base_fem

    fem = _compose(_compose(_base_fem(), frame_h5, "a", 0.0),
                   frame_h5, "b", 10.0)
    out = tmp_path / "composed.h5"
    fem.to_h5(str(out))
    back = FEMData.from_h5(str(out))
    for code, arr in fem.elements.entity.items():
        assert np.array_equal(back.elements.entity[code], arr)


# ── I6: an older file reads as no stream ───────────────────────────

def test_a_2_35_corpus_file_opens_with_no_stream():
    minor = schema.NEUTRAL_BEFORE_ENTITY.rsplit(".", 1)[0]
    path = CORPUS / f"neutral_{minor}.h5"
    assert path.is_file(), path
    with h5py.File(path, "r") as f:
        stamp = f["meta"].attrs["neutral_schema_version"]
        assert stamp == schema.NEUTRAL_BEFORE_ENTITY
        assert not any("entity" in f["elements"][a] for a in f["elements"])
    fem = FEMData.from_h5(str(path))
    assert len(fem.elements) > 0
    assert fem.elements.entity is None


# ── I5: refusals ────────────────────────────────────────────────────

def _edited(frame_h5: Path, tmp_path: Path, edit) -> Path:
    path = tmp_path / "edited.h5"
    shutil.copy(frame_h5, path)
    with h5py.File(path, "r+") as f:
        edit(f["elements"])
    return path


def _first(elements) -> h5py.Group:
    return elements[sorted(elements)[0]]


def _set_entity(value):
    def edit(elements):
        g = _first(elements)
        ent = g["entity"][...]
        ent[0] = value
        del g["entity"]
        g.create_dataset("entity", data=ent)
    return edit


def _short(elements):
    g = _first(elements)
    ent = g["entity"][...]
    del g["entity"]
    g.create_dataset("entity", data=ent[:-1])


def _partial(elements):
    del _first(elements)["entity"]


def _float(elements):
    g = _first(elements)
    ent = g["entity"][...].astype(np.float64)
    del g["entity"]
    g.create_dataset("entity", data=ent)


@pytest.mark.parametrize("edit, match", [
    (_set_entity(0), "neither -1 nor"),
    (_set_entity(-2), "neither -1 nor"),
    (_short, "shape"),
    (_partial, "some groups"),
    (_float, "neither -1 nor"),
], ids=["zero", "minus-two", "short", "partial", "float"])
def test_malformed_streams_refuse(frame_h5, tmp_path, edit, match):
    path = _edited(frame_h5, tmp_path, edit)
    with pytest.raises(MalformedH5Error, match=match):
        FEMData.from_h5(str(path))


def test_composite_refuses_a_bad_stream():
    groups = {1: _group(1, [1, 2, 3])}
    with pytest.raises(ValueError, match="shape"):
        _composite(groups, entity={1: np.array([1, 2], dtype=np.int32)})
    with pytest.raises(ValueError, match="no element group"):
        _composite(groups, entity={1: np.ones(3, np.int32),
                                   9: np.ones(1, np.int32)})
    for bad in (0, -2, 2**31):
        with pytest.raises(ValueError, match="positive"):
            _composite(groups, entity={1: np.array([1, bad, 2])})
    with pytest.raises(ValueError, match="integers"):
        _composite(groups, entity={1: np.array([1.0, 2.0, 3.0])})


def test_composite_fills_a_missing_code_with_minus_one():
    groups = {1: _group(1, [1, 2]), 8: _group(8, [7, 8, 9])}
    comp = _composite(groups, entity={1: np.array([4, 5])})
    assert comp.entity[1].dtype == np.int32
    assert comp.entity[1].tolist() == [4, 5]
    assert comp.entity[8].tolist() == [-1, -1, -1]


# ── source_entity ───────────────────────────────────────────────────

def test_source_entity_is_vectorised_over_one_dims_groups():
    groups = {
        1: _group(1, [10, 11, 12], dim=1),
        8: _group(8, [20, 21], dim=1),
        2: _group(2, [30, 31], dim=2),
    }
    comp = _composite(
        groups,
        entity={1: np.array([5, 5, 6]), 8: np.array([7, -1]),
                2: np.array([3, 3])},
        module_label={1: np.array(["a", "a", "b"], dtype=object)},
    )
    labels, ent = comp.source_entity([21, 12, 10, 20], dim=1)
    assert ent.dtype == np.int32
    assert ent.tolist() == [-1, 6, 5, 7]
    assert labels.tolist() == ["", "b", "a", ""]
    with pytest.raises(KeyError, match="dimension 1"):
        comp.source_entity([30], dim=1)  # a dim-2 element
    with pytest.raises(KeyError, match="dimension 1"):
        comp.source_entity([99], dim=1)
    labels, ent = comp.source_entity([], dim=3)
    assert labels.size == 0 and ent.size == 0


def test_source_entity_maps_interleaved_ids_through_the_sort():
    """Ids that interleave across groups: concatenation order is not id
    order, so a lookup that skips the sort permutation answers wrong."""
    groups = {2: _group(2, [1, 2, 81, 82]), 3: _group(3, [51, 52])}
    comp = _composite(groups, entity={2: np.array([2, 2, 3, 3]),
                                      3: np.array([7, 8])})
    _, ent = comp.source_entity([51, 81, 1], dim=1)
    assert ent.tolist() == [7, 3, 2]
    _, ent = comp.source_entity([82, 52, 2], dim=1)
    assert ent.tolist() == [3, 8, 2]
