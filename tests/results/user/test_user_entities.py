"""``/entities``: cuts, storeys, point sets, and the results that name them (VU8 #2692).

Oracles:

* the layout of ``architecture/h5-schema.md`` (``/entities``), read with h5py;
* a :class:`~apeGmsh.cuts.SectionCutDef` exported, written, read and turned back
  into a def equals the original field for field, so the cut kernel fed its
  ``to_spec()`` computes the same resultant;
* a cantilever's cut resultant by statics, stored about the cut's reference,
  reads back through its ``(ENTITY_KIND, ENTITY_ROW)`` and moves to another
  point by ``M + (reference - P) x F`` to the statics there.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.cuts import SectionCutDef
from apeGmsh.opensees._internal.schema_version import (
    ENTITIES,
    ENTITIES_KEY,
    SchemaVersionError,
    reader_version,
)
from apeGmsh.results.user import (
    CutEntity,
    Entities,
    PointSet,
    StateAddress,
    Storey,
    UserFile,
    UserResult,
    UserWriteRefused,
    ZoneRefused,
    staleness,
    write_user_file,
)
from tests.fixtures.schema import ENTITIES_CURRENT
from tests.results.user._user_helpers import nodal, sample_entities


def _bits(a: object) -> bytes:
    return np.asarray(a, dtype="f8").tobytes()


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------


def test_entities_round_trip_bit_equal(make_model, tmp_path: Path) -> None:
    e = sample_entities()
    path = write_user_file(tmp_path / "s.user.h5", make_model(), entities=e)
    back = UserFile(path).entities()
    assert back.same_bits(e)
    assert [c.name for c in back.cuts] == ["pier P1 base", "Delta_after (L/480)"]
    assert back.cuts[1].side == "negative" and back.cuts[1].polygon.shape == (3, 3)
    assert back.cuts[0].polygon.shape == (0, 3)
    assert back.storeys[0].level == 3.0 and back.storeys[0].node_ids.tolist() == [5, 6, 7, 8]
    assert back.point_sets[0].host_element.tolist() == [-1, 102]


def test_entities_layout_is_the_schema(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), entities=sample_entities())
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs[ENTITIES_KEY] == ENTITIES_CURRENT
        zone = f["entities"]
        assert zone.attrs["element_index_space"] == "opensees"
        cuts = zone["cuts"]
        assert [x.decode() for x in cuts["name"][...]] == ["pier P1 base", "Delta_after (L/480)"]
        assert cuts["point"].shape == cuts["normal"].shape == cuts["reference"].shape == (2, 3)
        assert cuts["element_offsets"][...].tolist() == [0, 2, 3]
        assert cuts["element_ids"][...].tolist() == [101, 102, 7]
        assert cuts["polygon_offsets"][...].tolist() == [0, 0, 3]
        assert cuts["polygon"].shape == (3, 3)
        storeys = zone["storeys"]
        assert storeys["level"][...].tolist() == [3.0]
        assert storeys["node_offsets"][...].tolist() == [0, 4]
        points = zone["point_sets"]
        assert points["point_offsets"][...].tolist() == [0, 2]
        assert points["xyz"].shape == (2, 3)
        assert points["host_element"][...].tolist() == [-1, 102]
        for ds in (cuts["element_offsets"], cuts["element_ids"], storeys["node_ids"],
                   points["host_element"]):
            assert ds.dtype == np.dtype("i4")  # never int64 in this file


def test_empty_tables_are_written_and_read(make_model, tmp_path: Path) -> None:
    only = Entities(storeys=sample_entities().storeys)
    path = write_user_file(tmp_path / "s.user.h5", make_model(), entities=only)
    with h5py.File(path, "r") as f:
        assert f["entities/cuts/element_offsets"][...].tolist() == [0]
        assert f["entities/cuts/point"].shape == (0, 3)
        assert f["entities/point_sets/xyz"].shape == (0, 3)
    back = UserFile(path).entities()
    assert back.cuts == () and back.point_sets == () and back.same_bits(only)


def test_a_file_without_entities_reads_empty(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal()])
    back = UserFile(path).entities()
    assert (back.cuts, back.storeys, back.point_sets) == ((), (), ())


def test_entities_replace_whole_and_none_leaves_them(make_model, tmp_path: Path) -> None:
    model = make_model()
    path = write_user_file(tmp_path / "s.user.h5", model, entities=sample_entities())
    write_user_file(path, model, results=[nodal()])
    assert UserFile(path).entities().same_bits(sample_entities())
    smaller = Entities(storeys=sample_entities().storeys)
    write_user_file(path, model, entities=smaller)
    assert UserFile(path).entities().same_bits(smaller)


def test_entities_go_stale_with_the_model(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), entities=sample_entities())
    assert not staleness(path, model_h5=make_model()).zones["entities"].stale
    other = make_model("other.h5", fem="0" * 32)
    assert staleness(path, model_h5=other).zones["entities"].stale
    with pytest.raises(UserWriteRefused, match="entities"):
        write_user_file(path, other)
    write_user_file(path, other, drop_stale=True)
    assert "entities" not in UserFile(path).zones
    with h5py.File(path, "r") as f:
        assert ENTITIES_KEY not in f["meta"].attrs


# ---------------------------------------------------------------------------
# The cut export
# ---------------------------------------------------------------------------


def _cut(**kw) -> SectionCutDef:
    fields = dict(
        plane_point=(0.0, 0.0, 1500.0), plane_normal=(0.0, 0.0, 2.0),
        element_ids=(11, 12, 40), label="Story 1 base shear",
        bounding_polygon=((-500, -500, 1500), (500, -500, 1500),
                          (500, 500, 1500), (-500, 500, 1500)),
    )
    fields.update(kw)
    return SectionCutDef(**fields)


@pytest.mark.parametrize("cut", [
    _cut(),
    _cut(side="negative", bounding_polygon=None, label=None),
    _cut(plane_normal=(1.0, 2.0, -3.0), plane_point=(0.1, 0.2, 0.3), bounding_polygon=None),
])
def test_a_cut_def_comes_back_field_for_field(make_model, tmp_path: Path,
                                              cut: SectionCutDef) -> None:
    entity = cut.to_entity(name="cut A")
    path = write_user_file(tmp_path / "s.user.h5", make_model(),
                           entities=Entities(cuts=(entity,)))
    again = UserFile(path).entities().row("cut", 0).to_cut_def()
    assert again.plane_point == cut.plane_point
    assert again.plane_normal == cut.plane_normal  # bit-equal: already unit
    assert again.element_ids == cut.element_ids
    assert again.side == cut.side
    assert again.bounding_polygon == cut.bounding_polygon
    assert again.label == "cut A"


def test_to_entity_defaults_and_refusals() -> None:
    entity = _cut().to_entity()
    assert entity.name == "Story 1 base shear"
    assert _bits(entity.reference) == _bits((0.0, 0.0, 1500.0))  # the plane point
    assert _bits(_cut().to_entity(reference=(1, 2, 3)).reference) == _bits((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="no label"):
        _cut(label=None).to_entity()
    with pytest.raises(ValueError, match="i4"):
        _cut(element_ids=(2 ** 31,)).to_entity()


# ---------------------------------------------------------------------------
# A cut resultant on the cut (closed form)
# ---------------------------------------------------------------------------

#: A cantilever column on z, fixed at 0, tip at H, loaded at the tip by P.
H, ZC = 4.0, 1.5
P_TIP = np.array([[30.0, -12.0, -200.0], [-45.0, 0.0, -350.0]])  # two load states


def _statics(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The force and moment the discarded (lower) part exerts on the kept upper
    part, about ``reference``: the kept part is in equilibrium under them and P."""
    tip = np.array([0.0, 0.0, H])
    force = -P_TIP
    moment = -np.cross(tip - reference, P_TIP)
    return force, moment


def test_a_cut_resultant_reads_back_on_its_cut(make_model, tmp_path: Path) -> None:
    cut = SectionCutDef(plane_point=(0.0, 0.0, ZC), plane_normal=(0.0, 0.0, 1.0),
                        element_ids=(2,), side="positive", label="column cut")
    entity = cut.to_entity(reference=(0.0, 0.0, ZC))
    force, moment = _statics(entity.reference)
    data = np.concatenate([force, moment], axis=1)[:, None, :]  # (S, n=1, 6)
    result = UserResult(
        name="cut resultant", target="entities", basis="model", linear=True,
        states=(StateAddress("stage", "push", 0, 0), StateAddress("stage", "push", 1, 0)),
        components=("Fx", "Fy", "Fz", "Mx", "My", "Mz"), data=data,
        entity_kind=("cut",), entity_row=np.array([0], dtype="i4"),
    )
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[result],
                           entities=Entities(cuts=(entity,)))
    uf = UserFile(path)
    back = uf.read_result("cut resultant")
    held = uf.entities().row(back.entity_kind[0], int(back.entity_row[0]))
    assert held.name == "column cut"
    f_back, m_back = back.data[:, 0, :3], back.data[:, 0, 3:]
    assert _bits(f_back) == _bits(force) and _bits(m_back) == _bits(moment)
    # The stored moment is about the cut's reference: moved to the base, it is
    # the statics about the base, which is the fixed end's reaction moment.
    base = np.array([0.0, 0.0, 0.0])
    moved = m_back + np.cross(held.reference - base, f_back)
    np.testing.assert_allclose(moved, _statics(base)[1], rtol=0, atol=1e-9)
    assert moved[0, 1] == pytest.approx(-H * P_TIP[0, 0])


# ---------------------------------------------------------------------------
# Binding: ENTITY_KIND and ENTITY_ROW
# ---------------------------------------------------------------------------


def _on(kind: str, row: int) -> UserResult:
    return UserResult(name="drift", target="entities", basis="model", components=("dx",),
                      data=np.zeros((1, 1, 1)), entity_kind=(kind,),
                      entity_row=np.array([row], dtype="i4"))


def test_a_result_naming_a_missing_row_refuses_the_write(make_model, tmp_path: Path) -> None:
    target = tmp_path / "s.user.h5"
    with pytest.raises(UserWriteRefused, match="storeys holds 1"):
        write_user_file(target, make_model(), results=[_on("storey", 1)],
                        entities=sample_entities())
    assert not target.exists() and not list(tmp_path.glob("*.partial"))
    with pytest.raises(UserWriteRefused, match="holds 0"):
        write_user_file(target, make_model(), results=[_on("cut", 0)])


def test_an_entities_rewrite_that_strands_a_result_refuses(make_model, tmp_path: Path) -> None:
    model = make_model()
    path = write_user_file(tmp_path / "s.user.h5", model, results=[_on("cut", 1)],
                           entities=sample_entities())
    with pytest.raises(UserWriteRefused, match="cut row 1"):
        write_user_file(path, model, entities=Entities(cuts=sample_entities().cuts[:1]))
    assert UserFile(path).entities().same_bits(sample_entities())  # left whole


def test_an_unknown_entity_kind_is_refused(make_model, tmp_path: Path) -> None:
    with pytest.raises(UserWriteRefused, match="not one of"):
        write_user_file(tmp_path / "s.user.h5", make_model(), results=[_on("storeys", 0)],
                        entities=sample_entities())
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[_on("storey", 0)],
                           entities=sample_entities())
    with h5py.File(path, "a") as f:
        f["results/r0/RESULTS/ON_ENTITIES/values/ENTITY_KIND"][0] = "wall"
    index = UserFile(path).results()
    assert index.entries == () and index.refused[0].reason == "bad-param"
    assert "wall" in index.refused[0].detail


# ---------------------------------------------------------------------------
# Refusals of the zone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entities, match", [
    (Entities(cuts=(sample_entities().cuts[0],) * 2), "given twice"),
    (Entities(storeys=(Storey(name=" L1", level=0.0, reference=(0, 0, 0),
                              node_ids=np.array([1])),)), "whitespace"),
    (Entities(cuts=(CutEntity(name="c", point=(0, 0, 0), normal=(0, 0, 2), reference=(0, 0, 0),
                              element_ids=np.array([1])),)), "unit length"),
    (Entities(cuts=(CutEntity(name="c", point=(0, 0, 0), normal=(0, 0, 1), reference=(0, 0, 0),
                              element_ids=np.array([1]), side="up"),)), "side"),
    (Entities(cuts=(CutEntity(name="c", point=(0, 0, 0), normal=(0, 0, 1), reference=(0, 0, 0),
                              element_ids=np.zeros(0, dtype="i4")),)), "no element"),
    (Entities(cuts=(CutEntity(name="c", point=(0, 0, 0), normal=(0, 0, 1), reference=(0, 0, 0),
                              element_ids=np.array([1]), polygon=np.zeros((2, 3))),)), "three"),
    (Entities(storeys=(Storey(name="L", level=0.0, reference=(0, 0, 0),
                              node_ids=np.zeros(0, dtype="i4")),)), "no node"),
    (Entities(point_sets=(PointSet(name="p", xyz=np.zeros((1, 3)),
                                   host_element=np.array([-2])),)), "below -1"),
])
def test_a_bad_entity_refuses_the_write(make_model, tmp_path: Path, entities, match) -> None:
    with pytest.raises(UserWriteRefused, match=match):
        write_user_file(tmp_path / "s.user.h5", make_model(), entities=entities)


def test_non_finite_geometry_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="finite"):
        CutEntity(name="c", point=(0, 0, np.nan), normal=(0, 0, 1), reference=(0, 0, 0),
                  element_ids=np.array([1]))
    with pytest.raises(ValueError, match="finite"):
        Storey(name="L", level=np.inf, reference=(0, 0, 0), node_ids=np.array([1]))


@pytest.mark.parametrize("damage, match", [
    (lambda f: f["entities"].attrs.__setitem__("element_index_space", "fem"), "element_index_space"),
    (lambda f: f["entities"].__delitem__("storeys"), "no storeys table"),
    (lambda f: f["entities/cuts"].__delitem__("reference"), "no reference column"),
    (lambda f: _replace(f, "entities/cuts/element_offsets", np.array([0, 2, 9], dtype="i4")),
     "does not run from 0 to 3"),
    (lambda f: _replace(f, "entities/cuts/element_ids", np.array([101, 102, 7], dtype="i8")),
     "expected i4"),
    (lambda f: _replace(f, "entities/cuts/normal", np.zeros((2, 3))), "unit length"),
    (lambda f: _replace(f, "entities/storeys/level", np.array([np.nan])), "finite"),
])
def test_a_damaged_zone_refuses_alone(make_model, tmp_path: Path, damage, match) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal()],
                           entities=sample_entities())
    with h5py.File(path, "a") as f:
        damage(f)
    with pytest.raises(ZoneRefused, match=match):
        UserFile(path).entities()
    assert UserFile(path).results().entries  # the other zones still read


def _replace(f: h5py.File, link: str, data: np.ndarray) -> None:
    del f[link]
    f[link] = data


def test_a_newer_entities_version_refuses(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), entities=sample_entities())
    newer = reader_version(ENTITIES)
    with h5py.File(path, "a") as f:
        f["meta"].attrs[ENTITIES_KEY] = f"{newer.major + 1}.0.0"
    with pytest.raises(SchemaVersionError):
        UserFile(path).entities()
    with pytest.raises(SchemaVersionError):
        write_user_file(path, make_model(), entities=sample_entities())


def test_a_zone_without_its_key_refuses(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), entities=sample_entities())
    with h5py.File(path, "a") as f:
        del f["meta"].attrs[ENTITIES_KEY]
    with pytest.raises(ZoneRefused, match="no version key"):
        UserFile(path).entities()
    with pytest.raises(UserWriteRefused, match="entities_schema_version"):
        write_user_file(path, make_model(), results=[nodal()])
