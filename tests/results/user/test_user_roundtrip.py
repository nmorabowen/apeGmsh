"""Write, read back bit-equal; ``/meta`` byte-equal; versions; verbatim copies."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._internal.schema_version import (
    COMBINATIONS,
    COMBINATIONS_KEY,
    SchemaVersionError,
    USER_META_KEY,
    USER_RESULTS,
    USER_RESULTS_KEY,
    reader_version,
)
from apeGmsh.results.readers._ladruno_element_io import _gp_param_for, parse_blocks
from apeGmsh.results.user import (
    Combination,
    ElementBucket,
    StateAddress,
    Term,
    UserFile,
    UserFileRefused,
    UserResult,
    UserWriteRefused,
    results_fingerprint,
    write_user_file,
)
from tests.fixtures.schema import (
    COMBINATIONS_CURRENT,
    USER_META_CURRENT,
    USER_RESULTS_CURRENT,
)

from .conftest import nodal

LINEAR = Combination(
    "1.2D+1.6L", "linear", (Term("stage", "D", -1, 0, 1.2), Term("stage", "L", 7, 0, 1.6)),
)
ENVELOPE = Combination(
    "S+E env", "envelope", (Term("combination", "1.2D+1.6L", factor=1.0),), envelope="max",
)
CQC = Combination("E_x", "cqc", (Term("spectrum", "SLE-X", -1, 0, 1.0),), damping=0.05)
DIRECTIONAL = Combination(
    "100/30", "directional",
    (Term("combination", "E_x"), Term("spectrum", "SLE-Y", -1, 3, -1.0)), fraction=0.3,
)
ROWS = [LINEAR, ENVELOPE, CQC, DIRECTIONAL]


def _bits(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype="f8").tobytes()


# ---- /meta -----------------------------------------------------------------


def test_meta_hashes_are_byte_equal_to_the_models_as_stored(make_model, tmp_path: Path) -> None:
    fem, model = "fem-Ω/hash with spaces", "MODEL 00ff"
    path = write_user_file(tmp_path / "s.user.h5", make_model(fem=fem, model=model))
    with h5py.File(path, "r") as out, h5py.File(tmp_path / "model.h5", "r") as src:
        for key in ("fem_hash", "model_hash"):
            assert out["meta"].attrs[key] == src["meta/lineage"].attrs[key]
    uf = UserFile(path)
    assert (uf.fem_hash, uf.model_hash) == (fem, model)


def test_a_model_with_only_fem_hash_writes_no_model_hash(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(model=None))
    with h5py.File(path, "r") as f:
        assert "model_hash" not in f["meta"].attrs
    assert UserFile(path).model_hash is None


def test_a_model_without_fem_hash_refuses(tmp_path: Path) -> None:
    model = tmp_path / "m.h5"
    with h5py.File(model, "w") as f:
        f.create_group("meta/lineage").attrs["model_hash"] = "x"
    with pytest.raises(UserWriteRefused, match="fem_hash"):
        write_user_file(tmp_path / "s.user.h5", model)
    assert not (tmp_path / "s.user.h5").exists()


def test_meta_records_results_hash_writer_units_and_revision(make_model, ladruno, tmp_path) -> None:
    path = tmp_path / "s.user.h5"
    write_user_file(path, make_model(), results_file=ladruno, units="kN,m", writer="yoo 1.0")
    uf = UserFile(path)
    assert uf.results_hash == results_fingerprint(ladruno)
    assert (uf.writer, uf.units, uf.revision) == ("yoo 1.0", "kN,m", 1)
    write_user_file(path, make_model())
    again = UserFile(path)
    assert again.revision == 2
    assert again.results_hash == uf.results_hash, "leaving results_file out keeps the hash"
    assert again.units == "kN,m"


def test_zone_keys_are_present_if_and_only_if_the_zone_is(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model())
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs[USER_META_KEY] == USER_META_CURRENT
        assert COMBINATIONS_KEY not in f["meta"].attrs and "combinations" not in f
        assert USER_RESULTS_KEY not in f["meta"].attrs and "results" not in f
    write_user_file(path, make_model(), combinations=[LINEAR], results=[nodal()])
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs[COMBINATIONS_KEY] == COMBINATIONS_CURRENT and "combinations" in f
        assert f["meta"].attrs[USER_RESULTS_KEY] == USER_RESULTS_CURRENT and "results" in f


# ---- /combinations ---------------------------------------------------------


def test_combinations_read_back_equal(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), combinations=ROWS)
    table = UserFile(path).combinations()
    assert table.refused == ()
    assert list(table.rows) == ROWS


def test_columns_match_the_layout_exactly(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), combinations=ROWS)
    with h5py.File(path, "r") as f:
        zone = f["combinations"]
        assert zone.attrs["current"] == 0 and zone.attrs["current"].dtype == np.int32
        g = zone["g0"]
        assert g["term_offsets"][...].tolist() == [0, 2, 3, 4, 6]
        assert g["term_offsets"].dtype == np.dtype("i4")
        assert g["fraction"].dtype == np.dtype("f8") and g["damping"].dtype == np.dtype("f8")
        assert np.isnan(g["fraction"][0]) and g["fraction"][3] == 0.3
        assert g["damping"][2] == 0.05
        assert g["envelope"].asstr()[...].tolist() == ["", "max", "", ""]
        terms = g["terms"]
        assert terms["kind"].asstr()[...].tolist()[:2] == ["stage", "stage"]
        assert terms["step"].dtype == np.dtype("i4") and terms["mode"].dtype == np.dtype("i4")
        assert terms["factor"].dtype == np.dtype("f8")
        assert terms["step"][1] == 7 and terms["factor"][1] == 1.6


def test_an_empty_table_round_trips(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), combinations=[])
    table = UserFile(path).combinations()
    assert table.rows == () and table.refused == ()


def test_each_save_adds_a_generation_and_keeps_the_old_ones(make_model, tmp_path: Path) -> None:
    path = tmp_path / "s.user.h5"
    write_user_file(path, make_model(), combinations=[LINEAR])
    write_user_file(path, make_model(), combinations=[CQC])
    uf = UserFile(path)
    assert uf.generations() == (0, 1)
    assert list(uf.combinations().rows) == [CQC]
    with h5py.File(path, "r") as f:
        assert f["combinations"].attrs["current"] == 1
        assert f["combinations/g0/name"].asstr()[...].tolist() == [LINEAR.name]


def test_a_write_without_combinations_leaves_the_generations(make_model, tmp_path: Path) -> None:
    path = tmp_path / "s.user.h5"
    write_user_file(path, make_model(), combinations=[LINEAR])
    write_user_file(path, make_model(), results=[nodal()])
    assert UserFile(path).generations() == (0,)


# ---- /results: every target, bit-equal --------------------------------------


def test_nodal_field_reads_back_bit_equal(make_model, tmp_path: Path) -> None:
    r = nodal()
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[r])
    back = UserFile(path).read_result(r.name)
    assert _bits(back.data) == _bits(r.data), "NaN cells included"
    assert back.ids.tolist() == [1, 2, 3]
    assert back.states == r.states and back.components == r.components and back.units == r.units
    assert back.linear is False and back.basis == "model" and back.target == "nodes"


def test_nodal_layout_is_a_small_stage(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal(linear=True)])
    with h5py.File(path, "r") as f:
        r = f["results/r0"]
        assert r.attrs["KIND"] == "static" and r.attrs["per_state"] == 1 and r.attrs["linear"] == 1
        assert r.attrs["per_state"].dtype == np.int8
        values = r["RESULTS/ON_NODES/values"]
        assert values["ID"].shape == (3, 1) and values["ID"].dtype == np.dtype("i4")
        assert values["DATA"].shape == (2, 3, 2) and values.attrs["COMPONENTS"] == "wx,wy"
        assert values["STEP"][...].tolist() == [0, 1] and values["TIME"][...].tolist() == [0.0, 1.0]
        assert r["STATES/kind"].asstr()[...].tolist() == ["stage", "combination"]


def test_linear_is_absent_when_it_is_not_declared(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal()])
    with h5py.File(path, "r") as f:
        assert "linear" not in f["results/r0"].attrs


def test_state_free_domain_value_round_trips(make_model, ladruno, tmp_path: Path) -> None:
    r = UserResult(name="base shear", target="domain", basis="results",
                   components=("Vx", "Vy"), data=np.array([[[12.5, -3.25]]]))
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results_file=ladruno, results=[r])
    uf = UserFile(path)
    entry = uf.results().by_name(r.name)
    assert entry.per_state is False and entry.target == "domain"
    back = uf.read_result(r.name)
    assert back.states is None and _bits(back.data) == _bits(r.data)
    with h5py.File(path, "r") as f:
        assert "STATES" not in f["results/r0"] and "ID" not in f["results/r0/RESULTS/ON_DOMAIN/values"]


def test_entity_field_round_trips(make_model, tmp_path: Path) -> None:
    r = UserResult(
        name="storey drift", target="entities", basis="model",
        components=("dx",), data=np.array([[[0.001], [np.nan]]]),
        entity_kind=("storey", "point_set"), entity_row=np.array([0, 4], dtype="i4"),
    )
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[r])
    back = UserFile(path).read_result(r.name)
    assert back.entity_kind == ("storey", "point_set") and back.entity_row.tolist() == [0, 4]
    assert _bits(back.data) == _bits(r.data)


def _bucket(**kw) -> ElementBucket:
    data = np.arange(12, dtype="f8").reshape(1, 2, 6) / 7.0
    fields = dict(
        name="204-ASDShellT3[102:0:0]", ids=np.array([11, 12], dtype="i4"), data=data,
        components=(("mxx", "myy", "mxy"),), gauss_ids=(0,), multiplicity=(2,),
        topology="tri", param_domain="bary", family="lagrange", order=(1, 1),
        gp_param=np.array([[0.2, 0.3], [0.1, 0.6]]),
    )
    fields.update(kw)
    return ElementBucket(**fields)


def _elements(bucket: ElementBucket) -> UserResult:
    return UserResult(name="wood_armer", target="elements", basis="model", buckets=(bucket,))


@pytest.mark.parametrize(
    "family, expected",
    [
        # Closed form: node-0-first (xi, eta) has node weights (xi, eta, 1-xi-eta),
        # and gmsh (u, v) is the weights of nodes 1 and 2.
        ("lagrange", [[0.3, 0.5], [0.6, 0.3]]),
        # A bernstein triangle's rows are gmsh already.
        ("bernstein", [[0.2, 0.3], [0.1, 0.6]]),
    ],
)
def test_the_recorder_read_path_places_the_gauss_points(make_model, tmp_path, family, expected) -> None:
    b = _bucket(family=family)
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[_elements(b)])
    with h5py.File(path, "r") as f:
        got = _gp_param_for(f["results/r0/MODEL/ELEMENTS"], b.name)
    assert np.allclose(got, expected, atol=1e-15)


def test_a_cartesian_rule_reads_through_the_same_path(make_model, tmp_path) -> None:
    gp = np.array([[-0.5, 0.5]])
    b = _bucket(topology="quad", param_domain="[-1,1]", order=(1, 1), gp_param=gp)
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[_elements(b)])
    with h5py.File(path, "r") as f:
        assert _gp_param_for(f["results/r0/MODEL/ELEMENTS"], b.name).tolist() == gp.tolist()
    assert UserFile(path).read_result("wood_armer").buckets[0].order == (1, 1)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"family": ""}, "family"),
        ({"family": "nurbs"}, "no known barycentric frame"),
        ({"order": ()}, "order"),
        ({"order": (1,)}, "ORDER"),
        ({"order": (0, 1)}, "ORDER"),
        ({"param_domain": "polar"}, "unknown PARAM_DOMAIN"),
        ({"gp_param": np.array([[0.5, 0.5, 0.0]]), "order": (1, 1, 1)}, "free"),
    ],
)
def test_the_writer_refuses_what_the_read_path_could_not_place(make_model, tmp_path, kw, match) -> None:
    target = tmp_path / "s.user.h5"
    with pytest.raises(UserWriteRefused, match=match):
        write_user_file(target, make_model(), results=[_elements(_bucket(**kw))])
    assert not target.exists()


@pytest.mark.parametrize(
    "damage",
    [
        lambda g: g.attrs.__delitem__("FAMILY"),
        lambda g: g.attrs.__delitem__("ORDER"),
        lambda g: g.attrs.__setitem__("ORDER", "two"),
        lambda g: g.attrs.__setitem__("ORDER", np.array([1], dtype="i4")),
        lambda g: g.attrs.__setitem__("FAMILY", "nurbs"),
        lambda g: g.attrs.__setitem__("NUM_GP", "two"),
        lambda g: g.attrs.__setitem__("NUM_GP", 2.5),
    ],
)
def test_a_placement_the_read_path_cannot_use_refuses_the_entry(make_model, tmp_path, damage) -> None:
    b = _bucket()
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[_elements(b), nodal("keep")])
    with h5py.File(path, "a") as f:
        damage(f[f"results/r0/MODEL/ELEMENTS/{b.name}"])
    index = UserFile(path).results()
    assert [e.name for e in index.entries] == ["keep"]
    assert [(r.name, r.reason) for r in index.refused] == [("wood_armer", "bad-param")]


def test_element_field_round_trips_and_the_recorder_reader_parses_it(make_model, tmp_path) -> None:
    b = _bucket()
    r = UserResult(name="wood_armer", target="elements", basis="model",
                   element_index_space="fem", buckets=(b,))
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[r])
    back = UserFile(path).read_result(r.name)
    got = back.buckets[0]
    assert back.element_index_space == "fem"
    assert _bits(got.data) == _bits(b.data) and _bits(got.gp_param) == _bits(b.gp_param)
    assert got.ids.tolist() == [11, 12] and got.components == b.components
    assert (got.gauss_ids, got.multiplicity, got.topology) == ((0,), (2,), "tri")
    assert (got.family, got.order, got.param_domain) == ("lagrange", (1, 1), "bary")
    with h5py.File(path, "r") as f:
        stored = f[f"results/r0/RESULTS/ON_ELEMENTS/values/{b.name}"]
        blocks = parse_blocks(stored)  # the .ladruno read path, unmodified
        assert [x.comp_names for x in blocks] == [("mxx", "myy", "mxy")]
        assert blocks[0].width == 6 and stored.attrs["NUM_COLUMNS"] == 6
        placed = f[f"results/r0/MODEL/ELEMENTS/{b.name}"]
        assert placed.attrs["NUM_GP"] == 2 and placed["QUADRATURE/GP_PARAM"].shape == (2, 2)
        assert placed.attrs["FAMILY"] == "lagrange"
        assert placed.attrs["ORDER"].tolist() == [1, 1] and placed.attrs["ORDER"].dtype == np.int32


def test_no_int64_in_the_file(make_model, ladruno, tmp_path: Path) -> None:
    """The integer policy: i1, i4 and f8 only (app, BigInt)."""
    r = UserResult(name="w", target="elements", basis="model", buckets=(_bucket(),))
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results_file=ladruno,
                           results=[r, nodal(ids=np.array([1, 2, 3], dtype="i8"))],
                           combinations=ROWS)
    seen: list[str] = []

    def visit(name: str, obj) -> None:
        for key, value in obj.attrs.items():
            a = np.asarray(value)
            if a.dtype.kind in "iu":
                assert a.dtype.itemsize < 8, f"{name}@{key} is {a.dtype}"
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "iu":
            seen.append(name)
            assert obj.dtype.itemsize <= 4, f"{name} is {obj.dtype}"

    with h5py.File(path, "r") as f:
        f.visititems(visit)
        visit("meta", f["meta"])
    assert seen, "the sweep looked at no integer dataset: vacuous"


def test_an_id_beyond_i4_refuses(make_model, tmp_path: Path) -> None:
    with pytest.raises(UserWriteRefused, match="i4"):
        write_user_file(tmp_path / "s.user.h5", make_model(),
                        results=[nodal(ids=np.array([1, 2, 2 ** 33], dtype="i8"))])
    assert not (tmp_path / "s.user.h5").exists()


def test_a_shape_that_disagrees_with_the_states_refuses(make_model, tmp_path: Path) -> None:
    with pytest.raises(UserWriteRefused, match="shape"):
        write_user_file(tmp_path / "s.user.h5", make_model(),
                        results=[nodal(states=(StateAddress("stage", "only"),))])


def test_same_name_replaces_its_entry_and_keeps_the_others(make_model, tmp_path: Path) -> None:
    path = tmp_path / "s.user.h5"
    write_user_file(path, make_model(), results=[nodal("a"), nodal("b")])
    newer = nodal("a", data=np.zeros((2, 3, 2)))
    write_user_file(path, make_model(), results=[newer])
    uf = UserFile(path)
    assert sorted(e.name for e in uf.results().entries) == ["a", "b"]
    assert np.all(uf.read_result("a").data == 0.0)
    assert np.isnan(uf.read_result("b").data[0, 1, 0])
    with h5py.File(path, "r") as f:
        assert sorted(f["results"]) == ["r0", "r1"]


# ---- names -----------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", " lead", "trail ", "a\tb", "a\x00b", "é"])
def test_writer_refuses_a_bad_name(make_model, tmp_path: Path, bad: str) -> None:
    target = tmp_path / "s.user.h5"
    with pytest.raises(UserWriteRefused):
        write_user_file(target, make_model(), results=[nodal(bad)])
    with pytest.raises(UserWriteRefused):
        write_user_file(target, make_model(), combinations=[Combination(bad, "linear", (Term("stage", "D"),))])
    assert not target.exists()


@pytest.mark.parametrize("name", ["Delta_after (L/480)", "100%/L", "..", "é", "a/b/c", "Ω 1"])
def test_a_real_name_round_trips_as_data_not_as_a_link(make_model, tmp_path: Path, name: str) -> None:
    path = write_user_file(
        tmp_path / "s.user.h5", make_model(), results=[nodal(name)],
        combinations=[Combination(name, "linear", (Term("stage", "D"),))],
    )
    uf = UserFile(path)
    assert uf.results().entries[0].name == name
    assert uf.combinations().rows[0].name == name
    with h5py.File(path, "r") as f:
        assert list(f["results"]) == ["r0"], "the name is not a link"
        assert f["results/r0"].attrs["name"] == name


# ---- versions --------------------------------------------------------------


def _stamp(path: Path, key: str, value: str) -> None:
    with h5py.File(path, "a") as f:
        f["meta"].attrs[key] = value


def test_a_newer_meta_refuses_the_file(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model())
    _stamp(path, USER_META_KEY, "1.9.0")
    with pytest.raises(SchemaVersionError, match="newer"):
        UserFile(path)


def test_a_newer_major_on_a_zone_refuses_that_zone_only(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(),
                           combinations=[LINEAR], results=[nodal()])
    _stamp(path, COMBINATIONS_KEY, "2.0.0")
    uf = UserFile(path)
    with pytest.raises(SchemaVersionError, match="different major"):
        uf.combinations()
    assert uf.results().entries[0].name == nodal().name


def test_a_file_without_the_key_is_not_a_user_file(tmp_path: Path) -> None:
    path = tmp_path / "s.user.h5"
    with h5py.File(path, "w") as f:
        f.create_group("meta")
    with pytest.raises(UserFileRefused, match="not a .user.h5"):
        UserFile(path)


def test_a_writer_refuses_to_replace_a_foreign_file(make_model, tmp_path: Path) -> None:
    path = tmp_path / "s.user.h5"
    with h5py.File(path, "w") as f:
        f.create_group("nodes")
    before = path.read_bytes()
    with pytest.raises(UserFileRefused):
        write_user_file(path, make_model())
    assert path.read_bytes() == before


def test_a_writer_will_not_touch_a_zone_newer_than_it_knows(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), combinations=[LINEAR], results=[nodal()])
    newer = f"{reader_version(COMBINATIONS).major}.{reader_version(COMBINATIONS).minor + 5}.0"
    _stamp(path, COMBINATIONS_KEY, newer)
    before = path.read_bytes()
    with pytest.raises(SchemaVersionError, match="newer"):
        write_user_file(path, make_model(), combinations=[CQC])
    assert path.read_bytes() == before
    write_user_file(path, make_model(), results=[nodal("other")])  # a zone it does not write
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs[COMBINATIONS_KEY] == newer, "copied verbatim, key included"


def test_a_combinations_only_write_stamps_no_results_zone(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), combinations=[LINEAR])
    write_user_file(path, make_model(), combinations=[CQC])
    with h5py.File(path, "r") as f:
        assert "results" not in f and USER_RESULTS_KEY not in f["meta"].attrs
    assert UserFile(path).results().entries == ()


def test_a_write_keeps_the_version_of_a_results_zone_it_did_not_write(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model(), results=[nodal()])
    older = str(reader_version(USER_RESULTS))
    with h5py.File(path, "a") as f:
        f["meta"].attrs[USER_RESULTS_KEY] = older + "-kept"
    write_user_file(path, make_model(), combinations=[LINEAR])
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs[USER_RESULTS_KEY] == older + "-kept"


def test_a_path_not_named_user_h5_refuses(make_model, tmp_path: Path) -> None:
    with pytest.raises(UserWriteRefused, match="user.h5"):
        write_user_file(tmp_path / "model.h5", make_model())


def test_zones_it_does_not_write_are_copied_verbatim(make_model, tmp_path: Path) -> None:
    stamp = "an unregistered key's value, carried as it is"
    path = write_user_file(tmp_path / "s.user.h5", make_model())
    with h5py.File(path, "a") as f:
        modes = f.create_group("modes")
        modes.attrs["components"] = "MX,MY"
        modes.create_dataset("lambda", data=np.array([1.5, 2.5]))
        f.create_group("future_zone").create_dataset("x", data=np.array([9], dtype="i4"))
        f["meta"].attrs["modes_schema_version"] = stamp
        f["meta"].attrs["unknown_attr"] = "kept"
    write_user_file(path, make_model(), results=[nodal()])
    with h5py.File(path, "r") as f:
        assert f["modes/lambda"][...].tolist() == [1.5, 2.5]
        assert f["modes"].attrs["components"] == "MX,MY"
        assert f["future_zone/x"][0] == 9
        assert f["meta"].attrs["modes_schema_version"] == stamp
        assert f["meta"].attrs["unknown_attr"] == "kept"


def test_a_reader_holds_no_handle_so_a_write_replaces_at_once(make_model, tmp_path: Path) -> None:
    path = write_user_file(tmp_path / "s.user.h5", make_model())
    uf = UserFile(path)
    write_user_file(path, make_model(), results=[nodal()])
    assert uf.results().entries[0].name == nodal().name
