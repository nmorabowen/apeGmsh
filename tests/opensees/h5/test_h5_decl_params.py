"""``/opensees/decl_params``: every declaration's parameters by field name
(ADR 0114 A6/Q4, program slice K1-7 #1464, opensees schema 2.26.0).

Oracles, each naming the right answer:

* **Every declaration's parameters read back by field name.** For every
  registered primitive of a model, the row the reader returns for its
  declaration has the primitive's class as ``type`` and exactly
  ``dataclasses.fields(prim)`` as ``params``, value for value: a scalar
  as is, a tuple as a tuple, a referenced primitive as the
  :class:`DeclRef` of its declaration key, a value dataclass as a
  :class:`DeclStruct`, an orientation as a :class:`DeclOpaque`.
* **References resolve to declaration keys.** A ``DeclRef``'s key is a
  row of ``/opensees/decls`` (``by_key`` finds it), the one the
  referenced primitive's own ``(kind, tag)`` joins to; ``transf_ref``,
  ``integration_ref`` and ``section_ref`` carry the same keys.
* **An unknown shape refuses at write.** A field holding an ``ndarray``,
  a set, a non-finite float, an int-keyed mapping or an unlisted object
  raises :class:`H5DeclParamsError` from ``set_decl_params``; nothing is
  skipped.
* **The encoder on real primitives.** Every contract-sampled primitive
  (the ``ALL_*`` rosters) encodes without refusal to its field names.
* **A derived view, hash-excluded.** ``model_hash`` is the same with and
  without the group, the group is in ``MODEL_HASH_EXCLUDED_CHILDREN``,
  and the decks are untouched. A rewrite echoes the group verbatim.
* **``params_names`` where the argv equals the fields.** ``Steel01``
  names ``fy E b``; ``Parallel`` without factors names its materials'
  slots; with ``-factors`` (a flag without a field) it stays unnamed, as
  does every element and transform.

The group is written by ``BuiltModel.emit``'s hook on every ``apeSees.h5``
emit; nothing here feeds the archive by hand.
"""
from __future__ import annotations

import dataclasses
import enum
import fractions
import json
import numbers
import shutil
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.lineage import (
    MODEL_HASH_EXCLUDED_CHILDREN,
    compute_model_hash,
)
from apeGmsh.opensees._internal.types import Primitive, UniaxialMaterial
from apeGmsh.opensees._orientation import Cartesian
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import (
    H5DeclParamsError,
    H5Emitter,
    decl_argv_names,
    encode_decl_params,
)
from apeGmsh.opensees.emitter.h5_reader import (
    DeclOpaque,
    DeclRef,
    DeclStruct,
    MalformedH5Error,
)
from tests.opensees.golden.builder import build_model
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem
from tests.opensees.h5.test_h5_decls import _flat_frame, _staged


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _params_frame() -> apeSees:
    """One column on a real FEMData with a force-based element, so a
    transform, an integration rule and a section are referenced, plus a
    Steel01 and two Parallel materials for ``params_names``."""
    ops = apeSees(cast("Any", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    steel = ops.uniaxialMaterial.Steel01(fy=420e6, E=200e9, b=0.01)
    visc = ops.uniaxialMaterial.Viscous(C=1.0e5)
    ops.uniaxialMaterial.Parallel(materials=(steel, visc))
    ops.uniaxialMaterial.Parallel(materials=(steel, visc), factors=(2.0, 3.0))
    sec = ops.section.Elastic(E=200e9, A=0.01, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.fix(nodes=[1], dofs=(1,) * 6)
    return ops


def _table(path: Path) -> h5_reader.DeclarationTable:
    with h5_reader.open(str(path)) as m:
        table = m.declarations()
    assert table is not None
    return table


def _stored_model_hash(path: Path) -> str:
    with h5py.File(str(path), "r") as f:
        return str(f["meta"]["lineage"].attrs["model_hash"])


def _plain(value: Any) -> Any:
    """A read-back value with its mapping proxies and structs as plain
    dicts, for equality against :func:`_expect`."""
    if isinstance(value, DeclStruct):
        return DeclStruct(type=value.type, fields=_plain(value.fields))
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    return value


def _expect(value: Any, key_of: Callable[[object], str]) -> Any:
    """The read-back form of a primitive field value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Primitive):
        return DeclRef(key=key_of(value))
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(_expect(v, key_of) for v in value)
    if isinstance(value, Mapping):
        return {k: _expect(v, key_of) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return DeclStruct(
            type=type(value).__name__,
            fields={f.name: _expect(getattr(value, f.name), key_of)
                    for f in dataclasses.fields(value)})
    return DeclOpaque(type=type(value).__name__)


# ---------------------------------------------------------------------------
# Every declaration's parameters read back by field name
# ---------------------------------------------------------------------------


def _assert_params_read_back(ops: apeSees, path: Path) -> None:
    table = _table(path)
    bm = ops.build()
    _rows, index = bm._declaration_rows()

    def key_of(prim: object) -> str:
        return bm._decls[id(prim)][0]

    # Every declaration has exactly one row; the primitives are among them.
    assert set(table.params) == set(range(len(table.decls)))
    assert {index[id(p)] for p in bm.primitives} <= set(table.params)
    for prim in bm.primitives:
        ro = table.params[index[id(prim)]]
        assert ro.type == type(prim).__name__
        assert list(ro.params) == [f.name for f in dataclasses.fields(prim)]
        for f in dataclasses.fields(prim):
            got = _plain(ro.params[f.name])
            assert got == _expect(getattr(prim, f.name), key_of), (
                type(prim).__name__, f.name)
        # The row's key is the primitive's own declaration.
        assert table.decls[index[id(prim)]].key == key_of(prim)
        # The stored text is the encoder's, byte for byte.
        assert json.loads(ro.params_json) == json.loads(json.dumps(
            encode_decl_params(prim, key_of)))


@pytest.mark.parametrize("build", [_staged, _flat_frame, _params_frame])
def test_every_declaration_reads_back_by_field_name(
    tmp_path: Path, build: Callable[..., apeSees],
) -> None:
    ops = build(named=False) if build is not _params_frame else build()
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    _assert_params_read_back(ops, p)


@pytest.mark.parametrize("mode", [
    "flat", "partitioned", "staged", "staged_partitioned",
])
def test_every_declaration_reads_back_golden(tmp_path: Path, mode: str) -> None:
    ops = build_model("two_column_frame_partitioned", mode, "recording")
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    _assert_params_read_back(ops, p)


def test_orientation_reads_back_opaque(tmp_path: Path) -> None:
    ops = build_model("arch_with_orientation_fan_out", "flat", "tcl")
    p = tmp_path / "arch.h5"
    ops.h5(str(p))
    table = _table(p)
    opaque = [
        v for ro in table.params.values() for v in ro.params.values()
        if isinstance(v, DeclOpaque)]
    assert opaque and {o.type for o in opaque} <= {
        "Cartesian", "Cylindrical", "Spherical", "AlongBeam"}
    _assert_params_read_back(ops, p)


# ---------------------------------------------------------------------------
# References resolve to declaration keys
# ---------------------------------------------------------------------------


def test_refs_resolve_to_declaration_keys(tmp_path: Path) -> None:
    ops = _params_frame()
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    table = _table(p)
    bm = ops.build()
    by_type = {type(prim).__name__: prim for prim in bm.primitives}
    rows = {name: table.params_for(bm._decls[id(prim)][0])
            for name, prim in by_type.items()}

    def key_of_tag(kind: str, prim: object) -> str:
        return table.for_tag(kind, bm.tag_for[id(prim)]).key

    # The element references its transform and its integration rule; the
    # rule references its section; the Parallel its two materials.
    el = rows["forceBeamColumn"]
    transf_key = key_of_tag("geomTransf", by_type["Linear"])
    integ_key = key_of_tag("beamIntegration", by_type["Lobatto"])
    sec_key = key_of_tag("section", by_type["ElasticSection"])
    assert el.params["transf"] == DeclRef(key=transf_key)
    assert el.params["integration"] == DeclRef(key=integ_key)
    assert (el.transf_ref, el.integration_ref, el.section_ref) == (
        (transf_key,), (integ_key,), ())
    lob = rows["Lobatto"]
    assert lob.params["section"] == DeclRef(key=sec_key)
    assert (lob.transf_ref, lob.integration_ref, lob.section_ref) == (
        (), (), (sec_key,))
    parallels = [ro for ro in table.params.values() if ro.type == "Parallel"]
    steel_key = key_of_tag("uniaxialMaterial", by_type["Steel01"])
    visc_key = key_of_tag("uniaxialMaterial", by_type["Viscous"])
    for ro in parallels:
        assert ro.params["materials"] == (
            DeclRef(key=steel_key), DeclRef(key=visc_key))
    # Every reference is a row of /opensees/decls.
    for ro in table.params.values():
        for v in ro.params.values():
            for ref in (v if isinstance(v, tuple) else (v,)):
                if isinstance(ref, DeclRef):
                    assert table.by_key(ref.key).family in {
                        "uniaxialMaterial", "section", "geomTransf",
                        "beamIntegration"}
    # A fix has a row of its own fields; an unknown key raises.
    assert set(table.params_for("opensees/fix/#1").params) == {"pg", "nodes", "dofs"}
    with pytest.raises(KeyError):
        table.params_for("opensees/uniaxialMaterial/nope")


# ---------------------------------------------------------------------------
# An unknown shape refuses at write
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class _Odd(UniaxialMaterial):
    """A primitive whose one field holds whatever the test puts there."""

    weird: Any

    def _emit(self, emitter: Any, tag: int) -> None:  # pragma: no cover
        emitter.uniaxialMaterial("Odd", tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@pytest.mark.parametrize("weird", [
    np.zeros(3),
    {1, 2},
    float("nan"),
    float("inf"),
    {1: 2.0},
    {"$decl": 1.0},
    object(),
    (1.0, {2, 3}),
], ids=["ndarray", "set", "nan", "inf", "int-keyed", "dollar-key", "object",
        "nested-set"])
def test_unknown_shape_refuses(weird: Any) -> None:
    prim = _Odd(weird=weird)
    with pytest.raises(H5DeclParamsError, match="_Odd.weird"):
        encode_decl_params(prim, lambda p: "k")
    emitter = H5Emitter(model_name="m", snapshot_id="")
    emitter.set_declarations([("opensees/uniaxialMaterial/#1",
                               "uniaxialMaterial", "", False)], [])
    with pytest.raises(H5DeclParamsError):
        emitter.set_decl_params([(0, prim)], lambda p: "k")


def test_unregistered_reference_refuses() -> None:
    other = _Odd(weird=1.0)
    prim = _Odd(weird=other)

    def key_of(p: object) -> str:
        raise KeyError(p)

    with pytest.raises(H5DeclParamsError, match="never registered"):
        encode_decl_params(prim, key_of)


def test_listed_opaque_and_value_shapes_encode() -> None:
    prim = _Odd(weird=(Cartesian(), {"a": (1, 2.5, "s", None, True)}, np.int64(3)))
    assert encode_decl_params(prim, lambda p: "k") == {"weird": [
        {"$opaque": "Cartesian"}, {"a": [1, 2.5, "s", None, True]}, 3]}


def test_set_decl_params_refuses_before_declarations_and_twice() -> None:
    emitter = H5Emitter(model_name="m", snapshot_id="")
    with pytest.raises(RuntimeError, match="set_declarations first"):
        emitter.set_decl_params([], lambda p: "k")
    emitter.set_declarations(
        [("opensees/uniaxialMaterial/#1", "uniaxialMaterial", "", False)], [])
    with pytest.raises(IndexError):
        emitter.set_decl_params([(1, _Odd(weird=1.0))], lambda p: "k")
    with pytest.raises(ValueError, match="two primitives"):
        emitter.set_decl_params(
            [(0, _Odd(weird=1.0)), (0, _Odd(weird=2.0))], lambda p: "k")
    emitter.set_decl_params([(0, _Odd(weird=1.0))], lambda p: "k")
    with pytest.raises(RuntimeError, match="already set"):
        emitter.set_decl_params([(0, _Odd(weird=1.0))], lambda p: "k")


def test_write_refuses_a_reference_the_declarations_lack(tmp_path: Path) -> None:
    emitter = H5Emitter(model_name="m", snapshot_id="")
    emitter.set_declarations(
        [("opensees/uniaxialMaterial/#1", "uniaxialMaterial", "", False)], [])
    emitter.set_decl_params(
        [(0, _Odd(weird=(_Odd(weird=1.0),)))], lambda p: "opensees/uniaxialMaterial/ghost")
    with pytest.raises(RuntimeError, match="does not declare"):
        emitter.write(str(tmp_path / "m.h5"))


# ---------------------------------------------------------------------------
# The encoder on real primitives: every contract-sampled class
# ---------------------------------------------------------------------------


def _contract_samples() -> list[Primitive]:
    from tests.opensees.contract import (
        test_analysis_contract as an,
        test_element_beam_column_contract as bc,
        test_element_shell_contract as sh,
        test_element_solid_contract as so,
        test_element_truss_contract as tr,
        test_element_zero_length_contract as zl,
        test_nd_material_contract as nd,
        test_pattern_contract as pa,
        test_recorder_contract as re_,
        test_section_contract as se,
        test_time_series_contract as ts,
        test_uniaxial_material_contract as un,
    )

    out: list[Primitive] = []
    out += [un._minimal(c) for c in un.ALL_UNIAXIAL]
    out += [nd._instantiate(c) for c in nd.ALL_ND]
    out += [se._make_minimal(c) for c in se.ALL_SECTIONS]
    out += [ts._minimal_instance(c) for c in ts.ALL_TIME_SERIES]
    out += [bc._minimal(c) for c in bc.ALL_BEAM_COLUMN_ELEMENTS]
    out += [tr._minimal(c) for c in tr.ALL_TRUSS_ELEMENTS]
    out += [zl._minimal(c) for c in zl.ALL_ZERO_LENGTH_ELEMENTS]
    out += [sh._make_minimal(c) for c in sh.ALL_SHELL_ELEMENTS]
    out += [so._make_minimal(c) for c in so.ALL_SOLID_ELEMENTS]
    out += [pa._minimal_instance(c) for c in pa.ALL_PATTERNS]
    out += [re_._minimal_instance(c) for c in re_.ALL_RECORDERS]
    out += [an._minimal(c) for c in an.ALL_ANALYSIS_COMPONENTS]
    return out


@pytest.mark.parametrize(
    "prim", _contract_samples(), ids=lambda p: type(p).__name__)
def test_encoder_on_real_primitives(prim: Primitive) -> None:
    def key_of(p: object) -> str:
        return f"opensees/x/{type(p).__name__}"

    encoded = encode_decl_params(prim, key_of)
    assert list(encoded) == [f.name for f in dataclasses.fields(prim)]
    text = json.dumps(encoded, allow_nan=False, separators=(",", ":"))
    assert json.loads(text) == encoded
    for f in dataclasses.fields(prim):
        value = getattr(prim, f.name)
        if isinstance(value, Primitive):
            assert encoded[f.name] == {"$decl": key_of(value)}


# ---------------------------------------------------------------------------
# A derived view, hash-excluded; a rewrite echoes it
# ---------------------------------------------------------------------------


def test_the_hook_writes_the_group_and_model_hash_excludes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BuiltModel.emit`` hands every primitive to the archive; the
    group is a derived view, so the hash with it deleted is the stamped
    one, and a build whose hook is silenced stamps the same hash."""
    with_ = tmp_path / "with.h5"
    _params_frame().h5(str(with_))
    with h5py.File(str(with_), "r") as f:
        assert "decl_params" in f["opensees"]
        assert len(f["opensees"]["decl_params"]["decl"]) == len(
            _table(with_).decls)
    assert "decl_params" in MODEL_HASH_EXCLUDED_CHILDREN
    stripped = tmp_path / "stripped.h5"
    shutil.copy(with_, stripped)
    with h5py.File(str(stripped), "a") as f:
        fem_hash = str(f["meta"]["lineage"].attrs["fem_hash"])
        del f["opensees"]["decl_params"]
        assert compute_model_hash(fem_hash, f["opensees"]) == (
            _stored_model_hash(with_))
    # The same model with the hook silenced: no group, the same hash.
    monkeypatch.setattr(
        H5Emitter, "set_decl_params", lambda self, items, key_of: None)
    without = tmp_path / "without.h5"
    _params_frame().h5(str(without))
    with h5py.File(str(without), "r") as f:
        assert "decl_params" not in f["opensees"]
    assert _stored_model_hash(with_) == _stored_model_hash(without)


def test_rewrite_echoes_decl_params(tmp_path: Path) -> None:
    p = tmp_path / "src.h5"
    q = tmp_path / "out.h5"
    _params_frame().h5(str(p))
    om = OpenSeesModel.from_h5(str(p))
    assert om.declarations is not None and om.declarations.params
    om.to_h5(str(q))
    src, out = _table(p), _table(q)
    assert out.decls == src.decls and out.tags == src.tags
    assert dict(out.params) == dict(src.params)
    assert [r.params_json for r in out.params.values()] == [
        r.params_json for r in src.params.values()]
    assert _stored_model_hash(q) == _stored_model_hash(p)


def test_a_source_without_the_group_rewrites_without_it(tmp_path: Path) -> None:
    p = tmp_path / "src.h5"
    q = tmp_path / "out.h5"
    _params_frame().h5(str(p))
    with h5py.File(str(p), "a") as f:
        del f["opensees"]["decl_params"]
    OpenSeesModel.from_h5(str(p)).to_h5(str(q))
    assert not _table(q).params
    with h5py.File(str(q), "r") as f:
        assert "decl_params" not in f["opensees"]


# ---------------------------------------------------------------------------
# params_names where the argv equals the fields (Q4)
# ---------------------------------------------------------------------------


def test_params_names_where_argv_equals_fields(tmp_path: Path) -> None:
    ops = _params_frame()
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    table = _table(p)
    by_type: dict[str, list[tuple[str, ...] | None]] = {}
    for ro in table.params.values():
        by_type.setdefault(ro.type, []).append(ro.params_names)
    assert by_type["Steel01"] == [("fy", "E", "b")]
    assert by_type["Lobatto"] == [("section", "n_ip")]
    assert by_type["ElasticSection"] == [("E", "A", "Iz", "Iy", "G", "J")]
    # Without factors the two material slots are the fields; with
    # ``-factors`` the argv carries a flag no field names: unnamed.
    assert by_type["Parallel"] == [("materials[0]", "materials[1]"), None]
    # An element row carries node tags, a transform a vecxz the store
    # keeps structured: never named.
    assert by_type["forceBeamColumn"] == [None]
    assert by_type["Linear"] == [None]


def test_decl_argv_names_rules() -> None:
    tag_of = {"k/a": 7, "k/b": None}.get
    params = {"x": 1.0, "m": {"$decl": "k/a"}, "pts": [[0.1, 0.2], [0.3, 0.4]],
              "opt": None}
    assert decl_argv_names(params, (1.0, 7, 0.1, 0.2, 0.3, 0.4), tag_of) == (
        "x", "m", "pts[0][0]", "pts[0][1]", "pts[1][0]", "pts[1][1]")
    # An int slot equals its float argv; a value or count mismatch, a
    # flag, a bool, a struct, a mapping and an unresolved reference are
    # not the fields.
    assert decl_argv_names({"n": 3}, (3.0,), tag_of) == ("n",)
    assert decl_argv_names({"n": 3}, (4,), tag_of) is None
    assert decl_argv_names({"n": 3}, (3, "-flag"), tag_of) is None
    assert decl_argv_names({"b": True}, (1,), tag_of) is None
    assert decl_argv_names({"s": {"$struct": "S", "fields": {}}}, (), tag_of) is None
    assert decl_argv_names({"d": {"a": 1}}, (1,), tag_of) is None
    assert decl_argv_names({"m": {"$decl": "k/b"}}, (1,), tag_of) is None
    assert decl_argv_names({}, (), tag_of) == ()


# ---------------------------------------------------------------------------
# The reader refuses a malformed group
# ---------------------------------------------------------------------------


def _tamper(path: Path, column: str, values: list[Any]) -> None:
    with h5py.File(str(path), "a") as f:
        g = f["opensees"]["decl_params"]
        dt = g[column].dtype
        del g[column]
        g.create_dataset(column, data=values, dtype=dt)


@pytest.mark.parametrize("column, value, message", [
    ("decl", 10**6, "points at declaration"),
    ("params", "[1, 2]", "not an object"),
    ("params", "{", "not JSON"),
    ("params", '{"m": {"$decl": "opensees/uniaxialMaterial/ghost"}}',
     "does not declare"),
    ("params", '{"m": {"$weird": 1}}', "unknown tag"),
    ("params_names", "{}", "not a JSON list"),
])
def test_reader_refuses_malformed_rows(
    tmp_path: Path, column: str, value: Any, message: str,
) -> None:
    p = tmp_path / "m.h5"
    _params_frame().h5(str(p))
    with h5py.File(str(p), "r") as f:
        n = len(f["opensees"]["decl_params"]["decl"])
        first = f["opensees"]["decl_params"][column][()][0]
    values = [value] + [first] * (n - 1)
    _tamper(p, column, values)
    with pytest.raises(MalformedH5Error, match=message):
        _table(p)


def test_reader_refuses_a_dangling_ref_run(tmp_path: Path) -> None:
    p = tmp_path / "m.h5"
    _params_frame().h5(str(p))
    with h5py.File(str(p), "a") as f:
        run = f["opensees"]["decl_params"]["transf_ref"]
        n_keys = len(run["key"])
        assert n_keys == 1
        dt = run["key"].dtype
        del run["key"]
        run.create_dataset("key", data=["opensees/geomTransf/ghost"], dtype=dt)
    with pytest.raises(MalformedH5Error, match="does not declare"):
        _table(p)
    _params_frame().h5(str(p))
    with h5py.File(str(p), "a") as f:
        run = f["opensees"]["decl_params"]["section_ref"]
        n = len(run["count"])
        del run["count"]
        run.create_dataset("count", data=np.full(n, 5, dtype=np.int64))
    with pytest.raises(MalformedH5Error, match="past its"):
        _table(p)


# ---------------------------------------------------------------------------
# Every key of /opensees/decls has exactly one row
# ---------------------------------------------------------------------------


def _broad_staged() -> apeSees:
    """The staged two-quad model plus global and region-scoped Rayleigh,
    a damping object attached ``on=`` a group, modal damping, a named
    region and a staged support (an equation constraint needs a Lagrange
    handler on the stage; the frame carries one)."""
    ops = _staged(named=True)
    ops.damping.rayleigh(alpha_m=0.1, beta_k=0.01)
    ops.damping.rayleigh(alpha_m=0.2, beta_k=0.02, on="Fill")
    ops.damping.uniform(ratio=0.05, freq_lower=1.0, freq_upper=10.0, on="Rock")
    ops.damping.modal((0.02, 0.03), modes=2)
    ops.region(name="rock_region", pg="Rock")
    return ops


def _broad_frame() -> apeSees:
    """The force-based frame plus an initial stress, a region and an
    equation constraint."""
    ops = _params_frame()
    ops.initial_stress(
        name="insitu", pg="Cols", sigma_xx=-100.0, sigma_yy=-200.0,
        sigma_zz=-300.0, ramp_steps=10, lambda_install=0.5)
    ops.region(name="cols_region", pg="Cols")
    ops.equation_constraint(constrained=(1, 1), retained=[(2, 1, 1.0)])
    ops.damping.rayleigh(alpha_m=0.1, beta_k=0.01, on="Cols")
    return ops


def _broad_partitioned() -> apeSees:
    return build_model("two_column_frame_partitioned", "staged_partitioned", "recording")


@pytest.mark.parametrize("build", [_broad_staged, _broad_frame, _broad_partitioned])
def test_every_key_has_exactly_one_row(
    tmp_path: Path, build: Callable[[], apeSees],
) -> None:
    ops = build()
    p = tmp_path / "m.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the equation-constraint ledger warning
        ops.h5(str(p))
    table = _table(p)
    with h5py.File(str(p), "r") as f:
        decl = f["opensees"]["decl_params"]["decl"][()].tolist()
    assert sorted(decl) == list(range(len(table.decls)))
    families = {d.family for d in table.decls}
    assert {"fix", "element"} <= families
    if build is _broad_staged:
        assert {"mass", "recorder", "rayleigh", "modal_damping", "damping",
                "region"} <= families
    if build is _broad_frame:
        assert {"initial_stress", "region", "equation_constraint",
                "rayleigh"} <= families
    # Each tagless record's own fields are its row: a fix's dofs, a
    # Rayleigh's coefficients and ``on``; a damping object its own fields.
    for i, d in enumerate(table.decls):
        ro = table.params[i]
        if d.family == "fix":
            assert "dofs" in ro.params, d.key
        if d.family == "rayleigh":
            assert {"alpha_m", "beta_k", "on"} <= set(ro.params), d.key
        if d.family == "damping":
            assert ro.type in {"Uniform", "SecStif", "URD", "URDbeta"}, d.key
            assert {"activate_time", "factor"} <= set(ro.params), d.key
    _assert_params_read_back(ops, p)


def test_model_wide_declaration_has_a_row(tmp_path: Path) -> None:
    ops = _flat_frame(named=False)
    ops.mass_from_model()
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    table = _table(p)
    rows = [ro for ro in table.params.values() if ro.type == "_ModelWideDeclaration"]
    assert [dict(ro.params) for ro in rows] == [{"verb": "mass_from_model"}]
    assert set(table.params) == set(range(len(table.decls)))


# ---------------------------------------------------------------------------
# Every reference of a family, in field order
# ---------------------------------------------------------------------------


def test_hinge_integration_lists_every_section_in_field_order(
    tmp_path: Path,
) -> None:
    ops = apeSees(cast("Any", build_simple_frame_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    s_i = ops.section.Elastic(E=1.0, A=1.0, Iz=1.0, Iy=1.0, G=1.0, J=1.0)
    s_j = ops.section.Elastic(E=2.0, A=1.0, Iz=1.0, Iy=1.0, G=1.0, J=1.0)
    s_e = ops.section.Elastic(E=3.0, A=1.0, Iz=1.0, Iy=1.0, G=1.0, J=1.0)
    hinge = ops.beamIntegration.HingeRadau(
        section_i=s_i, lp_i=0.1, section_j=s_j, lp_j=0.2, section_interior=s_e)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=hinge)
    p = tmp_path / "m.h5"
    ops.h5(str(p))
    table = _table(p)
    bm = ops.build()
    keys = [bm._declaration_key(s) for s in (s_i, s_j, s_e)]
    assert len(set(keys)) == 3
    ro = table.params_for(bm._declaration_key(hinge))
    assert ro.section_ref == tuple(keys)
    assert [ro.params[n] for n in ("section_i", "section_j", "section_interior")] == [
        DeclRef(key=k) for k in keys]
    assert ro.transf_ref == () and ro.integration_ref == ()
    el = table.params_for(bm._declaration_key(
        next(p_ for p_ in bm.primitives if type(p_).__name__ == "forceBeamColumn")))
    assert el.integration_ref == (bm._declaration_key(hinge),)
    # Echoed verbatim by a rewrite.
    q = tmp_path / "out.h5"
    OpenSeesModel.from_h5(str(p)).to_h5(str(q))
    assert _table(q).params_for(bm._declaration_key(hinge)).section_ref == tuple(keys)


# ---------------------------------------------------------------------------
# Enum and Fraction refuse explicitly
# ---------------------------------------------------------------------------


class _Colour(enum.Enum):
    RED = 1


class _Mode(str, enum.Enum):
    FAST = "fast"


class _Level(enum.IntEnum):
    LOW = 1


@pytest.mark.parametrize("weird", [
    _Colour.RED, _Mode.FAST, _Level.LOW, fractions.Fraction(1, 3),
    (1.0, _Level.LOW), {"a": fractions.Fraction(2, 5)},
], ids=["enum", "str-enum", "int-enum", "fraction", "nested-int-enum",
        "nested-fraction"])
def test_enum_and_fraction_refuse(weird: Any) -> None:
    with pytest.raises(H5DeclParamsError, match="Enum member or a Fraction"):
        encode_decl_params(_Odd(weird=weird), lambda p: "k")


def test_reader_refuses_a_repeated_declaration(tmp_path: Path) -> None:
    p = tmp_path / "m.h5"
    _params_frame().h5(str(p))
    with h5py.File(str(p), "r") as f:
        n = len(f["opensees"]["decl_params"]["decl"])
    _tamper(p, "decl", [0] * n)
    with pytest.raises(MalformedH5Error, match="repeat"):
        _table(p)
