"""M2-S2: the ``.ladruno`` particle reader (ADR 0134 D3, D4).

Every value is checked against a direct ``h5py`` read of the fork's
fixtures (``tests/fixtures/mpm/``, vendored by M2-S0), or of a copy the
test edits in place: the crash rule (contract 232a §2.5) on a torn tail,
the fork's attribute encodings beside h5py's, contiguous storage beside
the fork's chunks, and the refusals (C2-PART, C2-MINOR, C2-PAIR, an
inexact dtype).
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_particles import (
    ParticleFormatError,
    ParticleOutputsTruncatedWarning,
)
from apeGmsh.results.readers._protocol import ResultLevel

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MPM = FIXTURES / "mpm"
PARTICLES = MPM / "particles_v1m2.ladruno"
MIXED = MPM / "particles_mixed_v1m2.ladruno"
TORN = MPM / "particles_torn_v1m2.ladruno"   # the fork's torn file (M2-S6)
STAGE = "MODEL_STAGE[1]"           # the particle stage in both fixtures
ON_P = f"{STAGE}/RESULTS/ON_PARTICLES"
REG = f"{STAGE}/MODEL/PARTICLES"
FIELDS = ("POSITION", "VELOCITY", "STRESS", "STRAIN", "VOLUME",
          "DEFORMATION_GRADIENT", "STATE_eqPlasticStrain")


def _raw(path: Path) -> dict:
    """The particle stage as h5py reads it, whole."""
    with h5py.File(path, "r") as f:
        out = {k: f[f"{ON_P}/{k}"][...] for k in ("STEP", "TIME", "OFFSETS", "ID")}
        out["DATA"] = {n: f[f"{ON_P}/{n}/DATA"][...] for n in FIELDS}
        out["REG_ID"] = f[f"{REG}/ID"][...]
        out["BODY_ID"] = f[f"{REG}/BODY_ID"][...]
        out["MATERIAL_TAG"] = f[f"{REG}/MATERIAL_TAG"][...]
        out["MASS"] = f[f"{REG}/MASS"][...]
        names = f[f"{REG}/BODIES/NAME"][...]
        out["BODIES"] = dict(zip(f[f"{REG}/BODIES/ID"][...].tolist(),
                                 [n.decode() for n in names]))
    return out


@pytest.fixture
def copy_of(tmp_path):
    def _copy(src: Path, name: str | None = None) -> Path:
        dst = tmp_path / (name or src.name)
        shutil.copyfile(src, dst)
        return dst
    return _copy


@pytest.fixture
def opened():
    """Open Results and close them after the test."""
    held = []

    def _open(path, **kw):
        r = Results.from_ladruno(path, **kw)
        held.append(r)
        return r
    yield _open
    for r in held:
        r.close()


def _particle_stage_id(r) -> str:
    (sid,) = r.particles.stages()
    return sid


# ---------------------------------------------------------------------
# Both fixtures: values equal direct h5py reads
# ---------------------------------------------------------------------

@pytest.mark.parametrize("path", [PARTICLES, MIXED], ids=lambda p: p.name)
@pytest.mark.parametrize("field", FIELDS)
def test_field_equals_h5py(opened, path, field) -> None:
    raw = _raw(path)
    r = opened(path)
    s = r.particles.get(component=field, stage=_particle_stage_id(r))
    np.testing.assert_array_equal(s.values, raw["DATA"][field])
    np.testing.assert_array_equal(s.particle_ids, raw["ID"])
    np.testing.assert_array_equal(s.offsets, raw["OFFSETS"])
    np.testing.assert_array_equal(s.step, raw["STEP"])
    np.testing.assert_array_equal(s.time, raw["TIME"])
    np.testing.assert_array_equal(s.registry_ids, raw["REG_ID"])


@pytest.mark.parametrize("path", [PARTICLES, MIXED], ids=lambda p: p.name)
def test_positions_per_output(opened, path) -> None:
    raw = _raw(path)
    r = opened(path)
    s = r.particles.get(component="POSITION")
    off = raw["OFFSETS"]
    for k in range(off.size - 1):
        ids, vals = s.output(k)
        np.testing.assert_array_equal(ids, raw["ID"][off[k]:off[k + 1]])
        np.testing.assert_array_equal(
            vals, raw["DATA"]["POSITION"][off[k]:off[k + 1]])


def test_registry_bodies_grid_info(opened) -> None:
    raw = _raw(PARTICLES)
    r = opened(PARTICLES)
    reg = r.particles.registry()
    np.testing.assert_array_equal(reg.ids, raw["REG_ID"])
    np.testing.assert_array_equal(reg.body_ids, raw["BODY_ID"])
    np.testing.assert_array_equal(reg.material_tags, raw["MATERIAL_TAG"])
    np.testing.assert_array_equal(reg.mass, raw["MASS"])
    assert r.particles.bodies() == raw["BODIES"] == {0: "soil", 1: "block"}
    g = r.particles.grid()
    np.testing.assert_array_equal(g.origin, [0.0, 0.0])
    np.testing.assert_array_equal(g.cell_size, [0.5, 0.5])
    np.testing.assert_array_equal(g.num_cells, [4, 4])
    assert g.basis == "gimp"
    info = r.particles.info()
    assert (info.mode, info.epoch, info.plane_strain_thickness) == ("explicit", 0, 1.0)
    assert info.units == {"UNIT_LENGTH": "m", "UNIT_FORCE": "kN",
                          "UNIT_TIME": "s", "UNIT_SYSTEM": "kN_m_s"}
    assert info.source_set_id == "3f6c1d2e-8a4b-4c5d-9e7f-0a1b2c3d4e5f"


def test_state_is_nan_on_the_block(opened) -> None:
    raw = _raw(PARTICLES)
    r = opened(PARTICLES)
    dense = r.particles.get(component="STATE_eqPlasticStrain").dense()
    block = raw["BODY_ID"] == 1
    assert block.sum() == 4
    assert np.isnan(dense[:, block, :]).all()
    # The soil's values are the file's, where present.
    soil = r.particles.get(component="STATE_eqPlasticStrain", body="soil")
    assert np.isfinite(soil.values).all()
    assert set(soil.particle_ids.tolist()) <= set(raw["REG_ID"][~block].tolist())


def test_particle_11_is_absent_from_output_2(opened) -> None:
    raw = _raw(PARTICLES)
    r = opened(PARTICLES)
    s = r.particles.get(component="POSITION")
    dense = s.dense()
    row = int(np.searchsorted(raw["REG_ID"], 11))
    assert raw["REG_ID"][row] == 11
    assert dense.shape == (3, raw["REG_ID"].size, 2)
    assert np.isnan(dense[2, row]).all()
    off = raw["OFFSETS"]
    for k in (0, 1):
        ids = raw["ID"][off[k]:off[k + 1]]
        np.testing.assert_array_equal(
            dense[k, row], raw["DATA"]["POSITION"][off[k] + int(np.flatnonzero(ids == 11)[0])])
    assert 11 not in s.output(2)[0]


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------

def test_body_selection(opened) -> None:
    raw = _raw(PARTICLES)
    r = opened(PARTICLES)
    s = r.particles.get(component="STRESS", body="block")
    block_ids = raw["REG_ID"][raw["BODY_ID"] == 1]
    np.testing.assert_array_equal(s.registry_ids, block_ids)
    keep = np.isin(raw["ID"], block_ids)
    np.testing.assert_array_equal(s.particle_ids, raw["ID"][keep])
    np.testing.assert_array_equal(s.values, raw["DATA"]["STRESS"][keep])
    # Offsets count each output's kept rows.
    counts = [int(keep[a:b].sum()) for a, b in
              zip(raw["OFFSETS"][:-1], raw["OFFSETS"][1:])]
    np.testing.assert_array_equal(np.diff(s.offsets), counts)
    both = r.particles.get(component="STRESS", body=["soil", "block"])
    np.testing.assert_array_equal(both.values, raw["DATA"]["STRESS"])


def test_ids_selection_and_item(opened) -> None:
    raw = _raw(PARTICLES)
    r = opened(PARTICLES)
    s = r.particles.get(component="STRESS/sigma12", ids=[3], time=-1)
    off = raw["OFFSETS"]
    ids2 = raw["ID"][off[2]:off[3]]
    expect = raw["DATA"]["STRESS"][off[2] + int(np.flatnonzero(ids2 == 3)[0]), 2]
    assert s.item() == expect
    assert s.components == ("sigma12",)
    with pytest.raises(ValueError, match="item"):
        r.particles.get(component="STRESS", ids=[3]).item()


def test_output_selection_by_index_and_step(opened) -> None:
    raw = _raw(PARTICLES)
    r = opened(PARTICLES)
    off = raw["OFFSETS"]
    pos = raw["DATA"]["POSITION"]
    picked = r.particles.get(component="POSITION", time=[2, 0])
    np.testing.assert_array_equal(picked.step, raw["STEP"][[2, 0]])
    np.testing.assert_array_equal(
        picked.values, np.concatenate([pos[off[2]:off[3]], pos[off[0]:off[1]]]))
    by_step = r.particles.get(component="POSITION", step=20)
    np.testing.assert_array_equal(by_step.values, pos[off[2]:off[3]])
    sl = r.particles.get(component="POSITION", time=slice(1, None))
    np.testing.assert_array_equal(sl.values, pos[off[1]:off[3]])
    np.testing.assert_array_equal(sl.offsets, off[1:] - off[1])


@pytest.mark.parametrize("kw, exc, match", [
    ({"time": 0.001}, TypeError, "never by\\s+time"),
    ({"time": 3}, IndexError, "out of range"),
    ({"step": 5}, KeyError, "STEP"),
    ({"time": 0, "step": 0}, ValueError, "not both"),
    ({"body": "rock"}, KeyError, "soil"),
    ({"ids": [999]}, KeyError, "999"),
    ({"body": "soil", "ids": [0]}, ValueError, "not both"),
    ({"component": "STRESS/sigma33"}, KeyError, "sigma12"),
])
def test_selection_refusals(opened, kw, exc, match) -> None:
    r = opened(PARTICLES)
    kw = {"component": "POSITION", **kw}
    with pytest.raises(exc, match=match):
        r.particles.get(**kw)


# ---------------------------------------------------------------------
# The all-particle file opens without model_h5=, and has no FEM
# ---------------------------------------------------------------------

def test_all_particle_file_opens_without_a_model(opened) -> None:
    r = opened(PARTICLES)
    assert r.fem is None
    assert int(r.model.ndm) == 2
    assert r.inspect.components()["particles"] == r.particles.components()
    assert r.particles.components()[:2] == [
        "DEFORMATION_GRADIENT/F11", "DEFORMATION_GRADIENT/F12"]


@pytest.mark.parametrize("read", [
    lambda r: r.nodes.get(component="displacement_x"),
    lambda r: r.elements.get(component="globalForce"),
    lambda r: r.elements.gauss.get(component="stress_xx"),
    lambda r: r.elements.line_stations.get(component="axial_force"),
    lambda r: r.elements.fibers.get(component="fiber_stress"),
    lambda r: r.elements.layers.get(component="fiber_stress"),
], ids=["nodes", "elements", "gauss", "line_stations", "fibers", "layers"])
def test_fe_reads_of_a_particle_stage_refuse_by_name(opened, read) -> None:
    for path in (PARTICLES, MIXED):
        r = opened(path)
        scoped = r.stage(_particle_stage_id(r))
        with pytest.raises(ValueError, match=r"holds MPM\s+particles only"):
            read(scoped)


def test_fe_selection_without_fem_refuses_by_name(opened) -> None:
    r = opened(PARTICLES)
    with pytest.raises(RuntimeError, match="requires a bound FEMData"):
        r.nodes.get(component="displacement_x", pg="base")


def test_mixed_fe_stage_reads_as_before(opened) -> None:
    r = opened(MIXED)
    fe = r.stage("MODEL_STAGE[0]")
    with h5py.File(MIXED, "r") as f:
        nodes = f["MODEL_STAGE[0]/MODEL/NODES/ID"][...].ravel()
        ids = f["MODEL_STAGE[0]/RESULTS/ON_NODES/DISPLACEMENT/ID"][...].ravel()
        ux = f["MODEL_STAGE[0]/RESULTS/ON_NODES/DISPLACEMENT/DATA"][..., 0]
    slab = fe.nodes.get(component="displacement_x", ids=ids)
    np.testing.assert_array_equal(slab.node_ids, ids)
    np.testing.assert_array_equal(slab.values, ux)
    assert r.fem is not None and r.fem.nodes.ids.size == nodes.size
    with pytest.raises(ValueError, match="holds no particle group"):
        r.particles.get(component="POSITION", stage="MODEL_STAGE[0]")


def test_a_file_without_particles_refuses_particle_reads(opened) -> None:
    r = opened(FIXTURES / "ladruno" / "quad2d.ladruno")
    assert r.particles.stages() == []
    with pytest.raises(ValueError, match="no stage holds a particle group"):
        r.particles.get(component="POSITION")


def test_a_partitioned_set_refuses_particle_reads(opened) -> None:
    r = opened(FIXTURES / "ladruno" / "truss2d.part-0.ladruno")
    with pytest.raises(ValueError, match="LadrunoMultiPartitionReader"):
        r.particles.stages()


# ---------------------------------------------------------------------
# File-level refusals
# ---------------------------------------------------------------------

def test_part_file_with_particles_is_refused_by_name(copy_of) -> None:
    part = copy_of(PARTICLES, "case.part-0.ladruno")
    with pytest.raises(ParticleFormatError, match=r"case\.part-0\.ladruno.*C2-PART"):
        Results.from_ladruno(part, merge_partitions=False)


def test_partitioned_flag_with_particles_is_refused(copy_of) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        f["INFO"].attrs["PARTITIONED"] = np.array([1], dtype=np.int32)
    with pytest.raises(ParticleFormatError, match="C2-PART"):
        LadrunoReader(p)


def test_minor_1_with_particles_is_refused(copy_of) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        f["INFO"].attrs["FORMAT_MINOR_VERSION"] = np.array([1], dtype=np.int32)
    with pytest.raises(ParticleFormatError, match="C2-MINOR"):
        LadrunoReader(p)


def test_half_a_particle_group_is_refused(copy_of) -> None:
    p = copy_of(MIXED)
    with h5py.File(p, "r+") as f:
        del f[ON_P]
    with pytest.raises(ParticleFormatError, match="C2-PAIR"):
        LadrunoReader(p)


def test_an_inexact_dtype_is_refused(copy_of, opened) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        ids = f[f"{ON_P}/ID"][...]
        del f[f"{ON_P}/ID"]
        f[ON_P].create_dataset("ID", data=ids.astype(np.int32))
    r = opened(p)
    with pytest.raises(ParticleFormatError, match="never casts silently"):
        r.particles.get(component="POSITION")


def test_unsafe_integer_ids_are_refused(copy_of, opened) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        ds = f[f"{REG}/ID"]
        ids = ds[...]
        ids[-1] = 2 ** 53
        ds[...] = ids
    r = opened(p)
    with pytest.raises(ParticleFormatError, match="safe integer"):
        r.particles.registry()


# ---------------------------------------------------------------------
# Encodings and storage the contract allows
# ---------------------------------------------------------------------

def test_h5py_scalar_and_vlen_encodings_read_alike(copy_of, opened) -> None:
    """The fork writes 1-element arrays and fixed NULLTERM strings; h5py
    writers write scalars and variable-length strings. Both read alike,
    and an absent string attribute reads as ''."""
    ref = opened(PARTICLES)
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        for path in (REG, f"{REG}/GRID", ON_P, *(f"{ON_P}/{n}" for n in FIELDS)):
            attrs = f[path].attrs
            for k in list(attrs):
                v = np.asarray(attrs[k])
                if v.dtype.kind == "S":
                    attrs[k] = v.reshape(-1)[0].decode()     # vlen str scalar
                elif v.size == 1:
                    attrs[k] = v.reshape(-1)[0]              # scalar dataspace
        f[f"{STAGE}"].attrs["PARTICLE_EPOCH"] = np.int32(0)
        names = [n.decode() for n in f[f"{REG}/BODIES/NAME"][...]]
        del f[f"{REG}/BODIES/NAME"]
        f[f"{REG}/BODIES"].create_dataset(
            "NAME", data=names, dtype=h5py.string_dtype())
        del f[f"{ON_P}/STRESS"].attrs["DIMENSION"]
    r = opened(p)
    assert r.particles.info() == ref.particles.info()
    assert r.particles.bodies() == ref.particles.bodies()
    np.testing.assert_array_equal(r.particles.grid().num_cells,
                                  ref.particles.grid().num_cells)
    fields = r._reader.particle_stage(_particle_stage_id(r)).fields()
    assert fields["STRESS"].dimension == ""
    assert fields["DEFORMATION_GRADIENT"].dimension == ""   # absent in the fixture
    assert fields["POSITION"].dimension == "L"
    np.testing.assert_array_equal(
        r.particles.get(component="STRESS").values,
        ref.particles.get(component="STRESS").values)


def test_contiguous_storage_reads_alike(copy_of, opened) -> None:
    """Nothing depends on the chunk shape: contiguous, uncompressed
    datasets read the same values."""
    raw = _raw(PARTICLES)
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        def rewrite(grp, name):
            data = grp[name][...]
            del grp[name]
            grp.create_dataset(name, data=data)
        for name in ("STEP", "TIME", "OFFSETS", "ID"):
            rewrite(f[ON_P], name)
        for n in FIELDS:
            rewrite(f[f"{ON_P}/{n}"], "DATA")
        assert f[f"{ON_P}/ID"].chunks is None
    r = opened(p)
    for n in FIELDS:
        np.testing.assert_array_equal(
            r.particles.get(component=n).values, raw["DATA"][n])


# ---------------------------------------------------------------------
# The crash rule (contract §2.5): torn tails and a zero-output stage
# ---------------------------------------------------------------------

def _resize(ds, n):
    ds.resize((n,) + ds.shape[1:])


def test_closed_file_reads_without_a_truncation_warning(opened) -> None:
    r = opened(PARTICLES)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParticleOutputsTruncatedWarning)
        r.particles.get(component="POSITION")


@pytest.mark.parametrize("torn_field", ["POSITION", "STATE_eqPlasticStrain"])
def test_torn_data_drops_the_output_it_cuts(copy_of, opened, torn_field) -> None:
    """ID longer than one field's DATA: the output whose rows DATA lacks
    is not read, in any field; every output before it is."""
    raw = _raw(PARTICLES)
    off = raw["OFFSETS"]
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        _resize(f[f"{ON_P}/{torn_field}/DATA"], int(off[2]) + 3)
    r = opened(p)
    with pytest.warns(ParticleOutputsTruncatedWarning, match="2 output"):
        s = r.particles.get(component="VELOCITY")
    np.testing.assert_array_equal(s.offsets, off[:3])
    np.testing.assert_array_equal(s.step, raw["STEP"][:2])
    np.testing.assert_array_equal(s.values, raw["DATA"]["VELOCITY"][:off[2]])
    assert s.dense().shape == (2, raw["REG_ID"].size, 2)


def test_an_in_flight_output_is_ignored(copy_of, opened) -> None:
    """A run killed while writing output 3: its DATA, ID, STEP and TIME
    rows are there, OFFSETS is not. The three committed outputs read."""
    raw = _raw(PARTICLES)
    off = raw["OFFSETS"]
    extra = 7
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        for n in FIELDS:
            ds = f[f"{ON_P}/{n}/DATA"]
            _resize(ds, ds.shape[0] + extra)
            ds[-extra:] = 9.0e9
        _resize(f[f"{ON_P}/ID"], int(off[-1]) + extra)
        f[f"{ON_P}/ID"][-extra:] = np.arange(extra)
        _resize(f[f"{ON_P}/STEP"], 4)
        f[f"{ON_P}/STEP"][3] = 30
        _resize(f[f"{ON_P}/TIME"], 4)
    r = opened(p)
    with pytest.warns(ParticleOutputsTruncatedWarning, match="3 output"):
        s = r.particles.get(component="POSITION")
    np.testing.assert_array_equal(s.values, raw["DATA"]["POSITION"])
    np.testing.assert_array_equal(s.step, raw["STEP"])
    assert not (s.values == 9.0e9).any()


def test_offsets_ahead_of_id_drop_the_uncommitted_output(copy_of, opened) -> None:
    """``OFFSETS[T] > len(ID)``: the shapes of the full file are not
    trusted; the output ID does not cover is dropped."""
    raw = _raw(PARTICLES)
    off = raw["OFFSETS"]
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        _resize(f[f"{ON_P}/ID"], int(off[2]) + 1)
    r = opened(p)
    with pytest.warns(ParticleOutputsTruncatedWarning):
        s = r.particles.get(component="POSITION", time=-1)
    np.testing.assert_array_equal(s.step, raw["STEP"][1:2])
    np.testing.assert_array_equal(s.values, raw["DATA"]["POSITION"][off[1]:off[2]])


def test_a_zero_output_stage_reads_empty(copy_of, opened) -> None:
    raw = _raw(PARTICLES)
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        for name in ("STEP", "TIME", "ID"):
            _resize(f[f"{ON_P}/{name}"], 0)
        _resize(f[f"{ON_P}/OFFSETS"], 1)
        for n in FIELDS:
            _resize(f[f"{ON_P}/{n}/DATA"], 0)
    r = opened(p)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParticleOutputsTruncatedWarning)
        s = r.particles.get(component="POSITION")
    assert s.n_outputs == 0 and s.values.shape == (0, 2)
    assert s.dense().shape == (0, raw["REG_ID"].size, 2)
    with pytest.raises(IndexError):
        r.particles.get(component="POSITION", time=0)


def test_an_extended_but_unwritten_offset_is_a_torn_output(copy_of, opened) -> None:
    """The writer extended OFFSETS for output 3 (fill 0) after STEP and
    TIME: the last OFFSETS entry decreases. That output is torn; the three
    before it read."""
    raw = _raw(PARTICLES)
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        for name in ("STEP", "TIME"):
            _resize(f[f"{ON_P}/{name}"], 4)
        f[f"{ON_P}/STEP"][3] = 30
        f[f"{ON_P}/TIME"][3] = 0.003
        _resize(f[f"{ON_P}/OFFSETS"], 5)
        f[f"{ON_P}/OFFSETS"][4] = 0
    r = opened(p)
    with pytest.warns(ParticleOutputsTruncatedWarning, match="3 output"):
        s = r.particles.get(component="POSITION")
    np.testing.assert_array_equal(s.offsets, raw["OFFSETS"])
    np.testing.assert_array_equal(s.step, raw["STEP"])
    np.testing.assert_array_equal(s.values, raw["DATA"]["POSITION"])


@pytest.mark.parametrize("field", FIELDS)
def test_fork_torn_fixture_reads_its_three_committed_outputs(opened, field) -> None:
    """The fork's own torn file (M2-S6 #2990; fork e30872e34, contract
    §2.5): the all-particle fixture with a fourth output whose OFFSETS
    entry was extended but never written, ``OFFSETS = [0, 12, 24, 35, 0]``.
    Its DATA, ID, STEP and TIME rows for that output are on disk. Three
    outputs read, each equal to a direct h5py read of the committed rows
    and to the closed fixture; the fourth output's rows never surface."""
    torn = _raw(TORN)
    assert torn["OFFSETS"].tolist() == [0, 12, 24, 35, 0]
    assert torn["STEP"].size == 4
    assert torn["ID"].size > 35
    off = torn["OFFSETS"][:4]
    closed = _raw(PARTICLES)
    r = opened(TORN)
    with pytest.warns(ParticleOutputsTruncatedWarning, match="3 output"):
        s = r.particles.get(component=field)
    assert s.n_outputs == 3
    np.testing.assert_array_equal(s.offsets, off)
    np.testing.assert_array_equal(s.step, torn["STEP"][:3])
    np.testing.assert_array_equal(s.time, torn["TIME"][:3])
    np.testing.assert_array_equal(s.particle_ids, torn["ID"][:off[-1]])
    np.testing.assert_array_equal(s.values, torn["DATA"][field][:off[-1]])
    np.testing.assert_array_equal(s.registry_ids, torn["REG_ID"])
    # The torn file is the closed fixture plus one in-flight output.
    np.testing.assert_array_equal(s.offsets, closed["OFFSETS"])
    np.testing.assert_array_equal(s.step, closed["STEP"])
    np.testing.assert_array_equal(s.values, closed["DATA"][field])


def test_a_torn_entry_after_empty_outputs_reads_as_one_more_empty(copy_of, opened) -> None:
    """Every output so far is empty (all offsets 0): a torn OFFSETS
    entry (fill 0) equals its predecessor, so it cannot be told from a
    committed empty output and reads as one. That is accepted, not
    refused (the fork's rule, adopted with item (c))."""
    raw = _raw(PARTICLES)
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        _resize(f[f"{ON_P}/ID"], 0)
        for n in FIELDS:
            _resize(f[f"{ON_P}/{n}/DATA"], 0)
        # Two committed empty outputs, then output 2's STEP and TIME and
        # an OFFSETS extension that still holds its fill.
        f[f"{ON_P}/OFFSETS"][...] = np.zeros(4, dtype=np.int64)
    r = opened(p)
    s = r.particles.get(component="POSITION")
    assert s.n_outputs == 3
    np.testing.assert_array_equal(s.offsets, [0, 0, 0, 0])
    np.testing.assert_array_equal(s.step, raw["STEP"])
    assert s.values.shape == (0, 2)
    assert np.isnan(s.dense()).all()


@pytest.mark.parametrize("offsets, n_steps", [
    ([3, 12, 24, 35], 3),         # step (1): OFFSETS[0] is not 0
    ([0, 12, 5, 35], 3),          # step (3): a decrease before the last entry
    ([0, 12, 5, 35, 0], 4),       # step (3) after step (2) drops a torn entry
    ([0, 12, 5, 35], 1),          # step (3) runs on all of OFFSETS, before T'
], ids=["nonzero-start", "mid-decrease", "mid-decrease-and-torn",
        "decrease-past-T"])
def test_a_decrease_before_the_last_offset_refuses(copy_of, opened, offsets, n_steps) -> None:
    """The fork's §2.5 steps (1) and (3), on OFFSETS as it stands on disk."""
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        for name in ("STEP", "TIME"):
            _resize(f[f"{ON_P}/{name}"], n_steps)
        f[f"{ON_P}/STEP"][...] = np.arange(n_steps, dtype=np.int32) * 10
        _resize(f[f"{ON_P}/OFFSETS"], len(offsets))
        f[f"{ON_P}/OFFSETS"][...] = np.asarray(offsets, dtype=np.int64)
    r = opened(p)
    with pytest.raises(ParticleFormatError, match="C2-OFF"):
        r.particles.get(component="POSITION")


@pytest.mark.parametrize("dtype", [np.int32, np.uint64], ids=["int32", "uint64"])
def test_registry_ids_must_be_int64(copy_of, opened, dtype) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        ids = f[f"{REG}/ID"][...]
        del f[f"{REG}/ID"]
        f[REG].create_dataset("ID", data=ids.astype(dtype))
    r = opened(p)
    with pytest.raises(ParticleFormatError, match=r"MODEL/PARTICLES/ID is .*int64"):
        r.particles.registry()


@pytest.mark.parametrize("value", [None, "SUM"], ids=["absent", "SUM"])
def test_partition_reduction_must_be_none(copy_of, opened, value) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        attrs = f[ON_P].attrs
        if value is None:
            del attrs["PARTITION_REDUCTION"]
        else:
            attrs["PARTITION_REDUCTION"] = np.bytes_(value)
    r = opened(p)
    with pytest.raises(ParticleFormatError, match="C2-RED|C2-ATTR"):
        r.particles.get(component="POSITION")


def test_step_must_strictly_increase(copy_of, opened) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        f[f"{ON_P}/STEP"][...] = np.array([0, 10, 10], dtype=np.int32)
    r = opened(p)
    with pytest.raises(ParticleFormatError, match="STEP is not strictly increasing.*C2-AXES"):
        r.particles.get(component="POSITION")


def test_a_refused_stage_is_never_reported_as_read(copy_of, opened) -> None:
    """Every access to a refused stage raises, every time: no component
    listing, registry or partial read survives the first refusal."""
    p = copy_of(MIXED)
    with h5py.File(p, "r+") as f:
        f[f"{ON_P}/STEP"][...] = np.array([0, 20, 10], dtype=np.int32)
    r = opened(p)
    sid = _particle_stage_id(r)
    for _ in range(2):
        for access in (
            lambda: r.particles.components(),
            lambda: r.particles.registry(),
            lambda: r.particles.bodies(),
            lambda: r.particles.get(component="POSITION"),
            lambda: r._reader.available_components(sid, ResultLevel.PARTICLES),
            lambda: r.inspect.components(stage=sid),
        ):
            with pytest.raises(ParticleFormatError, match="C2-AXES"):
                access()
    # The FE stage of the same file still reads.
    assert r.stage("MODEL_STAGE[0]").nodes.available_components()


@pytest.mark.parametrize("row, value, match", [
    (11, 2 ** 60, "safe integer"),          # unsafe, and in no registry
    (11, 99, r"\[99\].*C2-ID"),              # not in the registry
    (13, None, "repeat or are not sorted.*C2-ID"),   # repeats its neighbour
], ids=["unsafe", "unregistered", "repeated"])
@pytest.mark.parametrize("body", [None, "soil"], ids=["all", "soil"])
def test_output_ids_are_checked_on_the_rows_read(copy_of, opened, row, value, match, body) -> None:
    """C2-ID on what get() reads: a bad output id refuses the read, with
    or without a body selection that would otherwise drop its row."""
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        ds = f[f"{ON_P}/ID"]
        ids = ds[...]
        ids[row] = ids[row - 1] if value is None else value
        ds[...] = ids
    r = opened(p)
    with pytest.raises(ParticleFormatError, match=match):
        r.particles.get(component="POSITION", body=body)


def test_registry_columns_of_unequal_length_refuse(copy_of, opened) -> None:
    """Contract §2.4: every registry column is ``[M]``; unequal columns are
    C2-REG. Since OpenSees#1081 (232a §2.5, crash rule step 0) such a stage
    was torn while it opened, and the stage rule skips it when it is the
    last stage of a file opened by step 1 or 3 (``ParticleFile``, below).
    ``LadrunoReader`` does not know the open step until it adopts the shared
    open (#3003), so through it the stage is still refused."""
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        _resize(f[f"{REG}/MASS"], f[f"{REG}/MASS"].shape[0] - 2)
    r = opened(p)
    with pytest.raises(ParticleFormatError, match="differ in length.*C2-REG"):
        r.particles.registry()


@pytest.mark.parametrize("edit, match", [
    (lambda a: a.__setitem__("PARTITION_REDUCTION", np.bytes_("SUM")), "C2-RED"),
    (lambda a: a.__delitem__("PARTITION_REDUCTION"), "C2-ATTR"),
    (lambda a: a.__delitem__("TYPE"), "'TYPE' is missing.*C2-ATTR"),
], ids=["reduction-SUM", "reduction-absent", "type-absent"])
def test_field_attributes_are_checked(copy_of, opened, edit, match) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        edit(f[f"{ON_P}/VELOCITY"].attrs)
    r = opened(p)
    with pytest.raises(ParticleFormatError, match=match):
        r.particles.components()


def test_result_level_has_particles() -> None:
    assert ResultLevel("particles") is ResultLevel.PARTICLES


# --- UNIT_SYSTEM against UNIT_LENGTH / UNIT_FORCE / UNIT_TIME (#2991) ------------------

@pytest.mark.parametrize("system, ok", [
    ("kN_m_s", True),          # the fixture's: kN, m, s
    ("N_mm_s", False),         # N, mm, s: force and length disagree
    ("tf_m_s", False),         # tf, m, s: force disagrees
    ("not_an_apegmsh_id", True),   # informative and unknown: not used, not checked
])
def test_unit_system_must_agree_with_the_unit_attributes(copy_of, opened, system, ok) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        f[REG].attrs["UNIT_SYSTEM"] = np.bytes_(system)
    r = opened(p)
    if ok:
        assert r.particles.get(component="POSITION").n_outputs == 3
    else:
        with pytest.raises(ParticleFormatError, match="UNIT_SYSTEM.*disagree.*C2-UNIT"):
            r.particles.get(component="POSITION")


# --- a killed run: the stage rule by open step (232a §2.5 at a1d4c7260, #2991) ---------

TORN_LAST = MPM / "particles_torn_last_stage_v1m2.ladruno"     # step 0 in the last stage
# no ON_PARTICLES in the last stage: built per test by make_torn_stage_fixtures.build(p, "half"),
# not committed (LadrunoReader refuses it at open, and every committed .ladruno must open)
HALF_LAST = MPM / "particles_half_last_stage_v1m2.ladruno"
TORN_EARLY = MPM / "particles_torn_early_stage_v1m2.ladruno"   # step 0 in the first stage
ORACLE = MPM / "fork_a1d4c7260"


@pytest.fixture(scope="module")
def fork():
    """The fork's ``mpm_contracts`` / ``ladruno_format`` at a1d4c7260 (the
    stage rule's reference), with ``sys.path`` / ``sys.modules`` restored."""
    import importlib
    import sys

    names = ("mpm_contracts", "ladruno_format", "ladruno_basis")
    saved_path = list(sys.path)
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    sys.path.insert(0, str(ORACLE))
    try:
        yield importlib.import_module("mpm_contracts"), importlib.import_module("ladruno_format")
    finally:
        sys.path[:] = saved_path
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def build_torn_stage(path: Path, kind: str) -> Path:
    """``tests/fixtures/mpm/make_torn_stage_fixtures.py``'s ``build``, loaded by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_make_torn_stage_fixtures", MPM / "make_torn_stage_fixtures.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build(path, kind)
    return path


def _flagged(fork, copy_of, src: Path, flags: int) -> Path:
    if src == HALF_LAST:
        p = build_torn_stage(copy_of(TORN_LAST, f"{src.stem}.{flags:02x}.ladruno"), "half")
    else:
        p = copy_of(src, f"{src.stem}.{flags:02x}.ladruno")
    if flags:
        fork[1].set_superblock(p, flags=flags)
    return p


def test_ladruno_reader_refuses_the_half_stage_file_at_open(copy_of) -> None:
    """``LadrunoReader`` opens plainly and has no stage rule until #3003: the half
    group of a last stage killed mid-layout is C2-PAIR at open. ``ParticleFile``
    skips that stage (below)."""
    p = build_torn_stage(copy_of(TORN_LAST, "half.ladruno"), "half")
    with pytest.raises(ParticleFormatError, match="C2-PAIR"):
        Results.from_ladruno(p)


def _states(pf) -> list:
    return [(s.group, s.status, s.committed) for s in pf.states]


@pytest.mark.parametrize("src", [TORN_LAST, HALF_LAST], ids=["registry", "half"])
@pytest.mark.parametrize("flags, step, last", [
    (0x00, 1, "skipped"), (0x05, 2, "refused"), (0x01, 3, "skipped"),
], ids=["step1", "step2", "step3"])
def test_a_torn_last_stage_by_open_step(fork, copy_of, src, flags, step, last) -> None:
    from apeGmsh.results.readers._ladruno_particles import (
        ParticleFile,
        ParticleStageSkippedWarning,
    )
    p = _flagged(fork, copy_of, src, flags)
    before = p.read_bytes()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ParticleFile(p) as pf:
            assert pf.step == step
            assert _states(pf) == [("MODEL_STAGE[1]", "read", 3),
                                   ("MODEL_STAGE[2]", last, 0)]
            assert pf.stage("MODEL_STAGE[1]").read("POSITION").n_outputs == 3
            with pytest.raises(ParticleFormatError, match=f"MODEL_STAGE\\[2\\] is {last}"):
                pf.stage("MODEL_STAGE[2]")
            reason = pf.states[1].reason
    skipped = [x for x in w if issubclass(x.category, ParticleStageSkippedWarning)]
    if last == "skipped":
        assert "torn while it was laid out" in reason and len(skipped) == 1
    else:
        assert "every stage must read" in reason and "corruption" in reason and not skipped
    assert p.read_bytes() == before


# --- HDF5-level damage (#3008 review F1-F3): the viewer's ladrunoParticles.test.ts flips the
# same bytes of the same fixture and expects the same outcome for every case.

TWO = MPM / "particles_two_stages_v1m2.ladruno"   # both stages whole, 3 outputs each
#: (label, byte offset, signature there, damaged stage (2 = the last), its index still commits outputs)
DAMAGE = [
    ("stage2-links-TREE", 66699, b"TREE", 2, False),
    ("stage2-links-SNOD", 83851, b"SNOD", 2, False),
    ("stage2-OFFSETS", 95043, b"TREE", 2, False),
    ("stage2-STRESS-DATA", 113531, b"TREE", 2, True),
    ("stage1-links-TREE", 2795, b"TREE", 1, False),
    ("stage1-OFFSETS", 33187, b"TREE", 1, False),
    ("stage1-STRESS-DATA", 51675, b"TREE", 1, True),
]


def _damaged(fork, copy_of, off: int, sig: bytes, flags: int) -> Path:
    p = copy_of(TWO, f"damage_{off}_{flags:02x}.ladruno")
    raw = bytearray(p.read_bytes())
    assert raw[off:off + 4] == sig
    raw[off] ^= 0x20
    p.write_bytes(bytes(raw))
    if flags:
        fork[1].set_superblock(p, flags=flags)
    return p


def _fates(pf) -> list:
    return [(s.group, s.status + (" (corruption)" if s.status == "refused"
                                  and "corruption" in s.reason else ""), s.committed)
            for s in pf.states]


@pytest.mark.parametrize("label, off, sig, damaged, holds_outputs", DAMAGE,
                         ids=[d[0] for d in DAMAGE])
@pytest.mark.parametrize("flags, step", [(0x00, 1), (0x05, 2), (0x01, 3)],
                         ids=["step1", "step2", "step3"])
def test_hdf5_damage_through_the_stage_rule(fork, copy_of, label, off, sig, damaged,
                                            holds_outputs, flags, step) -> None:
    """A damaged B-tree node fails its stage, whatever h5py raises (RuntimeError
    for a link B-tree, OSError for a chunk index, KeyError for a heap): a stage
    that holds no countable output is torn (skipped as the last stage after step
    1 or 3, refused after step 2); one with outputs, or one before another
    stage, is refused. The other stage reads; the file is never written."""
    from apeGmsh.results.readers._ladruno_particles import ParticleFile

    p = _damaged(fork, copy_of, off, sig, flags)
    before = p.read_bytes()
    fate = ("refused" if holds_outputs else
            "refused (corruption)" if damaged == 1 or step == 2 else "skipped")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ParticleFile(p) as pf:
            assert pf.step == step
            whole = ("MODEL_STAGE[2]" if damaged == 1 else "MODEL_STAGE[1]", "read", 3)
            hurt = (f"MODEL_STAGE[{damaged}]", fate, 0)
            assert _fates(pf) == ([hurt, whole] if damaged == 1 else [whole, hurt])
    assert p.read_bytes() == before


def _damage_sweep_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_damage_sweep", MPM / "damage_sweep.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Sweep cases where the Python reader is stricter than the fork's reference: the reference
#: reads the stage (it counts outputs from the index alone), and the Python reader does not,
#: because it also reads the stage's metadata, registry, field attributes and last committed
#: rows (a damaged registry or field heap, or a field attribute that does not read). 343
#: (stage, file) pairs: 204 from attribute-message damage, 118 from signature flips, 18 from
#: 1 KiB zeroed blocks, 3 from 256-byte ones. Pinned so a change is seen.
STRICTER_THAN_REFERENCE = 343


def test_the_damage_sweep_gives_the_recorded_outcomes(fork, tmp_path) -> None:
    """Every damaged copy of ``tests/fixtures/mpm/damage_sweep.py`` (built here, removed
    after; "hang" cases, where HDF5 itself does not return, are never opened):

    1. the Python reader gives the recorded outcome, and the viewer's
       ``ladrunoParticles.test.ts`` checks ``openLadruno`` against the same table, so
       the two readers agree file by file;
    2. Python never reads a stage that the fork's reference ``particle_stage_report``
       (a1d4c7260) skips or refuses. Python may be stricter (it reads more of a stage
       than the index the reference counts); those cases are counted, not refused."""
    import json

    mc, lf = fork
    mod = _damage_sweep_module()
    expected = json.loads(mod.EXPECTED.read_text(encoding="utf-8"))
    raw = mod.BASE.read_bytes()
    wrong, looser, stricter, n = [], [], [], 0
    for name, data in mod.cases(raw):
        n += 1
        if expected.get(name) == ["hang"]:
            continue
        p = tmp_path / f"{name}.ladruno"
        p.write_bytes(data)
        got = mod.outcome(p)
        if got != expected.get(name):
            wrong.append((name, expected.get(name), got))
        ref = mod.reference_reads(p, tmp_path, mc, lf)
        mine = {}
        if got != ["file-error"]:
            for row in got[1:]:
                stage, what = row.split(" ", 1)
                mine[stage] = what
        for stage, what in ref.items():
            if not what.startswith("read") and mine.get(stage, "").startswith("read"):
                looser.append((name, stage, what, mine[stage]))
            if what.startswith("read") and not mine.get(stage, "").startswith("read"):
                stricter.append((name, stage, what, mine.get(stage, "file-error")))
        p.unlink()
        for q in tmp_path.glob("ref_*"):
            q.unlink()
    assert n == len(expected) and n >= 750
    assert not wrong, f"{len(wrong)} case(s) differ from the table: {wrong[:5]}"
    assert not looser, f"Python reads a stage the reference skips or refuses: {looser[:5]}"
    assert len(stricter) == STRICTER_THAN_REFERENCE, stricter[:10]


def test_an_eoa_past_the_end_at_flags_0x00_opens_swmr_read(copy_of) -> None:
    """h5py's plain open refuses "truncated file"; the SWMR-read open (step 2)
    skips that check, as the fork's open_ladruno does (the viewer agrees)."""
    from apeGmsh.results.readers._ladruno_open import lookup3
    from apeGmsh.results.readers._ladruno_particles import ParticleFile

    p = copy_of(TWO, "eoa_past_end.ladruno")
    raw = bytearray(p.read_bytes())
    raw[28:36] = (len(raw) + 4096).to_bytes(8, "little")
    raw[44:48] = lookup3(bytes(raw[:44])).to_bytes(4, "little")
    p.write_bytes(bytes(raw))
    with pytest.raises(OSError, match="truncated file"):
        h5py.File(p, "r")
    with ParticleFile(p) as pf:
        assert pf.step == 2
        assert _states(pf) == [("MODEL_STAGE[1]", "read", 3), ("MODEL_STAGE[2]", "read", 3)]


def test_a_bug_is_not_a_failing_stage(copy_of, monkeypatch) -> None:
    """Only HDF5's errors fail a stage; a TypeError (a bug) propagates."""
    from apeGmsh.results.readers import _ladruno_particles as lp

    def broken(self):
        raise TypeError("a bug")

    monkeypatch.setattr(lp.ParticleStage, "validate", broken)
    with pytest.raises(TypeError, match="a bug"):
        lp.ParticleFile(TWO)


@pytest.mark.parametrize("flags", [0x00, 0x05, 0x01], ids=["step1", "step2", "step3"])
def test_an_earlier_torn_stage_is_corruption_after_any_step(fork, copy_of, flags) -> None:
    from apeGmsh.results.readers._ladruno_particles import ParticleFile

    p = _flagged(fork, copy_of, TORN_EARLY, flags)
    before = p.read_bytes()
    with ParticleFile(p) as pf:
        assert _states(pf) == [("MODEL_STAGE[1]", "refused", 0),
                               ("MODEL_STAGE[2]", "read", 3)]
        assert "not the file's last stage, so it is corruption" in pf.states[0].reason
        with pytest.raises(ParticleFormatError, match="corruption"):
            pf.stage("MODEL_STAGE[1]")
    assert p.read_bytes() == before


@pytest.mark.parametrize("edit, match", [
    (lambda f: _resize(f[f"{REG}/MASS"], 11), "differ in length"),
    (lambda f: f[ON_P].attrs.__setitem__("PARTITION_REDUCTION", np.bytes_("SUM")), "C2-RED"),
], ids=["registry", "reduction"])
def test_a_failing_last_stage_that_holds_outputs_is_refused_not_skipped(copy_of, edit, match) -> None:
    """The writer lays a stage out in full, registry included, before output 0
    and creates nothing after (§2.5), so a stage whose index commits outputs
    was not torn while laid out: its failure is refused even as the last
    stage of a plainly opened file."""
    from apeGmsh.results.readers._ladruno_particles import ParticleFile

    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        edit(f)
    with ParticleFile(p) as pf:
        assert pf.step == 1
        assert _states(pf) == [(STAGE, "refused", 0)]
        assert "holds 3 output(s), so it was not torn" in pf.states[0].reason
        with pytest.raises(ParticleFormatError, match=match):
            pf.stage(STAGE)


@pytest.mark.parametrize("src", [TORN_LAST, HALF_LAST, TORN_EARLY],
                         ids=["registry", "half", "early"])
@pytest.mark.parametrize("flags", [0x00, 0x05, 0x01], ids=["step1", "step2", "step3"])
def test_the_stage_rule_agrees_with_the_forks_reference(fork, copy_of, tmp_path, src, flags) -> None:
    """``particle_stage_report`` at a1d4c7260 (after ``open_ladruno``, or
    ``recover_particle_file`` for step 3) on the same file. Where it counts
    a stage, the counts agree; where it raises, the reader refuses. Its step
    0 returns 0 committed for a torn registry in any stage and after any
    step, where the merged text (§2.5, lines 515-517) and the owner's ruling
    say "skip the LAST stage after step 1 or 3, refuse otherwise": the
    reader follows the text, so there it only agrees that the stage holds no
    output."""
    from apeGmsh.results.readers._ladruno_particles import ParticleFile

    mc, lf = fork
    p = _flagged(fork, copy_of, src, flags)
    try:
        if flags == 0x01:
            ref = mc.recover_particle_file(p, tmp_path / "ref.ladruno")
        else:
            with lf.open_ladruno(p) as f:
                ref = mc.particle_stage_report(f)
    except mc.TornStageError:
        ref = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ParticleFile(p) as pf:
            ours = {s.group: s for s in pf.states}
    if ref is None:
        assert any(s.status == "refused" for s in ours.values())
        return
    for entry in ref:
        mine = ours[entry["stage"]]
        if entry.get("torn_while_laid_out"):
            assert mine.status in ("skipped", "refused") and mine.committed == 0
        elif mine.status == "read":
            assert mine.committed == entry["committed"]
        else:   # step 0 in the reference: no output
            assert entry["committed"] == 0 and mine.committed == 0


@pytest.mark.parametrize("src", [PARTICLES, MIXED, TORN], ids=lambda p: p.name)
def test_a_closed_fork_fixture_reads_every_stage_through_the_rule(fork, src) -> None:
    from apeGmsh.results.readers._ladruno_particles import ParticleFile

    mc, _ = fork
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParticleOutputsTruncatedWarning)
        with ParticleFile(src) as pf:
            assert pf.step == 1
            assert _states(pf) == [
                (STAGE, "read", mc.committed_outputs(str(src), stage=STAGE))]


# --- GRID/BASIS: the Contract-2 set (232a §2.4, amendment 232i; #3013) -------------------
#
# The oracle is the fork's checker at ``ladruno-v20261118``: ``ladruno_format.py`` holds
# ``C2_BASES = ("gimp", "linear", "bspline2", "bspline3")`` and refuses any other value as
# C2-GRID. The B-spline files are ``particles_v1m2.ladruno`` with only BASIS rewritten
# (``tests/fixtures/mpm/make_basis_fixtures.py``); the viewer's ``ladrunoParticles.test.ts``
# reads the same files and makes the same refused copies. This reader has no Contract-1
# (particle-set) counterpart: C1-BASIS is the viewer's ``particleSet.test.ts``.

BSPLINE = {v: MPM / f"particles_{v}_v1m2.ladruno" for v in ("bspline2", "bspline3")}


def _set_basis(path: Path, value: str) -> None:
    """The fixture generator's own edit, loaded by path (``sys.path`` is untouched)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_make_basis_fixtures", MPM / "make_basis_fixtures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.set_basis(path, value)


def test_the_contract_2_basis_set_is_the_forks() -> None:
    from apeGmsh.results.readers._ladruno_particles import C2_BASES

    assert C2_BASES == ("gimp", "linear", "bspline2", "bspline3")


@pytest.mark.parametrize("basis", sorted(BSPLINE))
def test_a_bspline_grid_reads_and_reports_its_basis(opened, basis) -> None:
    path = BSPLINE[basis]
    with h5py.File(path, "r") as f:
        assert f[f"{REG}/GRID"].attrs["BASIS"].tolist() == [basis.encode()]
    r = opened(path)
    g = r.particles.grid()
    assert g.basis == basis
    np.testing.assert_array_equal(g.num_cells, [4, 4])
    # Only BASIS differs from the GIMP fixture: every value reads as h5py reads it.
    raw = _raw(path)
    s = r.particles.get(component="POSITION", stage=_particle_stage_id(r))
    np.testing.assert_array_equal(s.values, raw["DATA"]["POSITION"])
    np.testing.assert_array_equal(s.offsets, raw["OFFSETS"])


@pytest.mark.parametrize("basis", ["linear", "bspline2", "bspline3"])
def test_every_contract_2_basis_reads_on_an_edited_copy(copy_of, opened, basis) -> None:
    p = copy_of(PARTICLES, f"{basis}.ladruno")
    _set_basis(p, basis)
    assert opened(p).particles.grid().basis == basis


@pytest.mark.parametrize("basis", ["bspline4", "bspline", "BSPLINE2", "quadratic"])
def test_a_basis_outside_contract_2_refuses_the_stage_by_name(copy_of, opened, basis) -> None:
    p = copy_of(PARTICLES, f"{basis}.ladruno")
    _set_basis(p, basis)
    r = opened(p)
    match = (rf"GRID: BASIS is '{basis}'; Contract 2 allows "
             r"gimp \| linear \| bspline2 \| bspline3 \(C2-GRID\)")
    for access in (r.particles.grid, lambda: r.particles.get(component="POSITION")):
        with pytest.raises(ParticleFormatError, match=match):
            access()


# --- the §2.6 field catalogue, and PORE_PRESSURE (fork WP-232l, P12; #3121) --------------
#
# The oracle is the fork's checker at the pin ``ladruno-v20261123b`` (``3eb195bbc``),
# vendored in ``tests/fixtures/mpm/fork_3eb195bbc/``: its ``PARTICLE_FIELDS`` refuses a
# reserved field whose COMPONENTS differ from the catalogue's (C2-CAT, an empty one too),
# and checks no other name. ``particles_pore_pressure_v1m2.ladruno`` is
# ``particles_v1m2.ladruno`` with a PORE_PRESSURE group added in the writer's encoding
# (``make_pore_pressure_fixture.py``); its values are ``GAMMA_W * (Y_TOP - y)`` of each
# row's position. The viewer's ``ladrunoParticles.test.ts`` makes the same cases.

PORE = MPM / "particles_pore_pressure_v1m2.ladruno"
PIN = MPM / "fork_3eb195bbc"


def _module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pin_format():
    """The fork's ``ladruno_format`` at the pin, with ``sys.path`` / ``sys.modules`` restored."""
    import importlib
    import sys

    names = ("ladruno_format", "ladruno_basis")
    saved_path = list(sys.path)
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules.pop(n, None)
    sys.path.insert(0, str(PIN))
    try:
        yield importlib.import_module("ladruno_format")
    finally:
        sys.path[:] = saved_path
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def _put_field(f: h5py.File, name: str, components: str, data: np.ndarray) -> None:
    """Write ``ON_PARTICLES/<name>`` (replacing one there) with ``components`` and ``data``."""
    on = f[ON_P]
    if name in on:
        del on[name]
    g = on.create_group(name)
    g.attrs["DISPLAY_NAME"] = name
    g.attrs["COMPONENTS"] = components          # variable-length: "" is writable
    g.attrs["TYPE"] = np.array([0], dtype="<i4")
    g.attrs["DATA_TYPE"] = np.array([0], dtype="<i4")
    g.attrs["PARTITION_REDUCTION"] = "NONE"
    g.create_dataset("DATA", data=data)


def _rows(f: h5py.File) -> int:
    return int(f[f"{ON_P}/ID"].shape[0])


def _as_3d(f: h5py.File) -> None:
    """The stage made 3-D: SPATIAL_DIM 3, a 3-D grid, no thickness, POSITION alone
    (``x, y, 0.25``): only what a 3-D stage must hold for both readers."""
    f["INFO"].attrs["SPATIAL_DIM"] = np.array([3], dtype="<i4")
    grid = f[f"{REG}/GRID"].attrs
    grid["ORIGIN"] = np.zeros(3, dtype="<f8")
    grid["CELL_SIZE"] = np.full(3, 0.5, dtype="<f8")
    grid["NUM_CELLS"] = np.full(3, 4, dtype="<i4")
    del f[REG].attrs["PLANE_STRAIN_THICKNESS"]
    pos = f[f"{ON_P}/POSITION/DATA"][...]
    for name in [n for n in f[ON_P] if isinstance(f[ON_P][n], h5py.Group)]:
        del f[ON_P][name]
    _put_field(f, "POSITION", "x,y,z",
               np.column_stack([pos, np.full(len(pos), 0.25)]).astype("<f8"))


def test_the_catalogue_is_the_pins(pin_format) -> None:
    from apeGmsh.results.readers._ladruno_particles import PARTICLE_FIELDS

    assert PARTICLE_FIELDS == pin_format.PARTICLE_FIELDS
    assert PARTICLE_FIELDS["PORE_PRESSURE"] == ("p", "p")
    assert set(_RESERVED) == set(PARTICLE_FIELDS), "the C2-CAT sweep below covers every name"


def test_the_pins_validator_passes_the_pore_pressure_fixture(pin_format) -> None:
    assert pin_format.validate(str(PORE)) == []


def test_the_fixture_is_the_generators_closed_form() -> None:
    gen = _module(MPM / "make_pore_pressure_fixture.py", "_make_pore_pressure_fixture")
    with h5py.File(PORE, "r") as f:
        g = f[f"{ON_P}/PORE_PRESSURE"]
        for k, v in gen.TEXT.items():
            assert g.attrs[k].tolist() == [v.encode()], k
        assert "DESCRIPTION" not in g.attrs
        for k, v in gen.INTS.items():
            assert g.attrs[k].dtype == np.dtype("<i4") and g.attrs[k].tolist() == [v], k
        np.testing.assert_array_equal(
            g["DATA"][...], gen.pressure(f[f"{ON_P}/POSITION/DATA"][...]))


def test_pore_pressure_reads_the_stored_values(opened) -> None:
    with h5py.File(PORE, "r") as f:
        raw = f[f"{ON_P}/PORE_PRESSURE/DATA"][...]
        y = f[f"{ON_P}/POSITION/DATA"][:, 1]
    r = opened(PORE)
    assert "PORE_PRESSURE/p" in r.particles.components()
    s = r.particles.get(component="PORE_PRESSURE/p")
    np.testing.assert_array_equal(np.asarray(s.values).reshape(-1), raw.reshape(-1))
    whole = r.particles.get(component="PORE_PRESSURE")
    np.testing.assert_array_equal(whole.values, raw)
    # compression positive, growing with depth: the base reads more than the top
    v = raw.reshape(-1)
    assert v[np.argmin(y)] > v[np.argmax(y)] > 0


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", ["<f8", "<f4"])
def test_pore_pressure_p_reads_in_2d_and_3d(copy_of, opened, dim, dtype) -> None:
    p = copy_of(PARTICLES, f"pore_{dim}d_{dtype[1:]}.ladruno")
    with h5py.File(p, "r+") as f:
        if dim == 3:
            _as_3d(f)
        values = np.linspace(0.0, 1.0, _rows(f)).reshape(-1, 1).astype(dtype)
        _put_field(f, "PORE_PRESSURE", "p", values)
    s = opened(p).particles.get(component="PORE_PRESSURE/p")
    np.testing.assert_array_equal(np.asarray(s.values).reshape(-1), values.reshape(-1))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("comps", ["pw", "p,q", ""], ids=["pw", "p,q", "empty"])
def test_pore_pressure_with_other_components_is_c2_cat(copy_of, opened, dim, comps) -> None:
    p = copy_of(PARTICLES, f"pore_bad_{dim}d.ladruno")
    with h5py.File(p, "r+") as f:
        if dim == 3:
            _as_3d(f)
        ncol = max(1, len(comps.split(",")))
        _put_field(f, "PORE_PRESSURE", comps, np.zeros((_rows(f), ncol)))
    r = opened(p)
    match = (rf"ON_PARTICLES/PORE_PRESSURE: COMPONENTS is '{comps}'; the reserved "
             r"PORE_PRESSURE is 'p' \(C2-CAT")
    for access in (r.particles.components, lambda: r.particles.get(component="POSITION")):
        with pytest.raises(ParticleFormatError, match=match):
            access()


def test_p_over_two_columns_is_c2_comp(copy_of, opened) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        _put_field(f, "PORE_PRESSURE", "p", np.zeros((_rows(f), 2)))
    with pytest.raises(ParticleFormatError,
                       match=r"PORE_PRESSURE/DATA has 2 columns but COMPONENTS names 1 \(C2-COMP\)"):
        opened(p).particles.components()


def test_nan_in_pore_pressure_reads_as_stored(copy_of, opened) -> None:
    """C2-FINITE stays the validator's check: the reader reads what is stored."""
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        values = np.arange(_rows(f), dtype="<f8").reshape(-1, 1)
        values[3, 0] = np.nan
        _put_field(f, "PORE_PRESSURE", "p", values)
    s = opened(p).particles.get(component="PORE_PRESSURE/p")
    np.testing.assert_array_equal(np.asarray(s.values).reshape(-1), values.reshape(-1))


def test_a_file_without_pore_pressure_reads(opened) -> None:
    r = opened(PARTICLES)
    assert not any(c.startswith("PORE_PRESSURE/") for c in r.particles.components())
    assert r.particles.get(component="POSITION").n_outputs == 3


@pytest.mark.parametrize("dim, comps, ok", [
    (2, "Ax,Ay", True), (3, "Ax,Ay,Az", True),
    (2, "Ax,Ay,Az", False), (3, "Ax,Ay", False), (2, "ax,ay", False),
])
def test_acceleration_is_reserved(copy_of, opened, dim, comps, ok) -> None:
    p = copy_of(PARTICLES, f"acc_{dim}d.ladruno")
    with h5py.File(p, "r+") as f:
        if dim == 3:
            _as_3d(f)
        _put_field(f, "ACCELERATION", comps,
                   np.ones((_rows(f), len(comps.split(","))), dtype="<f8"))
    r = opened(p)
    if ok:
        assert r.particles.get(component="ACCELERATION").values.shape[1] == dim
    else:
        with pytest.raises(ParticleFormatError,
                           match=r"ON_PARTICLES/ACCELERATION: COMPONENTS is .*\(C2-CAT"):
            r.particles.components()


def test_stress33_in_3d_is_c2_cat(copy_of, opened) -> None:
    p = copy_of(PARTICLES, "stress33_3d.ladruno")
    with h5py.File(p, "r+") as f:
        _as_3d(f)
        _put_field(f, "STRESS33", "sigma33", np.zeros((_rows(f), 1)))
    with pytest.raises(ParticleFormatError,
                       match=r"the reserved STRESS33 is refused in 3-D \(C2-CAT"):
        opened(p).particles.components()


def test_a_3d_stage_reads_its_3d_position(copy_of, opened) -> None:
    p = copy_of(PARTICLES, "plain_3d.ladruno")
    with h5py.File(p, "r+") as f:
        _as_3d(f)
    assert opened(p).particles.get(component="POSITION").values.shape[1] == 3


_RESERVED = ("POSITION", "VELOCITY", "ACCELERATION", "STRESS", "STRESS33", "STRAIN",
             "VOLUME", "GIMP_HALF_LENGTH", "DEFORMATION_GRADIENT", "PORE_PRESSURE")


@pytest.mark.parametrize("name", _RESERVED)
def test_every_reserved_field_with_other_components_is_c2_cat(copy_of, opened, name) -> None:
    p = copy_of(PARTICLES, f"cat_{name}.ladruno")
    with h5py.File(p, "r+") as f:
        _put_field(f, name, "q", np.zeros((_rows(f), 1)))
    with pytest.raises(ParticleFormatError,
                       match=rf"ON_PARTICLES/{name}: COMPONENTS is 'q'; the reserved {name} .*\(C2-CAT"):
        opened(p).particles.components()


@pytest.mark.parametrize("name, comps", [
    ("PORE_PRESSURE_X", "pw"), ("PORE_PRESSURE_X", "p,q"), ("HEAT", "T"),
])
def test_an_unreserved_name_reads_as_the_file_spells_it(copy_of, opened, name, comps) -> None:
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        ncol = len(comps.split(","))
        values = np.arange(_rows(f) * ncol, dtype="<f8").reshape(-1, ncol)
        _put_field(f, name, comps, values)
    r = opened(p)
    assert [c for c in r.particles.components() if c.startswith(f"{name}/")] == [
        f"{name}/{c}" for c in comps.split(",")]
    np.testing.assert_array_equal(r.particles.get(component=name).values, values)


@pytest.mark.parametrize("value, ok", [(None, False), (1, False), (b"2", True)],
                         ids=["absent", "1", "ascii-2"])
def test_spatial_dim_must_be_2_or_3_for_a_reserved_field(copy_of, opened, value, ok) -> None:
    """The catalogue is read at INFO/SPATIAL_DIM, as the fork's checker and the viewer
    read it: absent or 1 is C2-NDM (never guessed from the node coordinates); a value
    reads by the file-level rule (an ASCII "2" is 2, #2244). A value that rule refuses
    (4) refuses the file at open, before any particle read."""
    p = copy_of(PARTICLES)
    with h5py.File(p, "r+") as f:
        del f["INFO"].attrs["SPATIAL_DIM"]
        if value is not None:
            f["INFO"].attrs["SPATIAL_DIM"] = (
                np.array([value]) if isinstance(value, bytes)
                else np.array([value], dtype="<i4"))
    r = opened(p)
    if ok:
        assert "POSITION/x" in r.particles.components()
    else:
        with pytest.raises(ParticleFormatError, match=r"INFO: SPATIAL_DIM .*\(C2-NDM\)"):
            r.particles.components()
