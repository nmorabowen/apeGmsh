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
