"""The Contract-1 writer and the ADR 0134 D1 write rule.

Oracle: the fork's own executable spec, ``validate_particle_set``, vendored byte-identical
in ``tests/fixtures/mpm/`` (slice M2-S0), and the fork fixture ``particle_set_v1.h5``.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._mpm import (
    FORMAT,
    MPMBody,
    MPMGrid,
    ParticleSetError,
    ParticleSetOverwriteRefused,
    Provenance,
    Units,
    _particle_set,
    check_overwrite,
    particle_set_path,
    preflight_particle_set,
    py_particles_line,
    py_particles_path_expr,
    seed_lattice,
    tcl_particles_line,
    write_particle_set,
)

from . import _meshes as M
from .conftest import MPM_FIXTURES

SESSION = "9b2e7c41-5d3a-4f6e-8c1b-2a3d4e5f6a7b"
PROV = Provenance(session_id=SESSION, apegmsh_version="1.2.3", apegmsh_commit="unknown")
UNITS = Units("m", "kN", "s", "kN_m_s")
FAMILY_DIM = {"tri3": 2, "quad4": 2, "tet4": 3, "hex8": 3}


def seeded_box(family: str, n: int = 2):
    d = FAMILY_DIM[family]
    nodes, ids = M.grid_nodes([2] * d, [0.25] * d, [0.75] * d)
    els = {"tri3": M.tris, "quad4": M.quads, "hex8": M.hexes}.get(family)
    elements = els(ids) if els else M.tets(ids, nodes)
    fem = M.make_fem(nodes, {"core": elements})
    grid = MPMGrid((0.0,) * d, (0.25,) * d, (4,) * d)
    return seed_lattice(fem, grid, [MPMBody("core", "core", 2.0)], n=n,
                        thickness=0.5 if d == 2 else None)


def write(path, seeded, tags=None, **kw):
    prepared = preflight_particle_set(path, seeded, material_tags=tags or {"core": 7},
                                      units=kw.pop("units", UNITS),
                                      provenance=kw.pop("provenance", PROV))
    return write_particle_set(prepared)


def listing(folder: Path) -> list[str]:
    return sorted(os.listdir(folder))


# ---------------------------------------------------------------------------
# (a) the vendored validator passes on every seeded family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", list(FAMILY_DIM))
def test_vendored_validator_passes_on_seeded_files(tmp_path, contracts, family):
    seeded = seeded_box(family)
    out = tmp_path / "case1.particles.h5"
    set_id = write(out, seeded)
    warnings: list[str] = []
    assert contracts.validate_particle_set(str(out), warnings) == []
    assert warnings == []  # lattice: VOLUME equals the GIMP domain (232a section 1.4)
    with h5py.File(out, "r") as f:
        assert f.attrs["SET_ID"] == set_id
        assert int(f.attrs["NUM_PARTICLES"]) == seeded.num_particles
        assert f.attrs["NUM_PARTICLES"].dtype == np.int64
        assert f.attrs["NDM"].dtype == np.int32
        assert ("PLANE_STRAIN_THICKNESS" in f.attrs) == (seeded.ndm == 2)
        for name, ds in f["PARTICLES"].items():
            assert ds.chunks is not None and ds.chunks[0] >= 1, name
            assert ds.compression == "gzip" and ds.compression_opts == 1, name
            assert ds.shuffle, name
        np.testing.assert_array_equal(f["PARTICLES/MATERIAL_TAG"][...], 7)
        np.testing.assert_array_equal(f["PARTICLES/ID"][...], np.arange(seeded.num_particles))
        assert set(f["PROVENANCE"].attrs) == {"apegmsh_version", "apegmsh_commit",
                                              "session_id"}
        assert f["PROVENANCE"].attrs["session_id"] == SESSION
        np.testing.assert_array_equal(f["SEEDING"].attrs["PARTICLES_PER_CELL"],
                                      [2] * seeded.ndm)


def test_validator_rejects_what_the_writer_must_never_write(tmp_path, contracts):
    """The oracle has teeth: a zero volume written behind the writer's back is C1-VOL."""
    out = tmp_path / "a.particles.h5"
    write(out, seeded_box("quad4"))
    with h5py.File(out, "r+") as f:
        f["PARTICLES/VOLUME"][0] = 0.0
    assert any(p.startswith("C1-VOL") for p in contracts.validate_particle_set(str(out)))


def test_optional_lineage_hashes_are_written_when_given(tmp_path, contracts):
    prov = Provenance(SESSION, "1.2.3", "a" * 40, fem_hash="f" * 64, model_hash="m1")
    out = tmp_path / "a.particles.h5"
    write(out, seeded_box("tri3"), provenance=prov)
    assert contracts.validate_particle_set(str(out)) == []
    with h5py.File(out, "r") as f:
        assert f["PROVENANCE"].attrs["fem_hash"] == "f" * 64
        assert f["PROVENANCE"].attrs["model_hash"] == "m1"
        assert f["PROVENANCE"].attrs["apegmsh_commit"] == "a" * 40


# ---------------------------------------------------------------------------
# D1: the path and the deck lines
# ---------------------------------------------------------------------------

def test_particle_set_path_is_the_deck_stem_beside_the_deck():
    assert particle_set_path("runs/case1.tcl") == Path("runs/case1.particles.h5")
    assert particle_set_path(Path("runs") / "case1.py") == Path("runs/case1.particles.h5")
    assert particle_set_path("slab.v2.tcl") == Path("slab.v2.particles.h5")


def test_deck_lines_are_verbatim_d1():
    p = particle_set_path("runs/case1.tcl")
    assert tcl_particles_line(p) == (
        "mpm particles -file [file join [file dirname [info script]] {case1.particles.h5}]")
    assert py_particles_path_expr(p) == (
        'os.path.join(os.path.dirname(os.path.abspath(__file__)), "case1.particles.h5")')
    assert py_particles_line(p) == (
        "ops.mpm('particles', '-file', os.path.join(os.path.dirname("
        'os.path.abspath(__file__)), "case1.particles.h5"))')
    # a space is safe inside the brace-quoted Tcl word and the Python literal
    assert "{my case.particles.h5}" in tcl_particles_line("my case.particles.h5")


@pytest.mark.parametrize("name", ["a{b.particles.h5", "a}b.particles.h5",
                                  'a"b.particles.h5', "a\nb.particles.h5", "case1.h5",
                                  ".particles.h5"])
def test_unquotable_or_foreign_names_are_refused(name):
    with pytest.raises(ParticleSetError):
        tcl_particles_line(name)
    with pytest.raises(ParticleSetError):
        py_particles_path_expr(name)


# ---------------------------------------------------------------------------
# (c) the overwrite guard
# ---------------------------------------------------------------------------

def test_guard_replaces_an_apegmsh_file(tmp_path, contracts):
    out = tmp_path / "case1.particles.h5"
    first = write(out, seeded_box("quad4", n=1))
    second = write(out, seeded_box("quad4", n=2))
    assert first != second
    with h5py.File(out, "r") as f:
        assert f.attrs["SET_ID"] == second
        assert int(f.attrs["NUM_PARTICLES"]) == seeded_box("quad4", n=2).num_particles
    assert contracts.validate_particle_set(str(out)) == []
    assert listing(tmp_path) == ["case1.particles.h5"]


def test_guard_accepts_the_fork_fixture_written_as_apegmsh():
    check_overwrite(MPM_FIXTURES / "particle_set_v1.h5")  # GENERATOR = "apeGmsh"


def _foreign_h5(path: Path, **attrs) -> None:
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v


@pytest.mark.parametrize("kind", ["user_script", "other_format", "no_attrs", "text", "fixed"])
def test_guard_refuses_a_foreign_file_by_name(tmp_path, kind):
    out = tmp_path / "case1.particles.h5"
    if kind == "user_script":
        _foreign_h5(out, FORMAT=FORMAT, GENERATOR="my_seeder.py")
    elif kind == "other_format":
        _foreign_h5(out, FORMAT="something-else", GENERATOR="apeGmsh")
    elif kind == "no_attrs":
        _foreign_h5(out)
    elif kind == "fixed":  # fixed-length bytes, still not apeGmsh's
        _foreign_h5(out, FORMAT=np.bytes_(FORMAT), GENERATOR=np.bytes_("fork"))
    else:
        out.write_text("my notes\n")
    before = out.read_bytes()
    with pytest.raises(ParticleSetOverwriteRefused, match="refusing to overwrite .*case1"):
        write(out, seeded_box("quad4"))
    assert out.read_bytes() == before
    assert listing(tmp_path) == ["case1.particles.h5"]


def test_guard_accepts_fixed_length_apegmsh_attrs(tmp_path):
    out = tmp_path / "a.particles.h5"
    _foreign_h5(out, FORMAT=np.array([np.bytes_(FORMAT)]), GENERATOR=np.bytes_("apeGmsh"))
    check_overwrite(out)


@pytest.mark.parametrize("kind", ["foreign_h5", "text"])
def test_write_rechecks_the_guard_after_the_preflight(tmp_path, kind):
    """The path was free at preflight; a foreign file appears before the write. The write
    refuses it by name and leaves it byte-identical, with no temp file left behind."""
    out = tmp_path / "case1.particles.h5"
    prepared = preflight_particle_set(out, seeded_box("quad4"), material_tags={"core": 7},
                                      units=UNITS, provenance=PROV)
    if kind == "foreign_h5":
        _foreign_h5(out, FORMAT=FORMAT, GENERATOR="my_seeder.py")
    else:
        out.write_text("my notes\n")
    before = out.read_bytes()
    with pytest.raises(ParticleSetOverwriteRefused, match="refusing to overwrite .*case1"):
        write_particle_set(prepared)
    assert out.read_bytes() == before
    assert listing(tmp_path) == ["case1.particles.h5"]


def test_guard_refuses_a_directory(tmp_path):
    out = tmp_path / "case1.particles.h5"
    out.mkdir()
    with pytest.raises(ParticleSetOverwriteRefused, match="not a file"):
        write(out, seeded_box("quad4"))


# ---------------------------------------------------------------------------
# (d) a refused set writes nothing; a failed write leaves no half file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tags,msg", [
    ({}, "missing \\['core'\\]"),
    ({"core": 1, "ghost": 2}, "unknown \\['ghost'\\]"),
    ({"core": 0}, "material tag must be an integer"),
    ({"core": True}, "material tag must be an integer"),
    ({"core": 2 ** 31}, "material tag must be an integer"),
])
def test_refused_material_tags_write_nothing(tmp_path, tags, msg):
    out = tmp_path / "case1.particles.h5"
    with pytest.raises(ParticleSetError, match=msg):
        preflight_particle_set(out, seeded_box("tri3"), material_tags=tags, units=UNITS,
                               provenance=PROV)
    assert listing(tmp_path) == []


@pytest.mark.parametrize("kw,msg", [
    ({"session_id": "not-a-uuid"}, "session_id"),
    ({"session_id": "9B2E7C41-5D3A-4F6E-8C1B-2A3D4E5F6A7B"}, "session_id"),
    ({"apegmsh_commit": "abc123"}, "apegmsh_commit"),
    ({"apegmsh_version": ""}, "apegmsh_version"),
    ({"fem_hash": ""}, "fem_hash"),
])
def test_bad_provenance_is_refused(kw, msg):
    base = {"session_id": SESSION, "apegmsh_version": "1", "apegmsh_commit": "unknown"}
    with pytest.raises(ParticleSetError, match=msg):
        Provenance(**{**base, **kw})


@pytest.mark.parametrize("kw", [{"length": ""}, {"force": " "}, {"system": ""}])
def test_bad_units_are_refused(kw):
    with pytest.raises(ParticleSetError, match="Units"):
        Units(**{"length": "m", "force": "kN", "time": "s", **kw})


def test_failed_write_leaves_no_half_file_and_keeps_the_old_one(tmp_path, monkeypatch):
    out = tmp_path / "case1.particles.h5"
    old = write(out, seeded_box("quad4", n=1))
    before = out.read_bytes()
    real = _particle_set._write_contents

    def half_then_fail(f, prepared, set_id):
        real(f, prepared, set_id)
        f["PARTICLES"].create_dataset("JUNK", data=np.zeros(3))
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(_particle_set, "_write_contents", half_then_fail)
    with pytest.raises(OSError, match="simulated"):
        write(out, seeded_box("quad4", n=2))
    assert listing(tmp_path) == ["case1.particles.h5"]
    assert out.read_bytes() == before
    with h5py.File(out, "r") as f:
        assert f.attrs["SET_ID"] == old


def test_failed_first_write_leaves_nothing(tmp_path, monkeypatch):
    def fail(f, prepared, set_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(_particle_set, "_write_contents", fail)
    with pytest.raises(RuntimeError, match="boom"):
        write(tmp_path / "case1.particles.h5", seeded_box("hex8"))
    assert listing(tmp_path) == []


def test_preflight_refuses_a_zero_volume_set(tmp_path):
    """Contract 1 refuses zero or negative volume; the writer never writes one."""
    import dataclasses

    good = seeded_box("quad4")
    vol = good.volume.copy()
    vol[3] = 0.0
    bad = dataclasses.replace(good, volume=vol)
    with pytest.raises(ParticleSetError, match="C1-VOL"):
        preflight_particle_set(tmp_path / "a.particles.h5", bad, material_tags={"core": 1},
                               units=UNITS, provenance=PROV)
    assert listing(tmp_path) == []


def test_write_takes_only_a_preflighted_set(tmp_path):
    with pytest.raises(ParticleSetError, match="preflight_particle_set"):
        write_particle_set(seeded_box("quad4"))  # type: ignore[arg-type]
