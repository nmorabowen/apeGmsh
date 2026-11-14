"""The Contract-1 particle-set writer and the ADR 0134 D1 write rule.

Contract 1 is the fork's (``Ladruno_implementation/232a_mpm_contracts.md`` section 1):
``FORMAT = "ladruno-mpm-particles"``, integer ``FORMAT_VERSION = 1``. This module writes it
and provides the helpers the deck emitters call (chain M, link M3):

* :func:`particle_set_path` -- ``<deck stem>.particles.h5`` beside the deck;
* :func:`check_overwrite` -- replace only a Contract-1 file an apeGmsh emit wrote;
* :func:`preflight_particle_set` -- every refusal, before anything is written, so a deck
  emitter can check the set and its deck together ("together or not at all");
* :func:`write_particle_set` -- a temp file in the same folder, then an atomic replace;
* :func:`tcl_particles_line` / :func:`py_particles_line` -- the deck line, anchored to the
  deck's own folder, verbatim as D1 gives it.

``_artifact_policy`` does not govern this file: it needs ``/meta`` zone keys a Contract-1
file does not have, and it never replaces a file (ADR 0134 D1).
"""
from __future__ import annotations

import math
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from ._seeding import SeededParticles

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "GENERATOR",
    "PARTICLES_SUFFIX",
    "ParticleSetError",
    "ParticleSetOverwriteRefused",
    "PreparedParticleSet",
    "Provenance",
    "Units",
    "check_overwrite",
    "particle_set_path",
    "preflight_particle_set",
    "py_particles_line",
    "py_particles_path_expr",
    "tcl_particles_line",
    "write_particle_set",
]

FORMAT = "ladruno-mpm-particles"
FORMAT_VERSION = 1
GENERATOR = "apeGmsh"
PARTICLES_SUFFIX = ".particles.h5"
VOIGT_ORDER = {2: "11,22,12", 3: "11,22,33,12,23,13"}

_UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|unknown)$")
#: Characters a particle-set file name may not carry: braces and backslash break the
#: brace-quoted Tcl word, a double quote or backslash the Python string literal.
_UNSAFE_NAME = re.compile(r'[{}\\"\x00-\x1f\x7f]')
_INT32_MAX = int(np.iinfo(np.int32).max)
#: Per-particle datasets are chunked along axis 0 at about 1 MiB (contract section 0).
_CHUNK_BYTES = 1 << 20


class ParticleSetError(ValueError):
    """A particle set (or its write) the Contract-1 writer refuses."""


class ParticleSetOverwriteRefused(ParticleSetError):
    """The particle-set path holds a file this writer may not replace (ADR 0134 D1)."""


@dataclass(frozen=True)
class Provenance:
    """Contract 1 ``PROVENANCE``, from the emitting session's in-memory lineage."""

    session_id: str
    apegmsh_version: str
    apegmsh_commit: str
    fem_hash: str | None = None
    model_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not _UUID4.match(self.session_id):
            raise ParticleSetError(f"Provenance.session_id must be a canonical uuid4, got "
                                   f"{self.session_id!r}")
        if not isinstance(self.apegmsh_version, str) or not self.apegmsh_version:
            raise ParticleSetError("Provenance.apegmsh_version must be a non-empty string")
        if not isinstance(self.apegmsh_commit, str) or not _COMMIT.match(self.apegmsh_commit):
            raise ParticleSetError(
                f"Provenance.apegmsh_commit must be 40 lowercase hex characters or "
                f"'unknown', got {self.apegmsh_commit!r}")
        for name in ("fem_hash", "model_hash"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or not v):
                raise ParticleSetError(f"Provenance.{name} must be a non-empty string or "
                                       f"None, got {v!r}")


@dataclass(frozen=True)
class Units:
    """The producer's consistent unit system (contract section 0: no conversion)."""

    length: str
    force: str
    time: str
    system: str | None = None

    def __post_init__(self) -> None:
        for name in ("length", "force", "time"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v.strip():
                raise ParticleSetError(f"Units.{name} must be a non-empty unit symbol, "
                                       f"got {v!r}")
        if self.system is not None and (not isinstance(self.system, str) or not self.system):
            raise ParticleSetError(f"Units.system must be a non-empty id or None, got "
                                   f"{self.system!r}")


@dataclass(frozen=True)
class PreparedParticleSet:
    """A particle set that passed every refusal; :func:`write_particle_set` writes it."""

    path: Path
    seeded: SeededParticles
    material_tag: NDArray[np.int32]
    units: Units
    provenance: Provenance


# ---------------------------------------------------------------------------
# the write rule (ADR 0134 D1)
# ---------------------------------------------------------------------------

def particle_set_path(deck_path: str | os.PathLike[str]) -> Path:
    """``<deck stem>.particles.h5`` in the deck's folder: ``runs/case1.tcl`` gives
    ``runs/case1.particles.h5``."""
    deck = Path(deck_path)
    if not deck.stem or deck.name in (".", ".."):
        raise ParticleSetError(f"deck path {str(deck_path)!r} has no file stem")
    return deck.with_name(deck.stem + PARTICLES_SUFFIX)


def _attr_str(attrs: Any, name: str) -> str | None:
    """A string attribute in any encoding (scalar or 1-element, fixed or vlen)."""
    if name not in attrs:
        return None
    v = attrs[name]
    if isinstance(v, np.ndarray):
        if v.size != 1:
            return None
        v = v.reshape(-1)[0]
    if isinstance(v, (bytes, np.bytes_)):
        try:
            return bytes(v).decode("utf-8")
        except UnicodeDecodeError:
            return None
    return v if isinstance(v, str) else None


def check_overwrite(path: str | os.PathLike[str]) -> None:
    """Refuse, by name, an existing file at ``path`` that is not an apeGmsh Contract-1 file.

    A missing path passes. An existing file passes only when its root carries
    ``FORMAT = "ladruno-mpm-particles"`` and ``GENERATOR = "apeGmsh"``.
    """
    p = Path(path)
    if not os.path.lexists(p):
        return
    if not p.is_file():
        raise ParticleSetOverwriteRefused(
            f"refusing to write the particle set: {str(p)!r} exists and is not a file")
    try:
        with h5py.File(p, "r") as f:
            fmt = _attr_str(f.attrs, "FORMAT")
            gen = _attr_str(f.attrs, "GENERATOR")
    except OSError as exc:
        raise ParticleSetOverwriteRefused(
            f"refusing to overwrite {str(p)!r}: it is not an HDF5 particle set "
            f"({exc})") from exc
    if fmt != FORMAT or gen != GENERATOR:
        raise ParticleSetOverwriteRefused(
            f"refusing to overwrite {str(p)!r}: it is not a particle set an apeGmsh emit "
            f"wrote (FORMAT={fmt!r}, GENERATOR={gen!r}; replaceable only with "
            f"FORMAT={FORMAT!r} and GENERATOR={GENERATOR!r}). Move or delete it, or emit "
            f"the deck under another name")


def _safe_name(particle_path: str | os.PathLike[str]) -> str:
    name = Path(particle_path).name
    if not name.endswith(PARTICLES_SUFFIX) or name == PARTICLES_SUFFIX:
        raise ParticleSetError(f"{name!r} is not a '<stem>{PARTICLES_SUFFIX}' file name")
    if _UNSAFE_NAME.search(name):
        raise ParticleSetError(
            f"particle-set file name {name!r} carries a brace, backslash, double quote or "
            f"control character, which the deck line cannot quote; rename the deck")
    return name


def tcl_particles_line(particle_path: str | os.PathLike[str]) -> str:
    """``mpm particles -file [file join [file dirname [info script]] {<name>}]``."""
    name = _safe_name(particle_path)
    return f"mpm particles -file [file join [file dirname [info script]] {{{name}}}]"


def py_particles_path_expr(particle_path: str | os.PathLike[str]) -> str:
    """``os.path.join(os.path.dirname(os.path.abspath(__file__)), "<name>")``."""
    name = _safe_name(particle_path)
    return f'os.path.join(os.path.dirname(os.path.abspath(__file__)), "{name}")'


def py_particles_line(particle_path: str | os.PathLike[str]) -> str:
    """The openseespy call, in the py emitter's ``ops.X(...)`` form; the deck needs
    ``import os``."""
    return f"ops.mpm('particles', '-file', {py_particles_path_expr(particle_path)})"


# ---------------------------------------------------------------------------
# preflight and write
# ---------------------------------------------------------------------------

def preflight_particle_set(
    path: str | os.PathLike[str],
    seeded: SeededParticles,
    *,
    material_tags: Mapping[str, int],
    units: Units,
    provenance: Provenance,
) -> PreparedParticleSet:
    """Run every refusal the writer has, and write nothing.

    ``material_tags`` maps each body name to its ``nDMaterial`` tag (ADR 0134 D6: the
    bridge's ``TagAllocator`` result); it must name exactly the seeded bodies.
    """
    if not isinstance(seeded, SeededParticles):
        raise ParticleSetError(f"seeded must be SeededParticles, got {type(seeded).__name__}")
    if not isinstance(units, Units):
        raise ParticleSetError(f"units must be Units, got {type(units).__name__}")
    if not isinstance(provenance, Provenance):
        raise ParticleSetError(f"provenance must be Provenance, got "
                               f"{type(provenance).__name__}")
    p = Path(path)
    _safe_name(p)
    check_overwrite(p)

    names = seeded.body_names
    missing = [b for b in names if b not in material_tags]
    extra = sorted(str(k) for k in material_tags if k not in names)
    if missing or extra:
        raise ParticleSetError(
            f"material_tags must name exactly the bodies {list(names)}; missing "
            f"{missing}, unknown {extra}")
    tags: list[int] = []
    for b in names:
        t = material_tags[b]
        if isinstance(t, bool) or not isinstance(t, (int, np.integer)) \
                or not 0 < int(t) <= _INT32_MAX:
            raise ParticleSetError(f"body {b!r}: material tag must be an integer in "
                                   f"[1, {_INT32_MAX}], got {t!r}")
        tags.append(int(t))
    if seeded.body_id.size and not (0 <= int(seeded.body_id.min())
                                    and int(seeded.body_id.max()) < len(names)):
        raise ParticleSetError("BODY_ID names a body outside BODIES/ID (C1-BODYREF)")
    material_tag = np.asarray(tags, dtype=np.int32)[seeded.body_id]

    count = seeded.num_particles
    ndm = seeded.ndm
    if count < 1:
        raise ParticleSetError("a particle set needs at least one particle (C1-N)")
    if seeded.position.shape != (count, ndm) or seeded.gimp_half_length.shape != (count, ndm):
        raise ParticleSetError("POSITION and GIMP_HALF_LENGTH must be (N, ndm)")
    for name, arr in (("POSITION", seeded.position), ("VOLUME", seeded.volume),
                      ("DENSITY", seeded.density),
                      ("GIMP_HALF_LENGTH", seeded.gimp_half_length)):
        if not np.all(np.isfinite(arr)):
            raise ParticleSetError(f"{name} has non-finite values (C1-FINITE)")
    if np.any(seeded.volume <= 0.0):
        raise ParticleSetError("VOLUME must be > 0; a zero or negative volume is never "
                               "written (C1-VOL)")
    if np.any(seeded.density <= 0.0):
        raise ParticleSetError("DENSITY must be > 0 (C1-MASS)")
    if ndm == 2:
        if seeded.thickness is None or not math.isfinite(seeded.thickness) \
                or seeded.thickness <= 0.0:
            raise ParticleSetError("a 2-D set needs a finite plane-strain thickness > 0")
    elif seeded.thickness is not None:
        raise ParticleSetError("a 3-D set takes no plane-strain thickness (C1-THK)")
    return PreparedParticleSet(path=p, seeded=seeded, material_tag=material_tag,
                               units=units, provenance=provenance)


def write_particle_set(prepared: PreparedParticleSet) -> str:
    """Write the prepared set to its path and return its new ``SET_ID``.

    The file is written to a temporary name in the same folder and moved into place, so a
    failure leaves no half-written particle set and an existing file untouched.
    """
    if not isinstance(prepared, PreparedParticleSet):
        raise ParticleSetError("write_particle_set takes the result of "
                               "preflight_particle_set")
    path = prepared.path
    check_overwrite(path)  # again: the folder may have changed since the preflight
    set_id = str(uuid.uuid4())
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with h5py.File(tmp, "w", track_order=True) as f:
            _write_contents(f, prepared, set_id)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return set_id


def _per_particle(group: h5py.Group, name: str, data: NDArray[Any]) -> None:
    row_bytes = max(1, int(data.dtype.itemsize * int(np.prod(data.shape[1:], dtype=np.int64))))
    rows = max(1, min(data.shape[0], _CHUNK_BYTES // row_bytes))
    group.create_dataset(name, data=data, chunks=(rows,) + data.shape[1:],
                         compression="gzip", compression_opts=1, shuffle=True)


def _write_contents(f: h5py.File, prepared: PreparedParticleSet, set_id: str) -> None:
    s = prepared.seeded
    ndm = s.ndm
    count = s.num_particles
    f.attrs["FORMAT"] = FORMAT
    f.attrs["FORMAT_VERSION"] = np.int32(FORMAT_VERSION)
    f.attrs["GENERATOR"] = GENERATOR
    f.attrs["SET_ID"] = set_id
    f.attrs["CREATED_UTC"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    f.attrs["NDM"] = np.int32(ndm)
    f.attrs["NUM_PARTICLES"] = np.int64(count)
    if s.thickness is not None:  # preflight: present iff ndm == 2
        f.attrs["PLANE_STRAIN_THICKNESS"] = np.float64(s.thickness)
    f.attrs["UNIT_LENGTH"] = prepared.units.length
    f.attrs["UNIT_FORCE"] = prepared.units.force
    f.attrs["UNIT_TIME"] = prepared.units.time
    if prepared.units.system is not None:
        f.attrs["UNIT_SYSTEM"] = prepared.units.system
    f.attrs["MASS_SOURCE"] = "density"
    f.attrs["STRESS_SIGN"] = "tension-positive"
    f.attrs["VOIGT_ORDER"] = VOIGT_ORDER[ndm]

    grid = f.create_group("GRID")
    grid.attrs["ORIGIN"] = np.asarray(s.grid.origin, dtype="<f8")
    grid.attrs["CELL_SIZE"] = np.asarray(s.grid.cell_size, dtype="<f8")
    grid.attrs["NUM_CELLS"] = np.asarray(s.grid.num_cells, dtype="<i4")
    grid.attrs["BASIS"] = s.grid.basis

    bodies = f.create_group("BODIES")
    bodies.create_dataset("ID", data=np.arange(len(s.body_names), dtype="<i4"))
    bodies.create_dataset("NAME", data=list(s.body_names), dtype=h5py.string_dtype())

    prov = f.create_group("PROVENANCE")
    pv = prepared.provenance
    prov.attrs["apegmsh_version"] = pv.apegmsh_version
    prov.attrs["apegmsh_commit"] = pv.apegmsh_commit
    prov.attrs["session_id"] = pv.session_id
    if pv.fem_hash is not None:
        prov.attrs["fem_hash"] = pv.fem_hash
    if pv.model_hash is not None:
        prov.attrs["model_hash"] = pv.model_hash

    seeding = f.create_group("SEEDING")
    seeding.attrs["METHOD"] = "lattice"
    seeding.attrs["PARTICLES_PER_CELL"] = np.full(ndm, s.n, dtype="<i4")

    parts = f.create_group("PARTICLES")
    _per_particle(parts, "ID", np.arange(count, dtype="<i8"))
    _per_particle(parts, "POSITION", np.ascontiguousarray(s.position, dtype="<f8"))
    _per_particle(parts, "VOLUME", np.ascontiguousarray(s.volume, dtype="<f8"))
    _per_particle(parts, "DENSITY", np.ascontiguousarray(s.density, dtype="<f8"))
    _per_particle(parts, "GIMP_HALF_LENGTH",
                  np.ascontiguousarray(s.gimp_half_length, dtype="<f8"))
    _per_particle(parts, "BODY_ID", np.ascontiguousarray(s.body_id, dtype="<i4"))
    _per_particle(parts, "MATERIAL_TAG", np.ascontiguousarray(prepared.material_tag,
                                                              dtype="<i4"))
