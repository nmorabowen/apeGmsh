"""MPM particle sets (ADR 0134 D1, D2, D6): lattice seeding and the Contract-1 writer.

Internal to the bridge. The deck emitters call it once chain M link M3 adds the ``mpm``
primitive family; nothing here is public API and nothing here is wired into an emitter.

Charter P3/P9: the package imports neither gmsh nor anything of apeGmsh outside
``apeGmsh.mesh`` (and that only for typing). ``tests/opensees/mpm`` holds the proof.
"""
from __future__ import annotations

from ._particle_set import (
    FORMAT,
    FORMAT_VERSION,
    GENERATOR,
    PARTICLES_SUFFIX,
    ParticleSetError,
    ParticleSetOverwriteRefused,
    PreparedParticleSet,
    Provenance,
    Units,
    check_overwrite,
    particle_set_path,
    preflight_particle_set,
    py_particles_line,
    py_particles_path_expr,
    tcl_particles_line,
    write_particle_set,
)
from ._seeding import (
    SUPPORTED_FAMILIES,
    MPMBody,
    MPMGrid,
    MPMSeedError,
    SeededParticles,
    UniformityReport,
    seed_lattice,
)

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "GENERATOR",
    "PARTICLES_SUFFIX",
    "SUPPORTED_FAMILIES",
    "MPMBody",
    "MPMGrid",
    "MPMSeedError",
    "ParticleSetError",
    "ParticleSetOverwriteRefused",
    "PreparedParticleSet",
    "Provenance",
    "SeededParticles",
    "UniformityReport",
    "Units",
    "check_overwrite",
    "particle_set_path",
    "preflight_particle_set",
    "py_particles_line",
    "py_particles_path_expr",
    "seed_lattice",
    "tcl_particles_line",
    "write_particle_set",
]
