"""M2-S0: the fork's MPM fixtures, and what today's readers do with them (ADR 0134 D3).

``tests/fixtures/mpm/`` holds byte-identical copies of the fork's contract fixtures and
executable validators (see its README). The validators patch ``sys.path`` from their own
location, so they are loaded by putting that directory on ``sys.path`` here, never by
editing the copies.

These tests pin the behaviour **before** the particle reader lands:

1. the vendored validators accept the fixtures;
2. ``Results.from_ladruno`` opens the mixed fixture, reads its FE stage, and exposes
   nothing particle-shaped;
3. ``Results.from_ladruno`` on the all-particle fixture raises the "no MODEL group" error.

Pin 3 is meant to be INVERTED by the Python reader slice of M2: once particle-only files
open, replace it with the new contract.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from apeGmsh.results import Results

MPM = Path(__file__).resolve().parents[1] / "fixtures" / "mpm"
PARTICLE_SET = MPM / "particle_set_v1.h5"
PARTICLES = MPM / "particles_v1m2.ladruno"
MIXED = MPM / "particles_mixed_v1m2.ladruno"

_VENDORED = ("mpm_contracts", "ladruno_format", "ladruno_basis")


@pytest.fixture(scope="module")
def contracts():
    """The vendored ``mpm_contracts`` module, with ``sys.path``/``sys.modules`` restored."""
    saved_path = list(sys.path)
    saved_mods = {n: sys.modules.get(n) for n in _VENDORED}
    for n in _VENDORED:
        sys.modules.pop(n, None)
    sys.path.insert(0, str(MPM))
    try:
        yield importlib.import_module("mpm_contracts")
    finally:
        sys.path[:] = saved_path
        for n, m in saved_mods.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


# --- pin 1: the fixtures pass their own validators -------------------------------------

@pytest.mark.parametrize("path", [PARTICLES, MIXED], ids=lambda p: p.name)
def test_particle_results_validate(contracts, path: Path) -> None:
    assert contracts.validate_particle_results(str(path)) == []


def test_particle_set_validates(contracts) -> None:
    assert contracts.validate_particle_set(str(PARTICLE_SET)) == []


# --- pins 2 and 3: today's Python reader ------------------------------------------------

def test_mixed_fixture_opens_and_fe_stage_reads_without_particles() -> None:
    r = Results.from_ladruno(MIXED)
    try:
        assert r.stages, "the FE stage must be listed"
        stage = r.stage(r.stages[0].id)
        assert stage.nodes.available_components(), "the FE stage must expose nodal results"
        # Nothing particle-shaped is exposed by today's reader.
        for obj in (r, stage):
            assert not [n for n in dir(obj) if "particle" in n.lower()]
    finally:
        r.close()


def test_particle_only_fixture_is_refused_for_want_of_a_model_group() -> None:
    # INVERT ME: the Python reader slice of M2 makes a particle-only file open.
    # This pins the refusal as it stands today (ADR 0134 D3 "Before M2").
    with pytest.raises(ValueError, match=r"carries no MODEL group .*Pass model_h5="):
        Results.from_ladruno(PARTICLES)
