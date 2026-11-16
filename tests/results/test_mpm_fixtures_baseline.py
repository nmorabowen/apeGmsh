"""M2-S0: the fork's MPM fixtures, and what today's readers do with them (ADR 0134 D3).

``tests/fixtures/mpm/`` holds byte-identical copies of the fork's contract fixtures and
executable validators (see its README). The validators patch ``sys.path`` from their own
location, so they are loaded by putting that directory on ``sys.path`` here, never by
editing the copies.

These tests pin:

1. the vendored validators accept the fixtures;
2. ``Results.from_ladruno`` opens the mixed fixture and reads its FE stage as before;
3. ``Results.from_ladruno`` opens the all-particle fixture without ``model_h5=``
   (ADR 0134 D3, inverted by M2-S2 #2981: before M2 it raised the "no MODEL group"
   error);
4. the fork's torn file (M2-S6 #2990, fork e30872e34): the closed-file validator
   refuses it (C2-OFF), the reference ``committed_outputs`` counts 3 outputs, and the
   Python reader reads exactly that many;
5. the torn-stage fixtures (M2-S7 #2991): each whole stage counts 3 outputs, and
   ``LadrunoReader``, which does not know the open step until #3003, refuses the torn
   stage by stage (C2-REG).

The particle values themselves are checked in ``test_ladruno_particles.py``.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest

from apeGmsh.results import Results
from apeGmsh.results.readers._ladruno_particles import ParticleOutputsTruncatedWarning

MPM = Path(__file__).resolve().parents[1] / "fixtures" / "mpm"
PARTICLE_SET = MPM / "particle_set_v1.h5"
PARTICLES = MPM / "particles_v1m2.ladruno"
MIXED = MPM / "particles_mixed_v1m2.ladruno"
TORN = MPM / "particles_torn_v1m2.ladruno"

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

def test_mixed_fixture_opens_and_fe_stage_reads() -> None:
    r = Results.from_ladruno(MIXED)
    try:
        assert r.stages, "the FE stage must be listed"
        stage = r.stage(r.stages[0].id)
        assert stage.nodes.available_components(), "the FE stage must expose nodal results"
        # The FE stage holds no particles; the particle stage is the second one.
        assert r.particles.stages() == [r.stages[1].id]
    finally:
        r.close()


def test_particle_only_fixture_opens_without_a_model() -> None:
    # Pin 3, inverted by M2-S2 (#2981): an all-particle file opens without
    # model_h5= (ADR 0134 D3) and has no FEM.
    r = Results.from_ladruno(PARTICLES)
    try:
        assert r.fem is None
        assert r.particles.get(component="POSITION").n_outputs == 3
    finally:
        r.close()


# --- pin 4: the torn file (contract 232a §2.5, M2-S6 #2990) ------------------------------

def test_torn_fixture_is_refused_by_the_closed_file_validator(contracts) -> None:
    # The validator checks closed files only: OFFSETS[T] = len(ID) (C2-OFF).
    errors = contracts.validate_particle_results(str(TORN))
    assert len(errors) == 1 and errors[0].startswith("C2-OFF"), errors


@pytest.mark.parametrize("path, expected", [(PARTICLES, 3), (MIXED, 3), (TORN, 3)],
                         ids=lambda x: x.name if isinstance(x, Path) else str(x))
def test_reference_committed_outputs(contracts, path: Path, expected: int) -> None:
    assert contracts.committed_outputs(str(path), stage="MODEL_STAGE[1]") == expected


@pytest.mark.parametrize("path", [PARTICLES, MIXED, TORN], ids=lambda p: p.name)
def test_python_reader_reads_the_reference_committed_outputs(contracts, path: Path) -> None:
    expected = contracts.committed_outputs(str(path), stage="MODEL_STAGE[1]")
    r = Results.from_ladruno(path)
    try:
        sid = r.particles.stages()[-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ParticleOutputsTruncatedWarning)
            assert r.particles.get(component="POSITION", stage=sid).n_outputs == expected
    finally:
        r.close()


# --- pin 5: the torn-stage fixtures (M2-S7 #2991; built by make_torn_stage_fixtures.py) ---

TORN_LAST = MPM / "particles_torn_last_stage_v1m2.ladruno"
TORN_EARLY = MPM / "particles_torn_early_stage_v1m2.ladruno"


@pytest.mark.parametrize("path, stage", [(TORN_LAST, "MODEL_STAGE[1]"),
                                         (TORN_EARLY, "MODEL_STAGE[2]")],
                         ids=lambda x: x.name if isinstance(x, Path) else x)
def test_torn_stage_fixtures_whole_stage_reads_3(contracts, path: Path, stage: str) -> None:
    assert contracts.committed_outputs(str(path), stage=stage) == 3


@pytest.mark.parametrize("path, expected", [
    (TORN_LAST, ["read 3", "ParticleStageTornError"]),
    (TORN_EARLY, ["ParticleStageTornError", "read 3"]),
], ids=lambda x: x.name if isinstance(x, Path) else "")
def test_ladruno_reader_on_the_torn_stage_fixtures(path: Path, expected: list) -> None:
    """``LadrunoReader`` opens these files plainly and does not apply the stage
    rule (it adopts the shared open in #3003): a torn registry is refused by
    stage (C2-REG). ``ParticleFile`` applies the rule
    (``test_ladruno_particles.py``)."""
    r = Results.from_ladruno(path)
    try:
        got = []
        for sid in r.particles.stages():
            try:
                got.append(f"read {r.particles.get(component='POSITION', stage=sid).n_outputs}")
            except Exception as e:  # noqa: BLE001 - the pin records which error
                got.append(type(e).__name__)
        assert got == expected
    finally:
        r.close()
