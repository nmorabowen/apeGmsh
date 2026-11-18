"""A variant row never names a seeded particle set as its model (ADR 0134 D1; #2985 finding 5).

``<stem>.particles.h5`` is a sibling of a set, as ``.geometry.h5``, ``.results.h5``,
``.user.h5`` and ``.workspace.h5`` are, so ``path_problem`` refuses it, case-insensitively,
as apeGmshViewer's ``reader/variants.ts`` ``pathProblem`` does.
"""

from __future__ import annotations

import pytest

from apeGmsh.results.user import _variants

PARTICLE_SETS = ["slab.particles.h5", "sub/slab.PARTICLES.H5", "slab.Particles.h5"]


@pytest.mark.parametrize("path", PARTICLE_SETS)
def test_a_particle_set_is_a_sibling_never_a_model(path: str) -> None:
    assert _variants.path_problem(path) == (
        "it names a sibling file (.geometry.h5, .results.h5, .user.h5, .workspace.h5 or .particles.h5)")


def test_a_model_whose_stem_is_particles_is_still_a_model() -> None:
    assert _variants.path_problem("particles.h5") is None
    assert _variants.path_problem("runs/particles.h5") is None

