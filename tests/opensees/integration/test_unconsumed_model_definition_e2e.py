"""ADR 0051 §4 end to end — g.constraints.bc / g.masses vs the deck.

The audit at 07f757e0: a model with ``g.constraints.bc`` and
``g.masses`` resolved 174 SP + 339 mass records into the broker, and
the emitted deck carried 0 ``fix`` / 0 ``mass`` lines with no warning.
Restating them (``ops.fix`` + ``ops.mass_from_model()``) gave 58 / 339.
These tests replay that on a real mesh: the unrestated deck now warns
with the broker's counts, the restated deck stays silent and unchanged,
``ops.fix_from_model()`` emits the same fixes as the explicit
``ops.fix``, and the H5 archive (never solved) stays silent.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import UnconsumedModelDefinitionWarning, apeSees


def _box(name: str, *, partitions: int = 0):
    with apeGmsh(model_name=name, verbose=False) as g:
        g.model.geometry.add_box(0, 0, 0, 10, 10, 10, label="b")
        g.physical.add_volume("b", name="B")
        (g.model.select(dim=2).on_plane((0, 0, 0.0), (0, 0, 1), tol=1e-6)
            .to_physical("Base"))
        g.constraints.bc("Base", dofs=[1, 1, 1])
        g.masses.volume("B", density=2400.0)
        g.mesh.sizing.set_global_size(3.0)
        g.mesh.generation.generate(dim=3)
        if partitions:
            g.mesh.partitioning.partition(partitions)
        return g.mesh.queries.get_fem_data(dim=3)


@pytest.fixture(scope="module")
def fem():
    fem = _box("umd")
    assert any(r.is_homogeneous for r in fem.nodes.sp)
    assert len(fem.nodes.masses) > 0
    return fem


@pytest.fixture(scope="module")
def fem_partitioned():
    return _box("umd_p", partitions=3)


def _ops(fem):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=0.0)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    return ops


def _deck(ops, tmp_path: Path) -> list[str]:
    path = tmp_path / "deck.tcl"
    ops.tcl(str(path))
    # strip the indent so per-rank blocks compare like flat ones
    return [ln.strip() for ln in path.read_text().splitlines()]


def _lines(deck: list[str], verb: str) -> list[str]:
    return [ln for ln in deck if ln.startswith(verb + " ")]


def test_unrestated_deck_warns_with_the_broker_counts(fem, tmp_path):
    hom = [r for r in fem.nodes.sp if r.is_homogeneous]
    n_nodes = len({r.node_id for r in hom})
    with pytest.warns(UnconsumedModelDefinitionWarning) as caught:
        deck = _deck(_ops(fem), tmp_path)
    msg = str(caught[0].message)
    assert f"{len(hom)} homogeneous SP record(s) on {n_nodes} node(s)" in msg
    assert f"{len(fem.nodes.masses)} nodal mass record(s)" in msg
    # the drop itself is unchanged: the warning names it, emit stays put
    assert _lines(deck, "fix") == [] and _lines(deck, "mass") == []


def test_restated_deck_is_silent_and_unchanged(fem, tmp_path):
    n_nodes = len({r.node_id for r in fem.nodes.sp if r.is_homogeneous})
    ops = _ops(fem)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.mass_from_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnconsumedModelDefinitionWarning)
        deck = _deck(ops, tmp_path)
    assert len(_lines(deck, "fix")) == n_nodes
    assert len(_lines(deck, "mass")) == len(fem.nodes.masses)


def test_fix_from_model_matches_the_explicit_fix(fem, tmp_path):
    explicit = _ops(fem)
    explicit.fix(pg="Base", dofs=(1, 1, 1))
    explicit.mass_from_model()
    streamed = _ops(fem)
    streamed.fix_from_model()
    streamed.mass_from_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnconsumedModelDefinitionWarning)
        want = sorted(_lines(_deck(explicit, tmp_path), "fix"))
        got = sorted(_lines(_deck(streamed, tmp_path), "fix"))
    assert got == want and len(got) > 0


def test_fix_from_model_matches_the_explicit_fix_partitioned(
    fem_partitioned, tmp_path,
):
    """Materialized records ride the per-rank fix buckets: every owning
    rank gets its (idempotent) fix line, exactly as ops.fix(pg=) does."""
    explicit = _ops(fem_partitioned)
    explicit.fix(pg="Base", dofs=(1, 1, 1))
    explicit.mass_from_model()
    streamed = _ops(fem_partitioned)
    streamed.fix_from_model()
    streamed.mass_from_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnconsumedModelDefinitionWarning)
        want = sorted(_lines(_deck(explicit, tmp_path), "fix"))
        got = sorted(_lines(_deck(streamed, tmp_path), "fix"))
    assert got == want and len(got) > 0


def test_h5_archive_does_not_warn(fem, tmp_path):
    """An archival emit never solves, and the neutral zone keeps both
    record sets; mass_from_model() is refused there, so the warning's
    advice would fail — it stays silent."""
    ops = _ops(fem)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnconsumedModelDefinitionWarning)
        ops.h5(str(tmp_path / "model.h5"))
