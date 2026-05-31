"""FEMData cache + dirty-bit invalidation on broker mutations.

Phase 3B.2b-prep / ADR 0038 — verifies the session-level FEMData
cache behaviour that the upcoming chain-phase semantics
(Phase 3B.2c) will build on top of:

* ``g.mesh.queries.get_fem_data()`` returns the SAME FEMData object
  identity on repeated calls with no intervening broker mutation.
* Broker mutations — ``g.constraints.X`` / ``g.loads.X`` /
  ``g.masses.X`` (and ``g.node_ndf.X``) — bump the session counter
  so the next ``get_fem_data()`` re-extracts.
* The cache is keyed on the *canonical* signature only
  (``dim=None``, ``remove_orphans=False``); variant calls
  (``dim=3`` etc.) always re-extract and never poison the cache.

These tests use the full ``g`` fixture so the gmsh + extractor round
trip is exercised end-to-end.  The dimension here is intentionally
small (1-volume box, coarse mesh) to keep the tests fast — the
contract is about cache identity, not mesh content.
"""
from __future__ import annotations

import pytest


# =====================================================================
# Helpers
# =====================================================================

def _build_box(g) -> None:
    """Build a tiny single-box meshed model on the session."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label="Body")
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)
    g.node_ndf.set_default(ndf=3)


def _add_body_pg(g) -> None:
    """Register the body volume as a physical group named ``BodyVol``.

    Uses the Tier 1 ``"Body"`` label set by :func:`_build_box` —
    :meth:`PhysicalGroups.from_label` looks up entities under that
    label and adds them as a PG so the loads/masses resolver can
    match ``pg="BodyVol"`` without a live label-tier lookup.
    """
    g.physical.from_label("Body", name="BodyVol")


# =====================================================================
# 1. Repeated calls return the same FEMData identity
# =====================================================================

def test_get_fem_data_returns_cache_on_second_call(g):
    """Two ``get_fem_data()`` calls with no mutation between them
    return the same object identity."""
    _build_box(g)

    fem_a = g.mesh.queries.get_fem_data()
    fem_b = g.mesh.queries.get_fem_data()

    assert fem_a is fem_b, (
        "Repeat get_fem_data() calls without intervening mutation "
        "must return the cached FEMData (same object identity)."
    )


def test_no_mutation_no_invalidation(g):
    """Multiple ``get_fem_data()`` calls without intervening mutation
    all return the SAME object identity."""
    _build_box(g)

    fems = [g.mesh.queries.get_fem_data() for _ in range(5)]
    first = fems[0]
    for f in fems[1:]:
        assert f is first


# =====================================================================
# 2. Constraint mutation invalidates the cache
# =====================================================================

def test_constraint_mutation_invalidates_cache(g):
    """A ``g.constraints.bc(...)`` call between two get_fem_data()
    calls forces a fresh extraction.

    We use ``g.constraints.bc(...)`` instead of one of the
    master/slave constructors so we don't need to set up parts; the
    cache-bump contract is identical for every mutation method.
    """
    _build_box(g)
    # PG registration is not itself a broker mutation, so it does NOT
    # bump the FEMData counter.
    _add_body_pg(g)

    fem_a = g.mesh.queries.get_fem_data()
    # Identity-stable across no-mutation calls.
    assert g.mesh.queries.get_fem_data() is fem_a

    # Now a real broker mutation: ``g.constraints.bc(...)`` is the
    # lone direct-append path on ConstraintsComposite (it bypasses
    # ``_add_def``), so this exercises the ``bc()``-side bump hook
    # specifically.
    g.constraints.bc(pg="BodyVol", dofs=[1, 1, 1])

    fem_b = g.mesh.queries.get_fem_data()
    assert fem_b is not fem_a, (
        "Broker mutation must invalidate the cache — the second "
        "get_fem_data() should re-extract."
    )
    # And the new bc must appear in the fresh broker.
    assert len(fem_b.nodes.sp) > 0, (
        "The newly-declared bc() didn't materialise into "
        "fem.nodes.sp after re-extract."
    )


# =====================================================================
# 3. Load mutation invalidates the cache
# =====================================================================

def test_load_mutation_invalidates_cache(g):
    """A ``g.loads.X(...)`` call invalidates the cache."""
    _build_box(g)
    # The auto-PG path on apeGmsh leaves the label as a Tier 1 label
    # (not a PG); add an explicit PG that the loads/masses resolver
    # can match against without label-tier sugar.
    _add_body_pg(g)

    fem_a = g.mesh.queries.get_fem_data()

    # Body force on the box volume.  ``volume()`` goes through
    # ``_add_def`` which bumps the counter.
    g.loads.volume(pg="BodyVol", force_per_volume=(0.0, 0.0, -1.0))

    fem_b = g.mesh.queries.get_fem_data()
    assert fem_b is not fem_a, (
        "Load mutation must invalidate the cache."
    )


# =====================================================================
# 4. Mass mutation invalidates the cache
# =====================================================================

def test_mass_mutation_invalidates_cache(g):
    """A ``g.masses.X(...)`` call invalidates the cache."""
    _build_box(g)
    # The auto-PG path on apeGmsh leaves the label as a Tier 1 label
    # (not a PG); add an explicit PG that the loads/masses resolver
    # can match against without label-tier sugar.
    _add_body_pg(g)

    fem_a = g.mesh.queries.get_fem_data()

    # Point-like volumetric mass: every node of the body picks up
    # a small lumped mass.  Goes through ``_add_def``.
    g.masses.volume(pg="BodyVol", density=2400.0, reduction="lumped")

    fem_b = g.mesh.queries.get_fem_data()
    assert fem_b is not fem_a, (
        "Mass mutation must invalidate the cache."
    )


# =====================================================================
# 5. NodeNDF mutation invalidates the cache
# =====================================================================

def test_node_ndf_mutation_invalidates_cache(g):
    """A ``g.node_ndf.set(...)`` / ``set_default`` / ``clear`` call
    invalidates the cache.

    The existing :meth:`NodeNDFComposite._warn_if_post_extraction`
    will fire one ``UserWarning`` on the first post-extract mutation
    — that's the pre-existing contract and is orthogonal to the
    cache-identity check this test makes.
    """
    _build_box(g)
    # The auto-PG path on apeGmsh leaves the label as a Tier 1 label
    # (not a PG); add an explicit PG that the loads/masses resolver
    # can match against without label-tier sugar.
    _add_body_pg(g)

    fem_a = g.mesh.queries.get_fem_data()

    # Re-declare default ndf.  Triggers the historical
    # post-extract UserWarning + the new counter bump.
    with pytest.warns(UserWarning, match="get_fem_data"):
        g.node_ndf.set_default(ndf=6)

    fem_b = g.mesh.queries.get_fem_data()
    assert fem_b is not fem_a, (
        "node_ndf mutation must invalidate the cache."
    )


# =====================================================================
# 6. Variant calls (dim=, remove_orphans=) bypass the cache
# =====================================================================

def test_dim_variant_call_does_not_poison_cache(g):
    """``get_fem_data(dim=3)`` does not populate the default-signature
    cache.  A subsequent default-signature call still re-extracts
    (well — it extracts for the first time, since no default call has
    primed the cache)."""
    _build_box(g)

    fem_dim = g.mesh.queries.get_fem_data(dim=3)
    fem_default = g.mesh.queries.get_fem_data()

    # The two objects are distinct — the dim=3 call did NOT poison
    # the default-signature slot.
    assert fem_dim is not fem_default

    # Calling default again returns the cached default.
    fem_default_2 = g.mesh.queries.get_fem_data()
    assert fem_default_2 is fem_default


# =====================================================================
# 7. Cross-mutation invalidation
# =====================================================================

def test_multiple_mutations_each_invalidate(g):
    """Each mutation between cached extracts forces a fresh fem;
    cache identity must change after EACH mutation."""
    _build_box(g)
    # The auto-PG path on apeGmsh leaves the label as a Tier 1 label
    # (not a PG); add an explicit PG that the loads/masses resolver
    # can match against without label-tier sugar.
    _add_body_pg(g)

    fem_a = g.mesh.queries.get_fem_data()

    g.loads.volume(pg="BodyVol", force_per_volume=(0.0, 0.0, -1.0))
    fem_b = g.mesh.queries.get_fem_data()
    assert fem_b is not fem_a

    g.masses.volume(pg="BodyVol", density=2400.0, reduction="lumped")
    fem_c = g.mesh.queries.get_fem_data()
    assert fem_c is not fem_b
    assert fem_c is not fem_a
