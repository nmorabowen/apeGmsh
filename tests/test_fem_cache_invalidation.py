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
* Every declaration kind (section 8, the ``KINDS`` table) and every
  ``clear()`` (section 9) invalidates the cache.  ``contact`` /
  ``contact_plane`` / ``interface`` / ``g.reinforce`` / ``g.embed``
  once did not: the second ``get_fem_data()`` returned the first
  snapshot without the new def.  ``test_declaration_coverage.py``
  holds new verbs to the same path.

These tests use the full ``g`` fixture so the gmsh + extractor round
trip is exercised end-to-end.  The dimension here is intentionally
small (1-volume box, coarse mesh) to keep the tests fast — the
contract is about cache identity, not mesh content.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gmsh
import pytest

from apeGmsh import NormalLaw, TangentialLaw
from apeGmsh._kernel.defs.rebar import Cage
from apeGmsh.core._compose_errors import ChainPhaseError


# =====================================================================
# Helpers
# =====================================================================

def _build_box(g) -> None:
    """Build a tiny single-box meshed model on the session."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 10.0, 10.0, 10.0, label="Body")
    g.model.sync()
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)


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


# =====================================================================
# 8. Every declaration kind invalidates the cache
# =====================================================================
#
# One row per kind: a model it resolves on, the declaration, and how
# many of its records a snapshot holds.  ``owner`` names the composite
# whose ``clear()`` retracts the kind (section 9).
#
# contact / contact_plane / interface / reinforce / embed are the
# regression rows: before ``_DeclarationsMixin._declare`` their verbs
# appended without bumping, so a declaration made after the first
# ``get_fem_data()`` was left out of every later snapshot.

def _face_at_z(volume: int, z: float) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False):
        if dim == 2 and abs(gmsh.model.occ.getCenterOfMass(2, abs(tag))[2] - z) < 1e-6:
            return abs(tag)
    raise AssertionError(f"no boundary face of volume {volume} at z={z}")


def _curve_at_x(surface: int, x: float) -> int:
    for _dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < 1e-6 and abs(bb[3] - x) < 1e-6:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _box_faces(g) -> None:
    """A unit box: volume ``solid``, bottom face ``floor``, top ``roof``."""
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    g.model.sync()
    g.physical.add(3, [box], name="solid")
    g.physical.add(2, [_face_at_z(box, 0.0)], name="floor")
    g.physical.add(2, [_face_at_z(box, 1.0)], name="roof")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)


def _stacked_boxes(g) -> None:
    """Two boxes 0.05 apart: ``master`` top face below, ``slave`` above."""
    lower = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    upper = g.model.geometry.add_box(0, 0, 1.05, 1, 1, 1)
    g.model.sync()
    g.physical.add(3, [lower, upper], name="solid")
    g.physical.add(2, [_face_at_z(lower, 1.0)], name="master")
    g.physical.add(2, [_face_at_z(upper, 1.05)], name="slave")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)


def _abutting_squares(g) -> None:
    """Two unfused unit squares whose x=1 edges coincide: the 2D
    interface lane's ``face`` (master) and ``wire`` (slave)."""
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=2)
    g.mesh.generation.generate(dim=2)


def _box_with_bar(g) -> None:
    """A ``concrete`` box and an independently meshed interior ``rebar``
    line."""
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    p0 = gmsh.model.occ.addPoint(0.5, 0.5, 0.2)
    p1 = gmsh.model.occ.addPoint(0.5, 0.5, 0.8)
    line = gmsh.model.occ.addLine(p0, p1)
    g.model.sync()
    g.physical.add(3, [box], name="concrete")
    g.physical.add(1, [line], name="rebar")
    g.mesh.sizing.set_global_size(0.4)
    g.mesh.generation.generate(dim=3)


def _box_with_point(g) -> None:
    """A ``host`` box and a free interior point ``probe``."""
    box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    pt = gmsh.model.occ.addPoint(0.4, 0.4, 0.4)
    g.model.sync()
    g.physical.add(3, [box], name="host")
    g.physical.add(0, [pt], name="probe")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)


@dataclass(frozen=True)
class _Kind:
    build: Callable            # geometry + mesh + PGs, before any extraction
    declare: Callable          # the declaration under test
    count: Callable            # records of this kind in a snapshot
    owner: str | None = None   # composite whose clear() retracts it


KINDS = {
    "contact": _Kind(
        _stacked_boxes,
        lambda g: g.constraints.contact(
            "master", "slave", formulation="nts", kn=1.0e6),
        lambda fem: len(fem.elements.contacts),
        owner="constraints"),
    "contact_plane": _Kind(
        _box_faces,
        lambda g: g.constraints.contact_plane(
            "floor", normal=(0, 0, 1), point=(0, 0, 0), kn=1.0e7),
        lambda fem: len(fem.elements.contact_planes),
        owner="constraints"),
    "interface": _Kind(
        _abutting_squares,
        lambda g: g.constraints.interface(
            "face", "wire",
            normal=NormalLaw(kind="ent", k_per_area=1.0e9),
            tangential=TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5),
            thickness=0.5),
        lambda fem: len(fem.elements.interfaces),
        owner="constraints"),
    "reinforce": _Kind(
        _box_with_bar,
        lambda g: g.reinforce(
            host="concrete", bars="rebar", perfect=1.0e12, bar_diameter=0.025),
        lambda fem: len(fem.elements.reinforce_ties),
        owner="reinforce"),
    "embed": _Kind(
        _box_with_point,
        lambda g: g.embed(host="host", nodes="probe", k=1.0e12),
        lambda fem: len(fem.elements.embed_ties),
        owner="embed"),
    "bc": _Kind(
        _box_faces,
        lambda g: g.constraints.bc("floor"),
        lambda fem: len(fem.nodes.sp),
        owner="constraints"),
    "displacement": _Kind(
        _box_faces,
        lambda g: g.displacements.point("floor", values=(0.01, 0.0, 0.0)),
        lambda fem: len(fem.nodes.sp)),
    "surface_load": _Kind(
        _box_faces,
        lambda g: g.loads.surface.pressure("roof", magnitude=1.0),
        lambda fem: len(fem.nodes.loads)),
    "decouple_node": _Kind(
        _box_faces,
        lambda g: g.decouple_node(coords=(5.0, 5.0, 5.0), label="anchor"),
        lambda fem: len(fem.nodes.ids)),
}

CLEARABLE = sorted(k for k, kind in KINDS.items() if kind.owner is not None)


@pytest.mark.parametrize("name", sorted(KINDS))
def test_declaration_after_extraction_invalidates_cache(g, name):
    """A declaration made after the first ``get_fem_data()`` appears in
    the next one, which is a new snapshot."""
    kind = KINDS[name]
    kind.build(g)

    fem_a = g.mesh.queries.get_fem_data()
    kind.declare(g)
    fem_b = g.mesh.queries.get_fem_data()

    assert fem_b is not fem_a, (
        f"{name}: a declaration after the first get_fem_data() must "
        f"invalidate the cache; the second call returned the first "
        f"snapshot."
    )
    assert kind.count(fem_b) > kind.count(fem_a), (
        f"{name}: the declaration made after the first extraction is "
        f"missing from the next snapshot."
    )


# =====================================================================
# 9. clear() retracts every kind it owns and invalidates the cache
# =====================================================================

@pytest.mark.parametrize("name", CLEARABLE)
def test_clear_invalidates_cache(g, name):
    """``clear()`` after an extraction empties the kind from the next
    snapshot, which is a new one."""
    kind = KINDS[name]
    kind.build(g)
    kind.declare(g)

    fem_a = g.mesh.queries.get_fem_data()
    assert kind.count(fem_a) > 0
    getattr(g, kind.owner).clear()
    fem_b = g.mesh.queries.get_fem_data()

    assert fem_b is not fem_a, (
        f"{name}: g.{kind.owner}.clear() must invalidate the cache; the "
        f"next get_fem_data() returned the snapshot that still holds "
        f"the cleared records."
    )
    assert kind.count(fem_b) == 0, (
        f"{name}: g.{kind.owner}.clear() left its records in place."
    )


def test_constraints_clear_empties_every_def_list(g):
    """``g.constraints.clear()`` empties all five of its def lists, not
    only the MP ``constraint_defs``."""
    _box_faces(g)
    c = g.constraints
    c.equal_dof("floor", "roof")
    c.bc("floor")
    c.contact("roof", "floor", kn=1.0e6)
    c.contact_plane("floor", normal=(0, 0, 1), point=(0, 0, 0), kn=1.0e7)
    c.interface(
        "floor", "roof",
        normal=NormalLaw(kind="ent", k_per_area=1.0e9),
        tangential=TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5))

    c.clear()

    lists = ("constraint_defs", "_bc_defs", "contact_defs",
             "contact_plane_defs", "interface_defs")
    left = {a: len(getattr(c, a)) for a in lists if getattr(c, a)}
    assert not left, f"clear() left defs behind: {left}"


# =====================================================================
# 10. g.rebar.place — declared, and frozen once a snapshot exists
# =====================================================================

def test_rebar_place_declares_and_is_frozen_after_extraction(g):
    """``g.rebar.place`` records its placement through the declaration
    path, and cannot leave a snapshot stale: it mutates geometry, so the
    chain-phase guard refuses it once the first snapshot exists."""
    g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 2.0, label="ConcreteVol")
    bar = g.rebar.bar([(0.15, 0.15, 0.1), (0.15, 0.15, 1.9)],
                      db=0.0254, material="rebar", name="L1")
    cage = Cage(bars=(bar,))

    before = g._fem_counter
    g.rebar.place(cage, into="ConcreteVol", coupling="conformal",
                  emit_elements=True)
    assert g._fem_counter > before, (
        "g.rebar.place must declare through _declare (which bumps the "
        "FEMData counter)."
    )

    g.mesh.sizing.set_global_size(0.2)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data()
    assert len(fem.elements.rebar_elements) == 1

    with pytest.raises(ChainPhaseError):
        g.rebar.place(cage, into="ConcreteVol", coupling="conformal",
                      name="again")
