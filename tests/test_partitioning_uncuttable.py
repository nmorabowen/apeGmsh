"""Tests for ADR 0092 S2 — ``g.mesh.partitioning.partition(..., uncuttable_elements=)``.

INV-4: "the partitioner may cut the slave side of a contact interface,
never the master." A cut master surface leaves some facets' backing
solid off-rank, which disables the fork's ``-kn auto`` for those facets.

The mechanism is a **deterministic post-partition override**, not a
weighted bias: METIS vertex weights (the existing ``weights=`` path)
balance partition *size*, they do not stop METIS from cutting through a
specific group. ``uncuttable_elements`` instead tallies which partition
already holds the most of the named element tags after the normal
partition runs, and reassigns every tag in the group to that partition
via ``partition_explicit`` — see ``_Partitioning._enforce_uncuttable``.
"""
from __future__ import annotations

import gmsh
import pytest


# ── Helpers ──────────────────────────────────────────────────────────

def _build_two_body_contact_model(g, lc: float = 2.0) -> None:
    """Two separate, non-touching boxes — the shape of a contact
    interface (no shared mesh nodes across the gap). Box A (bottom, the
    larger body) plays the "master"; box B (top, smaller) the "slave"."""
    g.model.geometry.add_box(0, 0, 0, 10, 10, 5)      # master body
    g.model.geometry.add_box(0, 0, 5.5, 10, 10, 2)    # slave body, gapped
    g.model.sync()
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(3)


def _surface_tags_at_z(z: float, tol: float = 0.01) -> list[int]:
    tags = []
    for _, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(2, tag)
        if abs(bb[2] - z) < tol and abs(bb[5] - z) < tol:
            tags.append(tag)
    return tags


def _master_surface_and_backing_elements(
    master_surface_tags: list[int],
) -> tuple[list[int], list[int]]:
    """2D face element tags of *master_surface_tags* plus the 3D
    element(s) backing each face (elements sharing that face's corner
    node set) — the ADR 0092 INV-4 "master surface and its backing
    solid elements" group, computed directly from Gmsh (a later slice
    will derive the same shape from a ``ContactRecord.master_faces``).
    """
    master_2d: list[int] = []
    for tag in master_surface_tags:
        _, etags_list, _ = gmsh.model.mesh.getElements(2, tag)
        for etags in etags_list:
            master_2d.extend(int(t) for t in etags)

    master_keys: set[tuple[int, ...]] = set()
    for tag in master_surface_tags:
        etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(2, tag)
        for etype, etags, enodes in zip(etypes, etags_list, enodes_list):
            _, _, _, num_nodes, _, n_primary = (
                gmsh.model.mesh.getElementProperties(int(etype)))
            enodes = list(enodes)
            for i in range(len(etags)):
                # Corner nodes only (``n_primary``): tris give 3-node
                # keys, quads 4-node keys — matching the arity Gmsh uses
                # for the volume-side face keys below.
                corners = enodes[
                    i * num_nodes:(i + 1) * num_nodes][:n_primary]
                master_keys.add(tuple(sorted(int(n) for n in corners)))

    backing: list[int] = []
    for ent_dim, ent_tag in gmsh.model.getEntities(3):
        etypes, etags_list, _ = gmsh.model.mesh.getElements(ent_dim, ent_tag)
        for etype, etags in zip(etypes, etags_list):
            if len(etags) == 0:
                continue
            # tag=ent_tag scopes the query to THIS entity. The default
            # tag=-1 returns face data for every entity sharing the
            # element type, which silently misaligns the per-element
            # face count on any model with >1 same-type 3D entity.
            #
            # Both face arities are queried, for the same reason
            # ``_build_dual_graph`` does: tets expose only 3-node faces,
            # hexes only 4-node ones.  A single arity returns an empty
            # array for the family that lacks it, leaving ``backing``
            # empty and every INV-4 assertion below vacuously true.
            hits: set[int] = set()
            for arity in (3, 4):
                face_nodes = gmsh.model.mesh.getElementFaceNodes(
                    int(etype), arity, tag=ent_tag)
                if len(face_nodes) == 0:
                    continue
                n_keys = len(face_nodes) // (arity * len(etags))
                ptr = 0
                for elem_tag in etags:
                    for _ in range(n_keys):
                        key = tuple(sorted(
                            int(x) for x in face_nodes[ptr:ptr + arity]))
                        ptr += arity
                        if key in master_keys:
                            hits.add(int(elem_tag))
            backing.extend(sorted(hits))

    # Fail loudly rather than hand back an empty group: every assertion
    # in this file is about where these elements land, so an empty
    # ``backing`` would pass every test while checking nothing.
    assert backing, (
        "no backing solids found for master surfaces "
        f"{master_surface_tags} - the INV-4 assertions would be vacuous")
    return master_2d, backing


def _partitions_touched(fem, element_ids: set[int]) -> set[int]:
    touched = set()
    for rec in fem.partitions:
        if set(int(x) for x in rec.element_ids) & element_ids:
            touched.add(rec.id)
    return touched


# =====================================================================
# INV-4 — master surface + backing solids never split across ranks
# =====================================================================

class TestUncuttableMasterSurface:

    def test_master_surface_and_backing_solids_stay_on_one_partition(self, g):
        _build_two_body_contact_model(g)
        master_tags = _surface_tags_at_z(5.0)
        assert master_tags, "test setup: expected a master surface at z=5"
        master_2d, backing_3d = _master_surface_and_backing_elements(master_tags)
        assert master_2d and backing_3d, "test setup: expected a non-empty group"
        uncuttable = master_2d + backing_3d

        g.mesh.partitioning.partition(2, uncuttable_elements=uncuttable)
        fem = g.mesh.queries.get_fem_data(dim=None)

        touched = _partitions_touched(fem, set(uncuttable))
        assert len(touched) == 1, (
            f"master surface + backing solids spread across "
            f"{len(touched)} partitions: {touched}")

        # The owner rank must carry every node the group needs — a
        # replicated boundary node may ALSO show up on a neighbour rank
        # (that is normal ghost topology), but it must not be MISSING
        # from the owner.
        owner = next(iter(touched))
        owner_nodes = set(int(n) for n in fem.partitions[owner].node_ids)
        group_nodes: set[int] = set()
        for tag in uncuttable:
            elem_type, node_tags, _dim, _etag = gmsh.model.mesh.getElement(tag)
            group_nodes.update(int(n) for n in node_tags)
        assert group_nodes <= owner_nodes

    def test_group_is_a_small_subset_of_the_master_body(self, g):
        """Only the surface + its immediate backing layer is pinned —
        not the whole master volume — so the rest of the body stays
        free for METIS to balance. Compares the 3D backing elements
        only — the group also carries the master's 2D face elements,
        which are never part of a 3D volume's element set."""
        _build_two_body_contact_model(g)
        master_tags = _surface_tags_at_z(5.0)
        _master_2d, backing_3d = _master_surface_and_backing_elements(master_tags)

        _, etags_list, _ = gmsh.model.mesh.getElements(3, 1)
        master_body_all = {int(t) for etl in etags_list for t in etl}
        assert set(backing_3d) < master_body_all, (
            "expected the backing layer to be a proper subset of the "
            "master body's full 3D element set")


# =====================================================================
# Hard constraint — a plain partition() call is unaffected
# =====================================================================

class TestUncuttableIsStrictlyOptIn:
    """``uncuttable_elements=None`` (the default) or ``[]`` must never
    invoke the override — proved by spying on
    ``_Partitioning._enforce_uncuttable`` rather than diffing two METIS
    runs, because Gmsh's native partitioner is not itself
    call-to-call deterministic (confirmed empirically: two back-to-back
    unweighted ``partition(2)`` calls on an identical mesh produced
    different element counts per partition) — a fact that predates this
    slice and is not something this test should rely on.
    """

    @staticmethod
    def _spy(g, monkeypatch):
        from apeGmsh.mesh._mesh_partitioning import _Partitioning
        calls: list[tuple] = []
        original = _Partitioning._enforce_uncuttable

        def spy(self, *args, **kwargs):
            calls.append((args, kwargs))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(_Partitioning, "_enforce_uncuttable", spy)
        return calls

    def test_default_never_invokes_override(self, g, monkeypatch):
        _build_two_body_contact_model(g)
        calls = self._spy(g, monkeypatch)
        g.mesh.partitioning.partition(2)
        assert calls == []

    def test_explicit_none_never_invokes_override(self, g, monkeypatch):
        _build_two_body_contact_model(g)
        calls = self._spy(g, monkeypatch)
        g.mesh.partitioning.partition(2, uncuttable_elements=None)
        assert calls == []

    def test_empty_list_never_invokes_override(self, g, monkeypatch):
        _build_two_body_contact_model(g)
        calls = self._spy(g, monkeypatch)
        g.mesh.partitioning.partition(2, uncuttable_elements=[])
        assert calls == []

    def test_nonempty_group_invokes_override_exactly_once(self, g, monkeypatch):
        _build_two_body_contact_model(g)
        _, etags_list, _ = gmsh.model.mesh.getElements(3, 1)
        one_tag = int(etags_list[0][0])
        calls = self._spy(g, monkeypatch)
        g.mesh.partitioning.partition(2, uncuttable_elements=[one_tag])
        assert len(calls) == 1


# =====================================================================
# Error / edge cases
# =====================================================================

class TestUncuttableErrors:

    def test_element_tag_not_in_mesh_raises(self, g):
        _build_two_body_contact_model(g)
        with pytest.raises(ValueError, match="absent from the partitioned mesh"):
            g.mesh.partitioning.partition(2, uncuttable_elements=[999_999_999])

    def test_single_partition_is_a_noop(self, g):
        """n_parts=1: the group is trivially already "whole" — the
        no-op branch of ``_enforce_uncuttable`` (``len(tally) <= 1``)."""
        _build_two_body_contact_model(g)
        _, etags_list, _ = gmsh.model.mesh.getElements(3, 1)
        one_tag = int(etags_list[0][0])
        info = g.mesh.partitioning.partition(1, uncuttable_elements=[one_tag])
        assert info.n_parts == 1

    def test_all_original_elements_still_accounted_for(self, g):
        """The override must not lose or duplicate an original element:
        every tag that existed before partitioning is still assigned to
        a partition afterwards. Not an equality check — partitioning
        itself (with or without ``uncuttable_elements``) adds new ghost
        interface elements at the cut, which is expected Gmsh topology
        creation, not something this override introduces."""
        _build_two_body_contact_model(g)
        before: set[int] = set()
        for d in range(4):
            _, etags_list, _ = gmsh.model.mesh.getElements(dim=d, tag=-1)
            for etags in etags_list:
                before.update(int(t) for t in etags)

        master_tags = _surface_tags_at_z(5.0)
        master_2d, backing_3d = _master_surface_and_backing_elements(master_tags)
        g.mesh.partitioning.partition(
            2, uncuttable_elements=master_2d + backing_3d)

        fem = g.mesh.queries.get_fem_data(dim=None)
        after: set[int] = set()
        for rec in fem.partitions:
            after.update(int(x) for x in rec.element_ids)
        assert before <= after


# =====================================================================
# Weighted backend interop (Flavor B, pymetis) — CI-only
# =====================================================================

class TestUncuttableWithWeightedBackend:
    """``uncuttable_elements`` composes with ``weights=`` — the override
    runs after either flavour. Needs pymetis (not installable on
    Windows; CI's ``partition-pymetis`` extra covers it), so this skips
    locally like every other Flavor B test in ``test_partitioning.py``.
    """

    def test_uncuttable_after_weighted_partition(self, g):
        pytest.importorskip("pymetis")
        _build_two_body_contact_model(g)

        n = 0
        for d in range(4):
            _, etags_list, _ = gmsh.model.mesh.getElements(dim=d, tag=-1)
            n += sum(len(t) for t in etags_list)

        master_tags = _surface_tags_at_z(5.0)
        master_2d, backing_3d = _master_surface_and_backing_elements(master_tags)
        uncuttable = master_2d + backing_3d

        g.mesh.partitioning.partition(
            2, weights=[1.0] * n, uncuttable_elements=uncuttable)
        fem = g.mesh.queries.get_fem_data(dim=None)

        touched = _partitions_touched(fem, set(uncuttable))
        assert len(touched) == 1, (
            f"master group spread across partitions {touched} even "
            "under the weighted backend")


class TestUncuttablePreservesEverythingElse:
    """Assertions the first S2 suite lacked (review finding F2).

    The original battery proved the group ends up on ONE partition and that the
    override is strictly opt-in — but nothing checked what happened to the rest
    of the mesh. Five one-line mutations survived it, the worst being
    ``parts = [target for tag in elem_tags]`` (move the WHOLE mesh onto one
    partition): the group is still on one partition, its nodes are still a subset
    of the owner's, and no spy fires. These close that hole.
    """

    def test_non_group_elements_keep_their_partition(self, g):
        _build_two_body_contact_model(g)
        master_2d, backing_3d = _master_surface_and_backing_elements(
            _surface_tags_at_z(5.0))
        uncuttable = master_2d + backing_3d

        # Pin a DETERMINISTIC baseline first. Going through partition() twice
        # would compare two independent METIS runs, and gmsh's METIS is not
        # call-to-call deterministic — the diff would be METIS noise, not the
        # override's doing. (That nondeterminism is exactly why the opt-in tests
        # above use spies rather than diffing assignments.) So: fix an assignment
        # with partition_explicit, then drive the override directly.
        part = g.mesh.partitioning
        part.partition(2)
        baseline = dict(part._elem_partition_map())
        tags = list(baseline)
        part.partition_explicit(2, tags, [baseline[t] for t in tags])
        before = dict(part._elem_partition_map())

        with pytest.warns(UserWarning, match="uncuttable_elements moved"):
            part._enforce_uncuttable(2, uncuttable)
        after = dict(part._elem_partition_map())

        # Only group members may have moved. This is what kills the
        # "move the whole mesh" mutation.
        moved = {e for e in before if e in after and before[e] != after[e]}
        assert moved <= set(uncuttable), (
            f"{len(moved - set(uncuttable))} NON-group element(s) changed "
            f"partition; the override must touch only the group")

    def test_no_partition_is_left_empty(self, g):
        _build_two_body_contact_model(g)
        master_2d, backing_3d = _master_surface_and_backing_elements(
            _surface_tags_at_z(5.0))

        with pytest.warns(UserWarning):
            g.mesh.partitioning.partition(
                2, uncuttable_elements=master_2d + backing_3d)

        fem = g.mesh.queries.get_fem_data(dim=None)
        counts = [len(rec.element_ids) for rec in fem.partitions]
        assert len(fem.partitions) == 2, (
            f"a partition vanished: {len(fem.partitions)} records for 2 parts")
        assert all(c > 0 for c in counts), f"empty partition: {counts}"

    def test_weighted_cache_survives_the_override(self, g):
        """`weights_per_partition` must reflect the NEW assignment, not vanish."""
        pytest.importorskip("pymetis")
        _build_two_body_contact_model(g)
        master_2d, backing_3d = _master_surface_and_backing_elements(
            _surface_tags_at_z(5.0))
        _, etags, _ = gmsh.model.mesh.getElements(dim=3)
        weights = {int(t): 1.0 for t in etags[0]}

        with pytest.warns(UserWarning):
            info = g.mesh.partitioning.partition(
                2, weights=weights, backend="pymetis",
                uncuttable_elements=master_2d + backing_3d)

        assert info.weights_per_partition is not None, (
            "the override cleared the weight cache it was supposed to restore")
