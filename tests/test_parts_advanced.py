"""
Advanced Parts system tests — fragmentation edge cases, registry
operations, node/face maps, anchor rebinding, end-to-end assembly.

Complements the happy-path tests in test_parts_fragmentation.py,
test_part_autopersist.py, and test_part_anchors.py.
"""
import warnings

import gmsh
import numpy as np
import pytest

from apeGmsh import apeGmsh, Part


# =====================================================================
# Helpers
# =====================================================================

def _vol_count():
    return len(gmsh.model.getEntities(3))


def _surf_count():
    return len(gmsh.model.getEntities(2))


def _make_labeled_part(name, x, y, z, dx, dy, dz, label):
    """Build a Part with a single labeled box."""
    p = Part(name)
    p.begin()
    p.model.geometry.add_box(x, y, z, dx, dy, dz, label=label)
    p.end()
    return p


# =====================================================================
# Group 1: Fragmentation Edge Cases
# =====================================================================

class TestFragmentationEdgeCases:

    def test_fragment_three_overlapping_chain(self, g):
        """A--B--C chain: A∩B exists, B∩C exists, no A∩C.

        A = [0, 2], B = [1, 3], C = [2.5, 4.5]
        → A-only=[0,1], A∩B=[1,2], B-only=[2,2.5], B∩C=[2.5,3], C-only=[3,4.5]
        """
        with g.parts.part("a"):
            g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        with g.parts.part("b"):
            g.model.geometry.add_box(1, 0, 0, 2, 1, 1)      # overlaps A at [1,2]
        with g.parts.part("c"):
            g.model.geometry.add_box(2.5, 0, 0, 2, 1, 1)    # overlaps B at [2.5,3]

        g.parts.fragment_all()

        vols = gmsh.model.getEntities(3)
        # A-only, A∩B, B-only, B∩C, C-only = 5 volumes
        assert len(vols) == 5

        # Each instance should have entities
        for label in ["a", "b", "c"]:
            tags = g.parts.get(label).entities.get(3, [])
            assert len(tags) >= 1, f"Part {label} has no volumes after fragment"

    def test_fragment_disjoint_parts(self, g):
        """Non-overlapping parts: fragment is a no-op in volume count."""
        with g.parts.part("left"):
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        with g.parts.part("right"):
            g.model.geometry.add_box(5, 0, 0, 1, 1, 1)

        vols_before = _vol_count()
        g.parts.fragment_all()
        vols_after = _vol_count()

        assert vols_after == vols_before == 2

    def test_fragment_tangent_parts(self, g):
        """Boxes sharing a face: fragment keeps 2 volumes, conformal mesh."""
        with g.parts.part("left"):
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        with g.parts.part("right"):
            g.model.geometry.add_box(1, 0, 0, 1, 1, 1)  # shares x=1 face

        g.parts.fragment_all()

        # Still 2 volumes (touching, not overlapping)
        assert _vol_count() == 2

        # Mesh and verify shared interface
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)

        # Get nodes at the interface (x ≈ 1.0)
        all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(all_coords).reshape(-1, 3)
        interface_mask = np.abs(coords[:, 0] - 1.0) < 1e-6
        n_interface = interface_mask.sum()
        assert n_interface > 0, "No shared nodes at interface x=1"

    def test_fragment_preserves_labels(self, g):
        """Labels (Tier 1) survive fragmentation."""
        with g.parts.part("a"):
            g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="web")
        with g.parts.part("b"):
            g.model.geometry.add_box(1, 0, 0, 2, 1, 1, label="flange")

        g.parts.fragment_all()

        # Both labels should still be resolvable
        web_ents = g.labels.entities("web")
        flange_ents = g.labels.entities("flange")
        assert len(web_ents) > 0, "Label 'web' lost after fragmentation"
        assert len(flange_ents) > 0, "Label 'flange' lost after fragmentation"

    def test_fragment_repeated_is_idempotent(self, g):
        """Calling fragment_all() twice doesn't change anything."""
        with g.parts.part("a"):
            g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        with g.parts.part("b"):
            g.model.geometry.add_box(1, 0, 0, 2, 1, 1)

        g.parts.fragment_all()
        vols_after_first = _vol_count()
        a_tags_first = list(g.parts.get("a").entities.get(3, []))

        g.parts.fragment_all()
        vols_after_second = _vol_count()
        a_tags_second = list(g.parts.get("a").entities.get(3, []))

        assert vols_after_second == vols_after_first
        assert sorted(a_tags_second) == sorted(a_tags_first)


# =====================================================================
# Group 2: Registry Operations
# =====================================================================

class TestRegistryOperations:

    def test_from_model_captures_all_dims(self, g):
        """from_model with no dim filter captures everything."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        inst = g.parts.from_model("everything")
        # Should have at least volumes, surfaces, curves, points
        assert 3 in inst.entities and len(inst.entities[3]) > 0
        assert 2 in inst.entities and len(inst.entities[2]) > 0

    def test_from_model_dim_filter(self, g):
        """from_model with dim=2 captures only surfaces."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()

        inst = g.parts.from_model("surfs_only", dim=2)
        assert 2 in inst.entities and len(inst.entities[2]) > 0
        # Should NOT have volumes
        assert len(inst.entities.get(3, [])) == 0

    def test_rename_instance(self, g):
        """Rename changes label, preserves entities."""
        with g.parts.part("old_name"):
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

        old_tags = list(g.parts.get("old_name").entities.get(3, []))
        g.parts.rename("old_name", "new_name")

        assert "old_name" not in g.parts.labels()
        assert "new_name" in g.parts.labels()
        new_tags = list(g.parts.get("new_name").entities.get(3, []))
        assert sorted(new_tags) == sorted(old_tags)

    def test_delete_instance(self, g):
        """Delete removes from registry but entities survive in Gmsh."""
        with g.parts.part("doomed"):
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

        vols_before = _vol_count()
        g.parts.delete("doomed")

        assert "doomed" not in g.parts.labels()
        # Entities still exist in Gmsh
        assert _vol_count() == vols_before

    def test_delete_then_fragment_survives(self, g):
        """Deleting one part, then fragmenting the rest, doesn't crash."""
        with g.parts.part("a"):
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        with g.parts.part("b"):
            g.model.geometry.add_box(2, 0, 0, 1, 1, 1)

        g.parts.delete("a")
        # Only "b" remains — fragment_all should be a no-op (single part)
        result = g.parts.fragment_all()
        assert len(result) >= 1


# =====================================================================
# Group 3: Node & Face Maps
# =====================================================================

class TestNodeFaceMaps:

    def test_node_map_two_disjoint_parts(self, g):
        """Disjoint boxes → each part gets its own nodes, no overlap."""
        with g.parts.part("left"):
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        with g.parts.part("right"):
            g.model.geometry.add_box(5, 0, 0, 1, 1, 1)

        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)

        node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)

        assert "left" in node_map
        assert "right" in node_map

        left_set = node_map["left"]
        right_set = node_map["right"]

        # No overlap between disjoint parts
        assert len(left_set & right_set) == 0
        # Together they cover all nodes
        assert left_set | right_set == set(int(n) for n in fem.nodes.ids)

    def test_node_map_shared_interface(self, g):
        """Tangent boxes → interface nodes in both maps."""
        with g.parts.part("left"):
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        with g.parts.part("right"):
            g.model.geometry.add_box(1, 0, 0, 1, 1, 1)

        g.parts.fragment_all()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)

        node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)

        left_set = node_map["left"]
        right_set = node_map["right"]

        # Interface nodes should appear in both (bbox overlap at boundary)
        overlap = left_set & right_set
        assert len(overlap) > 0, "No shared interface nodes found"

    def test_node_map_empty_registry(self, g):
        """No parts → empty node map."""
        # Don't create any parts
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)

        node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
        assert len(node_map) == 0

    def test_face_map_basic(self, g):
        """Face map returns surface connectivity for a single part."""
        with g.parts.part("box"):
            g.model.geometry.add_box(0, 0, 0, 1, 1, 1)

        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)

        node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
        face_map = g.parts.build_face_map(node_map)

        assert "box" in face_map
        faces = face_map["box"]
        assert len(faces) > 0, "No faces found for part"


# =====================================================================
# Group 4: Anchor Rebinding Edge Cases
# =====================================================================

@pytest.fixture
def symmetric_part():
    """Part with two identical boxes side-by-side, each labeled.
    Built BEFORE the assembly session to avoid Gmsh session conflict."""
    p = Part("sym")
    with p:
        p.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="left_box")
        p.model.geometry.add_box(2, 0, 0, 1, 1, 1, label="right_box")
    yield p
    p.cleanup()


@pytest.fixture
def simple_labeled_part():
    """Part with one labeled box."""
    p = Part("simple")
    with p:
        p.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="core")
    yield p
    p.cleanup()


@pytest.fixture
def two_label_part():
    """Part with two labeled boxes far apart."""
    p = Part("two_labels")
    with p:
        p.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="near")
        p.model.geometry.add_box(100, 100, 100, 1, 1, 1, label="far_away")
    yield p
    p.cleanup()


@pytest.fixture
def unlabeled_line_part():
    """Part with a single line, no labels (hence no sidecar)."""
    p = Part("col")
    with p:
        p0 = p.model.geometry.add_point(0, 0, 0, lc=0.5)
        p1 = p.model.geometry.add_point(0, 0, 3, lc=0.5)
        p.model.geometry.add_line(p0, p1)
    yield p
    p.cleanup()


class TestAnchorRebinding:

    def test_rebind_symmetric_part(self, symmetric_part, g):
        """Two identical boxes side-by-side: greedy matching assigns each uniquely."""
        inst = g.parts.add(symmetric_part, label="sym_inst")
        names = inst.label_names
        assert "sym_inst.left_box" in names
        assert "sym_inst.right_box" in names
        # Both labels resolve to different entities
        left_ents = g.labels.entities("sym_inst.left_box")
        right_ents = g.labels.entities("sym_inst.right_box")
        assert len(left_ents) > 0
        assert len(right_ents) > 0
        # No entity overlap
        left_tags = set(left_ents)
        right_tags = set(right_ents)
        assert len(left_tags & right_tags) == 0

    def test_rebind_with_translate(self, simple_labeled_part, g):
        """Labels resolve after translating the imported part."""
        inst = g.parts.add(simple_labeled_part, label="shifted_inst",
                           translate=(10, 20, 30))
        name = "shifted_inst.core"
        assert name in inst.label_names
        ents = g.labels.entities(name)
        assert len(ents) > 0

    def test_rebind_both_labels_match(self, two_label_part, g):
        """Part with two labeled boxes far apart: both rebound."""
        inst = g.parts.add(two_label_part, label="dual_inst")
        assert "dual_inst.near" in inst.label_names
        assert "dual_inst.far_away" in inst.label_names

    def test_add_unlabeled_part_with_translate(self, unlabeled_line_part, g):
        """Regression: ``add(part, translate=...)`` on a Part with no
        labels (hence no sidecar) must not hit UnboundLocalError on
        ``labels_comp`` inside ``_import_cad``.
        """
        a = g.parts.add(unlabeled_line_part, label="col_0",
                        translate=(0.0, 0.0, 0.0))
        b = g.parts.add(unlabeled_line_part, label="col_1",
                        translate=(4.0, 0.0, 0.0))

        assert a.label == "col_0"
        assert b.label == "col_1"
        assert a.translate == (0.0, 0.0, 0.0)
        assert b.translate == (4.0, 0.0, 0.0)
        assert set(g.parts.labels()) == {"col_0", "col_1"}
        # No sidecar → no extra labels beyond the explicit ``label=`` kwarg.
        assert a.label_names == ["col_0"]
        assert b.label_names == ["col_1"]


# =====================================================================
# Group 5: End-to-End Assembly Workflow
# =====================================================================

class TestAssemblyWorkflow:

    def test_assembly_fragment_mesh_roundtrip(self, g):
        """Full workflow: parts → fragment → mesh → FEMData."""
        # Two overlapping parts
        with g.parts.part("col"):
            g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3)
        with g.parts.part("slab"):
            g.model.geometry.add_box(-1, -1, 2.5, 2, 2, 0.5)

        g.parts.fragment_all()

        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)

        fem = g.mesh.queries.get_fem_data(dim=3)

        assert fem.info.n_nodes > 0
        assert fem.info.n_elems > 0

        # Node map partitions correctly
        node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
        assert "col" in node_map
        assert "slab" in node_map
        assert len(node_map["col"]) > 0
        assert len(node_map["slab"]) > 0

    def test_assembly_with_labels_and_constraint(self, g):
        """Labels + constraints survive the full assembly pipeline."""
        # Column with labeled top face
        with g.parts.part("col"):
            g.model.geometry.add_box(0, 0, 0, 0.5, 0.5, 3, label="col_body")

        # Slab with labeled bottom face
        with g.parts.part("slab"):
            g.model.geometry.add_box(-1, -1, 3, 2, 2, 0.3, label="slab_body")

        g.parts.fragment_all()

        # Labels should survive fragmentation
        assert len(g.labels.entities("col_body")) > 0
        assert len(g.labels.entities("slab_body")) > 0

        # Define physical groups for constraint targets
        col_ents = g.labels.entities("col_body")
        slab_ents = g.labels.entities("slab_body")
        assert len(col_ents) > 0
        assert len(slab_ents) > 0

        # Mesh
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)

        fem = g.mesh.queries.get_fem_data(dim=3)
        assert fem.info.n_nodes > 0

    def test_assembly_multipart_node_ownership(self, g):
        """Three disjoint parts: node map gives clean partition."""
        for i, label in enumerate(["a", "b", "c"]):
            with g.parts.part(label):
                g.model.geometry.add_box(i * 3, 0, 0, 1, 1, 1)

        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)
        fem = g.mesh.queries.get_fem_data(dim=3)

        node_map = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)

        all_nodes = set(int(n) for n in fem.nodes.ids)
        assigned = set()
        for label in ["a", "b", "c"]:
            assert label in node_map
            assigned |= node_map[label]

        # All mesh nodes accounted for
        assert assigned == all_nodes

        # No pairwise overlap (disjoint)
        for i, l1 in enumerate(["a", "b", "c"]):
            for l2 in ["a", "b", "c"][i+1:]:
                overlap = node_map[l1] & node_map[l2]
                assert len(overlap) == 0, f"{l1} and {l2} share nodes"
