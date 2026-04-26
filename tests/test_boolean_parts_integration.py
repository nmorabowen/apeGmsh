"""Integration tests for the two v1.0.3 changes:

* Piece 1 — ``g.model.boolean.*`` keeps ``Instance.entities`` consistent
  after an OCC boolean through the ``_remap_from_result`` helper.
* Piece 2 — ``objects=`` / ``tools=`` accept label names, PG names,
  raw tags, dimtags, and lists mixing all of the above.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh, Part


# =====================================================================
# Piece 1 — low-level booleans preserve Instance.entities
# =====================================================================

class TestLowLevelBooleanRemapsInstances:
    """After a raw g.model.boolean.* on Instance-owned entities,
    ``inst.entities`` must reflect the new tags without the user
    having to call g.parts.fragment_* again."""

    def test_fragment_on_registered_instances(self, g):
        """Register two instances then fragment via g.model.boolean.
        Instance.entities must be remapped."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        g.parts.register("A", [(3, box_a)])
        g.parts.register("B", [(3, box_b)])

        g.model.boolean.fragment(box_a, box_b)

        a_tags = g.parts.get("A").entities.get(3, [])
        b_tags = g.parts.get("B").entities.get(3, [])
        all_vols = {t for _, t in gmsh.model.getEntities(3)}

        assert len(a_tags) >= 1
        assert len(b_tags) >= 1
        # Both instances together must cover every live volume —
        # nothing stale, nothing phantom.  Fragment produced 3 vols
        # (2 disjoint pieces + 1 shared overlap) across the two
        # instances.
        assert set(a_tags) | set(b_tags) == all_vols
        assert len(all_vols) == 3

    def test_fuse_on_registered_box_cylinder_instances(self, g):
        """Fuse via g.model.boolean must keep both Part instances pointing
        at the merged volume.  Box+cylinder is the pathological case:
        OCC returns ``result_map=[[], []]`` even though the union exists,
        so without the absorbed-fallback both Parts get ``entities[3]=[]``."""
        box = g.model.geometry.add_box(0, 0, 0, 5, 5, 5)
        cyl = g.model.geometry.add_cylinder(2.5, 2.5, 2.5, 0, 0, 5, 1)
        g.parts.register("solid_box", [(3, box)])
        g.parts.register("solid_cyl", [(3, cyl)])

        result = g.model.boolean.fuse(box, cyl)
        assert len(result) == 1
        fused = result[0]

        box_tags = g.parts.get("solid_box").entities.get(3, [])
        cyl_tags = g.parts.get("solid_cyl").entities.get(3, [])
        assert box_tags == [fused], (
            f"solid_box lost the fused volume; got {box_tags}"
        )
        assert cyl_tags == [fused], (
            f"solid_cyl lost the fused volume; got {cyl_tags}"
        )

    def test_intersect_on_registered_box_cylinder_instances(self, g):
        """Mirrors the fuse test for intersect — already worked because
        OCC populates ``result_map`` for intersect, but lock the contract."""
        box = g.model.geometry.add_box(0, 0, 0, 5, 5, 5)
        cyl = g.model.geometry.add_cylinder(2.5, 2.5, 2.5, 0, 0, 5, 1)
        g.parts.register("solid_box", [(3, box)])
        g.parts.register("solid_cyl", [(3, cyl)])

        result = g.model.boolean.intersect(box, cyl)
        assert len(result) == 1
        inter = result[0]

        box_tags = g.parts.get("solid_box").entities.get(3, [])
        cyl_tags = g.parts.get("solid_cyl").entities.get(3, [])
        assert box_tags == [inter]
        assert cyl_tags == [inter]

    def test_cut_on_registered_instances(self, g):
        """Cut via g.model.boolean must rewrite the object's entities."""
        box = g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
        cyl = g.model.geometry.add_cylinder(1, 1, 0, 0, 0, 2, 0.5)
        g.parts.register("solid", [(3, box)])
        g.parts.register("hole", [(3, cyl)])

        result = g.model.boolean.cut(box, cyl)

        solid_tags = g.parts.get("solid").entities.get(3, [])
        # The Instance entities must match what cut returned —
        # whether OCC reused the old tag or allocated a fresh one,
        # the remap keeps the two in sync.
        assert set(solid_tags) == set(result)

    def test_remap_handles_multi_dim_inputs(self, g):
        """_remap_from_result must build a per-dim map — not hardcode
        a single dim — when the caller's input_ents mixes dim-3 and
        dim-2 entries."""
        from apeGmsh.core._parts_registry import PartsRegistry

        # A hand-constructed result_map that simulates a mixed-dim
        # boolean: one volume + one surface in, both replaced by
        # one new entity each at the same dim.
        box_a = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        surf = gmsh.model.getBoundary([(3, box_a)], oriented=False)[0][1]

        inst = g.parts.register("A", [(3, box_a), (2, surf)])
        input_ents: list[tuple[int, int]] = [(3, box_a), (2, surf)]
        result_map = [[(3, 999)], [(2, 888)]]

        g.parts._remap_from_result(input_ents, result_map)

        assert inst.entities[3] == [999]
        assert inst.entities[2] == [888]

    def test_part_session_has_no_parts_composite(self):
        """Boolean inside a Part session must not crash — Part has no
        ``parts`` composite, so the getattr null-safety guard is what
        keeps this working."""
        p = Part("unit")
        p.begin()
        try:
            a = p.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            b = p.model.geometry.add_box(0.5, 0, 0, 1, 1, 1)
            result = p.model.boolean.fuse(a, b)
            assert len(result) == 1
        finally:
            p.end()


class TestExistingFragmentationStillWorks:
    """Regression sentinel — the Parts-level fragment/fuse paths must
    continue to produce the same observable state they did before
    the remap extraction."""

    def test_fragment_all_still_updates_entities(self, g):
        with g.parts.part("a"):
            g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        with g.parts.part("b"):
            g.model.geometry.add_box(1, 0, 0, 2, 1, 1)

        g.parts.fragment_all()

        a_tags = g.parts.get("a").entities.get(3, [])
        b_tags = g.parts.get("b").entities.get(3, [])
        assert len(a_tags) >= 1
        assert len(b_tags) >= 1
        all_vols = {t for _, t in gmsh.model.getEntities(3)}
        assert set(a_tags) | set(b_tags) == all_vols


# =====================================================================
# Piece 2 — booleans accept labels, PG names, tags, dimtags
# =====================================================================

class TestBooleanAcceptsLabels:
    """g.model.boolean.* now accepts the same input shape as
    g.physical.add — labels (Tier 1), physical-group names (Tier 2),
    raw tags, dimtags, or lists mixing them all."""

    def test_label_label(self, g):
        """Both inputs are label names."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="slab")
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1, label="col")
        result = g.model.boolean.fragment("slab", "col")
        assert len(result) == 3

    def test_label_equivalent_to_dimtag(self, g):
        """Label-based input produces the same number of survivors as
        an equivalent dimtag call in the same session."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="slab")
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1, label="col")
        res_label = g.model.boolean.fragment("slab", "col")
        n_label = len(res_label)
        # Reset for second pass
        gmsh.model.remove()
        gmsh.model.add("reset")
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        res_dt = g.model.boolean.fragment((3, box_a), (3, box_b))
        assert n_label == len(res_dt) == 3

    def test_pg_pg(self, g):
        """Both inputs are user-PG names."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        g.physical.add(3, [box_a], name="Slab")
        g.physical.add(3, [box_b], name="Col")
        result = g.model.boolean.fragment("Slab", "Col")
        assert len(result) == 3

    def test_mixed_list(self, g):
        """Mix label strings, raw tags, and dimtags in one input list."""
        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="slab")
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        box_c = g.model.geometry.add_box(0, 2, 0, 2, 1, 1)
        # objects: label + bare tag; tools: dimtag
        result = g.model.boolean.fragment(["slab", box_b], [(3, box_c)])
        # All three boxes pairwise-disjoint except slab/box_b overlap,
        # so fragment yields 3 + 1 = 4 volumes.
        vols = [t for _, t in gmsh.model.getEntities(3)]
        assert len(vols) == 4
        assert len(result) == 4

    def test_unknown_string_raises_keyerror(self, g):
        """An unknown string ref raises KeyError citing label and PG."""
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="slab")
        with pytest.raises(KeyError, match="does_not_exist"):
            g.model.boolean.fragment("does_not_exist", "slab")

    def test_label_in_part_session(self):
        """Label-based input resolves inside a Part session, which has
        ``g.labels`` but no ``g.parts``."""
        p = Part("unit")
        p.begin()
        try:
            p.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="a")
            p.model.geometry.add_box(1, 0, 0, 2, 1, 1, label="b")
            result = p.model.boolean.fragment("a", "b")
            assert len(result) == 3
        finally:
            p.end()

    def test_shadowed_name_resolves_label_first(self, g):
        """When a label and a user PG share a name, the label wins —
        matches the documented resolution order of resolve_to_tags."""
        from apeGmsh.core._helpers import resolve_to_dimtags

        box_a = g.model.geometry.add_box(0, 0, 0, 2, 1, 1)
        box_b = g.model.geometry.add_box(1, 0, 0, 2, 1, 1)
        # Label X → box_a ; user PG X → box_b.  Same name, two tiers.
        g.labels.add(3, [box_a], "X")
        g.physical.add(3, [box_b], name="X")

        resolved = resolve_to_dimtags("X", default_dim=3, session=g)
        # Label (Tier 1) must win — we expect box_a, NOT box_b.
        assert resolved == [(3, box_a)], (
            f"Expected label X (box_a={box_a}) to resolve first, "
            f"got {resolved}"
        )
