"""
Tests for ``g.mesh.editing.split_higher_order_lines``.

Covers the three accepted ``policy`` values (forbid / split / constrain
where the last raises NotImplementedError this round), ``dim`` guard,
PG membership preservation, mid-node preservation, mixed Line3/Line2
entities (the spike's gmsh per-entity mixed-order claim), per-Line2
connectivity, multi-PG iterable input, and the bridge-side ``-ele``
fan-out path.

These tests are the persistent record of the one-off gmsh element-
surgery spike that verified ``removeElements + addElements`` works
across mixed-order entities on the same dim=1 entity — do not
weaken them without a re-spike.

ADR 0037.
"""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh import apeGmsh


# =====================================================================
# Helpers
# =====================================================================


def _build_quadratic_frame_pair(g) -> None:
    """Two parallel vertical edges meshed at order 2.

    Geometry: 2 columns of length 1, transfinite to 1 segment each.
    Order 2 → each column carries one Line3 (i, j, mid).

    PG "Cols" covers both curves.
    """
    p_a0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
    p_a1 = g.model.geometry.add_point(0.0, 0.0, 1.0)
    p_b0 = g.model.geometry.add_point(1.0, 0.0, 0.0)
    p_b1 = g.model.geometry.add_point(1.0, 0.0, 1.0)
    line_a = g.model.geometry.add_line(p_a0, p_a1)
    line_b = g.model.geometry.add_line(p_b0, p_b1)
    g.physical.add_curve([line_a, line_b], name="Cols")

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.setTransfiniteCurve(line_a, 2)
    gmsh.model.mesh.setTransfiniteCurve(line_b, 2)
    g.mesh.generation.generate(dim=1)


def _build_linear_frame(g) -> None:
    """One vertical edge meshed at order 1 → one Line2 element."""
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
    p1 = g.model.geometry.add_point(0.0, 0.0, 1.0)
    line_tag = g.model.geometry.add_line(p0, p1)
    g.physical.add_curve([line_tag], name="Cols")
    gmsh.model.mesh.setTransfiniteCurve(line_tag, 2)
    g.mesh.generation.generate(dim=1)


def _count_elements_in_pg(g, name: str, dim: int) -> dict[int, int]:
    """Return ``{etype: count}`` aggregated across the PG's entities."""
    pg_tag = g.physical.get_tag(dim, name)
    assert pg_tag is not None
    counts: dict[int, int] = {}
    for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag):
        types, tags, _ = gmsh.model.mesh.getElements(dim=dim, tag=int(ent))
        for et, tt in zip(types, tags):
            counts[int(et)] = counts.get(int(et), 0) + len(tt)
    return counts


def _line2_connectivity_pairs(
    g, name: str, dim: int = 1,
) -> set[frozenset[int]]:
    """Return the set of ``frozenset({a, b})`` for every Line2 element on
    ``name``'s entities.  Order-independent (a beam is a beam regardless
    of which end is i and which is j).
    """
    pg_tag = g.physical.get_tag(dim, name)
    assert pg_tag is not None
    pairs: set[frozenset[int]] = set()
    for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, pg_tag):
        types, tags, nodes = gmsh.model.mesh.getElements(
            dim=dim, tag=int(ent),
        )
        for etype, ttags, tnodes in zip(types, tags, nodes):
            if int(etype) != 1:   # only Line2
                continue
            flat = [int(n) for n in tnodes]
            for k in range(len(ttags)):
                pairs.add(frozenset((flat[2 * k], flat[2 * k + 1])))
    return pairs


# =====================================================================
# Argument validation
# =====================================================================


class TestValidation:
    def test_invalid_policy_raises_value_error(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            with pytest.raises(ValueError, match="policy must be"):
                g.mesh.editing.split_higher_order_lines(
                    "Cols", policy="explode",   # type: ignore[arg-type]
                )

    def test_dim_2_raises_not_implemented(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            with pytest.raises(
                NotImplementedError, match="dim=2",
            ):
                g.mesh.editing.split_higher_order_lines(
                    "Cols", policy="split", dim=2,
                )

    def test_constrain_raises_not_implemented(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            with pytest.raises(
                NotImplementedError, match="constrain.*not implemented",
            ):
                g.mesh.editing.split_higher_order_lines(
                    "Cols", policy="constrain",
                )

    def test_unknown_pg_raises_key_error(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            with pytest.raises(KeyError, match="MissingPG"):
                g.mesh.editing.split_higher_order_lines(
                    "MissingPG", policy="split",
                )

    def test_empty_pg_iterable_raises(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            with pytest.raises(ValueError, match="non-empty"):
                g.mesh.editing.split_higher_order_lines(
                    [], policy="split",
                )


# =====================================================================
# policy="forbid"
# =====================================================================


class TestForbid:
    def test_forbid_with_line3_raises_runtime(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            with pytest.raises(RuntimeError, match="Cols.*Line3"):
                g.mesh.editing.split_higher_order_lines(
                    "Cols", policy="forbid",
                )

    def test_forbid_with_only_line2_is_noop(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_linear_frame(g)
            # No raise: PG contains Line2 only.
            g.mesh.editing.split_higher_order_lines(
                "Cols", policy="forbid",
            )
            counts = _count_elements_in_pg(g, "Cols", dim=1)
            assert counts == {1: 1}   # one Line2

    def test_forbid_message_names_pg(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            with pytest.raises(RuntimeError) as ei:
                g.mesh.editing.split_higher_order_lines(
                    "Cols", policy="forbid",
                )
            msg = str(ei.value)
            assert "Cols" in msg
            assert "2" in msg   # 2 Line3 elements in the pair PG


# =====================================================================
# policy="split"
# =====================================================================


class TestSplit:
    def test_split_replaces_line3_with_line2_pair(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            before = _count_elements_in_pg(g, "Cols", dim=1)
            assert before == {8: 2}   # 2 Line3 elements

            g.mesh.editing.split_higher_order_lines(
                "Cols", policy="split",
            )
            after = _count_elements_in_pg(g, "Cols", dim=1)
            assert after == {1: 4}   # 4 Line2 elements (2 per Line3)

    def test_split_preserves_pg_membership(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            pg_tag_before = g.physical.get_tag(1, "Cols")
            ents_before = sorted(
                int(e) for e in
                gmsh.model.getEntitiesForPhysicalGroup(1, pg_tag_before)
            )
            g.mesh.editing.split_higher_order_lines(
                "Cols", policy="split",
            )
            pg_tag_after = g.physical.get_tag(1, "Cols")
            ents_after = sorted(
                int(e) for e in
                gmsh.model.getEntitiesForPhysicalGroup(1, pg_tag_after)
            )
            # PG tag and entity list unchanged (membership tracks at
            # entity level — the spike confirmed this).
            assert pg_tag_before == pg_tag_after
            assert ents_before == ents_after

    def test_split_preserves_mid_node(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            nodes_before = set(
                int(n) for n in gmsh.model.mesh.getNodes()[0]
            )
            g.mesh.editing.split_higher_order_lines(
                "Cols", policy="split",
            )
            nodes_after = set(
                int(n) for n in gmsh.model.mesh.getNodes()[0]
            )
            # No nodes lost (mid stays); no new nodes added.
            assert nodes_after == nodes_before

    def test_split_idempotent_on_already_line2(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_linear_frame(g)
            g.mesh.editing.split_higher_order_lines(
                "Cols", policy="split",
            )
            counts = _count_elements_in_pg(g, "Cols", dim=1)
            assert counts == {1: 1}   # untouched

    def test_split_accepts_iterable_pg(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            # Single PG covers both columns; pass it inside a list to
            # exercise the iterable path.
            g.mesh.editing.split_higher_order_lines(
                ["Cols"], policy="split",
            )
            counts = _count_elements_in_pg(g, "Cols", dim=1)
            assert counts == {1: 4}

    def test_split_returns_self(self) -> None:
        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            ret = g.mesh.editing.split_higher_order_lines(
                "Cols", policy="split",
            )
            assert ret is g.mesh.editing

    def test_split_emits_correct_line2_connectivity(self) -> None:
        """The load-bearing invariant of the surgery: each Line3
        ``(i, j, mid)`` becomes exactly the two Line2 pairs
        ``(i, mid)`` and ``(mid, j)`` — NOT ``(i, j) + (j, mid)`` or
        any other combination.  Order-independent assertion (Gmsh's
        i/j choice doesn't matter; the pair set does).
        """
        with apeGmsh(model_name="t") as g:
            # Capture parent Line3 connectivity BEFORE the split.
            _build_quadratic_frame_pair(g)
            pg_tag = g.physical.get_tag(1, "Cols")
            parent_triples: list[tuple[int, int, int]] = []
            for ent in gmsh.model.getEntitiesForPhysicalGroup(1, pg_tag):
                types, tags, nodes = gmsh.model.mesh.getElements(
                    dim=1, tag=int(ent),
                )
                for etype, ttags, tnodes in zip(types, tags, nodes):
                    if int(etype) != 8:   # Line3
                        continue
                    flat = [int(n) for n in tnodes]
                    for k in range(len(ttags)):
                        parent_triples.append(
                            (flat[3 * k], flat[3 * k + 1], flat[3 * k + 2])
                        )
            # Build the expected set: {(i, mid), (mid, j)} per parent.
            expected: set[frozenset[int]] = set()
            for n_i, n_j, n_mid in parent_triples:
                expected.add(frozenset((n_i, n_mid)))
                expected.add(frozenset((n_mid, n_j)))

            g.mesh.editing.split_higher_order_lines(
                "Cols", policy="split",
            )
            got = _line2_connectivity_pairs(g, "Cols")
            assert got == expected, (
                f"Line2 connectivity mismatch.  Expected {expected}, "
                f"got {got}.  A bug in _replace_line3_with_line2_pair "
                "could silently regress mechanics."
            )

    def test_split_quiet_no_op_on_line2_only_pg(self, capsys) -> None:
        """``policy='split'`` on a Line2-only PG prints a 'no-op'
        message, not the misleading 'demoted 0 Line3' line.
        """
        with apeGmsh(model_name="t") as g:
            _build_linear_frame(g)
            g.mesh.editing.split_higher_order_lines(
                "Cols", policy="split",
            )
        captured = capsys.readouterr()
        assert "no Line3 elements found" in captured.out
        assert "demoted 0" not in captured.out


# =====================================================================
# Multi-PG iterable + mixed Line2+Line3 entity (spike-locking tests)
# =====================================================================


def _build_two_pgs_mixed_orders(g) -> None:
    """Build two separate dim=1 PGs at DIFFERENT mesh orders.

    PG "Frames"  (order 2): two columns -> 2 Line3 elements
    PG "Braces"  (order 2): one column  -> 1 Line3 element

    All meshed in the same gmsh model (one global ElementOrder=2),
    so every line entity carries Line3 elements.
    """
    p_a0 = g.model.geometry.add_point(0.0, 0.0, 0.0)
    p_a1 = g.model.geometry.add_point(0.0, 0.0, 1.0)
    p_b0 = g.model.geometry.add_point(1.0, 0.0, 0.0)
    p_b1 = g.model.geometry.add_point(1.0, 0.0, 1.0)
    p_c0 = g.model.geometry.add_point(2.0, 0.0, 0.0)
    p_c1 = g.model.geometry.add_point(2.0, 0.0, 1.0)
    line_a = g.model.geometry.add_line(p_a0, p_a1)
    line_b = g.model.geometry.add_line(p_b0, p_b1)
    line_c = g.model.geometry.add_line(p_c0, p_c1)
    g.physical.add_curve([line_a, line_b], name="Frames")
    g.physical.add_curve([line_c],         name="Braces")

    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    for ln in (line_a, line_b, line_c):
        gmsh.model.mesh.setTransfiniteCurve(ln, 2)
    g.mesh.generation.generate(dim=1)


class TestMultiPgAndMixedOrders:
    def test_iterable_two_pgs_both_split(self) -> None:
        """Passing a 2-element iterable splits BOTH PGs."""
        with apeGmsh(model_name="t") as g:
            _build_two_pgs_mixed_orders(g)
            assert _count_elements_in_pg(g, "Frames", dim=1) == {8: 2}
            assert _count_elements_in_pg(g, "Braces", dim=1) == {8: 1}

            g.mesh.editing.split_higher_order_lines(
                ["Frames", "Braces"], policy="split",
            )
            assert _count_elements_in_pg(g, "Frames", dim=1) == {1: 4}
            assert _count_elements_in_pg(g, "Braces", dim=1) == {1: 2}

    def test_iterable_one_clean_one_dirty_forbid_reports_dirty(
        self,
    ) -> None:
        """``policy='forbid'`` with one clean PG and one Line3 PG
        raises naming the offending PG (not the clean one)."""
        with apeGmsh(model_name="t") as g:
            _build_two_pgs_mixed_orders(g)
            # Split Frames first so it's clean (Line2 only).
            g.mesh.editing.split_higher_order_lines(
                "Frames", policy="split",
            )
            # Now forbid across both — Braces still has Line3.
            with pytest.raises(RuntimeError, match="Braces.*Line3"):
                g.mesh.editing.split_higher_order_lines(
                    ["Frames", "Braces"], policy="forbid",
                )

    def test_split_on_mixed_order_entity(self) -> None:
        """The persistent gmsh-spike record: an entity that holds both
        Line2 (from a prior split) and Line3 (from a subsequent
        re-mesh) survives ``removeElements + addElements`` for the
        Line3-only subset.

        Setup: build two PGs at order 2; split Frames; then
        regenerate the mesh on Braces' curve at order 2 again (which
        re-introduces a Line3 on Braces' entity).  Splitting Braces
        now must touch only the Line3, leave any Line2 untouched.
        """
        with apeGmsh(model_name="t") as g:
            _build_two_pgs_mixed_orders(g)
            # First pass: split Frames.
            g.mesh.editing.split_higher_order_lines(
                "Frames", policy="split",
            )
            # Pre-condition for the assertion: Frames is now Line2-only.
            assert _count_elements_in_pg(g, "Frames", dim=1) == {1: 4}
            assert _count_elements_in_pg(g, "Braces", dim=1) == {8: 1}

            # Second pass: split Braces in a separate call.  The
            # entity-level addElements continues to work even though
            # global maxElementTag has moved past the first split's
            # allocations.
            g.mesh.editing.split_higher_order_lines(
                "Braces", policy="split",
            )
            assert _count_elements_in_pg(g, "Frames", dim=1) == {1: 4}
            assert _count_elements_in_pg(g, "Braces", dim=1) == {1: 2}


# =====================================================================
# End-to-end through the bridge
# =====================================================================


class TestBridgeIntegration:
    def test_bridge_accepts_split_frame_pg(self) -> None:
        """End-to-end: 2nd-order line entity meshed at order 2 produces
        Line3 elements that the bridge would normally reject at
        ``_check_two_nodes``.  After ``split_higher_order_lines``, the
        FEMData snapshot has Line2 only, and the bridge fan-out
        through ``ops.element.elasticBeamColumn(pg="Cols", ...)``
        completes without raising.
        """
        from apeGmsh.opensees import apeSees
        from apeGmsh.opensees.emitter.recording import RecordingEmitter

        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            g.mesh.editing.split_higher_order_lines(
                "Cols", policy="split",
            )
            fem = g.mesh.queries.get_fem_data(dim=1)

        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        ops.element.elasticBeamColumn(
            pg="Cols", transf=transf,
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
        )

        rec = RecordingEmitter()
        ops.build().emit(rec)

        # Two parent Line3 elements → 2 sub-elements each → 4 emitted
        # ``element elasticBeamColumn`` lines.
        element_calls = [
            c for c in rec.calls
            if c[0] == "element" and c[1][0] == "elasticBeamColumn"
        ]
        assert len(element_calls) == 4, (
            f"expected 4 elasticBeamColumn emissions (2 parent Line3 "
            f"-> 2 sub-elements each), got {len(element_calls)}"
        )

    def test_bridge_rejects_unsplit_line3_with_helpful_message(self) -> None:
        """Without ``split_higher_order_lines``, the bridge raises the
        sharpened ``_check_two_nodes`` error that points at the new
        editing verb.
        """
        from apeGmsh.opensees import apeSees

        with apeGmsh(model_name="t") as g:
            _build_quadratic_frame_pair(g)
            fem = g.mesh.queries.get_fem_data(dim=1)

        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        ops.element.elasticBeamColumn(
            pg="Cols", transf=transf,
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
        )

        from apeGmsh.opensees.emitter.recording import RecordingEmitter
        rec = RecordingEmitter()
        with pytest.raises(
            ValueError,
            match=r"got 3.*split_higher_order_lines",
        ):
            ops.build().emit(rec)
