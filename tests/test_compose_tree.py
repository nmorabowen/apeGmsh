"""compose_tree() derived view — reconstructs nested-compose tree
from the flat ``composed_from`` storage shipped by PR #369.

PR #369 ships flat graft as canonical storage; this PR ships the
derived tree view that follow-up callers (viewer ColorMode.MODULE,
docs, debugging) need. The tree is built by parsing each joined
label via the separator-alternation rule (depth-1 ``.``, depth-2
``/``, depth-3 ``.``, ...) and stitching child records under their
parents.

Tests are pure FEMData / H5 (no Gmsh, no OpenSeesPy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh.mesh._compose import (
    ComposeError,
    ComposeTreeNode,
    _build_compose_tree,
    _join_module_label,
    _split_joined_label,
)
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


# ---------------------------------------------------------------------------
# Fixture builders (mirrors test_phase_3e_1.py shape).
# ---------------------------------------------------------------------------


def _make_fem(
    *,
    node_ids: "list[int] | np.ndarray",
    elem_ids: "list[int] | np.ndarray",
) -> FEMData:
    """Tiny FEMData with one Line2 group; empty arrays are supported."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    elem_ids = np.asarray(elem_ids, dtype=np.int64)
    n = node_ids.size
    if n > 0:
        node_coords = np.array(
            [[float(i), 0.0, 0.0] for i in range(n)],
            dtype=np.float64,
        )
    else:
        node_coords = np.zeros((0, 3), dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    if elem_ids.size > 0:
        conn = np.array(
            [
                [int(node_ids[i % n]), int(node_ids[(i + 1) % n])]
                for i in range(elem_ids.size)
            ],
            dtype=np.int64,
        )
    else:
        conn = np.zeros((0, 2), dtype=np.int64)
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )
    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords,
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}),
        labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=n,
        n_elems=elem_ids.size,
        bandwidth=1,
        types=[line_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


@pytest.fixture
def empty_h5(tmp_path: Path) -> Path:
    """Empty FEMData — serves as the chain-phase 'empty host'."""
    fem = _make_fem(node_ids=[], elem_ids=[])
    p = tmp_path / "empty.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def leaf_h5(tmp_path: Path) -> Path:
    """A depth-0 source: uncomposed leaf FEMData."""
    fem = _make_fem(node_ids=[1, 2, 3], elem_ids=[10, 11])
    p = tmp_path / "leaf.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def leaf2_h5(tmp_path: Path) -> Path:
    """A second depth-0 source for sibling-compose scenarios."""
    fem = _make_fem(node_ids=[100, 101, 102], elem_ids=[200, 201])
    p = tmp_path / "leaf2.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def depth_1_h5(tmp_path: Path, empty_h5: Path, leaf_h5: Path) -> Path:
    """Depth-1 source: ``[partA]``."""
    g = apeGmsh.from_h5(empty_h5)
    g.compose(leaf_h5, label="partA")
    out = tmp_path / "depth_1.h5"
    g.save(out)
    return out


@pytest.fixture
def depth_2_h5(
    tmp_path: Path, empty_h5: Path, depth_1_h5: Path,
) -> Path:
    """Depth-2 source: ``[assemblyM, assemblyM/partA]``."""
    g = apeGmsh.from_h5(empty_h5)
    g.compose(depth_1_h5, label="assemblyM")
    out = tmp_path / "depth_2.h5"
    g.save(out)
    return out


# ---------------------------------------------------------------------------
# _split_joined_label — the parser
# ---------------------------------------------------------------------------


class TestSplitJoinedLabel:
    """Inverse of :func:`_join_module_label`."""

    def test_empty_label_returns_empty_tuple(self) -> None:
        assert _split_joined_label("") == ()

    def test_depth_1_label_is_single_component(self) -> None:
        assert _split_joined_label("partA") == ("partA",)

    def test_depth_2_uses_slash(self) -> None:
        assert _split_joined_label("outer/inner") == ("outer", "inner")

    def test_depth_3_dot_slash(self) -> None:
        # Leftmost sep is depth-3 (".") then depth-2 ("/").
        assert _split_joined_label("top.middle/inner") == (
            "top", "middle", "inner",
        )

    def test_depth_4_slash_dot_slash(self) -> None:
        # Leftmost is depth-4 ("/") then depth-3 (".") then depth-2 ("/").
        assert _split_joined_label("a/b.c/d") == ("a", "b", "c", "d")

    def test_depth_5_dot_slash_dot_slash(self) -> None:
        # Depth-5 (".") depth-4 ("/") depth-3 (".") depth-2 ("/").
        assert _split_joined_label("a.b/c.d/e") == (
            "a", "b", "c", "d", "e",
        )

    def test_wrong_outermost_separator_raises(self) -> None:
        """``top/foo/bar`` has 2 separators (depth 3) but the
        leftmost is ``/`` where ``.`` is expected."""
        with pytest.raises(ComposeError, match="separator-alternation"):
            _split_joined_label("top/foo/bar")

    def test_wrong_inner_separator_raises(self) -> None:
        """``a.b.c`` has 2 separators (depth 3). Leftmost ``.`` is
        correct for depth 3, but the next must be ``/`` at depth 2."""
        with pytest.raises(ComposeError, match="separator-alternation"):
            _split_joined_label("a.b.c")

    def test_malformed_labels_fail_loud(self) -> None:
        """Adjacent separators, leading/trailing separators, and other
        malformed labels are not round-trippable — surface as
        :class:`ComposeError`.  These often trigger the alternation
        check before the empty-component check (which is the
        defensive fallback); both branches raise ``ComposeError``."""
        with pytest.raises(ComposeError):
            _split_joined_label("a..b")
        # Leading separator at depth-2 has 1 sep → expected ``/``.
        with pytest.raises(ComposeError):
            _split_joined_label("/abc")
        # Trailing separator.
        with pytest.raises(ComposeError):
            _split_joined_label("abc/")


class TestSplitJoinRoundTrip:
    """Round-trip property: every valid joined label survives
    split → re-join unchanged."""

    @pytest.mark.parametrize("label", [
        "partA",
        "bayP",
        "outer/inner",
        "bayP/assemblyM",
        "top.middle/inner",
        "bayP.assemblyM/partA",
        "a/b.c/d",
        "a.b/c.d/e",
    ])
    def test_round_trip_join_split_join(self, label: str) -> None:
        parts = _split_joined_label(label)
        # Reconstruct inside-out — same convention as the tree
        # builder uses internally.
        inner = parts[-1]
        n = len(parts)
        for k in range(n - 1, 0, -1):
            outer = parts[k - 1]
            result_depth = n - k + 1
            inner = _join_module_label(
                outer, inner, result_depth=result_depth,
            )
        assert inner == label


# ---------------------------------------------------------------------------
# _build_compose_tree — direct unit tests on synthetic records
# ---------------------------------------------------------------------------


def _rec(label: str, source_path: str = "x.h5") -> ComposeRecord:
    """Minimal ComposeRecord factory for tree-builder unit tests."""
    return ComposeRecord(
        label=label,
        source_path=source_path,
        source_fem_hash="hash_" + label.replace("/", "_").replace(".", "_"),
        source_neutral_schema_version="2.9.0",
        translate=(0.0, 0.0, 0.0),
    )


class TestBuildComposeTree:
    """Pure-function tests on synthetic ComposeRecord tuples — no H5."""

    def test_empty_records_returns_empty_tuple(self) -> None:
        assert _build_compose_tree(()) == ()

    def test_single_depth_1(self) -> None:
        r = _rec("partA")
        tree = _build_compose_tree((r,))
        assert len(tree) == 1
        assert isinstance(tree[0], ComposeTreeNode)
        assert tree[0].label == "partA"
        assert tree[0].record is r
        assert tree[0].children == ()

    def test_two_depth_1_siblings(self) -> None:
        a = _rec("A")
        b = _rec("B")
        tree = _build_compose_tree((a, b))
        labels = [n.label for n in tree]
        # Roots are sorted by component name (ComposeSet itself is
        # also sorted, but the builder sorts internally for
        # determinism on caller-supplied tuples).
        assert labels == ["A", "B"]
        assert tree[0].children == ()
        assert tree[1].children == ()

    def test_depth_2_one_parent_one_child(self) -> None:
        # Mirrors the flat-graft contract: parent + grafted child.
        outer = _rec("outer")
        child = _rec("outer/partA")
        tree = _build_compose_tree((outer, child))
        assert len(tree) == 1
        root = tree[0]
        assert root.label == "outer"
        assert root.record is outer
        assert len(root.children) == 1
        assert root.children[0].label == "partA"
        assert root.children[0].record is child
        assert root.children[0].children == ()

    def test_depth_3_three_levels(self) -> None:
        # Flat representation of a depth-3 chain.
        bayP = _rec("bayP")
        bayP_assemblyM = _rec("bayP/assemblyM")
        bayP_assemblyM_partA = _rec("bayP.assemblyM/partA")
        tree = _build_compose_tree(
            (bayP, bayP_assemblyM, bayP_assemblyM_partA),
        )
        assert len(tree) == 1
        root = tree[0]
        assert root.label == "bayP" and root.record is bayP
        assert len(root.children) == 1
        mid = root.children[0]
        assert mid.label == "assemblyM" and mid.record is bayP_assemblyM
        assert len(mid.children) == 1
        leaf = mid.children[0]
        assert leaf.label == "partA"
        assert leaf.record is bayP_assemblyM_partA
        assert leaf.children == ()

    def test_mixed_depth_siblings(self) -> None:
        """One root has a child (and grandchild); another sibling
        root is a bare leaf — verifies the tree handles mixed
        depths under different roots."""
        # Root A: depth-1 only.
        a = _rec("A")
        # Root B: depth-1 + depth-2 + depth-3 chain.
        b = _rec("B")
        b_inner = _rec("B/inner")
        b_inner_leaf = _rec("B.inner/leaf")
        tree = _build_compose_tree((a, b, b_inner, b_inner_leaf))
        labels = [n.label for n in tree]
        assert labels == ["A", "B"]
        # A — no children.
        assert tree[0].children == ()
        # B — chain of inner → leaf.
        root_b = tree[1]
        assert len(root_b.children) == 1
        inner = root_b.children[0]
        assert inner.label == "inner"
        assert inner.record is b_inner
        assert len(inner.children) == 1
        assert inner.children[0].label == "leaf"
        assert inner.children[0].record is b_inner_leaf

    def test_node_is_frozen_dataclass(self) -> None:
        node = _build_compose_tree((_rec("X"),))[0]
        with pytest.raises((AttributeError, Exception)):
            # Frozen dataclass: assignment to label raises.
            node.label = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FEMData.compose_tree() / Compose.compose_tree() end-to-end
# ---------------------------------------------------------------------------


class TestComposeTreeAPI:
    """End-to-end: compose into an apeGmsh session, then walk the
    derived tree."""

    def test_uncomposed_returns_empty(self, empty_h5: Path) -> None:
        g = apeGmsh.from_h5(empty_h5)
        # Both the session shim and the canonical FEMData primitive
        # return ().
        assert g.compose_tree() == ()
        assert g._fem.compose_tree() == ()

    def test_compose_facade_method_present(
        self, empty_h5: Path,
    ) -> None:
        """Facade exposes ``compose_tree`` alongside ``compose_list``."""
        g = apeGmsh.from_h5(empty_h5)
        facade = g._compose_facade()
        assert hasattr(facade, "compose_tree")
        assert facade.compose_tree() == ()

    def test_single_depth_1_compose(
        self, empty_h5: Path, leaf_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(leaf_h5, label="partA")
        tree = g.compose_tree()
        assert len(tree) == 1
        assert tree[0].label == "partA"
        assert tree[0].record.label == "partA"
        assert tree[0].children == ()

    def test_two_sibling_depth_1_composes(
        self, empty_h5: Path, leaf_h5: Path, leaf2_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(leaf_h5, label="A")
        g.compose(leaf2_h5, label="B")
        tree = g.compose_tree()
        labels = sorted(n.label for n in tree)
        assert labels == ["A", "B"]
        for node in tree:
            assert node.children == ()

    def test_depth_2_nested(
        self, empty_h5: Path, depth_1_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_1_h5, label="outer")
        tree = g.compose_tree()
        assert len(tree) == 1
        root = tree[0]
        assert root.label == "outer"
        # The root record is the joined label "outer" (depth-1).
        assert root.record.label == "outer"
        # The child record is the grafted "outer/partA".
        assert len(root.children) == 1
        child = root.children[0]
        assert child.label == "partA"
        assert child.record.label == "outer/partA"

    def test_depth_3_nested(
        self, empty_h5: Path, depth_2_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_2_h5, label="bayP")
        tree = g.compose_tree()
        assert len(tree) == 1
        root = tree[0]
        assert root.label == "bayP"
        assert root.record.label == "bayP"
        assert len(root.children) == 1
        mid = root.children[0]
        assert mid.label == "assemblyM"
        # Depth-2 separator = "/" → joined label is "bayP/assemblyM".
        assert mid.record.label == "bayP/assemblyM"
        assert len(mid.children) == 1
        leaf = mid.children[0]
        assert leaf.label == "partA"
        # Depth-3 separator = "." at outer, "/" at inner →
        # "bayP.assemblyM/partA".
        assert leaf.record.label == "bayP.assemblyM/partA"
        assert leaf.children == ()

    def test_mixed_depth_siblings_e2e(
        self,
        empty_h5: Path,
        leaf_h5: Path,
        depth_1_h5: Path,
    ) -> None:
        """Compose a depth-0 leaf and a depth-1 source side-by-side.
        Result: two roots, one with a child, one without."""
        g = apeGmsh.from_h5(empty_h5)
        g.compose(leaf_h5, label="bareA")
        g.compose(depth_1_h5, label="nestedB")
        tree = g.compose_tree()
        labels = sorted(n.label for n in tree)
        assert labels == ["bareA", "nestedB"]
        bareA = next(n for n in tree if n.label == "bareA")
        nestedB = next(n for n in tree if n.label == "nestedB")
        assert bareA.children == ()
        assert len(nestedB.children) == 1
        assert nestedB.children[0].label == "partA"
        assert nestedB.children[0].record.label == "nestedB/partA"

    def test_round_trip_tree_labels_match_flat(
        self, empty_h5: Path, depth_2_h5: Path,
    ) -> None:
        """Walking the tree and collecting every record's joined
        label reproduces the flat ``composed_from`` list (sorted)."""
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_2_h5, label="bayP")
        flat_labels = sorted(r.label for r in g._fem.composed_from)

        def _walk(nodes: "tuple[ComposeTreeNode, ...]") -> list[str]:
            out: list[str] = []
            for n in nodes:
                out.append(n.record.label)
                out.extend(_walk(n.children))
            return out

        tree_labels = sorted(_walk(g.compose_tree()))
        assert tree_labels == flat_labels

    def test_record_fidelity(
        self, empty_h5: Path, leaf_h5: Path,
    ) -> None:
        """Each tree node's ``record`` is the same ComposeRecord
        instance that ``fem.composed_from`` carries."""
        g = apeGmsh.from_h5(empty_h5)
        g.compose(leaf_h5, label="modA")
        tree = g.compose_tree()
        flat = g._fem.composed_from["modA"]
        assert tree[0].record is flat or tree[0].record == flat
        # Compare a couple of attributes that matter for callers.
        assert tree[0].record.label == flat.label
        assert tree[0].record.source_path == flat.source_path
        assert tree[0].record.source_fem_hash == flat.source_fem_hash

    def test_fem_compose_tree_matches_session(
        self, empty_h5: Path, depth_1_h5: Path,
    ) -> None:
        """The session shim delegates to the canonical primitive on
        FEMData — both return identical trees."""
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_1_h5, label="modX")
        session_tree = g.compose_tree()
        fem_tree = g._fem.compose_tree()
        # Compare by structure: equal tuple of equal ComposeTreeNodes.
        assert session_tree == fem_tree


class TestComposeTreeParserFailLoud:
    """Synthetic record with a malformed joined label exercises the
    parser's fail-loud branch end-to-end through ``compose_tree``."""

    def test_malformed_label_raises_through_facade(self) -> None:
        # Construct a FEMData by hand with a malformed compose record
        # (depth-3 label that uses "/" where "." is expected at the
        # outermost boundary).  This is the case the spec calls out.
        fem = _make_fem(node_ids=[], elem_ids=[])
        bad = _rec("top/foo/bar")
        # Inject directly — we want to exercise the parser, not
        # whatever upstream validator might have rejected this.
        from apeGmsh._kernel.record_sets import ComposeSet
        fem.composed_from = ComposeSet((bad,))
        with pytest.raises(ComposeError, match="separator-alternation"):
            fem.compose_tree()
