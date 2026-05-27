"""Phase 3E.1 — Nested compose: depth verifier, separator alternation,
recursive provenance graft.

Locks ADR 0038 §"Nested composition":

1. **Depth limit** — composing a source whose own ``composed_from``
   chain already sits at ``max_compose_depth`` (default 3) raises
   :class:`ComposeDepthExceededError`.  ``max_compose_depth=N``
   per-call or :data:`Compose.MAX_COMPOSE_DEPTH` class-level lifts
   the cap.
2. **Separator alternation** — the outer namespace separator
   alternates ``.`` ↔ ``/`` per compose depth so nested labels
   remain unambiguous on parse.  Convention: depth 1 → ``.``,
   depth 2 → ``/``, depth 3 → ``.``, … (odd = ``.``, even = ``/``).
3. **Provenance graft (flat)** — the source's own ``composed_from``
   records surface in the host's flat ``composed_from`` chain with
   their labels re-prefixed via the depth-N rule.  H5 round-trip
   preserves the joined labels via the existing 2.9.0 schema
   without further field additions.

These tests are pure FEMData / H5 (no Gmsh, no OpenSeesPy).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.records._compose import ComposeRecord
from apeGmsh.core._compose_errors import (
    ComposeDepthExceededError as CoreComposeDepthExceededError,
)
from apeGmsh.mesh._compose import (
    DEFAULT_MAX_COMPOSE_DEPTH,
    Compose,
    ComposeDepthExceededError,
    _compose_depth_of_records,
    _join_module_label,
    _label_depth,
    _prefix_namespaced_name,
    _read_source_composed_from,
    _separator_for_depth,
)
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees.emitter import h5_reader


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_fem(
    *,
    node_ids: "list[int] | np.ndarray",
    elem_ids: "list[int] | np.ndarray",
) -> FEMData:
    """Tiny single-Line2 FEMData (no compose state); supports empty
    fixtures so we can build an empty host for nested-compose tests
    without ambiguous "host has 1 record from from_h5" semantics.
    """
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
    """Empty FEMData saved to H5 — serves as the "fresh uncomposed
    host" for chain-phase compose tests."""
    fem = _make_fem(node_ids=[], elem_ids=[])
    p = tmp_path / "empty.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def leaf_h5(tmp_path: Path) -> Path:
    """Depth-0 source: a leaf FEMData with no compose state."""
    fem = _make_fem(node_ids=[1, 2, 3], elem_ids=[10, 11])
    p = tmp_path / "leaf.h5"
    fem.to_h5(str(p))
    return p


@pytest.fixture
def depth_1_h5(tmp_path: Path, empty_h5: Path, leaf_h5: Path) -> Path:
    """A depth-1 source: leaf composed under label ``partA`` into an
    empty host, saved.  Its ``composed_from`` is ``[partA]``."""
    g = apeGmsh.from_h5(empty_h5)
    g.compose(leaf_h5, label="partA")
    out = tmp_path / "depth_1.h5"
    g.save(out)
    return out


@pytest.fixture
def depth_2_h5(
    tmp_path: Path, empty_h5: Path, depth_1_h5: Path,
) -> Path:
    """A depth-2 source: depth-1 file composed under label
    ``assemblyM`` into an empty host, saved.  Its ``composed_from``
    is ``[assemblyM, assemblyM/partA]``."""
    g = apeGmsh.from_h5(empty_h5)
    g.compose(depth_1_h5, label="assemblyM")
    out = tmp_path / "depth_2.h5"
    g.save(out)
    return out


@pytest.fixture
def depth_3_h5(
    tmp_path: Path, empty_h5: Path, depth_2_h5: Path,
) -> Path:
    """A depth-3 source: depth-2 file composed under label ``bayP``
    into an empty host, saved.  Its ``composed_from`` is
    ``[bayP, bayP.assemblyM/partA, bayP/assemblyM]``."""
    g = apeGmsh.from_h5(empty_h5)
    g.compose(depth_2_h5, label="bayP")
    out = tmp_path / "depth_3.h5"
    g.save(out)
    return out


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


class TestLabelDepth:
    """Unit tests for the depth-counting + separator helpers."""

    def test_default_max_compose_depth_is_3(self) -> None:
        assert DEFAULT_MAX_COMPOSE_DEPTH == 3

    def test_compose_class_constant_mirrors_module_default(self) -> None:
        assert Compose.MAX_COMPOSE_DEPTH == DEFAULT_MAX_COMPOSE_DEPTH

    def test_label_depth_empty_is_zero(self) -> None:
        assert _label_depth("") == 0

    def test_label_depth_leaf_is_one(self) -> None:
        assert _label_depth("bolt") == 1

    def test_label_depth_one_sep_is_two(self) -> None:
        assert _label_depth("partA.bolt_head") == 2

    def test_label_depth_two_seps_is_three(self) -> None:
        assert _label_depth("assemblyM/partA.bolt_head") == 3

    def test_label_depth_three_seps_is_four(self) -> None:
        assert _label_depth("bayP.assemblyM/partA.bolt_head") == 4

    def test_label_depth_counts_both_separators(self) -> None:
        """Either ``.`` or ``/`` counts as one depth boundary."""
        assert _label_depth("a.b") == _label_depth("a/b") == 2
        assert _label_depth("a.b/c") == _label_depth("a/b.c") == 3

    def test_compose_depth_of_records_empty(self) -> None:
        assert _compose_depth_of_records(()) == 0

    def test_compose_depth_of_records_max_across_chain(self) -> None:
        rec1 = ComposeRecord(
            label="leaf",
            source_path="x.h5",
            source_fem_hash="h",
            source_neutral_schema_version="2.9.0",
            translate=(0.0, 0.0, 0.0),
        )
        rec2 = ComposeRecord(
            label="leaf2/inner.deeper",
            source_path="x.h5",
            source_fem_hash="h",
            source_neutral_schema_version="2.9.0",
            translate=(0.0, 0.0, 0.0),
        )
        assert _compose_depth_of_records((rec1, rec2)) == 3


class TestSeparatorAlternation:
    """The ``.`` ↔ ``/`` alternation rule (Phase 3E.1)."""

    def test_depth_1_uses_dot(self) -> None:
        assert _separator_for_depth(1) == "."

    def test_depth_2_uses_slash(self) -> None:
        assert _separator_for_depth(2) == "/"

    def test_depth_3_uses_dot(self) -> None:
        assert _separator_for_depth(3) == "."

    def test_depth_4_uses_slash(self) -> None:
        assert _separator_for_depth(4) == "/"

    def test_depth_0_raises(self) -> None:
        with pytest.raises(ValueError, match="depth 0"):
            _separator_for_depth(0)

    def test_join_module_label_depth_2(self) -> None:
        # Inner is a depth-1 compose label "leaf" → outer at depth 2.
        # The joined label's depth = 2; separator = "/".
        assert _join_module_label("outer", "leaf", result_depth=2) == (
            "outer/leaf"
        )

    def test_join_module_label_depth_3(self) -> None:
        # Inner is a depth-2 compose label "frame/conn" → outer at
        # depth 3.  Separator = ".".
        assert _join_module_label(
            "outer", "frame/conn", result_depth=3,
        ) == "outer.frame/conn"

    def test_join_module_label_empty_inner(self) -> None:
        assert _join_module_label(
            "outer", "", result_depth=1,
        ) == "outer"

    def test_prefix_namespaced_name_leaf(self) -> None:
        # A PG / label / part name from an uncomposed source has 0 seps
        # → outer prefix at depth 1 = ".".
        assert _prefix_namespaced_name(
            "outer", "top_flange",
        ) == "outer.top_flange"

    def test_prefix_namespaced_name_one_sep(self) -> None:
        # 1 sep in inner → outer prefix at depth 2 = "/".
        assert _prefix_namespaced_name(
            "outer", "partA.bolt",
        ) == "outer/partA.bolt"

    def test_prefix_namespaced_name_two_seps(self) -> None:
        # 2 seps in inner → outer prefix at depth 3 = ".".
        assert _prefix_namespaced_name(
            "outer", "frame/conn.bolt",
        ) == "outer.frame/conn.bolt"

    def test_prefix_namespaced_name_none(self) -> None:
        assert _prefix_namespaced_name("outer", None) is None


class TestReadSourceComposedFrom:
    """The H5 helper that probes ``/composed_from/`` without the full
    ``read_fem_h5`` cost."""

    def test_uncomposed_source_returns_empty(
        self, leaf_h5: Path,
    ) -> None:
        assert _read_source_composed_from(leaf_h5) == ()

    def test_depth_1_source_returns_one_record(
        self, depth_1_h5: Path,
    ) -> None:
        records = _read_source_composed_from(depth_1_h5)
        assert len(records) == 1
        assert records[0].label == "partA"

    def test_depth_2_source_returns_both_records(
        self, depth_2_h5: Path,
    ) -> None:
        records = _read_source_composed_from(depth_2_h5)
        labels = sorted(r.label for r in records)
        assert labels == ["assemblyM", "assemblyM/partA"]


# ---------------------------------------------------------------------------
# End-to-end nested compose
# ---------------------------------------------------------------------------


class TestDepthTracking:
    """Composing a source extends the resulting compose chain by one
    level; the depth check fires when the cap is exceeded."""

    def test_depth_0_compose_produces_depth_1(
        self, empty_h5: Path, leaf_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(leaf_h5, label="A")
        labels = [r.label for r in g._fem.composed_from]
        assert labels == ["A"]
        assert _compose_depth_of_records(
            tuple(g._fem.composed_from),
        ) == 1

    def test_depth_1_compose_produces_depth_2(
        self, empty_h5: Path, depth_1_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_1_h5, label="M2")
        labels = sorted(r.label for r in g._fem.composed_from)
        assert labels == ["M2", "M2/partA"]
        assert _compose_depth_of_records(
            tuple(g._fem.composed_from),
        ) == 2

    def test_depth_2_compose_produces_depth_3(
        self, empty_h5: Path, depth_2_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_2_h5, label="M3")
        labels = sorted(r.label for r in g._fem.composed_from)
        assert labels == [
            "M3", "M3.assemblyM/partA", "M3/assemblyM",
        ]
        assert _compose_depth_of_records(
            tuple(g._fem.composed_from),
        ) == 3

    def test_depth_3_compose_raises(
        self, empty_h5: Path, depth_3_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        with pytest.raises(ComposeDepthExceededError) as ei:
            g.compose(depth_3_h5, label="topLevel")
        # Message names host's source depth (3) and the cap (3).
        msg = str(ei.value)
        assert "max_compose_depth=3" in msg
        assert "depth is 3" in msg

    def test_depth_exceeded_is_core_error(
        self, empty_h5: Path, depth_3_h5: Path,
    ) -> None:
        """The facade exception inherits from the core canonical class
        so callers using ``except CoreComposeDepthExceededError`` catch
        it from outside the mesh package."""
        g = apeGmsh.from_h5(empty_h5)
        with pytest.raises(CoreComposeDepthExceededError):
            g.compose(depth_3_h5, label="topLevel")

    def test_depth_exceeded_also_value_error(
        self, empty_h5: Path, depth_3_h5: Path,
    ) -> None:
        """``ValueError`` continues to catch it — backward compat."""
        g = apeGmsh.from_h5(empty_h5)
        with pytest.raises(ValueError):
            g.compose(depth_3_h5, label="topLevel")


class TestMaxDepthOverride:
    """The per-call kwarg lifts the depth cap for a single compose."""

    def test_override_allows_deeper(
        self, empty_h5: Path, depth_3_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        # Cap = 4 lets the depth-3 source compose (yielding depth 4).
        g.compose(depth_3_h5, label="topLevel", max_compose_depth=4)
        labels = sorted(r.label for r in g._fem.composed_from)
        assert "topLevel" in labels
        # The result has at least one depth-4 entry (joined three times).
        assert max(_label_depth(r.label) for r in g._fem.composed_from) == 4

    def test_class_level_constant_can_be_lifted(
        self, empty_h5: Path, depth_3_h5: Path,
    ) -> None:
        """Subclassing :class:`Compose` with ``MAX_COMPOSE_DEPTH = 5``
        lifts the cap class-wide without per-call kwargs."""
        # Build a custom session class with a wider cap.
        g = apeGmsh.from_h5(empty_h5)
        original = Compose.MAX_COMPOSE_DEPTH
        try:
            Compose.MAX_COMPOSE_DEPTH = 5
            g.compose(depth_3_h5, label="topLevel")
            labels = [r.label for r in g._fem.composed_from]
            assert "topLevel" in labels
        finally:
            Compose.MAX_COMPOSE_DEPTH = original

    def test_override_too_small_raises(
        self, empty_h5: Path, depth_1_h5: Path,
    ) -> None:
        """Tightening the cap below the source's depth makes a
        previously-legal compose fail."""
        g = apeGmsh.from_h5(empty_h5)
        with pytest.raises(ComposeDepthExceededError):
            g.compose(depth_1_h5, label="X", max_compose_depth=1)

    def test_override_invalid_type_raises(
        self, empty_h5: Path, leaf_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        with pytest.raises(ValueError, match="must be an int"):
            g.compose(leaf_h5, label="X", max_compose_depth="three")  # type: ignore[arg-type]

    def test_override_below_one_raises(
        self, empty_h5: Path, leaf_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        with pytest.raises(ValueError, match=">= 1"):
            g.compose(leaf_h5, label="X", max_compose_depth=0)


class TestSeparatorAlternationEndToEnd:
    """Composing nested sources produces the expected joined labels
    with depth-N separator alternation."""

    def test_depth_2_uses_slash(
        self, empty_h5: Path, depth_1_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_1_h5, label="outer")
        labels = sorted(r.label for r in g._fem.composed_from)
        # Top-level "outer" + grafted "outer/partA" (depth-2 boundary
        # uses "/").
        assert "outer" in labels
        assert "outer/partA" in labels
        assert "outer.partA" not in labels  # no "." at depth 2

    def test_depth_3_uses_dot_at_outer_boundary(
        self, empty_h5: Path, depth_2_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_2_h5, label="top")
        labels = sorted(r.label for r in g._fem.composed_from)
        # Top-level "top" + grafted "top/assemblyM" (depth-2 inner
        # boundary uses "/") + grafted "top.assemblyM/partA"
        # (depth-3 boundary uses "." outer, "/" preserved inner).
        assert "top" in labels
        assert "top/assemblyM" in labels
        assert "top.assemblyM/partA" in labels

    def test_module_label_on_nodes_is_joined(
        self, empty_h5: Path, depth_1_h5: Path,
    ) -> None:
        """A depth-2 compose stamps the source's depth-1 rows with
        ``{outer}/{inner}`` on the merged ``module_label`` parallel
        dataset."""
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_1_h5, label="outer")
        # All nodes came from depth_1_h5, whose own rows had
        # module_label == "partA".  After compose: "outer/partA".
        ml = g._fem.nodes._module_label
        assert ml is not None
        # The empty host contributed 0 rows; all rows are from the
        # composed module → all labels are "outer/partA".
        labels = set(str(x) for x in ml)
        assert labels == {"outer/partA"}

    def test_module_label_on_elements_is_joined(
        self, empty_h5: Path, depth_1_h5: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_1_h5, label="outer")
        ml = g._fem.elements._module_label
        assert ml is not None
        for arr in ml.values():
            labels = set(str(x) for x in arr)
            assert labels == {"outer/partA"}


# ---------------------------------------------------------------------------
# H5 round-trip
# ---------------------------------------------------------------------------


class TestH5RoundTripNested:
    """The 2.9.0 schema preserves nested compose-records and joined
    module_labels across save/load cycles."""

    def test_round_trip_preserves_labels(
        self, empty_h5: Path, depth_2_h5: Path, tmp_path: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_2_h5, label="top")
        out = tmp_path / "round.h5"
        g.save(out)
        # Reload and compare labels.
        g2 = apeGmsh.from_h5(out)
        loaded_labels = sorted(r.label for r in g2._fem.composed_from)
        assert loaded_labels == [
            "top", "top.assemblyM/partA", "top/assemblyM",
        ]

    def test_round_trip_preserves_translate(
        self, empty_h5: Path, depth_1_h5: Path, tmp_path: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_1_h5, label="outer", translate=(5.0, 0.0, 0.0))
        out = tmp_path / "round.h5"
        g.save(out)
        g2 = apeGmsh.from_h5(out)
        # The top-level ComposeRecord carries the translate.
        outer_rec = g2._fem.composed_from["outer"]
        assert outer_rec.translate == (5.0, 0.0, 0.0)

    def test_round_trip_module_label_for_node(
        self, empty_h5: Path, depth_1_h5: Path, tmp_path: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_1_h5, label="outer")
        out = tmp_path / "round.h5"
        g.save(out)
        with h5_reader.open(str(out)) as model:
            ids = model.nodes()["ids"]
            for nid in ids:
                # Every node came from the composed module at depth 2.
                assert model.composed_for_node(int(nid)) == "outer/partA"

    def test_iter_composed_from_yields_nested_labels(
        self, empty_h5: Path, depth_2_h5: Path, tmp_path: Path,
    ) -> None:
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_2_h5, label="top")
        out = tmp_path / "round.h5"
        g.save(out)
        with h5_reader.open(str(out)) as model:
            labels = sorted(r.label for r in model.iter_composed_from())
        assert labels == [
            "top", "top.assemblyM/partA", "top/assemblyM",
        ]

    def test_double_round_trip_is_stable(
        self, empty_h5: Path, depth_2_h5: Path, tmp_path: Path,
    ) -> None:
        """save → load → save → load yields the same compose chain."""
        g1 = apeGmsh.from_h5(empty_h5)
        g1.compose(depth_2_h5, label="top")
        out1 = tmp_path / "r1.h5"
        g1.save(out1)
        g2 = apeGmsh.from_h5(out1)
        out2 = tmp_path / "r2.h5"
        g2.save(out2)
        g3 = apeGmsh.from_h5(out2)
        first_labels = sorted(r.label for r in g2._fem.composed_from)
        second_labels = sorted(r.label for r in g3._fem.composed_from)
        assert first_labels == second_labels

    def test_h5_uses_safe_group_names_for_slashed_labels(
        self, empty_h5: Path, depth_2_h5: Path, tmp_path: Path,
    ) -> None:
        """The H5 writer sanitises ``/`` to ``_`` for group names but
        round-trips the original label via the ``label`` attribute."""
        g = apeGmsh.from_h5(empty_h5)
        g.compose(depth_2_h5, label="top")
        out = tmp_path / "round.h5"
        g.save(out)
        with h5py.File(str(out), "r") as f:
            assert "composed_from" in f
            cf = f["composed_from"]
            # At least one group has a sanitised name (no "/" in group
            # keys) but its label attr carries the joined label.
            joined_labels: set[str] = set()
            for key in cf.keys():
                assert "/" not in key
                attrs = cf[key].attrs
                if "label" in attrs:
                    raw = attrs["label"]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    joined_labels.add(str(raw))
            assert "top/assemblyM" in joined_labels
            assert "top.assemblyM/partA" in joined_labels


# ---------------------------------------------------------------------------
# Edge cases — multiple modules, sibling composes, mixed inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Behaviour at unusual composition shapes."""

    def test_two_sibling_depth_1_composes_stay_depth_1(
        self, empty_h5: Path, leaf_h5: Path,
    ) -> None:
        """Composing two leaves under different outer labels yields
        two depth-1 entries (sibling modules; no nesting)."""
        g = apeGmsh.from_h5(empty_h5)
        g.compose(leaf_h5, label="A")
        # Build a second distinct leaf source.
        leaf2 = _make_fem(
            node_ids=[100, 200, 300], elem_ids=[1000, 1001],
        )
        leaf2_path = empty_h5.parent / "leaf2.h5"
        leaf2.to_h5(str(leaf2_path))
        g.compose(leaf2_path, label="B")
        labels = sorted(r.label for r in g._fem.composed_from)
        assert labels == ["A", "B"]
        # Sibling composes do NOT count as nested.
        assert max(
            _label_depth(L) for L in labels
        ) == 1

    def test_compose_source_with_multiple_modules(
        self, empty_h5: Path, leaf_h5: Path, tmp_path: Path,
    ) -> None:
        """A source whose own ``composed_from`` has multiple entries —
        the host inherits all of them through the graft."""
        # Build a host that composes two leaves.
        gA = apeGmsh.from_h5(empty_h5)
        gA.compose(leaf_h5, label="A")
        leaf2 = _make_fem(
            node_ids=[100, 200, 300], elem_ids=[1000, 1001],
        )
        leaf2_path = tmp_path / "leaf2.h5"
        leaf2.to_h5(str(leaf2_path))
        gA.compose(leaf2_path, label="B")
        multi_h5 = tmp_path / "multi.h5"
        gA.save(multi_h5)
        # Compose the multi-module source into a fresh host.
        g = apeGmsh.from_h5(empty_h5)
        g.compose(multi_h5, label="outer")
        labels = sorted(r.label for r in g._fem.composed_from)
        # Top-level "outer" + grafted "outer/A" + grafted "outer/B".
        assert labels == ["outer", "outer/A", "outer/B"]

    def test_uncomposed_source_does_not_graft(
        self, empty_h5: Path, leaf_h5: Path,
    ) -> None:
        """Composing a depth-0 source produces a single top-level
        entry; no graft records appear."""
        g = apeGmsh.from_h5(empty_h5)
        g.compose(leaf_h5, label="outer")
        labels = [r.label for r in g._fem.composed_from]
        assert labels == ["outer"]

    def test_compose_inspect_reports_nested_provenance(
        self, depth_2_h5: Path,
    ) -> None:
        """``compose_inspect`` surfaces the source's nested
        ``composed_from`` so callers can audit before composing."""
        from apeGmsh.mesh._compose import Compose

        # Build a session with no host; compose_inspect doesn't need
        # a session (Compose facade reads H5 metadata only).
        class _StubSession:
            _fem = None
            _fem_from_h5 = False

            class _MeshShim:
                class _Queries:
                    @staticmethod
                    def get_fem_data():
                        raise RuntimeError("no live session")

                queries = _Queries()

            mesh = _MeshShim()

        facade = Compose(_StubSession())
        info = facade.compose_inspect(depth_2_h5)
        graft_labels = sorted(r.label for r in info["composed_from"])
        assert graft_labels == ["assemblyM", "assemblyM/partA"]
