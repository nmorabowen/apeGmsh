"""Phase 3E.1 — Nested compose: depth verifier, separator alternation,
recursive provenance graft.

Locks ADR 0038 §"Nested composition", through the v2 ``Assembly``
(ADR 0117): a nested source is an assembly archive (``asm.h5``), and
instancing it in another assembly nests it one level deeper.

1. **Depth limit** — instancing a source whose own ``composed_from``
   chain already sits at the fixed cap of 3 raises
   :class:`ComposeDepthExceededError` at ``bridge()``.  The cap is not
   configurable (ruling G3, #1585).
2. **Separator alternation** — the outer namespace separator
   alternates ``.`` ↔ ``/`` per compose depth so nested labels
   remain unambiguous on parse.  Convention: depth 1 → ``.``,
   depth 2 → ``/``, depth 3 → ``.``, … (odd = ``.``, even = ``/``).
3. **Provenance graft (flat)** — the source's own ``composed_from``
   records surface in the bridge's flat ``composed_from`` chain with
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
from apeGmsh.assembly import Assembly
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
from apeGmsh.opensees import apeSees
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
    fixtures.
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


def _write_source(fem: FEMData, path: Path) -> Path:
    """Write ``fem`` as an instanceable source: ``/opensees`` carries
    ``model(ndm=3, ndf=3)`` (a bare ``fem.to_h5`` file is refused by
    ``bridge()``)."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    ops.h5(str(path))
    return path


def _assemble(*instances, out: "Path | None" = None, **kw):
    """Instance each ``(label, source)`` (plus ``kw`` on every one) in
    one assembly, bridge it, and write the archive to ``out`` if given.
    Returns the bridge (``apeSees``); its merged broker is ``.fem``."""
    asm = Assembly("nested")
    for label, source in instances:
        asm.instance(label, source, **kw)
    ops = asm.bridge(ndm=3, ndf=3)
    if out is not None:
        asm.h5(str(out))
    return ops


@pytest.fixture(scope="module")
def _dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("phase_3e_1")


@pytest.fixture(scope="module")
def leaf_h5(_dir: Path) -> Path:
    """Depth-0 source: a leaf FEMData with no compose state."""
    return _write_source(
        _make_fem(node_ids=[1, 2, 3], elem_ids=[10, 11]), _dir / "leaf.h5",
    )


@pytest.fixture(scope="module")
def leaf2_h5(_dir: Path) -> Path:
    """A second, distinct depth-0 leaf."""
    return _write_source(
        _make_fem(node_ids=[100, 200, 300], elem_ids=[1000, 1001]),
        _dir / "leaf2.h5",
    )


@pytest.fixture(scope="module")
def depth_1_h5(_dir: Path, leaf_h5: Path) -> Path:
    """A depth-1 source: an assembly archive instancing the leaf under
    label ``partA``.  Its ``composed_from`` is ``[partA]``."""
    out = _dir / "depth_1.h5"
    _assemble(("partA", leaf_h5), out=out)
    return out


@pytest.fixture(scope="module")
def depth_2_h5(_dir: Path, depth_1_h5: Path) -> Path:
    """A depth-2 source: the depth-1 archive instanced under label
    ``assemblyM``.  Its ``composed_from`` is
    ``[assemblyM, assemblyM/partA]``."""
    out = _dir / "depth_2.h5"
    _assemble(("assemblyM", depth_1_h5), out=out)
    return out


@pytest.fixture(scope="module")
def depth_3_h5(_dir: Path, depth_2_h5: Path) -> Path:
    """A depth-3 source: the depth-2 archive instanced under label
    ``bayP``.  Its ``composed_from`` is
    ``[bayP, bayP.assemblyM/partA, bayP/assemblyM]``."""
    out = _dir / "depth_3.h5"
    _assemble(("bayP", depth_2_h5), out=out)
    return out


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


class TestLabelDepth:
    """Unit tests for the depth-counting + separator helpers."""

    def test_default_max_compose_depth_is_3(self) -> None:
        assert DEFAULT_MAX_COMPOSE_DEPTH == 3

    def test_compose_class_has_no_depth_knob(self) -> None:
        # Ruling G3 (#1585): the cap is fixed at the module default;
        # the class-level ``Compose.MAX_COMPOSE_DEPTH`` knob is gone.
        assert not hasattr(Compose, "MAX_COMPOSE_DEPTH")

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
    """Instancing a source extends the resulting compose chain by one
    level; the depth check fires when the cap is exceeded."""

    def test_depth_0_compose_produces_depth_1(self, leaf_h5: Path) -> None:
        fem = _assemble(("A", leaf_h5)).fem
        labels = [r.label for r in fem.composed_from]
        assert labels == ["A"]
        assert _compose_depth_of_records(tuple(fem.composed_from)) == 1

    def test_depth_1_compose_produces_depth_2(
        self, depth_1_h5: Path,
    ) -> None:
        fem = _assemble(("M2", depth_1_h5)).fem
        labels = sorted(r.label for r in fem.composed_from)
        assert labels == ["M2", "M2/partA"]
        assert _compose_depth_of_records(tuple(fem.composed_from)) == 2

    def test_depth_2_compose_produces_depth_3(
        self, depth_2_h5: Path,
    ) -> None:
        fem = _assemble(("M3", depth_2_h5)).fem
        labels = sorted(r.label for r in fem.composed_from)
        assert labels == [
            "M3", "M3.assemblyM/partA", "M3/assemblyM",
        ]
        assert _compose_depth_of_records(tuple(fem.composed_from)) == 3

    def test_depth_3_compose_raises(self, depth_3_h5: Path) -> None:
        with pytest.raises(ComposeDepthExceededError) as ei:
            _assemble(("topLevel", depth_3_h5))
        # Message names the source depth (3) and the cap (3).
        msg = str(ei.value)
        assert "would exceed the maximum compose depth (3)" in msg
        assert "max_compose_depth" not in msg   # no knob (#1585, G3)
        assert "depth is 3" in msg

    def test_depth_exceeded_is_core_error(self, depth_3_h5: Path) -> None:
        """The facade exception inherits from the core canonical class
        so callers using ``except CoreComposeDepthExceededError`` catch
        it from outside the mesh package."""
        with pytest.raises(CoreComposeDepthExceededError):
            _assemble(("topLevel", depth_3_h5))

    def test_depth_exceeded_also_value_error(self, depth_3_h5: Path) -> None:
        """``ValueError`` continues to catch it — backward compat."""
        with pytest.raises(ValueError):
            _assemble(("topLevel", depth_3_h5))


class TestSeparatorAlternationEndToEnd:
    """Instancing nested sources produces the expected joined labels
    with depth-N separator alternation."""

    def test_depth_2_uses_slash(self, depth_1_h5: Path) -> None:
        fem = _assemble(("outer", depth_1_h5)).fem
        labels = sorted(r.label for r in fem.composed_from)
        # Top-level "outer" + grafted "outer/partA" (depth-2 boundary
        # uses "/").
        assert "outer" in labels
        assert "outer/partA" in labels
        assert "outer.partA" not in labels  # no "." at depth 2

    def test_depth_3_uses_dot_at_outer_boundary(
        self, depth_2_h5: Path,
    ) -> None:
        fem = _assemble(("top", depth_2_h5)).fem
        labels = sorted(r.label for r in fem.composed_from)
        # Top-level "top" + grafted "top/assemblyM" (depth-2 inner
        # boundary uses "/") + grafted "top.assemblyM/partA"
        # (depth-3 boundary uses "." outer, "/" preserved inner).
        assert "top" in labels
        assert "top/assemblyM" in labels
        assert "top.assemblyM/partA" in labels

    def test_module_label_on_nodes_is_joined(self, depth_1_h5: Path) -> None:
        """A depth-2 instance stamps the source's depth-1 rows with
        ``{outer}/{inner}`` on the merged ``module_label`` parallel
        dataset."""
        fem = _assemble(("outer", depth_1_h5)).fem
        # All nodes came from depth_1_h5, whose own rows had
        # module_label == "partA".  After the merge: "outer/partA".
        ml = fem.nodes._module_label
        assert ml is not None
        # The assembly's empty base contributed 0 rows; every row is
        # from the instance → all labels are "outer/partA".
        labels = set(str(x) for x in ml)
        assert labels == {"outer/partA"}

    def test_module_label_on_elements_is_joined(
        self, depth_1_h5: Path,
    ) -> None:
        fem = _assemble(("outer", depth_1_h5)).fem
        ml = fem.elements._module_label
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
        self, depth_2_h5: Path, tmp_path: Path,
    ) -> None:
        out = tmp_path / "round.h5"
        _assemble(("top", depth_2_h5), out=out)
        # Reload and compare labels.
        g2 = apeGmsh.from_h5(out)
        loaded_labels = sorted(r.label for r in g2._fem.composed_from)
        assert loaded_labels == [
            "top", "top.assemblyM/partA", "top/assemblyM",
        ]

    def test_round_trip_preserves_translate(
        self, depth_1_h5: Path, tmp_path: Path,
    ) -> None:
        out = tmp_path / "round.h5"
        _assemble(("outer", depth_1_h5), out=out, translate=(5.0, 0.0, 0.0))
        g2 = apeGmsh.from_h5(out)
        # The top-level ComposeRecord carries the translate.
        outer_rec = g2._fem.composed_from["outer"]
        assert outer_rec.translate == (5.0, 0.0, 0.0)

    def test_round_trip_module_label_for_node(
        self, depth_1_h5: Path, tmp_path: Path,
    ) -> None:
        out = tmp_path / "round.h5"
        _assemble(("outer", depth_1_h5), out=out)
        with h5_reader.open(str(out)) as model:
            ids = model.nodes()["ids"]
            assert len(ids) > 0
            for nid in ids:
                # Every node came from the instance at depth 2.
                assert model.composed_for_node(int(nid)) == "outer/partA"

    def test_iter_composed_from_yields_nested_labels(
        self, depth_2_h5: Path, tmp_path: Path,
    ) -> None:
        out = tmp_path / "round.h5"
        _assemble(("top", depth_2_h5), out=out)
        with h5_reader.open(str(out)) as model:
            labels = sorted(r.label for r in model.iter_composed_from())
        assert labels == [
            "top", "top.assemblyM/partA", "top/assemblyM",
        ]

    def test_double_round_trip_is_stable(
        self, depth_2_h5: Path, tmp_path: Path,
    ) -> None:
        """save → load → save → load yields the same compose chain."""
        out1 = tmp_path / "r1.h5"
        _assemble(("top", depth_2_h5), out=out1)
        g2 = apeGmsh.from_h5(out1)
        out2 = tmp_path / "r2.h5"
        g2.save(out2)
        g3 = apeGmsh.from_h5(out2)
        first_labels = sorted(r.label for r in g2._fem.composed_from)
        second_labels = sorted(r.label for r in g3._fem.composed_from)
        assert first_labels == second_labels
        assert first_labels == [
            "top", "top.assemblyM/partA", "top/assemblyM",
        ]

    def test_h5_uses_safe_group_names_for_slashed_labels(
        self, depth_2_h5: Path, tmp_path: Path,
    ) -> None:
        """The H5 writer sanitises ``/`` to ``_`` for group names but
        round-trips the original label via the ``label`` attribute."""
        out = tmp_path / "round.h5"
        _assemble(("top", depth_2_h5), out=out)
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
# Edge cases — multiple modules, sibling instances, mixed inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Behaviour at unusual composition shapes."""

    def test_two_sibling_depth_1_composes_stay_depth_1(
        self, leaf_h5: Path, leaf2_h5: Path,
    ) -> None:
        """Instancing two leaves under different labels yields two
        depth-1 entries (sibling modules; no nesting)."""
        fem = _assemble(("A", leaf_h5), ("B", leaf2_h5)).fem
        labels = sorted(r.label for r in fem.composed_from)
        assert labels == ["A", "B"]
        # Sibling instances do NOT count as nested.
        assert max(
            _label_depth(L) for L in labels
        ) == 1

    def test_compose_source_with_multiple_modules(
        self, leaf_h5: Path, leaf2_h5: Path, tmp_path: Path,
    ) -> None:
        """A source whose own ``composed_from`` has multiple entries —
        the bridge inherits all of them through the graft."""
        # Build an archive that instances two leaves.
        multi_h5 = tmp_path / "multi.h5"
        _assemble(("A", leaf_h5), ("B", leaf2_h5), out=multi_h5)
        # Instance the multi-module archive in a fresh assembly.
        fem = _assemble(("outer", multi_h5)).fem
        labels = sorted(r.label for r in fem.composed_from)
        # Top-level "outer" + grafted "outer/A" + grafted "outer/B".
        assert labels == ["outer", "outer/A", "outer/B"]

    def test_uncomposed_source_does_not_graft(self, leaf_h5: Path) -> None:
        """Instancing a depth-0 source produces a single top-level
        entry; no graft records appear."""
        fem = _assemble(("outer", leaf_h5)).fem
        labels = [r.label for r in fem.composed_from]
        assert labels == ["outer"]

    def test_compose_inspect_reports_nested_provenance(
        self, depth_2_h5: Path,
    ) -> None:
        """``compose_inspect`` surfaces the source's nested
        ``composed_from`` so callers can audit before instancing."""
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
