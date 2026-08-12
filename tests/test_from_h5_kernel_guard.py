"""from_h5 live-kernel guard — ChainPhaseError on introspection composites.

A session built via ``apeGmsh.from_h5`` has no gmsh kernel, but the
live-kernel composites (``g.inspect``, the ``g.physical`` and
``g.labels`` queries, ``g.view``, ``g.plot``) were still reachable and
died with raw gmsh errors ("Gmsh has not been initialized") — or, when
another session happened to hold the kernel, silently answered from an
unrelated model.  These tests lock the new contract: every such entry
point raises :class:`ChainPhaseError` naming the H5-safe alternatives
(``fem.inspect`` / ``results.inspect`` / ``fem.physical`` —
:class:`PhysicalGroupSet`; ``fem.nodes.labels`` — :class:`LabelSet`).

``__repr__`` is the deliberate exception: it *degrades* to a
descriptive string instead of raising, because Python calls it from
debuggers, logging and pytest's own failure output, where an
exception would mask whatever is actually being inspected.

Runs entirely off the FEMData broker — no live gmsh, no openseespy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh._kernel.payloads import ElementGroup
from apeGmsh._kernel.record_sets import ComposeSet
from apeGmsh.core._compose_errors import (
    ChainPhaseError,
    is_kernelless_session,
    raise_if_no_live_kernel,
)
from apeGmsh.mesh._element_types import make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


def _quad_fem() -> FEMData:
    """A single quad with one node PG — enough to round-trip H5."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    quad_info = make_type_info(
        code=3, gmsh_name="Quadrangle 4", dim=2, order=1, npe=4, count=1,
    )
    quad_group = ElementGroup(
        element_type=quad_info,
        ids=np.array([500], dtype=np.int64),
        connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64),
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet({
            (0, 1): {"name": "face",
                     "node_ids": node_ids.copy(),
                     "node_coords": coords.copy()},
        }),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={3: quad_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=4, n_elems=1, bandwidth=1, types=[quad_info])
    return FEMData(nodes=nodes, elements=elements, info=info,
                   composed_from=ComposeSet(()))


@pytest.fixture()
def g(tmp_path: Path) -> apeGmsh:
    path = tmp_path / "m.h5"
    _quad_fem().to_h5(str(path))
    return apeGmsh.from_h5(path)


# ---------------------------------------------------------------------
# g.inspect — introspects the live gmsh model
# ---------------------------------------------------------------------


class TestInspectRaises:
    def test_print_summary_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="fem.inspect"):
            g.inspect.print_summary()

    def test_get_geometry_info_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="live gmsh kernel"):
            g.inspect.get_geometry_info()

    def test_get_mesh_info_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="live gmsh kernel"):
            g.inspect.get_mesh_info()

    def test_message_names_all_alternatives(self, g: apeGmsh) -> None:
        """The message must point at every H5-safe surface."""
        with pytest.raises(ChainPhaseError) as exc:
            g.inspect.print_summary()
        msg = str(exc.value)
        assert "g.inspect.print_summary()" in msg
        assert "fem.inspect" in msg
        assert "results.inspect" in msg
        assert "PhysicalGroupSet" in msg


# ---------------------------------------------------------------------
# g.physical queries — read gmsh PGs directly
# ---------------------------------------------------------------------


class TestPhysicalQueriesRaise:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda g: g.physical.get_all(), id="get_all"),
            pytest.param(lambda g: g.physical.get_entities(2, 1),
                         id="get_entities"),
            pytest.param(lambda g: g.physical.entities("face"),
                         id="entities"),
            pytest.param(lambda g: g.physical.get_groups_for_entity(2, 1),
                         id="get_groups_for_entity"),
            pytest.param(lambda g: g.physical.get_name(2, 1), id="get_name"),
            pytest.param(lambda g: g.physical.get_tag(2, "face"),
                         id="get_tag"),
            pytest.param(lambda g: g.physical.summary(), id="summary"),
            pytest.param(lambda g: g.physical.get_nodes(2, 1),
                         id="get_nodes"),
        ],
    )
    def test_query_raises(self, g: apeGmsh, call) -> None:
        with pytest.raises(ChainPhaseError, match="PhysicalGroupSet"):
            call(g)

    def test_h5_safe_alternative_works(self, g: apeGmsh) -> None:
        """The alternative the message names actually answers."""
        fem = g.mesh.queries.get_fem_data()
        assert len(fem.nodes.physical) == 1


# ---------------------------------------------------------------------
# g.labels — Tier 1 naming, backed by prefixed gmsh PGs
# ---------------------------------------------------------------------


class TestLabelQueriesRaise:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda g: g.labels.entities("face"), id="entities"),
            pytest.param(lambda g: g.labels.get_all(), id="get_all"),
            pytest.param(lambda g: g.labels.summary(), id="summary"),
            pytest.param(lambda g: g.labels.has("face"), id="has"),
            pytest.param(lambda g: g.labels.reverse_map(), id="reverse_map"),
            pytest.param(lambda g: g.labels.labels_for_entity(2, 1),
                         id="labels_for_entity"),
        ],
    )
    def test_query_raises(self, g: apeGmsh, call) -> None:
        with pytest.raises(ChainPhaseError, match="live gmsh kernel"):
            call(g)

    def test_message_names_labelset_not_physicalgroupset(
        self, g: apeGmsh,
    ) -> None:
        """Labels snapshot onto the broker as a LabelSet — the message
        must point there, not at the default PhysicalGroupSet text."""
        with pytest.raises(ChainPhaseError) as exc:
            g.labels.get_all()
        msg = str(exc.value)
        assert "fem.nodes.labels" in msg
        assert "LabelSet" in msg
        assert "PhysicalGroupSet" not in msg

    def test_has_names_itself_not_the_delegate(self, g: apeGmsh) -> None:
        """``has()`` delegates to ``entities()``; the guard fires at the
        outer verb so the message names the call the user made."""
        with pytest.raises(ChainPhaseError, match=r"g\.labels\.has\('face'\)"):
            g.labels.has("face")


# ---------------------------------------------------------------------
# __repr__ degrades instead of raising
# ---------------------------------------------------------------------


class TestReprDegrades:
    """``__repr__`` must never raise — Python calls it from debuggers,
    logging, exception formatting and pytest assertion output, where an
    exception masks the object actually under inspection."""

    def test_physical_repr_reports_no_kernel(self, g: apeGmsh) -> None:
        text = repr(g.physical)
        assert "no live gmsh kernel" in text
        assert "from_h5" in text

    def test_labels_repr_reports_no_kernel(self, g: apeGmsh) -> None:
        text = repr(g.labels)
        assert "no live gmsh kernel" in text

    def test_labels_repr_not_mislabelled_session_closed(
        self, g: apeGmsh,
    ) -> None:
        """The pre-existing except-branch says "session closed", which
        is wrong here: a from_h5 session was never open."""
        assert "session closed" not in repr(g.labels)

    def test_repr_survives_in_f_string(self, g: apeGmsh) -> None:
        """The failure mode a raising __repr__ would cause: formatting
        a message that merely mentions the composite."""
        assert "no live gmsh kernel" in f"{g.physical!r} {g.labels!r}"


# ---------------------------------------------------------------------
# g.view — writes gmsh post-processing views
# ---------------------------------------------------------------------


class TestViewRaises:
    def test_add_node_scalar_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="g.view.add_node_scalar"):
            g.view.add_node_scalar("u", [1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0])

    def test_add_element_scalar_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="live gmsh kernel"):
            g.view.add_element_scalar("s", [500], [1.0])

    def test_add_node_vector_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="live gmsh kernel"):
            g.view.add_node_vector("d", [1, 2, 3, 4], np.zeros((4, 3)))

    def test_add_element_vector_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="live gmsh kernel"):
            g.view.add_element_vector("v", [500], np.zeros((1, 3)))


# ---------------------------------------------------------------------
# g.plot — reads gmsh geometry / mesh for matplotlib rendering
# ---------------------------------------------------------------------


class TestPlotRaises:
    @pytest.fixture(autouse=True)
    def _needs_mpl(self) -> None:
        pytest.importorskip("matplotlib")

    def test_geometry_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="g.plot.geometry"):
            g.plot.geometry()

    def test_mesh_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="g.plot.mesh"):
            g.plot.mesh()

    def test_physical_groups_raises(self, g: apeGmsh) -> None:
        with pytest.raises(ChainPhaseError, match="live gmsh kernel"):
            g.plot.physical_groups()


# ---------------------------------------------------------------------
# Live sessions and stub parents pass the guard untouched
# ---------------------------------------------------------------------


class TestGuardIsScoped:
    def test_stub_parent_passes(self) -> None:
        """Fixture stubs without _fem_from_h5 are not gated."""
        raise_if_no_live_kernel(type("_Stub", (), {})(), "x")

    def test_live_flag_false_passes(self) -> None:
        """A live session (_fem_from_h5 False) is not gated."""
        stub = type("_Stub", (), {"_fem_from_h5": False})()
        raise_if_no_live_kernel(stub, "x")

    def test_predicate_matches_guard(self, g: apeGmsh) -> None:
        """is_kernelless_session backs both the raise and the repr
        paths — they must agree on what counts as kernelless."""
        assert is_kernelless_session(g) is True
        assert is_kernelless_session(type("_Stub", (), {})()) is False
