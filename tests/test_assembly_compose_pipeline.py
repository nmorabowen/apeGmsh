"""ADR 0043 slice 1.4 — raw compose+couple pipeline (pre-Assembly).

De-risks the linchpin the 1.4 red/blue pass flagged: NO existing test
exercises an interface constraint over a *namespaced composed PG* through
the session (``from_h5 -> g.compose -> g.constraints.<kind>``). All prior
chain-phase constraint tests use hand-built FEMData with un-namespaced
labels. This verifies the underlying pipeline the future ``Assembly``
wrapper will sit on:

* the host part keeps its PGs UN-prefixed (``"base"``),
* a composed part's PGs are prefixed with its compose label (``"A.top"``),
* ``g.constraints.equal_dof("base", "A.top", ...)`` resolves both through
  ``FEMDataSource.nodes_for`` and actually appends a constraint record to
  ``_fem`` (i.e. the chain-phase route did NOT silently drop the couple).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._core import apeGmsh
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)


def _line_module(
    *, node_ids, coords, node_pgs, elem_ids, conn,
) -> FEMData:
    """Minimal single-Line2-type FEMData with node-side PGs."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    elem_ids = np.asarray(elem_ids, dtype=np.int64)
    conn = np.asarray(conn, dtype=np.int64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
        count=elem_ids.size,
    )
    line_group = ElementGroup(
        element_type=line_info, ids=elem_ids, connectivity=conn,
    )
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet(node_pgs), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet({}), labels=LabelSet({}),
    )
    info = MeshInfo(
        n_nodes=node_ids.size, n_elems=elem_ids.size, bandwidth=1,
        types=[line_info],
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


@pytest.fixture
def host_and_module(tmp_path: Path) -> tuple[Path, Path]:
    """Host with PG 'base' at x=2; module A with PG 'top' at local x=0.

    Composing A with translate=(2,0,0) lands A's 'top' node on the host's
    'base' node (co-located) so equal_dof pairs them.
    """
    host = _line_module(
        node_ids=[1, 2, 3],
        coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        node_pgs={(0, 100): {
            "name": "base",
            "node_ids": np.array([3], dtype=np.int64),
            "node_coords": np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
        }},
        elem_ids=[10, 11], conn=[[1, 2], [2, 3]],
    )
    mod_a = _line_module(
        node_ids=[1, 2, 3],
        coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        node_pgs={(0, 100): {
            "name": "top",
            "node_ids": np.array([1], dtype=np.int64),
            "node_coords": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        }},
        elem_ids=[10, 11], conn=[[1, 2], [2, 3]],
    )
    host_p = tmp_path / "host.h5"
    a_p = tmp_path / "module_a.h5"
    host.to_h5(str(host_p))
    mod_a.to_h5(str(a_p))
    return host_p, a_p


def _count_constraints(fem) -> int:
    return len(tuple(fem.nodes.constraints)) + len(
        tuple(fem.elements.constraints)
    )


class TestComposeCouplePipeline:
    def test_nodes_for_resolves_namespaced_composed_pg(
        self, host_and_module: tuple[Path, Path],
    ) -> None:
        """FEMDataSource.nodes_for must resolve both the host's bare PG
        and the composed part's namespaced PG."""
        from apeGmsh._kernel.resolvers._source import FEMDataSource

        host_p, a_p = host_and_module
        g = apeGmsh.from_h5(str(host_p))
        g.compose(str(a_p), label="A", translate=(2.0, 0.0, 0.0))

        src = FEMDataSource(g._fem)
        base_nodes = set(int(x) for x in src.nodes_for("base"))
        top_nodes = set(int(x) for x in src.nodes_for("A.top"))

        assert base_nodes == {3}, "host PG 'base' should stay un-namespaced"
        assert len(top_nodes) == 1, "composed PG 'A.top' must resolve"
        assert top_nodes != base_nodes

    def test_equal_dof_over_namespaced_pg_routes_onto_fem(
        self, host_and_module: tuple[Path, Path],
    ) -> None:
        """The couple must actually append a constraint record (not be
        silently dropped by the chain-phase router's KeyError swallow)."""
        host_p, a_p = host_and_module
        g = apeGmsh.from_h5(str(host_p))
        g.compose(str(a_p), label="A", translate=(2.0, 0.0, 0.0))

        before = _count_constraints(g._fem)
        g.constraints.equal_dof("base", "A.top", dofs=[1, 2, 3])
        after = _count_constraints(g._fem)

        assert after > before, (
            "equal_dof over a namespaced composed PG was silently dropped "
            "— the chain-phase route did not apply to _fem"
        )


class TestAssembly:
    """The Assembly wrapper over the (de-risked) compose+couple pipeline."""

    def test_materialize_applies_couple_like_raw_pipeline(
        self, host_and_module: tuple[Path, Path],
    ) -> None:
        from apeGmsh import Assembly

        host_p, a_p = host_and_module
        asm = Assembly("frame")
        asm.add("base_part", str(host_p))                      # host
        asm.add("top_part", str(a_p), translate=(2.0, 0.0, 0.0))
        asm.couple(
            "base_part", "top_part", kind="equal_dof",
            ports=("base", "top"), dofs=[1, 2, 3],
        )
        g = asm.materialize()

        # The couple landed a constraint, and the composed broker carries
        # both modules (host node 3 + A's offset nodes).
        assert _count_constraints(g._fem) >= 1
        assert g._fem.nodes.ids.size == 6  # 3 host + 3 composed

    def test_chainable_declaration(
        self, host_and_module: tuple[Path, Path],
    ) -> None:
        from apeGmsh import Assembly

        host_p, a_p = host_and_module
        g = (
            Assembly("frame")
            .add("base_part", str(host_p))
            .add("top_part", str(a_p), translate=(2.0, 0.0, 0.0))
            .couple(
                "base_part", "top_part", kind="equal_dof",
                ports=("base", "top"), dofs=[1, 2, 3],
            )
            .materialize()
        )
        assert _count_constraints(g._fem) >= 1

    def test_fail_loud_on_unresolvable_port(
        self, host_and_module: tuple[Path, Path],
    ) -> None:
        from apeGmsh import Assembly, AssemblyError

        host_p, a_p = host_and_module
        asm = Assembly("frame")
        asm.add("base_part", str(host_p))
        asm.add("top_part", str(a_p), translate=(2.0, 0.0, 0.0))
        # "nope" is not a PG on top_part → router swallows KeyError →
        # zero records → materialize must fail loud, not emit an untied model.
        asm.couple(
            "base_part", "top_part", kind="equal_dof",
            ports=("base", "nope"), dofs=[1, 2, 3],
        )
        with pytest.raises(
            AssemblyError, match="tied nothing|not a physical group",
        ):
            asm.materialize()

    def test_couple_unknown_part_raises(
        self, host_and_module: tuple[Path, Path],
    ) -> None:
        from apeGmsh import Assembly, AssemblyError

        host_p, a_p = host_and_module
        asm = Assembly("frame")
        asm.add("base_part", str(host_p))
        asm.add("top_part", str(a_p), translate=(2.0, 0.0, 0.0))
        asm.couple(
            "base_part", "ghost", kind="equal_dof",
            ports=("base", "top"), dofs=[1, 2, 3],
        )
        with pytest.raises(AssemblyError, match="unknown part"):
            asm.materialize()

    def test_validation(self, host_and_module: tuple[Path, Path]) -> None:
        from apeGmsh import Assembly, AssemblyError

        host_p, _ = host_and_module
        with pytest.raises(AssemblyError):
            Assembly("")  # empty name
        with pytest.raises(AssemblyError):
            Assembly("x").materialize()  # no parts
        asm = Assembly("x")
        asm.add("p", str(host_p))
        with pytest.raises(AssemblyError, match="duplicate"):
            asm.add("p", str(host_p))
        with pytest.raises(AssemblyError, match="unsupported kind"):
            asm.couple("p", "p", kind="welded", ports=("a", "b"))
        with pytest.raises(AssemblyError):
            asm.couple("p", "p", kind="equal_dof", ports=("only_one",))
