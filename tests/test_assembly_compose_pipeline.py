"""ADR 0043 slice 1.4 — the ``Assembly`` surface contracts, on the ADR
0117 (v2) form.

The raw compose+couple pipeline this file once de-risked (``from_h5 ->
g.compose -> g.constraints.<kind>``) is gone with v1; its oracles live in
``tests/assembly/test_couplings.py`` (the namespaced port resolves and the
coupling lands records) and ``tests/assembly/test_two_instances_one_tie.py``
(bad ports). What stays here:

* ``Assembly`` is a sub-path export, never top-level;
* declaration-time validation refuses before recording anything.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
def host_h5(tmp_path: Path) -> Path:
    """A saved part with node PG 'base' at x=2."""
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
    host_p = tmp_path / "host.h5"
    host.to_h5(str(host_p))
    return host_p


class TestAssembly:
    """The Assembly wrapper over the (de-risked) compose+couple pipeline."""

    def test_assembly_is_subpath_only_not_top_level(self) -> None:
        """Assembly lives at ``apeGmsh.assembly``, NOT top-level — the v1.0
        'session IS the assembly' guard (test_library_contracts) stays
        satisfied. Locks the slice-1.4 red/blue decision."""
        import apeGmsh
        from apeGmsh.assembly import Assembly as _SubPath

        assert _SubPath is not None
        assert not hasattr(apeGmsh, "Assembly"), (
            "Assembly must not be a top-level export (v1.0 contract); import "
            "it from apeGmsh.assembly."
        )

    def test_validation(self, host_h5: Path) -> None:
        from apeGmsh.assembly import Assembly, AssemblyError

        with pytest.raises(AssemblyError):
            Assembly("")  # empty name
        with pytest.raises(AssemblyError, match="no instances"):
            Assembly("x").bridge(ndm=3, ndf=3)  # no parts
        asm = Assembly("x")
        asm.instance("p", str(host_h5))
        with pytest.raises(AssemblyError, match="already declared"):
            asm.instance("p", str(host_h5))  # duplicate
        asm.node("ref", (2.0, 0.0, 0.0))
        with pytest.raises(AssemblyError, match="kind='welded'"):
            asm.couple("p.base", kind="welded", reference="ref")
        with pytest.raises(AssemblyError):
            # a bare (undotted) port names no instance
            asm.equal_dof("only_one", "p.base", dofs=[1, 2, 3])
        # Every refusal recorded nothing.
        assert [i.label for i in asm.instances] == ["p"]
        assert asm.ties == ()
