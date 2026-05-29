"""ADR 0043 slice 1.3 — MPCO read-join over a composed model.

Locks the correctness contract that compose (ADR 0038) broke:
``g.compose`` bakes per-module base-tag OFFSETS into element ``fem_eid``s,
while OpenSees ops element tags stay allocator-assigned (1-based, see
:class:`apeGmsh.opensees._internal.tag_allocator.TagAllocator`). So over a
composed model ``ops_tag != fem_eid``.

MPCO (STKO) records element results keyed by the global OpenSees ops tag
(the bucket ``ID`` dataset). The apeGmsh results API takes/returns
``fem_eid``s. The reader must therefore relabel through the persisted
``/opensees/element_meta/`` pairing (``FemToOpsTagMap``) — translating the
requested ``fem_eid`` filter to ops tags before the ``np.isin`` join, and
the returned ``element_index`` from ops tags back to ``fem_eid``s.

This test reproduces the composed condition WITHOUT the full compose
engine: a real ``model.h5`` whose two elements carry OFFSET fem_eids
(``1_000_001`` / ``1_000_002``). The bridge allocates ops tags ``1`` / ``2``
and pairs them with those fem_eids in ``element_meta``. The synthetic MPCO
bucket is keyed by the ops tags (what STKO writes); the query uses the
offset fem_eids (what the apeGmsh user holds).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import IntRule, flatten, lookup
from apeGmsh.results import Results

# Reuse the synthetic-MPCO builders from the element-mock suite.
from tests.test_results_mpco_element_mock import (
    _add_stress_bucket,
    _create_mpco_skeleton,
)


# Offset fem_eids — what a composed model carries.
FEM_EID_A = 1_000_001
FEM_EID_B = 1_000_002


def _build_offset_model_h5(tmp_path: Path) -> tuple[Path, dict[int, int]]:
    """Emit a model.h5 whose 2 elements have OFFSET fem_eids.

    Returns the path plus the ``{ops_tag: fem_eid}`` map read back from
    the emitted ``element_meta`` so the caller can key the MPCO bucket on
    the same ops tags the bridge allocated.
    """
    from apeGmsh._kernel.records._compose import ComposeRecord
    from apeGmsh.mesh._element_types import ElementGroup, make_type_info
    from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
    from apeGmsh.mesh.FEMData import (
        ElementComposite,
        FEMData,
        MeshInfo,
        NodeComposite,
    )
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.section.fiber import FiberPoint

    # Nodes are emitted with their fem id AS the ops node tag, so node
    # offsets never diverge; offset them too for faithfulness.
    node_ids = np.array(
        [1_000_001, 1_000_002, 1_000_003, 1_000_004], dtype=np.int64,
    )
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2,
    )
    line_group = ElementGroup(
        element_type=line_info,
        ids=np.array([FEM_EID_A, FEM_EID_B], dtype=np.int64),
        connectivity=np.array(
            [[1_000_001, 1_000_002], [1_000_003, 1_000_004]], dtype=np.int64,
        ),
    )
    pg = {(1, 100): {
        "name": "Cols",
        "node_ids": node_ids,
        "node_coords": node_coords,
        "element_ids": np.array([FEM_EID_A, FEM_EID_B], dtype=np.int64),
    }}
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups={1: line_group},
        physical=PhysicalGroupSet(pg), labels=LabelSet({}),
    )
    info = MeshInfo(n_nodes=4, n_elems=2, bandwidth=2, types=[line_info])
    # Stamp compose provenance so the read side recognises this as a
    # composed model (ADR 0043 slice 1.3 gates the ops↔fem relabel on
    # ``fem.composed_from``). The offset ids above already simulate the
    # post-compose tag-offset condition; this records the matching
    # provenance the real compose engine would write.
    composed_from = (
        ComposeRecord(
            label="A",
            source_path="moduleA.h5",
            source_fem_hash="deadbeef",
            source_neutral_schema_version="2.10.0",
            translate=(0.0, 0.0, 0.0),
        ),
    )
    fem = FEMData(
        nodes=nodes, elements=elements, info=info,
        composed_from=composed_from,
    )

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)

    out = tmp_path / "offset_model.h5"
    ops.h5(str(out))

    # Read back the (ops_tag -> fem_eid) pairing the bridge persisted.
    ops_to_fem: dict[int, int] = {}
    with h5py.File(out, "r") as f:
        meta = f["/opensees/element_meta"]
        for type_token in meta.keys():
            grp = meta[type_token]
            ids = np.asarray(grp["ids"][...], dtype=np.int64)
            fem_eids = np.asarray(grp["fem_eids"][...], dtype=np.int64)
            for ops_tag, fem_eid in zip(ids, fem_eids):
                ops_to_fem[int(ops_tag)] = int(fem_eid)
    return out, ops_to_fem


@pytest.fixture
def offset_composed_case(tmp_path: Path) -> tuple[Path, Path, dict[int, int]]:
    """(mpco_path, model_h5_path, ops_to_fem) for the composed-join test."""
    model_h5, ops_to_fem = _build_offset_model_h5(tmp_path)

    # Sanity: the bridge really did break tag==fem_eid (offset condition).
    assert set(ops_to_fem.values()) == {FEM_EID_A, FEM_EID_B}
    assert all(tag != fem for tag, fem in ops_to_fem.items()), (
        "fixture precondition: ops tags must differ from offset fem_eids"
    )

    ops_tags = sorted(ops_to_fem)  # the tags STKO would record

    layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
    mpco = tmp_path / "offset.mpco"
    f, stage_name = _create_mpco_skeleton(
        mpco,
        node_ids=np.array([1, 2, 3, 4, 5], dtype=np.int64),
        node_coords=np.eye(5, 3),
        n_steps=2,
        dt=0.5,
    )
    try:
        T, E = 2, len(ops_tags)
        # Encode a value carrying the element's POSITION so we can verify
        # the join lands on the right row: value(t,e) = e*100 + t*10.
        per_comp = {
            name: (
                np.arange(E, dtype=np.float64).reshape(1, E, 1) * 100.0
                + np.arange(T, dtype=np.float64).reshape(T, 1, 1) * 10.0
                + float(k)
            )
            for k, name in enumerate(layout.component_layout)
        }
        flat = flatten(per_comp, layout)
        _add_stress_bucket(
            f[stage_name],
            bracket_key=f"{layout.class_tag}-FourNodeTetrahedron[300:0:0]",
            class_tag=layout.class_tag,
            int_rule=IntRule.Tet_GL_1,
            element_ids=np.array(ops_tags, dtype=np.int64),  # ← ops tags
            flat_data=flat,
            n_gauss_points=1,
        )
    finally:
        f.close()
    return mpco, model_h5, ops_to_fem


class TestMpcoComposedJoin:
    def test_query_by_offset_fem_eid_returns_the_row(
        self, offset_composed_case: tuple[Path, Path, dict[int, int]],
    ) -> None:
        """Querying by a composed fem_eid must return that element's row.

        Pre-fix the reader filters the bucket's ops-tag ``ID`` by the
        fem_eid directly (``np.isin``), so this returns an empty slab.
        """
        mpco, model_h5, ops_to_fem = offset_composed_case
        # fem_eid -> ops_tag, to know which encoded row to expect.
        fem_to_ops = {fem: tag for tag, fem in ops_to_fem.items()}
        ops_tags_sorted = sorted(ops_to_fem)
        expected_pos = ops_tags_sorted.index(fem_to_ops[FEM_EID_A])

        with Results.from_mpco(mpco, model_h5=model_h5) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(
                component="stress_xx", ids=np.array([FEM_EID_A]),
            )
            # One element, one GP, two steps.
            assert slab.values.shape == (2, 1), (
                "join by composed fem_eid returned an empty slab — the MPCO "
                "reader compared ops-tag bucket IDs against a fem_eid filter"
            )
            # element_index is returned in fem_eid space (viewer contract).
            np.testing.assert_array_equal(slab.element_index, [FEM_EID_A])
            # Value encodes the element's bucket position: e*100 + t*10.
            np.testing.assert_array_equal(
                slab.values,
                [[expected_pos * 100.0], [expected_pos * 100.0 + 10.0]],
            )

    def test_unfiltered_element_index_is_fem_eid_space(
        self, offset_composed_case: tuple[Path, Path, dict[int, int]],
    ) -> None:
        """An unfiltered read must relabel the ops-keyed bucket to fem_eids."""
        mpco, model_h5, _ = offset_composed_case
        with Results.from_mpco(mpco, model_h5=model_h5) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            assert slab.values.shape[1] == 2
            assert set(int(e) for e in slab.element_index) == {
                FEM_EID_A, FEM_EID_B,
            }, (
                "element_index leaked ops tags instead of fem_eids — the "
                "viewer scatters slab.element_index into a fem_eid table"
            )


class TestElementTagTranslator:
    """Unit coverage for the all-or-nothing relabel contract."""

    def _make(self) -> "ElementTagTranslator":
        from apeGmsh.results.readers._tag_translation import (
            ElementTagTranslator,
        )
        # fem 1_000_001 ↔ ops 2, fem 1_000_002 ↔ ops 3.
        return ElementTagTranslator(
            fem_to_ops={FEM_EID_A: 2, FEM_EID_B: 3},
            ops_to_fem={2: FEM_EID_A, 3: FEM_EID_B},
        )

    def test_full_match_relabels(self) -> None:
        t = self._make()
        np.testing.assert_array_equal(
            t.to_ops(np.array([FEM_EID_A, FEM_EID_B])), [2, 3],
        )
        np.testing.assert_array_equal(
            t.to_fem(np.array([2, 3])), [FEM_EID_A, FEM_EID_B],
        )

    def test_partial_match_passes_through_whole_array(self) -> None:
        """A single unknown id ⇒ the model does not describe this result ⇒
        leave the array untouched (no partial / colliding relabel)."""
        t = self._make()
        # ops tag 2 is known, but 99 is not — relabel neither.
        np.testing.assert_array_equal(t.to_fem(np.array([2, 99])), [2, 99])
        np.testing.assert_array_equal(
            t.to_ops(np.array([FEM_EID_A, 7])), [FEM_EID_A, 7],
        )

    def test_none_and_empty(self) -> None:
        t = self._make()
        assert t.to_ops(None) is None
        assert t.to_fem(None) is None
        np.testing.assert_array_equal(
            t.to_fem(np.array([], dtype=np.int64)), [],
        )

    def test_empty_map_is_identity(self) -> None:
        from apeGmsh.results.readers._tag_translation import (
            ElementTagTranslator,
        )
        t = ElementTagTranslator(fem_to_ops={}, ops_to_fem={})
        assert t.is_empty
        np.testing.assert_array_equal(t.to_ops(np.array([5, 6])), [5, 6])
        np.testing.assert_array_equal(t.to_fem(np.array([5, 6])), [5, 6])
