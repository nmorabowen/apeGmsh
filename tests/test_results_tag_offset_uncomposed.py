"""ADR 0043 slice 1.3 follow-up — ops↔fem_eid relabel for UNCOMPOSED models.

Locks the correctness contract the compose-provenance gate broke: the
fem_eid↔ops-tag divergence is NOT a compose-only condition. gmsh numbers
lower-dimensional elements first, so almost any 3-D solid whose surfaces
are meshed (e.g. carries surface physical groups) has its volume
``fem_eid``s offset from the allocator-assigned 1-based ops tags — with
``fem.composed_from`` empty.

Pre-fix, ``Results.from_mpco`` / ``from_ladruno`` attached the
:class:`ElementTagTranslator` only when the bound model carried compose
provenance, so for this (ordinary) case:

* a ``fem_eid`` filter was compared directly against the bucket's
  ops-tag ``ID``s — elements whose ``fem_eid`` exceeded the max ops tag
  were **silently dropped**, the rest joined to the **wrong rows**;
* the unfiltered ``element_index`` leaked ops tags, scattering every
  element value onto a different element downstream.

This file reproduces the offset condition exactly like
``test_results_mpco_composed_join.py`` — same builders, same synthetic
MPCO bucket — but with **no** ``ComposeRecord`` stamped, so the model is
uncomposed and only the meaning of the pairing keeps the join right.

It also locks the new loud-failure contract for the *unpaired* case:
element-level reads over an externally-bound fem with no tag pairing at
all (``from_ladruno`` without ``model_h5=``) warn with
:class:`WarnElementTagPairingMissing`; nodal reads stay silent.
"""
from __future__ import annotations

import warnings
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

# Offset fem_eids — what a gmsh mesh whose boundary elements consumed
# the low ids carries. Uncomposed: no ComposeRecord anywhere.
FEM_EID_A = 501
FEM_EID_B = 502


def _build_offset_uncomposed_model_h5(
    tmp_path: Path,
) -> tuple[Path, dict[int, int]]:
    """Emit a model.h5 whose 2 elements have OFFSET fem_eids, UNCOMPOSED.

    Mirrors ``test_results_mpco_composed_join._build_offset_model_h5``
    minus the ``ComposeRecord`` — the divergence here comes purely from
    sparse renumbering (the gmsh boundary-elements-first condition), so
    ``fem.composed_from`` stays empty.

    Returns the path plus the ``{ops_tag: fem_eid}`` map read back from
    the emitted ``element_meta``.
    """
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

    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
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
        connectivity=np.array([[1, 2], [3, 4]], dtype=np.int64),
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
    # NO ComposeRecord — this is the uncomposed, sparsely-renumbered
    # model the old compose-provenance gate wrongly treated as identity.
    fem = FEMData(nodes=nodes, elements=elements, info=info)
    assert len(fem.composed_from) == 0, "fixture precondition: uncomposed"

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

    out = tmp_path / "offset_uncomposed_model.h5"
    ops.h5(str(out))

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
def offset_uncomposed_case(
    tmp_path: Path,
) -> tuple[Path, Path, dict[int, int]]:
    """(mpco_path, model_h5_path, ops_to_fem) for the uncomposed join."""
    model_h5, ops_to_fem = _build_offset_uncomposed_model_h5(tmp_path)

    # Sanity: the bridge really did break tag==fem_eid without compose.
    assert set(ops_to_fem.values()) == {FEM_EID_A, FEM_EID_B}
    assert all(tag != fem for tag, fem in ops_to_fem.items()), (
        "fixture precondition: ops tags must differ from offset fem_eids"
    )

    ops_tags = sorted(ops_to_fem)  # the tags the recorder writes

    layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
    mpco = tmp_path / "offset_uncomposed.mpco"
    f, stage_name = _create_mpco_skeleton(
        mpco,
        node_ids=np.array([1, 2, 3, 4, 5], dtype=np.int64),
        node_coords=np.eye(5, 3),
        n_steps=2,
        dt=0.5,
    )
    try:
        T, E = 2, len(ops_tags)
        # value(t, e) = e*100 + t*10 encodes the bucket row so the test
        # can verify the join lands on the right element.
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


class TestMpcoUncomposedOffsetJoin:
    """The bug this file exists for: uncomposed model, offset fem_eids."""

    def test_query_by_offset_fem_eid_returns_the_row(
        self, offset_uncomposed_case: tuple[Path, Path, dict[int, int]],
    ) -> None:
        """Querying by an offset fem_eid must return that element's row.

        Pre-fix, the compose-provenance gate skipped the translator for
        this uncomposed model, so the fem_eid filter was compared
        directly against the ops-tag bucket IDs — an empty (or wrong)
        slab: the SILENT-DROP symptom.
        """
        mpco, model_h5, ops_to_fem = offset_uncomposed_case
        fem_to_ops = {fem: tag for tag, fem in ops_to_fem.items()}
        ops_tags_sorted = sorted(ops_to_fem)
        expected_pos = ops_tags_sorted.index(fem_to_ops[FEM_EID_A])

        with Results.from_mpco(mpco, model_h5=model_h5) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(
                component="stress_xx", ids=np.array([FEM_EID_A]),
            )
            assert slab.values.shape == (2, 1), (
                "join by offset fem_eid returned an empty slab — the "
                "reader compared ops-tag bucket IDs against a fem_eid "
                "filter (the compose-provenance gate skipped the "
                "translator for an uncomposed model)"
            )
            np.testing.assert_array_equal(slab.element_index, [FEM_EID_A])
            np.testing.assert_array_equal(
                slab.values,
                [[expected_pos * 100.0], [expected_pos * 100.0 + 10.0]],
            )

    def test_unfiltered_element_index_is_fem_eid_space(
        self, offset_uncomposed_case: tuple[Path, Path, dict[int, int]],
    ) -> None:
        """An unfiltered read must relabel the ops-keyed bucket to
        fem_eids — otherwise every value scatters onto the WRONG element
        downstream (the SCRAMBLE symptom)."""
        mpco, model_h5, _ = offset_uncomposed_case
        with Results.from_mpco(mpco, model_h5=model_h5) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            assert slab.values.shape[1] == 2
            assert set(int(e) for e in slab.element_index) == {
                FEM_EID_A, FEM_EID_B,
            }, (
                "element_index leaked ops tags instead of fem_eids for an "
                "uncomposed offset model — spatially scrambled results"
            )


# ---------------------------------------------------------------------
# Unpaired element reads warn (from_ladruno without model_h5=)
# ---------------------------------------------------------------------

_LADRUNO_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ladruno"
_TRUSS = _LADRUNO_FIXTURES / "truss2d.ladruno"


def _external_stub_fem():
    """A FEMData that is NOT the reader's own embedded snapshot."""
    from tests.opensees.h5._opensees_model_fixtures import (
        build_simple_frame_fem,
    )
    return build_simple_frame_fem()


class TestReadTranslationConsistency:
    """One read translates BOTH directions or NEITHER (the stub trap).

    With pairing ``{fem 1 ↔ ops 2}`` (the session stub model's real
    shape — the allocator gave its one element ops tag 2), a filter
    ``[2]`` is no known fem_eid and passes through, selecting ops
    element 2. Deciding the reverse map independently would then see
    the returned index ``[2]`` fully covered and mis-relabel it to
    ``[1]``. ``read_translation`` pins the per-read pairing.
    """

    def _make(self):
        from apeGmsh.results.readers._tag_translation import (
            ElementTagTranslator,
        )
        return ElementTagTranslator(fem_to_ops={1: 2}, ops_to_fem={2: 1})

    def test_covered_filter_translates_both_ways(self) -> None:
        t = self._make()
        bucket_ids, back = t.read_translation(np.array([1]))
        np.testing.assert_array_equal(bucket_ids, [2])
        np.testing.assert_array_equal(back(np.array([2, 2])), [1, 1])

    def test_uncovered_filter_translates_neither_way(self) -> None:
        t = self._make()
        bucket_ids, back = t.read_translation(np.array([2]))
        np.testing.assert_array_equal(bucket_ids, [2])
        # The index selected by the untranslated filter must come back
        # untouched even though the reverse map covers it.
        np.testing.assert_array_equal(back(np.array([2, 2])), [2, 2])

    def test_unfiltered_read_is_all_or_nothing_on_the_index(self) -> None:
        t = self._make()
        bucket_ids, back = t.read_translation(None)
        assert bucket_ids is None
        # Fully covered → relabel; any unknown id → passthrough.
        np.testing.assert_array_equal(back(np.array([2])), [1])
        np.testing.assert_array_equal(back(np.array([1, 2])), [1, 2])


def _warning_cls():
    # Imported lazily so the *join* tests above still collect (and fail
    # meaningfully) on a pre-fix checkout that lacks the warning class.
    from apeGmsh.results.readers._tag_translation import (
        WarnElementTagPairingMissing,
    )
    return WarnElementTagPairingMissing


class TestUnpairedElementReadWarns:
    def test_element_read_with_external_fem_warns(self) -> None:
        r = Results.from_ladruno(_TRUSS).bind(_external_stub_fem())
        with pytest.warns(_warning_cls()):
            r.elements.line_stations.get(
                component="axial_force", ids=np.array([1]),
            )

    def test_self_sufficient_read_does_not_warn(self) -> None:
        """No external fem → the filter speaks the file's own id space."""
        r = Results.from_ladruno(_TRUSS)
        with warnings.catch_warnings():
            warnings.simplefilter("error", _warning_cls())
            r.elements.line_stations.get(component="axial_force")

    def test_nodal_read_never_warns(self) -> None:
        """Nodal ids match the file exactly — reads stay silent even
        with an external fem bound."""
        r = Results.from_ladruno(_TRUSS).bind(_external_stub_fem())
        with warnings.catch_warnings():
            warnings.simplefilter("error", _warning_cls())
            r.nodes.get(component="displacement_x")
