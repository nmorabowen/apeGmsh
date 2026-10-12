"""ADR 0043 slice 1.3 follow-up — live DomainCapture over a composed model.

The element-level capturers query the live ops domain with the resolved
record's ``element_ids`` (mesh ``fem_eid``s). Over a composed model
(ADR 0038) the ops element tag is allocator-assigned and differs from
``fem_eid``, so ``ops.eleResponse(fem_eid, …)`` would hit a nonexistent
element. The fix translates the record's ``element_ids`` fem→ops for the
queries (built from the composed model's ``element_meta``) and relabels
the captured ``element_index`` ops→fem on the way out.

This test pairs a composed ``model.h5`` (element fem_eids offset to
1_000_00x; bridge-allocated ops tags differ) with a fake ops domain that
ONLY knows the ops tags. Without the translation the capturer would query
fem_eids the fake domain has never heard of and raise; with it, capture
succeeds and the written ``element_index`` is in fem space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import IntRule, flatten, lookup
from apeGmsh.results.capture._domain import (
    DomainCapture,
    _capture_pairing,
)
from apeGmsh.results.capture.spec import (
    ResolvedDomainCaptureRecord,
    ResolvedDomainCaptureSpec,
)

# Reuse the composed offset-model builder from the MPCO read-join test.
from tests.test_results_mpco_composed_join import (
    FEM_EID_A,
    FEM_EID_B,
    _build_offset_model_h5,
)
from tests.test_results_domain_capture_gauss import _FakeOpsElements, _MockFem


class _RecordingOps(_FakeOpsElements):
    """Fake ops that records every element id it was queried with."""

    def __init__(self) -> None:
        super().__init__()
        self.queried_eids: list[int] = []

    def eleType(self, eid: int) -> str:
        self.queried_eids.append(int(eid))
        return super().eleType(eid)

    def eleResponse(self, eid: int, *args):
        self.queried_eids.append(int(eid))
        return super().eleResponse(eid, *args)


def _resolved_gauss_spec(layout, snapshot_id, *, element_ids):
    return ResolvedDomainCaptureSpec(
        fem_snapshot_id=snapshot_id,
        records=(
            ResolvedDomainCaptureRecord(
                category="gauss", name="solid_stress",
                components=tuple(layout.component_layout),
                dt=None, n_steps=None,
                element_ids=np.array(element_ids, dtype=np.int64),
            ),
        ),
        ndm=3, ndf=6,
    )


class TestComposedDomainCapture:
    def test_capture_queries_ops_tags_and_writes_fem_index(
        self, tmp_path: Path,
    ) -> None:
        model_h5, ops_to_fem = _build_offset_model_h5(tmp_path)
        # Precondition: composed ⇒ ops tags differ from offset fem_eids.
        assert all(tag != fem for tag, fem in ops_to_fem.items())
        ops_tags = sorted(ops_to_fem)  # the live-domain tags

        # The translator is built from the composed model's element_meta.
        pairing = _capture_pairing(model_h5)
        assert pairing, (
            "composed model should yield a tag translator (compose gate)"
        )

        layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")

        # Fake ops domain that ONLY knows the ops tags (NOT the fem_eids).
        ops = _RecordingOps()
        for k, tag in enumerate(ops_tags):
            ops.ele_class[tag] = "FourNodeTetrahedron"
            # value carries the tag's position so we can verify the join.
            ops.ele_response[(tag, "stresses")] = [
                float(k * 10 + c) for c in range(len(layout.component_layout))
            ]

        fem = _MockFem([1, 2, 3, 4])
        spec = _resolved_gauss_spec(
            layout, fem.snapshot_id,
            element_ids=[FEM_EID_A, FEM_EID_B],   # composed fem_eids
        )

        out = tmp_path / "composed_capture.h5"
        # tag_map injected (the bridge / from_h5 paths build it the same way).
        with DomainCapture(spec, out, fem, ops=ops, element_tags=pairing) as cap:
            cap.begin_stage("static", kind="static")
            cap.step(t=1.0)
            cap.end_stage()

        # The fake domain was queried with OPS tags, never the fem_eids —
        # proof the translation happened (else eleType would KeyError).
        assert set(ops.queried_eids) <= set(ops_tags)
        assert FEM_EID_A not in ops.queried_eids
        assert FEM_EID_B not in ops.queried_eids

        # A complete pairing translated every id, so the file is stamped.
        from apeGmsh.results.readers._native import NativeReader
        with NativeReader(out) as r:
            assert r.element_index_space == "fem"

        from apeGmsh.results import Results

        with Results.from_native(out, model=_open(model_h5)) as r:
            s = r.stage(r.stages[0].id)
            slab = s.elements.gauss.get(component="stress_xx")
            # element_index must be in fem space (the viewer contract).
            assert set(int(e) for e in slab.element_index) == {
                FEM_EID_A, FEM_EID_B,
            }


def _open(model_h5: Path):
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    return OpenSeesModel.from_h5(str(model_h5))
