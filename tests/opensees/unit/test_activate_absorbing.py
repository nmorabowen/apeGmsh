"""Unit tests for the absorbing-boundary stage flip (ADR 0054, AB-3).

Covers the emitter method (``flip_element_stage``), the build's flip path
(``absorbing_ele_tags`` — element resolution, per-partition filtering,
missing-eid fail-loud — as the tag plan runs it, then the real
``emit_activate_absorbing`` writing the planned lines), and the staged verb
validation.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import (
    ActivateAbsorbingRecord,
    BridgeError,
    absorbing_ele_tags,
    emit_activate_absorbing,
    parameter_flip_sites,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter

from tests.opensees._helpers.tag_plan import parameter_plan


def _emit(records, *, eid_to_tag, element_owner=None, partition_rank=None):
    """Plan the flips as ``plan_tags`` does (resolve each record's
    elements, then number the sites), then write them through the real
    ``emit_activate_absorbing``."""
    e = RecordingEmitter()
    fem = cast("object", MagicMock(name="FEMData"))
    resolved = [
        absorbing_ele_tags(rec, fem, eid_to_tag, element_owner, partition_rank)
        for rec in records
    ]
    plan = parameter_plan(
        parameter_flip_sites(
            "flip_element_stage", records, resolved, partition_rank),
        partitioned=partition_rank is not None)
    emit_activate_absorbing(records, e, plan, partition_rank=partition_rank)
    return [c for c in e.calls if c[0] == "flip_element_stage"]


class TestEmitterMethod:
    def test_flip_block_structure(self) -> None:
        e = RecordingEmitter()
        e.flip_element_stage(7, (10, 11, 12))
        assert e.calls == [("flip_element_stage", (7, (10, 11, 12)), {})]

    def test_tcl_rendering(self) -> None:
        from apeGmsh.opensees.emitter.tcl import TclEmitter

        e = TclEmitter()
        e.flip_element_stage(3, (10, 11))
        txt = "\n".join(e._lines)
        assert "parameter 3" in txt
        assert "addToParameter 3 element 10 stage" in txt
        assert "addToParameter 3 element 11 stage" in txt
        assert "updateParameter 3 1" in txt
        assert "remove parameter 3" in txt


class TestEmitFunction:
    def test_explicit_elements(self) -> None:
        rec = ActivateAbsorbingRecord(pg=None, elements=(101, 102))
        calls = _emit([rec], eid_to_tag={101: 5, 102: 6})
        assert calls == [("flip_element_stage", (1, (5, 6)), {})]

    def test_missing_eid_single_partition_raises(self) -> None:
        rec = ActivateAbsorbingRecord(pg=None, elements=(101, 999))
        with pytest.raises(BridgeError, match="not .*registered"):
            _emit([rec], eid_to_tag={101: 5})

    def test_partition_filters_to_owned(self) -> None:
        rec = ActivateAbsorbingRecord(pg=None, elements=(101, 102, 103))
        calls = _emit(
            [rec],
            eid_to_tag={101: 5, 102: 6, 103: 7},
            element_owner={101: 0, 102: 1, 103: 0},
            partition_rank=0,
        )
        # Only rank-0 elements (101, 103) flip on this rank.
        assert calls == [("flip_element_stage", (1, (5, 7)), {})]

    def test_partition_missing_eid_silently_skipped(self) -> None:
        rec = ActivateAbsorbingRecord(pg=None, elements=(101, 102))
        # 102 owned by rank 0 but absent from this rank's tag map → skip, no raise.
        calls = _emit(
            [rec],
            eid_to_tag={101: 5},
            element_owner={101: 0, 102: 0},
            partition_rank=0,
        )
        assert calls == [("flip_element_stage", (1, (5,)), {})]

    def test_no_owned_elements_emits_nothing(self) -> None:
        rec = ActivateAbsorbingRecord(pg=None, elements=(101,))
        calls = _emit(
            [rec],
            eid_to_tag={101: 5},
            element_owner={101: 1},
            partition_rank=0,
        )
        assert calls == []


class TestStageVerbValidation:
    def _ops(self) -> apeSees:
        return apeSees(cast("object", MagicMock(name="FEMData")))

    def test_requires_exactly_one_target(self) -> None:
        ops = self._ops()
        with pytest.raises(ValueError, match="exactly one"):
            with ops.stage(name="dyn") as s:
                s.activate_absorbing()  # neither pg nor elements
        with pytest.raises(ValueError, match="exactly one"):
            with ops.stage(name="dyn") as s:
                s.activate_absorbing(pg="abs", elements=[1])

    def test_records_pg(self) -> None:
        ops = self._ops()
        with ops.stage(name="dyn") as s:
            rec = s.activate_absorbing(pg="absorbing")
            # minimal chain so __exit__ doesn't raise
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-6, max_iter=20),
                algorithm=ops.algorithm.Newton(),
                integrator=ops.integrator.LoadControl(dlam=0.1),
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.RCM(),
                system=ops.system.UmfPack(),
                analysis=ops.analysis.Static(),
            )
            s.run(n_increments=1)
        assert isinstance(rec, ActivateAbsorbingRecord)
        assert rec.pg == "absorbing" and rec.elements is None
