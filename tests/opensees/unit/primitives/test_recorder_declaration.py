"""Unit tests for :class:`RecorderRecord` + :class:`RecorderDeclaration`.

Phase 9 commit 2 ships the typed-container surface for the unified
recorder declaration. Validation rules covered here:

- Category whitelist
- Per-category canonical component validation
- ``raw=`` escape hatch bypasses validation
- Cadence mutex (``dt`` XOR ``n_steps``)
- Selector mutex (``ids`` vs named selectors)
- Modal-only ``n_modes`` contract
- Indexed canonicals (``state_variable_<n>``) in element-level categories
- Frozen/slots dataclass shape
- :class:`RecorderDeclaration` is a :class:`Recorder` primitive

``_emit`` is not exercised end-to-end here — it raises
:class:`NotImplementedError` pre-commit-3 by design (bridge fan-out
supplies the per-record translation context).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from apeGmsh.opensees._internal.types import Primitive, Recorder
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.recorder import (
    ALL_RECORDER_CATEGORIES,
    RecorderDeclaration,
    RecorderRecord,
)


# ---------------------------------------------------------------------------
# RecorderRecord — construction + validation
# ---------------------------------------------------------------------------

class TestRecorderRecordConstruction:
    def test_minimal_nodes_record(self) -> None:
        r = RecorderRecord(category="nodes")
        assert r.category == "nodes"
        assert r.components == ()
        assert r.raw == ()

    def test_components_accepted_when_canonical(self) -> None:
        r = RecorderRecord(
            category="nodes",
            components=("displacement_x", "displacement_y"),
        )
        assert r.components == ("displacement_x", "displacement_y")

    def test_components_accepted_for_each_category(self) -> None:
        cases: list[tuple[str, tuple[str, ...]]] = [
            ("nodes",         ("displacement_x", "reaction_force_y")),
            ("elements",      ("nodal_resisting_force_x",)),
            ("line_stations", ("axial_force", "bending_moment_y")),
            ("gauss",         ("stress_xx", "strain_yy", "von_mises_stress")),
            ("fibers",        ("fiber_stress", "fiber_strain")),
            ("layers",        ("fiber_stress",)),
        ]
        for category, components in cases:
            r = RecorderRecord(category=category, components=components)
            assert r.category == category
            assert r.components == components

    def test_raw_escape_hatch_bypasses_canonical_validation(self) -> None:
        r = RecorderRecord(
            category="nodes",
            raw=("some_unsupported_token",),
        )
        assert r.raw == ("some_unsupported_token",)

    def test_indexed_canonical_in_element_category(self) -> None:
        # state_variable_<n> is recognized as canonical via regex,
        # allowed in element-level categories.
        r = RecorderRecord(
            category="gauss",
            components=("state_variable_0", "state_variable_5"),
        )
        assert r.components == ("state_variable_0", "state_variable_5")

    def test_indexed_fiber_canonical(self) -> None:
        r = RecorderRecord(
            category="fibers",
            components=("fiber_stress_3",),
        )
        assert r.components == ("fiber_stress_3",)


class TestRecorderRecordValidation:
    def test_unknown_category_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown category"):
            RecorderRecord(category="bogus")

    def test_unknown_component_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in canonical vocabulary"):
            RecorderRecord(category="nodes", components=("bogus_thing",))

    def test_component_from_wrong_category_rejected(self) -> None:
        # `axial_force` belongs to line_stations, not nodes.
        with pytest.raises(ValueError, match="not in canonical vocabulary"):
            RecorderRecord(category="nodes", components=("axial_force",))

    def test_cadence_mutex(self) -> None:
        with pytest.raises(ValueError, match="at most one"):
            RecorderRecord(category="nodes", dt=0.01, n_steps=5)

    def test_cadence_either_alone_is_ok(self) -> None:
        RecorderRecord(category="nodes", dt=0.01)
        RecorderRecord(category="nodes", n_steps=5)
        RecorderRecord(category="nodes")  # neither — every step

    def test_ids_mutex_with_pg(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            RecorderRecord(
                category="nodes",
                pg=("Top",),
                ids=(1, 2, 3),
            )

    def test_ids_mutex_with_label(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            RecorderRecord(
                category="nodes",
                label=("MyLabel",),
                ids=(1,),
            )

    def test_ids_mutex_with_selection(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            RecorderRecord(
                category="nodes",
                selection=("MySelection",),
                ids=(1,),
            )

    def test_named_selectors_can_combine(self) -> None:
        # pg + label + selection together is OK (treated as union)
        r = RecorderRecord(
            category="nodes",
            pg=("Top",),
            label=("MyLabel",),
            selection=("MySelection",),
        )
        assert r.pg == ("Top",)
        assert r.label == ("MyLabel",)
        assert r.selection == ("MySelection",)

    def test_modal_requires_n_modes(self) -> None:
        with pytest.raises(ValueError, match="n_modes"):
            RecorderRecord(category="modal")

    def test_modal_n_modes_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_modes"):
            RecorderRecord(category="modal", n_modes=0)

    def test_n_modes_invalid_for_non_modal(self) -> None:
        with pytest.raises(ValueError, match="only valid for category='modal'"):
            RecorderRecord(category="nodes", n_modes=5)


# ---------------------------------------------------------------------------
# RecorderRecord — dataclass shape
# ---------------------------------------------------------------------------

class TestRecorderRecordShape:
    def test_is_frozen_dataclass(self) -> None:
        assert is_dataclass(RecorderRecord)
        params = RecorderRecord.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen
        assert params.kw_only

    def test_frozen_raises_on_mutation(self) -> None:
        r = RecorderRecord(category="nodes")
        with pytest.raises(FrozenInstanceError):
            r.category = "elements"  # type: ignore[misc]

    def test_has_slots(self) -> None:
        assert hasattr(RecorderRecord, "__slots__")

    def test_all_fields_kw_only(self) -> None:
        for f in fields(RecorderRecord):
            assert f.kw_only is True


# ---------------------------------------------------------------------------
# RecorderDeclaration — bundle + Primitive contract
# ---------------------------------------------------------------------------

class TestRecorderDeclaration:
    def test_minimal_construction(self) -> None:
        d = RecorderDeclaration(records=())
        assert d.records == ()
        assert d.name == "default"
        assert d.ndm == 3
        assert d.ndf == 6

    def test_with_records(self) -> None:
        d = RecorderDeclaration(
            records=(
                RecorderRecord(category="nodes", components=("displacement_x",)),
                RecorderRecord(category="gauss", components=("stress_xx",)),
            ),
            name="primary",
            ndm=3,
            ndf=6,
        )
        assert len(d.records) == 2
        assert d.name == "primary"

    def test_ndm_ndf_captured(self) -> None:
        # Phase 9 D8: bridge passes its ndm/ndf at construction;
        # the declaration captures them verbatim.
        d = RecorderDeclaration(records=(), ndm=2, ndf=3)
        assert d.ndm == 2 and d.ndf == 3

    def test_inherits_from_recorder(self) -> None:
        assert issubclass(RecorderDeclaration, Recorder)
        assert issubclass(RecorderDeclaration, Primitive)

    def test_is_frozen_dataclass(self) -> None:
        assert is_dataclass(RecorderDeclaration)
        params = RecorderDeclaration.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen
        assert params.kw_only

    def test_has_slots(self) -> None:
        assert hasattr(RecorderDeclaration, "__slots__")

    def test_frozen_raises_on_mutation(self) -> None:
        d = RecorderDeclaration(records=())
        with pytest.raises(FrozenInstanceError):
            d.ndm = 99  # type: ignore[misc]

    def test_dependencies_empty(self) -> None:
        d = RecorderDeclaration(records=())
        assert d.dependencies() == ()

    def test_emit_raises_pre_commit_3(self) -> None:
        # Phase 9 commit 3 wires emit_recorder_spec to handle this;
        # direct _emit raises until then.
        d = RecorderDeclaration(records=())
        with pytest.raises(NotImplementedError, match="emit_recorder_spec"):
            d._emit(RecordingEmitter(), tag=1)

    def test_repr_includes_class_name(self) -> None:
        d = RecorderDeclaration(records=())
        assert "RecorderDeclaration" in repr(d)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestCategoryConstants:
    def test_all_categories_listed(self) -> None:
        # Defensive: if a new category is added in the source, this
        # test surfaces the change in CI immediately so the matching
        # consumers (emit, capture, h5 writer) can be updated.
        assert ALL_RECORDER_CATEGORIES == (
            "nodes",
            "elements",
            "line_stations",
            "gauss",
            "fibers",
            "layers",
            "modal",
        )
