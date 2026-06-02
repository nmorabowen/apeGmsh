"""Unit tests for :class:`apeGmsh.opensees.section.aggregator.Aggregator`.

Covers:

* construction (mapping of DOF code → UniaxialMaterial, optional
  ``base_section``)
* validation (empty mapping rejected; bad DOF code rejected;
  non-UniaxialMaterial value rejected; non-Section base_section rejected)
* ``dependencies()`` returns materials + base section in order
* ``_emit`` records the right ``section Aggregator`` call with tag
  resolution per material (and with ``-section $base_tag`` when a base
  is supplied)
* ``__repr__`` includes the class name
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.uniaxial import (
    ElasticMaterial,
    Steel02,
    Viscous,
)
from apeGmsh.opensees.section.aggregator import (
    AGGREGATOR_DOF_CODES,
    Aggregator,
)
from apeGmsh.opensees.section.beam import ElasticSection


class TestAggregatorConstruction:
    def test_single_dof(self) -> None:
        m = ElasticMaterial(E=2e11)
        agg = Aggregator(materials_by_dof={"P": m})
        assert agg.materials_by_dof["P"] is m
        assert agg.base_section is None

    def test_all_six_dofs(self) -> None:
        m = ElasticMaterial(E=2e11)
        mapping = {code: m for code in AGGREGATOR_DOF_CODES}
        agg = Aggregator(materials_by_dof=mapping)
        assert tuple(agg.materials_by_dof.keys()) == AGGREGATOR_DOF_CODES

    def test_with_base_section(self) -> None:
        m = ElasticMaterial(E=2e11)
        base = ElasticSection(E=2e11, A=0.01, Iz=1e-4)
        agg = Aggregator(materials_by_dof={"Mz": m}, base_section=base)
        assert agg.base_section is base


class TestAggregatorValidation:
    def test_empty_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Aggregator(materials_by_dof={})

    def test_unknown_dof_raises(self) -> None:
        m = ElasticMaterial(E=2e11)
        with pytest.raises(ValueError, match="unknown DOF code"):
            Aggregator(
                materials_by_dof={"BogusCode": m},  # type: ignore[dict-item]
            )

    def test_non_uniaxial_value_raises(self) -> None:
        class NotAMaterial:
            pass

        with pytest.raises(TypeError, match="UniaxialMaterial"):
            Aggregator(
                materials_by_dof={"P": NotAMaterial()},  # type: ignore[dict-item]
            )

    def test_non_section_base_raises(self) -> None:
        m = ElasticMaterial(E=2e11)

        class NotASection:
            pass

        with pytest.raises(TypeError, match="Section"):
            Aggregator(
                materials_by_dof={"P": m},
                base_section=NotASection(),  # type: ignore[arg-type]
            )

    def test_rate_dependent_material_raises(self) -> None:
        # A Viscous (rate-dependent) material is silently inert inside a
        # section Aggregator -> zeroLengthSection (no strain-rate
        # channel). The aggregator must fail loud rather than emit a
        # dead dashpot.
        dashpot = Viscous(C=1.0e5)
        with pytest.raises(ValueError, match="rate-dependent"):
            Aggregator(materials_by_dof={"Vy": dashpot})

    def test_rate_dependent_material_rejected_even_with_base(self) -> None:
        dashpot = Viscous(C=1.0e5)
        base = ElasticSection(E=2e11, A=0.01, Iz=1e-4)
        with pytest.raises(ValueError, match="rate-dependent"):
            Aggregator(
                materials_by_dof={"Vy": dashpot},
                base_section=base,
            )


class TestAggregatorDependencies:
    def test_pure_aggregator_deps_are_materials_only(self) -> None:
        m1 = ElasticMaterial(E=2e11)
        m2 = Steel02(fy=420e6, E=2e11, b=0.01)
        agg = Aggregator(materials_by_dof={"P": m1, "Mz": m2})
        deps = agg.dependencies()
        assert m1 in deps
        assert m2 in deps
        assert len(deps) == 2

    def test_base_section_is_appended_to_deps(self) -> None:
        m = ElasticMaterial(E=2e11)
        base = ElasticSection(E=2e11, A=0.01, Iz=1e-4)
        agg = Aggregator(materials_by_dof={"Mz": m}, base_section=base)
        deps = agg.dependencies()
        assert m in deps
        assert base in deps
        # Base goes after materials.
        assert deps.index(base) > deps.index(m)


class TestAggregatorEmit:
    def test_emit_single_dof(self) -> None:
        m = ElasticMaterial(E=2e11)
        agg = Aggregator(materials_by_dof={"P": m})
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 7 if p is m else 0)
        agg._emit(rec, tag=5)
        assert rec.calls == [
            ("section", ("Aggregator", 5, 7, "P"), {}),
        ]

    def test_emit_multi_dof_preserves_insertion_order(self) -> None:
        m_axial = ElasticMaterial(E=2e11)
        m_bending = Steel02(fy=420e6, E=2e11, b=0.01)
        m_shear = ElasticMaterial(E=8e10)
        # User-typed order: Mz first, then P, then Vy.  Aggregator must
        # round-trip THIS order, not an alphabetised or canonicalised
        # one.
        agg = Aggregator(
            materials_by_dof={
                "Mz": m_bending,
                "P":  m_axial,
                "Vy": m_shear,
            },
        )
        rec = RecordingEmitter()
        set_tag_resolver(
            rec,
            lambda p: {
                id(m_bending): 11,
                id(m_axial):   12,
                id(m_shear):   13,
            }[id(p)],
        )
        agg._emit(rec, tag=42)
        assert rec.calls == [
            (
                "section",
                ("Aggregator", 42, 11, "Mz", 12, "P", 13, "Vy"),
                {},
            )
        ]

    def test_emit_with_base_section_appends_section_flag(self) -> None:
        m = ElasticMaterial(E=2e11)
        base = ElasticSection(E=2e11, A=0.01, Iz=1e-4)
        agg = Aggregator(
            materials_by_dof={"Mz": m},
            base_section=base,
        )
        rec = RecordingEmitter()
        set_tag_resolver(
            rec,
            lambda p: {id(m): 9, id(base): 4}[id(p)],
        )
        agg._emit(rec, tag=20)
        assert rec.calls == [
            (
                "section",
                ("Aggregator", 20, 9, "Mz", "-section", 4),
                {},
            )
        ]


class TestAggregatorNamespace:
    def _make_ops(self) -> apeSees:
        return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]

    def test_namespace_constructs_and_registers(self) -> None:
        ops = self._make_ops()
        m = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
        agg = ops.section.Aggregator(materials_by_dof={"P": m})
        assert isinstance(agg, Aggregator)
        # Tag allocator uses per-family namespaces — uniaxials and
        # sections each begin numbering from 1.
        assert ops.tag_for(m) == 1
        assert ops.tag_for(agg) == 1

    def test_namespace_with_base_section(self) -> None:
        ops = self._make_ops()
        m = ops.uniaxialMaterial.ElasticMaterial(E=2e11)
        base = ops.section.Elastic(E=2e11, A=0.01, Iz=1e-4)
        agg = ops.section.Aggregator(
            materials_by_dof={"Mz": m},
            base_section=base,
        )
        assert agg.base_section is base


class TestAggregatorContract:
    def test_is_frozen(self) -> None:
        m = ElasticMaterial(E=2e11)
        agg = Aggregator(materials_by_dof={"P": m})
        with pytest.raises(Exception):  # FrozenInstanceError
            agg.base_section = None  # type: ignore[misc]

    def test_repr_includes_class_name(self) -> None:
        m = ElasticMaterial(E=2e11)
        agg = Aggregator(materials_by_dof={"P": m})
        assert "Aggregator" in repr(agg)

    def test_emit_without_tag_resolver_raises(self) -> None:
        m = ElasticMaterial(E=2e11)
        agg = Aggregator(materials_by_dof={"P": m})
        rec = RecordingEmitter()
        with pytest.raises(RuntimeError, match="tag resolver"):
            agg._emit(rec, tag=1)
