"""Unit tests for the beam-column element family.

Covers Phase 2α primitives:

* :class:`apeGmsh.opensees.element.beam_column.elasticBeamColumn`
* :class:`apeGmsh.opensees.element.beam_column.forceBeamColumn`
* :class:`apeGmsh.opensees.element.beam_column.dispBeamColumn`
* :class:`apeGmsh.opensees.element.beam_column.ElasticTimoshenkoBeam`

Each ``_emit`` requires both a tag-resolver context (for section /
transform tags) and an element-nodes context (for the per-element
i, j node tags) attached to the emitter. Tests install both via
:func:`set_tag_resolver` / :func:`set_element_nodes`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import (
    BeamIntegration,
    NDMaterial,
    Primitive,
    Section,
    UniaxialMaterial,
)
from apeGmsh.opensees.element.beam_column import (
    ElasticTimoshenkoBeam,
    LadrunoDispBeamColumn,
    LadrunoIMKBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.integration import Lobatto
from apeGmsh.opensees.transform import Corotational, Linear, PDelta


# ---------------------------------------------------------------------------
# Test-local fakes — same shape as the Phase 1 fiber tests.
# A real Section primitive is heavy to construct; a fake whose only
# job is to stand in as a typed dependency is plenty.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeMat(UniaxialMaterial):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:  # pragma: no cover
        emitter.uniaxialMaterial("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeSection(Section):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:  # pragma: no cover
        emitter.section("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeNDMat(NDMaterial):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:  # pragma: no cover
        emitter.nDMaterial("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


def _resolver_from(tags: dict[int, int]) -> object:
    """Return a callable Primitive -> tag via id-keyed map."""
    def _resolve(prim: Primitive) -> int:
        return tags[id(prim)]
    return _resolve


# ===========================================================================
# elasticBeamColumn
# ===========================================================================

class TestElasticBeamColumnConstruction:
    def test_construct_2d_minimum(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = elasticBeamColumn(
            pg="Cols", transf=t,
            A=0.04, E=2e11, Iz=2e-4,
        )
        assert e.pg == "Cols"
        assert e.transf is t
        assert e.A == 0.04
        assert e.E == 2e11
        assert e.Iz == 2e-4
        assert e.Iy is None
        assert e.G is None
        assert e.J is None
        assert e.mass is None
        assert e.c_mass is False

    def test_construct_3d_minimum(self) -> None:
        t = PDelta(vecxz=(0.0, 0.0, 1.0))
        e = elasticBeamColumn(
            pg="Cols", transf=t,
            A=0.04, E=2e11, Iz=2e-4,
            Iy=1.5e-4, G=8e10, J=1e-4,
        )
        assert e.Iy == 1.5e-4
        assert e.G == 8e10
        assert e.J == 1e-4

    def test_dependencies_are_transf(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = elasticBeamColumn(
            pg="Cols", transf=t, A=0.04, E=2e11, Iz=2e-4,
        )
        assert e.dependencies() == (t,)


class TestElasticBeamColumnValidation:
    def _t(self) -> Linear:
        return Linear(vecxz=(0.0, 0.0, 1.0))

    def test_pg_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="pg must be a non-empty"):
            elasticBeamColumn(
                pg="", transf=self._t(), A=0.04, E=2e11, Iz=2e-4,
            )

    @pytest.mark.parametrize("A", [0.0, -1e-3])
    def test_A_must_be_positive(self, A: float) -> None:
        with pytest.raises(ValueError, match="A must be > 0"):
            elasticBeamColumn(
                pg="Cols", transf=self._t(), A=A, E=2e11, Iz=2e-4,
            )

    @pytest.mark.parametrize("E", [0.0, -1e9])
    def test_E_must_be_positive(self, E: float) -> None:
        with pytest.raises(ValueError, match="E must be > 0"):
            elasticBeamColumn(
                pg="Cols", transf=self._t(), A=0.04, E=E, Iz=2e-4,
            )

    @pytest.mark.parametrize("Iz", [0.0, -1e-5])
    def test_Iz_must_be_positive(self, Iz: float) -> None:
        with pytest.raises(ValueError, match="Iz must be > 0"):
            elasticBeamColumn(
                pg="Cols", transf=self._t(), A=0.04, E=2e11, Iz=Iz,
            )

    def test_partial_3d_iy_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="3-D variant requires"):
            elasticBeamColumn(
                pg="Cols", transf=self._t(),
                A=0.04, E=2e11, Iz=2e-4, Iy=1.5e-4,
            )

    def test_partial_3d_g_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="3-D variant requires"):
            elasticBeamColumn(
                pg="Cols", transf=self._t(),
                A=0.04, E=2e11, Iz=2e-4, G=8e10,
            )

    def test_partial_3d_j_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="3-D variant requires"):
            elasticBeamColumn(
                pg="Cols", transf=self._t(),
                A=0.04, E=2e11, Iz=2e-4, J=1e-4,
            )

    def test_negative_Iy_rejected(self) -> None:
        with pytest.raises(ValueError, match="Iy must be > 0"):
            elasticBeamColumn(
                pg="Cols", transf=self._t(),
                A=0.04, E=2e11, Iz=2e-4,
                Iy=-1e-4, G=8e10, J=1e-4,
            )

    def test_mass_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="mass must be >= 0"):
            elasticBeamColumn(
                pg="Cols", transf=self._t(),
                A=0.04, E=2e11, Iz=2e-4, mass=-1.0,
            )


class TestElasticBeamColumnEmit:
    def _build_2d(self) -> tuple[elasticBeamColumn, Linear]:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        spec = elasticBeamColumn(
            pg="Cols", transf=t, A=0.04, E=2e11, Iz=2e-4,
        )
        return spec, t

    def _build_3d(self) -> tuple[elasticBeamColumn, PDelta]:
        t = PDelta(vecxz=(0.0, 0.0, 1.0))
        spec = elasticBeamColumn(
            pg="Cols", transf=t,
            A=0.04, E=2e11, Iz=2e-4,
            Iy=1.5e-4, G=8e10, J=1e-4,
        )
        return spec, t

    def test_emit_2d_records_correct_call(self) -> None:
        spec, t = self._build_2d()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 7}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        assert rec.calls == [
            (
                "element",
                ("elasticBeamColumn", 42, 10, 20, 0.04, 2e11, 2e-4, 7),
                {},
            )
        ]

    def test_emit_3d_records_correct_call(self) -> None:
        spec, t = self._build_3d()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 5}))
        set_element_nodes(rec, (1, 2))

        spec._emit(rec, tag=100)

        # 3-D: A E G J Iy Iz transfTag.
        assert rec.calls == [
            (
                "element",
                (
                    "elasticBeamColumn", 100, 1, 2,
                    0.04, 2e11, 8e10, 1e-4, 1.5e-4, 2e-4, 5,
                ),
                {},
            )
        ]

    def test_emit_with_mass_appends_flag(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        spec = elasticBeamColumn(
            pg="Cols", transf=t, A=0.04, E=2e11, Iz=2e-4, mass=12.5,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 3}))
        set_element_nodes(rec, (10, 11))

        spec._emit(rec, tag=1)

        assert rec.calls[0][1] == (
            "elasticBeamColumn", 1, 10, 11,
            0.04, 2e11, 2e-4, 3, "-mass", 12.5,
        )

    def test_emit_with_c_mass_appends_flag(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        spec = elasticBeamColumn(
            pg="Cols", transf=t, A=0.04, E=2e11, Iz=2e-4, c_mass=True,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 9}))
        set_element_nodes(rec, (1, 2))

        spec._emit(rec, tag=1)

        assert rec.calls[0][1] == (
            "elasticBeamColumn", 1, 1, 2, 0.04, 2e11, 2e-4, 9, "-cMass",
        )

    def test_emit_without_node_context_raises(self) -> None:
        spec, _t = self._build_2d()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({}))
        with pytest.raises(RuntimeError, match="element"):
            spec._emit(rec, tag=1)

    def test_emit_without_resolver_raises(self) -> None:
        spec, _t = self._build_2d()
        rec = RecordingEmitter()
        set_element_nodes(rec, (1, 2))
        with pytest.raises(RuntimeError, match="resolver"):
            spec._emit(rec, tag=1)

    def test_emit_with_three_nodes_rejected(self) -> None:
        spec, t = self._build_2d()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 1}))
        set_element_nodes(rec, (1, 2, 3))
        with pytest.raises(ValueError, match="expected 2 node tags"):
            spec._emit(rec, tag=1)


# ===========================================================================
# forceBeamColumn
# ===========================================================================

def _integ_from(sec: _FakeSection, n_ip: int = 5) -> Lobatto:
    """Build a Lobatto integration rule composing the fake section."""
    return Lobatto(section=sec, n_ip=n_ip)


class TestForceBeamColumnConstruction:
    def test_construct_minimum(self) -> None:
        sec = _FakeSection(name="col_sec")
        integ = _integ_from(sec)
        t = Corotational(vecxz=(0.0, 0.0, 1.0))
        e = forceBeamColumn(pg="Cols", transf=t, integration=integ)
        assert e.integration is integ
        assert e.transf is t
        assert e.mass is None
        assert e.max_iter is None
        assert e.tol is None

    def test_dependencies(self) -> None:
        """forceBeamColumn returns (integration, transf) — sections are
        composed into the integration rule, not the element."""
        sec = _FakeSection(name="col_sec")
        integ = _integ_from(sec)
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = forceBeamColumn(pg="Cols", transf=t, integration=integ)
        assert e.dependencies() == (integ, t)


class TestForceBeamColumnValidation:
    def _integ(self) -> BeamIntegration:
        return _integ_from(_FakeSection(name="x"))

    def _t(self) -> Linear:
        return Linear(vecxz=(0.0, 0.0, 1.0))

    def test_pg_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="pg must be a non-empty"):
            forceBeamColumn(pg="", transf=self._t(), integration=self._integ())

    def test_mass_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="mass must be >= 0"):
            forceBeamColumn(
                pg="Cols", transf=self._t(), integration=self._integ(),
                mass=-0.1,
            )

    def test_iter_partial_max_iter_only_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="max_iter and tol must be supplied together"
        ):
            forceBeamColumn(
                pg="Cols", transf=self._t(), integration=self._integ(),
                max_iter=10,
            )

    def test_iter_partial_tol_only_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="max_iter and tol must be supplied together"
        ):
            forceBeamColumn(
                pg="Cols", transf=self._t(), integration=self._integ(),
                tol=1e-8,
            )

    def test_iter_negative_max_iter_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be > 0"):
            forceBeamColumn(
                pg="Cols", transf=self._t(), integration=self._integ(),
                max_iter=0, tol=1e-8,
            )

    def test_iter_negative_tol_rejected(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            forceBeamColumn(
                pg="Cols", transf=self._t(), integration=self._integ(),
                max_iter=10, tol=0.0,
            )


class TestForceBeamColumnEmit:
    def _build(
        self,
    ) -> tuple[forceBeamColumn, BeamIntegration, Corotational]:
        sec = _FakeSection(name="col_sec")
        integ = _integ_from(sec)
        t = Corotational(vecxz=(0.0, 0.0, 1.0))
        spec = forceBeamColumn(pg="Cols", transf=t, integration=integ)
        return spec, integ, t

    def test_emit_records_correct_call(self) -> None:
        """Modern shape: `element forceBeamColumn tag iNode jNode transfTag integrationTag`."""
        spec, integ, t = self._build()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        assert rec.calls == [
            (
                "element",
                ("forceBeamColumn", 42, 10, 20, 2, 7),
                {},
            )
        ]

    def test_emit_with_mass_flag(self) -> None:
        spec_basic, integ, t = self._build()
        spec = forceBeamColumn(
            pg=spec_basic.pg, transf=t, integration=integ,
            mass=12.5,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        assert rec.calls[0][1] == (
            "forceBeamColumn", 42, 10, 20, 2, 7,
            "-mass", 12.5,
        )

    def test_emit_with_iter_flag(self) -> None:
        spec_basic, integ, t = self._build()
        spec = forceBeamColumn(
            pg=spec_basic.pg, transf=t, integration=integ,
            max_iter=20, tol=1e-9,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        assert rec.calls[0][1] == (
            "forceBeamColumn", 42, 10, 20, 2, 7,
            "-iter", 20, 1e-9,
        )

    def test_emit_with_mass_and_iter(self) -> None:
        spec_basic, integ, t = self._build()
        spec = forceBeamColumn(
            pg=spec_basic.pg, transf=t, integration=integ,
            mass=2.5, max_iter=20, tol=1e-9,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        # Order: -mass first, then -iter.
        assert rec.calls[0][1] == (
            "forceBeamColumn", 42, 10, 20, 2, 7,
            "-mass", 2.5,
            "-iter", 20, 1e-9,
        )

    def test_emit_without_node_context_raises(self) -> None:
        spec, integ, t = self._build()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        with pytest.raises(RuntimeError, match="element"):
            spec._emit(rec, tag=1)

    def test_emit_without_resolver_raises(self) -> None:
        spec, _integ, _t = self._build()
        rec = RecordingEmitter()
        set_element_nodes(rec, (10, 20))
        with pytest.raises(RuntimeError, match="resolver"):
            spec._emit(rec, tag=1)

    def test_emit_with_one_node_rejected(self) -> None:
        spec, integ, t = self._build()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        set_element_nodes(rec, (10,))
        with pytest.raises(ValueError, match="expected 2 node tags"):
            spec._emit(rec, tag=1)


# ===========================================================================
# dispBeamColumn
# ===========================================================================

class TestDispBeamColumnConstruction:
    def test_construct_minimum(self) -> None:
        sec = _FakeSection(name="x")
        integ = _integ_from(sec, n_ip=4)
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = dispBeamColumn(pg="Cols", transf=t, integration=integ)
        assert e.integration is integ
        assert e.transf is t
        assert e.c_mass is False
        assert e.mass is None

    def test_dependencies(self) -> None:
        sec = _FakeSection(name="x")
        integ = _integ_from(sec, n_ip=4)
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = dispBeamColumn(pg="Cols", transf=t, integration=integ)
        assert e.dependencies() == (integ, t)


class TestDispBeamColumnValidation:
    def _integ(self) -> BeamIntegration:
        return _integ_from(_FakeSection(name="x"))

    def _t(self) -> Linear:
        return Linear(vecxz=(0.0, 0.0, 1.0))

    def test_pg_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="pg must be a non-empty"):
            dispBeamColumn(pg="", transf=self._t(), integration=self._integ())

    def test_mass_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="mass must be >= 0"):
            dispBeamColumn(
                pg="Cols", transf=self._t(), integration=self._integ(),
                mass=-1.0,
            )


class TestDispBeamColumnEmit:
    def _build(self) -> tuple[dispBeamColumn, BeamIntegration, Linear]:
        sec = _FakeSection(name="x")
        integ = _integ_from(sec, n_ip=4)
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        spec = dispBeamColumn(pg="Cols", transf=t, integration=integ)
        return spec, integ, t

    def test_emit_records_correct_call(self) -> None:
        """Modern shape: `element dispBeamColumn tag iNode jNode transfTag integrationTag`."""
        spec, integ, t = self._build()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 11, id(t): 12}))
        set_element_nodes(rec, (3, 4))

        spec._emit(rec, tag=99)

        assert rec.calls == [
            (
                "element",
                ("dispBeamColumn", 99, 3, 4, 12, 11),
                {},
            )
        ]

    def test_emit_with_mass_and_c_mass(self) -> None:
        spec_basic, integ, t = self._build()
        spec = dispBeamColumn(
            pg=spec_basic.pg, transf=t, integration=integ,
            mass=3.0, c_mass=True,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 11, id(t): 12}))
        set_element_nodes(rec, (3, 4))

        spec._emit(rec, tag=1)

        assert rec.calls[0][1] == (
            "dispBeamColumn", 1, 3, 4, 12, 11,
            "-mass", 3.0, "-cMass",
        )


# ===========================================================================
# ElasticTimoshenkoBeam
# ===========================================================================

class TestElasticTimoshenkoBeamConstruction:
    def test_construct_2d_minimum(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = ElasticTimoshenkoBeam(
            pg="Cols", transf=t,
            E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
        )
        assert e.Iy is None
        assert e.J is None
        assert e.Avz is None

    def test_construct_3d_minimum(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = ElasticTimoshenkoBeam(
            pg="Cols", transf=t,
            E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
            Iy=1e-4, J=1e-4, Avz=0.025,
        )
        assert e.Iy == 1e-4
        assert e.J == 1e-4
        assert e.Avz == 0.025

    def test_dependencies(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = ElasticTimoshenkoBeam(
            pg="Cols", transf=t,
            E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
        )
        assert e.dependencies() == (t,)


class TestElasticTimoshenkoBeamValidation:
    def _t(self) -> Linear:
        return Linear(vecxz=(0.0, 0.0, 1.0))

    def test_pg_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="pg must be a non-empty"):
            ElasticTimoshenkoBeam(
                pg="", transf=self._t(),
                E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
            )

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("E", {"E": 0.0}),
            ("G", {"G": -1e10}),
            ("A", {"A": 0.0}),
            ("Iz", {"Iz": -1e-4}),
            ("Avy", {"Avy": 0.0}),
        ],
    )
    def test_required_positives(
        self, name: str, kwargs: dict[str, float],
    ) -> None:
        full: dict[str, float] = {
            "E": 2e11, "G": 8e10, "A": 0.04,
            "Iz": 2e-4, "Avy": 0.03,
        }
        full.update(kwargs)
        with pytest.raises(ValueError, match=f"{name} must be > 0"):
            ElasticTimoshenkoBeam(
                pg="Cols", transf=self._t(), **full,
            )

    def test_partial_3d_iy_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="3-D variant requires"):
            ElasticTimoshenkoBeam(
                pg="Cols", transf=self._t(),
                E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
                Iy=1e-4,
            )

    def test_partial_3d_avz_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="3-D variant requires"):
            ElasticTimoshenkoBeam(
                pg="Cols", transf=self._t(),
                E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
                Avz=0.025,
            )

    def test_negative_J_rejected(self) -> None:
        with pytest.raises(ValueError, match="J must be > 0"):
            ElasticTimoshenkoBeam(
                pg="Cols", transf=self._t(),
                E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
                Iy=1e-4, J=-1e-4, Avz=0.025,
            )


class TestElasticTimoshenkoBeamEmit:
    def _build_2d(self) -> tuple[ElasticTimoshenkoBeam, Linear]:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        spec = ElasticTimoshenkoBeam(
            pg="Cols", transf=t,
            E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
        )
        return spec, t

    def _build_3d(self) -> tuple[ElasticTimoshenkoBeam, PDelta]:
        t = PDelta(vecxz=(0.0, 0.0, 1.0))
        spec = ElasticTimoshenkoBeam(
            pg="Cols", transf=t,
            E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
            Iy=1e-4, J=1e-4, Avz=0.025,
        )
        return spec, t

    def test_emit_2d(self) -> None:
        spec, t = self._build_2d()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 7}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        # 2-D: E G A Iz Avy transfTag.
        assert rec.calls == [
            (
                "element",
                (
                    "ElasticTimoshenkoBeam", 42, 10, 20,
                    2e11, 8e10, 0.04, 2e-4, 0.03, 7,
                ),
                {},
            )
        ]

    def test_emit_3d(self) -> None:
        spec, t = self._build_3d()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 5}))
        set_element_nodes(rec, (1, 2))

        spec._emit(rec, tag=100)

        # 3-D: E G A J Iy Iz Avy Avz transfTag.
        assert rec.calls == [
            (
                "element",
                (
                    "ElasticTimoshenkoBeam", 100, 1, 2,
                    2e11, 8e10, 0.04, 1e-4, 1e-4, 2e-4, 0.03, 0.025, 5,
                ),
                {},
            )
        ]

    def test_emit_with_mass_and_c_mass(self) -> None:
        spec_basic, t = self._build_2d()
        spec = ElasticTimoshenkoBeam(
            pg=spec_basic.pg, transf=t,
            E=spec_basic.E, G=spec_basic.G, A=spec_basic.A,
            Iz=spec_basic.Iz, Avy=spec_basic.Avy,
            mass=4.0, c_mass=True,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 7}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=1)

        assert rec.calls[0][1] == (
            "ElasticTimoshenkoBeam", 1, 10, 20,
            2e11, 8e10, 0.04, 2e-4, 0.03, 7,
            "-mass", 4.0, "-cMass",
        )


# ===========================================================================
# Repr / family-base sanity
# ===========================================================================

class TestBeamColumnRepr:
    def test_elasticBeamColumn_repr(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = elasticBeamColumn(pg="Cols", transf=t, A=0.04, E=2e11, Iz=2e-4)
        assert "elasticBeamColumn" in repr(e)

    def test_forceBeamColumn_repr(self) -> None:
        sec = _FakeSection(name="x")
        integ = _integ_from(sec)
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = forceBeamColumn(pg="Cols", transf=t, integration=integ)
        assert "forceBeamColumn" in repr(e)

    def test_dispBeamColumn_repr(self) -> None:
        sec = _FakeSection(name="x")
        integ = _integ_from(sec, n_ip=4)
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = dispBeamColumn(pg="Cols", transf=t, integration=integ)
        assert "dispBeamColumn" in repr(e)

    def test_ElasticTimoshenkoBeam_repr(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = ElasticTimoshenkoBeam(
            pg="Cols", transf=t,
            E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
        )
        assert "ElasticTimoshenkoBeam" in repr(e)


# ===========================================================================
# LadrunoDispBeamColumn (Ladruno fork — disp-based + fork hinges)
# ===========================================================================

class TestLadrunoDispBeamColumn:
    def _bits(self) -> tuple[BeamIntegration, Linear]:
        return _integ_from(_FakeSection(name="sec")), Linear(vecxz=(0, 0, 1))

    def test_emit_minimal(self) -> None:
        integ, t = self._bits()
        spec = LadrunoDispBeamColumn(pg="C", transf=t, integration=integ)
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        set_element_nodes(rec, (10, 20))
        spec._emit(rec, tag=1)
        assert rec.calls[0][1] == ("LadrunoDispBeamColumn", 1, 10, 20, 2, 7)

    def test_emit_lch_nl_mass(self) -> None:
        integ, t = self._bits()
        spec = LadrunoDispBeamColumn(
            pg="C", transf=t, integration=integ,
            mass=3.0, c_mass=True, lch=0.25, nl=True,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        set_element_nodes(rec, (10, 20))
        spec._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "LadrunoDispBeamColumn", 2, 10, 20, 2, 7,
            "-mass", 3.0, "-cMass", "-lch", 0.25, "-nl",
        )

    def test_emit_lch_element_keyword(self) -> None:
        integ, t = self._bits()
        spec = LadrunoDispBeamColumn(
            pg="C", transf=t, integration=integ, lch="element")
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2}))
        set_element_nodes(rec, (10, 20))
        spec._emit(rec, tag=3)
        assert rec.calls[0][1][-2:] == ("-lch", "element")

    def test_emit_hinges_and_dependencies(self) -> None:
        integ, t = self._bits()
        hz = _FakeMat(name="hz")
        hy = _FakeMat(name="hy")
        spec = LadrunoDispBeamColumn(
            pg="C", transf=t, integration=integ, hinge=hz, hinge_y=hy)
        assert spec.dependencies() == (integ, t, hz, hy)
        rec = RecordingEmitter()
        set_tag_resolver(
            rec, _resolver_from({id(integ): 7, id(t): 2, id(hz): 31, id(hy): 32}))
        set_element_nodes(rec, (10, 20))
        spec._emit(rec, tag=4)
        assert rec.calls[0][1] == (
            "LadrunoDispBeamColumn", 4, 10, 20, 2, 7,
            "-hinge", 31, "-hingeY", 32,
        )

    def test_emit_hinge_biaxial(self) -> None:
        integ, t = self._bits()
        hb = _FakeNDMat(name="hb")
        spec = LadrunoDispBeamColumn(
            pg="C", transf=t, integration=integ, hinge_biaxial=hb)
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(integ): 7, id(t): 2, id(hb): 9}))
        set_element_nodes(rec, (10, 20))
        spec._emit(rec, tag=5)
        assert rec.calls[0][1][-2:] == ("-hingeBiaxial", 9)

    def test_rejects_nl_with_hinge(self) -> None:
        integ, t = self._bits()
        with pytest.raises(ValueError, match="-nl is mutually exclusive"):
            LadrunoDispBeamColumn(
                pg="C", transf=t, integration=integ, nl=True,
                hinge=_FakeMat(name="h"))

    def test_rejects_biaxial_with_block_hinge(self) -> None:
        integ, t = self._bits()
        with pytest.raises(ValueError, match="mutually exclusive"):
            LadrunoDispBeamColumn(
                pg="C", transf=t, integration=integ,
                hinge=_FakeMat(name="h"), hinge_biaxial=_FakeNDMat(name="b"))

    def test_rejects_hinge_y_without_hinge(self) -> None:
        integ, t = self._bits()
        with pytest.raises(ValueError, match="hinge_y requires hinge"):
            LadrunoDispBeamColumn(
                pg="C", transf=t, integration=integ, hinge_y=_FakeMat(name="h"))

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_bad_numeric_lch(self, bad: float) -> None:
        integ, t = self._bits()
        with pytest.raises(ValueError, match="numeric lch must be finite"):
            LadrunoDispBeamColumn(pg="C", transf=t, integration=integ, lch=bad)

    def test_rejects_bad_lch_keyword(self) -> None:
        integ, t = self._bits()
        with pytest.raises(ValueError, match="lch must be"):
            LadrunoDispBeamColumn(pg="C", transf=t, integration=integ, lch="bogus")


# ===========================================================================
# LadrunoIMKBeam (Ladruno fork — concentrated-plasticity IMK beam)
# ===========================================================================

class TestLadrunoIMKBeam:
    def test_emit_2d(self) -> None:
        t = Linear(vecxz=(0, 0, 1))
        hz = _FakeMat(name="hz")
        spec = LadrunoIMKBeam(
            pg="B", transf=t, A=0.04, E=2e11, Iz=2e-4, mat_z=hz)
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 2, id(hz): 31}))
        set_element_nodes(rec, (10, 20))
        spec._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "LadrunoIMKBeam", 1, 10, 20, 0.04, 2e11, 2e-4, 2, "-matZ", 31,
        )

    def test_emit_3d_with_per_end_and_mass(self) -> None:
        t = Linear(vecxz=(0, 0, 1))
        hz = _FakeMat(name="hz")
        hy = _FakeMat(name="hy")
        spec = LadrunoIMKBeam(
            pg="B", transf=t, A=0.04, E=2e11, Iz=2e-4,
            G=8e10, Jx=1e-4, Iy=1.5e-4,
            ends="i", mat_z=hz, mat_y=hy, mass=5.0,
        )
        assert spec.dependencies() == (t, hz, hy)
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(t): 2, id(hz): 31, id(hy): 32}))
        set_element_nodes(rec, (10, 20))
        spec._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "LadrunoIMKBeam", 2, 10, 20,
            0.04, 2e11, 8e10, 1e-4, 1.5e-4, 2e-4, 2,
            "-hinge", "i", "-matZ", 31, "-matY", 32, "-mass", 5.0,
        )

    def test_rejects_partial_3d(self) -> None:
        t = Linear(vecxz=(0, 0, 1))
        with pytest.raises(ValueError, match="3-D variant requires G, Jx, Iy"):
            LadrunoIMKBeam(pg="B", transf=t, A=0.04, E=2e11, Iz=2e-4, G=8e10)

    def test_rejects_matY_in_2d(self) -> None:
        t = Linear(vecxz=(0, 0, 1))
        with pytest.raises(ValueError, match="weak-axis hinges .* are 3-D only"):
            LadrunoIMKBeam(
                pg="B", transf=t, A=0.04, E=2e11, Iz=2e-4,
                mat_y=_FakeMat(name="hy"))

    def test_rejects_bad_ends(self) -> None:
        t = Linear(vecxz=(0, 0, 1))
        with pytest.raises(ValueError, match="ends must be"):
            LadrunoIMKBeam(
                pg="B", transf=t, A=0.04, E=2e11, Iz=2e-4, ends="middle")

    @pytest.mark.parametrize("field,val", [("A", 0.0), ("E", -1.0), ("Iz", 0.0)])
    def test_rejects_non_positive_props(self, field: str, val: float) -> None:
        t = Linear(vecxz=(0, 0, 1))
        kwargs = {"pg": "B", "transf": t, "A": 0.04, "E": 2e11, "Iz": 2e-4}
        kwargs[field] = val
        with pytest.raises(ValueError, match=f"{field} must be > 0"):
            LadrunoIMKBeam(**kwargs)  # type: ignore[arg-type]

    def test_repr(self) -> None:
        t = Linear(vecxz=(0, 0, 1))
        spec = LadrunoIMKBeam(pg="B", transf=t, A=0.04, E=2e11, Iz=2e-4)
        assert "LadrunoIMKBeam" in repr(spec)
