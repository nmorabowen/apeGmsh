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
    Primitive,
    Section,
    UniaxialMaterial,
)
from apeGmsh.opensees.element.beam_column import (
    ElasticTimoshenkoBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
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

class TestForceBeamColumnConstruction:
    def test_construct_minimum(self) -> None:
        sec = _FakeSection(name="col_sec")
        t = Corotational(vecxz=(0.0, 0.0, 1.0))
        e = forceBeamColumn(
            pg="Cols", section=sec, transf=t, n_ip=5,
        )
        assert e.section is sec
        assert e.transf is t
        assert e.n_ip == 5
        assert e.mass is None
        assert e.max_iter is None
        assert e.tol is None

    def test_dependencies(self) -> None:
        sec = _FakeSection(name="col_sec")
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = forceBeamColumn(pg="Cols", section=sec, transf=t, n_ip=5)
        assert e.dependencies() == (sec, t)


class TestForceBeamColumnValidation:
    def _sec(self) -> _FakeSection:
        return _FakeSection(name="x")

    def _t(self) -> Linear:
        return Linear(vecxz=(0.0, 0.0, 1.0))

    def test_pg_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="pg must be a non-empty"):
            forceBeamColumn(
                pg="", section=self._sec(), transf=self._t(), n_ip=3,
            )

    @pytest.mark.parametrize("n", [0, -1])
    def test_n_ip_must_be_positive(self, n: int) -> None:
        with pytest.raises(ValueError, match="n_ip must be >= 1"):
            forceBeamColumn(
                pg="Cols", section=self._sec(), transf=self._t(), n_ip=n,
            )

    def test_mass_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="mass must be >= 0"):
            forceBeamColumn(
                pg="Cols", section=self._sec(), transf=self._t(),
                n_ip=3, mass=-0.1,
            )

    def test_iter_partial_max_iter_only_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="max_iter and tol must be supplied together"
        ):
            forceBeamColumn(
                pg="Cols", section=self._sec(), transf=self._t(),
                n_ip=3, max_iter=10,
            )

    def test_iter_partial_tol_only_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="max_iter and tol must be supplied together"
        ):
            forceBeamColumn(
                pg="Cols", section=self._sec(), transf=self._t(),
                n_ip=3, tol=1e-8,
            )

    def test_iter_negative_max_iter_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be > 0"):
            forceBeamColumn(
                pg="Cols", section=self._sec(), transf=self._t(),
                n_ip=3, max_iter=0, tol=1e-8,
            )

    def test_iter_negative_tol_rejected(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            forceBeamColumn(
                pg="Cols", section=self._sec(), transf=self._t(),
                n_ip=3, max_iter=10, tol=0.0,
            )


class TestForceBeamColumnEmit:
    def _build(
        self,
    ) -> tuple[forceBeamColumn, _FakeSection, Corotational]:
        sec = _FakeSection(name="col_sec")
        t = Corotational(vecxz=(0.0, 0.0, 1.0))
        spec = forceBeamColumn(pg="Cols", section=sec, transf=t, n_ip=5)
        return spec, sec, t

    def test_emit_records_correct_call(self) -> None:
        spec, sec, t = self._build()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(sec): 1, id(t): 2}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        assert rec.calls == [
            (
                "element",
                (
                    "forceBeamColumn", 42, 10, 20, 2,
                    "-section", 1, 5,
                ),
                {},
            )
        ]

    def test_emit_with_mass_flag(self) -> None:
        spec_basic, sec, t = self._build()
        # New spec with mass set.
        spec = forceBeamColumn(
            pg=spec_basic.pg, section=sec, transf=t, n_ip=spec_basic.n_ip,
            mass=12.5,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(sec): 1, id(t): 2}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        assert rec.calls[0][1] == (
            "forceBeamColumn", 42, 10, 20, 2,
            "-section", 1, 5,
            "-mass", 12.5,
        )

    def test_emit_with_iter_flag(self) -> None:
        spec_basic, sec, t = self._build()
        spec = forceBeamColumn(
            pg=spec_basic.pg, section=sec, transf=t, n_ip=spec_basic.n_ip,
            max_iter=20, tol=1e-9,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(sec): 1, id(t): 2}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        assert rec.calls[0][1] == (
            "forceBeamColumn", 42, 10, 20, 2,
            "-section", 1, 5,
            "-iter", 20, 1e-9,
        )

    def test_emit_with_mass_and_iter(self) -> None:
        spec_basic, sec, t = self._build()
        spec = forceBeamColumn(
            pg=spec_basic.pg, section=sec, transf=t, n_ip=spec_basic.n_ip,
            mass=2.5, max_iter=20, tol=1e-9,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(sec): 1, id(t): 2}))
        set_element_nodes(rec, (10, 20))

        spec._emit(rec, tag=42)

        # Order: -mass first, then -iter.
        assert rec.calls[0][1] == (
            "forceBeamColumn", 42, 10, 20, 2,
            "-section", 1, 5,
            "-mass", 2.5,
            "-iter", 20, 1e-9,
        )

    def test_emit_without_node_context_raises(self) -> None:
        spec, sec, t = self._build()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(sec): 1, id(t): 2}))
        with pytest.raises(RuntimeError, match="element"):
            spec._emit(rec, tag=1)

    def test_emit_without_resolver_raises(self) -> None:
        spec, _sec, _t = self._build()
        rec = RecordingEmitter()
        set_element_nodes(rec, (10, 20))
        with pytest.raises(RuntimeError, match="resolver"):
            spec._emit(rec, tag=1)

    def test_emit_with_one_node_rejected(self) -> None:
        spec, sec, t = self._build()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(sec): 1, id(t): 2}))
        set_element_nodes(rec, (10,))
        with pytest.raises(ValueError, match="expected 2 node tags"):
            spec._emit(rec, tag=1)


# ===========================================================================
# dispBeamColumn
# ===========================================================================

class TestDispBeamColumnConstruction:
    def test_construct_minimum(self) -> None:
        sec = _FakeSection(name="x")
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = dispBeamColumn(pg="Cols", section=sec, transf=t, n_ip=4)
        assert e.section is sec
        assert e.transf is t
        assert e.n_ip == 4
        assert e.c_mass is False
        assert e.mass is None

    def test_dependencies(self) -> None:
        sec = _FakeSection(name="x")
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = dispBeamColumn(pg="Cols", section=sec, transf=t, n_ip=4)
        assert e.dependencies() == (sec, t)


class TestDispBeamColumnValidation:
    def _sec(self) -> _FakeSection:
        return _FakeSection(name="x")

    def _t(self) -> Linear:
        return Linear(vecxz=(0.0, 0.0, 1.0))

    def test_pg_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="pg must be a non-empty"):
            dispBeamColumn(
                pg="", section=self._sec(), transf=self._t(), n_ip=3,
            )

    @pytest.mark.parametrize("n", [0, -2])
    def test_n_ip_must_be_positive(self, n: int) -> None:
        with pytest.raises(ValueError, match="n_ip must be >= 1"):
            dispBeamColumn(
                pg="Cols", section=self._sec(), transf=self._t(), n_ip=n,
            )

    def test_mass_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="mass must be >= 0"):
            dispBeamColumn(
                pg="Cols", section=self._sec(), transf=self._t(),
                n_ip=3, mass=-1.0,
            )


class TestDispBeamColumnEmit:
    def _build(self) -> tuple[dispBeamColumn, _FakeSection, Linear]:
        sec = _FakeSection(name="x")
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        spec = dispBeamColumn(pg="Cols", section=sec, transf=t, n_ip=4)
        return spec, sec, t

    def test_emit_records_correct_call(self) -> None:
        spec, sec, t = self._build()
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(sec): 11, id(t): 12}))
        set_element_nodes(rec, (3, 4))

        spec._emit(rec, tag=99)

        # Tcl: tag iNode jNode numIntgrPts secTag transfTag
        assert rec.calls == [
            (
                "element",
                ("dispBeamColumn", 99, 3, 4, 4, 11, 12),
                {},
            )
        ]

    def test_emit_with_mass_and_c_mass(self) -> None:
        spec_basic, sec, t = self._build()
        spec = dispBeamColumn(
            pg=spec_basic.pg, section=sec, transf=t, n_ip=spec_basic.n_ip,
            mass=3.0, c_mass=True,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver_from({id(sec): 11, id(t): 12}))
        set_element_nodes(rec, (3, 4))

        spec._emit(rec, tag=1)

        assert rec.calls[0][1] == (
            "dispBeamColumn", 1, 3, 4, 4, 11, 12,
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
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = forceBeamColumn(pg="Cols", section=sec, transf=t, n_ip=5)
        assert "forceBeamColumn" in repr(e)

    def test_dispBeamColumn_repr(self) -> None:
        sec = _FakeSection(name="x")
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = dispBeamColumn(pg="Cols", section=sec, transf=t, n_ip=4)
        assert "dispBeamColumn" in repr(e)

    def test_ElasticTimoshenkoBeam_repr(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        e = ElasticTimoshenkoBeam(
            pg="Cols", transf=t,
            E=2e11, G=8e10, A=0.04, Iz=2e-4, Avy=0.03,
        )
        assert "ElasticTimoshenkoBeam" in repr(e)
