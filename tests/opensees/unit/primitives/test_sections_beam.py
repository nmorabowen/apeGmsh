"""Unit tests for :class:`apeGmsh.opensees.section.beam.ElasticSection`.

Covers:

* construction / defaults
* validation (E, A, Iz positive; partial 3-D specs rejected; alphaY
  in 2-D requires G; alphaZ in 2-D rejected)
* ``_emit`` records the right ``section`` call for both 2-D and 3-D
* ``dependencies()`` empty
* ``repr`` includes the type name
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.section.beam import ElasticSection


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------

class TestElasticSectionConstruction:
    def test_construct_2d_minimum(self) -> None:
        s = ElasticSection(E=2e11, A=0.01, Iz=1e-4)
        assert s.E == 2e11
        assert s.A == 0.01
        assert s.Iz == 1e-4
        assert s.Iy is None
        assert s.G is None
        assert s.J is None
        assert s.alphaY is None
        assert s.alphaZ is None

    def test_construct_3d_minimum(self) -> None:
        s = ElasticSection(
            E=2e11, A=0.01, Iz=1e-4, Iy=2e-4, G=8e10, J=3e-4,
        )
        assert s.Iy == 2e-4
        assert s.G == 8e10
        assert s.J == 3e-4

    def test_2d_with_g_and_alpha(self) -> None:
        s = ElasticSection(
            E=2e11, A=0.01, Iz=1e-4, G=8e10, alphaY=0.9,
        )
        assert s.G == 8e10
        assert s.alphaY == 0.9
        assert s.Iy is None  # still 2-D


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestElasticSectionValidation:
    @pytest.mark.parametrize("E", [0.0, -1.0, -1e9])
    def test_E_must_be_positive(self, E: float) -> None:
        with pytest.raises(ValueError, match="E must be > 0"):
            ElasticSection(E=E, A=0.01, Iz=1e-4)

    @pytest.mark.parametrize("A", [0.0, -1e-3])
    def test_A_must_be_positive(self, A: float) -> None:
        with pytest.raises(ValueError, match="A must be > 0"):
            ElasticSection(E=2e11, A=A, Iz=1e-4)

    @pytest.mark.parametrize("Iz", [0.0, -1e-5])
    def test_Iz_must_be_positive(self, Iz: float) -> None:
        with pytest.raises(ValueError, match="Iz must be > 0"):
            ElasticSection(E=2e11, A=0.01, Iz=Iz)

    def test_partial_3d_iy_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="3-D variant requires"):
            ElasticSection(E=2e11, A=0.01, Iz=1e-4, Iy=2e-4)

    def test_partial_3d_iy_g_no_j_rejected(self) -> None:
        with pytest.raises(ValueError, match="3-D variant requires"):
            ElasticSection(E=2e11, A=0.01, Iz=1e-4, Iy=2e-4, G=8e10)

    def test_alphaZ_in_2d_rejected(self) -> None:
        # alphaZ alone (without Iy/G/J) triggers 3-D; missing Iy
        # surfaces the error.
        with pytest.raises(ValueError, match="3-D variant requires"):
            ElasticSection(E=2e11, A=0.01, Iz=1e-4, alphaZ=0.9)

    def test_2d_alphaY_without_G_rejected(self) -> None:
        with pytest.raises(ValueError, match="alphaY requires G"):
            ElasticSection(E=2e11, A=0.01, Iz=1e-4, alphaY=0.9)

    def test_negative_Iy_rejected(self) -> None:
        with pytest.raises(ValueError, match="Iy must be > 0"):
            ElasticSection(
                E=2e11, A=0.01, Iz=1e-4, Iy=-1e-4, G=8e10, J=3e-4,
            )

    def test_negative_G_rejected(self) -> None:
        with pytest.raises(ValueError, match="G must be > 0"):
            ElasticSection(
                E=2e11, A=0.01, Iz=1e-4, Iy=2e-4, G=-1e10, J=3e-4,
            )


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

class TestElasticSectionEmit:
    def test_emit_2d_minimum(self) -> None:
        s = ElasticSection(E=2e11, A=0.01, Iz=1e-4)
        e = RecordingEmitter()
        s._emit(e, tag=7)
        assert e.calls == [
            ("section", ("Elastic", 7, 2e11, 0.01, 1e-4), {})
        ]

    def test_emit_2d_with_g(self) -> None:
        s = ElasticSection(E=2e11, A=0.01, Iz=1e-4, G=8e10)
        e = RecordingEmitter()
        s._emit(e, tag=2)
        assert e.calls == [
            ("section", ("Elastic", 2, 2e11, 0.01, 1e-4, 8e10), {})
        ]

    def test_emit_2d_with_g_and_alphaY(self) -> None:
        s = ElasticSection(
            E=2e11, A=0.01, Iz=1e-4, G=8e10, alphaY=0.9,
        )
        e = RecordingEmitter()
        s._emit(e, tag=3)
        assert e.calls == [
            ("section", ("Elastic", 3, 2e11, 0.01, 1e-4, 8e10, 0.9), {})
        ]

    def test_emit_3d_minimum(self) -> None:
        s = ElasticSection(
            E=2e11, A=0.01, Iz=1e-4, Iy=2e-4, G=8e10, J=3e-4,
        )
        e = RecordingEmitter()
        s._emit(e, tag=11)
        assert e.calls == [
            (
                "section",
                ("Elastic", 11, 2e11, 0.01, 1e-4, 2e-4, 8e10, 3e-4),
                {},
            )
        ]

    def test_emit_3d_with_alpha_pair(self) -> None:
        s = ElasticSection(
            E=2e11, A=0.01, Iz=1e-4,
            Iy=2e-4, G=8e10, J=3e-4,
            alphaY=0.9, alphaZ=0.85,
        )
        e = RecordingEmitter()
        s._emit(e, tag=4)
        assert e.calls == [
            (
                "section",
                ("Elastic", 4, 2e11, 0.01, 1e-4, 2e-4, 8e10, 3e-4,
                 0.9, 0.85),
                {},
            )
        ]

    def test_emit_3d_with_only_alphaY_pads_alphaZ_to_one(self) -> None:
        s = ElasticSection(
            E=2e11, A=0.01, Iz=1e-4,
            Iy=2e-4, G=8e10, J=3e-4,
            alphaY=0.9,
        )
        e = RecordingEmitter()
        s._emit(e, tag=5)
        # OpenSees expects the alphaY/alphaZ pair together — missing
        # alphaZ defaults to 1.0.
        assert e.calls[0][1][-2:] == (0.9, 1.0)


# ---------------------------------------------------------------------------
# Dependencies / repr
# ---------------------------------------------------------------------------

class TestElasticSectionMisc:
    def test_dependencies_is_empty(self) -> None:
        s = ElasticSection(E=2e11, A=0.01, Iz=1e-4)
        assert s.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        s = ElasticSection(E=2e11, A=0.01, Iz=1e-4)
        assert "ElasticSection" in repr(s)
