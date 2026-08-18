"""A ``stress_zz`` gauss record is promoted onto the fork's 4-component
plane-strain stress response — but only where it can actually be recorded.

``stress_zz`` is a valid canonical component that routes onto the plain
``stresses`` token, which carries only the three in-plane components — so
requesting it was a silent no-op and the value read back was the ν-estimate
σ_zz = ν(σ_xx+σ_yy), exact only while the material is elastic.

The fork's ``stressesPlaneStrain`` returns ``[σxx, σyy, σxy, σzz]`` per Gauss
point, a strict superset of ``stresses``.  Promoting to it is safe ONLY when
every targeted element reports a real σ33: ``NDMaterial::getStressZZ()``
returns ``quiet_NaN`` by default, so an un-gated promotion writes an all-NaN
column — strictly worse than the estimate it replaces.

These tests cover the gate (element class · plane_type · material), the
record-level promotion at emit, the emit-time warning when the gate refuses,
and the invariant that a record which does not ask for ``stress_zz`` is
untouched.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import (
    STRESS_PLANE_STRAIN_RESPONSE,
    element_records_stress_zz,
)
from apeGmsh.opensees._recorder_translate import (
    StressZZNotRecordedWarning,
    element_record_response_tokens,
)
from apeGmsh.opensees.element.solid import (
    BezierTri6,
    FourNodeQuad,
    LadrunoLST,
    LadrunoUP,
    SixNodeTri,
    stdBrick,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.nd import (
    ElasticIsotropic,
    LadrunoJ2,
    LadrunoJ2Finite,
)

from tests.opensees.h5.test_h5_stages_reader import build_two_quad_fem


IN_PLANE = ("stress_xx", "stress_yy", "stress_xy")
WITH_ZZ = IN_PLANE + ("stress_zz",)


@pytest.fixture(scope="module")
def box_fem():
    """One structured hex — the 3-D control case for the ndm gate."""
    g = apeGmsh(model_name="szz_box", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.structured.set_transfinite("soil", n=2)
        g.mesh.generation.generate(dim=3)
        yield g.mesh.queries.get_fem_data()
    finally:
        g.end()

_ELASTIC = ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
_J2 = LadrunoJ2(K=1.333e8, G=8e7, sig0=1.2e5, Hiso=2.0e6)
_J2_FINITE = LadrunoJ2Finite(K=1.333e8, G=8e7, sig0=1.2e5, Hiso=2.0e6)


# ---------------------------------------------------------------------------
# The capability gate
# ---------------------------------------------------------------------------


class TestSigmaZZCapability:
    def test_plane_strain_quad_over_elastic_is_capable(self) -> None:
        assert element_records_stress_zz(
            FourNodeQuad(pg="Rock", thickness=1.0, material=_ELASTIC)
        )

    def test_ladruno_lst_over_ladruno_j2_is_capable(self) -> None:
        """The live plastic case: LadrunoJ2 answers only at DIM_PSTRAIN,
        which is exactly what ``plane_type="PlaneStrain"`` selects."""
        assert element_records_stress_zz(
            LadrunoLST(pg="Body", thickness=0.1, material=_J2)
        )

    def test_bezier_tri6_is_capable(self) -> None:
        assert element_records_stress_zz(
            BezierTri6(pg="Body", thickness=1.0, material=_ELASTIC)
        )

    def test_plane_stress_is_not_capable(self) -> None:
        """Under plane stress the material's plane-stress view has no
        ``getStressZZ`` override — the response would be NaN (and σ_zz is
        0 there by definition anyway)."""
        assert not element_records_stress_zz(
            FourNodeQuad(
                pg="Rock", thickness=1.0, material=_ELASTIC,
                plane_type="PlaneStress",
            )
        )

    def test_material_without_getstresszz_is_not_capable(self) -> None:
        """LadrunoJ2Finite ships no ``getStressZZ`` override — recording
        the 4-component response there writes NaN.  A capable element at
        plane strain is not enough."""
        assert not element_records_stress_zz(
            LadrunoLST(
                pg="Body", thickness=0.1, material=_J2_FINITE, geom="finite",
            )
        )

    def test_six_node_tri_has_no_plane_strain_response(self) -> None:
        """A plane element is not enough: SixNodeTri's ``setResponse``
        has no ``stressesPlaneStrain`` branch in the fork."""
        assert not element_records_stress_zz(
            SixNodeTri(pg="Body", thickness=1.0, material=_ELASTIC)
        )

    def test_ladruno_up_has_no_plane_strain_response(self) -> None:
        assert not element_records_stress_zz(
            LadrunoUP(
                pg="Body", material=_ELASTIC,
                Kf=2.2e9, poro=0.4, rhoF=1000.0, perm=(1e-5, 1e-5),
            )
        )

    def test_solid_element_is_not_capable(self) -> None:
        """A 3-D element records σ_zz through ``stresses`` already; it has
        no ``plane_type`` and must never be promoted."""
        assert not element_records_stress_zz(
            stdBrick(pg="Body", material=_ELASTIC)
        )


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


class TestResponseTokenPromotion:
    def test_capable_stress_zz_record_promotes(self) -> None:
        assert element_record_response_tokens(
            "gauss", WITH_ZZ, sigma_zz_capable=True,
        ) == (STRESS_PLANE_STRAIN_RESPONSE,)

    def test_incapable_stress_zz_record_keeps_stresses_and_warns(self) -> None:
        with pytest.warns(StressZZNotRecordedWarning, match="stress_zz"):
            tokens = element_record_response_tokens(
                "gauss", WITH_ZZ, record_name="body", sigma_zz_capable=False,
            )
        assert tokens == ("stresses",)

    def test_unknown_capability_keeps_stresses_silently(self) -> None:
        """``None`` = the caller has no element plan (ModelData's
        fem-eid-verbatim rendering).  Warning there would be noise about
        something the caller cannot know."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", StressZZNotRecordedWarning)
            assert element_record_response_tokens(
                "gauss", WITH_ZZ, sigma_zz_capable=None,
            ) == ("stresses",)

    def test_in_plane_only_record_never_promotes(self) -> None:
        assert element_record_response_tokens(
            "gauss", IN_PLANE, sigma_zz_capable=True,
        ) == ("stresses",)

    def test_strain_record_is_untouched(self) -> None:
        assert element_record_response_tokens(
            "gauss", ("strain_xx", "strain_zz"), sigma_zz_capable=True,
        ) == ("strains",)


# ---------------------------------------------------------------------------
# End-to-end emit through the bridge
# ---------------------------------------------------------------------------


def _bridge(components, *, material="elastic", plane_type="PlaneStrain"):
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    if material == "elastic":
        mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    else:
        mat = ops.nDMaterial.LadrunoJ2Finite(
            K=1.333e8, G=8e7, sig0=1.2e5, Hiso=2.0e6,
        )
    ops.element.FourNodeQuad(
        pg="Rock", thickness=1.0, material=mat, plane_type=plane_type,
    )
    ops.element.FourNodeQuad(
        pg="Fill", thickness=1.0, material=mat, plane_type=plane_type,
    )
    ops.fix(pg="Base", dofs=(1, 1))
    ops.recorder.declare(gauss=components, pg="Rock", file_root="out")
    return ops


def _recorder_args(ops) -> tuple:
    rec = RecordingEmitter()
    ops.build().emit(rec)
    calls = [c for c in rec.calls if c[0] == "recorder"]
    assert len(calls) == 1
    return calls[0][1]


def _deck(ops) -> str:
    emitter = TclEmitter()
    ops.build().emit(emitter)
    return "\n".join(emitter.lines())


class TestDeclareEmitPromotion:
    def test_capable_deck_emits_plane_strain_response(self) -> None:
        args = _recorder_args(_bridge(WITH_ZZ))
        assert args[0] == "Element"
        assert args[-1] == STRESS_PLANE_STRAIN_RESPONSE

    def test_incapable_material_keeps_stresses_and_warns(self) -> None:
        ops = _bridge(WITH_ZZ, material="j2finite")
        with pytest.warns(StressZZNotRecordedWarning):
            args = _recorder_args(ops)
        assert args[-1] == "stresses"

    def test_mixed_targets_do_not_promote(self) -> None:
        """The gate is "every targeted element", not "any": one incapable
        element in the record would turn its Gauss points into a NaN
        column that the rest of the record cannot compensate for."""
        ops = apeSees(build_two_quad_fem(), default_orientation=None)
        ops.model(ndm=2, ndf=2)
        good = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
        bad = ops.nDMaterial.LadrunoJ2Finite(
            K=1.333e8, G=8e7, sig0=1.2e5, Hiso=2.0e6,
        )
        ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=good)
        ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=bad)
        ops.fix(pg="Base", dofs=(1, 1))
        ops.recorder.declare(
            gauss=WITH_ZZ, pg=("Rock", "Fill"), file_root="out",
        )
        with pytest.warns(StressZZNotRecordedWarning):
            args = _recorder_args(ops)
        assert args[-1] == "stresses"

    def test_plane_stress_deck_keeps_stresses(self) -> None:
        ops = _bridge(WITH_ZZ, plane_type="PlaneStress")
        with pytest.warns(StressZZNotRecordedWarning):
            args = _recorder_args(ops)
        assert args[-1] == "stresses"

    def test_record_without_stress_zz_is_byte_identical(self) -> None:
        """A deck that does not ask for ``stress_zz`` must be untouched by
        the promotion — same token, no warning, no new deck text.

        The sharper half of the guard is
        :meth:`test_promotion_changes_exactly_one_deck_line` below: the
        promotion may perturb the recorder line and nothing else.
        """
        ops = _bridge(IN_PLANE)
        with warnings.catch_warnings():
            warnings.simplefilter("error", StressZZNotRecordedWarning)
            deck = _deck(ops)
        assert STRESS_PLANE_STRAIN_RESPONSE not in deck
        recorder_lines = [
            ln for ln in deck.splitlines() if ln.startswith("recorder ")
        ]
        assert len(recorder_lines) == 1
        assert recorder_lines[0].endswith(" stresses")

    def test_three_dimensional_deck_is_left_alone_and_silent(
        self, box_fem,
    ) -> None:
        """A 3-D ``stresses`` response already carries σ_zz. Promoting is
        meaningless there and warning about it would be pure noise."""
        ops = apeSees(box_fem)
        ops.model(ndm=3, ndf=3)
        mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
        ops.element.stdBrick(pg="soil", material=mat)
        ops.recorder.declare(gauss=WITH_ZZ, pg="soil", file_root="out")
        with warnings.catch_warnings():
            warnings.simplefilter("error", StressZZNotRecordedWarning)
            args = _recorder_args(ops)
        assert args[-1] == "stresses"

    def test_promotion_changes_exactly_one_deck_line(self) -> None:
        """Record-level: promoting ``stress_zz`` swaps the response token
        on its own recorder line and leaves the whole rest of the deck
        (model, materials, elements, fixes) alone."""
        without = _deck(_bridge(IN_PLANE)).splitlines()
        with_zz = _deck(_bridge(WITH_ZZ)).splitlines()
        assert len(without) == len(with_zz)
        differing = [
            (a, b) for a, b in zip(without, with_zz) if a != b
        ]
        assert len(differing) == 1
        before, after = differing[0]
        assert before.endswith(" stresses")
        assert after.endswith(f" {STRESS_PLANE_STRAIN_RESPONSE}")
        assert before[: -len("stresses")] == after[
            : -len(STRESS_PLANE_STRAIN_RESPONSE)
        ]
