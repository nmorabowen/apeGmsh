"""The two remaining SANISAND deck gates from the integrator guide.

* **Section 4** — ``NormDispIncr`` is unreachable on this material, so a
  deck that declares one gets a :class:`ManzariConvergenceTestWarning`.
  Keyed on the material alone: the stall belongs to the integrator, and it
  was measured on a ``ManzariDafalias`` leg running the *elastic* tangent,
  so gating it on ``tan_type`` would miss the case that actually bit us.
* **Section 3** — ``max_substeps`` makes the material REFUSE an increment
  it cannot integrate. That only helps if the element acts on the refusal;
  under one that discards the return code the analysis converges on a
  partially integrated stress, which is worse than the uncapped
  force-accept it replaces. So the cap is refused outright anywhere but
  :class:`LadrunoBrick` — an ALLOW-list, because the fork's finding is
  that only that element propagates on every path today.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh.opensees._internal.build import (
    BridgeError,
    ManzariConvergenceTestWarning,
    validate_manzari_convergence_test,
    validate_sanisand_substep_cap,
)
from apeGmsh.opensees.element.solid import LadrunoBrick, stdBrick
from apeGmsh.opensees.material.nd import (
    ElasticIsotropic,
    LadrunoSANISAND,
    ManzariDafalias,
    PlaneStrain,
    SAniSandMS,
)

_GORINI = {
    "G0": 264.32, "nu": 0.3129, "e_init": 0.6944, "Mc": 1.33090, "c": 0.71,
    "lambda_c": 0.027, "e0": 0.83, "ksi": 0.45, "P_atm": 101.0, "m": 0.005,
    "h0": 1.3, "Ch": 0.968, "nb": 3.5, "A0": 0.05, "nd": 5.75,
    "z_max": 12.5, "cz": 1100.0, "rho": 2.0,
}


def _sand(**kw):
    return LadrunoSANISAND(**_GORINI, **kw)


def _test(name: str, **kw):
    import apeGmsh.opensees.analysis.test as testmod

    kw.setdefault("tol", 1e-8)
    kw.setdefault("max_iter", 200)
    return getattr(testmod, name)(**kw)


def _silent(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ManzariConvergenceTestWarning)
        fn(*a, **kw)


# ── section 4: the convergence test ─────────────────────────────────────────

def _flat_test(materials, tests):
    validate_manzari_convergence_test(
        materials, staged=False, flat_tests=tests, stage_tests=[],
    )


def _staged_test(materials, stage_tests):
    validate_manzari_convergence_test(
        materials, staged=True, flat_tests=[], stage_tests=stage_tests,
    )


class TestConvergenceTestGate:
    @pytest.mark.parametrize(
        "name", ["NormDispIncr", "RelativeNormDispIncr"]
    )
    def test_displacement_increment_warns(self, name: str) -> None:
        with pytest.warns(
            ManzariConvergenceTestWarning, match="UNREACHABLE"
        ):
            _flat_test([_sand()], [_test(name)])

    @pytest.mark.parametrize("name", ["NormUnbalance", "EnergyIncr"])
    def test_force_and_energy_are_silent(self, name: str) -> None:
        """EnergyIncr was measured to converge on the same deck."""
        _silent(_flat_test, [_sand()], [_test(name)])

    def test_fires_on_the_elastic_tangent_too(self) -> None:
        """The stall is the integrator's, not the tangent's.

        The leg that actually failed in our suite was ManzariDafalias at
        its vanilla ``tan_type=0``, so a tan_type-keyed gate would have
        stayed quiet on exactly the deck that was red.
        """
        md = ManzariDafalias(**_GORINI)
        assert md.tan_type == 0
        with pytest.warns(ManzariConvergenceTestWarning, match="Manzari"):
            _flat_test([md], [_test("NormDispIncr")])

    def test_sanisandms_is_in_the_family(self) -> None:
        with pytest.warns(ManzariConvergenceTestWarning):
            _flat_test(
                [SAniSandMS(
                    G0=125.0, nu=0.05, e_init=0.80, Mc=1.25, c=0.712,
                    lambda_c=0.019, e0=0.934, ksi=0.7, P_atm=101.3, m=0.01,
                    h0=7.05, Ch=0.968, nb=1.1, A0=0.704, nd=3.5,
                    zeta=0.0005, mu0=260.0, beta=1.0, rho=1.6,
                )],
                [_test("NormDispIncr")],
            )

    def test_unrelated_material_is_silent(self) -> None:
        _silent(
            _flat_test, [ElasticIsotropic(E=1e4, nu=0.3)],
            [_test("NormDispIncr")],
        )

    def test_no_declared_test_is_silent(self) -> None:
        """A missing test belongs to the analysis-chain validation."""
        _silent(_flat_test, [_sand()], [])

    def test_last_declared_test_is_the_effective_one(self) -> None:
        _silent(
            _flat_test, [_sand()],
            [_test("NormDispIncr"), _test("NormUnbalance")],
        )

    def test_per_stage_warns_naming_the_stage(self) -> None:
        with pytest.warns(ManzariConvergenceTestWarning, match="consolidate"):
            _staged_test(
                [_sand()], [("'consolidate'", _test("NormDispIncr"))]
            )

    def test_per_stage_force_residual_is_silent(self) -> None:
        _silent(
            _staged_test, [_sand()], [("'push'", _test("NormUnbalance"))]
        )

    def test_undeclared_stage_test_is_silent(self) -> None:
        _silent(_staged_test, [_sand()], [("'push'", None)])


# ── section 3: the substep cap ──────────────────────────────────────────────

class TestSubstepCapGate:
    def test_uncapped_is_allowed_anywhere(self) -> None:
        """Default 0 must change nothing for any existing deck."""
        validate_sanisand_substep_cap(
            [stdBrick(pg="soil", material=_sand())]
        )

    def test_capped_on_ladruno_brick_is_allowed(self) -> None:
        validate_sanisand_substep_cap(
            [LadrunoBrick(pg="soil", material=_sand(max_substeps=80))]
        )

    @pytest.mark.parametrize("formulation", ["std", "bbar", "uri", "ssp"])
    def test_every_ladruno_brick_formulation_is_allowed(
        self, formulation: str
    ) -> None:
        from apeGmsh.opensees.material.nd import SanisandIntegrationWarning

        with warnings.catch_warnings():
            # formulation="ssp" carries its own unrelated warning about the
            # stabilization reference stress; this test is about the cap.
            warnings.simplefilter("ignore", SanisandIntegrationWarning)
            brick = LadrunoBrick(
                pg="soil", material=_sand(max_substeps=80),
                formulation=formulation,
            )
        validate_sanisand_substep_cap([brick])

    def test_capped_on_a_non_propagating_element_raises(self) -> None:
        with pytest.raises(BridgeError, match="propagate a material refusal"):
            validate_sanisand_substep_cap(
                [stdBrick(pg="soil", material=_sand(max_substeps=80))]
            )

    def test_the_error_names_the_element_and_its_group(self) -> None:
        with pytest.raises(BridgeError) as exc:
            validate_sanisand_substep_cap(
                [stdBrick(pg="clay", material=_sand(max_substeps=40))]
            )
        assert "'stdBrick'" in str(exc.value)
        assert "'clay'" in str(exc.value)
        assert "LadrunoBrick" in str(exc.value)

    def test_a_wrapped_material_is_still_found(self) -> None:
        """PlaneStrain / LogStrain put the real model a level down."""
        with pytest.raises(BridgeError, match="propagate a material refusal"):
            validate_sanisand_substep_cap([
                stdBrick(
                    pg="soil",
                    material=PlaneStrain(base=_sand(max_substeps=80)),
                )
            ])

    def test_an_unrelated_material_is_ignored(self) -> None:
        validate_sanisand_substep_cap(
            [stdBrick(pg="soil", material=ElasticIsotropic(E=1e4, nu=0.3))]
        )

    def test_only_the_offending_element_matters(self) -> None:
        """A safe element in the same deck does not excuse an unsafe one."""
        capped = _sand(max_substeps=80)
        with pytest.raises(BridgeError, match="stdBrick"):
            validate_sanisand_substep_cap([
                LadrunoBrick(pg="rock", material=capped),
                stdBrick(pg="soil", material=capped),
            ])


# ── the primitive's own validation ──────────────────────────────────────────

class TestSubstepCapField:
    def test_default_is_uncapped(self) -> None:
        assert _sand().max_substeps == 0

    def test_rejects_a_negative_cap(self) -> None:
        with pytest.raises(ValueError, match="max_substeps must be >= 0"):
            _sand(max_substeps=-1)

    @pytest.mark.parametrize("scheme", [2, 45])
    def test_warns_when_the_scheme_skips_modified_euler(
        self, scheme: int
    ) -> None:
        from apeGmsh.opensees.material.nd import SanisandIntegrationWarning

        with pytest.warns(SanisandIntegrationWarning, match="NO EFFECT"):
            _sand(max_substeps=80, int_scheme=scheme)

    @pytest.mark.parametrize("scheme", [0, 1])
    def test_silent_when_the_scheme_reaches_modified_euler(
        self, scheme: int
    ) -> None:
        from apeGmsh.opensees.material.nd import SanisandIntegrationWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", SanisandIntegrationWarning)
            _sand(max_substeps=80, int_scheme=scheme)
