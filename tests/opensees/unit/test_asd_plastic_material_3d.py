"""Unit tests for the Phase SSI-1.5 ASDPlasticMaterial3D wrapper family.

Covers:

1. The generic :class:`ASDPlasticMaterial3D` typed dataclass — frozen,
   validates yf/pf/el/iv shape, emits the right Tcl card.
2. The :func:`MohrCoulombSoil` convenience constructor — builds the
   correct generic-class shape with MohrCoulomb_YF/PF + LinearIsotropic3D_EL
   + BackStress(NullHardeningTensorFunction):, with input validation.
3. The :class:`PlaneStrain` wrapper — emits ``nDMaterial PlaneStrain
   $tag $base_tag``.
4. The bridge namespace methods (``ops.nDMaterial.ASDPlasticMaterial3D``,
   ``ops.nDMaterial.MohrCoulombSoil``, ``ops.nDMaterial.PlaneStrain``)
   construct + register + emit correctly.
5. ADR 0105 D1 — the per-combination parameter schema
   (:func:`asdp_parameter_schema`) resolves every one of the fork's 46
   registered combinations (pinned fixture, captured from the fork's
   ``list`` verb on build ``3622d6214``), the fail-loud ``__post_init__``
   (foreign name / missing name / unknown component), and
   :func:`MohrCoulombSoil` emitting exactly its schema.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.nd import (
    _ASDP_OPTIONAL_PARAMS,
    _ASDP_PARAMS_BY_COMPONENT,
    _ASDP_RETURN_TO_YIELD_SURFACE,
    _ASDP_TANGENT_TYPES,
    ASDPlasticIntegrationWarning,
    ASDPlasticMaterial3D,
    HoekBrownRock,
    MohrCoulombSoil,
    MohrCoulombTensionCutoffSoil,
    PlaneStrain,
    asdp_parameter_schema,
)

from tests.opensees.fixtures.fem_stub import make_two_node_beam

_MC_IV = "BackStress(NullHardeningTensorFunction):"

#: The full MohrCoulomb_YF / _PF + LinearIsotropic3D_EL schema — what the
#: fork's ADR-94 parser requires (minus the two optional names).
_MC_SCHEMA = frozenset({
    "YoungsModulus", "PoissonsRatio", "MC_phi", "MC_c", "MC_ds", "MC_psi",
    "MassDensity", "InitialP0",
})
_MC_PARAMS = (
    ("YoungsModulus", 1e6), ("PoissonsRatio", 0.25), ("MC_phi", 30.0),
    ("MC_c", 100.0), ("MC_ds", 0.0), ("MC_psi", 30.0),
)

#: The pre-ADR-0105 ``MohrCoulombSoil`` parameter block, verbatim (the
#: 21-name superset the helper used to zero-fill).  Kept as a fixture,
#: not regenerated: it is what the fork's ADR-94 parser refuses.
_PRE_ADR94_SUPERSET = (
    ("AF_cr", 0.0), ("AF_ha", 0.0), ("DP_eta", 0.0), ("DP_etabar", 0.0),
    ("DP_xi_c", 0.0), ("Dilatancy", 0.0), ("DuncanChang_MaxSigma3", 0.0),
    ("DuncanChang_n", 0.0), ("InitialP0", 0.0), ("MC_c", 1014.0),
    ("MC_ds", 1e-5), ("MC_phi", 45.95), ("MC_psi", 11.49),
    ("MassDensity", 4.5), ("PoissonsRatio", 0.18),
    ("ReferencePressure", 0.0), ("ReferenceYoungsModulus", 0.0),
    ("ScalarLinearHardeningParameter", 0.0), ("TC_min_stress", 0.0),
    ("TensorLinearHardeningParameter", 0.0), ("YoungsModulus", 4080000.0),
)


def _mc(**kw) -> ASDPlasticMaterial3D:
    kw.setdefault("model_parameters", _MC_PARAMS)
    return ASDPlasticMaterial3D(
        yf="MohrCoulomb_YF", pf="MohrCoulomb_PF",
        el="LinearIsotropic3D_EL", iv=_MC_IV, **kw,
    )


# ---------------------------------------------------------------------------
# 1. Generic ASDPlasticMaterial3D dataclass
# ---------------------------------------------------------------------------


def test_asd_plastic_material_3d_validates_type_strings() -> None:
    with pytest.raises(ValueError, match="yf= must be non-empty"):
        ASDPlasticMaterial3D(
            yf="", pf="MohrCoulomb_PF", el="LinearIsotropic3D_EL",
            iv="BackStress(NullHardeningTensorFunction):",
        )
    with pytest.raises(ValueError, match="pf= must be non-empty"):
        ASDPlasticMaterial3D(
            yf="MohrCoulomb_YF", pf="", el="LinearIsotropic3D_EL",
            iv="BackStress(NullHardeningTensorFunction):",
        )
    with pytest.raises(ValueError, match="iv= must be non-empty"):
        ASDPlasticMaterial3D(
            yf="MohrCoulomb_YF", pf="MohrCoulomb_PF",
            el="LinearIsotropic3D_EL", iv="",
        )


def test_asd_plastic_material_3d_emit_shape() -> None:
    mat = _mc(
        internal_variables=(("BackStress", (0.0,) * 6),),
        model_parameters=_MC_PARAMS,
        integration_options=(
            ("integration_method", "Backward_Euler"),
            ("n_max_iterations", 50),
            ("f_absolute_tol", 1e-6),
        ),
    )
    e = TclEmitter()
    mat._emit(e, tag=5)
    line = e.lines()[-1]
    # Header tokens.
    assert "nDMaterial ASDPlasticMaterial3D 5" in line
    assert "MohrCoulomb_YF MohrCoulomb_PF LinearIsotropic3D_EL" in line
    assert "BackStress(NullHardeningTensorFunction):" in line
    # Blocks.
    assert "Begin_Internal_Variables BackStress 0.0 0.0 0.0 0.0 0.0 0.0 End_Internal_Variables" in line
    assert (
        "Begin_Model_Parameters YoungsModulus 1000000.0 PoissonsRatio 0.25 "
        "MC_phi 30.0 MC_c 100.0 MC_ds 0.0 MC_psi 30.0 End_Model_Parameters"
    ) in line
    # Integration options: float, int, and string enum render correctly.
    assert "Begin_Integration_Options" in line
    assert "integration_method Backward_Euler" in line
    assert "n_max_iterations 50" in line  # int, not 50.0
    assert "f_absolute_tol 1e-06" in line  # float repr


def test_asd_plastic_material_3d_no_dependencies() -> None:
    assert _mc().dependencies() == ()


# ---------------------------------------------------------------------------
# 1b. ADR 0105 D1 — the per-combination parameter schema
# ---------------------------------------------------------------------------

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "asdp_registered_combinations_3622d6214.txt"
)


def _registered_combinations() -> list[tuple[str, str, str, str]]:
    """The 46 ``(yf, pf, el, iv)`` tuples the fork's ``list`` verb printed.

    Captured once against fork build ``3622d6214`` (ADR-94 closeout,
    ASDP-equivalent to ``bbf657d49``) and pinned: the verb prints the four
    type strings only, never the parameter names, which is why the table
    under test exists at all (ADR 0105 D8 asks the fork for the names).
    """
    lines = _FIXTURE.read_text(encoding="utf-8").splitlines()
    out: list[tuple[str, str, str, str]] = []
    for i in range(0, len(lines), 4):
        block = dict(line.split(" = ", 1) for line in lines[i:i + 4])
        out.append((block["YF"], block["PF"], block["EL"], block["IV"]))
    return out


def test_fixture_carries_the_forks_46_registered_combinations() -> None:
    combos = _registered_combinations()
    assert len(combos) == 46
    assert len(set(combos)) == 46
    by_yf: dict[str, int] = {}
    for yf, _pf, _el, _iv in combos:
        by_yf[yf] = by_yf.get(yf, 0) + 1
    # The fork's _adr94_inventory.md §(b) grouping, verbatim.
    assert by_yf == {
        "VonMises_YF": 14, "DruckerPrager_YF": 14, "MohrCoulomb_YF": 7,
        "HoekBrown_YF": 7, "StiffSoilCap_YF": 2,
        "MohrCoulombTensionCutoff_YF": 1, "StiffSoilShear_YF": 1,
    }


@pytest.mark.parametrize(
    ("yf", "pf", "el", "iv"), _registered_combinations(),
    ids=lambda v: v if isinstance(v, str) and "_" in v and "(" not in v else None,
)
def test_schema_resolves_every_registered_combination(
    yf: str, pf: str, el: str, iv: str,
) -> None:
    """Every registered combination resolves without error.

    The StiffSoil family (3 of the 46) is outside the table by design and
    resolves to ``None`` — the escape hatch; every other one resolves to
    a set that carries the two optional names and the elastic pair.
    """
    schema = asdp_parameter_schema(yf, pf, el, iv)
    if el == "StiffSoil_EL":
        assert schema is None
        assert yf.startswith("StiffSoil")
        return
    assert schema is not None, (yf, pf, el, iv)
    assert _ASDP_OPTIONAL_PARAMS <= schema
    assert {"YoungsModulus", "PoissonsRatio"} <= schema
    # Every name is one the table knows — no component can smuggle in a
    # name that is not attributable to it.
    known = set().union(*_ASDP_PARAMS_BY_COMPONENT.values())
    assert schema - _ASDP_OPTIONAL_PARAMS <= known


def test_schema_is_the_union_of_the_components() -> None:
    # MohrCoulomb_YF (3) ∪ MohrCoulomb_PF (+MC_psi) ∪ LinearIsotropic3D_EL
    # ∪ NullHardeningTensorFunction (none) ∪ the two optional names.
    assert asdp_parameter_schema(
        "MohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL", _MC_IV,
    ) == _MC_SCHEMA
    # A hardening policy contributes its parameters ONCE however many IVs
    # carry it; a repeated IV name with different policies contributes
    # each policy's set.
    iv = (
        "BackStress(TensorLinearHardeningFunction):"
        "YieldStress(ScalarLinearHardeningFunction):"
        "BackStress(ArmstrongFrederickHardeningFunction):"
        "DP_cohesion(ScalarLinearHardeningFunction):"
    )
    assert asdp_parameter_schema(
        "VonMises_YF", "DruckerPrager_PF", "LinearIsotropic3D_EL", iv,
    ) == frozenset({
        "YoungsModulus", "PoissonsRatio", "DP_etabar",
        "TensorLinearHardeningParameter", "ScalarLinearHardeningParameter",
        "AF_ha", "AF_cr", "MassDensity", "InitialP0",
    })


def test_schema_is_none_for_an_unknown_component_or_malformed_iv() -> None:
    assert asdp_parameter_schema(
        "RoundedMohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL",
        _MC_IV,
    ) is None
    assert asdp_parameter_schema(
        "MohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL",
        "BackStress(SomeFuturePolicy):",
    ) is None
    assert asdp_parameter_schema(
        "MohrCoulomb_YF", "MohrCoulomb_PF", "LinearIsotropic3D_EL",
        "BackStress:",
    ) is None


def test_pre_adr94_superset_deck_is_refused_naming_the_first_foreign_name() -> None:
    """The old helper's 21-name block: refused at construction, naming the
    first foreign name (``AF_cr``, in the block's own order) and the schema."""
    with pytest.raises(ValueError) as exc:
        _mc(model_parameters=_PRE_ADR94_SUPERSET)
    msg = str(exc.value)
    assert "'AF_cr' is not a parameter of" in msg
    assert "DuncanChang_n" in msg          # every foreign name is listed
    assert "TC_min_stress" in msg
    assert "Schema for this combination" in msg
    assert "MC_psi" in msg


def test_missing_required_parameter_is_refused_listing_the_names() -> None:
    with pytest.raises(ValueError) as exc:
        _mc(model_parameters=(("YoungsModulus", 1e6), ("PoissonsRatio", 0.25)))
    msg = str(exc.value)
    assert "4 required model parameter(s) missing" in msg
    assert "MC_c, MC_ds, MC_phi, MC_psi" in msg
    # The optional pair is never reported missing.
    assert "MassDensity" not in msg.split("missing for")[1].split(".")[0]
    assert "InitialP0" not in msg.split("missing for")[1].split(".")[0]


def test_optional_parameters_may_be_omitted_or_given() -> None:
    _mc()                                              # neither
    _mc(model_parameters=_MC_PARAMS + (("MassDensity", 2.0),))
    _mc(model_parameters=_MC_PARAMS + (("InitialP0", -50.0),))


def test_unknown_component_is_accepted_unchanged() -> None:
    """The escape hatch: a combination the table does not cover is not
    validated here — the fork validates.  Foreign AND missing alike."""
    mat = ASDPlasticMaterial3D(
        yf="StiffSoilCap_YF", pf="StiffSoilCap_PF", el="StiffSoil_EL",
        iv="CapPressure(StiffSoilCapHardening):",
        model_parameters=(("WhateverTheForkAccepts", 1.0),),
    )
    assert dict(mat.model_parameters) == {"WhateverTheForkAccepts": 1.0}


# ---------------------------------------------------------------------------
# 2. MohrCoulombSoil convenience constructor
# ---------------------------------------------------------------------------


def test_mohr_coulomb_soil_builds_correct_generic_shape() -> None:
    mat = MohrCoulombSoil(
        c=1014.0, phi=45.95, psi=11.49,
        E=4080000.0, nu=0.18, rho=4.5,
    )
    assert isinstance(mat, ASDPlasticMaterial3D)
    assert mat.yf == "MohrCoulomb_YF"
    assert mat.pf == "MohrCoulomb_PF"
    assert mat.el == "LinearIsotropic3D_EL"
    assert mat.iv == "BackStress(NullHardeningTensorFunction):"

    iv_dict = dict(mat.internal_variables)
    assert iv_dict["BackStress"] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # DP_cohesion / YieldStress are NOT in the IV list — the parser
    # silently drops them for MohrCoulomb_YF; emitting them is noise.
    assert "DP_cohesion" not in iv_dict
    assert "YieldStress" not in iv_dict

    mp_dict = dict(mat.model_parameters)
    assert mp_dict["MC_c"] == 1014.0
    assert mp_dict["MC_phi"] == 45.95
    assert mp_dict["MC_psi"] == 11.49
    assert mp_dict["YoungsModulus"] == 4080000.0
    assert mp_dict["PoissonsRatio"] == 0.18
    assert mp_dict["MassDensity"] == 4.5
    # ADR 0105 D1: EXACTLY the schema — the 21-name superset is gone.
    assert set(mp_dict) == _MC_SCHEMA
    assert len(mat.model_parameters) == len(_MC_SCHEMA)

    io_dict = dict(mat.integration_options)
    assert io_dict["integration_method"] == "Backward_Euler"
    # ADR 0105 D2 / D3 defaults: strict on, relative tol off, Continuum.
    assert io_dict["tangent_type"] == "Continuum"
    assert io_dict["strict_convergence"] is True
    assert io_dict["f_relative_tol"] == 0.0
    assert io_dict["return_to_yield_surface"] == "Disabled"


def test_mohr_coulomb_soil_validates_inputs() -> None:
    with pytest.raises(ValueError, match="c must be >= 0"):
        MohrCoulombSoil(c=-1, phi=30, psi=0, E=1e6, nu=0.3)
    with pytest.raises(ValueError, match=r"phi must be in \[0, 90\)"):
        MohrCoulombSoil(c=0, phi=95, psi=0, E=1e6, nu=0.3)
    with pytest.raises(ValueError, match=r"psi must be in \[0, phi\]"):
        MohrCoulombSoil(c=0, phi=30, psi=50, E=1e6, nu=0.3)
    with pytest.raises(ValueError, match="E must be > 0"):
        MohrCoulombSoil(c=0, phi=30, psi=0, E=0, nu=0.3)
    with pytest.raises(ValueError, match=r"nu must be in \[0, 0.5\)"):
        MohrCoulombSoil(c=0, phi=30, psi=0, E=1e6, nu=0.5)
    with pytest.raises(ValueError, match="rho must be >= 0"):
        MohrCoulombSoil(c=0, phi=30, psi=0, E=1e6, nu=0.3, rho=-1)


def test_mohr_coulomb_soil_passes_integration_overrides() -> None:
    with pytest.warns(ASDPlasticIntegrationWarning, match="experimental"):
        mat = MohrCoulombSoil(
            c=100, phi=30, psi=10, E=1e6, nu=0.3,
            integration_method="Modified_Euler_Error_Control",
            tangent_type="Secant",
            n_max_iterations=200,
            f_absolute_tol=1e-8,
            f_relative_tol=1e-7,
            strict_convergence=False,
        )
    io_dict = dict(mat.integration_options)
    assert io_dict["integration_method"] == "Modified_Euler_Error_Control"
    assert io_dict["tangent_type"] == "Secant"
    assert io_dict["n_max_iterations"] == 200
    assert io_dict["f_absolute_tol"] == 1e-8
    assert io_dict["f_relative_tol"] == 1e-7
    assert io_dict["strict_convergence"] is False


# ---------------------------------------------------------------------------
# 2b. ADR 0105 D2 / D3 — option defaults, emitted shape, token validation
# ---------------------------------------------------------------------------


def _emitted(mat: ASDPlasticMaterial3D) -> str:
    e = TclEmitter()
    mat._emit(e, tag=7)
    return e.lines()[-1]


def test_mohr_coulomb_soil_emits_the_d2_d3_shape() -> None:
    line = _emitted(MohrCoulombSoil(c=100, phi=30, psi=10, E=1e6, nu=0.3))
    assert (
        "Begin_Model_Parameters YoungsModulus 1000000.0 PoissonsRatio 0.3 "
        "MC_phi 30.0 MC_c 100.0 MC_ds 1e-05 MC_psi 10.0 MassDensity 0.0 "
        "InitialP0 0.0 End_Model_Parameters"
    ) in line
    assert "strict_convergence 1" in line          # bool -> int token
    assert "f_relative_tol 0.0" in line
    assert "tangent_type Continuum" in line
    assert "integration_method Backward_Euler" in line
    for foreign in ("AF_cr", "DP_eta", "DuncanChang", "TC_min_stress",
                    "ReferencePressure", "LinearHardeningParameter"):
        assert foreign not in line


def test_strict_convergence_off_emits_zero() -> None:
    line = _emitted(MohrCoulombSoil(
        c=100, phi=30, psi=10, E=1e6, nu=0.3, strict_convergence=False,
    ))
    assert "strict_convergence 0" in line


@pytest.mark.parametrize(
    ("method", "reason"),
    [
        ("Backward_Euler_LineSearch", "ADR-94 M7"),
        ("Runge_Kutta_45_Error_Control_old", "ADR-94 M8"),
    ],
)
def test_refused_integrators_raise_with_the_forks_reason(
    method: str, reason: str,
) -> None:
    with pytest.raises(ValueError, match=f"REFUSED by the fork \\({reason}\\)"):
        MohrCoulombSoil(
            c=100, phi=30, psi=10, E=1e6, nu=0.3, integration_method=method,
        )
    # Same on the generic class — the check lives in __post_init__.
    with pytest.raises(ValueError, match=reason):
        _mc(integration_options=(("integration_method", method),))


def test_unknown_tokens_raise_naming_the_valid_set() -> None:
    with pytest.raises(ValueError, match="unknown integration_method 'Euler'"):
        _mc(integration_options=(("integration_method", "Euler"),))
    with pytest.raises(ValueError, match="unknown tangent_type 'Consistent'"):
        _mc(integration_options=(("tangent_type", "Consistent"),))
    with pytest.raises(
        ValueError, match="unknown return_to_yield_surface 'Always'",
    ):
        _mc(integration_options=(("return_to_yield_surface", "Always"),))


@pytest.mark.parametrize(
    "method",
    [
        "Forward_Euler", "Forward_Euler_Subincrement",
        "Modified_Euler_Error_Control", "Runge_Kutta_45_Error_Control",
    ],
)
def test_explicit_integrators_are_accepted_with_a_warning(method: str) -> None:
    with pytest.warns(ASDPlasticIntegrationWarning, match="experimental"):
        mat = _mc(integration_options=(("integration_method", method),))
    assert dict(mat.integration_options)["integration_method"] == method


def test_backward_euler_and_every_valid_token_are_silent() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for tangent in sorted(_ASDP_TANGENT_TYPES):
            for ret in sorted(_ASDP_RETURN_TO_YIELD_SURFACE):
                _mc(integration_options=(
                    ("integration_method", "Backward_Euler"),
                    ("tangent_type", tangent),
                    ("return_to_yield_surface", ret),
                ))


@pytest.mark.parametrize(
    "helper", [MohrCoulombSoil, MohrCoulombTensionCutoffSoil, HoekBrownRock],
    ids=lambda f: f.__name__,
)
def test_namespace_wrapper_mirrors_the_helper_defaults(helper) -> None:
    """``ops.nDMaterial.<helper>`` re-states every default in its own
    signature (the second site the SANISAND guide warns about)."""
    import inspect

    from apeGmsh.opensees._internal.ns.nd import _NDMaterialNS

    params = inspect.signature(helper).parameters
    wrapper = inspect.signature(getattr(_NDMaterialNS, helper.__name__)).parameters
    for key, param in params.items():
        assert key in wrapper, key
        assert wrapper[key].default == param.default, key
    assert set(wrapper) - set(params) == {"self", "name"}


# ---------------------------------------------------------------------------
# 2c. ADR 0105 D5 — MohrCoulombTensionCutoffSoil / HoekBrownRock
# ---------------------------------------------------------------------------

_HB = dict(E=5.0e7, nu=0.25, sigci=50000.0, mb=0.5741, s=0.0117, a=0.5028)


def test_mohr_coulomb_tension_cutoff_soil_emits_exactly_its_schema() -> None:
    mat = MohrCoulombTensionCutoffSoil(
        c=100.0, phi=30.0, psi=10.0, tension_cutoff=50.0, E=1e6, nu=0.3,
        rho=2.0,
    )
    assert (mat.yf, mat.pf, mat.el) == (
        "MohrCoulombTensionCutoff_YF", "MohrCoulombTensionCutoff_PF",
        "LinearIsotropic3D_EL",
    )
    assert mat.iv == _MC_IV
    assert dict(mat.internal_variables) == {"BackStress": (0.0,) * 6}
    assert set(dict(mat.model_parameters)) == _MC_SCHEMA | {"TC_min_stress"}
    assert set(dict(mat.model_parameters)) == asdp_parameter_schema(
        mat.yf, mat.pf, mat.el, mat.iv,
    )
    line = _emitted(mat)
    assert (
        "Begin_Model_Parameters YoungsModulus 1000000.0 PoissonsRatio 0.3 "
        "MC_phi 30.0 MC_c 100.0 MC_ds 1e-05 MC_psi 10.0 TC_min_stress 50.0 "
        "MassDensity 2.0 InitialP0 0.0 End_Model_Parameters"
    ) in line
    assert "strict_convergence 1" in line
    assert "tangent_type Continuum" in line
    assert "f_relative_tol 0.0" in line


def test_mohr_coulomb_tension_cutoff_soil_validates_inputs() -> None:
    with pytest.raises(ValueError, match="tension_cutoff must be >= 0"):
        MohrCoulombTensionCutoffSoil(
            c=100, phi=30, psi=0, tension_cutoff=-1.0, E=1e6, nu=0.3,
        )
    with pytest.raises(ValueError, match=r"MohrCoulombTensionCutoffSoil: psi"):
        MohrCoulombTensionCutoffSoil(
            c=100, phi=30, psi=40, tension_cutoff=1.0, E=1e6, nu=0.3,
        )
    with pytest.raises(ValueError, match="MohrCoulombTensionCutoffSoil: E"):
        MohrCoulombTensionCutoffSoil(
            c=100, phi=30, psi=0, tension_cutoff=1.0, E=0, nu=0.3,
        )


def test_hoek_brown_rock_emits_exactly_its_schema() -> None:
    mat = HoekBrownRock(**_HB, rho=2.6)
    assert (mat.yf, mat.pf, mat.el) == (
        "HoekBrown_YF", "HoekBrown_PF", "LinearIsotropic3D_EL",
    )
    assert mat.iv == _MC_IV
    mp = dict(mat.model_parameters)
    assert set(mp) == asdp_parameter_schema(mat.yf, mat.pf, mat.el, mat.iv)
    assert set(mp) == {
        "YoungsModulus", "PoissonsRatio", "HB_sigci", "HB_mb", "HB_s",
        "HB_a", "HB_mb_psi", "HB_ds", "MassDensity", "InitialP0",
    }
    assert mp["HB_mb_psi"] == mp["HB_mb"] == 0.5741      # associated by default
    assert mp["HB_ds"] == 0.0
    line = _emitted(mat)
    assert (
        "Begin_Model_Parameters YoungsModulus 50000000.0 PoissonsRatio 0.25 "
        "HB_sigci 50000.0 HB_mb 0.5741 HB_s 0.0117 HB_a 0.5028 "
        "HB_mb_psi 0.5741 HB_ds 0.0 MassDensity 2.6 InitialP0 0.0 "
        "End_Model_Parameters"
    ) in line
    assert "strict_convergence 1" in line
    assert "tangent_type Continuum" in line


def test_hoek_brown_rock_non_associated_and_validation() -> None:
    mat = HoekBrownRock(**_HB, mb_psi=0.2)
    assert dict(mat.model_parameters)["HB_mb_psi"] == 0.2
    for bad in (
        dict(sigci=0.0), dict(mb=0.0), dict(s=0.0), dict(s=1.5),
        dict(a=0.0), dict(a=1.2),
    ):
        with pytest.raises(ValueError, match="HoekBrownRock"):
            HoekBrownRock(**{**_HB, **bad})
    with pytest.raises(ValueError, match="mb_psi must be > 0"):
        HoekBrownRock(**_HB, mb_psi=0.0)


@pytest.mark.parametrize(
    "make",
    [
        lambda: MohrCoulombTensionCutoffSoil(
            c=100, phi=30, psi=0, tension_cutoff=10.0, E=1e6, nu=0.3,
        ),
        lambda: HoekBrownRock(**_HB),
    ],
    ids=["MohrCoulombTensionCutoffSoil", "HoekBrownRock"],
)
def test_d5_helpers_are_plane_strain_wrappable(make) -> None:
    base = make()
    wrapper = PlaneStrain(base=base)
    assert wrapper.dependencies() == (base,)


def test_ndmaterial_namespace_mohr_coulomb_tension_cutoff_soil() -> None:
    fem = make_two_node_beam()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.MohrCoulombTensionCutoffSoil(
        c=1014.0, phi=45.95, psi=11.49, tension_cutoff=100.0,
        E=4080000.0, nu=0.18, rho=4.5,
    )
    assert isinstance(mat, ASDPlasticMaterial3D)
    assert mat.yf == "MohrCoulombTensionCutoff_YF"
    assert dict(mat.model_parameters)["TC_min_stress"] == 100.0
    assert ops.tag_for(mat) == 1
    wrapper = ops.nDMaterial.PlaneStrain(base=mat)
    assert ops.tag_for(wrapper) == 2


def test_ndmaterial_namespace_hoek_brown_rock() -> None:
    fem = make_two_node_beam()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.HoekBrownRock(**_HB, f_relative_tol=1e-8)
    assert isinstance(mat, ASDPlasticMaterial3D)
    assert mat.yf == "HoekBrown_YF"
    assert dict(mat.integration_options)["f_relative_tol"] == 1e-8
    assert ops.tag_for(mat) == 1


# ---------------------------------------------------------------------------
# 3. PlaneStrain wrapper
# ---------------------------------------------------------------------------


def test_plane_strain_wraps_3d_material() -> None:
    """The wrapper carries its 3D base as a dependency so the
    topological sort emits the base first."""
    base = MohrCoulombSoil(c=100, phi=30, psi=0, E=1e6, nu=0.3)
    wrapper = PlaneStrain(base=base)
    deps = wrapper.dependencies()
    assert deps == (base,)


def test_plane_strain_emit_shape() -> None:
    """Through the bridge: PlaneStrain emits ``nDMaterial PlaneStrain
    $tag $base_tag`` with the bridge-resolved tag for ``base``."""
    fem = make_two_node_beam()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    base = ops.nDMaterial.MohrCoulombSoil(
        c=100, phi=30, psi=0, E=1e6, nu=0.3,
    )
    wrapper = ops.nDMaterial.PlaneStrain(base=base)
    bm = ops.build()
    e = TclEmitter()
    bm.emit(e)
    text = "\n".join(e.lines())
    # The base 3D material is emitted (tag is the bridge-allocated one).
    base_tag = bm.tag_for[id(base)]
    wrapper_tag = bm.tag_for[id(wrapper)]
    assert f"nDMaterial ASDPlasticMaterial3D {base_tag}" in text
    assert f"nDMaterial PlaneStrain {wrapper_tag} {base_tag}" in text


# ---------------------------------------------------------------------------
# 4. Namespace registration
# ---------------------------------------------------------------------------


def test_ndmaterial_namespace_asd_plastic_material_3d() -> None:
    fem = make_two_node_beam()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ASDPlasticMaterial3D(
        yf="MohrCoulomb_YF",
        pf="MohrCoulomb_PF",
        el="LinearIsotropic3D_EL",
        iv="BackStress(NullHardeningTensorFunction):",
        internal_variables={"BackStress": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        model_parameters=dict(_MC_PARAMS),
        integration_options={"integration_method": "Backward_Euler"},
    )
    assert isinstance(mat, ASDPlasticMaterial3D)
    # Bridge allocated a tag.
    tag = ops.tag_for(mat)
    assert tag == 1


def test_ndmaterial_namespace_internal_variables_scalar_normalizes() -> None:
    """Passing a scalar for an IV value normalizes to a 1-tuple."""
    fem = make_two_node_beam()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ASDPlasticMaterial3D(
        yf="X", pf="Y", el="Z", iv="W",
        internal_variables={"ScalarIV": 42.0},
    )
    iv_dict = dict(mat.internal_variables)
    assert iv_dict["ScalarIV"] == (42.0,)


def test_ndmaterial_namespace_mohr_coulomb_soil() -> None:
    fem = make_two_node_beam()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.MohrCoulombSoil(
        c=1014.0, phi=45.95, psi=11.49,
        E=4080000.0, nu=0.18, rho=4.5,
    )
    assert isinstance(mat, ASDPlasticMaterial3D)
    assert ops.tag_for(mat) == 1


def test_ndmaterial_namespace_plane_strain() -> None:
    fem = make_two_node_beam()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    base = ops.nDMaterial.MohrCoulombSoil(
        c=100, phi=30, psi=0, E=1e6, nu=0.3,
    )
    wrapper = ops.nDMaterial.PlaneStrain(base=base)
    assert isinstance(wrapper, PlaneStrain)
    assert wrapper.base is base
    # Both registered, both get tags from the nDMaterial bucket.
    assert ops.tag_for(base) == 1
    assert ops.tag_for(wrapper) == 2
