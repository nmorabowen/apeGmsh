"""The self-describing .ladruno reader surfaces a recorded out-of-plane
stress as ``stress_zz`` with no catalog change.

When the OpenSees/Ladruno fork records a 4th Gauss-point stress component
for plane-strain elements (labelled ``sigma33`` in the digit convention or
``sigma_zz`` in the axis convention), the token→canonical mapping in the
.ladruno element reader turns it into ``stress_zz`` — which the derived
layer then consumes as the true out-of-plane stress. This locks that
mapping so the fork side and apeGmsh side stay in agreement.
"""
from __future__ import annotations

from apeGmsh.results.readers._ladruno_element_io import (
    continuum_canonical,
    material_bucket_canonicals,
)


def test_sigma33_digit_form_maps_to_stress_zz():
    assert continuum_canonical("sigma33") == "stress_zz"


def test_sigma_zz_axis_form_maps_to_stress_zz():
    assert continuum_canonical("sigma_zz") == "stress_zz"


def test_out_of_plane_strain_forms_map():
    assert continuum_canonical("eta33") == "strain_zz"
    assert continuum_canonical("eps_zz") == "strain_zz"


def test_case_insensitive():
    # The recorder's casing shouldn't matter.
    assert continuum_canonical("SIGMA33") == "stress_zz"
    assert continuum_canonical("Sigma_ZZ") == "stress_zz"
    assert continuum_canonical("SIGMA11") == "stress_xx"


def test_in_plane_forms_unchanged():
    # regression: the recovery/4th-component work must not disturb the
    # existing in-plane mappings.
    assert continuum_canonical("sigma11") == "stress_xx"
    assert continuum_canonical("sigma22") == "stress_yy"
    assert continuum_canonical("sigma12") == "stress_xy"


def test_eps_digit_form_maps_to_strain():
    # LadrunoBrick tags total strains ``eps11..eps13`` (digit form) —
    # previously unmapped, so gauss strain reads from a LadrunoBrick
    # .ladruno silently returned empty.
    assert continuum_canonical("eps11") == "strain_xx"
    assert continuum_canonical("eps33") == "strain_zz"
    assert continuum_canonical("eps12") == "strain_xy"
    assert continuum_canonical("eps13") == "strain_xz"


def test_epsp_forms_map_to_plastic_strain():
    # Plastic-strain tensor labels (fork element branch, following the
    # eps11 convention): digit + axis forms → plastic_strain_*.
    assert continuum_canonical("epsp11") == "plastic_strain_xx"
    assert continuum_canonical("epsp33") == "plastic_strain_zz"
    assert continuum_canonical("epsp12") == "plastic_strain_xy"
    assert continuum_canonical("epsp_xx") == "plastic_strain_xx"
    assert continuum_canonical("epsp_zz") == "plastic_strain_zz"
    # epsilon long-form still total strain; unknown stems still None.
    assert continuum_canonical("epsilon11") == "strain_xx"
    assert continuum_canonical("plasticStrain") is None


def test_asdplastic_pstrain_forms_map():
    # ASDPlasticMaterial3D spells its plastic-strain response ``pstrain``.
    assert continuum_canonical("pstrain11") == "plastic_strain_xx"
    assert continuum_canonical("pstrain33") == "plastic_strain_zz"
    assert continuum_canonical("pstrain12") == "plastic_strain_xy"
    assert continuum_canonical("pstrain_xx") == "plastic_strain_xx"


def test_accumulated_peeq_scalar_labels_map():
    # Accumulated equivalent plastic strain, per material spelling:
    # ASDPlastic ``eqpstrain``; LadrunoJ2 ``equivalentPlasticStrain`` /
    # ``plasticStrainEq`` / ``ebarP``. All → equivalent_plastic_strain.
    assert continuum_canonical("eqpstrain") == "equivalent_plastic_strain"
    assert continuum_canonical("ebarP") == "equivalent_plastic_strain"
    assert continuum_canonical("equivalentPlasticStrain") == "equivalent_plastic_strain"
    assert continuum_canonical("plasticStrainEq") == "equivalent_plastic_strain"


# ---------------------------------------------------------------------------
# ADR 0105 Amendment 1 — the material-level buckets, keyed by TOKEN
# ---------------------------------------------------------------------------
#
# ``ASDPlasticMaterial3D::setResponse`` labels its material-level scalars
# with the material's own private names. Every one of them was unmapped,
# so the column was dropped in silence: recording ``material.PStress``
# and its siblings produced buckets in the file that no reader surfaced.
#
# The mapping is keyed by the BUCKET TOKEN, never by the label, because
# the labels are not unique across levels: ``material.PStress`` labels
# its column ``p``, and the section axial force is ``P`` —
# canonicalisation is case-insensitive, so a label key would turn every
# force-based beam station into a Gauss mean stress.


def test_material_bucket_invariants_map_by_token():
    # Provenance-distinct names: these are what the MATERIAL computed,
    # NOT the reader's tensor-derived mean_stress / j2_stress / ... .
    assert material_bucket_canonicals("material.PStress") == (
        "material_mean_stress",
    )
    assert material_bucket_canonicals("material.J2Stress") == (
        "material_j2_stress",
    )
    assert material_bucket_canonicals("material.VolStrain") == (
        "material_volumetric_strain",
    )
    assert material_bucket_canonicals("material.J2Strain") == (
        "material_j2_strain",
    )


def test_material_bucket_token_is_case_insensitive():
    assert material_bucket_canonicals("MATERIAL.PSTRESS") == (
        "material_mean_stress",
    )
    assert material_bucket_canonicals(" material.j2strain ") == (
        "material_j2_strain",
    )


def test_backstress_bucket_maps_in_voigt_order():
    # Six columns, positional, in the material's Voigt order
    # 11, 22, 33, 12, 23, 13 — the order the epsP1.. columns already use.
    assert material_bucket_canonicals("material.BackStress") == (
        "back_stress_xx", "back_stress_yy", "back_stress_zz",
        "back_stress_xy", "back_stress_yz", "back_stress_xz",
    )


def test_scalar_internal_variable_buckets_map():
    assert material_bucket_canonicals("material.YieldStress") == (
        "yield_stress",
    )
    assert material_bucket_canonicals("material.DP_cohesion") == (
        "dp_cohesion",
    )
    assert material_bucket_canonicals("material.CapPressure") == (
        "cap_pressure",
    )
    assert material_bucket_canonicals("material.EpsQpShear") == (
        "eps_qp_shear",
    )


def test_unknown_material_bucket_is_none():
    # No generic pass-through: a bucket nobody mapped stays None so the
    # dropped column surfaces as a GaussColumnDroppedWarning instead of
    # being guessed into a canonical.
    assert material_bucket_canonicals("material.SomeNewIV") is None


# ---------------------------------------------------------------------------
# TIMs A12 — LadrunoSANISAND's IMPL-EX/state responses (fork PR #805/#820)
# ---------------------------------------------------------------------------


def test_sanisand_scalar_buckets_map():
    assert material_bucket_canonicals("material.psi") == ("state_parameter",)
    assert material_bucket_canonicals("material.yieldDistance") == (
        "yield_distance",
    )
    assert material_bucket_canonicals("material.implexError") == (
        "implex_error",
    )
    assert material_bucket_canonicals("material.avgImplexError") == (
        "avg_implex_error",
    )


def test_sanisand_multi_slot_buckets_map():
    assert material_bucket_canonicals("material.substeps") == (
        "substeps_me", "substeps_cap_hit",
    )
    assert material_bucket_canonicals("material.implexDetail") == (
        "implex_detail_total", "implex_detail_dev", "implex_detail_vol",
        "implex_detail_clamp_fired", "implex_detail_clamp_count",
        "implex_detail_f",
    )
    assert material_bucket_canonicals("material.implexRefusals") == (
        "implex_refusals_total", "implex_refusals_sign_change",
        "implex_refusals_control", "implex_refusals_companion",
    )


def test_sanisand_bucket_token_is_case_insensitive():
    assert material_bucket_canonicals("MATERIAL.PSI") == ("state_parameter",)
    assert material_bucket_canonicals(" material.implexdetail ") == (
        "implex_detail_total", "implex_detail_dev", "implex_detail_vol",
        "implex_detail_clamp_fired", "implex_detail_clamp_count",
        "implex_detail_f",
    )


def test_sanisand_aliases_resolve_to_same_canonicals_as_primary():
    assert material_bucket_canonicals(
        "material.stateParameter"
    ) == material_bucket_canonicals("material.psi")
    assert material_bucket_canonicals(
        "material.yieldFunction"
    ) == material_bucket_canonicals("material.yieldDistance")
    assert material_bucket_canonicals(
        "material.substepsME"
    ) == material_bucket_canonicals("material.substeps")
    assert material_bucket_canonicals(
        "material.ladrunoSubsteps"
    ) == material_bucket_canonicals("material.substeps")


def test_material_labels_are_not_in_the_label_map():
    # The whole point of keying by token. ``p`` is the section axial
    # force ``P`` under case-insensitive matching; the label map must not
    # claim it (or any material-private label) at the Gauss level.
    assert continuum_canonical("p") is None
    assert continuum_canonical("P") is None
    assert continuum_canonical("J2stress") is None
    assert continuum_canonical("epsVol") is None
    assert continuum_canonical("BackStress_1") is None
    assert continuum_canonical("YieldStress") is None


def test_self_describing_material_buckets_keep_the_label_path():
    # ``material.stress`` / ``material.strain`` / ``material.pstrain`` /
    # ``material.eqpstrain`` label their own columns, so they are not in
    # the token table and their labels still map.
    assert material_bucket_canonicals("material.stress") is None
    assert material_bucket_canonicals("material.eqpstrain") is None
    assert continuum_canonical("sigma11") == "stress_xx"
    assert continuum_canonical("epsP11") == "plastic_strain_xx"
    assert continuum_canonical("eqpstrain") == "equivalent_plastic_strain"


def test_non_material_tokens_are_never_token_mapped():
    assert material_bucket_canonicals("section.force") is None
    assert material_bucket_canonicals("stress") is None
