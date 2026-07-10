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

from apeGmsh.results.readers._ladruno_element_io import continuum_canonical


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
