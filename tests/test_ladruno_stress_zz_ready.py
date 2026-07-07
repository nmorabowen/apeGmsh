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
