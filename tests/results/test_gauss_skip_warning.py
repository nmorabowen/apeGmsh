"""ADR 0074 review fix (F3): DomainCapture must WARN when a gauss (continuum
stress/strain) capture drops elements whose class has no RESPONSE_CATALOG
layout (e.g. LadrunoUP) — otherwise the record's dataset is silently empty
with zero user-facing signal after a full transient run.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from apeGmsh.results.capture._domain import DomainCapture


def _capture_with_gauss_skips(skips):
    """A DomainCapture with one fake gauss capturer carrying ``skips`` and no
    line-station capturers (bypasses __init__ — we only exercise the warning
    method)."""
    dc = DomainCapture.__new__(DomainCapture)
    dc._gauss_capturers = [SimpleNamespace(skipped_elements=list(skips))]
    dc._line_station_capturers = []
    return dc


def test_gauss_skip_emits_warning():
    dc = _capture_with_gauss_skips(
        [(10, "class LadrunoUP not in catalog"),
         (11, "class LadrunoUP not in catalog")],
    )
    with pytest.warns(UserWarning, match="gauss") as rec:
        dc._warn_about_skipped_gauss_elements()
    msg = str(rec[0].message)
    assert "2 element(s)" in msg
    assert "LadrunoUP" in msg          # names the offending class reason
    assert "10" in msg and "11" in msg  # names the skipped eids


def test_no_gauss_skips_is_silent():
    import warnings

    dc = _capture_with_gauss_skips([])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dc._warn_about_skipped_gauss_elements()  # must not warn


def test_gauss_warning_points_at_pressure_alternatives():
    dc = _capture_with_gauss_skips([(1, "class LadrunoUP not in catalog")])
    with pytest.warns(UserWarning) as rec:
        dc._warn_about_skipped_gauss_elements()
    msg = str(rec[0].message)
    # The message steers the user to the working pore-pressure / stress paths.
    assert "-dof" in msg or "ladruno" in msg.lower() or "MPCO" in msg
