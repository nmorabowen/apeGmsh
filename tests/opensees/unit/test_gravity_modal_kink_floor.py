"""The per-mode kink verdict and the ``-increment`` leg guard of
``spectrum_after_gravity`` (ADR 0124 D4, #2545), on synthetic forces.

A leg's change in an element's resisting force is modelled as
``f±(s) = ±k s + c s² + n±``: ``k`` the tangent, ``c`` a smooth second-order
term, and ``n±`` round-off of a fixed size. A kink is a different tangent on
each side, ``f+ = k₁ s``, ``f- = -k₂ s``. The second difference over the
first (``|f+ + f-| / |f+ - f-|``) is then:

* round-off: ``|n+ + n-| / (2 k s)``, shrinking as the probe ``s`` grows;
* smooth: ``c s / k``, growing with it;
* kink: ``|k₁ - k₂| / (k₁ + k₂)``, the same at every ``s``.

The round-off numbers are the issue's (#2545): a near-rigid link in the
centred-mass diaphragm (``I`` x 1e6 under 30 times the gravity) read a
second difference of 0.040 against a first of 27.3 at the probe, 1.48e-3,
just above the limit, and the same 0.03 against 2732 at 100 times it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from apeGmsh.opensees.procedures import _gravity_modal as gm

ELEMENTS = (1, 2)


def _legs(f_plus: dict[int, float], f_minus: dict[int, float]) -> list[dict[int, np.ndarray]]:
    return [{e: np.asarray([v]) for e, v in f_plus.items()},
            {e: np.asarray([v]) for e, v in f_minus.items()}]


def _family(k: float, *, c: float = 0.0, n_plus: float = 0.0,
            n_minus: float = 0.0, k_minus: float | None = None,
            stiff: float = 6.0e6) -> Any:
    """Legs at probe multiple ``s``: element 1 is the one judged, element 2 a
    stiff column the mode deforms (so the floor does not decide)."""
    km = k if k_minus is None else k_minus

    def at(s: float) -> list[dict[int, np.ndarray]]:
        return _legs({1: k * s + c * s * s + n_plus, 2: stiff * s},
                     {1: -km * s + c * s * s + n_minus, 2: -stiff * s})
    return at


class _LegsAt:
    """The ``legs_at`` callable the driver passes: records the scales it
    ran, and returns the scale as the legs' stand-in."""

    def __init__(self, family: Any) -> None:
        self.family = family
        self.scales: list[float] = []

    def __call__(self, scale: float) -> tuple[float, list[dict[int, np.ndarray]]]:
        self.scales.append(scale)
        return scale, self.family(scale)


def test_round_off_in_a_near_rigid_link_passes_on_the_reprobe() -> None:
    # The issue's link: 0.040 against 27.3 is 1.48e-3, above the limit.
    family = _family(13.66, n_plus=0.0202, n_minus=0.0202)
    worst, ratio = gm._kink_ratio(ELEMENTS, family(1.0))
    assert worst == 1
    assert ratio == pytest.approx(0.0404 / 27.32, rel=1e-12)
    assert ratio > gm._KINK_TOL
    legs_at = _LegsAt(family)
    worst, ratio, scale, legs = gm._judge_kink(ELEMENTS, legs_at)
    assert legs_at.scales == [1.0, gm._REPROBE]
    # The same round-off against a first difference 100 times larger,
    # and the values come from those legs.
    assert ratio == pytest.approx(0.0404 / 2732.0, rel=1e-12)
    assert ratio <= gm._KINK_TOL
    assert scale == legs == gm._REPROBE


@pytest.mark.parametrize(("k1", "k2"), [
    (1.0, 0.0),        # a no-tension brace at zero strain: 1.0
    (1.0, 0.86),       # Concrete01 on its backbone: 7.4e-2 live
    (1.0, 0.99),       # a kink just above the limit: 5.0e-3
])
def test_a_kink_is_refused_at_both_amplitudes(k1: float, k2: float) -> None:
    family = _family(k1 * 1e3, k_minus=k2 * 1e3)
    legs_at = _LegsAt(family)
    worst, ratio, _, _ = gm._judge_kink(ELEMENTS, legs_at)
    assert legs_at.scales == [1.0, gm._REPROBE]
    assert worst == 1
    assert ratio == pytest.approx(abs(k1 - k2) / (k1 + k2), rel=1e-9)
    assert ratio > gm._KINK_TOL


def test_a_kink_hidden_under_round_off_is_still_refused() -> None:
    # Round-off on top of a kink: the re-probe drops the round-off and
    # leaves the kink's own ratio, 5e-3.
    family = _family(1e3, k_minus=0.99e3, n_plus=3.0, n_minus=3.0)
    _, ratio, _, _ = gm._judge_kink(ELEMENTS, _LegsAt(family))
    assert ratio > gm._KINK_TOL
    assert ratio == pytest.approx(0.01 / 1.99, rel=1e-2)


def test_a_smooth_element_is_not_reprobed() -> None:
    # c s / k = 1e-6 at the probe (the PDelta frames read ~1e-8).
    family = _family(1e3, c=1e-3)
    legs_at = _LegsAt(family)
    _, ratio, scale, legs = gm._judge_kink(ELEMENTS, legs_at)
    assert legs_at.scales == [1.0]
    assert scale == legs == 1.0
    assert ratio == pytest.approx(2e-3 / 2e3, rel=1e-9)


def test_a_barely_deformed_element_is_judged_against_the_floor() -> None:
    # Element 1 barely moves in the mode (first difference 2e-12) and
    # carries 1e-9 of round-off: judged against 1e-6 of the column's
    # first difference (1.2e7), not its own, so 1e-9 / 12 passes.
    family = _family(1e-12, n_plus=5e-10, n_minus=5e-10)
    legs_at = _LegsAt(family)
    worst, ratio, _, _ = gm._judge_kink(ELEMENTS, legs_at)
    assert legs_at.scales == [1.0]
    assert worst == 1
    assert ratio == pytest.approx(1e-9 / (gm._ELEMENT_FLOOR * 1.2e7), rel=1e-6)


# -- the -increment leg always reverts -----------------------------------------

class _Ops:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def responseSpectrumAnalysis(self, *args: Any) -> None:  # noqa: N802
        self.calls.append(args)


def test_an_imposed_leg_reverts_after_reading() -> None:
    ops = _Ops()
    assert gm._imposed_leg(ops, lambda: None, lambda: 7) == 7
    assert ops.calls == [("-revert",)]


def test_an_imposed_leg_off_the_mode_still_reverts() -> None:
    ops = _Ops()

    def impose() -> None:
        raise RuntimeError("imposed a displacement 1e-3 away")

    with pytest.raises(RuntimeError, match="imposed a displacement"):
        gm._imposed_leg(ops, impose, lambda: 7)
    assert ops.calls == [("-revert",)]


def test_an_extractor_raising_on_an_imposed_leg_still_reverts() -> None:
    ops = _Ops()

    def read() -> int:
        raise ValueError("extractor 'roof' changed length")

    with pytest.raises(ValueError, match="changed length"):
        gm._imposed_leg(ops, lambda: None, read)
    assert ops.calls == [("-revert",)]
