"""The mass guards of ``spectrum_after_gravity`` (chain A, link A1; gap 6).

Γ divides by the free mass, ``modalProperties``' ``totalFreeMass``. Under
``constraints Transformation`` stock OpenSees and Ladruno forks older than
``ladruno-v20261027`` (WP-211) count the mass on a rigid diaphragm's
constrained nodes as fixed: ``totalMass`` matches the nodes, so the
element-mass check passes, but the free mass and every Γ are zero and the
per-mode response is zero, silently.

The numbers are V1's one-storey diaphragm (``studies/rsa-gaps/v1``): 400 t
spread over the four column tops, ``rigidDiaphragm 3`` to a massless centre
node. On the old build ``modalProperties`` gives ``totalMass`` 400,
``totalFreeMass`` 0 and ``partiFactorMX`` 0; with WP-211 it gives the free
mass 400 and Γ 1 in the sway mode.

The driver's ``_per_mode_spectrum`` runs here on a stand-in domain that
answers the mass queries and stops at the first per-mode command
(``timeSeries``): reaching it means every guard passed.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.analysis.spectrum import DesignSpectrum
from apeGmsh.opensees.procedures import _gravity_modal as gm

CONTEXT = "apeSees.spectrum_after_gravity"
M_TOT = 400.0
TOPS = (11, 12, 13, 14)
MASTER = 100
EIGENVALUES = np.asarray([204.8, 388.0, 997.0])


class _PastTheGuards(Exception):
    """The driver reached its first per-mode command."""


class _Ops:
    """The queries the guards read, from per-node mass rows."""

    def __init__(self, masses: dict[int, tuple[float, ...]]) -> None:
        self._masses = masses

    def getNodeTags(self) -> list[int]:
        return list(self._masses)

    def nodeMass(self, tag: int) -> list[float]:
        return list(self._masses[tag])

    def timeSeries(self, *args: Any) -> None:
        raise _PastTheGuards


def _slave_mass() -> dict[int, tuple[float, ...]]:
    m = M_TOT / len(TOPS)
    rows: dict[int, tuple[float, ...]] = {
        t: (m, m, 0.0, 0.0, 0.0, 0.0) for t in TOPS}
    rows[MASTER] = (0.0,) * 6
    return rows


def _properties(*, free: float, gamma_x: tuple[float, ...],
                gamma_y: tuple[float, ...] = (0.0, 1.0, 0.0),
                total: float = M_TOT) -> dict[str, list[float]]:
    props = {
        "totalMass": [total, total, 0.0, 0.0, 0.0, 0.0],
        "totalFreeMass": [free, free, 0.0, 0.0, 0.0, 0.0],
        "partiFactorMX": list(gamma_x),
        "partiFactorMY": list(gamma_y),
        "partiFactorMZ": [0.0, 0.0, 0.0],
    }
    for comp in ("MX", "MY", "MZ"):
        props[f"partiMass{comp}"] = [
            g * g * free for g in props[f"partiFactor{comp}"]]
    return props


def _run(properties: dict[str, list[float]], *directions: int,
         masses: dict[int, tuple[float, ...]] | None = None) -> None:
    from apeGmsh.opensees.analysis.algorithm import Linear

    host = SimpleNamespace(_primitives=[Linear()], _ndm=3)
    live = SimpleNamespace(ops=_Ops(masses or _slave_mass()))
    spectra = tuple(DesignSpectrum((0.0, 2.0), (5.0, 5.0), direction=d)
                    for d in directions or (1,))
    gm._GravityModalMixin._per_mode_spectrum(
        host,  # type: ignore[arg-type]
        live,  # type: ignore[arg-type]
        EIGENVALUES, properties, spectra, {}, context=CONTEXT,
    )


# -- A1-1: nodal mass, no free mass --------------------------------------------

def test_mass_on_constrained_nodes_before_wp211_is_refused() -> None:
    """V1 on the old build: totalMass 400, free mass 0, Γ 0. Without the
    guard the driver goes on to the per-mode loop and returns zeros."""
    props = _properties(free=0.0, gamma_x=(0.0, 0.0, 0.0),
                        gamma_y=(0.0, 0.0, 0.0))
    with pytest.raises(BridgeError) as err:
        _run(props, 1)
    msg = str(err.value)
    assert "direction 1" in msg
    assert "free mass of 0" in msg and "carry 400" in msg
    assert "Transformation" in msg and "WP-211" in msg
    assert "ladruno-v20261027" in msg


def test_mass_on_constrained_nodes_with_wp211_runs() -> None:
    """V1 with WP-211: the free mass is the nodes', Γ_X = 1 in the sway
    mode; every guard passes."""
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0))
    with pytest.raises(_PastTheGuards):
        _run(props, 1)


def test_each_loaded_direction_is_checked() -> None:
    """Free mass lost along Y only: the X spectrum alone runs, adding the
    Y spectrum is refused on direction 2."""
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0))
    props["totalFreeMass"][1] = 0.0
    with pytest.raises(_PastTheGuards):
        _run(props, 1)
    with pytest.raises(BridgeError, match="direction 2 acts on a free mass"):
        _run(props, 1, 2)


def test_a_free_mass_at_round_off_is_refused() -> None:
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0))
    props["totalFreeMass"][0] = 1e-12 * M_TOT
    with pytest.raises(BridgeError, match="free mass"):
        _run(props, 1)


def test_a_build_without_total_free_mass_is_refused() -> None:
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0))
    del props["totalFreeMass"]
    with pytest.raises(BridgeError, match="no 'totalFreeMass'"):
        _run(props, 1)


# -- A1-2: every Γ of a loaded direction zero ----------------------------------

def test_a_loaded_direction_with_every_gamma_zero_is_refused() -> None:
    """Free mass counted, but no requested mode participates along X."""
    props = _properties(free=M_TOT, gamma_x=(0.0, 0.0, 0.0))
    with pytest.raises(BridgeError,
                       match="zero participation factor .* direction 1"):
        _run(props, 1)


def test_a_zero_gamma_off_the_loaded_directions_is_not_refused() -> None:
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0),
                        gamma_y=(0.0, 0.0, 0.0))
    with pytest.raises(_PastTheGuards):
        _run(props, 1)
    with pytest.raises(BridgeError, match="direction 2"):
        _run(props, 1, 2)


def test_a_direction_without_nodal_mass_falls_to_the_gamma_guard() -> None:
    """No nodal mass along Y: no free mass is expected there, so the
    free-mass guard does not fire; Γ_Y is zero, so the Γ guard does."""
    masses = {t: (m[0], 0.0, 0.0, 0.0, 0.0, 0.0)
              for t, m in _slave_mass().items()}
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0),
                        gamma_y=(0.0, 0.0, 0.0))
    props["totalMass"][1] = props["totalFreeMass"][1] = 0.0
    with pytest.raises(BridgeError, match="zero participation factor"):
        _run(props, 2, masses=masses)


# -- A1-3: the element-mass refusal is unchanged ---------------------------------

def test_element_mass_is_still_refused_against_total_mass() -> None:
    """totalMass above the nodal mass is element mass, refused as before,
    whatever the free mass."""
    props = _properties(free=M_TOT + 100.0, gamma_x=(1.0, 0.0, 0.0),
                        total=M_TOT + 100.0)
    with pytest.raises(BridgeError, match="element mass"):
        _run(props, 1)


def test_element_mass_refusal_comes_first() -> None:
    """With both faults, the element-mass refusal answers (its question,
    gap 5, is asked first)."""
    props = _properties(free=0.0, gamma_x=(0.0, 0.0, 0.0),
                        total=M_TOT + 100.0)
    with pytest.raises(BridgeError, match="element mass"):
        _run(props, 1)
