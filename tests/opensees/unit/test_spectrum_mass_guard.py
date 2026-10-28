"""The mass guards of ``spectrum_after_gravity`` (chain A, link A1; gap 6).

Γ divides by the free mass, ``modalProperties``' ``totalFreeMass``. Under
``constraints Transformation`` stock OpenSees and Ladruno forks older than
``ladruno-v20261027`` (WP-211) count the mass on a rigid diaphragm's
constrained nodes as fixed: ``totalMass`` matches the nodes, so the
element-mass check passes, but the free mass falls short of the nodal mass
on unfixed DOFs. When all of it is lost every Γ is zero and the per-mode
response is zero; when one floor's share is lost Γ is too low. Either way
there is no error.

The numbers are V1's one-storey diaphragm (``studies/rsa-gaps/v1``): 400 t
spread over the four column tops, ``rigidDiaphragm 3`` to a massless centre
node. On the old build ``modalProperties`` gives ``totalMass`` 400,
``totalFreeMass`` 0 and ``partiFactorMX`` 0; with WP-211 it gives the free
mass 400 and Γ 1 in the sway mode. The review's two-storey variant (floor 1
on the column tops, floor 2 on the master node) gives a free mass of 400 of
800 and a Γ_X 15% low.

The driver's ``_per_mode_spectrum`` runs here on a stand-in domain that
answers the mass and fix queries and stops at the first per-mode command
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

Rows = dict[int, tuple[float, ...]]


class _PastTheGuards(Exception):
    """The driver reached its first per-mode command."""


class _Ops:
    """The queries the guards read, from per-node mass rows and fixes."""

    def __init__(self, masses: Rows, fixed: dict[int, list[int]]) -> None:
        self._masses = masses
        self._fixed = fixed

    def getNodeTags(self) -> list[int]:
        return list(self._masses)

    def nodeMass(self, tag: int) -> list[float]:
        return list(self._masses[tag])

    def getFixedNodes(self) -> list[int]:
        return list(self._fixed)

    def getFixedDOFs(self, tag: int) -> list[int]:
        return list(self._fixed[tag])

    def timeSeries(self, *args: Any) -> None:
        raise _PastTheGuards


def _slave_mass(floors: int = 1) -> Rows:
    m = M_TOT / len(TOPS)
    rows: Rows = {}
    for f in range(floors):
        rows.update({1000 * f + t: (m, m, 0.0, 0.0, 0.0, 0.0) for t in TOPS})
        rows[1000 * f + MASTER] = (0.0,) * 6
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
         masses: Rows | None = None,
         fixed: dict[int, list[int]] | None = None, ndm: int = 3) -> None:
    from apeGmsh.opensees.analysis.algorithm import Linear

    host = SimpleNamespace(_primitives=[Linear()], _ndm=ndm)
    live = SimpleNamespace(ops=_Ops(masses or _slave_mass(), fixed or {}))
    spectra = tuple(DesignSpectrum((0.0, 2.0), (5.0, 5.0), direction=d)
                    for d in directions or (1,))
    gm._GravityModalMixin._per_mode_spectrum(
        host,  # type: ignore[arg-type]
        live,  # type: ignore[arg-type]
        EIGENVALUES, properties, spectra, {}, context=CONTEXT,
    )


# -- A1-1: the free mass falls short of the unfixed nodal mass ------------------

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


def test_a_partial_loss_is_refused() -> None:
    """The review's two-storey V1 on the old build: floor 1 on the column
    tops (lost), floor 2 on the master node (kept). The free mass is 400
    of 800 and Γ is low but not zero."""
    masses = _slave_mass(2)
    for t in TOPS:
        masses[1000 + t] = (0.0,) * 6
    masses[1000 + MASTER] = (M_TOT, M_TOT, 0.0, 0.0, 0.0, 0.0)
    props = _properties(free=M_TOT, gamma_x=(1.2, -0.4, 0.0),
                        total=2.0 * M_TOT)
    with pytest.raises(BridgeError) as err:
        _run(props, 1, masses=masses)
    assert "free mass of 400" in str(err.value)
    assert "carry 800" in str(err.value)
    props["totalFreeMass"][0] = 2.0 * M_TOT * (1.0 - 1e-8)  # round-off
    with pytest.raises(_PastTheGuards):
        _run(props, 1, masses=masses)


def test_mass_on_sp_fixed_dofs_is_not_expected_free() -> None:
    """A support node with mass along X, fixed along X: its mass is not
    free on any build, so it is not asked of the free mass."""
    masses = _slave_mass()
    masses[1] = (50.0, 50.0, 0.0, 0.0, 0.0, 0.0)
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0),
                        total=M_TOT + 50.0)
    with pytest.raises(_PastTheGuards):
        _run(props, 1, masses=masses, fixed={1: [1, 2, 3, 4, 5, 6]})
    with pytest.raises(BridgeError, match="carry 450"):
        _run(props, 1, masses=masses, fixed={1: [2, 3, 4, 5, 6]})


def test_a_free_mass_above_the_unfixed_mass_runs() -> None:
    """Penalty and Lagrange count the SP-fixed mass as free too."""
    masses = _slave_mass()
    masses[1] = (50.0, 50.0, 0.0, 0.0, 0.0, 0.0)
    props = _properties(free=M_TOT + 50.0, gamma_x=(1.0, 0.0, 0.0),
                        total=M_TOT + 50.0)
    with pytest.raises(_PastTheGuards):
        _run(props, 1, masses=masses, fixed={1: [1, 2, 3, 4, 5, 6]})


def test_each_loaded_direction_is_checked() -> None:
    """Free mass lost along Y only: the X spectrum alone runs, adding the
    Y spectrum is refused on direction 2."""
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0))
    props["totalFreeMass"][1] = 0.0
    with pytest.raises(_PastTheGuards):
        _run(props, 1)
    with pytest.raises(BridgeError, match="direction 2 acts on a free mass"):
        _run(props, 1, 2)


def test_a_build_without_total_free_mass_is_refused() -> None:
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0))
    del props["totalFreeMass"]
    with pytest.raises(BridgeError, match="no 'totalFreeMass'"):
        _run(props, 1)


# -- A1-2: the modes carry none of a loaded direction's mass ---------------------

def test_a_loaded_direction_with_every_gamma_zero_is_refused() -> None:
    """Free mass counted, but no requested mode participates along X."""
    props = _properties(free=M_TOT, gamma_x=(0.0, 0.0, 0.0))
    with pytest.raises(BridgeError,
                       match="direction 1, .* zero to round-off"):
        _run(props, 1)


def test_a_gamma_at_round_off_is_refused() -> None:
    """The review's repro: one Y-sway mode under Penalty gave
    Γ_X = -2.8e-8, not 0."""
    props = _properties(free=M_TOT, gamma_x=(-2.8e-8,), gamma_y=(1.0,))
    props["partiFactorMZ"] = [0.0]
    props["partiMassMZ"] = [0.0]
    with pytest.raises(BridgeError, match="zero to round-off"):
        _run(props, 1)
    with pytest.raises(_PastTheGuards):
        _run(props, 2)


def test_the_gamma_guard_does_not_depend_on_sign() -> None:
    props = _properties(free=M_TOT, gamma_x=(-1.0, 0.0, 0.0))
    with pytest.raises(_PastTheGuards):
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
    free-mass guard does not fire; the modes carry none, so the Γ guard
    does."""
    masses = {t: (m[0], 0.0, 0.0, 0.0, 0.0, 0.0)
              for t, m in _slave_mass().items()}
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0),
                        gamma_y=(0.0, 0.0, 0.0))
    props["totalMass"][1] = props["totalFreeMass"][1] = 0.0
    with pytest.raises(BridgeError, match="zero to round-off"):
        _run(props, 2, masses=masses)


def test_a_nan_effective_mass_is_refused() -> None:
    props = _properties(free=M_TOT, gamma_x=(1.0, 0.0, 0.0))
    props["partiMassMX"][2] = float("nan")
    with pytest.raises(BridgeError, match="effective mass of nan"):
        _run(props, 1)


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
