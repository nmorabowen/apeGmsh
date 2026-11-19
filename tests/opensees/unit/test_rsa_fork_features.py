"""Chain A link A4-3: the chain M / P1 capability probe.

:func:`rsa_features_of` answers which of the fork's modal-basis and RSA
commands (``residualModes``, ``ritzVectors``, ``responseSpectrumAnalysis
-requireMass``, ``responseSpectrumPerMode``) the in-process build has, and
:meth:`RsaFeatures.require` refuses with "fork too old: needs chain M
(ladruno-v20261027+)" instead of the opaque error the command would give.
Every build here is a stub module, so the tests never touch the real
(maybe stale) engine.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

from apeGmsh.opensees import _target
from apeGmsh.opensees._target import (
    CHAIN_M_MIN_RELEASE,
    ForkTooOldError,
    RsaFeatures,
    probe_live_capabilities,
    require_rsa_feature,
    rsa_features_of,
)
from apeGmsh.opensees.emitter import live

_SHA = "0123456789abcdef0123456789abcdef01234567"

CHAIN_M = ("residualModes", "ritzVectors", "responseSpectrumAnalysis -requireMass")
FEATURES = CHAIN_M + ("responseSpectrumPerMode",)


def _never_called(*args: Any) -> None:
    raise AssertionError("the probe must not run an OpenSees command")


def _build(*commands: str, fork: bool = True) -> ModuleType:
    """A stub ``opensees`` module registering ``commands``."""
    mod = ModuleType("opensees")
    if fork:
        setattr(mod, "ladrunoBuild", lambda: _SHA)
    for name in ("eigen", "modalProperties", "responseSpectrumAnalysis", *commands):
        setattr(mod, name, _never_called)
    return mod


def old_build() -> ModuleType:
    """ladruno-v20261023: a fork with RSA but none of chain M / P1."""
    return _build()


def new_build() -> ModuleType:
    """ladruno-v20261027+: chain M and P1."""
    return _build("residualModes", "ritzVectors", "responseSpectrumPerMode")


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_target, "_RSA_CACHE", {})


def test_old_build_has_nothing() -> None:
    feats = rsa_features_of(old_build())
    assert feats == RsaFeatures(
        residual_modes=False, ritz_vectors=False, rsa_require_mass=False,
        rsa_per_mode=False, build=_SHA,
    )
    assert not feats.chain_m


def test_new_build_has_everything() -> None:
    feats = rsa_features_of(new_build())
    assert feats.chain_m and feats.rsa_per_mode
    for feature in FEATURES:
        feats.require(feature)          # no raise


@pytest.mark.parametrize("feature", CHAIN_M)
def test_old_build_refuses_chain_m_by_name(feature: str) -> None:
    with pytest.raises(ForkTooOldError) as exc:
        rsa_features_of(old_build()).require(feature)
    msg = str(exc.value)
    assert msg.startswith(f"fork too old: needs chain M ({CHAIN_M_MIN_RELEASE}+)")
    assert feature in msg and _SHA[:12] in msg


def test_old_build_refuses_p1() -> None:
    with pytest.raises(ForkTooOldError, match=r"fork too old: needs P1 \(ladruno-v20261027\+\)"):
        rsa_features_of(old_build()).require("responseSpectrumPerMode")


def test_chain_m_without_p1() -> None:
    """A build between fork PRs #1027 and #1032: chain M, no P1."""
    feats = rsa_features_of(_build("residualModes", "ritzVectors"))
    assert feats.chain_m and not feats.rsa_per_mode
    feats.require("residualModes")
    with pytest.raises(ForkTooOldError, match="needs P1"):
        feats.require("responseSpectrumPerMode")


def test_require_mass_follows_residual_modes() -> None:
    """``-requireMass`` (#1021) merged before ``residualModes`` (#1023)."""
    assert rsa_features_of(_build("residualModes")).rsa_require_mass
    assert not rsa_features_of(_build("ritzVectors")).rsa_require_mass


def test_stock_build_is_named_in_the_refusal() -> None:
    feats = rsa_features_of(_build(fork=False))
    assert feats.build is None and not feats.chain_m
    with pytest.raises(ForkTooOldError, match="stock openseespy"):
        feats.require("residualModes")


def test_unknown_feature_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unknown fork feature"):
        rsa_features_of(new_build()).require("modalResponseHistory")


def test_cached_per_module() -> None:
    mod = old_build()
    first = rsa_features_of(mod)
    setattr(mod, "residualModes", _never_called)   # the build cannot change
    assert rsa_features_of(mod) is first
    other = new_build()
    assert rsa_features_of(other).chain_m          # another module, its own probe
    assert rsa_features_of(mod) is first


def test_require_rsa_feature_probes_the_live_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live, "_get_ops", old_build)
    with pytest.raises(ForkTooOldError, match="needs chain M"):
        require_rsa_feature("residualModes")
    monkeypatch.setattr(live, "_get_ops", new_build)
    require_rsa_feature("residualModes")


def test_capabilities_carry_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = new_build()
    monkeypatch.setattr(live, "_get_ops", lambda: fake)
    monkeypatch.setattr(live, "_BACKEND_INFO", live._BACKEND_INFO)  # restore
    caps = probe_live_capabilities()
    assert caps.rsa is rsa_features_of(fake)
    assert caps.rsa is not None and caps.rsa.chain_m
