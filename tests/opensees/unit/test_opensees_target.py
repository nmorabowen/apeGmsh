"""OpenSeesTarget: runtime resolution + capability / require_fork seam.

Covers the single seam that says *which* OpenSees each subprocess path
binds (``binary`` / ``python``) and the live fork expectation
(``require_fork``).  Resolution tests need no openseespy; the live
capability probe is gated behind its availability.
"""
from __future__ import annotations

import inspect
from typing import cast

import pytest

from apeGmsh.opensees import OpenSeesCapabilities, OpenSeesTarget, apeSees
from apeGmsh.opensees._target import (
    resolve_opensees_binary,
    resolve_python_binary,
)


def _has_openseespy() -> bool:
    try:
        import openseespy.opensees  # noqa: F401
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------
# Binary / python resolution precedence
# --------------------------------------------------------------------------
def test_binary_precedence_explicit_over_target_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENSEES_BIN", raising=False)
    target = OpenSeesTarget(binary="T:/fork/OpenSees.exe")
    # explicit bin= wins over everything
    assert resolve_opensees_binary("E:/explicit.exe", target) == "E:/explicit.exe"
    # target wins over env
    monkeypatch.setenv("OPENSEES_BIN", "env-bin")
    assert resolve_opensees_binary(None, target) == "T:/fork/OpenSees.exe"
    # env used when no explicit / target
    assert resolve_opensees_binary(None, None) == "env-bin"


def test_binary_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENSEES_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="OpenSeesTarget"):
        resolve_opensees_binary(None, None)


def test_python_precedence_explicit_over_target_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENSEES_VENV", raising=False)
    target = OpenSeesTarget(python="T:/fork/python.exe")
    assert resolve_python_binary("E:/py.exe", target) == "E:/py.exe"
    assert resolve_python_binary(None, target) == "T:/fork/python.exe"
    # always resolves to *something* with no explicit / target
    assert resolve_python_binary(None, None)


# --------------------------------------------------------------------------
# Bridge wiring
# --------------------------------------------------------------------------
def test_target_stored_and_exposed() -> None:
    target = OpenSeesTarget(binary="b", python="p", require_fork=True)
    ops = apeSees(cast("object", object()))  # type: ignore[arg-type]
    assert ops.opensees is None
    ops2 = apeSees(cast("object", object()), opensees=target)  # type: ignore[arg-type]
    assert ops2.opensees is target


def test_py_accepts_python_kwarg_mirroring_tcl_bin() -> None:
    assert "python" in inspect.signature(apeSees.py).parameters
    assert "bin" in inspect.signature(apeSees.tcl).parameters


# --------------------------------------------------------------------------
# require_fork — fail-loud at the live boundary
# --------------------------------------------------------------------------
def test_require_fork_noop_without_target() -> None:
    ops = apeSees(cast("object", object()))  # type: ignore[arg-type]
    ops._assert_fork_if_required()  # no target -> no probe, no raise


def test_require_fork_raises_on_stock_verdict() -> None:
    ops = apeSees(  # type: ignore[arg-type]
        cast("object", object()), opensees=OpenSeesTarget(require_fork=True)
    )
    # Force a "stock" capability verdict without needing a real build.
    ops.capabilities = lambda: OpenSeesCapabilities(  # type: ignore[method-assign]
        source="live", has_fork=False, has_profiler=False, version="3.8.0"
    )
    with pytest.raises(RuntimeError, match="require_fork=True"):
        ops._assert_fork_if_required()


def test_require_fork_passes_on_fork_verdict() -> None:
    ops = apeSees(  # type: ignore[arg-type]
        cast("object", object()), opensees=OpenSeesTarget(require_fork=True)
    )
    ops.capabilities = lambda: OpenSeesCapabilities(  # type: ignore[method-assign]
        source="live", has_fork=True, has_profiler=True, version="3.8.0"
    )
    ops._assert_fork_if_required()  # no raise


# --------------------------------------------------------------------------
# Live capability probe (needs openseespy installed)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _has_openseespy(), reason="openseespy not installed")
def test_capabilities_probe_shape() -> None:
    ops = apeSees(cast("object", object()))  # type: ignore[arg-type]
    caps = ops.capabilities()
    assert isinstance(caps, OpenSeesCapabilities)
    assert caps.source == "live"
    assert isinstance(caps.has_fork, bool)
    assert isinstance(caps.has_profiler, bool)
    # has_fork tracks the fork-only profiler command
    assert caps.has_fork == caps.has_profiler
