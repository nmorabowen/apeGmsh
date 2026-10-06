"""OpenSeesTarget: runtime resolution + capability / require_fork seam.

Covers the single seam that says *which* OpenSees each subprocess path
binds (``binary`` / ``python``) and the live fork expectation
(``require_fork``).  Resolution tests need no openseespy; the live
capability probe is gated behind its availability.
"""
from __future__ import annotations

import inspect
from pathlib import Path
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


def test_binary_directory_resolves_to_exe_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.name", "nt")
    bin_dir = tmp_path / "dist_bin"
    bin_dir.mkdir()
    exe = bin_dir / "OpenSees.exe"
    exe.write_text("stub")
    # directory at each precedence level resolves to the exe inside it
    assert resolve_opensees_binary(str(bin_dir), None) == str(exe)
    target = OpenSeesTarget(binary=str(bin_dir))
    assert resolve_opensees_binary(None, target) == str(exe)
    monkeypatch.setenv("OPENSEES_BIN", str(bin_dir))
    assert resolve_opensees_binary(None, None) == str(exe)


def test_binary_directory_without_exe_raises_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("os.name", "nt")
    empty_dir = tmp_path / "empty_bin"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="OpenSees.exe"):
        resolve_opensees_binary(str(empty_dir), None)


def test_binary_file_path_unchanged() -> None:
    assert resolve_opensees_binary("E:/explicit.exe", None) == "E:/explicit.exe"


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
# has_fork is the resolver's verdict, not the profiler command
# --------------------------------------------------------------------------
_SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize("fork, profiler", [
    (False, True),     # a profiler command does not make a fork
    (True, False),     # nor does its absence make stock
])
def test_has_fork_is_the_resolvers_verdict(
    monkeypatch: pytest.MonkeyPatch, fork: bool, profiler: bool,
) -> None:
    from types import ModuleType

    from apeGmsh.opensees._target import probe_live_capabilities
    from apeGmsh.opensees.emitter import live

    fake = ModuleType("opensees")   # stubbed: never the real (maybe stale) build
    # criticalTimeStep on both: it is no longer a fork signal
    fake.criticalTimeStep = lambda: 1.0  # type: ignore[attr-defined]
    if fork:
        fake.ladrunoBuild = lambda: _SHA  # type: ignore[attr-defined]
    if profiler:
        fake.profiler = lambda *args: None  # type: ignore[attr-defined]
    monkeypatch.setattr(live, "_get_ops", lambda: fake)
    monkeypatch.setattr(live, "_BACKEND_INFO", live._BACKEND_INFO)  # restore

    caps = probe_live_capabilities()
    assert caps.has_fork is fork
    assert caps.has_fork is (live.get_backend_name() == "ladruno-fork")
    assert caps.build == (_SHA if fork else None)
    assert caps.has_ladruno_up is caps.has_fork
    assert caps.has_profiler is profiler


# --------------------------------------------------------------------------
# Target mode: AUTO by default, explicit pins, require_fork means fork
# --------------------------------------------------------------------------
def test_default_mode_is_auto_and_follows_the_binary() -> None:
    from apeGmsh.opensees._target import BackendInfo

    target = OpenSeesTarget()
    assert target.mode == "auto"
    assert target.require_fork is False
    fork = BackendInfo(kind="fork", build=_SHA, version="3.7", source="x")
    stock = BackendInfo(kind="stock", build=None, version="3.7", source="y")
    assert target.resolve_kind(fork) == "fork"
    assert target.resolve_kind(stock) == "stock"


@pytest.mark.parametrize("mode", ["fork", "stock"])
def test_explicit_mode_pins_the_kind(mode: str) -> None:
    from apeGmsh.opensees._target import BackendInfo

    target = OpenSeesTarget(mode=mode)  # type: ignore[arg-type]
    for kind in ("fork", "stock"):
        info = BackendInfo(
            kind=kind,  # type: ignore[arg-type]
            build=_SHA if kind == "fork" else None,
            version=None, source="m",
        )
        assert target.resolve_kind(info) == mode


def test_require_fork_means_fork_and_fork_means_require_fork() -> None:
    # The bridge's live gate reads require_fork, so a fork pin must set it.
    assert OpenSeesTarget(require_fork=True).mode == "fork"
    assert OpenSeesTarget(mode="fork").require_fork is True
    assert OpenSeesTarget(require_fork=True) == OpenSeesTarget(mode="fork")
    assert OpenSeesTarget(mode="stock").require_fork is False


def test_contradictory_or_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        OpenSeesTarget(require_fork=True, mode="stock")
    with pytest.raises(ValueError, match="mode must be one of"):
        OpenSeesTarget(mode="ladruno")  # type: ignore[arg-type]


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
    # has_fork is the resolver's verdict, the one the live emitter gates on
    from apeGmsh.opensees.emitter.live import get_backend_name

    assert caps.has_fork == (get_backend_name() == "ladruno-fork")
    # build: the exact git hash of the engine binary on fork builds shipping
    # ladrunoBuild (fork PR #718); None on stock openseespy or an older fork.
    assert caps.build is None or (
        isinstance(caps.build, str) and len(caps.build) == 40
    )
    if caps.build is not None:
        assert caps.has_fork, "only fork builds expose a build stamp"
