"""BackendInfo: one fork signal (``ladrunoBuild()`` returning a sha).

Fork-free: every case classifies a fake module, never the real (maybe
stale) build. The oracle for each is the slice card's (#1498):

* ``ladrunoBuild`` returning a sha → fork, with that sha;
* ``criticalTimeStep`` without ``ladrunoBuild`` (a fork build predating
  fork PR #718) → stock;
* a ``ladrunoBuild`` that raises → stock with ``build=None``, no exception;
* an empty or non-sha build string → ``None``;
* the resolver names the module actually imported, even when a ``.pth``
  aliases ``openseespy.opensees`` to the fork;
* the resolver's cache follows the bound module;
* a native ``results.h5`` carries the stamp as root attrs, absent when
  unknown.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from apeGmsh.opensees._target import BackendInfo, backend_info_of
from apeGmsh.opensees.emitter import live

_SHA = "a240b9183c0ffee0123456789abcdef012345678"


@pytest.fixture(autouse=True)
def _restore_resolver_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test leaves the resolver's caches as it found them."""
    monkeypatch.setattr(live, "_OPS_CACHE", live._OPS_CACHE)
    monkeypatch.setattr(live, "_BACKEND_INFO", live._BACKEND_INFO)


def _fake(
    name: str = "opensees", *, build: Any = None, raises: bool = False,
    critical: bool = False, file: str | None = None,
) -> ModuleType:
    """An OpenSees-shaped module (wipe / model / element)."""
    mod = ModuleType(name)
    for cmd in ("wipe", "model", "element"):
        setattr(mod, cmd, lambda *a: None)
    mod.version = lambda: "3.7.1"  # type: ignore[attr-defined]
    if critical:
        mod.criticalTimeStep = lambda: 1.0  # type: ignore[attr-defined]
    if raises:
        def _boom() -> str:
            raise RuntimeError("See stderr output")
        mod.ladrunoBuild = _boom  # type: ignore[attr-defined]
    elif build is not None:
        mod.ladrunoBuild = lambda: build  # type: ignore[attr-defined]
    if file is not None:
        mod.__file__ = file
    return mod


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------
def test_ladruno_build_sha_is_fork_with_that_sha() -> None:
    info = backend_info_of(_fake(build=_SHA, critical=True))
    assert info == BackendInfo(
        kind="fork", build=_SHA, version="3.7.1", source="opensees",
    )


def test_critical_time_step_alone_is_stock() -> None:
    # A fork build predating ladrunoBuild (fork PR #718) reads as stock.
    info = backend_info_of(_fake(critical=True))
    assert (info.kind, info.build) == ("stock", None)


def test_raising_ladruno_build_is_stock_not_an_exception() -> None:
    info = backend_info_of(_fake(raises=True, critical=True))
    assert (info.kind, info.build) == ("stock", None)


@pytest.mark.parametrize("raw", ["", "   ", "unknown", _SHA[:9], 7, None])
def test_degenerate_build_is_none(raw: Any) -> None:
    mod = _fake()
    mod.ladrunoBuild = lambda: raw  # type: ignore[attr-defined]
    info = backend_info_of(mod)
    assert (info.kind, info.build) == ("stock", None)


def test_padded_sha_is_stripped() -> None:
    assert backend_info_of(_fake(build=f" {_SHA}\n")).build == _SHA


def test_version_that_raises_is_none() -> None:
    mod = _fake(build=_SHA)

    def _boom() -> str:
        raise RuntimeError("no version")
    mod.version = _boom  # type: ignore[attr-defined]
    assert backend_info_of(mod).version is None


def test_source_is_the_file_actually_imported() -> None:
    mod = _fake(build=_SHA, file="C:/Ladruno/bin/opensees.pyd")
    assert backend_info_of(mod).source == "C:/Ladruno/bin/opensees.pyd"


# --------------------------------------------------------------------------
# The resolver owns it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("build, name, stamp", [
    (_SHA, "ladruno-fork", _SHA),
    (None, "stock-openseespy", None),
])
def test_name_build_and_info_agree(
    monkeypatch: pytest.MonkeyPatch, build: str | None, name: str,
    stamp: str | None,
) -> None:
    mod = _fake(build=build, critical=True)
    monkeypatch.setattr(live, "_get_ops", lambda: mod)
    assert live.get_backend_name() == name
    assert live.get_backend_build() == stamp
    assert live.get_backend_info().build == stamp


def test_info_follows_the_bound_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fork, stock = _fake(build=_SHA), _fake()
    bound = [fork]
    monkeypatch.setattr(live, "_get_ops", lambda: bound[0])
    monkeypatch.setattr(live, "_BACKEND_INFO", None)
    assert live.get_backend_info().kind == "fork"
    bound[0] = stock            # a re-resolve binds another module
    assert live.get_backend_info().kind == "stock"


def test_cache_reset_with_the_ops_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resetting _OPS_CACHE (the reload seam) drops the cached verdict too.
    old, new = _fake(build=_SHA), _fake()
    monkeypatch.setattr(live, "_OPS_CACHE", None)
    monkeypatch.setattr(live, "_BACKEND_INFO", (old, backend_info_of(old)))
    monkeypatch.setattr(live, "_resolve_ops", lambda: new)
    assert live._get_ops() is new
    assert live._BACKEND_INFO is None
    assert live.get_backend_info().kind == "stock"


def test_pth_alias_reports_the_module_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fork's .pth may bind ``openseespy.opensees`` to the fork module:
    # the stock loader then returns the fork, and the verdict says so.
    fork = _fake(
        "openseespy.opensees", build=_SHA,
        file="C:/Program Files/Ladruno/OpenSees/bin/opensees.pyd",
    )
    pkg = ModuleType("openseespy")
    pkg.opensees = fork  # type: ignore[attr-defined]
    monkeypatch.delenv("APEGMSH_OPENSEES_BIN", raising=False)
    monkeypatch.delenv("LADRUNO_OPENSEES_QUIET", raising=False)
    monkeypatch.setitem(sys.modules, "opensees", None)  # bare fork absent
    monkeypatch.setitem(sys.modules, "openseespy", pkg)
    monkeypatch.setitem(sys.modules, "openseespy.opensees", fork)
    monkeypatch.setattr(live, "_OPS_CACHE", None)
    monkeypatch.setattr(live, "_BACKEND_INFO", None)

    info = live.get_backend_info()
    assert live._get_ops() is fork
    assert (info.kind, info.build) == ("fork", _SHA)
    assert info.source == fork.__file__
    assert live.get_backend_name() == "ladruno-fork"


# --------------------------------------------------------------------------
# results.h5 root attrs
# --------------------------------------------------------------------------
def _open_attrs(path: Path, **kw: Any) -> dict[str, Any]:
    import h5py

    from apeGmsh.results.writers._native import NativeWriter

    with NativeWriter(path) as w:
        w.open(**kw)
        # stamped with the header: present before any stage data lands
        assert w._h5 is not None
        before = dict(w._h5.attrs)
    with h5py.File(path, "r") as f:
        assert dict(f.attrs).keys() >= before.keys()
        return {k: v for k, v in f.attrs.items()}


def test_results_h5_carries_the_fork_stamp(tmp_path: Path) -> None:
    attrs = _open_attrs(
        tmp_path / "r.h5", opensees_backend="fork", opensees_build=_SHA,
    )
    assert attrs["opensees_backend"] == "fork"
    assert attrs["opensees_build"] == _SHA


def test_results_h5_stock_has_no_build(tmp_path: Path) -> None:
    attrs = _open_attrs(tmp_path / "r.h5", opensees_backend="stock")
    assert attrs["opensees_backend"] == "stock"
    assert "opensees_build" not in attrs


@pytest.mark.parametrize("build", [None, ""])
def test_results_h5_unknown_writes_nothing(tmp_path: Path, build: Any) -> None:
    attrs = _open_attrs(tmp_path / "r.h5", opensees_build=build)
    assert "opensees_backend" not in attrs
    assert "opensees_build" not in attrs


def test_results_h5_refuses_unknown_kind_and_stock_build(tmp_path: Path) -> None:
    from apeGmsh.results.writers._native import NativeWriter

    with pytest.raises(ValueError, match="expected one of"):
        NativeWriter(tmp_path / "a.h5").open(opensees_backend="ladruno")
    with pytest.raises(ValueError, match="only a fork build"):
        NativeWriter(tmp_path / "b.h5").open(
            opensees_backend="stock", opensees_build=_SHA,
        )
    assert not (tmp_path / "a.h5").exists()
    assert not (tmp_path / "b.h5").exists()


# --------------------------------------------------------------------------
# The writer's callers and the other backend tags read the same signal
# --------------------------------------------------------------------------
def _capture_attrs(tmp_path: Path, ops: Any) -> dict[str, Any]:
    import h5py

    from apeGmsh.results.capture._domain import DomainCapture
    from tests.test_results_domain_capture import _make_spec, _MockFem

    fem = _MockFem([1, 2])
    path = tmp_path / "cap.h5"
    with DomainCapture(
        _make_spec(snapshot_id=fem.snapshot_id), path, fem, ops=ops,
    ):
        pass
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


def test_domain_capture_stamps_the_fork_it_samples(tmp_path: Path) -> None:
    attrs = _capture_attrs(tmp_path, _fake(build=_SHA))
    assert attrs["opensees_backend"] == "fork"
    assert attrs["opensees_build"] == _SHA


def test_domain_capture_unstamped_fork_is_stock(tmp_path: Path) -> None:
    attrs = _capture_attrs(tmp_path, _fake(critical=True))
    assert attrs["opensees_backend"] == "stock"
    assert "opensees_build" not in attrs


def test_strut_tie_backend_tag_uses_the_one_signal() -> None:
    from apeGmsh.interop.strut_tie import _backend_tag

    assert _backend_tag(_fake(build=_SHA)) == "ladruno-fork"
    assert _backend_tag(_fake(critical=True)) == "stock-openseespy"
