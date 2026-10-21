"""``Results.from_ladruno`` / ``from_mpco`` fail closed and free the file.

Two contracts:

* A ``.ladruno`` with no ``MODEL`` group (no node geometry) gives the
  reader nothing to build an ``OpenSeesModel`` from. Without ``model_h5=``
  the open raises a ``ValueError`` that names the file and the missing
  group. The guard is a raise, so it holds under ``python -O`` too. With
  ``model_h5=`` the file opens and the archive supplies the model.
* A step after the reader opens that raises (``OpenSeesModel.from_h5``,
  the FEMData resolution, stage-name binding) closes the reader before
  the error propagates. The oracle is the file itself: while the
  traceback is still held, ``h5py.File(p, "r+")`` reopens every file the
  reader opened and ``os.remove`` deletes it (HDF5 refuses a second open
  of a file it holds open read-only, on every platform). The error that
  surfaces is the original object, and ``sys.unraisablehook`` stays
  silent (no ``__del__`` noise).
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import pytest

from apeGmsh.results import Results
from tests.test_results_ladruno_stage_binding import _capture, _model_h5, _ux

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "ladruno"


class _Boom(Exception):
    """The error a step after the open raises."""


class _Interrupt(BaseException):
    """A ``BaseException`` (as ``KeyboardInterrupt``) from the same step."""


def _drop_model(*paths: Path) -> None:
    for p in paths:
        with h5py.File(p, "r+") as f:
            for name in list(f.keys()):
                if name.startswith("MODEL_STAGE[") and "MODEL" in f[name]:
                    del f[f"{name}/MODEL"]


def _modelless_truss(tmp_path: Path, parts: bool) -> "tuple[Path, list[Path]]":
    """The committed fork capture (or its two-part split) with ``MODEL``
    removed; returns the path to open and every file it reads."""
    names = (["truss2d.part-0.ladruno", "truss2d.part-1.ladruno"]
             if parts else ["truss2d.ladruno"])
    files = []
    for n in names:
        shutil.copyfile(FIXTURES / n, tmp_path / n)
        files.append(tmp_path / n)
    _drop_model(*files)
    return files[0], files


def _assert_free(files: "list[Path]") -> None:
    for p in files:
        with h5py.File(p, "r+"):
            pass
        os.remove(p)
        if p.exists():
            raise AssertionError(f"{p} still exists after os.remove")


@pytest.fixture
def unraisable(monkeypatch: pytest.MonkeyPatch) -> "list[Any]":
    seen: "list[Any]" = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    return seen


# =====================================================================
# A model-less .ladruno
# =====================================================================

@pytest.mark.parametrize("parts", [False, True], ids=["file", "parts"])
def test_modelless_file_raises_a_named_error(
    tmp_path: Path, parts: bool, unraisable: "list[Any]",
) -> None:
    path, files = _modelless_truss(tmp_path, parts)
    with pytest.raises(ValueError) as ei:
        Results.from_ladruno(path)
    msg = str(ei.value)
    if path.name not in msg or "MODEL group" not in msg \
            or "model_h5=" not in msg:
        raise AssertionError(f"unnamed refusal: {msg!r}")
    _assert_free(files)
    del ei
    gc.collect()
    if unraisable:
        raise AssertionError(f"unraisable: {unraisable!r}")


def test_modelless_file_opens_with_model_h5(tmp_path: Path) -> None:
    path = _capture(tmp_path / "r.ladruno", "ladruno", ["static"])
    _drop_model(path)
    model_h5 = _model_h5(tmp_path / "model.h5", ("pull", "static"))
    with Results.from_ladruno(path, model_h5=model_h5) as r:
        if r.model.ndm != 2:
            raise AssertionError(r.model.ndm)
        # The program's stage name binds: the model came from model_h5.
        if [s.name for s in r.stages if s.kind != "mode"] != ["pull"]:
            raise AssertionError([s.name for s in r.stages])
        slab = r.stage("pull").nodes.get(component="displacement_x")
        last = [float(v) for v in slab.values[-1]]
        if last != [_ux(1, 1, n) for n in (1, 2, 3)]:
            raise AssertionError(last)


# =====================================================================
# A step after the open raises: the reader is closed, the error is the
# original
# =====================================================================

def _fail_from_h5(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    from apeGmsh.opensees.opensees_model import OpenSeesModel

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise exc
    monkeypatch.setattr(OpenSeesModel, "from_h5", _raise)


def _fail_resolve_fem(monkeypatch: pytest.MonkeyPatch,
                      exc: BaseException) -> None:
    results_mod = sys.modules["apeGmsh.results.Results"]

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise exc
    monkeypatch.setattr(results_mod, "_resolve_fem", _raise)
    monkeypatch.setattr(results_mod, "_resolve_fem_via_model", _raise)


def _fail_bind_stage_names(monkeypatch: pytest.MonkeyPatch,
                           exc: BaseException) -> None:
    results_mod = sys.modules["apeGmsh.results.Results"]

    def _raise(*_a: Any, **_k: Any) -> Any:
        raise exc
    monkeypatch.setattr(results_mod, "_bind_stage_names", _raise)


_STEPS = {
    "from_h5": _fail_from_h5,
    "resolve_fem": _fail_resolve_fem,
    "bind_stage_names": _fail_bind_stage_names,
}


def _ladruno_files(tmp_path: Path, parts: bool) -> "tuple[Path, list[Path]]":
    names = (["truss2d.part-0.ladruno", "truss2d.part-1.ladruno"]
             if parts else ["truss2d.ladruno"])
    files = []
    for n in names:
        shutil.copyfile(FIXTURES / n, tmp_path / n)
        files.append(tmp_path / n)
    return files[0], files


@pytest.mark.parametrize("exc_type", [_Boom, _Interrupt])
@pytest.mark.parametrize("parts", [False, True], ids=["file", "parts"])
@pytest.mark.parametrize("step", sorted(_STEPS))
def test_ladruno_failure_after_open_frees_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    unraisable: "list[Any]", step: str, parts: bool,
    exc_type: "type[BaseException]",
) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5", ("pull", "static"))
    path, files = _ladruno_files(tmp_path, parts)
    exc = exc_type("after the open")
    _STEPS[step](monkeypatch, exc)
    with pytest.raises(exc_type) as ei:
        Results.from_ladruno(path, model_h5=model_h5)
    if ei.value is not exc:
        raise AssertionError(f"a different error surfaced: {ei.value!r}")
    _assert_free(files)
    del ei
    gc.collect()
    if unraisable:
        raise AssertionError(f"unraisable: {unraisable!r}")


@pytest.mark.parametrize("exc_type", [_Boom, _Interrupt])
@pytest.mark.parametrize("step", ["from_h5", "bind_stage_names"])
def test_mpco_failure_after_open_frees_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    unraisable: "list[Any]", step: str, exc_type: "type[BaseException]",
) -> None:
    model_h5 = _model_h5(tmp_path / "model.h5", ("pull", "static"))
    path = _capture(tmp_path / "r.mpco", "mpco", [None])
    exc = exc_type("after the open")
    _STEPS[step](monkeypatch, exc)
    with pytest.raises(exc_type) as ei:
        Results.from_mpco(path, model_h5=model_h5)
    if ei.value is not exc:
        raise AssertionError(f"a different error surfaced: {ei.value!r}")
    _assert_free([path])
    del ei
    gc.collect()
    if unraisable:
        raise AssertionError(f"unraisable: {unraisable!r}")
