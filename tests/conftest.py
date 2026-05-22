"""Shared test fixtures for apeGmsh test suite."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import gmsh


@pytest.fixture
def gmsh_session():
    """Bare Gmsh session for low-level tests (Labels, helpers)."""
    gmsh.initialize()
    gmsh.model.add("test")
    yield
    gmsh.finalize()


@pytest.fixture
def g():
    """Full apeGmsh session with all composites wired up."""
    from apeGmsh import apeGmsh
    session = apeGmsh(model_name="test", verbose=False)
    session.begin()
    yield session
    session.end()


# ---------------------------------------------------------------------------
# Phase 8 (ADR 0020 INV-1) — ``model=`` is required on every
# :class:`Results` constructor.  Tests use this helper to build an
# :class:`OpenSeesModel` from a file path (composed-file pattern: the
# results.h5 carries ``/opensees/`` and can be both the results AND
# the model source).
# ---------------------------------------------------------------------------


def _open_model_from_h5(path: "str | Path") -> Any:
    """Return an :class:`OpenSeesModel` loaded from ``path``.

    Auto-detects Composed-file shape (FEM under ``/model``) versus
    standalone model.h5 (FEM at root).  When ``path`` carries neither
    the rich neutral zone NOR a ``/opensees/`` bridge zone (e.g. a
    synthetic ``NativeWriter`` file opened with
    ``source_type="domain_capture"`` and no ``model_h5_src=``), this
    helper returns a tiny stub :class:`OpenSeesModel` so the Phase 8
    ``model=`` contract is satisfied; tests that don't read
    ``r.model`` won't notice the difference.

    Tests can call this helper directly when they need the broker
    without taking a fixture.  Imported lazily so test modules that
    don't touch OpenSees don't pay the import cost.
    """
    import h5py

    from apeGmsh.opensees import OpenSeesModel

    try:
        with h5py.File(str(path), "r") as f:
            has_model = "model" in f
            has_opensees = "opensees" in f
    except OSError:
        return _stub_opensees_model()

    if has_model and has_opensees:
        # Composed file: FEM neutral zone under /model.
        return OpenSeesModel.from_h5(path, fem_root="/model")
    if has_opensees:
        # Standalone model.h5: FEM at root.
        return OpenSeesModel.from_h5(path)
    # Synthetic results file with no model surface — return a stub.
    return _stub_opensees_model()


def _stub_opensees_model() -> Any:
    """Return a tiny stub :class:`OpenSeesModel` for tests that
    don't exercise the model surface.

    Built by direct dataclass construction (no h5 round-trip);
    every record collection is empty.  Tests pulling
    ``r.materials()``/etc. against this will surface empty tuples;
    tests that only need ``model=`` to satisfy the Phase 8 contract
    are unaffected.
    """
    from types import MappingProxyType, SimpleNamespace

    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.opensees._internal.lineage import Lineage

    fem = SimpleNamespace(
        snapshot_id="stub",
        nodes=SimpleNamespace(ids=(), coords=()),
        elements=SimpleNamespace(),
    )
    return OpenSeesModel(
        _fem=fem,
        _model_name="",
        _ndm=0,
        _ndf=0,
        _snapshot_id="stub",
        _materials_by_family=MappingProxyType({}),
        _sections=(),
        _transforms=(),
        _beam_integration=(),
        _time_series=(),
        _elements=(),
        _fixes=(),
        _masses=(),
        _patterns=(),
        _recorders=(),
        _analysis_attrs=MappingProxyType({}),
        _analyze_call=None,
        _cuts=(),
        _sweeps=(),
        _lineage=Lineage(),
    )


def _stub_model_h5_path() -> Path:
    """Return a path to a session-scoped stub ``model.h5`` file.

    Built once per session at a stable temp location so Phase 8
    ``model_h5=`` calls on MPCO tests that don't have a real
    sibling model can point at SOMETHING the loader can parse.
    """
    import os
    import tempfile

    cache_dir = Path(tempfile.gettempdir()) / "apegmsh_stub_models"
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / "stub_model.h5"
    if path.is_file():
        return path

    # Build a minimal model.h5 by going through the same code path the
    # real fixtures use.  Anchored on a tiny FEMData (one node, one
    # line element) so the broker can rehydrate.
    from tests.opensees.h5._opensees_model_fixtures import (
        build_simple_frame_h5,
    )

    with tempfile.TemporaryDirectory() as td:
        built, _fem = build_simple_frame_h5(Path(td))
        # Copy to the stable location.
        import shutil
        shutil.copy(built, path)
    return path


@pytest.fixture(scope="session")
def stub_model_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped stub ``model.h5`` for tests that need
    ``model_h5=`` but don't otherwise build one.
    """
    return _stub_model_h5_path()


def _open_model_or_none(path: "str | Path") -> "Any | None":
    """Return an :class:`OpenSeesModel` loaded from ``path``, or
    ``None`` when the file lacks the bridge ``/opensees/`` zone.

    Used by legacy fixtures that pre-date the Composed-file pattern.
    Tests that construct :class:`Results` against such files must
    either embed a fresh ``/opensees/`` zone (via NativeWriter's
    ``model_h5_src=``) or supply ``model=`` directly.
    """
    import h5py
    try:
        with h5py.File(str(path), "r") as f:
            if "opensees" not in f:
                return None
    except OSError:
        return None
    return _open_model_from_h5(path)
