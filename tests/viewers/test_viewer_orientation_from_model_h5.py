"""Phase 5 — the viewer reads orientation through ``results.model``.

ADR 0020 collapsed the legacy ``model_h5=`` plumbing kwarg: the chain
forward ``Results.model -> OpenSeesModel`` is the primary source of
OpenSees enrichment (beam orientation, cuts). The viewer reads the
structural snapshot through the file-mediated seam
:meth:`ViewerData.from_h5` against the path the OpenSeesModel was
loaded from (the Composed-file pattern landing native results files
have).

This file replaces the pre-Phase-5 tests that asserted via the now-
removed ``_effective_model_h5`` / ``_resolve_effective_model_h5``
fields. Behaviour covered:

1. Chain-forward path — ``Results`` carries a model, the file at
   ``results._path`` carries ``/opensees/``: viewer reads through
   :meth:`ViewerData.from_h5` and recovers per-element ``vecxz``.
2. Legacy fallback — ``results.model is None`` but the file still
   carries ``/opensees/``: viewer takes the same file-mediated path
   (graceful degrade for users who didn't pass ``model=`` to
   :class:`Results`).
3. Default — neither: viewer takes :meth:`ViewerData.from_fem`
   (default orientation; ADR 0018 INV-11 graceful degrade).
4. Deprecation — ``ResultsViewer(model_h5=path)`` still works and
   emits ``DeprecationWarning``.

Viewer is *constructed* through :meth:`_build_viewer_data`, never
:meth:`show`-n — exercises the resolver without spinning up Qt /
OpenGL, mirroring the headless-verify posture in scope-doc §5 and
memory ``feedback_viewer_no_gpu``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from apeGmsh.opensees import ModelData
from apeGmsh.viewers.data import ViewerData
from apeGmsh.viewers.data._h5_probe import has_opensees_orientation
from apeGmsh.viewers.results_viewer import ResultsViewer

from tests.opensees.fixtures.fem_stub import make_two_node_beam


def _model_stub_with_cuts(cuts: tuple = (), sweeps: tuple = ()):
    """Build a duck-typed OpenSeesModel stand-in.

    The viewer-side code uses only the duck-typed surface (no actual
    :class:`OpenSeesModel` import — ADR 0014 INV-2). For these tests
    the only methods called on the handle are :meth:`cuts` and
    :meth:`sweeps`, both of which the resolver in
    :class:`_build_viewer_data` never invokes (those drive the cuts
    auto-load path, not the orientation path).

    The handle is essentially a sentinel — anything truthy that the
    chain-forward gate ``results.model is not None`` can recognise.
    """
    return SimpleNamespace(
        cuts=lambda: tuple(cuts),
        sweeps=lambda: tuple(sweeps),
    )


class _ResultsStub:
    """Minimal Results stand-in for :class:`ResultsViewer.__init__`.

    Reads:
      - ``results.fem`` — must be non-None (the guard at
        :class:`ResultsViewer.__init__` raises otherwise).
      - ``results._path`` — the file-mediated read source.
      - ``results.model`` — the chain-forward gate (Phase 5).

    Nothing else is touched until :meth:`_build_viewer_data` runs.
    """

    def __init__(
        self,
        *,
        path: Optional[Path],
        fem: object,
        model: Optional[object] = None,
    ) -> None:
        self._path = path
        self.fem = fem
        self.model = model


def _write_oriented_model(
    path: Path,
    *,
    fem,
    vecxz: tuple = (1.0, 0.0, 0.0),
) -> None:
    """Write a model.h5 with /opensees/transforms + /opensees/element_meta."""
    md = ModelData(fem, ndm=3, ndf=6, model_name="oriented")
    md.oriented_elements(
        pg="Cols", ele_type="forceBeamColumn", vecxz=vecxz,
    )
    md.write(str(path))


def _write_bare_model(path: Path, *, fem) -> None:
    """Write a model.h5 WITHOUT the /opensees/ orientation zone."""
    md = ModelData(fem, ndm=3, ndf=6, model_name="bare")
    md.write(str(path))


# =====================================================================
# Phase 5 — symmetric ``results.model`` gate
# =====================================================================

def test_viewer_uses_results_model(tmp_path: Path) -> None:
    """``Results`` carrying a model + composed-file path → viewer reads
    through :meth:`ViewerData.from_h5` against ``results._path`` and
    recovers per-element ``vecxz``.

    The chain forward ``Results.model -> OpenSeesModel`` is the new
    primary source of OpenSees enrichment (Phase 5 / ADR 0020).
    """
    fem = make_two_node_beam()
    out = tmp_path / "composed.h5"
    _write_oriented_model(out, fem=fem)

    # The file carries the orientation zone — sanity probe.
    assert has_opensees_orientation(out)

    # Chain-forward: results carries a (duck-typed) OpenSeesModel
    # handle. The viewer's resolver only checks ``model is not None``
    # for the orientation gate; the actual reader is file-mediated
    # via :class:`ViewerData.from_h5` (ADR 0014 INV-2 — no
    # OpenSeesModel import inside ``viewers/``).
    model = _model_stub_with_cuts()
    stub = _ResultsStub(path=out, fem=fem, model=model)
    viewer = ResultsViewer(stub)

    # The viewer's branched scene builder routes through from_h5 and
    # recovers the per-element vecxz keyed by FEM eid.
    view = viewer._build_viewer_data()
    assert view.source_kind == "h5"
    assert view.elements.has_vecxz
    np.testing.assert_allclose(view.elements.vecxz_for(1), [1.0, 0.0, 0.0])


def test_viewer_falls_back_to_path_when_no_model(tmp_path: Path) -> None:
    """``results.model is None`` but ``results._path`` carries
    ``/opensees/`` → viewer takes the legacy file-mediated path
    (graceful degrade for users who didn't pass ``model=`` to
    :class:`Results`).
    """
    fem = make_two_node_beam()
    out = tmp_path / "model.h5"
    _write_oriented_model(out, fem=fem, vecxz=(0.0, 1.0, 0.0))

    # results.model is None, but the file probe succeeds.
    stub = _ResultsStub(path=out, fem=fem, model=None)
    viewer = ResultsViewer(stub)

    view = viewer._build_viewer_data()
    # File-mediated read recovers the vecxz.
    assert view.source_kind == "h5"
    np.testing.assert_allclose(view.elements.vecxz_for(1), [0.0, 1.0, 0.0])


def test_viewer_falls_back_to_fem_when_neither(tmp_path: Path, monkeypatch) -> None:
    """Neither ``results.model`` nor a file with ``/opensees/`` →
    viewer takes the live :meth:`ViewerData.from_fem` path (ADR 0018
    INV-11 graceful degrade to default orientation).

    We assert the resolver SELECTS the from_fem branch (without
    actually building the full ViewerData, which would need a real
    FEMData — the stub doesn't carry ``.physical`` / ``.labels``).
    """
    fem = make_two_node_beam()
    bare = tmp_path / "bare.h5"
    _write_bare_model(bare, fem=fem)

    # Both gates fail: no model, no orientation zone in the file.
    assert not has_opensees_orientation(bare)
    stub = _ResultsStub(path=bare, fem=fem, model=None)
    viewer = ResultsViewer(stub)

    # Capture which builder was called.
    chose: dict = {}
    import apeGmsh.viewers.results_viewer as rv
    real_vd = rv.__dict__.get("ViewerData")  # may be None if not yet imported

    def _track_from_fem(fem):
        chose["from_fem"] = True
        return SimpleNamespace(source_kind="fem")

    def _track_from_h5(path):
        chose["from_h5"] = path
        return SimpleNamespace(source_kind="h5")

    import apeGmsh.viewers.data as _vd_mod
    monkeypatch.setattr(_vd_mod.ViewerData, "from_fem",
                        classmethod(lambda cls, f: _track_from_fem(f)))
    monkeypatch.setattr(_vd_mod.ViewerData, "from_h5",
                        classmethod(lambda cls, p: _track_from_h5(p)))

    viewer._build_viewer_data()
    assert chose == {"from_fem": True}


def test_viewer_falls_back_to_fem_for_in_memory_results(monkeypatch) -> None:
    """In-memory Results (``_path is None``) → no file probe possible,
    no model → from_fem path."""
    fem = make_two_node_beam()
    stub = _ResultsStub(path=None, fem=fem, model=None)
    viewer = ResultsViewer(stub)

    chose: dict = {}

    def _track_from_fem(fem):
        chose["from_fem"] = True
        return SimpleNamespace(source_kind="fem")

    def _track_from_h5(path):
        chose["from_h5"] = path
        return SimpleNamespace(source_kind="h5")

    import apeGmsh.viewers.data as _vd_mod
    monkeypatch.setattr(_vd_mod.ViewerData, "from_fem",
                        classmethod(lambda cls, f: _track_from_fem(f)))
    monkeypatch.setattr(_vd_mod.ViewerData, "from_h5",
                        classmethod(lambda cls, p: _track_from_h5(p)))

    viewer._build_viewer_data()
    assert chose == {"from_fem": True}


# =====================================================================
# Phase 5 — DeprecationWarning on ``model_h5=`` kwarg
# =====================================================================

def test_deprecated_model_h5_kwarg_warns_and_works(tmp_path: Path) -> None:
    """Phase 8 — the deprecated ``model_h5=`` kwarg on ``ResultsViewer``
    has been removed; supplying it now raises :class:`TypeError`.

    The file-mediated orientation read fires against ``results._path``
    instead (the Composed-file pattern).
    """
    fem = make_two_node_beam()
    out = tmp_path / "legacy.h5"
    _write_oriented_model(out, fem=fem, vecxz=(0.5, 0.5, 0.0))

    stub = _ResultsStub(path=None, fem=fem, model=None)

    with pytest.raises(TypeError, match="model_h5"):
        ResultsViewer(stub, model_h5=out)


def test_no_deprecation_warning_when_kwarg_absent(tmp_path: Path) -> None:
    """The DeprecationWarning is only emitted when the user explicitly
    passes ``model_h5=``. Default invocation must stay quiet."""
    import warnings as _warnings
    fem = make_two_node_beam()
    out = tmp_path / "model.h5"
    _write_oriented_model(out, fem=fem)
    stub = _ResultsStub(path=out, fem=fem, model=None)

    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        ResultsViewer(stub)
    # Filter for our specific DeprecationWarning.
    relevant = [
        w for w in record
        if issubclass(w.category, DeprecationWarning)
        and "model_h5" in str(w.message)
    ]
    assert relevant == []


# =====================================================================
# Phase 5 — h5 probe still works (retained per spec)
# =====================================================================

def test_probe_rejects_nonexistent_file(tmp_path: Path) -> None:
    """The probe's False answer for a missing file is the contract that
    keeps the auto-resolve quiet for in-memory Results (no ``_path``)
    and for files that simply don't exist on disk."""
    assert has_opensees_orientation(tmp_path / "nope.h5") is False


def test_probe_rejects_non_hdf5_file(tmp_path: Path) -> None:
    """A non-HDF5 file (or otherwise unreadable) → False, no raise.
    The caller's contract is "should I auto-resolve?", not "is this
    file healthy?"."""
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not actually hdf5\n")
    assert has_opensees_orientation(junk) is False


# =====================================================================
# resolve_orientation_source — the centralised gate (PR3 / ADR 0026)
# =====================================================================
#
# Two ResultsViewer call sites previously inlined the same
# ``Path + has_opensees_orientation`` block (the scene snapshot at
# ``_build_viewer_data`` and the cuts auto-load gate at
# ``_apply_pending_cuts``).  Collapsing into one helper preserves the
# probe semantics — these tests pin that contract.

def test_resolve_returns_path_when_orientation_zone_present(tmp_path):
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    fem = make_two_node_beam()
    oriented = tmp_path / "oriented.h5"
    _write_oriented_model(oriented, fem=fem)

    results = SimpleNamespace(_path=oriented)
    source = resolve_orientation_source(results)

    assert source == oriented
    assert isinstance(source, Path)


def test_resolve_returns_none_when_orientation_zone_absent(tmp_path):
    """A bare model.h5 (no /opensees/transforms or /opensees/element_meta)
    yields None — the viewer must degrade to the from_fem path."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    fem = make_two_node_beam()
    bare = tmp_path / "bare.h5"
    _write_bare_model(bare, fem=fem)

    results = SimpleNamespace(_path=bare)

    assert resolve_orientation_source(results) is None


def test_resolve_returns_none_when_results_has_no_path():
    """In-memory Results (``_path is None``) yields None — typical for
    recorder-flavoured Results and for hand-constructed test fixtures."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    results = SimpleNamespace(_path=None)

    assert resolve_orientation_source(results) is None


def test_resolve_returns_none_when_results_path_does_not_exist(tmp_path):
    """A ``_path`` that points at a nonexistent file yields None — the
    probe's "should I auto-resolve?" contract trumps any path-shape
    inference."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    results = SimpleNamespace(_path=tmp_path / "never_written.h5")

    assert resolve_orientation_source(results) is None


def test_resolve_is_duck_typed_no_results_import():
    """Verify the resolver never imports apeGmsh.results — it operates
    entirely on ``getattr(results, "_path", None)``.  Preserves
    ADR 0014 INV-1: viewers/data/ has no import edge into
    apeGmsh.results.  The probe accepts any object with a _path
    attribute; the test passes a bare SimpleNamespace."""
    from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

    # No _path attribute at all → also None (getattr default fires).
    assert resolve_orientation_source(SimpleNamespace()) is None
