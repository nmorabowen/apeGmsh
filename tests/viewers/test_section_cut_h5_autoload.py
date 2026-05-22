"""v4-4 tests — director ``load_cuts_from_h5`` and viewer auto-load wiring.

Two layers:

* **Director**: ``ResultsDirector.load_cuts_from_h5()`` reads
  ``/opensees/cuts/`` and ``/opensees/sweeps/`` from the bound source
  (Phase 5: ``OpenSeesModel.cuts()/sweeps()`` first, falling back to a
  bound ``model.h5`` file path), and dispatches to the existing
  ``add_section_cut*`` methods. Tests mock those dispatch methods so
  the test bed doesn't need a full bound Results / fem / scene.

* **Viewer**: ``ResultsViewer._apply_pending_cuts`` decides whether to
  apply the explicit ``cuts=`` kwarg (kwarg-wins) or auto-load from
  the persisted source. Phase 5 (ADR 0020) — auto-load gate is
  ``results.model is not None`` (symmetric with orientation
  auto-resolve, INV-5). Tests drive the method on a
  ``SimpleNamespace`` stub carrying just the attrs the method
  touches — that side-steps the full Qt + Results + FEM construction
  needed by a real ``ResultsViewer`` while still exercising the
  production code path.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import h5py
import pytest

from apeGmsh.cuts import SectionCutDef, SectionSweepDef, persist_to_h5
from apeGmsh.viewers.diagrams._director import ResultsDirector
from apeGmsh.viewers.results_viewer import ResultsViewer

from tests.fixtures.schema import OPENSEES_CURRENT


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _make_minimal_h5(path: Path, schema_version: str = OPENSEES_CURRENT) -> None:
    """Minimum the cuts reader needs: ``/meta/schema_version``."""
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs.create(
            "schema_version", schema_version,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def _make_h5_with_cuts(
    path: Path,
    cuts: list[SectionCutDef] | None = None,
    sweeps: list[SectionSweepDef] | None = None,
) -> None:
    """Write a minimal model.h5 then append cuts/sweeps via persist_to_h5."""
    _make_minimal_h5(path)
    persist_to_h5(path, cuts=cuts or [], sweeps=sweeps or [])


def _sample_cut(label: str = "cut", z: float = 1.0) -> SectionCutDef:
    return SectionCutDef(
        plane_point=(0.0, 0.0, z),
        plane_normal=(0.0, 0.0, 1.0),
        element_ids=(1,),
        label=label,
    )


# --------------------------------------------------------------------- #
# Director — load_cuts_from_h5
# --------------------------------------------------------------------- #
def test_director_load_cuts_from_h5_raises_without_source() -> None:
    """No cuts source bound (neither OpenSeesModel nor model.h5 path)
    → load_cuts_from_h5 raises."""
    # Bypass full ResultsDirector construction — we only need the
    # state and the method.
    director = ResultsDirector.__new__(ResultsDirector)
    director._model_h5 = None
    director._opensees_model = None
    with pytest.raises(RuntimeError, match="no cuts source bound"):
        ResultsDirector.load_cuts_from_h5(director)


def test_director_load_cuts_from_h5_walks_cuts_and_sweeps(
    tmp_path: Path,
) -> None:
    """Each persisted cut / sweep dispatches to the right director method.

    Mocks ``add_section_cut`` / ``add_section_cut_sweep`` so the test
    doesn't need a bound Results + scene + registry — only the dispatch
    logic of ``load_cuts_from_h5`` is under test here.

    Phase 5 — exercises the file-path fallback path
    (``_opensees_model is None``, ``_model_h5`` is set).
    """
    cut_a = _sample_cut(label="standalone A", z=1.0)
    cut_b = _sample_cut(label="standalone B", z=2.0)
    sweep_cuts = (
        _sample_cut(label="sweep 0", z=10.0),
        _sample_cut(label="sweep 1", z=20.0),
    )
    sweep = SectionSweepDef(cuts=sweep_cuts)
    path = tmp_path / "model.h5"
    _make_h5_with_cuts(path, cuts=[cut_a, cut_b], sweeps=[sweep])

    director = ResultsDirector.__new__(ResultsDirector)
    director._model_h5 = path
    director._opensees_model = None
    director._tag_map_cache = None  # tag_map property guard
    director.add_section_cut = MagicMock(return_value="diag")
    director.add_section_cut_sweep = MagicMock(return_value=["diag1", "diag2"])

    attached = ResultsDirector.load_cuts_from_h5(director)

    # add_section_cut called once per standalone cut in writer order
    assert director.add_section_cut.call_count == 2
    director.add_section_cut.assert_any_call(cut_a)
    director.add_section_cut.assert_any_call(cut_b)

    # add_section_cut_sweep called once per sweep
    assert director.add_section_cut_sweep.call_count == 1
    director.add_section_cut_sweep.assert_called_with(sweep)

    # Returned list flattens: 2 standalone diagrams + 2 sweep diagrams
    assert attached == ["diag", "diag", "diag1", "diag2"]


def test_director_load_cuts_from_h5_on_pre_v4_file_is_empty(
    tmp_path: Path,
) -> None:
    """In-window file with no /opensees/cuts/ → no calls, empty return.

    Per ADR 0023 the fixture must be inside the two-version reader
    window (2.7.x / 2.8.x); the test exercises "no cuts group" handling,
    not "pre-window file" handling.
    """
    path = tmp_path / "pre_v4.h5"
    _make_minimal_h5(path, schema_version=OPENSEES_CURRENT)

    director = ResultsDirector.__new__(ResultsDirector)
    director._model_h5 = path
    director._opensees_model = None
    director._tag_map_cache = None
    director.add_section_cut = MagicMock()
    director.add_section_cut_sweep = MagicMock()

    attached = ResultsDirector.load_cuts_from_h5(director)

    assert attached == []
    director.add_section_cut.assert_not_called()
    director.add_section_cut_sweep.assert_not_called()


def test_director_load_cuts_from_opensees_model_preferred(
    tmp_path: Path,
) -> None:
    """Phase 5 — when an OpenSeesModel is bound via :meth:`set_model`,
    its ``cuts()`` / ``sweeps()`` accessors are the cuts source (not
    the file). The director still requires a ``_model_h5`` path
    *somewhere* for the tag_map build, but the cuts iteration itself
    routes through the chain-forward handle."""
    cut_a = _sample_cut(label="from-model A", z=1.0)
    cut_b = _sample_cut(label="from-model B", z=2.0)
    sweep = SectionSweepDef(cuts=(_sample_cut("sweep 0", z=10.0),))

    # Stub OpenSeesModel — duck-typed; the director never imports it.
    model = SimpleNamespace(
        cuts=MagicMock(return_value=(cut_a, cut_b)),
        sweeps=MagicMock(return_value=(sweep,)),
    )
    director = ResultsDirector.__new__(ResultsDirector)
    director._model_h5 = None  # no fallback path bound
    director._opensees_model = model
    director._tag_map_cache = None
    director.add_section_cut = MagicMock(return_value="diag")
    director.add_section_cut_sweep = MagicMock(return_value=["sd"])

    attached = ResultsDirector.load_cuts_from_h5(director)
    # The file at ``_model_h5`` was never read — the cuts came from
    # the OpenSeesModel handle.
    model.cuts.assert_called_once_with()
    model.sweeps.assert_called_once_with()
    assert director.add_section_cut.call_count == 2
    director.add_section_cut_sweep.assert_called_once_with(sweep)
    assert attached == ["diag", "diag", "sd"]


def test_director_internal_bind_model_h5_does_not_warn(tmp_path: Path) -> None:
    """Phase 8 — ``set_model_h5`` (the deprecated public verb) is gone.

    The internal ``_bind_model_h5`` helper still exists (used by
    viewer-internal callers); it must NOT emit a DeprecationWarning.
    """
    path = tmp_path / "model.h5"
    director = ResultsDirector.__new__(ResultsDirector)
    director._model_h5 = None
    director._opensees_model = None
    director._tag_map_cache = None

    # The internal helper does NOT emit.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning → error → test fails
        ResultsDirector._bind_model_h5(director, path)
    assert director._model_h5 == path


# --------------------------------------------------------------------- #
# Viewer — auto-load + kwarg-wins (Phase 5 INV-5)
# --------------------------------------------------------------------- #
def _make_orientation_h5(path: Path) -> None:
    """Minimal model.h5 carrying ``/opensees/transforms`` and
    ``/opensees/element_meta`` so :func:`has_opensees_orientation`
    returns True against it (gate for the Phase 5 results.model+path
    symmetric auto-load path)."""
    _make_minimal_h5(path)
    with h5py.File(path, "a") as f:
        f.create_group("opensees/transforms")
        f.create_group("opensees/element_meta")


def _viewer_stub(
    *,
    pending_cuts: tuple = (),
    legacy_model_h5: Path | None = None,
    results_path: Path | None = None,
    model_cuts: tuple = (),
    model_sweeps: tuple = (),
) -> SimpleNamespace:
    """Build the minimal attribute surface ``_apply_pending_cuts`` reads.

    Phase 8 (ADR 0020 INV-1) — the method now reads:
    * ``self._director``
    * ``self._pending_cuts``
    * ``self._results._path``, ``self._results.model`` (always set
      post-prune; the chain-forward gate INV-5 evaluates against
      ``.cuts()`` / ``.sweeps()`` on the model)

    ``model_cuts`` / ``model_sweeps`` populate a stub OpenSeesModel
    handle on ``self._results.model`` so the symmetric auto-load
    gate evaluates as if a real broker were bound.  ``legacy_model_h5``
    is no longer carried by ``ResultsViewer`` (Phase 8 prune) and is
    retained on the stub only for call-site compatibility.
    """
    director_mock = MagicMock()
    # Phase 8 — model is always non-None.  Build an empty stub when
    # the caller didn't supply cuts/sweeps.
    model = SimpleNamespace(
        cuts=MagicMock(return_value=tuple(model_cuts)),
        sweeps=MagicMock(return_value=tuple(model_sweeps)),
    )
    results = SimpleNamespace(_path=results_path, model=model)
    return SimpleNamespace(
        _director=director_mock,
        _pending_cuts=pending_cuts,
        _legacy_model_h5=legacy_model_h5,
        _results=results,
    )


def test_viewer_autoload_when_results_model_has_cuts(tmp_path: Path) -> None:
    """Phase 5 INV-5 — ``results.model`` carries cuts AND ``results._path``
    carries ``/opensees/`` → load_cuts_from_h5 fires symmetric with
    the orientation auto-resolve in ``_build_viewer_data``."""
    path = tmp_path / "model.h5"
    _make_orientation_h5(path)
    cut = _sample_cut(label="from-model", z=1.0)
    stub = _viewer_stub(
        pending_cuts=(),
        results_path=path,
        model_cuts=(cut,),
    )
    ResultsViewer._apply_pending_cuts(stub)

    # The director was wired with the chain-forward handle and
    # the path (for tag_map). Auto-load fired.
    stub._director.set_model.assert_called_once_with(stub._results.model)
    stub._director._bind_model_h5.assert_called_once_with(path)
    stub._director.load_cuts_from_h5.assert_called_once_with()
    stub._director.add_section_cut.assert_not_called()


def test_viewer_no_autoload_when_results_model_has_no_cuts(
    tmp_path: Path,
) -> None:
    """``results.model`` is bound but ``.cuts()/.sweeps()`` are empty
    → no auto-load (the symmetric INV-5 gate also checks non-empty)."""
    path = tmp_path / "model.h5"
    _make_orientation_h5(path)
    stub = _viewer_stub(
        pending_cuts=(),
        results_path=path,
        model_cuts=(),
        model_sweeps=(),
    )
    # Make _results.model non-None by injecting an empty stub.
    stub._results.model = SimpleNamespace(
        cuts=MagicMock(return_value=()),
        sweeps=MagicMock(return_value=()),
    )
    ResultsViewer._apply_pending_cuts(stub)
    stub._director.load_cuts_from_h5.assert_not_called()
    stub._director.add_section_cut.assert_not_called()


def test_viewer_legacy_model_h5_kwarg_still_autoloads(tmp_path: Path) -> None:
    """Phase 8 — the deprecated ``model_h5=`` kwarg is gone.

    The corresponding autoload path was removed; this test stays as a
    placeholder asserting that the no-orientation no-model code path
    is a noop (cuts/sweeps source absent → no director calls fire).
    """
    stub = _viewer_stub(pending_cuts=())
    ResultsViewer._apply_pending_cuts(stub)
    stub._director.add_section_cut.assert_not_called()
    stub._director.load_cuts_from_h5.assert_not_called()


def test_viewer_kwarg_wins_when_cuts_supplied(tmp_path: Path) -> None:
    """Explicit ``cuts=[c]`` suppresses h5 auto-load (kwarg-wins, H14)."""
    path = tmp_path / "model.h5"
    _make_orientation_h5(path)
    cut = _sample_cut(label="explicit", z=5.0)
    autoload_cut = _sample_cut(label="autoload-target", z=99.0)
    stub = _viewer_stub(
        pending_cuts=(cut,),
        results_path=path,
        model_cuts=(autoload_cut,),
    )
    ResultsViewer._apply_pending_cuts(stub)

    # Chain-forward bind still fires (orientation source).
    stub._director.set_model.assert_called_once_with(stub._results.model)
    stub._director._bind_model_h5.assert_called_once_with(path)
    # But explicit cuts go through add_section_cut, and load_cuts_from_h5
    # is NOT called (H14 — kwarg wins).
    stub._director.add_section_cut.assert_called_once_with(cut)
    stub._director.load_cuts_from_h5.assert_not_called()


def test_viewer_noop_when_no_cuts_and_no_source() -> None:
    """No cuts kwarg, no model, no orientation source → no director calls."""
    stub = _viewer_stub(pending_cuts=())
    ResultsViewer._apply_pending_cuts(stub)

    stub._director.set_model.assert_not_called()
    stub._director._bind_model_h5.assert_not_called()
    stub._director.add_section_cut.assert_not_called()
    stub._director.load_cuts_from_h5.assert_not_called()


def test_viewer_kwarg_cuts_without_h5() -> None:
    """Cuts kwarg supplied but no orientation source → cuts apply, no h5 bind.

    Phase 8 — ``set_model(results.model)`` always fires (the model is
    always non-None post-prune); the orientation-binding path
    (``_bind_model_h5``) only fires when ``results._path`` carries
    ``/opensees/``.
    """
    cut = _sample_cut(label="standalone", z=3.0)
    stub = _viewer_stub(pending_cuts=(cut,))
    ResultsViewer._apply_pending_cuts(stub)

    # set_model always fires post-prune (model is the chain-forward
    # source).
    stub._director.set_model.assert_called_once_with(stub._results.model)
    stub._director._bind_model_h5.assert_not_called()
    stub._director.add_section_cut.assert_called_once_with(cut)
    stub._director.load_cuts_from_h5.assert_not_called()


def test_viewer_autoload_swallows_director_errors(tmp_path: Path) -> None:
    """``load_cuts_from_h5`` failure is logged, not propagated.

    Phase 8 — exercise the chain-forward model-mediated autoload path
    (results_path carries /opensees/ + model.cuts() non-empty) and
    assert a director.load_cuts_from_h5 raise is swallowed.
    """
    path = tmp_path / "model.h5"
    _make_orientation_h5(path)
    cut = _sample_cut(label="will-fail", z=1.0)
    stub = _viewer_stub(
        pending_cuts=(),
        results_path=path,
        model_cuts=(cut,),
    )
    stub._director.load_cuts_from_h5.side_effect = FileNotFoundError(
        "intentional",
    )
    # Must not raise.
    ResultsViewer._apply_pending_cuts(stub)
    stub._director.load_cuts_from_h5.assert_called_once_with()


def test_viewer_clears_pending_cuts_queue(tmp_path: Path) -> None:
    """Calling _apply_pending_cuts twice → cuts NOT double-applied."""
    cut = _sample_cut(label="once", z=1.0)
    stub = _viewer_stub(pending_cuts=(cut,))
    ResultsViewer._apply_pending_cuts(stub)
    ResultsViewer._apply_pending_cuts(stub)

    # add_section_cut called once total, despite two _apply invocations.
    assert stub._director.add_section_cut.call_count == 1
