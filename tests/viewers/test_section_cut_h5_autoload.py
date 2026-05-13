"""v4-4 tests — director ``load_cuts_from_h5`` and viewer auto-load wiring.

Two layers:

* **Director**: ``ResultsDirector.load_cuts_from_h5()`` reads
  ``/opensees/cuts/`` and ``/opensees/sweeps/`` from the bound
  ``model.h5`` and dispatches to the existing
  ``add_section_cut*`` methods. Tests mock those dispatch methods so
  the test bed doesn't need a full bound Results / fem / scene.

* **Viewer**: ``ResultsViewer._apply_pending_cuts`` decides whether to
  apply the explicit ``cuts=`` kwarg (kwarg-wins) or auto-load from
  ``model.h5``. Tests drive the method on a ``SimpleNamespace`` stub
  carrying just the attrs the method touches — that side-steps the
  full Qt + Results + FEM construction needed by a real
  ``ResultsViewer`` while still exercising the production code path.
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


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _make_minimal_h5(path: Path, schema_version: str = "2.5.0") -> None:
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
def test_director_load_cuts_from_h5_raises_without_model_h5() -> None:
    """No model.h5 bound → load_cuts_from_h5 raises."""
    # Bypass full ResultsDirector construction — we only need the
    # state and the method.
    director = ResultsDirector.__new__(ResultsDirector)
    director._model_h5 = None
    with pytest.raises(RuntimeError, match="no model.h5 bound"):
        ResultsDirector.load_cuts_from_h5(director)


def test_director_load_cuts_from_h5_walks_cuts_and_sweeps(
    tmp_path: Path,
) -> None:
    """Each persisted cut / sweep dispatches to the right director method.

    Mocks ``add_section_cut`` / ``add_section_cut_sweep`` so the test
    doesn't need a bound Results + scene + registry — only the dispatch
    logic of ``load_cuts_from_h5`` is under test here.
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
    """File at 2.2.0 with no /opensees/cuts/ → no calls, empty return."""
    path = tmp_path / "pre_v4.h5"
    _make_minimal_h5(path, schema_version="2.2.0")

    director = ResultsDirector.__new__(ResultsDirector)
    director._model_h5 = path
    director._tag_map_cache = None
    director.add_section_cut = MagicMock()
    director.add_section_cut_sweep = MagicMock()

    attached = ResultsDirector.load_cuts_from_h5(director)

    assert attached == []
    director.add_section_cut.assert_not_called()
    director.add_section_cut_sweep.assert_not_called()


# --------------------------------------------------------------------- #
# Viewer — auto-load + kwarg-wins
# --------------------------------------------------------------------- #
def _viewer_stub(
    *,
    pending_cuts: tuple = (),
    pending_model_h5: Path | None = None,
) -> SimpleNamespace:
    """Build the minimal attribute surface ``_apply_pending_cuts`` reads.

    ``_apply_pending_cuts`` only touches ``self._director``,
    ``self._pending_cuts``, and ``self._pending_model_h5``; the rest
    of ResultsViewer is irrelevant to the dispatch decision.
    """
    director_mock = MagicMock()
    return SimpleNamespace(
        _director=director_mock,
        _pending_cuts=pending_cuts,
        _pending_model_h5=pending_model_h5,
    )


def test_viewer_autoload_when_cuts_kwarg_absent(tmp_path: Path) -> None:
    """``ResultsViewer(model_h5=p)`` with no ``cuts=`` → load_cuts_from_h5 fires."""
    stub = _viewer_stub(
        pending_cuts=(),
        pending_model_h5=tmp_path / "model.h5",
    )
    ResultsViewer._apply_pending_cuts(stub)

    stub._director.set_model_h5.assert_called_once_with(
        tmp_path / "model.h5",
    )
    stub._director.load_cuts_from_h5.assert_called_once_with()
    stub._director.add_section_cut.assert_not_called()


def test_viewer_kwarg_wins_when_cuts_supplied(tmp_path: Path) -> None:
    """Explicit ``cuts=[c]`` suppresses h5 auto-load (kwarg-wins, H14)."""
    cut = _sample_cut(label="explicit", z=5.0)
    stub = _viewer_stub(
        pending_cuts=(cut,),
        pending_model_h5=tmp_path / "model.h5",
    )
    ResultsViewer._apply_pending_cuts(stub)

    # model_h5 still bound — autoload guard checks _pending_cuts, not model_h5
    stub._director.set_model_h5.assert_called_once_with(
        tmp_path / "model.h5",
    )
    # But the kwarg cuts go through add_section_cut, and load_cuts_from_h5
    # is NOT called.
    stub._director.add_section_cut.assert_called_once_with(cut)
    stub._director.load_cuts_from_h5.assert_not_called()


def test_viewer_noop_when_no_cuts_and_no_h5() -> None:
    """No cuts kwarg, no model_h5 → no director calls."""
    stub = _viewer_stub(pending_cuts=(), pending_model_h5=None)
    ResultsViewer._apply_pending_cuts(stub)

    stub._director.set_model_h5.assert_not_called()
    stub._director.add_section_cut.assert_not_called()
    stub._director.load_cuts_from_h5.assert_not_called()


def test_viewer_kwarg_cuts_without_h5() -> None:
    """Cuts kwarg supplied but no model_h5 → cuts apply, no h5 binding."""
    cut = _sample_cut(label="standalone", z=3.0)
    stub = _viewer_stub(pending_cuts=(cut,), pending_model_h5=None)
    ResultsViewer._apply_pending_cuts(stub)

    stub._director.set_model_h5.assert_not_called()
    stub._director.add_section_cut.assert_called_once_with(cut)
    stub._director.load_cuts_from_h5.assert_not_called()


def test_viewer_autoload_swallows_director_errors(tmp_path: Path) -> None:
    """``load_cuts_from_h5`` failure is logged, not propagated.

    Mirrors the existing kwarg-cut error handling — a bad h5 must not
    prevent the rest of the viewer from opening.
    """
    stub = _viewer_stub(
        pending_cuts=(),
        pending_model_h5=tmp_path / "missing.h5",
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
    stub = _viewer_stub(
        pending_cuts=(cut,),
        pending_model_h5=tmp_path / "model.h5",
    )
    ResultsViewer._apply_pending_cuts(stub)
    ResultsViewer._apply_pending_cuts(stub)

    # add_section_cut called once total, despite two _apply invocations.
    assert stub._director.add_section_cut.call_count == 1
