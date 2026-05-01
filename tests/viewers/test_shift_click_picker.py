"""ShiftClickPicker — observer + filtering tests.

The picker is a thin VTK observer; we exercise it by simulating the
``LeftButtonPressEvent`` callback with stub objects that mimic the
parts of ``vtkRenderWindowInteractor`` it queries (``GetShiftKey`` /
``GetEventPosition``). The picker mutates a real ``vtk.vtkCellPicker``
inside, so we patch that out with a stub recorder.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_picker(*, plotter, on_shift_pick):
    from apeGmsh.viewers.overlays.shift_click_picker import ShiftClickPicker
    return ShiftClickPicker(plotter, on_shift_pick)


def _stub_caller(*, shift: bool, x: int = 100, y: int = 100):
    caller = MagicMock()
    caller.GetShiftKey.return_value = 1 if shift else 0
    caller.GetEventPosition.return_value = (x, y)
    return caller


def _stub_plotter():
    """Minimal plotter mimic — has .iren and .renderer."""
    iren = MagicMock()
    iren.AddObserver.return_value = 7
    plotter = MagicMock()
    plotter.iren = iren
    plotter.iren.interactor = iren  # Picker accepts either form.
    plotter.renderer = MagicMock()
    return plotter, iren


# =====================================================================
# Plain clicks fall through
# =====================================================================

def test_plain_click_does_not_invoke_callback():
    plotter, _ = _stub_plotter()
    seen = []
    pk = _make_picker(
        plotter=plotter, on_shift_pick=lambda w: seen.append(w),
    )
    caller = _stub_caller(shift=False)
    pk._on_press(caller, "LeftButtonPressEvent")
    assert seen == []


# =====================================================================
# Shift-click invokes the callback when a cell is hit
# =====================================================================

def test_shift_click_with_hit_invokes_callback():
    plotter, _ = _stub_plotter()
    seen = []
    pk = _make_picker(
        plotter=plotter, on_shift_pick=lambda w: seen.append(w),
    )
    pk._picker = MagicMock()
    pk._picker.GetCellId.return_value = 5
    pk._picker.GetPickPosition.return_value = (1.0, 2.0, 3.0)

    pk._on_press(_stub_caller(shift=True), "LeftButtonPressEvent")
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], np.array([1.0, 2.0, 3.0]))


# =====================================================================
# Shift-click that misses geometry (cell_id = -1) is suppressed
# =====================================================================

def test_shift_click_miss_does_not_invoke_callback():
    plotter, _ = _stub_plotter()
    seen = []
    pk = _make_picker(
        plotter=plotter, on_shift_pick=lambda w: seen.append(w),
    )
    pk._picker = MagicMock()
    pk._picker.GetCellId.return_value = -1
    pk._picker.GetPickPosition.return_value = (0.0, 0.0, 0.0)

    pk._on_press(_stub_caller(shift=True), "LeftButtonPressEvent")
    assert seen == []


# =====================================================================
# Detach removes the observer
# =====================================================================

def test_detach_removes_observer():
    plotter, iren = _stub_plotter()
    pk = _make_picker(plotter=plotter, on_shift_pick=lambda w: None)
    iren.AddObserver.assert_called_once()
    pk.detach()
    iren.RemoveObserver.assert_called_once_with(7)


def test_detach_is_idempotent():
    plotter, iren = _stub_plotter()
    pk = _make_picker(plotter=plotter, on_shift_pick=lambda w: None)
    pk.detach()
    pk.detach()
    # Only one removal even though detach was called twice.
    assert iren.RemoveObserver.call_count == 1


# =====================================================================
# Callback exceptions are caught (don't crash the VTK loop)
# =====================================================================

def test_callback_exception_is_swallowed(capsys):
    plotter, _ = _stub_plotter()
    pk = _make_picker(
        plotter=plotter,
        on_shift_pick=lambda w: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    pk._picker = MagicMock()
    pk._picker.GetCellId.return_value = 1
    pk._picker.GetPickPosition.return_value = (0.0, 0.0, 0.0)
    # Must not raise.
    pk._on_press(_stub_caller(shift=True), "LeftButtonPressEvent")
    captured = capsys.readouterr()
    assert "callback raised" in captured.err
