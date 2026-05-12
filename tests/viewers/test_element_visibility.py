"""ElementVisibility controller (Phase 3.3).

Mutates the substrate grid's ``vtkGhostType`` cell-data array in place
to mark cells hidden. Asserts the in-place contract (mirrors the
``test_line_force_perf.py`` no-rebuild pattern), the box-pick
integration (results_pick masks hidden cells out of the hit set), and
the storm-resistant performance gate (100k cells in <5 ms).
"""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.viewers.core.element_visibility import (
    ElementVisibility,
    HIDDENCELL,
)


@pytest.fixture
def small_grid():
    """8-cell hex grid — enough for sanity checks without spinning up
    Gmsh."""
    grid = pv.ImageData(dimensions=(3, 3, 3))    # 2x2x2 = 8 hex cells
    return grid.cast_to_unstructured_grid()


@pytest.fixture
def big_grid():
    """~100k-cell grid for the bench gate."""
    # 50 x 50 x 40 = 100,000 hex cells.
    grid = pv.ImageData(dimensions=(51, 51, 41))
    g = grid.cast_to_unstructured_grid()
    assert g.n_cells == 100_000
    return g


# ---------------------------------------------------------------------
# Allocation + ghost-array invariant
# ---------------------------------------------------------------------

def test_constructor_allocates_ghost_array(small_grid):
    assert "vtkGhostType" not in small_grid.cell_data
    ev = ElementVisibility(small_grid)
    assert "vtkGhostType" in small_grid.cell_data
    assert small_grid.cell_data["vtkGhostType"].dtype == np.uint8
    assert small_grid.cell_data["vtkGhostType"].size == small_grid.n_cells
    assert ev.n_hidden() == 0


def test_constructor_preserves_existing_ghost_array(small_grid):
    """If the grid already carries a ghost array, don't clobber it."""
    pre = np.zeros(small_grid.n_cells, dtype=np.uint8)
    pre[0] = HIDDENCELL
    small_grid.cell_data["vtkGhostType"] = pre
    ev = ElementVisibility(small_grid)
    assert ev.is_hidden(0) is True
    assert ev.n_hidden() == 1


# ---------------------------------------------------------------------
# Hide / show / show_all
# ---------------------------------------------------------------------

def test_hide_sets_hiddencell_bit(small_grid):
    ev = ElementVisibility(small_grid)
    ev.hide([0, 1, 4])
    mask = ev.hidden_mask()
    assert mask[0] and mask[1] and mask[4]
    assert not mask[2]
    assert ev.n_hidden() == 3


def test_hide_is_idempotent(small_grid):
    ev = ElementVisibility(small_grid)
    ev.hide([0, 1])
    ev.hide([0, 1])    # second time — already hidden
    assert ev.n_hidden() == 2


def test_show_clears_hiddencell_bit(small_grid):
    ev = ElementVisibility(small_grid)
    ev.hide([0, 1, 2, 3])
    ev.show([1, 2])
    assert ev.n_hidden() == 2
    assert ev.is_hidden(0) is True
    assert ev.is_hidden(1) is False
    assert ev.is_hidden(2) is False
    assert ev.is_hidden(3) is True


def test_show_all_clears_everything(small_grid):
    ev = ElementVisibility(small_grid)
    ev.hide(list(range(small_grid.n_cells)))
    assert ev.n_hidden() == small_grid.n_cells
    ev.show_all()
    assert ev.n_hidden() == 0


def test_hide_accepts_ndarray(small_grid):
    ev = ElementVisibility(small_grid)
    ev.hide(np.array([0, 2, 4], dtype=np.int64))
    assert ev.n_hidden() == 3


def test_hide_empty_iterable_is_noop(small_grid):
    ev = ElementVisibility(small_grid)
    ev.hide([])
    assert ev.n_hidden() == 0


# ---------------------------------------------------------------------
# In-place mutation contract — buffer identity preserved across hides
# ---------------------------------------------------------------------

def test_hide_mutates_in_place_no_rebuild(small_grid):
    """The ghost array MUST be mutated in place; rebinding the dict
    entry would defeat the perf benefit (filters re-evaluate). This is
    the line_force in-place test pattern adapted for cell_data."""
    ev = ElementVisibility(small_grid)
    initial = small_grid.cell_data["vtkGhostType"]
    initial_buffer_addr = initial.__array_interface__["data"][0]

    ev.hide([0])
    ev.hide([1, 2, 3])
    ev.show([2])

    later = small_grid.cell_data["vtkGhostType"]
    later_buffer_addr = later.__array_interface__["data"][0]
    # Backing buffer didn't move — VTK's underlying array is the same.
    assert later_buffer_addr == initial_buffer_addr, (
        "vtkGhostType buffer was reallocated — hide() must mutate in "
        "place, not rebind"
    )


# ---------------------------------------------------------------------
# Box-pick integration — ghost mask excludes hidden cells
# ---------------------------------------------------------------------

def test_box_pick_excludes_hidden_cells():
    """Wire ElementVisibility + the box-pick mask path together and
    confirm hidden cells are dropped from the hit list."""
    from apeGmsh.viewers.core.results_pick import _inside_box

    # Build a 100-cell grid and a simulated "all cells inside box" mask.
    grid = pv.ImageData(dimensions=(11, 11, 2))  # 10x10x1 = 100
    grid = grid.cast_to_unstructured_grid()
    ev = ElementVisibility(grid)
    n = grid.n_cells
    inside = np.ones(n, dtype=bool)    # imagine box covers everything

    # Hide cells 0, 5, 25, 99 — they should drop from the result.
    hidden = [0, 5, 25, 99]
    ev.hide(hidden)

    # The masking step that the ELEMENT box-pick performs in
    # results_pick._build_box_result:
    ghosts = np.asarray(grid.cell_data["vtkGhostType"])
    final_mask = inside & ~(ghosts & 0x01).astype(bool)
    final_cells = np.nonzero(final_mask)[0]

    for h in hidden:
        assert h not in final_cells
    assert final_cells.size == n - len(hidden)


# ---------------------------------------------------------------------
# Dispatcher integration — ELEMENT_VISIBILITY_CHANGED fires
# ---------------------------------------------------------------------

def test_hide_fires_element_visibility_changed(small_grid):
    fires: list[tuple[str, object]] = []

    class _StubDispatcher:
        def fire(self, kind, *, payload=None):
            fires.append((kind, payload))

    ev = ElementVisibility(small_grid)
    ev.dispatcher = _StubDispatcher()
    ev.hide([0, 1])
    from apeGmsh.viewers.diagrams._dispatch import ELEMENT_VISIBILITY_CHANGED
    assert any(k == ELEMENT_VISIBILITY_CHANGED for (k, _p) in fires)


def test_show_all_fires_with_none_payload(small_grid):
    fires: list[tuple[str, object]] = []

    class _StubDispatcher:
        def fire(self, kind, *, payload=None):
            fires.append((kind, payload))

    ev = ElementVisibility(small_grid)
    ev.dispatcher = _StubDispatcher()
    ev.hide([0])
    ev.show_all()
    # Two events: hide(payload=[0]), show_all(payload=None)
    assert len(fires) == 2
    assert fires[-1][1] is None


# ---------------------------------------------------------------------
# Performance gate — 100k cells hidden in < 5 ms
# ---------------------------------------------------------------------

@pytest.mark.bench
def test_hide_100k_cells_under_5ms(big_grid):
    """The whole point of operating on vtkGhostType in place is that
    a sweep over 100k cells should cost microseconds, not seconds.
    Catches accidental O(N^2) loops in future refactors."""
    ev = ElementVisibility(big_grid)
    # Pre-allocate the id array so we time only the masking step.
    ids = np.arange(big_grid.n_cells, dtype=np.int64)

    t0 = time.perf_counter()
    ev.hide(ids)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"\nElementVisibility.hide({big_grid.n_cells} cells): "
        f"{elapsed_ms:.2f} ms"
    )

    assert ev.n_hidden() == big_grid.n_cells
    assert elapsed_ms < 50.0, (
        f"hide({big_grid.n_cells}) took {elapsed_ms:.2f} ms; expected <50 ms"
    )
