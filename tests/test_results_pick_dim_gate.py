"""S4b acceptance — the results pick dim-gate (_accept_cell_dim).

Headless: the gate is a pure function extracted from the VTK-observer pick
closures, so the dim-filter decision is testable without a live interactor.
The live FilterController wiring + box path are eyeball-gated.
"""
from __future__ import annotations

import numpy as np

from apeGmsh.viewers.core.results_pick import _accept_cell_dim, ResultsPickController

CELL_DIM = np.array([1, 3, 2, 3], dtype=np.int8)   # line, hex, quad, hex


def test_none_filter_accepts_everything() -> None:
    assert _accept_cell_dim(CELL_DIM, 2, None) is True


def test_gates_by_active_dims() -> None:
    assert _accept_cell_dim(CELL_DIM, 1, frozenset({3})) is True   # cell 1 → dim 3
    assert _accept_cell_dim(CELL_DIM, 2, frozenset({3})) is False  # cell 2 → dim 2
    assert _accept_cell_dim(CELL_DIM, 2, frozenset({2, 3})) is True


def test_empty_cell_dim_accepts() -> None:
    assert _accept_cell_dim(np.array([], dtype=np.int8), 0, frozenset({3})) is True


def test_out_of_range_cell_rejected() -> None:
    assert _accept_cell_dim(CELL_DIM, 99, frozenset({3})) is False
    assert _accept_cell_dim(CELL_DIM, -1, frozenset({3})) is False


def test_controller_active_dims_defaults_none() -> None:
    assert ResultsPickController().active_dims is None
