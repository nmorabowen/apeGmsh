"""#2345 — a click never resolves to a non-finite (unplaced) Gauss row.

Since #2315 / #2327 an unplaced Gauss point carries NaN coordinates.
``np.argmin`` returns the first NaN, so before the fix every Gauss pick
in a model with one unplaced GP selected that element. Headless: plain
arrays and stub session/view/backend, no Qt, no VTK.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from apeGmsh.viewers.session._pick import PanePick, _nearest

NAN = np.nan


def _gauss_coords() -> np.ndarray:
    # Row 0 is the unplaced GP (NaN); row 1 is placed at (5, 5, 0);
    # row 2 is placed far away.
    return np.array([
        [NAN, NAN, NAN],
        [5.0, 5.0, 0.0],
        [50.0, 50.0, 0.0],
    ])


def test_nearest_skips_a_nan_row():
    assert _nearest(_gauss_coords(), (5.0, 5.0, 0.0)) == 1
    assert _nearest(_gauss_coords(), (49.0, 49.0, 0.0)) == 2


def test_nearest_skips_a_partially_nan_row():
    coords = np.array([[1.0, NAN, 0.0], [5.0, 5.0, 0.0]])
    assert _nearest(coords, (1.0, 0.0, 0.0)) == 1


def test_nearest_all_nan_is_no_pick():
    coords = np.full((3, 3), NAN)
    assert _nearest(coords, (0.0, 0.0, 0.0)) is None


def test_nearest_empty_is_no_pick():
    assert _nearest(np.zeros((0, 3)), (0.0, 0.0, 0.0)) is None


# -- the click path, through PanePick._on_pick -------------------------


class _Selection:
    def __init__(self) -> None:
        self.calls: list = []

    def set_gauss(self, pairs):
        self.calls.append(("set_gauss", list(pairs)))

    def toggle_gauss(self, eid, gp):
        self.calls.append(("toggle_gauss", (eid, gp)))

    def clear(self):
        self.calls.append(("clear",))


class _Backend:
    def install(self, on_pick, on_box):
        self.on_pick = on_pick

    def uninstall(self):
        pass


def _pane_pick(coords: np.ndarray):
    view = SimpleNamespace(pick_target="gauss", clip_planes=())
    selection = _Selection()
    session = SimpleNamespace(
        selection=selection, pane=lambda pane_id: view,
    )
    gauss = SimpleNamespace(
        coords=coords,
        element_ids=np.array([7, 8, 9]),
        gp_indices=np.array([0, 0, 0]),
    )
    realized = SimpleNamespace(
        pane_id="p0", targets=SimpleNamespace(gauss=gauss),
    )
    backend = _Backend()
    PanePick(session, backend, lambda: realized)
    return backend, selection


def test_click_on_a_placed_gp_selects_it_beside_an_unplaced_one():
    backend, selection = _pane_pick(_gauss_coords())
    backend.on_pick(SimpleNamespace(world=(5.0, 5.0, 0.0)), None)
    assert selection.calls == [("set_gauss", [(8, 0)])]


def test_click_with_every_gp_unplaced_leaves_the_set_alone():
    backend, selection = _pane_pick(np.full((3, 3), NAN))
    backend.on_pick(SimpleNamespace(world=(5.0, 5.0, 0.0)), None)
    assert selection.calls == []
