"""ADR 0098 A6 G3 — the legend gesture on a real interactor. NOT A GATE.

Run by hand::

    pytest tests/viewers/manual_legend_gesture.py -p no:cacheprovider

Named ``manual_*`` so pytest does not collect it, following
``manual_check.py``. It is not a gate, and this file says so rather
than sitting in the suite as a permanent skip — a skipped test reads as
coverage, which is the exact illusion A6 exists to remove.

**Why it is not a gate yet.** It needs a SHOWN pane, because an unmapped
render window never takes a real size and the legend layout is then
legitimately degenerate. A shown pane plus the rest of the qt lane
takes an access violation on this Windows/GPU host — the multi-context
fragility ``_pane.py::dispose`` warns about. Measured, in one process:

* this file alone — passes;
* with ``test_viewport_presentation.py`` — passes (4 passed), once the
  teardown below releases the context hard (hide, dispose, deleteLater,
  pump); without that it HUNG;
* the whole qt lane — access violation.

Locally the lane is no oracle either: every other GL test SKIPS here, so
this would be the only real context in the process. CI (Linux/Mesa) is
where the lane actually runs, and that is where this has to be proven
before it can be a gate — carefully, because a hung job burns the
runner rather than failing fast.

**It has already paid for itself.** Driving a real interactor is what
found the two defects A6 fixed: a layout resolved against a ~30 px
viewport and never brought current, so the hit test missed a bar on
screen; and a negative extent from the margin clamp at small viewports.
Both now have offscreen gates in ``test_legend_binding.py`` that run in
every lane. What is unproven in CI is only the last link — that a mouse
event reaches priority 12 — which this script exercises on demand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest



@pytest.fixture
def qt_results(g, tmp_path: Path):
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Box")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    disp = np.zeros((2, node_ids.size))
    disp[1] = np.asarray(fem.nodes.coords, dtype=np.float64)[:, 2] * 10.0

    path = tmp_path / "legend_gesture_qt.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", stage_id="grav",
            time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _drag(app, iren, start, end):
    """Press, move, release — through the interactor, not the API."""
    iren.SetEventPosition(int(start[0]), int(start[1]))
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.SetEventPosition(int(end[0]), int(end[1]))
    iren.InvokeEvent("MouseMoveEvent")
    iren.SetEventPosition(int(end[0]), int(end[1]))
    iren.InvokeEvent("LeftButtonReleaseEvent")
    app.processEvents()


def test_a_real_mouse_drag_moves_the_scale_and_records_it(qt_results):
    """The end-to-end claim no offscreen test can make.

    Covers the miss case in the same context: a drag in open space must
    leave the legend alone, because the press observer returns WITHOUT
    aborting on a miss — if it aborted, camera and pick behaviour
    outside a legend would change.
    """
    from qtpy import QtWidgets

    from apeGmsh.results.session import Contour
    from apeGmsh.viewers.session._pane import MeshPane

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = qt_results.session()
    view = session.panes[0]
    view.contour = Contour("displacement_z")

    pane = MeshPane(session, pane_id=view.id, defer_fn=lambda fn: fn())
    try:
        pane.resize(900, 700)
        pane.show()
        for _ in range(20):
            app.processEvents()
        pane.reconciler.flush_now()
        app.processEvents()

        realized = pane.reconciler.realized
        controller = realized.legend_controller if realized else None
        assert controller is not None, "expected a realized legend"
        assert pane._legends._interactor is not None, (
            "a pane with a real QtInteractor must install the legend "
            "gesture — the S6a gap Amendment 5 closed."
        )

        # Bring the layout current, the way the first hit-test would.
        controller.ensure_current()
        app.processEvents()
        entry = next(e for e in controller.entries() if e.visible)
        key = entry.key
        vw, vh = controller.viewport_px()
        w, h = entry.extent
        assert w > 0 and h > 0, f"degenerate legend box {(w, h)}"

        iren = controller._backend.plotter.iren.interactor

        # -- a miss must change nothing ---------------------------------
        before = tuple(controller.entry(key).anchor)
        _drag(app, iren, (12, 12), (80, 80))
        assert tuple(controller.entry(key).anchor) == before
        assert view.legend_placement(key[1]) is None

        # -- a hit must move it, and reach the session ------------------
        ax, ay = before
        start = ((ax + w * 0.5) * vw, (ay + h * 0.5) * vh)
        _drag(app, iren, start, (start[0] - 90, start[1] - 90))

        after = tuple(controller.entry(key).anchor)
        assert after != before, (
            "the drag did not reach the legend. Either no interactor is "
            "installed, or another observer aborted the event before "
            "priority 12."
        )
        placement = view.legend_placement(key[1])
        assert placement is not None, "a real drag must reach the view"
        assert placement.anchor == pytest.approx(after)
    finally:
        # Release the GL context HARD. A pane whose context outlives
        # the test takes the next interactor in the process with it —
        # measured as a HANG of the whole qt lane on Windows when this
        # ran before ``test_viewport_presentation``. hide() unmaps the
        # window before dispose() closes the interactor, and
        # deleteLater()+processEvents actually reaps the widget rather
        # than leaving it for interpreter shutdown.
        pane.hide()
        app.processEvents()
        pane.dispose()
        app.processEvents()
        pane.setParent(None)
        pane.deleteLater()
        for _ in range(5):
            app.processEvents()
