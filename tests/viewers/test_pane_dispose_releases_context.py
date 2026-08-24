"""``MeshPane.dispose()`` must release the pane's GL context.

ADR 0098 Amendment 1 caution 1, asked one level down. ``dispose()``
documents itself as closing the context — "a pane that does not close
its own leaves a live render window behind every time a pane is closed
or a window shuts". Amendment 3 then made panes dock widgets that a
user closes routinely, so the cost of being wrong went up.

The evidence that prompted this: ``manual_legend_gesture.py`` only
stopped hanging the qt lane once its teardown added ``hide`` +
``deleteLater`` + pumping ON TOP of ``dispose()``, and its comment reads
as though ``dispose()`` were insufficient.

**Answer, measured: it is not leaking.** Three shown panes, disposed
with ``dispose()`` and nothing else, take the open-plotter count back to
baseline. The measurement is mutation-proven — with
``self._surface.close()`` removed from ``dispose()`` this file reports
"left 3 of 3 render window(s) open", so a pass here means the context
really is released rather than that the probe sees nothing.

What the harness's extra teardown is about is therefore NOT a GL-context
leak; the widgets survive ``dispose()``, which is correct, because
destroying them is Qt's parent-ownership job. The last test pins that
line. This file exists so the caution is a gate instead of a suspicion.

Measured against pyvista's own ``_ALL_PLOTTERS`` registry — the thing
``pv.close_all()`` walks — because "leaked" has to mean something
checkable, not "felt slow".

Marked ``qt``: it needs a real interactor. Run it in its own process,
the way the lane does:

    pytest -m qt tests/viewers/test_pane_dispose_releases_context.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.qt

N_PANES = 3


@pytest.fixture
def qt_results(g, tmp_path: Path):
    """A tiny meshed box with one nodal component — enough to realize."""
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
    disp[1] = np.asarray(fem.nodes.coords, dtype=np.float64)[:, 2]

    path = tmp_path / "dispose_ctx.h5"
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


def _open_plotters() -> int:
    """Plotters pyvista still considers OPEN, globally.

    ``_ALL_PLOTTERS`` is pyvista's own registry; ``_closed`` is the flag
    its ``close()`` sets. Counting the not-yet-closed entries asks
    exactly "how many render windows are still live in this process",
    which is the question ``dispose()`` claims to answer.
    """
    from pyvista.plotting import plotter as P

    return sum(
        1 for p in P._ALL_PLOTTERS.values() if not getattr(p, "_closed", True)
    )


def _build_pane(app, results):
    from apeGmsh.results.session import Contour
    from apeGmsh.viewers.session._pane import MeshPane

    session = results.session()
    view = session.panes[0]
    view.contour = Contour("displacement_z")
    pane = MeshPane(session, pane_id=view.id, defer_fn=lambda fn: fn())
    pane.resize(400, 300)
    pane.show()
    for _ in range(10):
        app.processEvents()
    pane.reconciler.flush_now()
    app.processEvents()
    return pane


def test_dispose_alone_closes_every_pane_context(qt_results):
    """``dispose()`` with NOTHING after it must close the context.

    No ``hide``, no ``deleteLater``, no pumping — because a user closing
    a dock widget gets ``closeEvent`` → ``dispose()`` and nothing else.
    If the count does not come back to baseline here, every pane close
    leaks a live render window.

    Several panes rather than one: a single leak is easy to mistake for
    a fixture artefact, and the failure mode this guards (the NEXT
    interactor in the process dying) needs more than one context to
    exist at all.
    """
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    baseline = _open_plotters()
    panes = [_build_pane(app, qt_results) for _ in range(N_PANES)]
    built = _open_plotters()
    assert built - baseline == N_PANES, (
        f"expected {N_PANES} new open plotters, saw {built - baseline} — "
        f"the fixture is not building real contexts, so a passing "
        f"teardown assertion below would prove nothing."
    )

    for pane in panes:
        pane.dispose()
    app.processEvents()

    after = _open_plotters()
    assert after == baseline, (
        f"dispose() left {after - baseline} of {N_PANES} render window(s) "
        f"open. Every pane close leaks a GL context — ADR 0098 A1 "
        f"caution 1, moved from the shell to the panes."
    )


def test_dispose_is_idempotent(qt_results):
    """``closeEvent`` calls it, and a host may call it again."""
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    baseline = _open_plotters()
    pane = _build_pane(app, qt_results)
    pane.dispose()
    pane.dispose()
    app.processEvents()
    assert _open_plotters() == baseline


def test_dispose_releases_the_context_but_is_not_a_destructor(qt_results):
    """The boundary that explains ``manual_legend_gesture.py``'s teardown.

    That harness releases the context "HARD" — ``hide`` + ``dispose`` +
    ``deleteLater`` + pumping — after a measured hang of the qt lane,
    and its comment reads as though ``dispose()`` were insufficient.
    What is actually true, measured here: ``dispose()`` closes the
    plotter, and the WIDGETS survive, because destroying them is Qt's
    parent-ownership job and not this method's.

    Pinned so the next reader of that teardown does not re-open the
    "does dispose() leak a context" question. It does not — the test
    above proves that, and this says where the line is.
    """
    from qtpy import QtWidgets

    # Liveness of a C++ QObject is binding-specific; skip rather than
    # error on a lane built against a different Qt binding.
    shiboken6 = pytest.importorskip(
        "shiboken6", reason="PySide6-only widget-liveness probe",
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    baseline = _open_plotters()
    pane = _build_pane(app, qt_results)
    surface = pane.surface

    pane.dispose()
    app.processEvents()

    assert _open_plotters() == baseline, "the context must be released"
    assert shiboken6.isValid(pane), (
        "dispose() must NOT destroy the pane widget — Qt owns that, and "
        "a pane that deleted itself would take its dock down with it"
    )
    assert shiboken6.isValid(surface)
