"""Criterion 21 — the ADR 0098 Amendment 1 screenshot review bar.

Renders 1-pane / 2-pane / 4-pane captures of the real
:class:`SessionWindow` in ``catppuccin_mocha``, ``neutral_studio`` and
``paper``, at both densities, and one deliberately narrow capture that
puts a pane AT the 240 px width floor so the header can be read for
clipping (0087 INV-6).

Untracked scratch — regenerate at will. It restores the persisted
theme / density on the way out, so it never leaves the viewer in
another look::

    python scratchpad_shots/take_pane_host_shots.py

Needs a real GL context (it opens a window); it is not a test and no
lane runs it.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "pane_host"
THEMES = ("catppuccin_mocha", "neutral_studio", "paper")
DENSITIES = ("comfortable", "compact")


def _results():
    from apeGmsh import apeGmsh
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    from tests.conftest import _open_model_from_h5

    g = apeGmsh(model_name="pane_host_shots", verbose=False)
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 2, 1, 1, label="deck")
    g.model.geometry.add_box(2, 0, 0, 1, 1, 1, label="pier")
    g.physical.add_volume("deck", name="Deck")
    g.physical.add_volume("pier", name="Pier")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    disp = np.zeros((3, node_ids.size))
    for t in range(3):
        disp[t] = (t + 1) * coords[:, 2] * 0.05

    path = Path(tempfile.mkdtemp()) / "pane_host_shots.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", stage_id="grav",
            time=np.arange(3, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.end_stage()
    g.end()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _qimage(img):
    """A ``QImage`` owning a copy of an ``(h, w, 3|4)`` uint8 array."""
    from qtpy import QtGui

    img = np.ascontiguousarray(img[:, :, :3], dtype=np.uint8)
    height, width, _ = img.shape
    return QtGui.QImage(
        img.data, width, height, 3 * width,
        QtGui.QImage.Format.Format_RGB888,
    ).copy()


def _shot(qapp, window, out: Path) -> None:
    """Composite the window: Qt chrome + each pane's own frame.

    ``QWidget.grab()`` renders the Qt widget tree only, so a pane's
    native OpenGL child window comes out blank — the wrong evidence for
    a review bar about N live viewports. ``QScreen.grabWindow`` gets
    the GL but captures whatever is on top of a shared desktop, which
    lost a focus race to an unrelated editor window. Painting each
    pane's OWN ``screenshot()`` into the grabbed chrome needs no focus,
    races nothing, and is the same pixels the pane renders.
    """
    from qtpy import QtCore, QtGui

    qapp.processEvents()
    for frame in window.host.pane_frames:
        if frame.pane is not None:
            frame.pane.reconciler.flush_now()
        if frame.plotter is not None:
            frame.plotter.render()
    qapp.processEvents()

    win = window.shell.window
    pix = win.grab()
    painter = QtGui.QPainter(pix)
    try:
        for frame in window.host.pane_frames:
            plotter = frame.plotter
            if plotter is None:
                continue
            image = _qimage(np.asarray(plotter.screenshot(return_img=True)))
            corner = plotter.mapTo(win, QtCore.QPoint(0, 0))
            painter.drawImage(QtCore.QRect(corner, plotter.size()), image)
    finally:
        painter.end()
    out.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out))
    print(f"saved {out}")


def main() -> int:
    from qtpy import QtWidgets
    from qtpy.QtCore import QSettings

    from apeGmsh.results.session import Contour, Deform, MeshStyle, Scope
    from apeGmsh.viewers.session import SessionResultsWindow, SessionWindow
    from apeGmsh.viewers.ui.density import DENSITY
    from apeGmsh.viewers.ui.theme import THEME

    # A throwaway layout scope: the review bar must show the SHIPPED
    # boot arrangement (0088 D1), not whatever the developer last
    # dragged into their own QSettings.
    ini = str(Path(tempfile.mkdtemp()) / "shots.ini")
    SessionResultsWindow._layout_settings = staticmethod(  # noqa: SLF001
        lambda: QSettings(ini, QSettings.Format.IniFormat),
    )

    saved_theme = THEME.current.name
    saved_density = DENSITY.current.name
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    results = _results()

    try:
        for density in DENSITIES:
            DENSITY.set_density(density)
            for theme in THEMES:
                THEME.set_theme(theme)
                session = results.session()
                window = SessionWindow(session, title=f"{theme} · {density}")
                window.shell.window.resize(1280, 800)
                window.show(blocking=False)
                _shot(qapp, window, OUT / f"{theme}_{density}_1pane.png")

                second = session.add_view("Deck only")
                second.scope = Scope("physical_groups", ("Deck",))
                session.panes[0].contour = Contour("displacement_z")
                _shot(qapp, window, OUT / f"{theme}_{density}_2pane.png")

                third = session.add_view("Deformed")
                third.deform = Deform("displacement")
                third.style = MeshStyle(
                    mesh=False, outlines=True, nodes=True, gauss=False,
                )
                # §6: a plot IS a pane — same host, same frame, same
                # tiling, and (0087 INV-2) no style buttons, because
                # they do not act on a plot.
                session.add_plot(name="Tip Uz")
                _shot(qapp, window, OUT / f"{theme}_{density}_4pane.png")

                # The 240 px floor: shrink the window until the panes
                # sit ON it, so the headers can be read for clipping.
                window.shell.window.resize(1000, 700)
                window.host.setFixedWidth(2 * 240 + 4)
                _shot(qapp, window, OUT / f"{theme}_{density}_floor.png")

                window.close()
                qapp.processEvents()
    finally:
        THEME.set_theme(saved_theme)
        DENSITY.set_density(saved_density)
    return 0


if __name__ == "__main__":
    sys.exit(main())
