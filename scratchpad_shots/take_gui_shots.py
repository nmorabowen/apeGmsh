"""Phase-2a verification shots — real ResultsViewer window.

Opens the results viewer on a tiny synthetic model (the
``test_results_viewer_smoke.small_results`` recipe), forces the
``catppuccin_mocha`` review palette, and saves to ``scratchpad_shots/``:

* ``p2a_full_window.png``   — whole window (icon rollout overview)
* ``p2a_scrubber.png``      — time-scrubber closeup (transport glyphs)
* ``p2a_view_menu.png``     — View menu opened (Focus mode / Camera /
  Orbit axis / Theme)
* ``p2a_camera_submenu.png`` — Camera submenu grab
* ``p2a_file_menu.png``     — File menu (Open results… Ctrl+O etc.)

THEME persists to QSettings — snapshotted before, restored after.

Run::

    python scratchpad_shots/take_gui_shots.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

OUT = REPO / "scratchpad_shots"


def _build_results():
    from apeGmsh import apeGmsh
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    g = apeGmsh(model_name="shots", verbose=False)
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = Path(tempfile.mkdtemp()) / "shots.h5"
    rng = np.random.default_rng(7)
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": rng.normal(0, 0.01, (3, n_nodes)),
                "displacement_y": rng.normal(0, 0.01, (3, n_nodes)),
                "displacement_z": rng.normal(0, 0.02, (3, n_nodes)),
            },
        )
        w.end_stage()
    g.end()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _screen_grab(qapp, widget, out: Path, *, pad: int = 24) -> None:
    """Grab the SCREEN region around ``widget``'s top-level window —
    captures native popups (menus) that widget.grab() misses."""
    from qtpy import QtCore
    win = widget.window()
    screen = qapp.primaryScreen()
    geo = win.frameGeometry()
    pix = screen.grabWindow(
        0,
        max(0, geo.x() - pad), max(0, geo.y() - pad),
        geo.width() + 2 * pad, geo.height() + 2 * pad,
    )
    pix.save(str(out))
    print(f"saved {out}")


def main() -> int:
    from qtpy import QtCore, QtWidgets
    from qtpy.QtCore import QSettings

    settings = QSettings("apeGmsh", "viewer")
    saved_theme = settings.value("theme", None)

    from apeGmsh.viewers.ui.theme import THEME
    THEME.set_theme("catppuccin_mocha")

    try:
        results = _build_results()
        from apeGmsh.viewers.results_viewer import ResultsViewer
        viewer = ResultsViewer(results)
        viewer.show(enter_loop=False)
        qapp = QtWidgets.QApplication.instance()
        win = viewer._win  # ResultsWindow
        qwin = win.window

        def pump(n=8):
            for _ in range(n):
                qapp.processEvents()
                QtCore.QThread.msleep(60)

        pump(20)

        # 1 — full window.
        _screen_grab(qapp, qwin, OUT / "p2a_full_window.png")

        # 2 — scrubber closeup.
        scrubber = viewer._time_scrubber
        pix = scrubber.widget.grab()
        pix.save(str(OUT / "p2a_scrubber.png"))
        print(f"saved {OUT / 'p2a_scrubber.png'}")

        # 3 — View menu opened.
        mb = qwin.menuBar()
        view_act = next(a for a in mb.actions() if a.text() == "View")
        view_menu = view_act.menu()
        pos = mb.mapToGlobal(mb.actionGeometry(view_act).bottomLeft())
        view_menu.popup(pos)
        pump(6)
        _screen_grab(qapp, qwin, OUT / "p2a_view_menu.png")
        menu_pix = view_menu.grab()
        menu_pix.save(str(OUT / "p2a_view_menu_only.png"))
        view_menu.close()
        pump(2)

        # 4 — Camera submenu grab (rendered standalone).
        cam_act = next(a for a in view_menu.actions()
                       if a.menu() and a.text() == "Camera")
        cam_menu = cam_act.menu()
        cam_menu.popup(pos)
        pump(4)
        cam_menu.grab().save(str(OUT / "p2a_camera_submenu.png"))
        print(f"saved {OUT / 'p2a_camera_submenu.png'}")
        cam_menu.close()
        pump(2)

        # 5 — File menu opened.
        file_act = next(a for a in mb.actions() if a.text() == "File")
        file_menu = file_act.menu()
        fpos = mb.mapToGlobal(mb.actionGeometry(file_act).bottomLeft())
        file_menu.popup(fpos)
        pump(6)
        _screen_grab(qapp, qwin, OUT / "p2a_file_menu.png")
        file_menu.grab().save(str(OUT / "p2a_file_menu_only.png"))
        file_menu.close()
        pump(2)

        qwin.close()
        pump(4)
        return 0
    finally:
        # Restore the persisted theme exactly as found.
        if saved_theme is None:
            settings.remove("theme")
        else:
            settings.setValue("theme", saved_theme)
        settings.sync()


if __name__ == "__main__":
    raise SystemExit(main())
