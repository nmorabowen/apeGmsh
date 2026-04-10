"""
pyGmshViewer — Entry point.

Usage:
    python -m pyGmshViewer                    # Launch empty
    python -m pyGmshViewer model.vtu          # Open a file on startup
    python -m pyGmshViewer results.pvd        # Open a time-series
"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    # Must set PyVista to use the Qt backend BEFORE importing Qt
    import pyvista as pv
    pv.set_plot_theme("dark")
    pv.global_theme.font.color = "white"

    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QFont
    from apeGmshViewer.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("pyGmsh Viewer")
    app.setOrganizationName("pyGmsh")

    # Default font — use platform-safe fallback
    font = QFont()
    font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"])
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    # Load files from command line args
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.exists():
            try:
                window.load_file(p)
            except Exception as e:
                print(f"Warning: Could not load '{p}': {e}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
