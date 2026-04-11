"""apeGmshViewer — Standalone 3D viewer for pyGmsh meshes and FEM results."""

__version__ = "0.1.0"


def _in_jupyter() -> bool:
    """Detect if running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def show(*filepaths, blocking=None):
    """Launch the apeGmshViewer from a notebook or script.

    Parameters
    ----------
    *filepaths : str or Path
        One or more VTU/PVD/MSH files to open on launch.
    blocking : bool or None
        If True, runs the Qt event loop and blocks until the window
        is closed — safe for scripts.
        If False, launches in a subprocess so the notebook stays alive.
        If None (default), auto-detects: False in Jupyter, True in scripts.

    Examples
    --------
    From a notebook cell::

        from apeGmshViewer import show
        show("results.vtu")            # auto-detects Jupyter → subprocess

    From a script::

        from apeGmshViewer import show
        show("results.vtu")            # auto-detects script → blocking
    """
    from pathlib import Path

    if blocking is None:
        blocking = not _in_jupyter()

    paths = [str(Path(f).resolve()) for f in filepaths]

    if not blocking:
        # Non-blocking: launch as a subprocess so the notebook stays alive
        import subprocess
        import sys
        cmd = [sys.executable, "-m", "apeGmshViewer"] + paths
        subprocess.Popen(cmd)
        return

    # Run the Qt event loop in this process
    _launch_viewer(
        lambda win: [win.load_file(p) for p in paths],
        blocking=blocking,
    )


def show_mesh_data(mesh_data, *, blocking=True):
    """Launch apeGmshViewer with a MeshData object directly (no file I/O).

    This always runs in-process (blocking) since MeshData cannot be
    serialized to a subprocess.  For non-blocking use, call
    ``results.viewer(blocking=False)`` which writes temp files and
    launches a subprocess.

    Parameters
    ----------
    mesh_data : MeshData
        Pre-built mesh data, e.g. from ``Results.to_mesh_data()``.
    blocking : bool
        Must be True (default). In-process Qt event loop.
    """
    _launch_viewer(
        lambda win: win.load_mesh_data(mesh_data, mesh_data.name),
        blocking=blocking,
    )


def _launch_viewer(load_fn, *, blocking=False):
    """Create a MainWindow, load data, and optionally block."""
    import pyvista as pv
    pv.set_plot_theme("dark")
    pv.global_theme.font.color = "white"

    import sys
    from qtpy.QtWidgets import QApplication
    from apeGmshViewer.main_window import MainWindow

    app = QApplication.instance()
    own_app = app is None
    if own_app:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.showMaximized()

    # Load data after show so the VTK render window is realized
    try:
        load_fn(window)
    except Exception as e:
        print(f"Warning: could not load data: {e}")

    if blocking and own_app:
        app.exec()


# Keep old name as alias for backward compatibility
_launch_blocking = _launch_viewer
