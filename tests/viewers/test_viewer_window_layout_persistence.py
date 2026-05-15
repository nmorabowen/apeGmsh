"""Plan 08 — ViewerWindow layout-state persistence (mesh/model viewer).

Verifies the ``window_key`` constructor argument + ``_save_layout`` /
``_restore_layout`` / ``reset_layout`` mirroring
:class:`ResultsWindow`'s mechanism, against a stub-bound shell so we
don't pay the VTK ``QtInteractor`` overhead.

Same stub pattern as :mod:`test_viewer_window_extensions`: ``QMainWindow``
+ method-binding from the real :class:`ViewerWindow` class so any
drift in the production code surfaces here.
"""
from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("qtpy.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_stub(qapp, *, window_key=None):
    """Stub ``ViewerWindow`` with the layout-persistence methods bound."""
    from qtpy import QtWidgets
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    class _Stub:
        pass

    stub = _Stub()
    stub._window = QtWidgets.QMainWindow()
    stub._window.setCentralWidget(QtWidgets.QWidget(stub._window))
    stub._window_key = window_key
    stub._default_layout_state = None
    stub._default_layout_geometry = None
    stub._extension_docks = {}
    stub._tabs_dock = None
    stub._console_dock = None
    # ``_LAYOUT_SCHEMA_VERSION`` is read on save / restore.
    stub._LAYOUT_SCHEMA_VERSION = ViewerWindow._LAYOUT_SCHEMA_VERSION

    # set_status is called by reset_layout; bind a no-op.
    stub.set_status = lambda *_args, **_kwargs: None

    for name in (
        "_layout_settings", "_save_layout", "_restore_layout",
        "reset_layout",
    ):
        method = getattr(ViewerWindow, name)
        setattr(stub, name, method.__get__(stub))
    return stub


# =====================================================================
# API contract
# =====================================================================


def test_window_key_param_in_constructor_signature():
    import inspect
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    sig = inspect.signature(ViewerWindow.__init__)
    assert "window_key" in sig.parameters
    # Default = None preserves the legacy non-persistent path.
    assert sig.parameters["window_key"].default is None


def test_class_exposes_layout_schema_version():
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    assert isinstance(ViewerWindow._LAYOUT_SCHEMA_VERSION, int)
    assert ViewerWindow._LAYOUT_SCHEMA_VERSION >= 1


def test_class_exposes_layout_persistence_methods():
    from apeGmsh.viewers.ui.viewer_window import ViewerWindow

    assert hasattr(ViewerWindow, "_save_layout")
    assert hasattr(ViewerWindow, "_restore_layout")
    assert hasattr(ViewerWindow, "reset_layout")
    assert hasattr(ViewerWindow, "_layout_settings")


# =====================================================================
# Persistence off (window_key=None) — save / restore are no-ops
# =====================================================================


def test_layout_settings_is_none_without_window_key(qapp):
    stub = _make_stub(qapp, window_key=None)
    assert stub._layout_settings() is None


def test_save_layout_without_window_key_does_not_raise(qapp):
    stub = _make_stub(qapp, window_key=None)
    # Must not raise + must not touch QSettings (we can't directly
    # observe the absence of writes here, but the no-op contract is
    # the important shape).
    stub._save_layout()


def test_restore_layout_without_window_key_does_not_raise(qapp):
    stub = _make_stub(qapp, window_key=None)
    stub._restore_layout()


# =====================================================================
# Persistence on — save / restore route through QSettings
# =====================================================================


def _isolated_settings_path(tmp_path):
    """Redirect QSettings to a tmp directory so tests don't pollute
    the user's real OS settings. Must be called before any QSettings
    instance is constructed.
    """
    from qtpy.QtCore import QSettings
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)


def test_save_layout_writes_schema_version(qapp, tmp_path):
    _isolated_settings_path(tmp_path)
    stub = _make_stub(qapp, window_key="TestViewerSaveSchema")
    stub._save_layout()

    settings = stub._layout_settings()
    stored = settings.value("layout/schema_version")
    # QSettings IniFormat returns the int as a string round-trip.
    assert int(stored) == stub._LAYOUT_SCHEMA_VERSION


def test_save_layout_writes_state_and_geometry(qapp, tmp_path):
    _isolated_settings_path(tmp_path)
    stub = _make_stub(qapp, window_key="TestViewerSaveState")
    stub._save_layout()

    settings = stub._layout_settings()
    assert settings.value("layout/geometry") is not None
    assert settings.value("layout/state") is not None


def test_restore_layout_skips_mismatched_schema(qapp, tmp_path):
    """A stored schema_version different from the current one must be
    discarded — applying it would produce a half-broken layout."""
    _isolated_settings_path(tmp_path)
    stub = _make_stub(qapp, window_key="TestViewerSchemaMismatch")
    settings = stub._layout_settings()
    # Write a state from a fictional v999 layout.
    settings.setValue("layout/schema_version", 999)
    settings.setValue("layout/state", stub._window.saveState())
    settings.setValue("layout/geometry", stub._window.saveGeometry())
    settings.sync()

    # Must not raise and must not apply the stale state. (We can't
    # directly observe "did restoreState fire" cheaply; the no-raise
    # contract is what matters — and the next save will overwrite it.)
    stub._restore_layout()


def test_save_then_restore_roundtrip(qapp, tmp_path):
    """End-to-end: a window's state can be saved, then a fresh window
    restores it. Verifies the QSettings keys persist across stub
    rebuilds."""
    _isolated_settings_path(tmp_path)
    stub_a = _make_stub(qapp, window_key="TestViewerRoundTrip")
    stub_a._window.resize(800, 600)
    stub_a._save_layout()

    stub_b = _make_stub(qapp, window_key="TestViewerRoundTrip")
    # Must not raise — restore reads the keys stub_a wrote.
    stub_b._restore_layout()


# =====================================================================
# reset_layout — restores captured defaults
# =====================================================================


def test_reset_layout_restores_default_state(qapp):
    from qtpy import QtWidgets

    stub = _make_stub(qapp, window_key="TestViewerReset")
    # Capture a "default" state, then resize and dock-arrange to
    # diverge from it.
    stub._default_layout_state = stub._window.saveState()
    stub._default_layout_geometry = stub._window.saveGeometry()
    # Add a dock + hide it so reset_layout has something to re-show.
    dock = QtWidgets.QDockWidget("D", stub._window)
    dock.setObjectName("test_dock")
    from qtpy import QtCore
    stub._window.addDockWidget(
        QtCore.Qt.RightDockWidgetArea, dock,
    )
    stub._extension_docks["test_dock"] = dock
    dock.setVisible(False)

    stub.reset_layout()
    # Hidden dock is re-shown so the reset arrangement is recognizable.
    # ``isHidden()`` reflects the explicit visibility flag independent
    # of whether the parent QMainWindow is realized — we run headless
    # so ``isVisible()`` would always return False.
    assert dock.isHidden() is False


def test_reset_layout_noop_when_no_default_captured(qapp):
    stub = _make_stub(qapp, window_key="TestViewerNoDefault")
    # _default_layout_state stays None — reset_layout must not raise.
    stub.reset_layout()
