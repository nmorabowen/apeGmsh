from __future__ import annotations

import os

from apeGmsh.viewers.ui._qt_env import prepare_qt_environment


def test_prepare_qt_environment_prefers_xcb_on_linux_wayland(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland")

    prepare_qt_environment()

    assert os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_prepare_qt_environment_preserves_explicit_offscreen(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    prepare_qt_environment()

    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"


def test_prepare_qt_environment_replaces_kvantum(monkeypatch):
    monkeypatch.setenv("QT_STYLE_OVERRIDE", "kvantum")

    prepare_qt_environment()

    assert os.environ["QT_STYLE_OVERRIDE"] == "Fusion"
