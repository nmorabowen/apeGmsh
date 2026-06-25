"""Qt process-environment guards for desktop viewers."""
from __future__ import annotations


def prepare_qt_environment() -> None:
    """Apply platform guards before importing/creating Qt objects.

    PyVistaQt + Qt can crash on some Linux Wayland sessions during startup.
    Prefer XCB there unless the user explicitly selected another platform.
    """
    import os
    import sys

    if sys.platform.startswith("linux"):
        qpa = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
        if qpa == "" or qpa.startswith("wayland"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"

    # Kvantum can crash or render incorrectly in embedded VTK Qt windows.
    if os.environ.get("QT_STYLE_OVERRIDE", "").lower() == "kvantum":
        os.environ["QT_STYLE_OVERRIDE"] = "Fusion"


__all__ = ["prepare_qt_environment"]
