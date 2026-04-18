from .model_viewer import ModelViewer
from .mesh_viewer import MeshViewer
from .geom_transf_viewer import GeomTransfViewer


def settings() -> int:
    """Open the global preferences editor (modal dialog).

    Persists changes to the JSON file at
    ``PreferencesManager.path`` (platform-appropriate config dir).
    Spins up a ``QApplication`` if none exists.

    Returns the dialog result code (``QDialog.Accepted`` / ``Rejected``).
    """
    from .ui.preferences_dialog import open_preferences_dialog
    return open_preferences_dialog()


__all__ = [
    "ModelViewer",
    "MeshViewer",
    "GeomTransfViewer",
    "settings",
]
