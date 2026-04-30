from .model_viewer import ModelViewer
from .mesh_viewer import MeshViewer
from .geom_transf_viewer import GeomTransfViewer
from .results_viewer import ResultsViewer


def settings() -> int:
    """Open the global preferences editor (modal dialog).

    Persists changes to the JSON file at
    ``PreferencesManager.path`` (platform-appropriate config dir).
    Spins up a ``QApplication`` if none exists.

    Returns the dialog result code (``QDialog.Accepted`` / ``Rejected``).
    """
    from .ui.preferences_dialog import open_preferences_dialog
    return open_preferences_dialog()


def theme_editor() -> int:
    """Open the theme editor (modal dialog with live preview).

    Custom themes are persisted under ``ThemeManager.themes_dir()``
    (platform-appropriate config dir). Spins up a ``QApplication`` if
    none exists.

    Returns the dialog result code (``QDialog.Accepted`` / ``Rejected``).
    """
    from .ui.theme_editor_dialog import open_theme_editor
    return open_theme_editor()


__all__ = [
    "ModelViewer",
    "MeshViewer",
    "GeomTransfViewer",
    "ResultsViewer",
    "settings",
    "theme_editor",
]
