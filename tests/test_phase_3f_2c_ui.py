"""Phase 3F.2c — UI registry includes 'Module' color mode.

Tiny slice: adds 'Module' to ``COLOR_MODES`` in
``src/apeGmsh/viewers/ui/mesh_tabs.py``. The controller-side dispatch
(``"Module"`` branch + ``_module_idle`` callback) shipped in 3F.2b
(PR #373); the data layer (``ViewerElements.module_for`` etc.)
shipped in 3F.2a (PR #372). This slice exposes the mode in the user-
facing dropdown.

No GPU verification here per ``feedback_viewer_no_gpu``; the user
eyeballs the dropdown after merge.
"""
from __future__ import annotations

from apeGmsh.viewers.ui.mesh_tabs import COLOR_MODES


class TestColorModesRegistry:
    def test_module_present(self) -> None:
        """The Module mode is registered in the UI dropdown list."""
        assert "Module" in COLOR_MODES

    def test_module_between_partition_and_quality(self) -> None:
        """Module sits between Partition and Quality.

        Order matters: the dispatch in ``ColorModeController.set_mode``
        also dispatches Module after Partition, so the dropdown order
        matches the controller's ``elif`` chain reading order. Locks
        the ordering against accidental reshuffles.
        """
        i_partition = COLOR_MODES.index("Partition")
        i_module = COLOR_MODES.index("Module")
        i_quality = COLOR_MODES.index("Quality")
        assert i_partition < i_module < i_quality

    def test_no_duplicates(self) -> None:
        """No duplicate entries (catches double-add regressions)."""
        assert len(COLOR_MODES) == len(set(COLOR_MODES))

    def test_existing_modes_preserved(self) -> None:
        """The pre-3F.2c modes are all still present."""
        expected_pre_existing = {
            "Default",
            "Element Type",
            "Physical Group",
            "Partition",
            "Quality",
        }
        assert expected_pre_existing.issubset(set(COLOR_MODES))
