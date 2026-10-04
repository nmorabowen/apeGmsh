### FIXED — the qt lane now runs the sections and studio qt files (#1241)

`qt-window-tests` only searched `tests/viewers`, so the qt-marked
`tests/sections` files and `tests/studio/test_{phase_selector,refresh,watch}.py`
ran in no lane. The lane now searches all three directories and runs the
sections files under `QT_QPA_PLATFORM=offscreen` (they skip otherwise).
`test_builder_gui.py`, `test_builder_gui_b7.py` and `test_inspector_s6.py` are
marked `qt` as well, and `tests/test_qt_lane_coverage.py` fails when a qt-marked
file lies outside the lane's directories.
