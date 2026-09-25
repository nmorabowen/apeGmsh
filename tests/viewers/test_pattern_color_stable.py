"""``loads_tab.pattern_color`` gives a load pattern the same colour in every session.

It used ``abs(hash(name))``. ``hash(str)`` is randomized per process
(``PYTHONHASHSEED``), so a pattern's arrow colour in the mesh viewer
changed between sessions, against its own "Stable color" docstring.
Same fix as the colour modes in #374 (3b5cd921) and 184b5734:
``zlib.crc32``.

The cross-process test runs each seed in a fresh interpreter, because
within one process ``hash()`` is constant and an in-process check
passes on the broken code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_NAMES = ("dead", "live", "wind", "snow", "EQ_x", "EQ_y", "pattern_1", "Pattern 2")

# Pure: loads_tab imports Qt only inside _qt(), so no display is needed.
_PROBE = (
    "import json, apeGmsh.viewers.ui.loads_tab as m\n"
    f"print(json.dumps([m.__file__, [m.pattern_color(n) for n in {_NAMES!r}]]))\n"
)


def _colors_under_seed(seed: str) -> list[str]:
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(_SRC))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env, capture_output=True, text=True, timeout=120, check=True,
    )
    module_file, colors = json.loads(out.stdout.strip().splitlines()[-1])
    # A stale editable install would test some other tree's loads_tab.
    assert Path(module_file).resolve().is_relative_to(_SRC.resolve()), module_file
    return colors


def test_pattern_color_is_the_same_across_hash_seeds() -> None:
    assert _colors_under_seed("1") == _colors_under_seed("2")


def test_pattern_color_uses_crc32() -> None:
    # Pins the mapping: a constant colour would also pass the test above.
    from apeGmsh.viewers.ui.loads_tab import _PATTERN_PALETTE, pattern_color

    for name in _NAMES:
        idx = zlib.crc32(name.encode("utf-8")) % len(_PATTERN_PALETTE)
        assert pattern_color(name) == _PATTERN_PALETTE[idx], name
