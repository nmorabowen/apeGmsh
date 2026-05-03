"""``apeGmsh.workdir`` — tiny per-script output directory helper."""
from __future__ import annotations

from pathlib import Path


def workdir(name: str | Path = "outputs") -> Path:
    """Return ``Path(name)`` after ensuring it exists.

    Convention for example notebooks: every script puts its
    artifacts (``capture.h5``, ``recorders/``, exports, etc.) under
    a sibling ``outputs/`` folder so the example directory stays
    self-contained. Typical use::

        from apeGmsh import workdir
        OUT = workdir()                 # ./outputs/
        cap_path = OUT / 'capture.h5'

    Pass an explicit name for nested or non-default layouts
    (``workdir('outputs/run_42')``).
    """
    p = Path(name)
    p.mkdir(parents=True, exist_ok=True)
    return p
