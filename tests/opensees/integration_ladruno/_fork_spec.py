"""The fork's executable ``.ladruno`` spec, fetched at ``FORK_PIN``.

``ladruno_format.py`` (with its import, ``ladruno_basis.py``) is the
fork-owned form of schema v1: ``validate()`` and the canonical
``normalize_nodal`` / ``normalize_element`` normalizers. It is not shipped
with the fork build, so a live test downloads it from the fork repository
at the ``FORK_PIN`` commit (as ``test_ladruno_provenance_live.py`` does,
#2111) and imports it for the duration of a ``with`` block. Nothing of it
is copied into apeGmsh (ADR 0122 D4, D7).
"""
from __future__ import annotations

import importlib
import sys
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

_PIN = Path(__file__).resolve().parents[3] / "FORK_PIN"
_SPEC_DIR = "Ladruno_scripts/ladruno_recorder_tests"
_SPEC_FILES = ("ladruno_format.py", "ladruno_basis.py")
_MODULES = ("ladruno_format", "ladruno_basis")


def pin(key: str) -> str:
    """One ``key=value`` line of the repo-root ``FORK_PIN``."""
    for line in _PIN.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == key:
            return v.strip()
    raise RuntimeError(f"no {key}= line in {_PIN}")


def fetch(dest: Path) -> Path:
    """Download the spec modules at ``FORK_PIN`` into ``dest``; return it."""
    sha, repo = pin("sha"), pin("repo")
    dest.mkdir(parents=True, exist_ok=True)
    for name in _SPEC_FILES:
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{_SPEC_DIR}/{name}"
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            (dest / name).write_bytes(resp.read())
    return dest


@contextmanager
def ladruno_format(spec_dir: Path) -> Iterator[ModuleType]:
    """Import the fetched ``ladruno_format``; restore ``sys`` on exit."""
    saved_path = list(sys.path)
    saved = {k: sys.modules[k] for k in _MODULES if k in sys.modules}
    for k in _MODULES:
        sys.modules.pop(k, None)
    sys.path.insert(0, str(spec_dir))
    try:
        yield importlib.import_module("ladruno_format")
    finally:
        for k in _MODULES:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
