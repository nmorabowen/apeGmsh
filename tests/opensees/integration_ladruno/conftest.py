"""Fork-lane guards for ``tests/opensees/integration_ladruno``.

* ``APEGMSH_FORK_PIN_ENFORCE=1`` (set by the nightly ``live-fork`` lane):
  every ``ladruno_fork`` test skips unless the imported engine's build stamp
  equals the ``sha`` in the repo-root ``FORK_PIN``. Unset, local runs beside
  a dev fork build are unaffected.
* ``ladruno_mkl`` tests skip with a reason when ``system Pardiso`` is not
  available in the imported build (the Linux tarball may lack MKL).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_PIN_FILE = Path(__file__).resolve().parents[3] / "FORK_PIN"
_PARDISO: "tuple[bool, str] | None" = None


def _pinned_sha() -> str:
    for line in _PIN_FILE.read_text(encoding="utf-8").splitlines():
        key, sep, val = line.partition("=")
        if sep and key.strip() == "sha":
            return val.strip()
    raise RuntimeError(f"no sha= line in {_PIN_FILE}")


def _pardiso_available() -> "tuple[bool, str]":
    global _PARDISO
    if _PARDISO is None:
        from apeGmsh.opensees.emitter.live import get_ops
        try:
            ops = get_ops()
            ops.wipe()
            ops.model("basic", "-ndm", 1, "-ndf", 1)
            rc = ops.system("Pardiso")
            ops.wipe()
            _PARDISO = (rc in (0, None), f"system Pardiso returned {rc!r}")
        except Exception as e:  # unknown system type may raise
            _PARDISO = (False, f"system Pardiso raised {type(e).__name__}: {e}")
    return _PARDISO


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("ladruno_fork") is None:
        return
    if os.environ.get("APEGMSH_FORK_PIN_ENFORCE") == "1":
        pinned = _pinned_sha()
        from apeGmsh.opensees.emitter.live import get_backend_build
        try:
            imported = get_backend_build()
        except Exception:
            imported = None
        if imported != pinned:
            pytest.skip(
                f"fork build mismatch: pinned {pinned[:12]}, "
                f"imported {imported[:12] if imported else 'none'}"
            )
    if item.get_closest_marker("ladruno_mkl") is not None:
        ok, why = _pardiso_available()
        if not ok:
            pytest.skip(f"ladruno_mkl: Pardiso unavailable ({why})")
