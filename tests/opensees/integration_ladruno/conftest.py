"""Fork-lane guards for ``tests/opensees/integration_ladruno``.

* ``ladruno_mkl`` tests skip with a reason when ``system Pardiso`` is not
  available in the imported build (the Linux tarball may lack MKL).

The ``APEGMSH_FORK_PIN_ENFORCE=1`` build-stamp skip lives in the root
``tests/conftest.py``, so it reaches every ``ladruno_fork`` test, not only
the ones in this directory.
"""
from __future__ import annotations

import pytest

_PARDISO: "tuple[bool, str] | None" = None


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


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    # trylast: the root conftest's backend and FORK_PIN skips run first, so
    # an off-pin build reads "fork build mismatch", never "ladruno_mkl:"
    # (the live-fork lane fails on any ladruno_mkl skip).
    if item.get_closest_marker("ladruno_fork") is None:
        return
    if item.get_closest_marker("ladruno_mkl") is not None:
        ok, why = _pardiso_available()
        if not ok:
            pytest.skip(f"ladruno_mkl: Pardiso unavailable ({why})")
