"""Backend wiring — apeGmsh's live runner resolves the Ladruno fork.

Proves Part A of the apeGmsh<->fork "online" work: with
``APEGMSH_OPENSEES_BIN`` pointing at the fork's ``dist\\bin`` (the runner
sets it), the live backend resolver imports the fork's ``opensees`` module
rather than stock ``openseespy``. The whole module is gated by the
``ladruno_fork`` marker, which the root conftest auto-skips unless the
resolved backend is the fork — so this asserts the *positive* path only
where it can hold.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.ladruno_fork


def test_backend_name_is_fork() -> None:
    from apeGmsh.opensees.emitter.live import get_backend_name

    assert get_backend_name() == "ladruno-fork"


def test_live_module_exposes_fork_only_symbol() -> None:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    e = LiveOpsEmitter(wipe=True)
    # criticalTimeStep is a fork-only command (the resolver keys fork
    # detection off it). Its presence confirms the live emitter is bound
    # to the fork build, not stock openseespy.
    assert hasattr(e.ops, "criticalTimeStep")


def test_backend_build_stamp_identifies_the_engine() -> None:
    """The resolved fork reports WHICH build it is, not just that it is a fork.

    ``get_backend_build()`` forwards the fork's ``ladrunoBuild`` command (fork
    PR #718) — the git hash compiled into the binary. Fork builds predating
    that command return ``None``, which is a stale-fork signal, not a failure
    of this wiring, so that case is skipped rather than failed.
    """
    import re

    from apeGmsh.opensees.emitter.live import get_backend_build

    stamp = get_backend_build()
    if stamp is None:
        pytest.skip("fork build predates ops.ladrunoBuild (fork PR #718)")
    assert re.fullmatch(r"[0-9a-f]{40}", stamp), (
        f"backend build stamp {stamp!r} is not a 40-char git hash"
    )


def test_capabilities_carries_the_build_stamp() -> None:
    """``capabilities().build`` is the same stamp, so a script can pin the
    engine it ran against without reaching into the emitter internals."""
    from apeGmsh.opensees._target import probe_live_capabilities
    from apeGmsh.opensees.emitter.live import get_backend_build

    assert probe_live_capabilities().build == get_backend_build()
