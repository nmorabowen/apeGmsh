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
