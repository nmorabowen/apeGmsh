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


def test_backend_info_parses_the_ladruno_schema() -> None:
    """The bound fork's ``ladrunoSchema`` (fork PR #972, v1 since WP-185 /
    apeGmsh#2517) parses (#1931).

    In the ``live-fork`` lane (``APEGMSH_FORK_PIN_ENFORCE=1``) the pinned
    build carries the schema, so it must parse. A local dev build predating
    the command skips; a build that has the command but whose schema reads
    as unknown fails everywhere.
    """
    import os

    from apeGmsh.opensees.emitter.live import get_backend_info, get_ops

    ops = get_ops()
    if (not hasattr(ops, "ladrunoSchema")
            and os.environ.get("APEGMSH_FORK_PIN_ENFORCE") != "1"):
        pytest.skip("fork build predates ops.ladrunoSchema (fork PR #972)")
    info = get_backend_info()
    assert info.schema is not None, "ladrunoSchema read as unknown"
    # FORK_PIN ladruno-v20261030b (23d937bc4, c9004f48d plus fork PRs
    # #1043 and #1042, which touch no schema code) answers v1 (fork
    # WP-185, apeGmsh#2517): v0 plus ``capabilities``.
    assert info.schema.schema_version == 1
    assert "mdStageFlagPreserved" in info.schema.capabilities
    assert info.schema.build == ops.ladrunoBuild().strip() == info.build
    assert info.supports_option("LadrunoRCConcrete", "-implexControl") is True
    assert info.has_command("ladrunoSchema") is True
