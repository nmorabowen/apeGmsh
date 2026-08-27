"""Studio tests must not inherit a habitat from the developer's shell.

``resolve_root`` reads ``APEGMSH_ROOT`` *and* ``APEGMSH_STUDIO_ROOT``
(ADR 0095 INV-15). Running the suite from inside a habitat — or from a
terminal that sourced ``scripts/_habitat.py`` — would otherwise bind
every cwd-fallback test to that habitat. Tests that want a root set it
themselves; this only clears the ambient one.
"""

from __future__ import annotations

import pytest

from apeGmsh.studio._paths import ROOT_ENVS


@pytest.fixture(autouse=True)
def _clear_habitat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ROOT_ENVS:
        monkeypatch.delenv(name, raising=False)
