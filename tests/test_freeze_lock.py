"""Lock: the frozen registries equal their census (ADR 0121 D4).

Invariant 6: each frozen registry equals ``tests/freeze_census.py``. A
member with no census entry, or a census entry with no member behind it,
fails. Invariant 7: every entry added after the freeze point is in
``ADMITTED`` and cites an ADR whose Status, as
``scripts/adr_index.py::parse_adr`` reads it, is Accepted.

A red lock means a registry grew without an accepted ADR. Remove the
member, or accept the ADR first and add an ``ADMITTED`` line citing it.
Never edit ``FROZEN`` or ``FREEZE_POINT_DIGEST`` to make room.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from tests.freeze_census import ADMITTED, FROZEN

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "architecture" / "decisions"

#: sha256 of ``FROZEN`` at the freeze point (see ``_digest``).
FREEZE_POINT_DIGEST = "aab89c40784a30bd7e55ff980381675c26870052d95e8a5117521f37b45bf307"

_spec = importlib.util.spec_from_file_location(
    "adr_index", ROOT / "scripts" / "adr_index.py")
assert _spec is not None and _spec.loader is not None
adr_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adr_index)


def _primitive_families() -> set[str]:
    from tests.families import FAMILIES
    return {f.name for f in FAMILIES}


def _constraint_kinds() -> set[str]:
    from apeGmsh._kernel.records._kinds import ConstraintKind
    return {v for k, v in vars(ConstraintKind).items()
            if k.isupper() and isinstance(v, str)}


class _RecordingServer:
    """Stands in for ``FastMCP``: records every tool ``build_server`` registers.

    It covers ``@mcp.tool()``, ``mcp.tool()(f)``, ``mcp.tool(f)`` and
    ``mcp.add_tool(f)``, under the function's name or ``name=``. Any other
    server attribute raises ``AttributeError``, so a new way of registering
    fails the lock instead of slipping past it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tools: list[str] = []

    def tool(self, name_or_fn: Any = None, *args: Any,
             name: str | None = None, **kwargs: Any) -> Any:
        if callable(name_or_fn):
            return self._record(name_or_fn, name)
        return lambda fn: self._record(fn, name or name_or_fn)

    def add_tool(self, fn: Any, *args: Any, name: str | None = None,
                 **kwargs: Any) -> None:
        self._record(fn, name or (args[0] if args else None))

    def _record(self, fn: Any, name: str | None) -> Any:
        self.tools.append(name or fn.__name__)
        return fn


def _mcp_tools() -> set[str]:
    # Build the real server against a recording stub: the MCP SDK is
    # optional and absent in lock-tests, and reading decorators off the
    # AST misses a tool registered by a direct call or an alias.
    from apeGmsh.studio import mcp

    with mock.patch.object(mcp, "_require_fastmcp", lambda: _RecordingServer):
        server = mcp.build_server()
    assert isinstance(server, _RecordingServer)
    return set(server.tools)


def _habitat_tools() -> set[str]:
    from apeGmsh.studio._profile import HABITAT_TOOLS
    return set(HABITAT_TOOLS)


def _verb_families() -> set[str]:
    from apeGmsh.opensees.emitter import verbs
    return set(verbs.FAMILIES) | {row.family for row in verbs.VERBS.values()}


LIVE: dict[str, Callable[[], set[str]]] = {
    "primitive families": _primitive_families,
    "constraint kinds": _constraint_kinds,
    "studio mcp tools": _mcp_tools,
    "habitat tools": _habitat_tools,
    "emitter verb families": _verb_families,
}


def _digest(frozen: dict[str, frozenset[str]]) -> str:
    canon = json.dumps({k: sorted(v) for k, v in frozen.items()}, sort_keys=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def test_census_names_exactly_the_frozen_registries() -> None:
    assert set(FROZEN) == set(LIVE)
    assert set(ADMITTED) == set(LIVE)


@pytest.mark.parametrize("registry", sorted(LIVE))
def test_registry_equals_census(registry: str) -> None:
    live = LIVE[registry]()
    census = set(FROZEN[registry]) | set(ADMITTED[registry])
    unlisted = sorted(live - census)
    stale = sorted(census - live)
    assert not unlisted, (
        f"{registry}: {unlisted} joined the registry with no census entry. "
        "A new registry member needs an Accepted ADR and an ADMITTED line "
        "citing it (ADR 0121 D4)")
    assert not stale, (
        f"{registry}: census entries {stale} have no member behind them; "
        "delete them from tests/freeze_census.py")


def test_admitted_entries_cite_an_accepted_adr() -> None:
    bad: list[str] = []
    for registry, entries in ADMITTED.items():
        for entry, number in entries.items():
            where = f"{registry} / {entry!r} cites {number!r}"
            if entry in FROZEN[registry]:
                bad.append(f"{where}: already frozen, admit it once")
                continue
            if not re.fullmatch(r"\d{4}", number):
                bad.append(f"{where}: not a four-digit ADR number")
                continue
            files = sorted(DECISIONS.glob(f"{number}-*.md"))
            if len(files) != 1:
                bad.append(f"{where}: {len(files)} ADR files match")
                continue
            _, status = adr_index.parse_adr(files[0])
            if status != "Accepted":
                bad.append(f"{where}: {files[0].name} is {status}, not Accepted")
    assert not bad, "\n".join(bad)


def test_freeze_point_is_pinned() -> None:
    assert _digest(FROZEN) == FREEZE_POINT_DIGEST, (
        "FROZEN changed after the freeze point. A new entry goes in ADMITTED "
        "with an Accepted ADR; a removed member's entry leaves FROZEN only "
        "with the digest updated in the same PR")
