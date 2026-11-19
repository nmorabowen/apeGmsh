"""``live-fork``: the ``mpm`` fork-surface snapshot is the parser at ``FORK_PIN``.

The suite lane checks the snapshot offline (``tests/opensees/parity/
test_mpm_command_parity.py``): its sha against ``FORK_PIN`` and its names
against the members and ``NOT_OFFERED``. Offline, it cannot tell a refreshed
snapshot from one whose ``fork_sha`` was typed in by hand. This test reads
``OPS_LadrunoMPM.cpp`` at ``FORK_PIN``'s sha from the public fork and runs the
refresh script's own extractor on it, so a hand-edited snapshot fails here.
``land_pr.py`` refuses a ``FORK_PIN`` change without a green ``live-fork``
run on its head, so this test gates every pin bump before it merges.

A failed read fails the test: a skip here would pass a stale snapshot.
"""
from __future__ import annotations

import importlib.util
import urllib.request
from typing import Any

import pytest

from tests.opensees.parity import mpm_fork_surface as surface

pytestmark = pytest.mark.ladruno_fork

RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"


def _refresh_script() -> Any:
    path = surface.REPO / "scripts" / "refresh_mpm_fork_surface.py"
    spec = importlib.util.spec_from_file_location("_refresh_mpm_fork_surface", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser_at_pin(r: Any, pin: dict[str, str]) -> str:
    url = RAW.format(repo=pin["repo"], sha=pin["sha"], path=r.SOURCE)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:  # any failed read is a failure, never a skip
        pytest.fail(f"could not read the fork's mpm parser at FORK_PIN "
                    f"({url}): {type(e).__name__}: {e}")


def test_mpm_fork_surface_snapshot_is_the_parser_at_fork_pin() -> None:
    r = _refresh_script()
    pin = r.read_pin()
    snap = surface.load_snapshot()
    assert (snap["fork_sha"], snap["fork_tag"]) == (pin["sha"], pin["tag"]), (
        f"the snapshot was not read at FORK_PIN: run `{surface.REFRESH}`")
    fork = r.extract_subcommands(_parser_at_pin(r, pin))
    assert snap["subcommands"] == fork, (
        f"the mpm fork-surface snapshot is not the parser at FORK_PIN "
        f"{pin['tag']} ({pin['sha'][:9]}): only in the parser "
        f"{sorted(set(fork) - set(snap['subcommands']))}, only in the snapshot "
        f"{sorted(set(snap['subcommands']) - set(fork))}; run `{surface.REFRESH}`")
