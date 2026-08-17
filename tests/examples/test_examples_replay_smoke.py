"""Replay-to-mesh smoke lane for the curated ``examples/`` scripts.

Each included example is copied into a tmp habitat and run through
``ReplayRunner.run_until(..., phase="mesh")`` — geometry + ``generate()``
only, stopping before ``apeSees``/``Results`` (no openseespy, no GL, no
Qt). This is the cheap, CI-safe way to catch import errors and API rot
in ``examples/`` — dogfood pass #1 found ``shoebuckle_arch.py`` had
silently rotted twice before it was fixed in PR #990.

Curation (why each example is in / out) is documented next to EXAMPLES.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from apeGmsh.studio import ReplayRunner

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

EXAMPLES = [
    "shoebuckle_arch.py",
    "partition_frame.py",
    # etabs_import_demo.py: excluded. Its FIXTURE constant is a
    # hardcoded absolute path into the *sibling* apeETABS repo
    # (C:/Users/.../Github/apeETABS/schema/examples/two_story_frame.sm.json),
    # which a CI checkout of apeGmsh alone does not have. Not an
    # ETABS-COM dependency — apeGmsh.interop has no COM/win32 import —
    # just a non-portable fixture path outside this repo.
]


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_replays_to_mesh_gate(name: str, tmp_path: Path) -> None:
    # Copy into tmp_path so .apegmsh/ and any relative outputs the
    # script would otherwise write land in the tmp habitat, never in
    # the repo.
    src = EXAMPLES_DIR / name
    dst = tmp_path / name
    shutil.copy(src, dst)

    runner = ReplayRunner(root=tmp_path)
    result = runner.run_until(dst, phase="mesh")

    assert result.ok, result.error
    # Both curated examples instantiate apeSees(fem) right after the
    # mesh is built, so the mesh-phase gate always stops there (same
    # convention as tests/studio/test_replay.py's
    # test_mesh_phase_generates_but_skips_apesees).
    assert result.stopped_at == "apeSees"
