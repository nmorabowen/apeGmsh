"""Tag law (K1-3, #1361): a fresh interpreter mints the same tag stream.

``BuiltModel.emit`` mints derived tags on a fresh allocator per emit.
Anything keyed on ``id()`` or on a ``str`` hash (whose seed changes per
process unless ``PYTHONHASHSEED`` is fixed) could reorder that minting
between runs, and an in-process test cannot see it. This test builds every
model in ``tests/opensees/contract/_tag_streams.py`` in-process and again
in fresh interpreters under two fixed, different hash seeds, and requires
the in-order ``(kind, tag)`` streams to be identical.

The models cover the flat, partitioned, staged, staged-partitioned and
split emit paths, with element fan-out, per-element ``geomTransf``
overrides, recorder regions and parameter tags.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.opensees.contract import _tag_streams as ts

pytestmark = pytest.mark.subprocess

_ROOT = Path(__file__).resolve().parents[3]

_CHILD = (
    "import json, sys\n"
    "from tests.opensees.contract._tag_streams import all_streams\n"
    "with open(sys.argv[1], 'w', encoding='utf-8') as f:\n"
    "    json.dump(all_streams(), f)\n"
)


def _streams_in_subprocess(out: Path, hash_seed: str) -> dict[str, object]:
    env = dict(os.environ)
    # The worktree's sources, never the editable install's checkout.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT / "src"), str(_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(out)],
        cwd=str(_ROOT), env=env, capture_output=True, text=True,
        timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    data: dict[str, object] = json.loads(out.read_text(encoding="utf-8"))
    return data


def test_tag_streams_match_a_fresh_interpreter(tmp_path: Path) -> None:
    here = ts.all_streams()
    assert any(k.endswith("/flat") for k in here)
    assert any(k.endswith("/partitioned") for k in here)
    assert any(k.endswith("/staged") for k in here)
    assert all(here.values()), "a model drove no tagged verb"

    for seed in ("0", "4242"):
        there = _streams_in_subprocess(tmp_path / f"streams_{seed}.json", seed)
        assert sorted(there) == sorted(here)
        for name, stream in here.items():
            assert there[name] == stream, (
                f"{name}: the (kind, tag) stream differs in a fresh "
                f"interpreter (PYTHONHASHSEED={seed})"
            )
