"""``_stable_section_tag`` gives one name the same tag in every process.

Its docstring promised a "Deterministic positive int tag derived from a
section name", but it took the builtin ``hash()`` of the name, which
Python salts per process (``PYTHONHASHSEED``): measured on the unfixed
source, ``"LayeredShell_A"`` came out ``1509370562`` under seed 1 and
``1288700299`` under seed 2. It is now a CRC-32 of the UTF-8 name.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from apeGmsh.results.capture.spec import _stable_section_tag

SRC = Path(__file__).resolve().parents[2] / "src"
PROBE = (
    "from apeGmsh.results.capture.spec import _stable_section_tag as t; "
    "print(t('LayeredShell_A'))"
)


def _tag_in_a_fresh_process(seed: str) -> int:
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "APEGMSH_QUIET": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")])),
    }
    done = subprocess.run(
        [sys.executable, "-c", PROBE],
        env=env, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return int(done.stdout.strip().splitlines()[-1])


def test_the_tag_is_the_same_under_two_hash_seeds() -> None:
    assert _tag_in_a_fresh_process("1") == _tag_in_a_fresh_process("2")


def test_the_tag_is_pinned() -> None:
    # A CRC-32 is fixed by its definition, so this value holds on every
    # platform and Python version. If it moves, every tag moved with it.
    assert _stable_section_tag("LayeredShell_A") == 1788243603


@pytest.mark.parametrize("name", ["", "a", "LayeredShell_A", "Hormigón"])
def test_the_tag_is_a_positive_opensees_int(name: str) -> None:
    tag = _stable_section_tag(name)
    assert isinstance(tag, int)
    assert 1 <= tag <= 2**31 - 2


def test_the_tag_is_the_crc32_of_the_utf8_name() -> None:
    name = "Hormigón"
    assert _stable_section_tag(name) == (zlib.crc32(name.encode("utf-8")) % (2**31 - 1) or 1)
