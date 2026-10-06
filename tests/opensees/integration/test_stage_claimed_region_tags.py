"""#1446: a stage-claimed filtered recorder writes its region tags once.

The partitioned pre-plan of filtered recorders used to walk every recorder,
stage-claimed ones included, and give each of their regions a tag that it
wrote on every rank before the first stage. The stage pass then emitted
the recorder through ``materialize``, which minted fresh tags and wrote
the regions again inside the stage. The pre-planned tags were declared and
never referenced: three orphan regions per rank in
:func:`~tests.opensees.contract._tag_streams.stage_claimed_regions`.

The oracle is a count: the model declares twelve regions (module
docstring of the fixture), so each deck declares exactly the tags 1..12.
Each recorder references a region its deck declares, and a stage-claimed
recorder's regions are declared inside its stage, after the stage's
``domainChange`` has put the stage's elements in the domain (OpenSees
binds a region only to members already in the domain).
"""
from __future__ import annotations

import re

import pytest

from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.contract import _tag_streams as ts

#: Global: two named regions, one scoped Rayleigh, one damping attach,
#: one filtered MPCO. Stage ``probe``: two named regions, one scoped
#: Rayleigh, one damping attach, one filtered MPCO, one Ladruno with a
#: filter and an energy region.
_N_REGIONS = 2 + 1 + 1 + 1 + 2 + 1 + 1 + 1 + 2

_REGION = re.compile(r"^\s*region (\d+)\b")
_REFS = re.compile(r"(?:-R|-G energy) (\d+)")


def _deck(mode: str) -> list[str]:
    em = TclEmitter()
    ts.stage_claimed_regions(mode).build().emit(em)
    return em.lines()


@pytest.mark.parametrize("mode", ts.STAGE_CLAIMED_MODES)
def test_each_region_tag_is_declared_by_one_site(mode: str) -> None:
    lines = _deck(mode)
    declared = {int(m.group(1)) for ln in lines if (m := _REGION.match(ln))}
    assert sorted(declared) == list(range(1, _N_REGIONS + 1)), mode


@pytest.mark.parametrize("mode", ts.STAGE_CLAIMED_MODES)
def test_recorders_reference_declared_regions(mode: str) -> None:
    lines = _deck(mode)
    probe = lines.index("# === Stage: probe ===")
    declared: dict[int, int] = {}
    for i, ln in enumerate(lines):
        if m := _REGION.match(ln):
            declared.setdefault(int(m.group(1)), i)
        if not ln.lstrip().startswith("recorder "):
            continue
        for ref in map(int, _REFS.findall(ln)):
            assert ref in declared, f"{mode}: {ln!r} references no region"
            if i > probe:
                assert declared[ref] > probe, (
                    f"{mode}: the stage-claimed recorder {ln!r} references "
                    f"region {ref}, declared before its stage")
