"""#1446: a stage-claimed filtered recorder writes its region tags once.

The partitioned pre-plan of filtered recorders used to walk every recorder,
stage-claimed ones included, and give each of their regions a tag that it
wrote on every rank before the first stage. The stage pass then emitted
the recorder through ``materialize``, which minted fresh tags and wrote
the regions again inside the stage. The pre-planned tags were declared and
never referenced: three orphan regions per rank in
:func:`~tests.opensees.contract._tag_streams.stage_claimed_regions`.

The model declares twelve regions (module docstring of the fixture), so
each deck declares exactly the tags 1..12, in a closed-form order: the
oracle is the ordered list of ``region`` lines, so a region written twice
fails it as surely as an orphan does. Each recorder references a region
its deck declares, and a stage-claimed recorder's regions are declared
inside its stage, after the stage's ``domainChange`` has put the stage's
elements in the domain (OpenSees binds a region only to members already
in the domain).
"""
from __future__ import annotations

import re
from typing import Any

import pytest

from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.contract import _tag_streams as ts

#: Global: two named regions, one scoped Rayleigh, one damping attach,
#: one filtered MPCO. Stage ``probe``: two named regions, one scoped
#: Rayleigh, one damping attach, one filtered MPCO, one Ladruno with a
#: filter and an energy region.
_N_REGIONS = 2 + 1 + 1 + 1 + 2 + 1 + 1 + 1 + 2

#: The ``region`` lines of each deck, in deck order. The flat deck writes
#: each region once, in mint order. The partitioned deck writes a region
#: in every rank block that holds a member: rank 0's block writes ``west``
#: (2) and the global MPCO's filter region (1), rank 1's ``east`` (3) and
#: the filter region again; the global Rayleigh and damping regions (4,
#: 5) follow. In stage ``probe`` rank 0 writes ``p_west`` (6, on ``Top``,
#: whose nodes both ranks hold) and rank 1 ``p_east`` (7) and ``p_west``;
#: the stage's Rayleigh, damping and claimed-recorder regions (8..12)
#: follow once.
_DECLARED: dict[str, list[int]] = {
    "staged": list(range(1, _N_REGIONS + 1)),
    "staged_partitioned": [2, 1, 3, 1, 4, 5, 6, 7, 6, 8, 9, 10, 11, 12],
}

_REGION = re.compile(r"^\s*region (\d+)\b")
_REFS = re.compile(r"(?:-R|-G energy) (\d+)")


def _deck(mode: str) -> list[str]:
    em = TclEmitter()
    ts.stage_claimed_regions(mode).build().emit(em)
    return em.lines()


def _declared(lines: list[str]) -> list[int]:
    return [int(m.group(1)) for ln in lines if (m := _REGION.match(ln))]


@pytest.mark.parametrize("mode", ts.STAGE_CLAIMED_MODES)
def test_each_region_tag_is_declared_by_one_site(mode: str) -> None:
    declared = _declared(_deck(mode))
    assert sorted(set(declared)) == list(range(1, _N_REGIONS + 1)), mode
    assert declared == _DECLARED[mode], mode


@pytest.mark.parametrize(("mode", "writer"), [
    ("staged", "_emit_stage_regions"),
    ("staged_partitioned", "_emit_stage_regions_partitioned"),
])
def test_a_duplicated_region_line_fails_the_order(
    mode: str, writer: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: the stage's named-region writer runs twice.

    The deck then declares the very same set of tags, so a comparison of
    sets passes it; the ordered list fails it.
    """
    from apeGmsh.opensees.apesees import BuiltModel

    orig = getattr(BuiltModel, writer)

    def twice(self: Any, *args: Any, **kwargs: Any) -> None:
        orig(self, *args, **kwargs)
        orig(self, *args, **kwargs)

    monkeypatch.setattr(BuiltModel, writer, twice)
    declared = _declared(_deck(mode))
    assert set(declared) == set(_DECLARED[mode])
    assert declared != _DECLARED[mode]


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
