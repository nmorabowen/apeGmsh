"""Pins for the tag-law ledger's replay waivers (K1-3, #1361).

``compose.py`` still mints tags on replay; ``tag_law_ledger.txt`` waives
each site by name, and K2 removes them. Until then, these tests pin what the
re-minting produces: the replayed deck carries exactly the tags the forward
emit minted, in the same order. If K2 (or anything else) changes a replayed
tag, the pin fails before a reader keyed on that tag can disagree with it.

Each test name carries its waiver's name; ``test_tag_law_lock.py`` checks
that every ledger waiver has one.
"""
from __future__ import annotations

import re
from pathlib import Path

from apeGmsh.opensees import OpenSeesModel, apeSees

_PARAM = re.compile(
    r"^\s*(parameter|updateParameter|addToParameter|remove parameter)\s+(\d+)",
    re.M,
)
_ELEMENT = re.compile(r"^\s*element\s+(\S+)\s+(\d+)", re.M)


def _forward_and_replayed(ops_factory, tmp_path: Path) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    """The bridge's own Tcl deck and the deck replayed from its archive."""
    fwd = tmp_path / "forward.tcl"
    ops: apeSees = ops_factory()
    ops.tcl(str(fwd), progress=False)
    archive = tmp_path / "model.h5"
    ops_factory().h5(str(archive))
    replayed = OpenSeesModel.from_h5(str(archive)).build("tcl")
    assert isinstance(replayed, str)
    return fwd.read_text(encoding="utf-8"), replayed


def _param_rows(deck: str) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in _PARAM.finditer(deck)]


def test_reinforce_tie_replay_reuses_the_forward_tie_tags(tmp_path: Path) -> None:
    """Step 8b: the replay seeds the element counter past the replayed
    elements and re-mints the tie tags; they equal the forward tags, which
    run contiguously above every host element."""
    from tests.opensees.h5.test_reinforce_deck_replay import _reinforced_ops

    fwd, replayed = _forward_and_replayed(
        lambda: _reinforced_ops(bond_by_name=False), tmp_path)
    ties_fwd = [
        int(t) for k, t in _ELEMENT.findall(fwd) if k == "LadrunoEmbeddedRebar"]
    ties_rep = [
        int(t) for k, t in _ELEMENT.findall(replayed)
        if k == "LadrunoEmbeddedRebar"]
    hosts = [
        int(t) for k, t in _ELEMENT.findall(replayed)
        if k != "LadrunoEmbeddedRebar"]
    assert len(ties_fwd) >= 2
    assert ties_rep == ties_fwd
    top = max(hosts)
    assert ties_rep == list(range(top + 1, top + 1 + len(ties_rep)))


def test_initial_stress_replay_reuses_the_forward_parameter_tags(
    tmp_path: Path,
) -> None:
    """A flat archive's global initial stress: the replay re-mints the
    parameter tags (1, 2, 3 for sigma_xx/yy/zz) exactly as the bridge did."""
    from tests.opensees.h5.test_h5_initial_stress import _build_frame

    fwd, replayed = _forward_and_replayed(
        lambda: _build_frame(with_initial_stress=True), tmp_path)
    rows = _param_rows(fwd)
    assert [t for v, t in rows if v == "parameter"] == [1, 2, 3]
    assert _param_rows(replayed) == rows


def test_staged_replay_params_reuse_the_forward_parameter_tags(
    tmp_path: Path,
) -> None:
    """A staged archive: the replay's shared allocator re-mints the stage
    initial-stress tags and the absorbing-flip tag exactly as the bridge did."""
    from tests.opensees.h5.test_h5_stages_reader import (
        _real_kitchen_sink_bridge,
        _real_two_stage_bridge,
    )

    for name, factory in (
        ("initial_stress", _real_two_stage_bridge),
        ("absorbing", _real_kitchen_sink_bridge),
    ):
        out = tmp_path / name
        out.mkdir()
        fwd, replayed = _forward_and_replayed(factory, out)
        rows = _param_rows(fwd)
        assert rows, f"{name}: the fixture mints no parameter tag"
        assert _param_rows(replayed) == rows, name
