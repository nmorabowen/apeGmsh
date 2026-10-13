"""Pins for the tag-law ledger's replay waivers (K1-3, #1361).

``compose.py`` still mints tags on replay; ``tag_law_ledger.txt`` waives
each site by name. The reinforce ties mint for an archive below opensees
2.29.0 only (K2b, #1795: a 2.29 archive stores them); the initial-stress
and staged parameters mint for an archive below opensees 2.27.0 only (K1-8,
#1465: a 2.27 archive stores them, and its replay reads them). These tests pin what the re-minting produces: the replayed deck
carries exactly the tags the forward emit minted, in the same order. If
anything changes a replayed tag, the pin fails before a reader keyed on that
tag can disagree with it.

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


def _forward_and_replayed(ops_factory, tmp_path: Path, *, pre_2_27: bool = False, pre_2_29: bool = False) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    """The bridge's own Tcl deck and the deck replayed from its archive
    (``pre_2_27``: from that archive as a pre-2.27 writer left it, so the
    replay re-mints its parameter tags; ``pre_2_29``: as a pre-2.29 writer
    left it, so the replay re-derives its reinforce ties)."""
    fwd = tmp_path / "forward.tcl"
    ops: apeSees = ops_factory()
    ops.tcl(str(fwd), progress=False)
    archive = tmp_path / "model.h5"
    # The factory builds a second session of the same name: give it its
    # own artifact directory, or its end() would meet the first bridge's
    # full model.h5 and refuse it with a warning (#1307).
    import os

    second = tmp_path / "second_run_artifacts"
    second.mkdir()
    previous = os.environ.get("APEGMSH_ARTIFACT_DIR")
    os.environ["APEGMSH_ARTIFACT_DIR"] = str(second)
    try:
        ops_factory().h5(str(archive))
    finally:
        if previous is None:
            os.environ.pop("APEGMSH_ARTIFACT_DIR", None)
        else:
            os.environ["APEGMSH_ARTIFACT_DIR"] = previous
    if pre_2_27:
        from tests.opensees.h5.test_h5_param_tags import (
            downgrade_to_pre_param_tags,
        )

        downgrade_to_pre_param_tags(archive)
    if pre_2_29:
        from tests.opensees.h5.test_h5_embedded_rebar_archive import (
            downgrade_to_pre_embedded_rebar,
        )

        downgrade_to_pre_embedded_rebar(archive)
    replayed = OpenSeesModel.from_h5(str(archive)).build("tcl")
    assert isinstance(replayed, str)
    return fwd.read_text(encoding="utf-8"), replayed


def _param_rows(deck: str) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in _PARAM.finditer(deck)]


def test_pre_2_29_reinforce_tie_replay_reuses_the_forward_tie_tags(
    tmp_path: Path,
) -> None:
    """A pre-2.29 archive: the replay seeds the element counter past the
    replayed elements and re-mints the tie tags; they equal the forward
    tags, which run contiguously above every host element."""
    from tests.opensees.h5.test_reinforce_deck_replay import _reinforced_ops

    fwd, replayed = _forward_and_replayed(
        lambda: _reinforced_ops(bond_by_name=False), tmp_path, pre_2_29=True)
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


def test_pre_2_27_initial_stress_replay_reuses_the_forward_parameter_tags(
    tmp_path: Path,
) -> None:
    """A pre-2.27 flat archive's global initial stress: the replay re-mints
    the parameter tags (1, 2, 3 for sigma_xx/yy/zz) exactly as the bridge
    did."""
    from tests.opensees.h5.test_h5_initial_stress import _build_frame

    fwd, replayed = _forward_and_replayed(
        lambda: _build_frame(with_initial_stress=True), tmp_path,
        pre_2_27=True)
    rows = _param_rows(fwd)
    assert [t for v, t in rows if v == "parameter"] == [1, 2, 3]
    assert _param_rows(replayed) == rows


def test_pre_2_27_staged_replay_params_reuse_the_forward_parameter_tags(
    tmp_path: Path,
) -> None:
    """A pre-2.27 staged archive: the replay's shared allocator re-mints the
    stage initial-stress tags and the absorbing-flip tag exactly as the
    bridge did."""
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
        fwd, replayed = _forward_and_replayed(factory, out, pre_2_27=True)
        rows = _param_rows(fwd)
        assert rows, f"{name}: the fixture mints no parameter tag"
        assert _param_rows(replayed) == rows, name
