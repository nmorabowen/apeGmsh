"""M2-S8 (#3003), the 2026-11-17 acceptance: an empty modes group is no mode.

Since OpenSees#1084 (release ``ladruno-v20261116``, the ``FORK_PIN``) a
recorder asked for ``modesOfVibration`` lays out ``MODES_OF_VIBRATION(U|R)``
on EVERY stage, with ``ID`` and an empty ``DATA`` group, and adds a
``DATA/STEP_<k>`` snapshot only when an eigen analysis runs (fork schema
§7b.2, "the groups of a modes result"). A stage whose group holds no snapshot
is not a modal stage: the readers must list no mode stage for it, and its own
steps stay readable as a host stage's.

Oracles: the raw layout h5py reads (which stage holds which ``STEP_<k>``),
and the eigenvalues the snapshot's ``LAMBDA`` attributes record. The
committed ``killed_swmr.ladruno`` (made with the release build; see its
README) has a snapshot in ``MODEL_STAGE[1]`` and the empty groups in
``MODEL_STAGE[2]``; the live test makes a cleanly closed file of the same
deck at the pin.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import h5py
import pytest

from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._ladruno_open import open_ladruno
from apeGmsh.results.readers._protocol import ResultLevel

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno_killed"
KILLED = FIX / "killed_swmr.ladruno"
MODES_U = "RESULTS/ON_NODES/MODES_OF_VIBRATION(U)"


def _layout(path: Path) -> "dict[str, list[str]]":
    """``{stage: [STEP_<k>, ...]}`` of each stage's modes group, as h5py
    lists it; and the snapshots' eigenvalues."""
    out: "dict[str, list[str]]" = {}
    with open_ladruno(path) as o:
        for name in o.h5:
            if not name.startswith("MODEL_STAGE["):
                continue
            g = o.h5[f"{name}/{MODES_U}"]
            assert "ID" in g and isinstance(g["DATA"], h5py.Group)
            out[name] = sorted(g["DATA"])
    return out


def _lambdas(path: Path, stage: str, step: str) -> "list[float]":
    with open_ladruno(path) as o:
        data = o.h5[f"{stage}/{MODES_U}/DATA/{step}"]
        return [float(data[k].attrs["LAMBDA"][0]) for k in sorted(data)]


def _check(path: Path) -> None:
    layout = _layout(path)
    # The fork lays the modes group out on both stages; only the first holds
    # a snapshot (the deck runs ``eigen 2`` once, in MODEL_STAGE[1]).
    assert layout == {"MODEL_STAGE[1]": ["STEP_10"], "MODEL_STAGE[2]": []}
    lams = _lambdas(path, "MODEL_STAGE[1]", "STEP_10")
    with LadrunoReader(path) as r:
        stages = r.stages()
        modes = [s for s in stages if s.kind == "mode"]
        hosts = [s for s in stages if s.kind != "mode"]
        assert [s.name for s in hosts] == ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]
        # Every mode stage comes from MODEL_STAGE[1]'s snapshot; the empty
        # group of MODEL_STAGE[2] gives none.
        assert [s.id for s in modes] == ["stage_0_mode_1", "stage_0_mode_2"]
        assert [s.eigenvalue for s in modes] == pytest.approx(lams, rel=0, abs=0)
        # The listing order is the host, then its modes, then the next host.
        assert [s.id for s in stages] == [
            "stage_0", "stage_0_mode_1", "stage_0_mode_2", "stage_1"]
        # The host without a snapshot keeps its steps and reads as a host.
        s2 = hosts[1]
        assert s2.n_steps > 0
        assert r.available_components(s2.id, ResultLevel.NODES) == [
            "displacement_x", "displacement_y"]
        assert r.read_nodes(s2.id, "displacement_x").values.shape[0] == s2.n_steps


def test_the_empty_modes_group_of_a_killed_run_is_no_mode() -> None:
    _check(KILLED)


@pytest.mark.ladruno_fork
def test_live_a_closed_run_at_the_pin_lists_modes_only_where_a_snapshot_is(tmp_path) -> None:
    """The same deck at the pin, run to completion and closed."""
    spec = importlib.util.spec_from_file_location(
        "_apegmsh_make_killed_run", FIX / "make_killed_run.py")
    assert spec is not None and spec.loader is not None
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    out = tmp_path / "closed.ladruno"
    bin_dir = os.environ.get("APEGMSH_OPENSEES_BIN") or "-"
    proc = mk.start(bin_dir, out, tmp_path, sleep=0.0, nsteps=30)
    try:
        lines = proc.communicate(timeout=120)[0].splitlines()
    finally:
        proc.kill()
        proc.wait(timeout=120)
    build = next((ln.split()[1] for ln in lines if ln.startswith("BUILD ")), "")
    pin = mk.pinned_sha()
    if build != pin:
        msg = f"the child ran build {build[:12]!r}, not the FORK_PIN build {pin[:12]}"
        if os.environ.get("APEGMSH_FORK_PIN_ENFORCE") == "1":
            pytest.fail(msg)
        pytest.skip(msg + "; set APEGMSH_OPENSEES_BIN to the pinned build's dist/bin")
    assert proc.returncode == 0 and "CHILD DONE" in lines
    with open(out, "rb") as fh:
        assert fh.read(12)[11] == 0x00   # closed cleanly
    _check(out)
