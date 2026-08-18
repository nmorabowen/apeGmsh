"""ADR 0098 S6a — the open policy: what a window boots with, and saves.

§11 puts "the restore/save snapshot semantics" in the flip and §1 says
what they are. This file is the policy's oracle, and it is Qt-free
because the policy is: ``boot_session`` decides, the window projects.

**INV-SESSION-OPEN — the window always opens.** The three tests that
actually earn this file are the ones where the file on disk is bad:

* the legacy gate firing IN ANGER (a real v13 file from the real old
  writer, met by the real loader at the real adopted path);
* the SECOND flipped open, where ``rename_legacy_aside`` refuses
  because an aside is already there;
* a snapshot the loader refuses on schema grounds.

All three must open a window, keep every byte on disk, and — the part
that is easy to get wrong — **disarm auto-save**, because at this flip
the save target IS the file that was just refused. A boot that opens
fresh with auto-save armed would overwrite on close exactly what the
refusal protected, and every "never destroys" test in
``test_snapshot_legacy_gate.py`` would still pass while the product
destroyed the file one close later. That is why the assertions here are
about the FILES after a simulated close, not only about the notice.

Degradation is not refusal (plan decision 15): a snapshot naming a
stage these results no longer have restores, notices, and keeps
auto-save. Told apart from a schema violation on purpose.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import (
    LEGACY_SUFFIX,
    Contour,
    Instant,
    default_snapshot_path,
    save_snapshot,
    snapshot,
)
from apeGmsh.results.session._boot import (
    boot_session,
    save_on_close,
    viewer_snapshot_path,
)
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams._session import serialize_session

from tests.conftest import _open_model_from_h5

STAGE = "grav"
N_STEPS = 3


# =====================================================================
# Fixture — the smallest real Results with a stage to validate against
# =====================================================================

@pytest.fixture
def results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    disp = np.tile(node_ids.astype(np.float64), (N_STEPS, 1))

    path = tmp_path / "boot.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.arange(N_STEPS, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_z": disp},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _saved_session(results, *, contour: str = "displacement_z"):
    """Arrange something recognisable and save it where the flip looks."""
    session = results.session()
    session.panes[0].contour = Contour(contour)
    path = save_snapshot(session)
    return session, path


def _write_v13(path: Path) -> dict:
    """A genuine old-window envelope, from the REAL old writer.

    Not a hand-written dict: the gate has to fire on what the retired
    window actually wrote on close, which is the whole point of testing
    the flip rather than the predicate.
    """
    payload = serialize_session(
        specs=[],
        results_path=str(path).replace(".viewer-session.json", ""),
        fem_snapshot_id="snap-1",
        active_stage_id=STAGE,
        active_step=1,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# =====================================================================
# The adopted path (plan decision 11)
# =====================================================================

def test_the_flip_adopted_the_old_window_s_filename(results):
    """Decision 11: the save target IS the old viewer's name now.

    The whole legacy gate exists because of this line. If the new
    session still wrote ``.session.json`` the two files would never
    meet and every refusal below would be unreachable.
    """
    path = viewer_snapshot_path(results)

    assert path is not None
    assert path.name.endswith(".viewer-session.json")
    assert path == default_snapshot_path(results._path)


def test_in_memory_results_neither_restores_nor_saves(results):
    results._path = None

    boot = boot_session(results, restore=True, save=True)

    assert viewer_snapshot_path(results) is None
    assert boot.save_path is None
    assert boot.autosaves is False
    assert save_on_close(boot) is None


# =====================================================================
# The ordinary paths
# =====================================================================

def test_no_saved_session_boots_the_default_picture(results):
    boot = boot_session(results, restore=True, save=True)

    assert boot.restored is False
    assert boot.notices == ()
    assert len(boot.session.panes) == 1
    assert boot.session.panes[0].slots == {}
    assert boot.autosaves is True


def test_restore_true_restores_and_stays_armed(results):
    _saved_session(results)

    boot = boot_session(results, restore=True, save=True)

    assert boot.restored is True
    assert boot.source == viewer_snapshot_path(results)
    assert boot.session.panes[0].contour.quantity == "displacement_z"
    assert boot.autosaves is True


def test_restore_false_ignores_the_file_but_still_saves(results):
    _, path = _saved_session(results)

    boot = boot_session(results, restore=False, save=True)

    assert boot.restored is False
    assert boot.session.panes[0].slots == {}
    # Explicitly asked to start fresh AND to save: overwriting on close
    # is the instruction, not an accident.
    assert boot.save_path == path


def test_save_false_disarms_without_touching_restore(results):
    _saved_session(results)

    boot = boot_session(results, restore=True, save=False)

    assert boot.restored is True
    assert boot.autosaves is False
    assert save_on_close(boot) is None


def test_save_on_close_writes_what_is_on_screen(results):
    boot = boot_session(results, restore=True, save=True)
    boot.session.panes[0].contour = Contour("displacement_z")

    written = save_on_close(boot)

    assert written == viewer_snapshot_path(results)
    on_disk = json.loads(written.read_text(encoding="utf-8"))
    # ``saved_at`` is in the payload by design (S5a), so two snapshots
    # of one session are never byte-identical — compare everything else.
    assert on_disk.pop("saved_at")
    expected = snapshot(boot.session)
    expected.pop("saved_at")
    assert on_disk == expected


def test_an_unknown_restore_token_is_an_argument_error(results):
    """Not a file problem — a caller problem, and silently meaning
    "restore" is the one outcome nobody asked for."""
    with pytest.raises(ValueError, match="True, False or 'prompt'"):
        boot_session(results, restore="yes")


# =====================================================================
# restore="prompt"
# =====================================================================

def test_prompt_restores_when_confirmed(results):
    _, path = _saved_session(results)
    asked: list = []

    def _confirm(p, restored):
        asked.append((p, restored))
        return True

    boot = boot_session(results, restore="prompt", confirm=_confirm)

    assert [p for p, _ in asked] == [path]
    assert asked[0][1].session.panes[0].contour is not None
    assert boot.restored is True


def test_prompt_declined_boots_fresh_and_keeps_auto_save(results):
    """Declining is a choice, not a failure: the dialog says the saved
    session is replaced on close, so auto-save is left as asked."""
    _, path = _saved_session(results)

    boot = boot_session(
        results, restore="prompt", confirm=lambda p, r: False,
    )

    assert boot.restored is False
    assert boot.session.panes[0].slots == {}
    assert boot.save_path == path


def test_prompt_with_nobody_to_ask_declines(results):
    """``confirm=None`` is headless. "Ask first" cannot be honoured by
    guessing, so it declines rather than restoring silently."""
    _saved_session(results)

    boot = boot_session(results, restore="prompt", confirm=None)

    assert boot.restored is False


def test_prompt_is_not_asked_when_there_is_no_file(results):
    asked: list = []

    boot = boot_session(
        results, restore="prompt",
        confirm=lambda p, r: asked.append(p) or True,
    )

    assert asked == []
    assert boot.restored is False


# =====================================================================
# THE LEGACY GATE, IN ANGER — first flipped open
# =====================================================================

def test_first_flipped_open_renames_v13_aside_and_boots_fresh(
    results, capsys,
):
    """The upgrade. A real v13 file sits at the newly adopted path."""
    path = viewer_snapshot_path(results)
    original = _write_v13(path)

    boot = boot_session(results, restore=True, save=True)
    out = capsys.readouterr().out

    aside = Path(str(path) + LEGACY_SUFFIX)
    assert aside.exists(), "the v13 file must be kept, not deleted"
    assert json.loads(aside.read_text(encoding="utf-8")) == original
    assert not path.exists(), "the adopted path is free after the rename"
    assert "old viewer session" in out
    assert aside.name in out

    assert boot.restored is False
    assert boot.session.panes[0].slots == {}
    # The path is FREE now — nothing left there to destroy — so this
    # window may save.
    assert boot.save_path == path


def test_after_the_rename_the_window_s_own_save_lands_on_the_path(
    results,
):
    """End to end: old writer → flip → open → close → the new file is
    the one at the adopted path, and the v13 bytes are still aside."""
    path = viewer_snapshot_path(results)
    original = _write_v13(path)

    boot = boot_session(results, restore=True, save=True)
    boot.session.panes[0].contour = Contour("displacement_z")
    written = save_on_close(boot)

    assert written == path
    fresh = json.loads(path.read_text(encoding="utf-8"))
    assert fresh["kind"] == "apegmsh.results.session"
    aside = Path(str(path) + LEGACY_SUFFIX)
    assert json.loads(aside.read_text(encoding="utf-8")) == original


def test_the_second_open_reads_back_what_the_first_one_saved(results):
    """The point of the whole slice: close the window, open it again,
    the picture is there."""
    path = viewer_snapshot_path(results)
    _write_v13(path)

    first = boot_session(results, restore=True, save=True)
    first.session.panes[0].contour = Contour("displacement_z")
    first.session.time = Instant(STAGE, 1)
    save_on_close(first)

    second = boot_session(results, restore=True, save=True)

    assert second.restored is True
    assert second.session.panes[0].contour.quantity == "displacement_z"
    assert second.session.time == Instant(STAGE, 1)


# =====================================================================
# THE DOUBLE UPGRADE — upgrade, downgrade, upgrade
# =====================================================================

def _downgrade_then_upgrade(results) -> Path:
    """Play the sequence that takes the ``.legacy`` slot.

    1. flipped open renames the user's v13 aside, saves a new session;
    2. they run the OLD build, which overwrites the adopted path with
       v13 again on close;
    3. they upgrade again.
    """
    path = viewer_snapshot_path(results)
    _write_v13(path)                                    # their real v13
    first = boot_session(results, restore=True, save=True)
    save_on_close(first)                                # new schema now
    _write_v13(path)                                    # downgraded run
    return path


def test_second_upgrade_opens_a_window_instead_of_refusing(results):
    """A human who just wanted their window back gets it."""
    path = _downgrade_then_upgrade(results)

    boot = boot_session(results, restore=True, save=True)

    assert boot.session is not None
    assert len(boot.session.panes) == 1
    assert boot.restored is False


def test_second_upgrade_disarms_auto_save_so_nothing_is_destroyed(
    results,
):
    """The assertion this whole file exists for.

    ``rename_legacy_aside`` refused, so the adopted path still holds
    the human's v13. If auto-save stayed armed, closing the window
    would overwrite it — and every "never destroys" test would still be
    green.
    """
    path = _downgrade_then_upgrade(results)
    before = path.read_text(encoding="utf-8")
    aside = Path(str(path) + LEGACY_SUFFIX)
    aside_before = aside.read_text(encoding="utf-8")

    boot = boot_session(results, restore=True, save=True)
    assert boot.autosaves is False

    assert save_on_close(boot) is None
    assert path.read_text(encoding="utf-8") == before
    assert aside.read_text(encoding="utf-8") == aside_before


def test_second_upgrade_says_which_file_to_move(results):
    path = _downgrade_then_upgrade(results)

    boot = boot_session(results, restore=True, save=True)

    assert len(boot.notices) == 1
    notice = boot.notices[0]
    assert path.name in notice
    assert (path.name + LEGACY_SUFFIX) in notice
    assert "already exists" in notice


# =====================================================================
# Refusals that are NOT the legacy gate
# =====================================================================

def test_a_schema_violation_opens_fresh_disarmed_and_keeps_the_file(
    results,
):
    """A near-miss, not a missing key: the marker is RIGHT and the slot
    category is wrong. A file with no ``kind`` at all would be refused
    by the legacy/marker branch instead, and would prove nothing about
    the catalog being closed.
    """
    session, path = _saved_session(results)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "apegmsh.results.session"
    slots = payload["panes"][0]["slots"]
    slots["fiber_section"] = slots.pop("contour")
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    boot = boot_session(results, restore=True, save=True)

    assert boot.restored is False
    assert boot.autosaves is False
    assert any("fiber_section" in n for n in boot.notices)
    assert save_on_close(boot) is None
    assert path.read_text(encoding="utf-8") == before


def test_unreadable_json_opens_fresh_disarmed_and_keeps_the_file(
    results,
):
    path = viewer_snapshot_path(results)
    path.write_text("{not json at all", encoding="utf-8")

    boot = boot_session(results, restore=True, save=True)

    assert boot.restored is False
    assert boot.autosaves is False
    assert boot.notices and path.name in boot.notices[0]
    assert save_on_close(boot) is None
    assert path.read_text(encoding="utf-8") == "{not json at all"


# =====================================================================
# Degradation is NOT refusal (plan decision 15)
# =====================================================================

def test_a_dead_stage_degrades_with_a_notice_and_still_saves(results):
    """The data-mismatch family. The file is a valid session; the
    results moved. Every pane, slot and scope survives, the instant
    drops, and the window keeps its save target — it is showing what
    the file said, as faithfully as these results allow."""
    session = results.session()
    session.panes[0].contour = Contour("displacement_z")
    payload = snapshot(session)
    payload["time"] = {"stage": "a-stage-that-was-renamed", "step": 0}
    path = viewer_snapshot_path(results)
    path.write_text(json.dumps(payload), encoding="utf-8")

    boot = boot_session(results, restore=True, save=True)

    assert boot.restored is True
    assert boot.session.panes[0].contour.quantity == "displacement_z"
    assert boot.session.time is None
    assert any("a-stage-that-was-renamed" in n for n in boot.notices)
    assert boot.autosaves is True
