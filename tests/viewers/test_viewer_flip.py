"""ADR 0098 S6a — the flip: ``viewer()`` opens a ``ResultsSession``.

§11: ``viewer()`` "flips to ``session().show()`` at S6, together with
the ``python -m apeGmsh.viewers`` subprocess path and the restore/save
snapshot semantics. After the flip ``viewer()`` returns the
``ResultsSession``."

Three surfaces, one PR, and this file covers all three:

* the door — what ``viewer()`` constructs and returns, per blocking
  mode (plan decision 13: only the blocking path returns the session;
  ``blocking=False`` keeps ``Popen``; the notebook fallback keeps the
  ``WebViewer`` handle — documented, not unified);
* the de-publication — ``apeGmsh.viewers`` no longer exports
  ``ResultsViewer``, while the MODULE stays importable, because S6a is
  de-publication and S6b is deletion;
* the argv hop — the child process must honour the same open policy
  the caller asked for, which before this flip it silently did not.

The window itself is stubbed at ``ResultsSession.show`` for the default
lane. That is the real seam (``Results._show_session_window`` calls it
and nothing else), so stubbing it tests the door rather than Qt; the
qt-marked smoke at the bottom drives the real window under xvfb.

The policy's own oracle — every restore/save/refusal branch — lives in
``tests/results/session/test_boot_policy.py``, Qt-free. Here we only
prove ``viewer()`` is wired to it, plus the one END-TO-END sequence
that no unit can stand in for: the old writer's v13 file meeting the
flipped door on a real upgrade.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.session import (
    LEGACY_SUFFIX,
    Contour,
    ResultsSession,
)
from apeGmsh.results.session._boot import viewer_snapshot_path
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams._session import serialize_session

from tests.conftest import _open_model_from_h5

STAGE = "grav"


@pytest.fixture
def small_results(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "flip.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name=STAGE, kind="static", stage_id=STAGE,
            time=np.array([0.0, 0.5, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={"displacement_x": np.zeros((3, node_ids.size))},
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


@pytest.fixture
def shown(monkeypatch):
    """Stub the window at the one seam ``viewer()`` uses.

    Records the session it was asked to show, so a test can assert the
    door handed the window the session the policy built — and can mutate
    that session "while it is open" before save-on-close runs.
    """
    seen: list = []

    def _show(self, *, blocking=True, title=None):
        seen.append({"session": self, "blocking": blocking,
                     "title": title})
        return None

    monkeypatch.setattr(ResultsSession, "show", _show)
    return seen


def _fake_popen(monkeypatch) -> dict:
    captured: dict = {}

    class _FakePopen:
        def __init__(self, args, *_, **__):
            captured["args"] = list(args)

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    captured["cls"] = _FakePopen
    return captured


def _write_v13(results) -> tuple[Path, dict]:
    path = viewer_snapshot_path(results)
    payload = serialize_session(
        specs=[],
        results_path=str(results._path),
        fem_snapshot_id="snap-1",
        active_stage_id=STAGE,
        active_step=1,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, payload


# =====================================================================
# The door — what viewer() opens and returns
# =====================================================================

def test_viewer_opens_a_session_window(small_results, shown):
    out = small_results.viewer(restore_session=False)

    assert len(shown) == 1
    assert shown[0]["session"] is out
    assert shown[0]["blocking"] is True


def test_viewer_returns_the_session_not_a_viewer(small_results, shown):
    """Decision 13 / §11: after the flip the blocking path returns the
    ``ResultsSession`` — still live once the window is gone."""
    out = small_results.viewer(restore_session=False)

    assert isinstance(out, ResultsSession)
    assert out.results is small_results
    assert len(out.panes) == 1


def test_viewer_boots_the_default_empty_mesh_view(small_results, shown):
    """Consequences: ``viewer()`` becomes ``session().show()`` "with a
    default empty mesh view"."""
    out = small_results.viewer(restore_session=False)
    view = out.panes[0]

    assert view.slots == {}
    assert view.deform is None


def test_viewer_does_not_construct_the_retired_viewer(
    small_results, shown, monkeypatch,
):
    """The flip's negative half. ``results_viewer`` is still importable
    (S6a is de-publication, S6b is deletion) — but this door must not
    touch it."""
    from apeGmsh.viewers import results_viewer as viewer_mod

    def _boom(*_a, **_kw):
        raise AssertionError(
            "viewer() constructed the retired ResultsViewer"
        )

    monkeypatch.setattr(viewer_mod, "ResultsViewer", _boom)
    small_results.viewer(restore_session=False)

    assert len(shown) == 1


def test_viewer_passes_the_title_through(small_results, shown):
    small_results.viewer(title="My Run", restore_session=False)

    assert shown[0]["title"] == "My Run"


def test_skip_env_still_short_circuits(small_results, shown, monkeypatch):
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")

    assert small_results.viewer() is None
    assert shown == []


def test_skip_env_touches_no_file(small_results, monkeypatch):
    """The skip is BEFORE the policy: a CI run must not rename a v13
    file aside as a side effect of a cell it did not execute."""
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    path, payload = _write_v13(small_results)

    small_results.viewer()

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not Path(str(path) + LEGACY_SUFFIX).exists()


def test_cuts_kwarg_is_retired(small_results, shown):
    """§1: ``viewer(cuts=)`` is retired; cuts are clip-on-a-view."""
    with pytest.raises(TypeError, match="cuts"):
        small_results.viewer(cuts=[])


# =====================================================================
# De-publication (not deletion)
# =====================================================================

def test_viewers_package_no_longer_exports_results_viewer():
    import apeGmsh.viewers as viewers

    assert "ResultsViewer" not in viewers.__all__
    assert not hasattr(viewers, "ResultsViewer")


def test_the_root_package_no_longer_exports_it_either():
    """There were TWO exports, and this is the more public one.

    ``from apeGmsh import ResultsViewer`` was the shortest way to reach
    the retired class; de-publishing only ``apeGmsh.viewers`` would have
    left the flip cosmetic.
    """
    import apeGmsh

    assert "ResultsViewer" not in apeGmsh.__all__
    assert not hasattr(apeGmsh, "ResultsViewer")


def test_the_module_is_still_importable_by_path():
    """S6a de-publishes; S6b deletes. The ``show_web`` hatch and
    ``export_animation`` (INV-11) still import it by module path."""
    from apeGmsh.viewers.results_viewer import ResultsViewer

    assert ResultsViewer is not None


def test_the_other_viewers_are_untouched():
    import apeGmsh.viewers as viewers

    assert {"ModelViewer", "MeshViewer", "GeomTransfViewer"} <= set(
        viewers.__all__
    )


# =====================================================================
# Decision 13 — the non-blocking returns stay what they were
# =====================================================================

def test_non_blocking_still_returns_popen(small_results, monkeypatch):
    captured = _fake_popen(monkeypatch)

    handle = small_results.viewer(blocking=False)

    assert isinstance(handle, captured["cls"])
    assert captured["args"][:3] == [sys.executable, "-m", "apeGmsh.viewers"]


def test_notebook_in_memory_still_falls_back_to_web(
    small_results, monkeypatch,
):
    module = sys.modules[Results.__module__]
    monkeypatch.setattr(module, "_in_notebook_kernel", lambda: True)
    sentinel = object()
    monkeypatch.setattr(Results, "show_web", lambda self, **kw: sentinel)
    small_results._path = None

    assert small_results.viewer() is sentinel


# =====================================================================
# The argv hop — the child honours the caller's policy
# =====================================================================

def test_default_policy_adds_no_flags(small_results, monkeypatch):
    """The common argv is unchanged, so nothing downstream that reads it
    has to learn two shapes."""
    captured = _fake_popen(monkeypatch)

    small_results.viewer(blocking=False)

    assert "--restore-session" not in captured["args"]
    assert "--no-save-session" not in captured["args"]


@pytest.mark.parametrize(
    "restore, token", [(True, "yes"), (False, "no")],
)
def test_restore_policy_crosses_the_argv_hop(
    small_results, monkeypatch, restore, token,
):
    captured = _fake_popen(monkeypatch)

    small_results.viewer(blocking=False, restore_session=restore)

    args = captured["args"]
    assert args[args.index("--restore-session") + 1] == token


def test_save_session_false_crosses_the_argv_hop(
    small_results, monkeypatch,
):
    """It did NOT before the flip: the child used the defaults and saved
    anyway, quietly breaking a documented promise."""
    captured = _fake_popen(monkeypatch)

    small_results.viewer(blocking=False, save_session=False)

    assert "--no-save-session" in captured["args"]


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], ("prompt", True)),
        (["--restore-session", "yes"], (True, True)),
        (["--restore-session", "no"], (False, True)),
        (["--no-save-session"], ("prompt", False)),
        (["--restore-session", "no", "--no-save-session"], (False, False)),
    ],
)
def test_the_child_parses_the_policy_it_was_sent(
    small_results, monkeypatch, argv, expected,
):
    """The other half of the hop. Without this the flags could be
    emitted into a child that ignores them — which is exactly the bug
    they exist to fix."""
    from apeGmsh.viewers import __main__ as viewers_main

    seen: dict = {}

    def _viewer(self, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(Results, "viewer", _viewer)
    monkeypatch.setattr(
        viewers_main, "_open_results", lambda *_a, **_kw: small_results,
    )

    rc = viewers_main.main([str(small_results._path), *argv])

    assert rc == 0
    assert seen["blocking"] is True
    assert (seen["restore_session"], seen["save_session"]) == expected


# =====================================================================
# END TO END — the upgrade, through the real door
# =====================================================================

def test_upgrading_onto_a_real_v13_file_renames_it_and_opens(
    small_results, shown, capsys,
):
    """The whole sequence, not the gate in isolation: the old window
    wrote v13 on close (real writer), the build flips, the human opens
    the results, and they get a window plus their file kept aside."""
    path, payload = _write_v13(small_results)

    out = small_results.viewer(restore_session=True)
    printed = capsys.readouterr().out

    aside = Path(str(path) + LEGACY_SUFFIX)
    assert json.loads(aside.read_text(encoding="utf-8")) == payload
    assert "old viewer session" in printed
    assert isinstance(out, ResultsSession)
    assert out.panes[0].slots == {}


def test_the_upgraded_window_saves_the_new_schema_on_close(
    small_results, shown, monkeypatch,
):
    """Save-on-close is real, and it lands on the ADOPTED path — the
    same name the v13 file had, which is why the rename-aside had to
    happen first."""
    path, payload = _write_v13(small_results)

    def _show(self, *, blocking=True, title=None):
        # What the human does while the window is open.
        self.panes[0].contour = Contour("displacement_x")
        return None

    monkeypatch.setattr(ResultsSession, "show", _show)
    small_results.viewer(restore_session=True)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["kind"] == "apegmsh.results.session"
    assert saved["panes"][0]["slots"]["contour"]["quantity"] == (
        "displacement_x"
    )
    aside = Path(str(path) + LEGACY_SUFFIX)
    assert json.loads(aside.read_text(encoding="utf-8")) == payload


def test_the_second_upgrade_opens_and_destroys_nothing(
    small_results, monkeypatch,
):
    """Upgrade, downgrade, upgrade — through ``viewer()``.

    The aside is taken, so ``rename_legacy_aside`` refuses. A human who
    only wanted their window back gets it, both files survive the close,
    and the notice names the file to move.
    """
    path, _ = _write_v13(small_results)

    monkeypatch.setattr(
        ResultsSession, "show", lambda self, **kw: None,
    )
    small_results.viewer(restore_session=True)     # first upgrade
    _write_v13(small_results)                      # downgraded run

    before = path.read_text(encoding="utf-8")
    aside = Path(str(path) + LEGACY_SUFFIX)
    aside_before = aside.read_text(encoding="utf-8")

    def _show(self, *, blocking=True, title=None):
        self.panes[0].contour = Contour("displacement_x")
        return None

    monkeypatch.setattr(ResultsSession, "show", _show)
    out = small_results.viewer(restore_session=True)

    assert isinstance(out, ResultsSession)
    assert out.panes[0].contour is not None, "the window really was used"
    assert path.read_text(encoding="utf-8") == before
    assert aside.read_text(encoding="utf-8") == aside_before


def test_save_session_false_leaves_the_file_alone(
    small_results, monkeypatch,
):
    from apeGmsh.results.session import save_snapshot

    fresh = small_results.session()
    fresh.panes[0].contour = Contour("displacement_x")
    path = save_snapshot(fresh)
    before = path.read_text(encoding="utf-8")

    def _show(self, *, blocking=True, title=None):
        self.panes[0].contour = None
        return None

    monkeypatch.setattr(ResultsSession, "show", _show)
    small_results.viewer(restore_session=True, save_session=False)

    assert path.read_text(encoding="utf-8") == before


# =====================================================================
# The prompt's summary (Qt-free half)
# =====================================================================

def test_describe_names_panes_and_filled_slots(small_results):
    from apeGmsh.results.session import RestoredSession
    from apeGmsh.viewers.session._restore_prompt import describe

    session = small_results.session()
    session.panes[0].contour = Contour("displacement_x")
    session.add_view()

    text = describe(RestoredSession(session=session, notices=()))

    assert "2 mesh views" in text
    assert "contour" in text


def test_describe_singularises_one_pane(small_results):
    from apeGmsh.results.session import RestoredSession
    from apeGmsh.viewers.session._restore_prompt import describe

    text = describe(
        RestoredSession(session=small_results.session(), notices=())
    )

    assert "1 mesh view" in text
    assert "mesh views" not in text


# =====================================================================
# Real window, real GL — the qt lane (xvfb in CI; skipped locally)
# =====================================================================

@pytest.mark.qt
def test_viewer_opens_the_real_session_window(small_results):
    """The flip on the real oracle: ``viewer()``'s own body, the real
    ``SessionWindow``, a real GL context — closed from a timer so the
    blocking call returns.

    Windows/GPU green proves nothing for Mesa; this is the lane that
    runs under xvfb.
    """
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from qtpy.QtCore import QTimer

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _close_everything():
        for widget in qapp.topLevelWidgets():
            widget.close()
        qapp.quit()

    QTimer.singleShot(2500, _close_everything)
    out = small_results.viewer(
        title="s6a-flip-smoke", restore_session=False,
    )

    assert isinstance(out, ResultsSession)
    assert len(out.panes) == 1
    # The window closed, and save-on-close ran against the adopted path.
    path = viewer_snapshot_path(small_results)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["kind"] == (
        "apegmsh.results.session"
    )
