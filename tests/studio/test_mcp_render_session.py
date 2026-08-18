"""ADR 0098 S5c — ``render(session=…)``, the MCP door.

Every assertion here is IN-PROCESS. The studio verbs shell out and wait
up to ``APEGMSH_STUDIO_TIMEOUT`` (600 s by default), so a test that
drove the subprocess to check an *argument* rule would turn a defect
into a stall instead of a failure — the trap that cost S5b's first
mutation pass an hour. The subprocess path itself is exercised once, in
``tests/viewers/test_session_snapshot_render.py``, where a still is
actually drawn.

Two rules this door owes:

* the closed ``view`` vocabulary does NOT apply — a snapshot's content
  is the §4 slot catalog — so the old-ontology knobs are REFUSED rather
  than ignored. Silently drawing a snapshot's own instant while the
  caller passed ``step=3`` would hand them a different picture than the
  one they asked for and call it theirs;
* an old ``.viewer-session.json`` is refused and **never renamed**. The
  rename-aside is the human flow's (ADR 0098 Consequences); a batch verb
  that moved a user's file would be doing surgery nobody asked for.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apeGmsh.studio._mcp import render
from apeGmsh.viewers.diagrams._session import serialize_session

_SNAPSHOT = {
    "kind": "apegmsh.results.session",
    "version": 1,
    "results_path": None,
    "panes": [{"pane": "mesh", "id": "mesh-1"}],
}


def _habitat(tmp_path: Path) -> Path:
    (tmp_path / ".apegmsh").mkdir(exist_ok=True)
    return tmp_path


def _snapshot(tmp_path: Path, name: str = "out.h5.session.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(_SNAPSHOT), encoding="utf-8")
    return path


def _legacy(tmp_path: Path) -> Path:
    path = tmp_path / "out.h5.viewer-session.json"
    path.write_text(
        json.dumps(serialize_session(
            specs=[], results_path=str(tmp_path / "out.h5"),
            fem_snapshot_id=None,
        )),
        encoding="utf-8",
    )
    return path


# =====================================================================
# The old-ontology knobs are refused, not ignored
# =====================================================================

@pytest.mark.parametrize("kwargs,named", [
    ({"view": "contour"}, "view"),
    ({"component": "S1"}, "component"),
    ({"step": 3}, "step"),
    ({"deform": "2.0"}, "deform"),
    ({"pack": True}, "pack"),
])
def test_a_view_knob_with_session_is_invalid_args(
    tmp_path: Path, kwargs: dict, named: str,
):
    _habitat(tmp_path)

    out = render(session=_snapshot(tmp_path), root=tmp_path, **kwargs)

    assert out["ok"] is False
    assert out["error"]["code"] == "INVALID_ARGS"
    assert named in out["error"]["message"]
    assert out["written"] == []


def test_the_refusal_names_every_knob_that_was_passed(tmp_path: Path):
    _habitat(tmp_path)

    out = render(
        session=_snapshot(tmp_path), view="contour", step=2,
        root=tmp_path,
    )

    message = out["error"]["message"]
    assert "view" in message and "step" in message


def test_camera_is_NOT_refused(tmp_path: Path):
    """The complement — without it the rule reads as "session takes no
    options at all", which is not what §4 says. A camera is a property
    of the STILL, not of the picture the snapshot describes.

    Stated as what actually happens: the call clears argument validation
    and fails downstream instead, on this fixture snapshot naming no
    results file. A disjunction ("not INVALID_ARGS or camera not in the
    message") would also pass if the verb refused for some unrelated
    reason, which is not what is being claimed.
    """
    _habitat(tmp_path)

    out = render(session=_snapshot(tmp_path), camera="xy", root=tmp_path)

    assert out["ok"] is False
    assert out["error"]["code"] == "RENDER_FAILED"
    assert "results_path" in out["error"]["message"]


def test_the_results_door_is_unchanged(tmp_path: Path):
    """``view`` became Optional so an explicit value is distinguishable
    from an unset one; the results door must still default to
    ``contour`` and still refuse a token outside the closed set."""
    _habitat(tmp_path)
    results = tmp_path / "out.h5"
    results.write_bytes(b"not really h5")

    bad = render(results, view="banana", root=tmp_path)

    assert bad["error"]["code"] == "INVALID_VIEW"
    assert bad["view"] == "banana"

    # ... and an UNSET view still means contour. Only the explicit-token
    # half was asserted before, so making `view` Optional could have
    # dropped the default and nothing would have noticed.
    unset = render(results, root=tmp_path)
    assert unset["view"] == "contour"
    assert unset["error"]["code"] != "INVALID_VIEW"


def test_a_foreign_kind_is_refused_by_the_session_door(tmp_path: Path):
    """The discriminating case for the marker check.

    A file with NO ``kind`` is refused whether the door compares the
    value or merely looks for the key — so a wrong-``kind`` file is the
    only shape that can tell those two apart. (The same gap turned up in
    S5b's pin schema; it is the shape of mistake worth naming twice.)
    """
    _habitat(tmp_path)
    impostor = tmp_path / "impostor.session.json"
    impostor.write_text(
        json.dumps({"kind": "something.else.entirely", "version": 1}),
        encoding="utf-8",
    )

    out = render(session=impostor, root=tmp_path)

    assert out["error"]["code"] == "INVALID_ARGS"
    assert "apegmsh.results.session" in out["error"]["message"]


def test_neither_path_nor_session_refuses(tmp_path: Path):
    _habitat(tmp_path)

    out = render(root=tmp_path)

    assert out["error"]["code"] == "INVALID_ARGS"
    assert "path=" in out["error"]["message"]
    assert "session=" in out["error"]["message"]


# =====================================================================
# The legacy refusal — and the directory is left exactly as found
# =====================================================================

def test_an_old_viewer_session_is_refused_and_not_renamed(tmp_path: Path):
    _habitat(tmp_path)
    legacy = _legacy(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())

    out = render(session=legacy, root=tmp_path)

    assert out["error"]["code"] == "INVALID_ARGS"
    assert "OLD viewer session" in out["error"]["message"]
    assert "never renames" in out["error"]["message"]
    # Nothing moved, nothing appeared, the original is intact.
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not (tmp_path / (legacy.name + ".legacy")).exists()
    assert legacy.is_file()


def test_json_without_the_marker_is_refused_by_name(tmp_path: Path):
    _habitat(tmp_path)
    other = tmp_path / "notes.json"
    other.write_text('{"panes": []}', encoding="utf-8")

    out = render(session=other, root=tmp_path)

    assert out["error"]["code"] == "INVALID_ARGS"
    assert "apegmsh.results.session" in out["error"]["message"]


def test_unreadable_json_is_refused_before_anything_spawns(tmp_path: Path):
    _habitat(tmp_path)
    broken = tmp_path / "broken.session.json"
    broken.write_text("{not json", encoding="utf-8")

    out = render(session=broken, root=tmp_path)

    assert out["error"]["code"] == "INVALID_ARGS"
    assert "not readable JSON" in out["error"]["message"]


def test_a_missing_snapshot_is_not_found(tmp_path: Path):
    _habitat(tmp_path)

    out = render(session=tmp_path / "nope.json", root=tmp_path)

    assert out["error"]["code"] == "NOT_FOUND"


def test_the_marker_the_verb_checks_is_the_writers(tmp_path: Path):
    """This door recognises a snapshot by the same literal the pin does
    (S5b) — importing ``results.session`` here would pull ``viewers``
    into ``_mcp``, which promises none."""
    from apeGmsh.results.session import SNAPSHOT_KIND
    from apeGmsh.studio._bundle import _SESSION_SNAPSHOT_KIND

    assert _SESSION_SNAPSHOT_KIND == SNAPSHOT_KIND
    assert _SNAPSHOT["kind"] == SNAPSHOT_KIND
