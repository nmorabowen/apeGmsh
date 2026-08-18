"""ADR 0098 S5b — ``results_pin(session_snapshot=)``.

The pin carries the picture a human arranged, so an agent can draw a
still of it later (§11 S5). Three things this slice has to get right,
each with its own test below:

* the ledger key is ``session_snapshot``, never ``session`` — a ledger
  record already uses ``session`` for the run's session name (plan
  decision 11);
* the file is **copied** into the pin, not merely hashed. ``model_h5``
  and ``results`` are large and nobody rewrites them under you; the live
  ``<results>.session.json`` is rewritten on every save, so a stamp
  alone would pin content that is already gone. The discriminating test
  overwrites the live file and asks what the pin still holds;
* a file that is not a session snapshot is refused HERE. The near-miss
  is the retired ``<results>.viewer-session.json``, which sits in the
  same directory under a similar name — pinned blind it becomes an
  opaque hash that S5c refuses at render time, one step too late and
  against the wrong artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apeGmsh.results.session import (
    Contour,
    Instant,
    ResultsSession,
    save_snapshot,
)
from apeGmsh.studio._bundle import _SESSION_SNAPSHOT_KIND, results_pin
from apeGmsh.studio._contract import validate
from apeGmsh.studio._ledger import read_runs
from apeGmsh.studio._paths import (
    ledger_path,
    pin_session_snapshot_path,
)
from apeGmsh.viewers.diagrams._session import serialize_session


def _habitat(tmp_path: Path) -> Path:
    (tmp_path / ".apegmsh").mkdir(exist_ok=True)
    return tmp_path


def _snapshot_file(tmp_path: Path, *, panes: int = 1) -> Path:
    session = ResultsSession()
    for _ in range(panes):
        session.add_view().contour = Contour("stress_xx")
    session.time = Instant("grav", 2)
    path = tmp_path / "out.h5.session.json"
    save_snapshot(session, path)
    return path


# =====================================================================
# The record
# =====================================================================

def test_the_ledger_key_is_session_snapshot_not_session(tmp_path: Path):
    """``session`` is taken (plan decision 11): in a ledger record it is
    the run's session NAME. Two meanings for one key is how a consumer
    reads a run's name and gets a file stamp."""
    _habitat(tmp_path)
    snapshot = _snapshot_file(tmp_path)

    record = results_pin(session_snapshot=snapshot, root=tmp_path)["pin"]

    assert record["session_snapshot"]["path"] == str(snapshot.resolve())
    assert record["session_snapshot"]["size"] == snapshot.stat().st_size
    assert record["session_snapshot"]["sha256"]
    assert "session" not in record


def test_a_snapshot_alone_is_enough_to_pin(tmp_path: Path):
    """It is a pinnable artifact in its own right — the ADR's "agent
    still of what the human built" needs no model_h5 beside it."""
    _habitat(tmp_path)

    out = results_pin(
        session_snapshot=_snapshot_file(tmp_path), root=tmp_path,
    )

    assert out["ok"] is True
    (row,) = [r for r in read_runs(ledger_path(tmp_path))
              if r.get("kind") == "pin"]
    assert row["id"] == out["id"]
    assert row["model_h5"] is None and row["results"] is None


def test_pinning_nothing_at_all_still_refuses(tmp_path: Path):
    _habitat(tmp_path)

    with pytest.raises(ValueError, match="session_snapshot"):
        results_pin(root=tmp_path)


def test_the_record_validates_against_the_published_pin_schema(
    tmp_path: Path,
):
    _habitat(tmp_path)

    record = results_pin(
        session_snapshot=_snapshot_file(tmp_path), root=tmp_path,
    )["pin"]

    validate("ledger_pin", record)


# =====================================================================
# The copy — the Amendment-5 lifecycle, and why a stamp is not enough
# =====================================================================

def test_the_snapshot_is_copied_into_the_pin(tmp_path: Path):
    _habitat(tmp_path)
    snapshot = _snapshot_file(tmp_path)

    out = results_pin(session_snapshot=snapshot, root=tmp_path)

    copy = pin_session_snapshot_path(out["id"], tmp_path)
    assert copy.is_file()
    assert json.loads(copy.read_text(encoding="utf-8")) == json.loads(
        snapshot.read_text(encoding="utf-8"),
    )


def test_the_pinned_copy_survives_the_next_save(tmp_path: Path):
    """THE reason this one is copied and the h5 files are not.

    The human pins, keeps working, saves again. A pin that had only
    stamped a hash would now name a file whose content is different —
    provenance for something nobody can look at. Written as a change the
    pin must NOT see: one pane at pin time, three afterwards.
    """
    _habitat(tmp_path)
    snapshot = _snapshot_file(tmp_path, panes=1)

    out = results_pin(session_snapshot=snapshot, root=tmp_path)
    pinned_sha = out["pin"]["session_snapshot"]["sha256"]

    # ... work continues, and the live snapshot is rewritten.
    session = ResultsSession()
    for _ in range(3):
        session.add_view()
    save_snapshot(session, snapshot)

    live = json.loads(snapshot.read_text(encoding="utf-8"))
    copy = pin_session_snapshot_path(out["id"], tmp_path)
    pinned = json.loads(copy.read_text(encoding="utf-8"))

    assert len(live["panes"]) == 3          # the live file moved on
    assert len(pinned["panes"]) == 1        # the pin did not
    # And the stamp still describes what the pin holds, not what is live.
    import hashlib
    assert pinned_sha == hashlib.sha256(
        copy.read_bytes(),
    ).hexdigest()
    assert pinned_sha != hashlib.sha256(snapshot.read_bytes()).hexdigest()


def test_two_pins_in_one_second_do_not_share_a_folder(
    tmp_path: Path, monkeypatch,
):
    """Pin ids are second-granular; the clock is FROZEN here so the
    collision is certain rather than likely.

    Written after reproducing the loss: two pins in one second shared an
    id, so the second pin's copy landed on top of the first's and the
    earlier picture was gone — and gone for good, because the live
    snapshot it came from had already been overwritten. The earlier
    version of this test raced the wall clock and skipped instead of
    proving anything, which is the same as not having it.
    """
    from datetime import datetime, timezone

    import apeGmsh.studio._bundle as bundle

    frozen = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenClock:
        @staticmethod
        def now(tz=None):
            return frozen

    monkeypatch.setattr(bundle, "datetime", _FrozenClock)

    _habitat(tmp_path)
    snapshot = _snapshot_file(tmp_path, panes=1)
    first = results_pin(session_snapshot=snapshot, root=tmp_path)
    save_snapshot(_two_pane_session(), snapshot)
    second = results_pin(session_snapshot=snapshot, root=tmp_path)

    assert first["id"] != second["id"]
    first_copy = pin_session_snapshot_path(first["id"], tmp_path)
    second_copy = pin_session_snapshot_path(second["id"], tmp_path)
    assert first_copy != second_copy
    # Each pin still holds the picture it was given.
    assert len(
        json.loads(first_copy.read_text(encoding="utf-8"))["panes"],
    ) == 1
    assert len(
        json.loads(second_copy.read_text(encoding="utf-8"))["panes"],
    ) == 2
    # And both are addressable in the ledger.
    ids = [r["id"] for r in read_runs(ledger_path(tmp_path))
           if r.get("kind") == "pin"]
    assert ids == [first["id"], second["id"]]
    assert len(set(ids)) == 2


def test_a_taken_pin_folder_alone_is_enough_to_disambiguate(
    tmp_path: Path, monkeypatch,
):
    """The ledger is not the only witness: a pin folder left behind by a
    run whose ledger line was lost (a torn append is tolerated by design,
    INV-16) must still not be written over."""
    from datetime import datetime, timezone

    import apeGmsh.studio._bundle as bundle
    from apeGmsh.studio._paths import pin_dir

    frozen = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenClock:
        @staticmethod
        def now(tz=None):
            return frozen

    monkeypatch.setattr(bundle, "datetime", _FrozenClock)

    _habitat(tmp_path)
    squatter = pin_dir("pin-20260818T120000Z", tmp_path)
    squatter.mkdir(parents=True)
    (squatter / "session_snapshot.json").write_text("older", encoding="utf-8")

    out = results_pin(
        session_snapshot=_snapshot_file(tmp_path), root=tmp_path,
    )

    assert out["id"] == "pin-20260818T120000Z-2"
    assert (squatter / "session_snapshot.json").read_text(
        encoding="utf-8",
    ) == "older"


def _two_pane_session() -> ResultsSession:
    session = ResultsSession()
    session.add_view()
    session.add_view()
    return session


# =====================================================================
# Refusals — the near-miss file, and the marker's single source of truth
# =====================================================================

def test_the_old_viewer_session_is_refused_by_name_of_what_it_is(
    tmp_path: Path,
):
    """A v13 file from the REAL old writer, so the refusal is pinned to
    what today's window actually leaves on disk."""
    _habitat(tmp_path)
    legacy = tmp_path / "out.h5.viewer-session.json"
    legacy.write_text(
        json.dumps(serialize_session(
            specs=[], results_path=str(tmp_path / "out.h5"),
            fem_snapshot_id=None,
        )),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        results_pin(session_snapshot=legacy, root=tmp_path)

    message = str(excinfo.value)
    assert "not an apeGmsh session snapshot" in message
    assert "OLD viewer" in message          # says which file it caught
    assert not list((tmp_path / ".apegmsh").glob("pins/*"))


@pytest.mark.parametrize("body", [
    "not json at all",
    '{"panes": []}',                  # a dict, but no marker
    '["apegmsh.results.session"]',    # the marker, in the wrong shape
])
def test_anything_without_the_marker_is_refused(tmp_path: Path, body: str):
    _habitat(tmp_path)
    path = tmp_path / "candidate.json"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match="session_snapshot="):
        results_pin(session_snapshot=path, root=tmp_path)


def test_a_missing_file_is_refused_before_anything_is_written(
    tmp_path: Path,
):
    _habitat(tmp_path)

    with pytest.raises(ValueError, match="file not found"):
        results_pin(session_snapshot=tmp_path / "nope.json", root=tmp_path)

    assert not ledger_path(tmp_path).exists()


def test_a_refused_snapshot_leaves_no_pin_line(tmp_path: Path):
    """The copy happens BEFORE ``append_run`` for this reason: a ledger
    line claiming a pinned snapshot that was never written would be a
    record of something that did not happen."""
    _habitat(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        results_pin(session_snapshot=bad, root=tmp_path)

    rows = read_runs(ledger_path(tmp_path)) if ledger_path(
        tmp_path,
    ).exists() else []
    assert [r for r in rows if r.get("kind") == "pin"] == []


# =====================================================================
# The MCP verb — in-process, so no subprocess timeout can hide a defect
# =====================================================================

def test_the_mcp_verb_threads_the_snapshot_through(tmp_path: Path):
    """Called in-process on purpose. The subprocess-backed studio verbs
    wait up to ``APEGMSH_STUDIO_TIMEOUT`` (600 s by default), which turns
    a mutation pass over this behaviour into a stall rather than a
    failure — so the thin verbs are asserted here, directly."""
    from apeGmsh.studio._mcp import results_pin as mcp_results_pin

    _habitat(tmp_path)
    snapshot = _snapshot_file(tmp_path)

    out = mcp_results_pin(session_snapshot=snapshot, root=tmp_path)

    assert out["ok"] is True
    assert out["error"] is None
    assert out["pin"]["session_snapshot"]["sha256"]
    assert pin_session_snapshot_path(out["id"], tmp_path).is_file()


def test_the_mcp_verb_reports_a_bad_snapshot_as_invalid_args(
    tmp_path: Path,
):
    """MCP verbs return an error envelope; they do not raise into the
    adapter (S5c's convention)."""
    from apeGmsh.studio._mcp import results_pin as mcp_results_pin

    _habitat(tmp_path)
    legacy = tmp_path / "out.h5.viewer-session.json"
    legacy.write_text(
        json.dumps(serialize_session(
            specs=[], results_path=str(tmp_path / "out.h5"),
            fem_snapshot_id=None,
        )),
        encoding="utf-8",
    )

    out = mcp_results_pin(session_snapshot=legacy, root=tmp_path)

    assert out["ok"] is False
    assert out["error"]["code"] == "INVALID_ARGS"
    assert "not an apeGmsh session snapshot" in out["error"]["message"]


def test_the_studio_marker_literal_matches_the_writers(tmp_path: Path):
    """``studio`` recognises a snapshot by its marker instead of
    importing the reader, because ``results.session`` imports
    ``viewers.core.selection`` (S0's recorded layering exception) and
    ``studio._bundle`` promises no viewers. That duplication is only safe
    while this assertion holds."""
    from apeGmsh.results.session import SNAPSHOT_KIND

    assert _SESSION_SNAPSHOT_KIND == SNAPSHOT_KIND


def test_studio_bundle_does_not_import_the_session_reader():
    """The purity claim in ``_bundle``'s docstring, as a test.

    Importing ``apeGmsh.results.session`` here would pull ``viewers`` in
    behind it — which is exactly what the duplicated marker avoids.
    """
    import ast

    source = Path(
        "src/apeGmsh/studio/_bundle.py",
    ).resolve()
    if not source.is_file():  # installed elsewhere
        import apeGmsh.studio._bundle as mod
        source = Path(mod.__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert not [m for m in imported if "viewers" in m or "results" in m]
