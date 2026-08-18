"""ADR 0098 S5a — the legacy gate. It must NEVER destroy.

Through S2–S5 the old viewer still owns and overwrites
``<results>.viewer-session.json`` on close; the new session writes
``<results>.session.json`` and adopts the old path at the S6 flip (plan
decision 11). A v13-shaped file meeting the new loader therefore earns
a notice and a ``.legacy`` rename-aside — the ADR keeps the old file so
a one-shot importer would still have its input.

That promise is worth nothing if the SECOND open overwrites the first
rename, so the tests below prove the collision is refused AND that the
first ``.legacy`` still holds its original bytes afterwards. A test
that only proves the first rename would stay green against a product
that clobbers.

The v13 payloads here come from the REAL old writer
(``viewers.diagrams._session.serialize_session``) rather than a
hand-written dict: the gate has to fire on what today's window actually
writes, and this file goes red if that envelope changes shape before
S6.
"""
from __future__ import annotations

import json

import pytest

from apeGmsh.results.session import (
    LEGACY_SUFFIX,
    Contour,
    LegacySessionFile,
    ResultsSession,
    SnapshotError,
    legacy_shape,
    load_snapshot,
    rename_legacy_aside,
    save_snapshot,
    snapshot,
)
from apeGmsh.viewers.diagrams._session import (
    SESSION_SCHEMA_VERSION,
    serialize_session,
)


def _v13_payload(results_path: str = "/models/out.h5") -> dict:
    """A genuine old-viewer session envelope, from the old writer."""
    return serialize_session(
        specs=[],
        results_path=results_path,
        fem_snapshot_id="snap-1",
        active_stage_id="grav",
        active_step=3,
    )


def _write_v13(path, results_path: str = "/models/out.h5"):
    payload = _v13_payload(results_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _new_snapshot_file(path):
    session = ResultsSession()
    session.add_view().contour = Contour("stress_xx")
    save_snapshot(session, path)
    return session


# =====================================================================
# Detection — both directions
# =====================================================================

def test_the_real_v13_envelope_is_detected_as_legacy():
    payload = _v13_payload()

    assert payload["schema_version"] == SESSION_SCHEMA_VERSION
    assert legacy_shape(payload) is True


def test_a_new_snapshot_is_not_legacy():
    """The complement — without it the predicate could just return
    True."""
    session = ResultsSession()
    session.add_view()

    assert legacy_shape(snapshot(session)) is False


@pytest.mark.parametrize("data", [
    None, [], "text", 13, {}, {"kind": "apegmsh.results.session"},
    {"schema_version": 13},              # no old-ontology block
    {"diagrams": []},                    # no version
    {"schema_version": True, "diagrams": []},  # bool is not a version
])
def test_unrelated_json_is_not_mistaken_for_a_legacy_session(data):
    """Only the old envelope earns a rename. Anything else that is not
    a snapshot refuses instead — a rename is a write, and this loader
    does not write to files it cannot identify."""
    assert legacy_shape(data) is False


# =====================================================================
# The human flow — notice + rename aside
# =====================================================================

def test_a_v13_file_is_renamed_aside_with_a_notice(tmp_path, capsys):
    path = tmp_path / "out.h5.viewer-session.json"
    payload = _write_v13(path)

    result = load_snapshot(path)

    assert result is None                      # nothing restorable
    assert not path.exists()                   # moved, not copied
    aside = tmp_path / (path.name + LEGACY_SUFFIX)
    assert json.loads(aside.read_text(encoding="utf-8")) == payload

    notice = capsys.readouterr().out
    assert "[session]" in notice
    assert path.name in notice and aside.name in notice
    assert str(SESSION_SCHEMA_VERSION) in notice


def test_a_second_collision_is_refused_and_the_first_aside_survives(
    tmp_path,
):
    """THE never-destroy test.

    Two closes of the old viewer, two opens of the new one. The second
    rename has nowhere to go, and the only acceptable answer is to
    refuse — with BOTH files still on disk and the first ``.legacy``
    byte-for-byte what it was.
    """
    path = tmp_path / "out.h5.viewer-session.json"
    first = _write_v13(path, results_path="/models/FIRST.h5")
    load_snapshot(path)
    aside = tmp_path / (path.name + LEGACY_SUFFIX)
    aside_bytes = aside.read_bytes()

    # The old viewer writes a new session at the same path.
    second = _write_v13(path, results_path="/models/SECOND.h5")
    assert second != first

    with pytest.raises(FileExistsError) as excinfo:
        load_snapshot(path)

    message = str(excinfo.value)
    assert aside.name in message
    assert "destroy" in message                # says WHY it refused
    # Nothing was destroyed, in either direction:
    assert aside.read_bytes() == aside_bytes   # the first is untouched
    assert path.exists()                       # the second is untouched
    assert json.loads(path.read_text(encoding="utf-8")) == second
    # ... and no third file was invented.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [path.name, aside.name],
    )


def test_rename_aside_leaves_no_placeholder_when_it_refuses(tmp_path):
    """The exclusive-create reservation must not become a 0-byte file
    that blocks every future retry."""
    path = tmp_path / "s.json"
    path.write_text("{}", encoding="utf-8")
    aside = tmp_path / (path.name + LEGACY_SUFFIX)
    aside.write_text("older", encoding="utf-8")

    with pytest.raises(FileExistsError):
        rename_legacy_aside(path)

    assert aside.read_text(encoding="utf-8") == "older"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        path.name, aside.name,
    ]


def test_rename_aside_appends_and_never_substitutes_the_suffix(tmp_path):
    path = tmp_path / "out.h5.viewer-session.json"
    path.write_text("{}", encoding="utf-8")

    dest = rename_legacy_aside(path)

    assert dest.name == "out.h5.viewer-session.json.legacy"


# =====================================================================
# The MCP flow (S5c) — refuse, rename NOTHING
# =====================================================================

def test_the_no_rename_flow_refuses_and_touches_no_file(tmp_path):
    """S5c's contract: MCP render refuses old-schema files, never
    renames — the rename belongs to the human flow."""
    path = tmp_path / "out.h5.viewer-session.json"
    payload = _write_v13(path)

    with pytest.raises(LegacySessionFile) as excinfo:
        load_snapshot(path, rename_legacy=False)

    assert excinfo.value.path == path
    assert excinfo.value.schema_version == SESSION_SCHEMA_VERSION
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


# =====================================================================
# The gate does not stand in the way of a real snapshot
# =====================================================================

def test_a_real_snapshot_loads_through_the_same_door(tmp_path):
    path = tmp_path / "out.h5.session.json"
    session = _new_snapshot_file(path)

    restored = load_snapshot(path)

    assert restored is not None
    assert restored.notices == ()
    assert restored.session.panes[0].contour == Contour("stress_xx")
    assert path.exists()
    assert not (tmp_path / (path.name + LEGACY_SUFFIX)).exists()


def test_json_that_is_neither_refuses_without_renaming(tmp_path):
    """Not a snapshot and not the old envelope: refuse loudly and leave
    the file alone. Renaming an unidentified file would be a write this
    loader has no business making."""
    path = tmp_path / "notes.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")

    with pytest.raises(SnapshotError, match="Not an apeGmsh session"):
        load_snapshot(path)

    assert [p.name for p in tmp_path.iterdir()] == [path.name]
