"""ADR 0095 S2c — append-only run ledger."""
from __future__ import annotations

import json
from pathlib import Path

from apeGmsh.studio import last_run, ledger_path, read_runs
from apeGmsh.studio._ledger import append_run, make_record
from apeGmsh.studio._replay import ReplayRunner, _exec_hold_open


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    first = make_record(
        script=tmp_path / "a.py",
        phase="model",
        ok=True,
        hash="abc",
        stopped_at="g.mesh.generation.generate",
        session="box",
        manifest={
            "labels": ["body"],
            "physical_groups": ["Body"],
            "counts": {"entities": {"3": 1}, "elements": {"3": 0}},
        },
        ts="2026-08-14T06:00:00Z",
    )
    second = make_record(
        script=tmp_path / "a.py",
        phase="mesh",
        ok=True,
        hash="abc",
        stopped_at="apeSees",
        session="box",
        manifest={
            "labels": ["body"],
            "physical_groups": ["Body"],
            "counts": {"entities": {"3": 1}, "elements": {"3": 12}},
        },
        ts="2026-08-14T06:01:00Z",
    )
    append_run(path, first)
    append_run(path, second)
    rows = read_runs(path)
    assert len(rows) == 2
    assert rows[0]["phase"] == "model"
    assert rows[0]["counts"]["elements"]["3"] == 0
    assert rows[1]["phase"] == "mesh"
    assert last_run(path)["stopped_at"] == "apeSees"
    assert last_run(path)["assess"] is None
    assert last_run(path)["visors"] == []
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 2
    json.loads(raw[0])


def test_missing_ledger_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    assert read_runs(path) == []
    assert last_run(path) is None


def test_read_runs_skips_torn_lines(tmp_path: Path) -> None:
    from apeGmsh.studio._ledger import read_runs_safe

    path = tmp_path / "runs.jsonl"
    good = make_record(
        script=tmp_path / "a.py",
        phase="model",
        ok=True,
        ts="2026-08-14T06:00:00Z",
    )
    path.write_text(
        json.dumps(good, ensure_ascii=False) + "\n"
        + "{not json\n"
        + '"just a string"\n'
        + json.dumps({**good, "phase": "mesh", "ts": "2026-08-14T06:01:00Z"})
        + "\n",
        encoding="utf-8",
    )
    rows, err = read_runs_safe(path)
    assert err is not None
    assert "skipped 2" in err
    assert len(rows) == 2
    assert rows[0]["phase"] == "model"
    assert rows[1]["phase"] == "mesh"
    assert read_runs(path) == rows
    assert last_run(path)["phase"] == "mesh"


def test_read_runs_keeps_rows_after_utf8_tear(tmp_path: Path) -> None:
    """A torn multibyte append must not discard earlier good lines (S5f)."""
    from apeGmsh.studio._ledger import read_runs_safe

    path = tmp_path / "runs.jsonl"
    good = make_record(
        script=tmp_path / "a.py",
        phase="model",
        ok=True,
        ts="2026-08-14T06:00:00Z",
        manifest={
            "labels": ["columna"],
            "physical_groups": ["Columna"],
            "counts": None,
        },
    )
    # Partial UTF-8 for 'ñ' (0xc3 0xb1) — only the lead byte, as a crash mid-append.
    body = (
        json.dumps(good, ensure_ascii=False) + "\n"
        + '{"schema":1,"phase":"mesh","labels":["se'
    ).encode("utf-8") + b"\xc3"
    path.write_bytes(body)
    rows, err = read_runs_safe(path)
    assert err is not None
    assert "skipped" in err
    assert len(rows) == 1
    assert rows[0]["phase"] == "model"
    assert rows[0]["labels"] == ["columna"]


def test_error_is_capped(tmp_path: Path) -> None:
    rec = make_record(
        script=tmp_path / "a.py",
        phase="model",
        ok=False,
        error="x" * 20_000,
    )
    assert rec["error"] is not None
    assert len(rec["error"]) == 8000


def test_run_until_appends_and_skip_does_not(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "box.py"
    script.write_text(
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='studio_box') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n"
        "    g.physical.from_label('body', name='Body')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner()
    r1 = runner.run_until(script, phase="model")
    r2 = runner.run_until(script, phase="model")
    try:
        assert r1.ok, r1.error
        assert r2.skipped is True
        path = ledger_path(tmp_path)
        rows = read_runs(path)
        assert len(rows) == 1
        rec = rows[0]
        assert rec["ok"] is True
        assert rec["phase"] == "model"
        assert rec["session"] == "studio_box"
        assert "body" in rec["labels"]
        assert "Body" in rec["physical_groups"]
        assert rec["counts"]["elements"]["3"] == 0
        assert rec["hash"] == r1.geometry_hash
        assert rec["assess"] is None
        assert rec["visors"] == []
    finally:
        if r1.session is not None and r1.session.is_active:
            r1.session.end()


def test_failed_exec_appends_error_line(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "bad.py"
    script.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    result = ReplayRunner().run_until(script, phase="model")
    assert result.ok is False
    rec = last_run(ledger_path(tmp_path))
    assert rec is not None
    assert rec["ok"] is False
    assert rec["phase"] == "model"
    assert rec["counts"] is None
    assert rec["error"] is not None
    assert "boom" in rec["error"]


# =====================================================================
# ADR 0095 Amendment 11 — how long did that take
# =====================================================================


def test_a_real_replay_stamps_started_at_and_duration(
    tmp_path: Path, monkeypatch
) -> None:
    import time

    script = tmp_path / "box.py"
    script.write_text(
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='studio_box') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    wall_before = time.monotonic()
    result = _exec_hold_open(script, "model")
    wall_after = time.monotonic()
    try:
        rec = last_run(ledger_path(tmp_path))
        assert rec is not None
        # ``ts`` is when the line was written (the END of the run), so it
        # cannot double as a start time — that is why there are two keys.
        assert rec["started_at"].endswith("Z")
        assert rec["started_at"] <= rec["ts"]
        assert 0.0 <= rec["duration_s"] <= (wall_after - wall_before)
    finally:
        if result.session is not None and result.session.is_active:
            result.session.end()


def test_a_failed_replay_is_timed_too(tmp_path: Path, monkeypatch) -> None:
    """The number a human most wants when a replay is slow AND broken."""
    script = tmp_path / "bad.py"
    script.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert ReplayRunner().run_until(script, phase="model").ok is False

    rec = last_run(ledger_path(tmp_path))
    assert rec is not None and rec["ok"] is False
    assert rec["duration_s"] >= 0.0
    assert rec["started_at"].endswith("Z")


def test_a_skipped_run_writes_no_line_at_all(tmp_path: Path, monkeypatch) -> None:
    """Why "what duration does a skip report" is not a question.

    The same-hash / same-phase short circuit returns before the ledger
    is touched, so there is no record to leave un-timed and no honest
    duration to invent.
    """
    script = tmp_path / "box.py"
    script.write_text(
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='studio_box') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    runner = ReplayRunner()
    first = runner.run_until(script, phase="model")
    try:
        assert first.ok, first.error
        second = runner.run_until(script, phase="model")
        assert second.skipped is True

        rows = read_runs(ledger_path(tmp_path))
        assert len(rows) == 1
        assert "duration_s" in rows[0]
    finally:
        if first.session is not None and first.session.is_active:
            first.session.end()


def test_last_ledger_line_matches_names_json(
    tmp_path: Path, monkeypatch
) -> None:
    from apeGmsh.studio._paths import names_path

    script = tmp_path / "box.py"
    script.write_text(
        "from apeGmsh import apeGmsh\n"
        "with apeGmsh(model_name='studio_box') as g:\n"
        "    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='body')\n"
        "    g.physical.from_label('body', name='Body')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = _exec_hold_open(script, "model")
    try:
        assert result.ok, result.error
        names = json.loads(names_path(tmp_path).read_text(encoding="utf-8"))
        rec = last_run(ledger_path(tmp_path))
        assert rec is not None
        assert rec["phase"] == names["phase"]
        assert rec["labels"] == names["labels"]
        assert rec["physical_groups"] == names["physical_groups"]
        assert rec["counts"] == names["counts"]
    finally:
        if result.session is not None and result.session.is_active:
            result.session.end()
