"""ADR 0095 S5b — atomic habitat writes + torn-read status (INV-16)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from apeGmsh.studio import SelectionEnvelope, collect_status, write_envelope
from apeGmsh.studio._envelope import PickEvidence
from apeGmsh.studio._highlight import write_highlight
from apeGmsh.studio._names import write_names
from apeGmsh.studio._paths import atomic_write_text
from apeGmsh.studio._status import has_studio_state


def test_atomic_write_replaces_without_temp_left(tmp_path: Path) -> None:
    dest = tmp_path / "payload.json"
    atomic_write_text(dest, '{"ok": true}\n')
    assert dest.read_text(encoding="utf-8") == '{"ok": true}\n'
    leftovers = list(tmp_path.glob(".payload.json.*.tmp"))
    assert leftovers == []


def test_write_envelope_names_highlight_are_atomic(tmp_path: Path) -> None:
    env = SelectionEnvelope(
        phase="model",
        substrate="model_brep",
        labels=("body",),
        physical_groups=("Body",),
        unnamed=(),
        evidence=(
            PickEvidence(
                substrate="model_brep",
                dim=3,
                key=1,
                labels=("body",),
                physical_groups=("Body",),
            ),
        ),
    )
    write_envelope(tmp_path / "selection.json", env)
    write_names(
        tmp_path / "names.json",
        {"schema": 1, "phase": "model", "labels": ["body"], "entities": []},
    )
    write_highlight(
        tmp_path / "highlight.json", names=["body"], dimtags=[(3, 1)],
    )
    assert json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert json.loads((tmp_path / "names.json").read_text(encoding="utf-8"))
    assert json.loads((tmp_path / "highlight.json").read_text(encoding="utf-8"))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_status_degrades_on_torn_names(tmp_path: Path) -> None:
    (tmp_path / ".apegmsh").mkdir()
    (tmp_path / ".apegmsh" / "names.json").write_text(
        '{"schema": 1, "phase": "model", "labels": ["bo',
        encoding="utf-8",
    )
    payload = collect_status(tmp_path)
    assert payload["names"] is None
    assert payload["names_error"]
    assert has_studio_state(payload) is False


def test_status_poll_during_rewrite_never_raises(tmp_path: Path) -> None:
    """Concurrent rewrite + status: no exception, no partial dict."""
    dest = tmp_path / ".apegmsh" / "names.json"
    dest.parent.mkdir()
    body = {"schema": 1, "phase": "model", "labels": ["body"], "entities": []}
    write_names(dest, body)

    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        i = 0
        while not stop.is_set():
            i += 1
            fat = {
                **body,
                "labels": ["body"] * (50 + (i % 20)),
                "pad": "x" * (10_000 + (i % 500)),
            }
            try:
                write_names(dest, fat)
            except BaseException as exc:  # noqa: BLE001 — collect for assert
                errors.append(exc)
                return

    def reader() -> None:
        while not stop.is_set():
            try:
                payload = collect_status(tmp_path)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return
            names = payload.get("names")
            if names is not None:
                assert isinstance(names, dict)
                assert "schema" in names
            # names_error is fine mid-flight if a race hits a missing file

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    threading.Event().wait(0.4)
    stop.set()
    for t in threads:
        t.join(timeout=2)
    assert errors == [], errors
