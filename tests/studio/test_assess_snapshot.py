"""ADR 0095 Amendment 5 — assess snapshot reaches the report.

``--assess`` writes ``.apegmsh/assess.json`` atomically; ``ReportBundle``
reads it (pin-preferred when a pin is referenced) and ``emit_report``
quotes its verdict / findings / skipped list / target / timestamp
instead of always printing "(none — run assess first)".
"""
from __future__ import annotations

import json
from pathlib import Path

from apeGmsh.studio.__main__ import main
from apeGmsh.studio._assess_snapshot import (
    ASSESS_SCHEMA,
    build_snapshot,
    read_snapshot_safe,
    write_snapshot,
)
from apeGmsh.studio._bundle import collect_bundle, emit_report, results_pin
from apeGmsh.studio._contract import CONTRACT_VERSION
from apeGmsh.studio._paths import assess_path, pin_assess_path


def _fake_assess_payload(
    src: Path,
    *,
    text: str = "Verdict: ok\n",
    findings: list[dict] | None = None,
    skipped: list[dict] | None = None,
    figures: list[str] | None = None,
) -> dict:
    return {
        "ok": True,
        "returncode": 0,
        "source": "fem",
        "path": str(src.resolve()),
        "text": text,
        "findings": findings or [],
        "skipped": skipped or [],
        "figures": figures or [],
    }


# ---------------------------------------------------------------------------
# Unit level — build_snapshot / read_snapshot_safe
# ---------------------------------------------------------------------------


def test_build_snapshot_root_relativizes_targets_and_figures(tmp_path: Path) -> None:
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    visor = tmp_path / ".apegmsh" / "visors" / "contour.png"
    visor.parent.mkdir(parents=True)
    visor.write_bytes(b"png")
    payload = _fake_assess_payload(src, figures=[str(visor)])
    snapshot = build_snapshot(
        payload,
        target=Path(payload["path"]),
        model_h5=None,
        root=tmp_path,
        ts="2026-08-17T00:00:00Z",
    )
    assert snapshot["schema"] == ASSESS_SCHEMA
    assert snapshot["targets"] == {"path": "model.h5", "model_h5": None}
    assert snapshot["figures"] == [".apegmsh/visors/contour.png"]
    assert snapshot["text"] == "Verdict: ok\n"


def test_read_snapshot_safe_missing_returns_none_none(tmp_path: Path) -> None:
    snapshot, error = read_snapshot_safe(tmp_path / "assess.json")
    assert snapshot is None
    assert error is None


def test_read_snapshot_safe_torn_degrades(tmp_path: Path) -> None:
    path = tmp_path / "assess.json"
    path.write_text('{"schema": 1, "ts": "2026', encoding="utf-8")
    snapshot, error = read_snapshot_safe(path)
    assert snapshot is None
    assert error


# ---------------------------------------------------------------------------
# CLI verb path — --assess writes the snapshot atomically
# ---------------------------------------------------------------------------


def test_assess_cli_writes_snapshot_atomically(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    visor = tmp_path / ".apegmsh" / "visors" / "contour.png"
    visor.parent.mkdir(parents=True)
    visor.write_bytes(b"png")

    monkeypatch.setattr(
        "apeGmsh.studio._verbs.assess_artifact",
        lambda *a, **k: _fake_assess_payload(
            src,
            findings=[
                {
                    "code": "MODEL.EMPTY",
                    "severity": "error",
                    "message": "no elements",
                    "detail": None,
                },
            ],
            skipped=[{"code": "RES.NAN", "reason": "no results loaded"}],
            figures=[str(visor)],
        ),
    )
    assert main(["--assess", str(src), "--json"]) == 0

    dest = assess_path(tmp_path)
    assert dest.is_file()
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["schema"] == 1
    assert data["targets"]["path"] == "model.h5"
    assert data["findings"][0]["code"] == "MODEL.EMPTY"
    assert data["skipped"][0]["code"] == "RES.NAN"
    assert data["figures"] == [".apegmsh/visors/contour.png"]
    assert "Verdict: ok" in data["text"]
    # atomic (INV-16): no leftover temp file next to the destination
    assert list(dest.parent.glob(f".{dest.name}.*.tmp")) == []


def test_assess_cli_failure_does_not_write_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    monkeypatch.setattr(
        "apeGmsh.studio._verbs.assess_artifact",
        lambda *a, **k: {
            "ok": False,
            "returncode": 2,
            "error": "boom",
            "path": str(src),
        },
    )
    main(["--assess", str(src), "--json"])
    assert not assess_path(tmp_path).is_file()


# ---------------------------------------------------------------------------
# ReportBundle degrade + no-snapshot behaviour
# ---------------------------------------------------------------------------


def test_bundle_degrades_on_torn_assess_snapshot(tmp_path: Path) -> None:
    (tmp_path / ".apegmsh").mkdir()
    (tmp_path / ".apegmsh" / "assess.json").write_text(
        '{"schema": 1, "ts": "2026', encoding="utf-8",
    )
    bundle = collect_bundle(root=tmp_path)
    assert bundle["assess"] is None
    assert bundle["assess_error"]


def test_bundle_no_snapshot_is_none_no_error(tmp_path: Path) -> None:
    bundle = collect_bundle(root=tmp_path)
    assert bundle["assess"] is None
    assert bundle["assess_error"] is None


def test_emit_without_assess_still_says_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    assert main(["--pin", "--model-h5", str(src), "--json"]) == 0
    assert main(["--emit-report", "--json"]) == 0
    text = (tmp_path / "docs" / "studio-report.md").read_text(encoding="utf-8")
    assert "(none — run assess first)" in text


# ---------------------------------------------------------------------------
# Dogfood repro — assess, pin, emit (ADR 0095 Amendment 5 acceptance)
# ---------------------------------------------------------------------------


def test_assess_pin_emit_chapter_shows_verdict_and_findings(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    monkeypatch.setattr(
        "apeGmsh.studio._verbs.assess_artifact",
        lambda *a, **k: _fake_assess_payload(
            src,
            text=(
                "# Assessment (fem)\n\n"
                "**Verdict:** 1 error(s), 0 warning(s), 0 info\n\n"
                "## Errors\n\n- `MODEL.EMPTY` — no elements\n\n"
                "## Skipped checks\n\nNone.\n"
            ),
            findings=[
                {
                    "code": "MODEL.EMPTY",
                    "severity": "error",
                    "message": "no elements",
                    "detail": None,
                },
            ],
        ),
    )
    assert main(["--assess", str(src), "--json"]) == 0
    assert main(["--pin", "--model-h5", str(src), "--json"]) == 0
    assert main(["--emit-report", "--json"]) == 0

    text = (tmp_path / "docs" / "studio-report.md").read_text(encoding="utf-8")
    assert "**Verdict:** 1 error(s)" in text
    assert "MODEL.EMPTY" in text
    assert "target: `model.h5`" in text
    assert "assessed:" in text


def test_pin_copies_snapshot_into_pin_folder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    monkeypatch.setattr(
        "apeGmsh.studio._verbs.assess_artifact",
        lambda *a, **k: _fake_assess_payload(src),
    )
    assert main(["--assess", str(src), "--json"]) == 0
    pin = results_pin(src, root=tmp_path)
    pinned = pin_assess_path(pin["id"], tmp_path)
    assert pinned.is_file()
    data = json.loads(pinned.read_text(encoding="utf-8"))
    assert data["text"] == "Verdict: ok\n"


def test_pin_preferred_over_deleted_live_snapshot(
    tmp_path: Path, monkeypatch,
) -> None:
    """Deleting the live assess.json after pinning still shows the pinned verdict."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    monkeypatch.setattr(
        "apeGmsh.studio._verbs.assess_artifact",
        lambda *a, **k: _fake_assess_payload(src, text="Verdict: pinned-verdict\n"),
    )
    assert main(["--assess", str(src), "--json"]) == 0
    assert main(["--pin", "--model-h5", str(src), "--json"]) == 0

    assess_path(tmp_path).unlink()

    assert main(["--emit-report", "--json"]) == 0
    text = (tmp_path / "docs" / "studio-report.md").read_text(encoding="utf-8")
    assert "pinned-verdict" in text


def test_pin_preferred_over_changed_live_snapshot(
    tmp_path: Path, monkeypatch,
) -> None:
    """A later, different live snapshot does not leak into an already-pinned chapter."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".apegmsh").mkdir()
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    monkeypatch.setattr(
        "apeGmsh.studio._verbs.assess_artifact",
        lambda *a, **k: _fake_assess_payload(src, text="Verdict: pinned-verdict\n"),
    )
    assert main(["--assess", str(src), "--json"]) == 0
    assert main(["--pin", "--model-h5", str(src), "--json"]) == 0

    write_snapshot(
        tmp_path,
        {
            "schema": 1,
            "ts": "2026-08-18T00:00:00Z",
            "targets": {"path": "model.h5", "model_h5": None},
            "findings": [],
            "skipped": [],
            "figures": [],
            "text": "Verdict: new-live-verdict\n",
        },
    )

    assert main(["--emit-report", "--json"]) == 0
    text = (tmp_path / "docs" / "studio-report.md").read_text(encoding="utf-8")
    assert "pinned-verdict" in text
    assert "new-live-verdict" not in text


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_contract_version_is_1_8_0() -> None:
    # 1.8.0 (ADR 0095 Amendment 11): `progress.json` joined the published
    # kinds, run lines gained `started_at` / `duration_s`, and
    # `names.json` / `progress.json` now stamp `contract_version` so a
    # file-only consumer can engage the INV-17 major gate. Additive
    # throughout — no existing reader's behaviour changes, so the major
    # stays 1.
    assert CONTRACT_VERSION == "1.8.0"
