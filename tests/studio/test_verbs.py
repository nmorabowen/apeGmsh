"""ADR 0095 S4b — assess / animate workers (no live Gmsh in the test)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apeGmsh.studio.__main__ import main
from apeGmsh.studio._verbs import animate_artifact, assess_artifact


def _report(**kwargs):
    defaults = dict(
        text="Verdict: ok",
        findings=(
            SimpleNamespace(
                code="RES.U_VS_DIAG",
                severity="info",
                message="ratio noted",
                detail={"ratio": 0.01},
            ),
        ),
        skipped=(("RES.ENERGY_ERR", "no energy channel"),),
        figures=(),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_assess_artifact_fem(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    fem = SimpleNamespace(assess=lambda **kw: _report())
    monkeypatch.setattr("apeGmsh.studio._verbs._is_model_only", lambda p: True)
    monkeypatch.setattr("apeGmsh.studio._verbs._open_fem", lambda p: fem)
    payload = assess_artifact(src, figures=False)
    assert payload["ok"] is True
    assert payload["source"] == "fem"
    assert payload["findings"][0]["code"] == "RES.U_VS_DIAG"
    assert payload["skipped"][0]["code"] == "RES.ENERGY_ERR"


def test_assess_artifact_missing(tmp_path: Path) -> None:
    payload = assess_artifact(tmp_path / "nope.h5")
    assert payload["ok"] is False
    assert payload["returncode"] == 2


def test_assess_cli_json(tmp_path: Path, monkeypatch, capsys) -> None:
    src = tmp_path / "model.h5"
    src.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "apeGmsh.studio._verbs.assess_artifact",
        lambda *a, **k: {
            "ok": True,
            "returncode": 0,
            "source": "fem",
            "text": "ok",
            "findings": [],
            "skipped": [],
            "figures": [],
        },
    )
    assert main(["--assess", str(src), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["source"] == "fem"


def test_animate_kind_not_shipped(tmp_path: Path) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    try:
        animate_artifact(src, tmp_path / "out.gif", kind="formation")
    except ValueError as exc:
        assert "not shipped" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_animate_artifact_history(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    dest = tmp_path / "clip.gif"
    results = SimpleNamespace(
        export_animation=lambda path, **kw: Path(path),
    )
    monkeypatch.setattr(
        "apeGmsh.studio._verbs._open_results", lambda p, m: results,
    )
    payload = animate_artifact(src, dest, kind="history")
    assert payload["ok"] is True
    assert payload["kind"] == "history"
    assert payload["output"] == str(dest)
    assert payload["skipped"] is False


def test_animate_artifact_yield(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    dest = tmp_path / "clip.gif"
    seen: dict = {}

    def export_animation(path, **kw):
        seen.update(kw)
        seen["path"] = Path(path)
        return Path(path)

    class FakeNodes:
        def get(self, component=None, time=None):
            n = 2
            if component == "displacement_x":
                return SimpleNamespace(values=np.array([[1.0, 0.0]]))
            if component in ("displacement_y", "displacement_z"):
                return SimpleNamespace(values=np.zeros((1, n)))
            raise KeyError(component)

    inspect = SimpleNamespace(
        components=lambda **kw: {
            "gauss": [
                "stress_xx", "stress_yy", "stress_zz",
                "stress_xy", "stress_yz", "stress_xz",
            ],
        },
    )
    results = SimpleNamespace(
        export_animation=export_animation,
        inspect=inspect,
        fem=SimpleNamespace(
            nodes=SimpleNamespace(coords=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])),
        ),
        nodes=FakeNodes(),
    )
    monkeypatch.setattr(
        "apeGmsh.studio._verbs._open_results", lambda p, m: results,
    )
    payload = animate_artifact(src, dest, kind="yield")
    assert payload["ok"] is True
    assert payload["kind"] == "yield"
    assert payload["component"] == "von_mises_stress"
    assert payload["deform_scale"] == pytest.approx(1.2)
    assert "iso-clip" in payload["picture"]
    assert payload["output"] == str(dest)
    assert seen.get("setup") is not None
    assert seen.get("deform")[0] == "displacement"
    assert seen.get("deform")[1] == pytest.approx(1.2)


def test_auto_deform_scale_stills_fraction() -> None:
    import numpy as np

    from apeGmsh.studio._verbs import auto_deform_scale

    class FakeNodes:
        def get(self, component=None, time=None):
            if component == "displacement_x":
                return SimpleNamespace(values=np.array([[2.0]]))
            return SimpleNamespace(values=np.array([[0.0]]))

    results = SimpleNamespace(
        fem=SimpleNamespace(
            nodes=SimpleNamespace(coords=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])),
        ),
        nodes=FakeNodes(),
    )
    # 0.12 * 10 / 2 = 0.6
    assert auto_deform_scale(results) == pytest.approx(0.6)


def test_animate_yield_requires_fluency(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    inspect = SimpleNamespace(
        components=lambda **kw: {"nodes": ["displacement_z"]},
    )
    results = SimpleNamespace(
        export_animation=lambda *a, **k: None,
        inspect=inspect,
    )
    monkeypatch.setattr(
        "apeGmsh.studio._verbs._open_results", lambda p, m: results,
    )
    try:
        animate_artifact(src, tmp_path / "out.gif", kind="yield")
    except ValueError as exc:
        assert "von_mises" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_animate_cli_yield_exits_2(tmp_path: Path, capsys) -> None:
    src = tmp_path / "run.h5"
    src.write_bytes(b"x")
    assert main(["--animate", str(src), "--kind", "formation"]) == 2
    assert "not shipped" in capsys.readouterr().err


def test_status_and_assess_conflict(capsys) -> None:
    assert main(["--status", "--assess", "x.h5"]) == 2
    assert "only one" in capsys.readouterr().err
