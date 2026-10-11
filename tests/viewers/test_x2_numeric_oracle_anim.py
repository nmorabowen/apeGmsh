"""X2-e â€” numeric oracle for ``Results.export_animation`` (qt lane).

The drawn-data counterpart of ``test_x2_export_animation_golden.py``:
the same three cases (``default``, ``deform=``, ``setup=`` contour), the
same public call, but what is compared is the data in the scene at each
captured frame (``capture_drawn`` in ``test_x2_numeric_oracle.py``, which
holds the seam, the store and the tolerances), not its pixels.

``qt``-marked because the old path builds a Qt ``ResultsViewer``; the qt
lane runs it in its own process. When X2-b moves the verb onto the
session path (no Qt), drop the mark: the calls, the ``.npz`` and these
assertions stay as they are.

Per case and frame the oracle holds the drawn (deformed) coordinates
and, for the contour case, the scalars and their clim. The frame
schedule ``[0, 3, 4]`` is asserted from the data: each frame is decoded
back to the one step whose ``reference + scale * u(step)`` (deform) or
``displacement_x(step)`` (contour) it draws.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.results import Results
from tests.viewers import test_x2_golden_stills as G
from tests.viewers import test_x2_numeric_oracle as NO

pytestmark = pytest.mark.qt

_ANIM = G._MANIFEST["animation"]
_CASES = sorted(_ANIM["cases"])
_STEPS = list(range(int(G._MANIFEST["fixture"]["n_steps"])))


class _NullWriter:
    """Stand-in imageio writer (neither CI lane installs the extra)."""

    def __enter__(self) -> _NullWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def append_data(self, frame: Any) -> None:
        return None


def _stub_imageio(monkeypatch: pytest.MonkeyPatch) -> None:
    v2 = types.ModuleType("imageio.v2")
    v2.get_writer = lambda *_a, **_k: _NullWriter()  # type: ignore[attr-defined]
    pkg = types.ModuleType("imageio")
    pkg.v2 = v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)


@pytest.fixture(scope="module")
def plate_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return G._write_plate_results(tmp_path_factory.mktemp("x2_numeric_anim"))


@pytest.fixture
def plate(plate_h5: Path, monkeypatch: pytest.MonkeyPatch):
    from apeGmsh.viewers.ui.theme import PALETTES, THEME
    from tests.conftest import _open_model_from_h5

    pytest.importorskip("qtpy.QtWidgets")
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    monkeypatch.setattr(THEME, "_current", PALETTES[G._MANIFEST["theme"]])
    _stub_imageio(monkeypatch)
    with Results.from_native(
        plate_h5, model=_open_model_from_h5(plate_h5),
    ) as r:
        yield r


def _contour_setup(plotter: Any, director: Any) -> None:
    """X2-a's ``setup=``: the viewer's S1 starter contour."""
    from apeGmsh.viewers.diagrams._starter import add_default_contour

    add_default_contour(director)


def _animate(
    plate: Results, tmp_path: Path, case: str, **over: Any,
) -> list[dict[str, Any]]:
    """The frames' drawn data of ``export_animation`` for ``case``."""
    spec = _ANIM["cases"][case]
    kw: dict[str, Any] = dict(
        fps=_ANIM["fps"], step_stride=_ANIM["step_stride"],
        camera=_ANIM["camera"], window_size=tuple(_ANIM["window"]),
    )
    if spec.get("deform") is not None:
        kw["deform"] = tuple(spec["deform"])
    if spec.get("contour") is not None:
        kw["setup"] = _contour_setup
    kw.update(over)
    with NO.capture_drawn() as recs:
        out = plate.export_animation(tmp_path / f"{case}.gif", **kw)
    assert out == tmp_path / f"{case}.gif"
    frames = [r for r in recs if r["kind"] == "vtk"]
    assert frames, "export_animation shot no frame through the seam"
    return frames


def _decode_steps(
    frames: list[dict[str, Any]], f: dict[str, Any], *,
    scale: float, contour: str | None,
) -> list[int | None]:
    """The step each frame draws (``None`` if no step-dependent data)."""
    out: list[int | None] = []
    for rec in frames:
        hits = []
        for s in _STEPS:
            key = "probe"
            gold = NO.flatten(key, rec)
            try:
                NO.check_golden_physics(
                    gold, key, f, step=s, scale=scale, contour=contour)
            except AssertionError:
                continue
            hits.append(s)
        out.append(hits[0] if len(hits) == 1 else None)
    return out


def _case_meta(case: str) -> tuple[float, str | None]:
    spec = _ANIM["cases"][case]
    scale = float(spec["deform"][1]) if spec.get("deform") else 0.0
    return scale, ("displacement_x" if spec.get("contour") else None)


@pytest.mark.parametrize("case", _CASES)
def test_export_animation_draws_golden_data(
    plate, tmp_path: Path, case: str,
) -> None:
    frames = _animate(plate, tmp_path, case)
    scale, contour = _case_meta(case)
    if NO.REGEN:
        entries: dict[str, np.ndarray] = {}
        for i, rec in enumerate(frames):
            entries.update(NO.flatten(f"anim.{case}.f{i}", rec))
        NO.write_golden(entries, {f"anim.{case}": {
            "frames": len(frames),
            "frame_steps": _ANIM["frame_steps"],
            "deform_scale": scale,
            "contour": None if frames[0]["contour"] is None
            else frames[0]["contour"]["name"],
        }})
        return
    man = NO.load_manifest()[f"anim.{case}"]
    assert len(frames) == man["frames"] == len(_ANIM["frame_steps"])
    gold = NO.load_golden()
    for i, rec in enumerate(frames):
        NO.assert_record(rec, f"anim.{case}.f{i}", gold)
    if scale or contour:
        f = NO.fixture_fields(plate)
        steps = _decode_steps(frames, f, scale=scale, contour=contour)
        assert steps == man["frame_steps"] == [0, 3, 4], steps


def test_animation_goldens_are_the_fixture_data(plate) -> None:
    if NO.REGEN:
        pytest.skip("regenerating")
    gold, man, f = NO.load_golden(), NO.load_manifest(), NO.fixture_fields(plate)
    for case in _CASES:
        m = man[f"anim.{case}"]
        assert m["frame_steps"] == [0, 3, 4]
        for i, step in enumerate(m["frame_steps"]):
            NO.check_golden_physics(
                gold, f"anim.{case}.f{i}", f, step=step,
                scale=m["deform_scale"], contour=m["contour"],
            )


# =====================================================================
# Revert-proof (kept as tests): each mutant must FAIL the oracle
# =====================================================================


def _fails(frames: list[dict[str, Any]], case: str) -> bool:
    gold = NO.load_golden()
    if len(frames) != NO.load_manifest()[f"anim.{case}"]["frames"]:
        return True
    try:
        for i, rec in enumerate(frames):
            NO.assert_record(rec, f"anim.{case}.f{i}", gold)
    except AssertionError:
        return True
    return False


@pytest.mark.parametrize("case", ["deform", "contour"])
def test_oracle_rejects_wrong_schedule(plate, tmp_path: Path, case: str) -> None:
    """Every step drawn once (stride 1), then re-sequenced: the true
    schedule passes (positive control); a wrong step or [0,2,4] fails."""
    if NO.REGEN:
        pytest.skip("regenerating")
    every = _animate(plate, tmp_path, case, step_stride=1)
    assert len(every) == len(_STEPS)
    pick = lambda *s: [every[i] for i in s]  # noqa: E731
    assert not _fails(pick(0, 3, 4), case), "true schedule rejected"
    assert _fails(pick(0, 3, 3), case), "frame 2 drawn at step 3 accepted"
    assert _fails(pick(0, 2, 4), case), "schedule [0,2,4] accepted"
    assert _fails(_animate(plate, tmp_path, case, step_stride=2), case), (
        "export_animation(step_stride=2) accepted")


def test_oracle_rejects_wrong_deform_scale(plate, tmp_path: Path) -> None:
    if NO.REGEN:
        pytest.skip("regenerating")
    frames = _animate(plate, tmp_path, "deform", deform=("displacement", 1.01))
    assert _fails(frames, "deform")


def test_oracle_rejects_wrong_contour(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The starter drawing ``displacement_y``, and a 1% wrong clim."""
    if NO.REGEN:
        pytest.skip("regenerating")
    frames = _animate(plate, tmp_path, "contour")
    assert not _fails(frames, "contour")
    bad = [{**r, "contour": {**r["contour"], "clim": r["contour"]["clim"] * 1.01}}
           for r in frames]
    assert _fails(bad, "contour"), "wrong clim accepted"
    from apeGmsh.viewers.diagrams import _starter

    monkeypatch.setattr(
        _starter, "default_contour_component", lambda _d: "displacement_y")
    assert _fails(_animate(plate, tmp_path, "contour"), "contour"), (
        "wrong contour component accepted")
