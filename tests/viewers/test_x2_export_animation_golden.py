"""X2-a — golden frames for the public ``Results.export_animation``.

The X2 oracle for the animation verb (see ``test_x2_golden_stills.py`` for
the stills and the comparison). Since X2-b (#1663) the verb re-steps a
``ResultsSession`` pane offscreen (no Qt), so this runs in ``suite``; it
was ``qt``-marked while the verb built a Qt ``ResultsViewer``.

Each case calls only ``Results.export_animation(path, ...)`` and records
what reaches the encoder. ``imageio.v2`` is replaced by a frame recorder:
neither CI lane installs the ``animation`` extra, and the encoding is
imageio's, while the frame schedule and pixels are apeGmsh's. With
``n_steps=5`` and ``step_stride=3`` the schedule is steps ``[0, 3, 4]``
(``range(0, 5, 3)`` plus the forced last step), so 3 frames.

Cases (``manifest.json`` ``animation.cases``):

* ``default``: no ``deform=``, no ``setup=``: the viewer's boot scene;
* ``deform``: ``deform=("displacement", 1.0)``: the public deform knob;
* ``contour``: ``setup=`` adds the S1 starter's contour (its component
  heuristic, ``default_contour_component``; with no ``displacement_z``
  recorded it picks ``displacement_x``). ``setup(plotter, view)`` is the
  documented escape hatch; X2-b re-expressed it for the session (the
  view's contour slot), the other two cases run unchanged.

The frame schedule ``[0, 3, 4]`` is asserted from the session instant at
each captured frame, not only the count.

The goldens are CI's ``suite`` frames of the session path (Linux, Mesa),
re-baselined by X2-b under the maintainer ruling of 2026-10-11 (#1663):
the session look is accepted, the drawn data is held by the numeric
oracle (``test_x2_numeric_oracle_anim.py``), the tolerance is unchanged.
Off Linux a miss skips and names the platform; on Linux it fails and
carries the actual frame as base64 (see the stills module for
regeneration and the runner-image risk).
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

_ANIM = G._MANIFEST["animation"]


class _RecordingWriter:
    """Stand-in for an imageio writer: keeps every appended frame."""

    def __init__(self, sink: list[np.ndarray]) -> None:
        self._sink = sink

    def __enter__(self) -> _RecordingWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def append_data(self, frame: np.ndarray) -> None:
        self._sink.append(np.array(frame, copy=True))


def _record_frames(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    sink: list[np.ndarray] = []
    v2 = types.ModuleType("imageio.v2")
    def get_writer(path: str, **_k: Any) -> _RecordingWriter:
        Path(path).touch()  # a real encoder opens its file (X2-b F5)
        return _RecordingWriter(sink)

    v2.get_writer = get_writer  # type: ignore[attr-defined]
    pkg = types.ModuleType("imageio")
    pkg.v2 = v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)
    return sink


@pytest.fixture(scope="module")
def plate_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return G._write_plate_results(tmp_path_factory.mktemp("x2_anim"))


@pytest.fixture
def plate(plate_h5: Path, monkeypatch: pytest.MonkeyPatch):
    from apeGmsh.viewers.ui.theme import PALETTES, THEME
    from tests.conftest import _open_model_from_h5

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    monkeypatch.setattr(THEME, "_current", PALETTES[G._MANIFEST["theme"]])
    with Results.from_native(
        plate_h5, model=_open_model_from_h5(plate_h5),
    ) as r:
        yield r


def _contour_setup(results: Results):
    """The S1 starter's contour, on the session view's contour slot."""
    from apeGmsh.results.session import Contour
    from apeGmsh.viewers.diagrams import _starter

    def setup(plotter: Any, view: Any) -> None:
        owner = types.SimpleNamespace(
            results=results, stages=lambda: list(results.stages))
        view.contour = Contour(_starter.default_contour_component(owner))

    return setup


def _record_steps(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The session instant at every captured frame."""
    from pyvista.plotting.plotter import BasePlotter

    sessions: list[Any] = []
    steps: list[int] = []
    make, shot = Results.session, BasePlotter.screenshot

    def session(self: Results) -> Any:
        sessions.append(make(self))
        return sessions[-1]

    def screenshot(self: Any, *a: Any, **k: Any) -> Any:
        steps.append(sessions[-1].time.step)
        return shot(self, *a, **k)

    monkeypatch.setattr(Results, "session", session)
    monkeypatch.setattr(BasePlotter, "screenshot", screenshot)
    return steps


@pytest.mark.parametrize("case", sorted(_ANIM["cases"]))
def test_export_animation_matches_golden(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    spec = _ANIM["cases"][case]
    frames = _record_frames(monkeypatch)
    steps = _record_steps(monkeypatch)
    kwargs: dict[str, Any] = {}
    if spec.get("deform") is not None:
        kwargs["deform"] = tuple(spec["deform"])
    if spec.get("contour") is not None:
        kwargs["setup"] = _contour_setup(plate)
    out = tmp_path / f"{case}.gif"
    written = plate.export_animation(
        out,
        fps=_ANIM["fps"],
        step_stride=_ANIM["step_stride"],
        camera=_ANIM["camera"],
        window_size=tuple(_ANIM["window"]),
        **kwargs,
    )
    assert written == out
    assert len(frames) == _ANIM["frame_count"] == len(_ANIM["frame_steps"])
    assert steps == _ANIM["frame_steps"] == [0, 3, 4], steps
    G._assert_goldens((frames[0], spec["first"]), (frames[-1], spec["last"]))
