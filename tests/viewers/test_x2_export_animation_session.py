"""X2-b (#1663) — ``export_animation`` re-steps a ``ResultsSession``.

``Results.export_animation`` and the studio ``animate`` verb used to
build a Qt ``ResultsViewer`` and drive its ``ResultsDirector``. They now
loop over one session mesh pane (``viewers/session/_animation.py``). The
drawn data is held by the X2-e numeric oracle
(``test_x2_numeric_oracle_anim.py``) and the look by the re-baselined
``anim_*`` goldens; this file holds the rest of the contract:

* neither public door constructs a ``ResultsViewer`` or a
  ``ResultsDirector``, nor calls the legacy ``viewers.animation`` loop
  (the regression proof: with the old body restored these fail);
* the frame schedule is ``range(0, n, stride)`` plus the last step;
* a failed frame leaves no file at ``path`` and no ``.partial``;
* ``setup(plotter, view)`` gets the session ``MeshView``, and a callback
  written against the retired director fails loudly;
* the studio ``animate`` verb, ``kind=history`` and ``kind=yield``,
  reaches the same path through its public verb entry.
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


class _Writer:
    """Stand-in imageio writer: records frames, may fail on one."""

    def __init__(self, path: str, sink: list, fail_at: int | None) -> None:
        self._sink = sink
        self._fail_at = fail_at
        Path(path).write_bytes(b"partial")   # the encoder opened its file

    def __enter__(self) -> _Writer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def append_data(self, frame: Any) -> None:
        if self._fail_at is not None and len(self._sink) == self._fail_at:
            raise RuntimeError("encoder failed mid-run")
        self._sink.append(np.asarray(frame).shape)


def _stub_imageio(
    monkeypatch: pytest.MonkeyPatch, fail_at: int | None = None,
) -> tuple[list, list]:
    sink: list = []
    paths: list = []

    def get_writer(path: str, **_k: Any) -> _Writer:
        paths.append(path)
        return _Writer(path, sink, fail_at)

    v2 = types.ModuleType("imageio.v2")
    v2.get_writer = get_writer  # type: ignore[attr-defined]
    pkg = types.ModuleType("imageio")
    pkg.v2 = v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)
    return sink, paths


@pytest.fixture
def no_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the old director, or calling the old loop, fails the
    test. (The old window itself was deleted at X3-d, #1734.)"""
    from apeGmsh.viewers import animation
    from apeGmsh.viewers.diagrams import _director

    def _boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("the legacy viewer / director path was reached")

    monkeypatch.setattr(_director.ResultsDirector, "__init__", _boom)
    monkeypatch.setattr(animation, "export_animation", _boom)


@pytest.fixture(scope="module")
def plate_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return G._write_plate_results(tmp_path_factory.mktemp("x2b_plate"))


@pytest.fixture
def plate(plate_h5: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.conftest import _open_model_from_h5

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    with Results.from_native(
        plate_h5, model=_open_model_from_h5(plate_h5),
    ) as r:
        yield r


def _require_gl(written: Any) -> None:
    import os

    if written is None and not os.environ.get("APEGMSH_EXPECT_GL"):
        pytest.skip("no offscreen GL context")
    assert written is not None


def _session_steps(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The session instant at every captured frame."""
    from pyvista.plotting.plotter import BasePlotter

    sessions: list[Any] = []
    steps: list[int] = []
    make = Results.session
    shot = BasePlotter.screenshot

    def session(self: Results) -> Any:
        s = make(self)
        sessions.append(s)
        return s

    def screenshot(self: Any, *a: Any, **k: Any) -> Any:
        steps.append(sessions[-1].time.step)
        return shot(self, *a, **k)

    monkeypatch.setattr(Results, "session", session)
    monkeypatch.setattr(BasePlotter, "screenshot", screenshot)
    return steps


@pytest.mark.parametrize("stride, expected", [
    (3, [0, 3, 4]), (1, [0, 1, 2, 3, 4]), (2, [0, 2, 4]), (10, [0, 4]),
])
def test_results_export_animation_is_a_session_restep(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_legacy: None,
    stride: int, expected: list[int],
) -> None:
    sink, paths = _stub_imageio(monkeypatch)
    steps = _session_steps(monkeypatch)
    out = tmp_path / "a.gif"
    written = plate.export_animation(
        out, fps=10, step_stride=stride, camera="xy",
        deform=("displacement", 1.0), window_size=(320, 180),
    )
    _require_gl(written)
    assert written == out and out.read_bytes() == b"partial"
    assert steps == expected
    assert len(sink) == len(expected)
    assert {s[:2] for s in sink} == {(180, 320)}
    assert paths == [str(tmp_path / "a.partial.gif")]
    assert not (tmp_path / "a.partial.gif").exists()


def test_a_failed_frame_leaves_no_file(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_legacy: None,
) -> None:
    _stub_imageio(monkeypatch, fail_at=1)
    out = tmp_path / "b.gif"
    try:
        written = plate.export_animation(
            out, fps=10, step_stride=1, window_size=(160, 90),
        )
    except RuntimeError as exc:
        assert "encoder failed" in str(exc)
    else:
        _require_gl(written)
        pytest.fail("a failed frame was reported as success")
    assert not out.exists()
    assert not (tmp_path / "b.partial.gif").exists()


def test_setup_gets_the_session_view(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_legacy: None,
) -> None:
    from apeGmsh.results.session import Contour, MeshView

    _stub_imageio(monkeypatch)
    seen: list[Any] = []

    def setup(plotter: Any, view: Any) -> None:
        seen.append(view)
        view.contour = Contour("displacement_x")

    written = plate.export_animation(
        tmp_path / "c.gif", step_stride=3, window_size=(160, 90),
        setup=setup,
    )
    _require_gl(written)
    assert len(seen) == 1 and isinstance(seen[0], MeshView)
    assert seen[0].contour == Contour("displacement_x")


def test_a_director_style_setup_fails_loudly(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_imageio(monkeypatch)

    def old_setup(plotter: Any, director: Any) -> None:
        director.registry.add(object())

    with pytest.raises(TypeError, match="not a ResultsDirector"):
        written = plate.export_animation(
            tmp_path / "d.gif", window_size=(160, 90), setup=old_setup,
        )
        _require_gl(written)
    assert not (tmp_path / "d.gif").exists()
    assert not (tmp_path / "d.partial.gif").exists()


def test_a_callback_bug_is_not_misdiagnosed(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F4: only director-surface lookups on the view are mapped."""
    _stub_imageio(monkeypatch)

    def buggy(plotter: Any, view: Any) -> None:
        object().missing  # noqa: B018 - a real bug in the callback

    with pytest.raises(AttributeError, match="missing"):
        written = plate.export_animation(
            tmp_path / "f.gif", window_size=(160, 90), setup=buggy,
        )
        _require_gl(written)


def test_an_encoder_that_wrote_nothing_is_not_success(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F5: no ``.partial`` file means nothing was written."""
    _stub_imageio(monkeypatch)
    import imageio.v2 as v2  # the stub

    real = v2.get_writer

    def no_file(path: str, **k: Any) -> Any:
        w = real(path, **k)
        Path(path).unlink()
        return w

    monkeypatch.setattr(v2, "get_writer", no_file)
    out = tmp_path / "g.gif"
    try:
        written = plate.export_animation(out, window_size=(160, 90))
    except RuntimeError as exc:
        assert "wrote no file" in str(exc)
    else:
        _require_gl(written)
        pytest.fail(f"reported {written!r} with no file written")
    assert not out.exists()


def test_bad_arguments_refuse_before_any_frame(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unsupported suffix"):
        plate.export_animation(tmp_path / "e.avi")
    with pytest.raises(ValueError, match="fps must be positive"):
        plate.export_animation(tmp_path / "e.gif", fps=0)
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    assert plate.export_animation(tmp_path / "e.gif") is None
    assert not (tmp_path / "e.gif").exists()


# =====================================================================
# stage=None keeps the old resolution (review F2)
# =====================================================================


@pytest.fixture
def two_stage(g, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Stage ``s0`` (2 steps) then ``s1`` (4 steps) on one cube."""
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
    path = tmp_path / "two.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        for sid_name, n in (("s0", 2), ("s1", 4)):
            sid = w.begin_stage(
                name=sid_name, kind="static", stage_id=sid_name,
                time=np.arange(n, dtype=np.float64),
            )
            w.write_nodes(sid, "partition_0", node_ids=ids, components={
                f"displacement_{a}": np.stack(
                    [xyz[:, i] * 0.01 * (t + 1) for t in range(n)])
                for i, a in enumerate("xyz")
            })
            w.end_stage()
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
        yield r


def _session_instants(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    from pyvista.plotting.plotter import BasePlotter

    sessions: list[Any] = []
    seen: list[tuple] = []
    make, shot = Results.session, BasePlotter.screenshot

    def session(self: Results) -> Any:
        sessions.append(make(self))
        return sessions[-1]

    def screenshot(self: Any, *a: Any, **k: Any) -> Any:
        t = sessions[-1].time
        seen.append((t.stage, t.step))
        return shot(self, *a, **k)

    monkeypatch.setattr(Results, "session", session)
    monkeypatch.setattr(BasePlotter, "screenshot", screenshot)
    return seen


@pytest.mark.parametrize("how, expected", [
    ("scoped", [("s0", 0), ("s0", 1)]),
    ("explicit", [("s1", 0), ("s1", 1), ("s1", 2), ("s1", 3)]),
])
def test_stage_resolution_matches_the_old_path(
    two_stage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    no_legacy: None, how: str, expected: list,
) -> None:
    """A ``.stage()``-scoped Results animates ITS stage, not the last."""
    _stub_imageio(monkeypatch)
    seen = _session_instants(monkeypatch)
    out = tmp_path / f"{how}.gif"
    if how == "scoped":
        written = two_stage.stage("s0").export_animation(
            out, window_size=(160, 90))
    else:
        written = two_stage.export_animation(
            out, stage="s1", window_size=(160, 90))
    _require_gl(written)
    assert seen == expected


def test_several_stages_and_no_pick_refuses(
    two_stage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_imageio(monkeypatch)
    with pytest.raises(RuntimeError, match="stage="):
        two_stage.export_animation(tmp_path / "h.gif", window_size=(160, 90))
    assert not (tmp_path / "h.gif").exists()


def test_combined_stage_refuses_and_names_the_alternative(
    two_stage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Maintainer ruling 2026-10-11 (#1663): ``stage='__all__'`` (the
    director's combined stage) refuses clearly instead of a KeyError."""
    _stub_imageio(monkeypatch)
    out = tmp_path / "all.gif"
    with pytest.raises(ValueError, match=r"stage=<id or name>") as info:
        two_stage.export_animation(out, stage="__all__")
    assert "'s0'" in str(info.value) and "'s1'" in str(info.value)
    assert not out.exists() and not (tmp_path / "all.partial.gif").exists()


# =====================================================================
# The studio MCP ``animate`` verb — the second public door
# =====================================================================

_T = 3


@pytest.fixture
def stress_h5(g, tmp_path: Path) -> Path:
    """A cube with nodal displacement and a Gauss stress tensor, so
    ``kind=yield`` resolves ``von_mises_stress``."""
    from apeGmsh.results.writers import NativeWriter

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
    eids = np.concatenate(
        [np.asarray(grp.ids, dtype=np.int64) for grp in fem.elements])
    nodal = {
        f"displacement_{a}": np.stack(
            [xyz[:, i] * 0.01 * (t + 1) for t in range(_T)])
        for i, a in enumerate("xyz")
    }
    gauss = {}
    for k, comp in enumerate(("stress_xx", "stress_yy", "stress_zz",
                              "stress_xy", "stress_yz", "stress_xz")):
        vals = np.zeros((_T, eids.size, 1))
        for t in range(_T):
            vals[t, :, 0] = (eids % 7) * (k + 1) + t
        gauss[comp] = vals
    path = tmp_path / "stress.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="load", kind="static", stage_id="load",
            time=np.arange(_T, dtype=np.float64),
        )
        w.write_nodes(sid, "partition_0", node_ids=ids, components=nodal)
        w.write_gauss_group(
            sid, "partition_0", "group_0", class_tag=4, int_rule=1,
            element_index=eids, natural_coords=np.zeros((1, 3)),
            components=gauss,
        )
        w.end_stage()
    return path


@pytest.mark.parametrize("kind", ["history", "yield"])
def test_studio_animate_verb_reaches_the_session_path(
    stress_h5: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    no_legacy: None, kind: str,
) -> None:
    from apeGmsh.studio._verbs import animate_artifact

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    sink, _ = _stub_imageio(monkeypatch)
    steps = _session_steps(monkeypatch)
    dest = tmp_path / f"{kind}.gif"
    payload = animate_artifact(stress_h5, dest, kind=kind, fps=5)
    assert payload["ok"] is True and payload["kind"] == kind
    _require_gl(payload["output"])
    assert payload["output"] == str(dest)
    assert steps == list(range(_T)) and len(sink) == _T
    if kind == "yield":
        assert payload["component"] == "von_mises_stress"
