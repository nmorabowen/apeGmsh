"""X2-d (#1665) — ``render`` / ``render_pack`` / ``assess`` / CLI on the
session path.

What the session path DRAWS is pinned by the numeric oracle
(``test_x2_numeric_oracle.py``) and what it LOOKS like by the image
goldens (``test_x2_golden_stills.py``). This module pins the move itself:

1. every public door (``results.render`` per view, ``render_pack``,
   ``assess(figures=True)``, ``python -m apeGmsh.viewers render``)
   draws with no ``ResultsDirector`` constructed;
2. the ``view=`` token becomes the session IR the module docstring of
   ``viewers/session/_results_stills.py`` documents (no GL needed);
3. the two stills the session path refuses (maintainer ruling
   2026-10-11): per-axis reactions and ``deform=`` by an arbitrary
   field. Both refuse before the skip check, so they run everywhere.

Revert-proof: with ``Results.render`` / ``render_pack`` pointed back at
``viewers/render.py`` the director tests fail (the patched constructor
raises) and the refusal tests fail (the old path returns ``None`` under
``APEGMSH_SKIP_VIEWER=1`` instead of raising).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import apeGmsh
from apeGmsh.results import Results
from tests.viewers import test_x2_golden_stills as G

_SRC = str(Path(apeGmsh.__file__).resolve().parents[1])
_WINDOW = (320, 180)


@pytest.fixture(scope="module")
def plate_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return G._write_plate_results(tmp_path_factory.mktemp("x2_render_session"))


@pytest.fixture
def plate(plate_h5: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.conftest import _open_model_from_h5

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    with Results.from_native(
        plate_h5, model=_open_model_from_h5(plate_h5),
    ) as r:
        yield r


@pytest.fixture
def no_director(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any ``ResultsDirector`` construction fails the test."""
    from apeGmsh.viewers.diagrams import _director

    def _boom(self: Any, *a: Any, **k: Any) -> None:
        raise AssertionError("a ResultsDirector was constructed")

    monkeypatch.setattr(_director.ResultsDirector, "__init__", _boom)


def _gl_or_skip(written: Any) -> None:
    if written:
        return
    if os.environ.get("APEGMSH_EXPECT_GL"):
        raise AssertionError("no still written on a GL lane")
    pytest.skip("no offscreen GL context")


# =====================================================================
# 1. No director behind any door
# =====================================================================


@pytest.mark.parametrize("view", ["mesh", "contour", "deformed", "reactions"])
def test_render_constructs_no_director(
    plate, no_director, tmp_path: Path, view: str,
) -> None:
    out = tmp_path / f"{view}.png"
    written = plate.render(
        out, view=view, camera="xy", window_size=_WINDOW,
        component="reactions" if view == "reactions" else None,
    )
    _gl_or_skip(written)
    assert written == out and out.stat().st_size > 0


def test_render_pack_constructs_no_director(
    plate, no_director, tmp_path: Path,
) -> None:
    written = plate.render_pack(tmp_path / "pack", window_size=_WINDOW)
    _gl_or_skip(written)
    assert [p.name for p in written] == G._MANIFEST["render_pack"]["files"]


def test_assess_figures_construct_no_director(
    plate, no_director, tmp_path: Path,
) -> None:
    report = plate.assess(figures=True, out_dir=tmp_path / "assess")
    _gl_or_skip(report.figures)
    assert [p.name for p in report.figures] == G._MANIFEST["assess_figures"]


# =====================================================================
# 2. view= -> session IR (no GL)
# =====================================================================


def test_view_tokens_become_one_session_view(plate) -> None:
    from apeGmsh.results.session import (
        Contour, Deform, Instant, MeshStyle, Reactions,
    )
    from apeGmsh.viewers.session._results_stills import session_for_view

    stage = plate.stages[-1].id
    expect = {
        "mesh": ({}, None, MeshStyle(mesh=True, outlines=False)),
        "contour": (
            {"contour": Contour("displacement_x")}, None,
            MeshStyle(mesh=False, outlines=False),
        ),
        "deformed": (
            {"contour": Contour("displacement_x")}, Deform(),
            MeshStyle(mesh=False, outlines=False),
        ),
        "reactions": (
            {"reactions": Reactions()}, None,
            MeshStyle(mesh=True, outlines=False),
        ),
    }
    for view, (slots, deform, style) in expect.items():
        s = session_for_view(
            plate, view=view, component=None, step=-1, deform_spec=None,
        )
        (pane,) = s.panes
        assert dict(pane.slots) == slots, view
        assert pane.deform == deform, view
        assert pane.style == style, view
        assert s.time == Instant(stage=stage, step=4), view
    s = session_for_view(
        plate, view="contour", component="displacement_y", step=1,
        deform_spec=("displacement", 2.0),
    )
    (pane,) = s.panes
    assert pane.contour == Contour("displacement_y")
    assert pane.deform == Deform(field="displacement", scale=2.0)
    assert s.time == Instant(stage=stage, step=1)


# =====================================================================
# 3. The two refusals (maintainer ruling 2026-10-11)
# =====================================================================


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_per_axis_reactions_refused_naming_alternative(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, axis: str,
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    with pytest.raises(ValueError, match="component='reactions'") as info:
        plate.render(
            tmp_path / "r.png", view="reactions", component=f"reaction_{axis}",
        )
    assert f"reaction_force_{axis}" in str(info.value)
    assert not (tmp_path / "r.png").exists()


@pytest.mark.parametrize("view", ["contour", "deformed", "reactions"])
def test_arbitrary_deform_field_refused_naming_alternative(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, view: str,
) -> None:
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    with pytest.raises(ValueError, match="deform=SCALE") as info:
        plate.render(
            tmp_path / "d.png", view=view, deform=("reaction_force", 1.0),
        )
    assert "'displacement'" in str(info.value)


@pytest.mark.parametrize("view", ["contour", "deformed"])
def test_unrecorded_contour_keeps_the_runtimeerror_type(
    plate, tmp_path: Path, view: str,
) -> None:
    """The old path raised ``NoDataError`` (a ``RuntimeError``) for a
    contour of an unrecorded component. The session path keeps that
    type, and is a ``ValueError`` as well. Raised before any plotter
    opens, so no GL is needed."""
    with pytest.raises(RuntimeError, match="not recorded") as info:
        plate.render(
            tmp_path / "c.png", view=view, component="no_such_component",
        )
    assert isinstance(info.value, ValueError)
    assert not (tmp_path / "c.png").exists()


def test_supported_forms_still_skip_cleanly(
    plate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusals are narrow: the supported spellings still pass."""
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    for kw in (
        dict(view="reactions", component="reactions"),
        dict(view="reactions", component=None),
        dict(view="contour", deform=("velocity", 1.0)),
        dict(view="mesh", deform=("reaction_force", 1.0)),  # mesh: no warp
    ):
        assert plate.render(tmp_path / "ok.png", **kw) is None, kw


# =====================================================================
# 4. The CLI door, as a real subprocess
# =====================================================================


def _cli(args: list[str], *, skip: bool, code: str | None = None):
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("APEGMSH_SKIP_VIEWER", None)
    if skip:
        env["APEGMSH_SKIP_VIEWER"] = "1"
    cmd = (
        [sys.executable, "-m", "apeGmsh.viewers", *args] if code is None
        else [sys.executable, "-c", code, *args]
    )
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=300,
    )


# Runs the real CLI main with ResultsDirector.__init__ made to raise.
_NO_DIRECTOR_MAIN = (
    "import sys\n"
    "from apeGmsh.viewers.diagrams import _director\n"
    "def _boom(self, *a, **k):\n"
    "    raise AssertionError('a ResultsDirector was constructed')\n"
    "_director.ResultsDirector.__init__ = _boom\n"
    "from apeGmsh.viewers.__main__ import main\n"
    "sys.exit(main(sys.argv[1:]))\n"
)


@pytest.mark.subprocess
def test_cli_render_and_pack_construct_no_director(
    plate_h5: Path, tmp_path: Path,
) -> None:
    out = tmp_path / "deformed.png"
    one = _cli(
        ["render", str(plate_h5), str(out), "--view", "deformed",
         "--camera", "xy", "--window", "320x180"],
        skip=False, code=_NO_DIRECTOR_MAIN,
    )
    assert one.returncode == 0, one.stderr
    dest = tmp_path / "pack"
    pack = _cli(
        ["render", str(plate_h5), "--pack", str(dest), "--window", "320x180"],
        skip=False, code=_NO_DIRECTOR_MAIN,
    )
    assert pack.returncode == 0, pack.stderr
    if not out.exists():
        _gl_or_skip(None)
    assert out.stat().st_size > 0
    assert sorted(p.name for p in dest.iterdir()) == sorted(
        G._MANIFEST["render_pack"]["files"])


@pytest.mark.subprocess
def test_cli_refuses_per_axis_reactions(plate_h5: Path, tmp_path: Path) -> None:
    proc = _cli(
        ["render", str(plate_h5), str(tmp_path / "r.png"),
         "--view", "reactions", "--component", "reaction_x"],
        skip=True,
    )
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "component='reactions'" in proc.stderr
