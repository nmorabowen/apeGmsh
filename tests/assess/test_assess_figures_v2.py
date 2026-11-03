"""VA6b (ADR 0128 slice 6): ``results.assess(figures=True)``, ``results.render``
and ``results.render_pack`` draw through ``apegmsh-view/1`` on Windows.

The parity plan (``internal_docs/plan_viewer_agent_stills_parity.md``)
rules the outcomes: assess's pack is ONE v2 pack spec (G5), a ``Results``
reaches the app only through the ``.ladruno`` it was read from and every
other source is refused by name (R22b), no app raises (G9), and Linux keeps
the VTK path (G2). ``Results.path`` is the public accessor that names the
source file.

The ``app_backend`` fixture takes the Windows path on any platform with
the app replaced by a fake that records each spec and writes stand-in
PNGs, as ``apeGmshViewer --render`` does on exit 0. No test here opens a
window or draws a VTK frame. A live check against a built apeGmshViewer is
the reviewer's (see the PR body): there is no live oracle in this file.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.assess._report import NO_STILLS
from apeGmsh.results import Results

from tests.assess.test_assess import _make_fem, _write_nodes
from tests.conftest import _open_model_from_h5


class _FakeApp:
    """``agent_view.render_stills`` without the app: records each spec and
    writes one stand-in PNG per shot plus an all-ok manifest."""

    def __init__(self) -> None:
        self.specs: list[dict] = []

    def __call__(self, spec, out, **kwargs):
        self.specs.append(json.loads(json.dumps(spec)))
        out = Path(out)
        assert not out.exists(), "the app refuses an --out that is not empty"
        out.mkdir(parents=True)
        shots = []
        for shot in spec["shots"]:
            (out / f"{shot['name']}.png").write_bytes(b"\x89PNG stand-in")
            shots.append({"name": shot["name"], "status": "ok", "png": f"{shot['name']}.png"})
        manifest = {"spec": spec["spec"], "shots": shots}
        (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


def _boom(*args, **kwargs):
    raise AssertionError("agent_view.render_stills was called")


@pytest.fixture
def app_backend(monkeypatch):
    """The Windows path, on any platform, with a fake app."""
    import apeGmsh.agent_view as agent_view
    from apeGmsh.viewers import render as render_mod

    fake = _FakeApp()
    monkeypatch.setattr(render_mod, "_stills_backend", lambda: "app")
    monkeypatch.setattr(agent_view, "render_stills", fake)
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    return fake


@pytest.fixture
def ladruno_set(tmp_path: Path) -> Path:
    """The demo run as one stem, ``run.h5`` beside ``run.ladruno``: the
    pair the app opens (its ``main/pairing.ts`` stem rule)."""
    src = tmp_path / "demo"
    Results.demo(n_steps=3, n_elements=4, path=src).close()
    run = tmp_path / "set"
    run.mkdir()
    shutil.copy(src / "demo_model.h5", run / "run.h5")
    shutil.copy(src / "demo_results.ladruno", run / "run.ladruno")
    return run / "run.ladruno"


def _auto_scale_oracle(ladruno: Path) -> float:
    """G3's closed form from the raw slabs: 0.12 of the bounding-box
    diagonal over the largest |u| component at the last step."""
    with Results.from_ladruno(ladruno, model_h5=ladruno.with_suffix(".h5")) as r:
        coords = np.asarray(r.fem.nodes.coords, dtype=float)
        stage = r.stage(r.stages[-1].id)
        comps = set(stage.nodes.available_components())
        peak = max(
            float(np.nanmax(np.abs(np.asarray(stage.nodes.get(component=c, time=-1).values))))
            for c in ("displacement_x", "displacement_y", "displacement_z") if c in comps
        )
    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    return 0.12 * diag / peak


# ---------------------------------------------------------------------
# The accessor
# ---------------------------------------------------------------------


def test_results_path_names_the_source_file(ladruno_set: Path) -> None:
    with Results.from_ladruno(ladruno_set) as r:
        assert r.path == ladruno_set
        # Stage scopes carry it: a scoped still draws the same file.
        assert r.stage(r.stages[0].id).path == ladruno_set
        with pytest.raises(AttributeError):
            r.path = Path("elsewhere.ladruno")  # read-only


# ---------------------------------------------------------------------
# assess(figures=True): one v2 pack spec (G5)
# ---------------------------------------------------------------------


def test_assess_figures_is_one_v2_pack_spec(app_backend, ladruno_set, tmp_path):
    dest = tmp_path / "assess"
    with Results.from_ladruno(ladruno_set) as r:
        report = r.assess(figures=True, out_dir=dest)
    assert len(app_backend.specs) == 1
    spec = app_backend.specs[0]
    assert spec["spec"] == "apegmsh-view/1"
    assert spec["chrome"] == "model"  # the canvas only, not the app's panels
    assert spec["files"] == {
        "model": str(ladruno_set.with_suffix(".h5").resolve()),
        "results": str(ladruno_set.resolve()),
    }
    shots = {s["name"]: s for s in spec["shots"]}
    assert list(shots) == ["mesh", "contour", "deformed"]  # reactions dropped (G1)
    assert "stage" not in shots["mesh"]
    for name in ("contour", "deformed"):
        assert shots[name]["stage"] == 0
        assert shots[name]["step"] == 2  # the last of 3 steps
        assert shots[name]["contour"] == {"field": "DISPLACEMENT", "component": "z"}
    assert "deform" not in shots["contour"]
    assert shots["deformed"]["deform"]["scale"] == pytest.approx(
        _auto_scale_oracle(ladruno_set), rel=1e-12,
    )
    names = [p.name for p in report.figures]
    assert names[:3] == ["mesh.png", "contour.png", "deformed.png"]
    assert set(names) <= {"mesh.png", "contour.png", "deformed.png", "history.png"}
    for p in report.figures:
        assert p.is_file() and p.parent == dest
    assert (dest / "manifest.json").is_file()
    assert NO_STILLS not in report.text


def test_render_pack_matches_assess_spec(app_backend, ladruno_set, tmp_path):
    """assess and render_pack draw the same spec: assess calls render_pack."""
    with Results.from_ladruno(ladruno_set) as r:
        r.render_pack(tmp_path / "pack")
        r.assess(figures=True, out_dir=tmp_path / "assess")
    a, b = app_backend.specs
    assert a == b


def test_render_maps_the_arguments(app_backend, ladruno_set, tmp_path):
    out = tmp_path / "stills" / "ux.png"
    with Results.from_ladruno(ladruno_set) as r:
        written = r.render(
            out, view="deformed", component="displacement_x", step=-2,
            deform=40.0, camera="xz", window_size=(640, 480),
        )
    assert written == out and out.is_file()
    (spec,) = app_backend.specs
    assert spec["size"] == [640, 480]
    (shot,) = spec["shots"]
    assert shot == {
        "name": "still", "camera": "-y", "stage": 0, "step": 1,
        "contour": {"field": "DISPLACEMENT", "component": "x"},
        "deform": {"scale": 40.0},
    }


# ---------------------------------------------------------------------
# R22b: .ladruno only, every other source refused by name
# ---------------------------------------------------------------------


def _native_results(tmp_path: Path) -> Results:
    fem = _make_fem(node_ids=[1, 2], coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    path = _write_nodes(
        tmp_path, node_ids=np.array([1, 2], dtype=np.int64),
        components={"displacement_x": np.zeros((1, 2))},
    )
    return Results.from_native(path, model=_open_model_from_h5(path), fem=fem)


@pytest.mark.parametrize("verb", ["assess", "render", "render_pack"])
def test_native_source_is_refused_by_name(app_backend, tmp_path, verb):
    with _native_results(tmp_path) as r:
        with pytest.raises(ValueError, match=r"\.ladruno") as info:
            if verb == "assess":
                r.assess(figures=True, out_dir=tmp_path / "a")
            elif verb == "render":
                r.render(tmp_path / "x.png", view="mesh")
            else:
                r.render_pack(tmp_path / "p")
    assert "results.h5" in str(info.value)
    assert app_backend.specs == []


def test_units_view_is_refused(app_backend, ladruno_set, tmp_path):
    with Results.from_ladruno(ladruno_set) as r:
        view = r.to_units("kN_m_s", source="N_m_s")
        with pytest.raises(ValueError, match="to_units"):
            view.render(tmp_path / "x.png", view="mesh")
    assert app_backend.specs == []


def test_no_source_file_is_refused() -> None:
    from apeGmsh.viewers.session._results_stills import results_files

    class _InMemory:
        path = None

    with pytest.raises(ValueError, match=r"\.ladruno"):
        results_files(_InMemory(), "render")


def test_missing_stem_model_is_refused(app_backend, ladruno_set, tmp_path):
    """The app opens <stem>.h5 beside the .ladruno; without it, no run."""
    lone = tmp_path / "lone"
    lone.mkdir()
    shutil.copy(ladruno_set, lone / "solo.ladruno")
    with Results.from_ladruno(lone / "solo.ladruno") as r:
        with pytest.raises(FileNotFoundError, match="solo.h5"):
            r.render(tmp_path / "x.png", view="mesh")
    assert app_backend.specs == []


def test_stored_gauss_component_is_a_named_carve_out(app_backend, tmp_path):
    """R16b stays refused until follow-up slice #2794; the refusal says so.

    The fork's quad2d fixture records ``stress_xx`` at the Gauss points.
    Its ``<stem>.h5`` is written from the file's own mesh, so the pair holds.
    """
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno" / "quad2d.ladruno"
    src = tmp_path / "quad2d.ladruno"
    shutil.copy(fixture, src)
    with Results.from_ladruno(src) as r:
        r.fem.to_h5(str(tmp_path / "quad2d.h5"))
        stage_id = r.stages[0].id
        assert "stress_xx" in r.inspect.components(stage=stage_id)["gauss"]
        with pytest.raises(ValueError, match="#2794"):
            r.render(tmp_path / "x.png", view="contour", component="stress_xx")
    assert app_backend.specs == []


def _fem_of_length(tmp_path: Path, length: float):
    from apeGmsh.opensees import OpenSeesModel

    Results.demo(n_steps=3, n_elements=4, length=length, path=tmp_path / "other").close()
    return OpenSeesModel.from_h5(tmp_path / "other" / "demo_model.h5").fem


def test_fem_override_is_refused(app_backend, ladruno_set, tmp_path):
    """R22b: the app draws run.h5 (length 10); a fem= of length 40 is not it."""
    with Results.from_ladruno(ladruno_set, fem=_fem_of_length(tmp_path, 40.0)) as r:
        with pytest.raises(ValueError, match="override"):
            r.render(tmp_path / "x.png", view="deformed", component="displacement_x")
    assert app_backend.specs == []


def test_bind_override_is_refused(app_backend, ladruno_set, tmp_path):
    with Results.from_ladruno(ladruno_set) as r:
        bound = r.bind(_fem_of_length(tmp_path, 40.0))
        with pytest.raises(ValueError, match="override"):
            bound.render_pack(tmp_path / "p")
    assert app_backend.specs == []


def test_the_pair_fem_passes_with_or_without_model_h5(app_backend, ladruno_set, tmp_path):
    """The file's own MODEL group (nodes only here) and run.h5's FEMData both draw."""
    with Results.from_ladruno(ladruno_set) as r:
        r.render(tmp_path / "a.png", view="mesh")
    with Results.from_ladruno(ladruno_set, model_h5=ladruno_set.with_suffix(".h5")) as r:
        r.render(tmp_path / "b.png", view="mesh")
    assert len(app_backend.specs) == 2


def _line2_fem(node_ids, coords, conn):
    from apeGmsh.mesh._element_types import make_type_info

    line2 = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=len(conn))
    return _make_fem(
        node_ids=list(node_ids), coords=[tuple(c) for c in coords],
        groups=[(line2, list(range(1, len(conn) + 1)), [list(c) for c in conn])],
    )


@pytest.fixture
def truss_pair(tmp_path: Path):
    """``truss2d.ladruno`` (elements typed by OpenSees class, ``truss``) beside a
    ``truss2d.h5`` of the same nodes and connectivity, typed gmsh ``Line 2`` and
    numbered 1..n. Returns the .ladruno, its node ids, coordinates and cells."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "ladruno" / "truss2d.ladruno"
    src = tmp_path / "truss2d.ladruno"
    shutil.copy(fixture, src)
    with Results.from_ladruno(src) as r:
        ids = np.asarray(r.fem.nodes.ids).tolist()
        xyz = np.asarray(r.fem.nodes.coords, dtype=float).tolist()
        cells = [row for g in r.fem.elements for row in np.asarray(g.connectivity).tolist()]
        assert {g.element_type.name for g in r.fem.elements} != {"line2"}
    _line2_fem(ids, xyz, cells).to_h5(str(tmp_path / "truss2d.h5"))
    return src, ids, xyz, cells


def test_a_matching_pair_is_never_refused_by_element_type_name(app_backend, truss_pair, tmp_path):
    """Same nodes and connectivity, other type names and element ids: the pair draws."""
    src = truss_pair[0]
    with Results.from_ladruno(src) as r:
        r.render(tmp_path / "x.png", view="mesh")
    assert len(app_backend.specs) == 1


@pytest.mark.parametrize("how", ["fem=", "bind()"])
def test_reordered_nodes_within_a_cell_still_pair(app_backend, truss_pair, tmp_path, how):
    """[[2, 1], [3, 2]] is [[1, 2], [2, 3]]: a writer may reorder a cell (hex20)."""
    src, ids, xyz, cells = truss_pair
    flipped = _line2_fem(ids, xyz, [row[::-1] for row in cells])
    opened = Results.from_ladruno(src, fem=flipped) if how == "fem=" else Results.from_ladruno(src)
    with opened as r:
        (r if how == "fem=" else r.bind(flipped)).render(tmp_path / "x.png", view="mesh")
    assert len(app_backend.specs) == 1


@pytest.mark.parametrize("how", ["fem=", "bind()"])
def test_same_nodes_other_connectivity_is_an_override(app_backend, truss_pair, tmp_path, how):
    """[[1, 3]] is not [[1, 2], [2, 3]]: the nodes match, the mesh does not."""
    src, ids, xyz, cells = truss_pair
    other = _line2_fem(ids, xyz, [cells[0][:1] + cells[1][1:]])
    assert sorted(map(tuple, [cells[0][:1] + cells[1][1:]])) != sorted(map(tuple, cells))
    opened = Results.from_ladruno(src, fem=other) if how == "fem=" else Results.from_ladruno(src)
    with opened as r:
        bound = r if how == "fem=" else r.bind(other)
        with pytest.raises(ValueError, match="override"):
            bound.render(tmp_path / "x.png", view="mesh")
    assert app_backend.specs == []


def test_a_stamped_fem_hash_of_another_model_does_not_pair(app_backend, truss_pair, tmp_path):
    import h5py

    from apeGmsh.opensees import OpenSeesModel

    src = truss_pair[0]
    assert OpenSeesModel.from_h5(tmp_path / "truss2d.h5").lineage.fem_hash
    with h5py.File(src, "r+") as f:
        f.require_group("INFO/PROVENANCE").attrs["fem_hash"] = "0" * 32
    with Results.from_ladruno(src) as r:
        with pytest.raises(ValueError, match="do not pair") as info:
            r.render(tmp_path / "x.png", view="mesh")
    assert "override" not in str(info.value)
    assert app_backend.specs == []


# ---------------------------------------------------------------------
# G9 and G2
# ---------------------------------------------------------------------


def test_no_app_raises_from_assess(ladruno_set, tmp_path, monkeypatch):
    """G9: no app is an error, never a silent ``no stills`` report."""
    import apeGmsh.agent_view as agent_view
    from apeGmsh.viewers import render as render_mod

    def _missing(spec, out, **kwargs):
        raise FileNotFoundError("apeGmshViewer was not found")

    monkeypatch.setattr(render_mod, "_stills_backend", lambda: "app")
    monkeypatch.setattr(agent_view, "render_stills", _missing)
    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    with Results.from_ladruno(ladruno_set) as r:
        with pytest.raises(FileNotFoundError, match="apeGmshViewer"):
            r.assess(figures=True, out_dir=tmp_path / "a")


def test_skip_env_still_skips_on_the_app_path(app_backend, ladruno_set, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    with Results.from_ladruno(ladruno_set) as r:
        report = r.assess(figures=True, out_dir=tmp_path / "a")
    assert report.figures == ()
    assert NO_STILLS in report.text
    assert "[skip viewer]" in capsys.readouterr().out
    assert app_backend.specs == []


@pytest.mark.parametrize("verb", ["assess", "render"])
def test_skip_env_never_loses_a_report_on_another_source(app_backend, tmp_path, monkeypatch, capsys, verb):
    """The skip is checked before the .ladruno refusal: a native .h5 under
    APEGMSH_SKIP_VIEWER=1 keeps its assess report and skips its still."""
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    with _native_results(tmp_path) as r:
        if verb == "assess":
            report = r.assess(figures=True, out_dir=tmp_path / "a")
            assert report.findings and report.figures == () and NO_STILLS in report.text
        else:
            assert r.render(tmp_path / "x.png", view="mesh") is None
    assert "[skip viewer]" in capsys.readouterr().out
    assert app_backend.specs == []


def test_vtk_backend_never_reaches_agent_view(ladruno_set, tmp_path, monkeypatch, capsys):
    """G2: Linux keeps the VTK path; agent_view is never called there."""
    import apeGmsh.agent_view as agent_view
    from apeGmsh.viewers import render as render_mod

    monkeypatch.setattr(render_mod, "_stills_backend", lambda: "vtk")
    monkeypatch.setattr(agent_view, "render_stills", _boom)
    monkeypatch.setenv("APEGMSH_SKIP_VIEWER", "1")
    with Results.from_ladruno(ladruno_set) as r:
        report = r.assess(figures=True, out_dir=tmp_path / "a")
        assert r.render(tmp_path / "x.png") is None
    assert report.figures == ()
    assert "[skip viewer]" in capsys.readouterr().out
