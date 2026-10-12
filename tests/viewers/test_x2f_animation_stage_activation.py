"""X2-f (#1722) — ``Results.export_animation`` honours stage activation.

Before X2-b (#1700) the verb drove the Qt ``ResultsViewer``, whose
``director.set_stage`` fired ``_apply_stage_activation``: on a STAGED
model each animated stage drew only the elements active in it. X2-b
moved the verb onto the session path, which applied no activation and
drew every element at every stage. This pins the restored behaviour,
and pins that the stills (``Results.render``) still draw every element:
they never filtered, and whether they should is a separate question.

Fixture (``test_stage_activation.py::staged_results``, two steps per
stage so the re-step fast path runs): Rock (z 0-1) is global, Fill
(z 1-2) activates in ``construction``, Rock is removed in ``excavate``.

Seam: every ``BasePlotter.screenshot`` records, per visible 3-D actor,
its dataset's cell count and z range (the ``capture_drawn`` seam of
``test_x2_numeric_oracle.py``, extended with cells).
"""
from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

pytest.importorskip("pyvista")


# ---------------------------------------------------------------------
# Seam
# ---------------------------------------------------------------------


def _drawn(plotter: Any) -> list[dict[str, Any]]:
    out = []
    for renderer in plotter.renderers:
        for name, actor in renderer.actors.items():
            if not actor.GetVisibility() or actor.IsA("vtkActor2D"):
                continue
            mapper = getattr(actor, "mapper", None)
            ds = getattr(mapper, "dataset", None) if mapper is not None else None
            if ds is None or ds.n_points == 0:
                continue
            z = np.asarray(ds.points)[:, 2]
            cell_data = getattr(ds, "cell_data", {})
            out.append({
                "name": str(name),
                                "n_cells": int(ds.n_cells),
                "zmin": float(z.min()),
                "zmax": float(z.max()),
                # The render-surface fast path stamps this on the
                # extracted boundary of a volume layer: the substrate.
                "surface": "_apegmsh_orig_cell_ids" in cell_data,
            })
    return out


@contextlib.contextmanager
def _capture() -> Iterator[list[list[dict[str, Any]]]]:
    from pyvista.plotting.plotter import BasePlotter

    frames: list[list[dict[str, Any]]] = []
    shot = BasePlotter.screenshot

    def _screenshot(self: Any, *a: Any, **k: Any) -> Any:
        frames.append(_drawn(self))
        return shot(self, *a, **k)

    BasePlotter.screenshot = _screenshot  # type: ignore[method-assign]
    try:
        yield frames
    finally:
        BasePlotter.screenshot = shot  # type: ignore[method-assign]


class _NullWriter:
    def __enter__(self) -> "_NullWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def append_data(self, frame: Any) -> None:
        return None


def _stub_imageio(monkeypatch: pytest.MonkeyPatch) -> None:
    v2 = types.ModuleType("imageio.v2")

    def get_writer(path: str, **_k: Any) -> _NullWriter:
        Path(path).touch()
        return _NullWriter()

    v2.get_writer = get_writer  # type: ignore[attr-defined]
    pkg = types.ModuleType("imageio")
    pkg.v2 = v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", pkg)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)


# ---------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------


@pytest.fixture
def staged(g, tmp_path, monkeypatch):
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.results import Results
    from apeGmsh.results.capture.spec import DomainCaptureSpec
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    from tests.conftest import _open_model_from_h5
    from tests.test_results_domain_capture import _FakeOps

    monkeypatch.delenv("APEGMSH_SKIP_VIEWER", raising=False)
    _stub_imageio(monkeypatch)

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="rock")
    g.model.geometry.add_box(0, 0, 1, 1, 1, 1, label="fill")
    g.physical.add_volume("rock", name="Rock")
    g.physical.add_volume("fill", name="Fill")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3)
    ops.element.FourNodeTetrahedron(pg="Rock", material=mat)
    ops.element.FourNodeTetrahedron(pg="Fill", material=mat)

    def chain():
        return {
            "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            "algorithm":   ops.algorithm.Newton(),
            "integrator":  ops.integrator.LoadControl(dlam=0.1),
            "constraints": ops.constraints.Plain(),
            "numberer":    ops.numberer.RCM(),
            "system":      ops.system.UmfPack(),
            "analysis":    ops.analysis.Static(),
        }

    with ops.stage(name="g") as s:
        s.analysis(**chain())
        s.run(n_increments=1)
    with ops.stage(name="construction") as s:
        s.activate(pgs=["Fill"])
        s.analysis(**chain())
        s.run(n_increments=1)
    with ops.stage(name="excavate") as s:
        s.remove_element(pg="Rock")
        s.analysis(**chain())
        s.run(n_increments=1)

    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(
        ids=[int(i) for i in fem.nodes.ids],
        components=["displacement_x", "displacement_y", "displacement_z"],
        name="all",
    )
    out = tmp_path / "staged_run.h5"
    with ops.domain_capture(spec, path=str(out), ops=_FakeOps()) as cap:
        for i, name in enumerate(["g", "construction", "excavate"]):
            cap.begin_stage(name, kind="static")
            cap.step(t=float(i + 1))
            cap.step(t=float(i + 1) + 0.5)
            cap.end_stage()
    results = Results.from_native(out, model=_open_model_from_h5(out))
    n_total = int(build_fem_scene(results.fem).grid.n_cells)
    try:
        yield results, n_total
    finally:
        results.close()


#: stage -> (z range the drawn cells must span, Rock drawn, Fill drawn)
_EXPECT = {
    "g":            ((0.0, 1.0), True, False),
    "construction": ((0.0, 2.0), True, True),
    "excavate":     ((1.0, 2.0), False, True),
}


def _pg_eids(results: Any) -> dict[str, set[int]]:
    return {
        pg: {int(e) for grp in results.fem.elements.select(pg=pg).groups()
             for e in grp.ids}
        for pg in ("Rock", "Fill")
    }


def _surface_cells(results: Any, eids: "set[int] | None") -> int:
    """Boundary faces of the cell set ``eids`` (``None``: every cell),
    extracted independently with the backend's own surface filter —
    what the substrate actor draws for exactly that set."""
    from apeGmsh.viewers.backends.pyvista_qt import extract_render_surface
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    scene = build_fem_scene(results.fem)
    grid = scene.grid
    if eids is not None:
        cells = sorted(scene.element_id_to_cell[e] for e in eids)
        grid = grid.extract_cells(np.asarray(cells, dtype=np.int64))
    extracted = extract_render_surface(grid)
    assert extracted is not None
    return int(extracted[0].n_cells)


def _substrate(frame: list[dict[str, Any]]) -> dict[str, Any]:
    """The one volume cell-set actor: the grey substrate, drawn as its
    pre-extracted boundary surface (outlines are a separate line set)."""
    subs = [a for a in frame if a["surface"]]
    assert len(subs) == 1, f"expected one substrate actor, got {frame}"
    return subs[0]


@pytest.mark.parametrize("deform", [None, 1.0])
@pytest.mark.parametrize("stage", sorted(_EXPECT))
def test_export_animation_draws_only_active_elements(
    staged, tmp_path, stage, deform,
):
    results, _n_total = staged
    pgs = _pg_eids(results)
    (zlo, zhi), rock, fill = _EXPECT[stage]
    active = (pgs["Rock"] if rock else set()) | (pgs["Fill"] if fill else set())
    want = _surface_cells(results, active)

    with _capture() as frames:
        out = results.export_animation(
            tmp_path / f"{stage}.gif", stage=stage, fps=2,
            window_size=(160, 120), deform=deform,
        )
    assert out == tmp_path / f"{stage}.gif"
    assert len(frames) == 2  # two steps, both drawn (re-step frame too)
    for frame in frames:
        sub = _substrate(frame)
        assert sub["n_cells"] == want, (stage, sub["n_cells"], want)
        if deform is None:
            assert sub["zmin"] == pytest.approx(zlo)
            assert sub["zmax"] == pytest.approx(zhi)
            # No drawn actor (outlines too) reaches outside the active
            # cells' z range.
            for actor in frame:
                assert actor["zmin"] >= zlo - 1e-9, actor
                assert actor["zmax"] <= zhi + 1e-9, actor


def test_render_still_draws_every_element(staged, tmp_path):
    """Stills never filtered (measured at 52fe77b9); X2-f leaves them so."""
    results, _n_total = staged
    everything = _surface_cells(results, None)
    for stage in _EXPECT:
        with _capture() as frames:
            results.stage(stage).render(tmp_path / f"{stage}.png")
        assert frames, "render took no shot"
        sub = _substrate(frames[-1])
        assert sub["n_cells"] == everything
        assert sub["zmin"] == pytest.approx(0.0)
        assert sub["zmax"] == pytest.approx(2.0)


@pytest.mark.parametrize("why", ["no_program", "unmatched_stage"])
def test_no_activation_data_draws_everything(staged, tmp_path, monkeypatch, why):
    """Fail-soft where the old viewer was: no staged program, or a
    capture stage pairing with no program stage -> no filter (the whole
    mesh), never an empty picture."""
    from apeGmsh.viewers.data import _stage_activation as SA

    results, _n_total = staged
    if why == "no_program":
        monkeypatch.setattr(SA, "build_from_model", lambda *a, **k: None)
    else:
        monkeypatch.setattr(SA, "pair_capture_to_program", lambda *a, **k: {})
    with _capture() as frames:
        results.export_animation(
            tmp_path / "g.gif", stage="g", fps=2, window_size=(160, 120),
        )
    assert frames
    everything = _surface_cells(results, None)
    for frame in frames:
        assert _substrate(frame)["n_cells"] == everything


def test_stage_with_no_active_element_refuses(staged, tmp_path):
    """Fail closed: a view scoped to Fill, animated at 'g' (Fill not yet
    active), has nothing to draw: it raises and leaves no file, instead
    of silently drawing the unfiltered mesh."""
    from apeGmsh.results.session import Scope

    results, _n_total = staged

    def setup(plotter: Any, view: Any) -> None:
        view.scope = Scope("physical_groups", ("Fill",))

    with pytest.raises(ValueError, match="activates none"):
        results.export_animation(
            tmp_path / "g.gif", stage="g", fps=2,
            window_size=(160, 120), setup=setup,
        )
    assert not (tmp_path / "g.gif").exists()
    assert not (tmp_path / "g.partial.gif").exists()


def test_scope_and_activation_intersect(staged, tmp_path):
    """A scoped view draws scope AND active: Rock-scoped at
    'construction' draws Rock only (Fill is active but out of scope)."""
    from apeGmsh.results.session import Scope

    results, _n_total = staged
    want = _surface_cells(results, _pg_eids(results)["Rock"])

    def setup(plotter: Any, view: Any) -> None:
        view.scope = Scope("physical_groups", ("Rock",))

    with _capture() as frames:
        results.export_animation(
            tmp_path / "c.gif", stage="construction", fps=2,
            window_size=(160, 120), setup=setup,
        )
    assert len(frames) == 2
    for frame in frames:
        sub = _substrate(frame)
        assert sub["n_cells"] == want
        assert sub["zmax"] == pytest.approx(1.0)
