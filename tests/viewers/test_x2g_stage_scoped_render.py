"""X2-g (#1742) â€” a stage-scoped still draws THAT stage.

``results.stage(s).render()`` / ``.render_pack()`` /
``.assess(figures=True)`` drew the LAST stage whatever the scope
(``_results_stills._resolve_stage`` took ``results.stages[-1]``; a
scoped Results lists every stage too). Now the scope wins, as in
``export_animation`` (``Results._resolve_stage``); unscoped stills keep
the last-stage default (pinned by ``test_x2f_animation_stage_activation``).

Fixture: the X2-f staged model (Rock z 0-1 global, Fill z 1-2 activated
in ``construction``, Rock removed in ``excavate``), with a distinct
displacement per stage: ``u_z = a_s * (k + 1) * (1 + z)`` at step ``k``
of stage ``s``, ``a = 0.1 / 0.2 / 0.3``. Checked through the numeric
seam (``test_x2_numeric_oracle.capture_drawn``): drawn contour scalars
are stage ``s``'s data at its last step, and the drawn substrate cells
are its activation set (``test_x2f_animation_stage_activation``'s seam).
On ``main`` before the fix, ``g`` and ``construction`` fail (they drew
``excavate``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("pyvista")

from tests.viewers.test_x2_numeric_oracle import capture_drawn  # noqa: E402
from tests.viewers.test_x2f_animation_stage_activation import (  # noqa: E402
    _EXPECT,
    _capture,
    _pg_eids,
    _stub_imageio,
    _substrate,
    _surface_cells,
)

_AMP = {"g": 0.1, "construction": 0.2, "excavate": 0.3}
_STAGES = ["g", "construction", "excavate"]
_WIN = (160, 120)


def _build(g, tmp_path, monkeypatch, reactions_in=None):
    """The staged-disp file; with ``reactions_in``, reactions are
    recorded too and then stripped from every OTHER stage, so exactly
    that stage carries a reaction field."""
    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.results import Results
    from apeGmsh.results.capture.spec import DomainCaptureSpec

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

    ids = [int(i) for i in fem.nodes.ids]
    z = {int(n): float(c[2]) for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(
        ids=ids,
        components=["displacement_x", "displacement_y", "displacement_z"],
        name="all",
    )
    if reactions_in is not None:
        spec.nodes(ids=ids, components=["reaction_force_z"], name="rxn")
    out = tmp_path / "staged_disp.h5"
    fake = _FakeOps()
    with ops.domain_capture(spec, path=str(out), ops=fake) as cap:
        for i, name in enumerate(_STAGES):
            cap.begin_stage(name, kind="static")
            for k in range(2):
                fake.disp_table = {
                    (n, 3): _AMP[name] * (k + 1) * (1.0 + z[n]) for n in ids
                }
                fake.reaction_table = {(n, 3): 1.0 + z[n] for n in ids}
                cap.step(t=float(i + 1) + 0.5 * k)
            cap.end_stage()
    if reactions_in is not None:
        _strip_reactions(out, keep=reactions_in)
    return Results.from_native(out, model=_open_model_from_h5(out))


def _strip_reactions(path: Any, keep: str) -> None:
    """Delete every reaction dataset outside stage ``keep``."""
    import h5py

    with h5py.File(path, "r+") as f:
        doomed: list[str] = []

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and "reaction" in name:
                doomed.append(name)

        f.visititems(visit)
        stage_of = {
            sid: str(grp.attrs.get("name", sid))
            for sid, grp in f["stages"].items()
        }
        dropped = 0
        for name in doomed:
            parts = name.split("/")
            if parts[0] == "stages" and stage_of.get(parts[1]) != keep:
                del f[name]
                dropped += 1
        assert dropped, f"no reaction datasets stripped; paths {doomed}"


@pytest.fixture
def staged_disp(g, tmp_path, monkeypatch):
    results = _build(g, tmp_path, monkeypatch)
    try:
        yield results
    finally:
        results.close()


@pytest.fixture
def staged_reactions(g, tmp_path, monkeypatch):
    results = _build(g, tmp_path, monkeypatch, reactions_in="construction")
    try:
        yield results
    finally:
        results.close()


def _active(results: Any, stage: str) -> set[int]:
    pgs = _pg_eids(results)
    _z, rock, fill = _EXPECT[stage]
    return (pgs["Rock"] if rock else set()) | (pgs["Fill"] if fill else set())


def _active_node_z(results: Any, stage: str) -> np.ndarray:
    """Reference z of every node of the stage's active elements."""
    _z, rock, fill = _EXPECT[stage]
    pgs = [pg for pg, on in (("Rock", rock), ("Fill", fill)) if on]
    nodes: set[int] = set()
    for pg in pgs:
        for grp in results.fem.elements.select(pg=pg).groups():
            nodes.update(int(n) for n in np.asarray(grp.connectivity).ravel())
    coords = {
        int(n): c for n, c in zip(results.fem.nodes.ids, results.fem.nodes.coords)
    }
    return np.array([coords[n][2] for n in sorted(nodes)], dtype=np.float64)


def _assert_stage_drawn(
    results: Any,
    stage: str,
    recs: list[dict[str, Any]],
    frames: list[list[dict[str, Any]]],
) -> None:
    """Every VTK shot draws ``stage``'s active cells and, where a contour
    is drawn, exactly ``stage``'s last-step u_z on its active nodes."""
    vtk = [r for r in recs if r["kind"] == "vtk"]
    assert vtk and len(vtk) == len(frames), (len(vtk), len(frames))
    want_cells = _surface_cells(results, _active(results, stage))
    want_vals = np.unique(np.round(
        _AMP[stage] * 2.0 * (1.0 + _active_node_z(results, stage)), 9))
    for rec, frame in zip(vtk, frames):
        assert _substrate(frame)["n_cells"] == want_cells, (
            stage, rec["name"], _substrate(frame)["n_cells"], want_cells)
        if not _is_deformed(rec):
            # Rock and Fill mesh alike, so the cell COUNT cannot tell
            # ``g`` from ``excavate``; the undeformed z range can.
            (zlo, zhi), _rock, _fill = _EXPECT[stage]
            sub = _substrate(frame)
            assert sub["zmin"] == pytest.approx(zlo), (stage, rec["name"])
            assert sub["zmax"] == pytest.approx(zhi), (stage, rec["name"])
        contour = rec["contour"]
        if contour is None:
            continue
        _assert_values(contour["scalars"], want_vals, stage, rec["name"])
        if not _is_deformed(rec):
            # Undeformed: each drawn point carries u_z of ITS z, pointwise.
            z = np.asarray(contour["points"])[:, 2]
            np.testing.assert_allclose(
                contour["scalars"], _AMP[stage] * 2.0 * (1.0 + z),
                atol=1e-6, err_msg=f"{stage} {rec['name']}")


def _is_deformed(rec: dict[str, Any]) -> bool:
    return bool(rec["name"]) and "deformed" in str(rec["name"])


def _assert_values(
    scalars: np.ndarray, want_vals: np.ndarray, stage: str, name: Any,
) -> None:
    """Drawn scalars are nodal values of ``stage`` (each one a value of
    an active node), spanning its whole active range."""
    drawn = np.unique(np.round(np.asarray(scalars, dtype=np.float64), 6))
    gap = np.min(np.abs(drawn[:, None] - want_vals[None, :]), axis=1)
    assert gap.max() <= 1e-6, (stage, name, drawn[gap > 1e-6])
    assert drawn.min() == pytest.approx(want_vals.min(), abs=1e-6), (stage, name)
    assert drawn.max() == pytest.approx(want_vals.max(), abs=1e-6), (stage, name)


@pytest.mark.parametrize("view", ["mesh", "contour", "deformed"])
@pytest.mark.parametrize("stage", _STAGES)
def test_scoped_render_draws_that_stage(staged_disp, tmp_path, stage, view):
    results = staged_disp
    kw: dict[str, Any] = {"window_size": _WIN}
    if view != "mesh":
        kw["component"] = "displacement_z"
    with capture_drawn() as recs, _capture() as frames:
        out = results.stage(stage).render(
            tmp_path / f"{stage}_{view}.png", view=view, **kw)
    assert out is not None
    assert len(frames) == 1
    if view != "mesh":
        assert recs[0]["contour"] is not None
    _assert_stage_drawn(results, stage, recs, frames)


@pytest.mark.parametrize("stage", _STAGES)
def test_scoped_render_pack_draws_that_stage(staged_disp, tmp_path, stage):
    results = staged_disp
    with capture_drawn() as recs, _capture() as frames:
        written = results.stage(stage).render_pack(
            tmp_path / stage, window_size=_WIN)
    names = {p.name for p in written}
    assert {"mesh.png", "contour.png", "deformed.png"} <= names, names
    assert len(frames) == 3
    assert sum(r["contour"] is not None for r in recs if r["kind"] == "vtk") == 2
    _assert_stage_drawn(results, stage, recs, frames)


@pytest.mark.parametrize("stage", _STAGES)
def test_scoped_assess_figures_draw_that_stage(staged_disp, tmp_path, stage):
    results = staged_disp
    with capture_drawn() as recs, _capture() as frames:
        results.stage(stage).assess(figures=True, out_dir=tmp_path / stage)
    assert len(frames) == 3
    _assert_stage_drawn(results, stage, recs, frames)


def test_scoped_step_indexes_the_scoped_stage(staged_disp, tmp_path):
    """``step=0`` of a scoped still is the scoped stage's first step."""
    results = staged_disp
    with capture_drawn() as recs:
        results.stage("g").render(
            tmp_path / "g0.png", view="contour",
            component="displacement_z", step=0, window_size=_WIN)
    contour = recs[0]["contour"]
    z = np.asarray(contour["points"])[:, 2]
    np.testing.assert_allclose(contour["scalars"], 0.1 * (1.0 + z), atol=1e-6)


def test_unknown_scope_refuses(staged_disp, tmp_path):
    """Fail closed: a scope that names no stage in the file raises; it
    never falls back to drawing the last stage."""
    scoped = staged_disp.stage("g")
    scoped._stage_id = "no-such-stage"
    with pytest.raises(RuntimeError, match="no-such-stage"):
        scoped.render(tmp_path / "x.png", view="mesh", window_size=_WIN)


@pytest.mark.parametrize("stage", _STAGES)
def test_scoped_pack_reactions_follow_the_scope(
    staged_reactions, tmp_path, stage,
):
    """Reactions are recorded in ``construction`` only: its scoped pack
    writes ``reactions.png``; the other stages' scoped packs do not
    (unscoped, any stage's reactions counted, so every pack drew one)."""
    results = staged_reactions
    names = {
        p.name for p in results.stage(stage).render_pack(
            tmp_path / stage, window_size=_WIN)
    }
    assert ("reactions.png" in names) == (stage == "construction"), (
        stage, names)
