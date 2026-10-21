"""The transient ``Subdivide`` rung (ADR 0057 Amendment 1, #2403).

A failed transient increment is retried in ``n`` pieces of its remaining
time; the increment always ends on the step grid, and exhausting every
rung still aborts the deck.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.algorithm import Newton
from apeGmsh.opensees.analysis.strategy import Ladder, Subdivide
from apeGmsh.opensees.emitter.base import SUBDIVIDE_TOKEN

DT, N = 0.01, 20


def test_subdivide_validates_n() -> None:
    for bad in (1, 0, 2.5, True):
        with pytest.raises(ValueError, match="int >= 2"):
            Subdivide(n=bad)  # type: ignore[arg-type]


def test_to_spec_encodes_the_rung_and_keeps_rung_0_an_algorithm() -> None:
    lad = Ladder(rungs=(Newton(), Subdivide(4), Subdivide(16)))
    assert lad.subdivides
    spec = lad.to_spec(base=Newton())
    assert spec.rungs[1:] == ((SUBDIVIDE_TOKEN, 4), (SUBDIVIDE_TOKEN, 16))
    with pytest.raises(ValueError, match="rung 0 must be an algorithm"):
        Ladder(rungs=(Subdivide(4), Newton())).to_spec(base=None)


@pytest.fixture(scope="module")
def column() -> Any:
    g = apeGmsh(model_name="subdiv", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_rectangle(0, 0, 0, 1, 4, label="s")
        g.physical.add_surface("s", name="S")
        g.model.select(None, dim=1).on_plane(
            (0, 0, 0), (0, 1, 0), tol=1e-6).to_physical("Base")
        g.model.select(None, dim=1).on_plane(
            (0, 4, 0), (0, 1, 0), tol=1e-6).to_physical("Top")
        g.mesh.sizing.set_global_size(1.0)
        g.mesh.structured.set_transfinite_automatic()
        g.mesh.generation.generate(dim=2)
        return g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()


def _staged(fem: Any, *, tol: float, dt: "float | None" = DT,
            out: str = "t.out") -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    m = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3, rho=2.0)
    ops.element.FourNodeQuad(pg="S", thickness=1.0, material=m)
    ops.fix(pg="Base", dofs=(1, 1))
    top = int(fem.nodes.select(pg="Top").ids[0])
    lad = ops.strategy.Ladder(rungs=(
        ops.algorithm.Newton(), ops.strategy.Subdivide(4),
        ops.strategy.Subdivide(16)))
    rec = ops.recorder.Node(file=out, response="disp", nodes=(top,),
                            dofs=(1,), time_format="dt")
    with ops.stage(name="shake") as s:
        series = ops.timeSeries.Path(
            dt=DT, values=tuple(float(i % 7) for i in range(50)))
        with s.pattern(series=series) as p:
            p.load(node=top, forces=(1.0, 0.0))
        s.recorder(rec)
        s.analysis(
            test=ops.test.NormDispIncr(tol=tol, max_iter=2),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.Newmark(gamma=0.5, beta=0.25),
            constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(), analysis=ops.analysis.Transient())
        s.run(n_increments=N, dt=dt, strategy=lad)
    return ops


def test_a_static_stage_refuses_the_rung(column) -> None:
    with pytest.raises(ValueError, match="needs a transient stage"):
        _staged(column, tol=1e-8, dt=None)


@pytest.mark.parametrize("fmt", ["py", "tcl"])
def test_the_deck_carries_the_piece_loop(column, fmt, tmp_path: Path) -> None:
    ops = _staged(column, tol=1e-8)
    path = tmp_path / f"deck.{fmt}"
    getattr(ops, fmt)(str(path))
    text = path.read_text()
    assert "_apesees_t1" in text and SUBDIVIDE_TOKEN in text
    assert "_apesees_h" in text


@pytest.mark.live
@pytest.mark.parametrize("tol, ok", [(1e-8, True), (1e-30, False)])
def test_the_py_deck_ends_on_the_grid_or_aborts(column, tol, ok,
                                                tmp_path: Path) -> None:
    pytest.importorskip("openseespy.opensees")
    ops = _staged(column, tol=tol, out="t.out")
    ops.py(str(tmp_path / "deck.py"))
    r = subprocess.run([sys.executable, "deck.py"], cwd=str(tmp_path),
                       capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        t = np.loadtxt(tmp_path / "t.out")[:, 0]
        assert len(t) == N and t[-1] == pytest.approx(N * DT)
    else:
        assert r.returncode != 0
        out = r.stdout + r.stderr
        assert out.count("-> rung") == 2                 # both Subdivide rungs
        assert "exhausting strategy ladder" in out


def _rescue_model(fem: Any) -> apeSees:
    """A loading that a one-iteration Newton cannot always carry: the
    Subdivide rungs rescue increments (and the deck still finishes)."""
    import math

    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    m = ops.nDMaterial.ElasticIsotropic(E=1e5, nu=0.3, rho=2.0)
    ops.element.FourNodeQuad(pg="S", thickness=1.0, material=m)
    ops.fix(pg="Base", dofs=(1, 1))
    top = int(fem.nodes.select(pg="Top").ids[0])
    lad = ops.strategy.Ladder(rungs=(
        ops.algorithm.Newton(), ops.strategy.Subdivide(4),
        ops.algorithm.ModifiedNewton(), ops.strategy.Subdivide(16)))
    rec = ops.recorder.Node(file="t.out", response="disp", nodes=(top,),
                            dofs=(1,), time_format="dt")
    n = 50
    with ops.stage(name="shake") as s:
        series = ops.timeSeries.Path(dt=DT, values=tuple(
            math.sin(0.37 * i) * (1 + i % 5) for i in range(n + 5)))
        with s.pattern(series=series) as p:
            p.load(node=top, forces=(1.0, 0.0))
        s.recorder(rec)
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-3, max_iter=1),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.Newmark(gamma=0.5, beta=0.25),
            constraints=ops.constraints.Plain(), numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(), analysis=ops.analysis.Transient())
        s.run(n_increments=n, dt=DT, strategy=lad)
    return ops


def test_a_static_stage_with_dt_refuses_the_rung(column) -> None:
    ops = apeSees(column)
    ops.model(ndm=2, ndf=2)
    lad = ops.strategy.Ladder(rungs=(ops.algorithm.Newton(),
                                     ops.strategy.Subdivide(4)))
    with pytest.raises(ValueError, match="under Static"):
        with ops.stage(name="st") as s:
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-8, max_iter=5),
                algorithm=ops.algorithm.Newton(),
                integrator=ops.integrator.LoadControl(dlam=DT),
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.RCM(), system=ops.system.UmfPack(),
                analysis=ops.analysis.Static())
            s.run(n_increments=5, dt=DT, strategy=lad)


@pytest.mark.live
def test_rescued_increments_land_on_the_grid_in_py_and_tcl(
        column, tmp_path: Path) -> None:
    pytest.importorskip("openseespy.opensees")
    import shutil

    ops = _rescue_model(column)
    (tmp_path / "py").mkdir()
    ops.py(str(tmp_path / "py" / "deck.py"))
    r = subprocess.run([sys.executable, "deck.py"], cwd=str(tmp_path / "py"),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "Subdivide(4) committed 4/4 pieces" in r.stdout
    rows = np.loadtxt(tmp_path / "py" / "t.out")
    assert len(rows) > 50                        # pieces record too ...
    assert rows[-1, 0] == pytest.approx(50 * DT, abs=1e-9)   # ... on the grid
    exe = shutil.which("OpenSees") or shutil.which("OpenSees.exe")
    if exe is None:
        return
    (tmp_path / "tcl").mkdir()
    ops.tcl(str(tmp_path / "tcl" / "deck.tcl"))
    subprocess.run([exe, "deck.tcl"], cwd=str(tmp_path / "tcl"),
                   capture_output=True, text=True)
    rows_tcl = np.loadtxt(tmp_path / "tcl" / "t.out")
    assert rows_tcl.shape == rows.shape
    assert np.allclose(rows_tcl, rows, rtol=1e-6, atol=1e-12)
